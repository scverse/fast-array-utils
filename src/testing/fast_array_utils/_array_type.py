# SPDX-License-Identifier: MPL-2.0
"""ArrayType class and helpers."""

from __future__ import annotations

import enum
from dataclasses import KW_ONLY, dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np


if TYPE_CHECKING:
    from typing import Any, Literal, Protocol, SupportsFloat, TypeAlias

    import h5py
    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from fast_array_utils import types
    from fast_array_utils.types import CSBase

    Array: TypeAlias = (
        NDArray[Any]
        | types.CSBase
        | types.CupyArray
        | types.DaskArray
        | types.H5Dataset
        | types.ZarrArray
    )

    Arr = TypeVar("Arr", bound=Array, default=Array)
    Arr_co = TypeVar("Arr_co", bound=Array, covariant=True)

    Inner = TypeVar("Inner", bound="ArrayType[Any, None] | None", default=Any)

    class ToArray(Protocol, Generic[Arr_co]):
        """Convert to a supported array."""

        def __call__(self, data: ArrayLike, /, *, dtype: DTypeLike | None = None) -> Array: ...

    _DTypeLikeFloat32 = np.dtype[np.float32] | type[np.float32]
    _DTypeLikeFloat64 = np.dtype[np.float64] | type[np.float64]
else:
    Arr = TypeVar("Arr")
    Inner = TypeVar("Inner")


__all__ = ["ArrayType", "ConversionContext", "ToArray"]


class Flags(enum.Flag):
    """Array classification flags.

    Use ``Flags(0)`` and ``~Flags(0)`` for no or all flags set.
    """

    Any = enum.auto()
    Sparse = enum.auto()
    Gpu = enum.auto()
    Dask = enum.auto()
    Disk = enum.auto()

    def __repr__(self) -> str:
        if self is Flags(0):
            return "Flags(0)"
        if self is ~Flags(0):
            return "~Flags(0)"
        return super().__repr__()


@dataclass
class ConversionContext:
    """Conversion context required for h5py."""

    hdf5_file: h5py.File


@dataclass(frozen=True)
class ArrayType(Generic[Arr, Inner]):
    """Supported array type with methods for conversion and random generation.

    Examples
    --------
    >>> at = ArrayType("numpy", "ndarray")
    >>> arr = at([1, 2, 3])
    >>> arr
    array([1, 2, 3])
    >>> assert isinstance(arr, at.cls)

    """

    mod: str
    """Module name."""
    name: str
    """Array class name."""
    flags: Flags = Flags.Any
    """Classification flags."""

    _: KW_ONLY

    inner: Inner = None  # type: ignore[assignment]
    """Inner array type (e.g. for dask)."""
    conversion_context: ConversionContext | None = field(default=None, compare=False)
    """Conversion context required for converting to h5py."""

    def __repr__(self) -> str:
        rv = f"{self.mod}.{self.name}"
        return f"{rv}[{self.inner}]" if self.inner else rv

    @cached_property
    def cls(self) -> type[Arr]:  # noqa: PLR0911
        """Array class for :func:`isinstance` checks."""
        match self.mod, self.name, self.inner:
            case "numpy", "ndarray", None:
                return np.ndarray  # type: ignore[return-value]
            case "scipy.sparse", (
                "csr_array" | "csc_array" | "csr_matrix" | "csc_matrix"
            ) as cls_name, None:
                import scipy.sparse

                return getattr(scipy.sparse, cls_name)  # type: ignore[no-any-return]
            case "cupy", "ndarray", None:
                import cupy as cp

                return cp.ndarray  # type: ignore[no-any-return]
            case "cupyx.scipy.sparse", ("csr_matrix" | "csc_matrix") as cls_name, None:
                import cupyx.scipy.sparse as cu_sparse

                return getattr(cu_sparse, cls_name)  # type: ignore[no-any-return]
            case "dask.array", "Array", _:
                if TYPE_CHECKING:
                    from dask.array.core import Array as DaskArray
                else:
                    from dask.array import Array as DaskArray

                return DaskArray  # type: ignore[return-value]
            case "h5py", "Dataset", _:
                import h5py

                return h5py.Dataset  # type: ignore[no-any-return]
            case "zarr", "Array", _:
                import zarr

                return zarr.Array  # type: ignore[return-value]
            case _:
                msg = f"Unknown array class: {self}"
                raise ValueError(msg)

    def random(
        self,
        shape: tuple[int, int],
        *,
        dtype: _DTypeLikeFloat32 | _DTypeLikeFloat64 | None,
        gen: np.random.Generator | None = None,
        # sparse only
        density: SupportsFloat = 0.01,
    ) -> Arr:
        """Create a random array."""
        gen = np.random.default_rng(gen)

        match self.mod, self.name, self.inner:
            case "numpy", "ndarray", None:
                return gen.random(shape, dtype=dtype or np.float64)  # type: ignore[return-value]
            case "scipy.sparse", (
                "csr_array" | "csc_array" | "csr_matrix" | "csc_matrix"
            ) as cls_name, None:
                fmt: Literal["csr", "csc"]
                container: Literal["array", "matrix"]
                fmt, container = cls_name.split("_")  # type: ignore[assignment]
                return random_mat(  # type: ignore[no-any-return]
                    shape, density=density, format=fmt, container=container, dtype=dtype
                )
            case "cupy", "ndarray", None:
                raise NotImplementedError
            case "cupyx.scipy.sparse", ("csr_matrix" | "csc_matrix") as cls_name, None:
                raise NotImplementedError
            case "dask.array", "Array", _:
                if TYPE_CHECKING:
                    from dask.array.wrap import zeros
                else:
                    from dask.array import zeros

                arr = zeros(shape, dtype=dtype, chunks=_half_chunk_size(shape))
                return arr.map_blocks(  # type: ignore[no-any-return]
                    lambda x: self.random(x.shape, dtype=x.dtype, gen=gen, density=density),
                    dtype=dtype,
                )
            case "h5py", "Dataset", _:
                raise NotImplementedError
            case "zarr", "Array", _:
                raise NotImplementedError
            case _:
                msg = f"Unknown array class: {self}"
                raise ValueError(msg)

    def __call__(self, x: ArrayLike, /, *, dtype: DTypeLike | None = None) -> Arr:
        """Convert to this array type."""
        from fast_array_utils import types

        fn: ToArray[Arr]
        if self.cls is np.ndarray:
            fn = np.asarray  # type: ignore[assignment]
        elif self.cls is types.DaskArray:
            if self.inner is None:
                msg = "Cannot convert to dask array without inner array type"
                raise AssertionError(msg)
            fn = self._to_dask_array
        elif self.cls is types.H5Dataset:
            fn = self._to_h5py_dataset
        elif self.cls is types.ZarrArray:
            fn = self._to_zarr_array
        elif self.cls is types.CupyArray:
            import cupy as cu

            fn = cu.asarray
        else:
            fn = self.cls  # type: ignore[assignment]

        return fn(x, dtype=dtype)  # type: ignore[return-value]

    def _to_dask_array(self, x: ArrayLike, /, *, dtype: DTypeLike | None = None) -> types.DaskArray:
        """Convert to a dask array."""
        if TYPE_CHECKING:
            import dask.array.core as da
        else:
            import dask.array as da

        assert self.inner is not None

        arr = self.inner(x, dtype=dtype)
        return da.from_array(arr, _half_chunk_size(arr.shape))  # type: ignore[no-untyped-call,no-any-return]

    def _to_h5py_dataset(
        self, x: ArrayLike, /, *, dtype: DTypeLike | None = None
    ) -> types.H5Dataset:
        """Convert to a h5py dataset."""
        if (ctx := self.conversion_context) is None:
            msg = "`conversion_context` must be set for h5py"
            raise RuntimeError(msg)
        arr = np.asarray(x, dtype=dtype)
        return ctx.hdf5_file.create_dataset("data", arr.shape, arr.dtype, data=arr)

    @staticmethod
    def _to_zarr_array(x: ArrayLike, /, *, dtype: DTypeLike | None = None) -> types.ZarrArray:
        """Convert to a zarr array."""
        import zarr

        arr = np.asarray(x, dtype=dtype)
        za = zarr.create_array({}, shape=arr.shape, dtype=arr.dtype)
        za[...] = arr
        return za


def random_mat(
    shape: tuple[int, int],
    *,
    density: SupportsFloat = 0.01,
    format: Literal["csr", "csc"] = "csr",  # noqa: A002
    dtype: DTypeLike | None = None,
    container: Literal["array", "matrix"] = "array",
    gen: np.random.Generator | None = None,
) -> CSBase:
    """Create a random matrix."""
    from scipy.sparse import random as random_spmat
    from scipy.sparse import random_array as random_sparr

    m, n = shape
    return (
        random_spmat(m, n, density=density, format=format, dtype=dtype, random_state=gen)
        if container == "matrix"
        else random_sparr(shape, density=density, format=format, dtype=dtype, random_state=gen)
    )


def _half_chunk_size(a: tuple[int, ...]) -> tuple[int, ...]:
    def half_rounded_up(x: int) -> int:
        div, mod = divmod(x, 2)
        return div + (mod > 0)

    return tuple(half_rounded_up(x) for x in a)
