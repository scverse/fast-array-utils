# SPDX-License-Identifier: MPL-2.0
"""ArrayType class and helpers."""

from __future__ import annotations

import enum
from dataclasses import KW_ONLY, dataclass, field
from functools import cached_property, partial
from importlib.metadata import version
from typing import TYPE_CHECKING, Generic, TypedDict, TypeVar, cast

import numpy as np
from packaging.version import Version

from fast_array_utils import types
from fast_array_utils.conv import to_dense


if TYPE_CHECKING:
    from typing import Any, Literal, Protocol, TypeAlias

    import h5py
    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from fast_array_utils.typing import CpuArray, DiskArray, GpuArray

    InnerArray = CpuArray | GpuArray | DiskArray
    Array: TypeAlias = InnerArray | types.DaskArray | types.CSDataset
    ExtendedArray = Array | types.COOBase | types.CupyCOOMatrix

    Arr = TypeVar("Arr", bound=ExtendedArray, default=Array)
    Arr_co = TypeVar("Arr_co", bound=ExtendedArray, covariant=True)

    Inner = TypeVar("Inner", bound="ArrayType[InnerArray, None] | None", default=Any)

    class ToArray(Protocol, Generic[Arr_co]):
        """Convert to a supported array."""

        def __call__(self, data: ArrayLike | Array, /, *, dtype: DTypeLike | None = None) -> Arr_co: ...

    class MkArray(Protocol):
        def __call__(self, shape: tuple[int, int], /, *, dtype: DTypeLike | None = None) -> Array: ...

    _DTypeLikeFloat32 = np.dtype[np.float32] | type[np.float32]
    _DTypeLikeFloat64 = np.dtype[np.float64] | type[np.float64]
    _DTypeLikeInt32 = np.dtype[np.int32] | type[np.int32]
    _DTypeLikeIn64 = np.dtype[np.int64] | type[np.int64]
    _DTypeLikeNum = _DTypeLikeFloat32 | _DTypeLikeFloat64 | _DTypeLikeInt32 | _DTypeLikeIn64
else:
    Arr = TypeVar("Arr")
    Inner = TypeVar("Inner")


__all__ = ["ArrayType", "ConversionContext", "ToArray"]


class Flags(enum.Flag):
    """Array classification flags."""

    None_ = 0
    """No array type."""
    Any = enum.auto()
    """Any array type."""

    Sparse = enum.auto()
    """Sparse array."""
    Matrix = enum.auto()
    """Matrix API (``A * B`` means ``A @ B``)."""
    Gpu = enum.auto()
    """GPU array."""
    Dask = enum.auto()
    """Dask array."""
    Disk = enum.auto()
    """On-disk array."""


@dataclass
class ConversionContext:
    """Conversion context required for h5py."""

    hdf5_file: h5py.File  # TODO(flying-sheep): ReadOnly <https://peps.python.org/pep-0767/>


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
                return cast("type[Arr]", np.ndarray)
            case "scipy.sparse", ("csr_array" | "csc_array" | "coo_array" | "csr_matrix" | "csc_matrix" | "coo_matrix") as cls_name, None:
                import scipy.sparse

                return cast("type[Arr]", getattr(scipy.sparse, cls_name))
            case "cupy", "ndarray", None:
                import cupy as cp

                return cast("type[Arr]", cp.ndarray)
            case "cupyx.scipy.sparse", ("csr_matrix" | "csc_matrix" | "coo_matrix") as cls_name, None:
                import cupyx.scipy.sparse as cu_sparse

                return cast("type[Arr]", getattr(cu_sparse, cls_name))
            case "dask.array", "Array", _:
                import dask.array as da

                return cast("type[Arr]", da.Array)
            case "h5py", "Dataset", _:
                import h5py

                return cast("type[Arr]", h5py.Dataset)
            case "zarr", "Array", _:
                import zarr

                return cast("type[Arr]", zarr.Array)
            case "anndata.abc", ("CSCDataset" | "CSRDataset") as cls_name, _:
                import anndata.abc  # type: ignore[import-untyped]

                return cast("type[Arr]", getattr(anndata.abc, cls_name))
            case _:
                msg = f"Unknown array class: {self}"
                raise ValueError(msg)

    def random(
        self,
        shape: tuple[int, int],
        *,
        dtype: _DTypeLikeNum | None = None,
        gen: np.random.Generator | None = None,
        # sparse only
        density: float | np.floating[Any] = 0.01,
    ) -> Arr:
        """Create a random array."""
        gen = np.random.default_rng(gen)

        match self.mod, self.name, self.inner:
            case "numpy", "ndarray", None:
                return cast("Arr", random_array(shape, dtype=dtype, rng=gen))
            case "scipy.sparse", ("csr_array" | "csc_array" | "csr_matrix" | "csc_matrix") as cls_name, None:
                fmt, container = cast('tuple[Literal["csr", "csc"], Literal["array", "matrix"]]', cls_name.split("_"))
                return cast(
                    "Arr",
                    random_mat(shape, density=density, format=fmt, container=container, dtype=dtype),
                )
            case "cupy", "ndarray", None:
                return self(random_array(shape, dtype=dtype, rng=gen))
            case "cupyx.scipy.sparse", ("csr_matrix" | "csc_matrix") as cls_name, None:
                import cupy as cu

                fmt = cast('Literal["csr", "csc"]', cls_name[:3])
                m = random_mat(shape, density=density, format=fmt, container="matrix", dtype=dtype)
                d, i, p = tuple(cu.asarray(p) for p in (m.data, m.indices, m.indptr))
                cls = cast("type[types.CupyCSMatrix]", self.cls)
                return cast("Arr", cls((d, i, p), shape=shape))
            case "dask.array", "Array", _:
                import dask.array as da

                arr = da.zeros(shape, dtype=dtype, chunks=_half_chunk_size(shape))
                return cast(
                    "Arr",
                    arr.map_blocks(
                        lambda x: self.random(x.shape, dtype=x.dtype, gen=gen, density=density),  # type: ignore[attr-defined]
                        dtype=dtype,
                    ),
                )
            case "h5py", "Dataset", _:
                raise NotImplementedError
            case "zarr", "Array", _:
                raise NotImplementedError
            case "anndata.abc", ("CSCDataset" | "CSRDataset"), _:
                raise NotImplementedError
            case _:
                msg = f"Unknown array class: {self}"
                raise ValueError(msg)

    def __call__(self, x: ArrayLike | Array, /, *, dtype: DTypeLike | None = None) -> Arr:
        """Convert to this array type."""
        fn: ToArray[Arr]
        if self.cls is np.ndarray:
            fn = cast("ToArray[Arr]", self._to_numpy_array)
        elif issubclass(self.cls, (types.spmatrix, types.sparray)):
            fn = cast("ToArray[Arr]", self._to_scipy_sparse)
        elif self.cls is types.DaskArray:
            if self.inner is None:
                msg = "Cannot convert to dask array without inner array type"
                raise AssertionError(msg)
            fn = cast("ToArray[Arr]", self._to_dask_array)
        elif self.cls is types.H5Dataset:
            fn = cast("ToArray[Arr]", self._to_h5py_dataset)
        elif self.cls is types.ZarrArray:
            fn = cast("ToArray[Arr]", self._to_zarr_array)
        elif self.cls in {types.CSCDataset, types.CSRDataset}:
            fn = cast("ToArray[Arr]", self._to_cs_dataset)
        elif self.cls is types.CupyArray:
            fn = cast("ToArray[Arr]", self._to_cupy_array)
        elif issubclass(self.cls, types.CupySpMatrix):
            fn = cast("ToArray[Arr]", self._to_cupy_sparse)
        else:
            fn = cast("ToArray[Arr]", self.cls)

        return fn(x, dtype=dtype)

    @staticmethod
    def _to_numpy_array(x: ArrayLike | Array, /, *, dtype: DTypeLike | None = None) -> NDArray[np.number[Any]]:
        """Convert to a numpy array."""
        x = to_dense(x, to_cpu_memory=True)
        return x if dtype is None else x.astype(dtype)

    def _to_dask_array(self, x: ArrayLike | Array, /, *, dtype: DTypeLike | None = None) -> types.DaskArray:
        """Convert to a dask array."""
        import dask.array as da

        assert self.inner is not None
        if TYPE_CHECKING:
            assert isinstance(self.inner, ArrayType[CpuArray | GpuArray, None])  # type: ignore[misc]

        if isinstance(x, types.DaskArray):
            if isinstance(x._meta, self.inner.cls):  # noqa: SLF001
                return x
            return x.map_blocks(self.inner, dtype=dtype, meta=self.inner([[1]], dtype=dtype or x.dtype))

        arr = self.inner(x, dtype=dtype)
        return da.from_array(arr, _half_chunk_size(arr.shape))

    def _to_h5py_dataset(self, x: ArrayLike | Array, /, *, dtype: DTypeLike | None = None) -> types.H5Dataset:
        """Convert to a h5py dataset."""
        if (ctx := self.conversion_context) is None:
            msg = "`conversion_context` must be set for h5py"
            raise RuntimeError(msg)

        arr = self._to_numpy_array(x, dtype=dtype)
        return ctx.hdf5_file.create_dataset("data", arr.shape, arr.dtype, data=arr)

    @classmethod
    def _to_zarr_array(cls, x: ArrayLike | Array, /, *, dtype: DTypeLike | None = None) -> types.ZarrArray:
        """Convert to a zarr array."""
        import zarr

        arr = cls._to_numpy_array(x, dtype=dtype)
        if Version(version("zarr")) >= Version("3"):
            za = zarr.create_array({}, shape=arr.shape, dtype=arr.dtype)
        else:
            za = zarr.create(shape=arr.shape, dtype=arr.dtype)
        za[...] = arr
        return za

    def _to_cs_dataset(self, x: ArrayLike | Array, /, *, dtype: DTypeLike | None = None) -> types.CSDataset:
        """Convert to a scipy sparse dataset."""
        import anndata.io  # type: ignore[import-untyped]
        from scipy.sparse import csc_array, csr_array

        assert self.inner is not None

        grp: types.H5Group | types.ZarrGroup
        if self.inner.cls is types.ZarrArray:
            import zarr

            grp = zarr.group()
        elif self.inner.cls is types.H5Dataset:
            if (ctx := self.conversion_context) is None:
                msg = "`conversion_context` must be set for h5py"
                raise RuntimeError(msg)
            grp = ctx.hdf5_file
        else:
            raise NotImplementedError

        cls = cast("type[types.csr_array[Any, tuple[int, int]] | types.csc_array]", csr_array if self.cls is types.CSRDataset else csc_array)
        x_sparse = self._to_scipy_sparse(x, dtype=dtype, cls=cls)
        anndata.io.write_elem(grp, "/", x_sparse)
        return anndata.io.sparse_dataset(grp)

    def _to_scipy_sparse(
        self,
        x: ArrayLike | Array | types.spmatrix | types.sparray | types.CupySpMatrix,
        /,
        *,
        dtype: DTypeLike | None = None,
        cls: type[types.CSBase] | None = None,
    ) -> types.CSBase:
        """Convert to a scipy sparse matrix/array."""
        if isinstance(x, types.DaskArray):
            x = x.compute()
        if isinstance(x, types.CupySpMatrix):
            x = x.get()  # can be a coo_matrix due to dask concatenation
        elif not isinstance(x, types.spmatrix | types.sparray | np.ndarray):
            x = to_dense(x, to_cpu_memory=True)

        cls = cast("type[types.CSBase]", cls or self.cls)
        return cls(x, dtype=dtype)  # type: ignore[arg-type]

    def _to_cupy_array(self, x: ArrayLike | Array, /, *, dtype: DTypeLike | None = None) -> types.CupyArray:
        import cupy as cu

        if isinstance(x, types.DaskArray):
            x = x.compute()  # this could now be a cupy array already
        if isinstance(x, types.CupySpMatrix):
            x = x.toarray()
        if isinstance(x, types.CSDataset | types.CSBase):
            x = to_dense(x, to_cpu_memory=True)

        return cu.asarray(x, dtype=None if dtype is None else np.dtype(dtype))

    def _to_cupy_sparse(
        self,
        x: ArrayLike | Array | types.spmatrix | types.sparray | types.CupySpMatrix,
        /,
        *,
        dtype: DTypeLike | None = None,
    ) -> types.CupyCSMatrix:
        if not isinstance(x, types.spmatrix | types.sparray | types.CupyArray | types.CupySpMatrix):
            x = self._to_cupy_array(x, dtype=dtype)

        return self.cls(x)  # type: ignore[call-arg,arg-type, return-value]


def random_array(
    shape: tuple[int, int],
    *,
    dtype: _DTypeLikeNum | None = None,
    rng: np.random.Generator | None = None,
) -> Array:
    """Create a random array."""
    rng = np.random.default_rng(rng)
    f: MkArray
    match np.dtype(dtype or "f").kind:
        case "f":
            f = rng.random  # type: ignore[assignment]
        case "i" | "u":
            f = partial(rng.integers, 0, 10_000)
        case _:
            raise NotImplementedError
    return f(shape, dtype=dtype)


def random_mat(
    shape: tuple[int, int],
    *,
    density: float | np.floating[Any] = 0.01,
    format: Literal["csr", "csc"] = "csr",  # noqa: A002
    dtype: _DTypeLikeNum | None = None,
    container: Literal["array", "matrix"] = "array",
    rng: np.random.Generator | None = None,
) -> types.CSBase:
    """Create a random sparse matrix/array."""
    from scipy.sparse import random as random_spmat
    from scipy.sparse import random_array as random_sparr

    m, n = shape
    return cast(
        "types.CSBase",
        random_spmat(m, n, density=density, format=format, dtype=dtype, **_rng_kw(rng))
        if container == "matrix"
        else random_sparr(shape, density=density, format=format, dtype=dtype, **_rng_kw(rng)),
    )


class RngKw(TypedDict):
    rng: np.random.Generator | None


def _rng_kw(rng: np.random.Generator | None) -> RngKw:
    return RngKw(rng=rng) if Version(version("scipy")) >= Version("1.15") else cast("RngKw", dict(random_state=rng))


def _half_chunk_size(a: tuple[int, ...]) -> tuple[int, ...]:
    def half_rounded_up(x: int) -> int:
        div, mod = divmod(x, 2)
        return div + (mod > 0)

    return tuple(half_rounded_up(x) for x in a)
