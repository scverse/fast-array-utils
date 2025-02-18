# SPDX-License-Identifier: MPL-2.0
"""Testing utilities."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from typing import Any, Generic, Literal, Protocol, SupportsFloat, TypeAlias, TypeVar

    from numpy.typing import ArrayLike, NDArray

    from fast_array_utils import types
    from fast_array_utils.types import CSBase

    _SCT_co = TypeVar("_SCT_co", covariant=True, bound=np.generic)
    _SCT_contra = TypeVar("_SCT_contra", contravariant=True, bound=np.generic)
    _SCT_float = TypeVar("_SCT_float", np.float32, np.float64)

    Array: TypeAlias = (
        NDArray[_SCT_co]
        | types.CSBase[_SCT_co]
        | types.CupyArray[_SCT_co]
        | types.DaskArray
        | types.H5Dataset
        | types.ZarrArray
    )

    class ToArray(Protocol, Generic[_SCT_contra]):
        """Convert to a supported array."""

        def __call__(  # noqa: D102
            self, data: ArrayLike, /, *, dtype: _SCT_contra | None = None
        ) -> Array[_SCT_contra]: ...


RE_ARRAY_QUAL = re.compile(r"(?P<mod>(?:\w+\.)*\w+)\.(?P<name>[^\[]+)(?:\[(?P<inner>[\w.]+)\])?")


def get_array_cls(qualname: str) -> type[Array[Any]]:  # noqa: PLR0911
    """Get a supported array class by qualname."""
    m = RE_ARRAY_QUAL.fullmatch(qualname)
    assert m
    match m["mod"], m["name"], m["inner"]:
        case "numpy", "ndarray", None:
            return np.ndarray
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
        case "dask.array", cls_name, _:
            if TYPE_CHECKING:
                from dask.array.core import Array as DaskArray
            else:
                from dask.array import Array as DaskArray

            return DaskArray
        case "h5py", "Dataset", _:
            import h5py

            return h5py.Dataset  # type: ignore[no-any-return]
        case "zarr", "Array", _:
            import zarr

            return zarr.Array
        case _:
            msg = f"Unknown array class: {qualname}"
            raise ValueError(msg)


def random_mat(
    shape: tuple[int, int],
    *,
    density: SupportsFloat = 0.01,
    format: Literal["csr", "csc"] = "csr",  # noqa: A002
    dtype: np.dtype[_SCT_float] | type[_SCT_float] | None = None,
    container: Literal["array", "matrix"] = "array",
    gen: np.random.Generator | None = None,
) -> CSBase[_SCT_float]:
    """Create a random matrix."""
    from scipy.sparse import random as random_spmat
    from scipy.sparse import random_array as random_sparr

    m, n = shape
    return (
        random_spmat(m, n, density=density, format=format, dtype=dtype, random_state=gen)
        if container == "matrix"
        else random_sparr(shape, density=density, format=format, dtype=dtype, random_state=gen)
    )


def random_array(
    qualname: str,
    shape: tuple[int, int],
    *,
    dtype: np.dtype[_SCT_float] | type[_SCT_float] | None,
    gen: np.random.Generator | None = None,
) -> Array[_SCT_float]:
    """Create a random array."""
    gen = np.random.default_rng(gen)

    m = RE_ARRAY_QUAL.fullmatch(qualname)
    assert m
    match m["mod"], m["name"], m["inner"]:
        case "numpy", "ndarray", None:
            return gen.random(shape, dtype=dtype or np.float64)
        case "scipy.sparse", (
            "csr_array" | "csc_array" | "csr_matrix" | "csc_matrix"
        ) as cls_name, None:
            fmt, container = cls_name.split("_")
            return random_mat(shape, format=fmt, container=container, dtype=dtype)  # type: ignore[arg-type]
        case "cupy", "ndarray", None:
            raise NotImplementedError
        case "cupyx.scipy.sparse", ("csr_matrix" | "csc_matrix") as cls_name, None:
            raise NotImplementedError
        case "dask.array", cls_name, _:
            raise NotImplementedError
        case "h5py", "Dataset", _:
            raise NotImplementedError
        case "zarr", "Array", _:
            raise NotImplementedError
        case _:
            msg = f"Unknown array class: {qualname}"
            raise ValueError(msg)
