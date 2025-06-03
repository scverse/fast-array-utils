# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

import warnings
from functools import partial, singledispatch
from typing import TYPE_CHECKING, cast

import numpy as np

from .. import types
from ..typing import CpuArray, DiskArray, GpuArray  # noqa: TC001


if TYPE_CHECKING:
    from typing import Any, Literal

    from numpy.typing import NDArray


# fallback’s arg0 type has to include types of registered functions
@singledispatch
def to_dense_(
    x: CpuArray | GpuArray | DiskArray | types.DaskArray | types.sparray | types.spmatrix | types.CupySpMatrix,
    /,
    *,
    order: Literal["K", "A", "C", "F"] = "K",
    to_cpu_memory: bool = False,
) -> NDArray[Any] | types.CupyArray | types.DaskArray:
    del to_cpu_memory  # it already is
    return np.asarray(x, order=order)


@to_dense_.register(types.spmatrix | types.sparray)
def _to_dense_cs(x: types.spmatrix | types.sparray, /, *, order: Literal["K", "A", "C", "F"] = "K", to_cpu_memory: bool = False) -> NDArray[Any]:
    from . import scipy

    del to_cpu_memory  # it already is
    return scipy.to_dense(x, order=sparse_order(x, order=order))


@to_dense_.register(types.DaskArray)
def _to_dense_dask(x: types.DaskArray, /, *, order: Literal["K", "A", "C", "F"] = "K", to_cpu_memory: bool = False) -> NDArray[Any] | types.DaskArray:
    from . import to_dense

    if order == "F":
        msg = f"{order=!r} will probably be ignored: Dask can not be made to emit F-contiguous arrays reliably."
        warnings.warn(msg, RuntimeWarning, stacklevel=4)
    x = x.map_blocks(partial(to_dense, order=order, to_cpu_memory=to_cpu_memory))
    return x.compute() if to_cpu_memory else x  # type: ignore[return-value]


@to_dense_.register(types.CSDataset)
def _to_dense_ooc(x: types.CSDataset, /, *, order: Literal["K", "A", "C", "F"] = "K", to_cpu_memory: bool = False) -> NDArray[Any]:
    from . import to_dense

    if not to_cpu_memory:
        msg = "to_cpu_memory must be True if x is an CS{R,C}Dataset"
        raise ValueError(msg)
    # TODO(flying-sheep): why is to_memory of type Any?  # noqa: TD003
    return to_dense(cast("types.CSBase", x.to_memory()), order=sparse_order(x, order=order))


@to_dense_.register(types.CupyArray | types.CupySpMatrix)
def _to_dense_cupy(x: GpuArray, /, *, order: Literal["K", "A", "C", "F"] = "K", to_cpu_memory: bool = False) -> NDArray[Any] | types.CupyArray:
    import cupy as cu

    x = x.toarray(sparse_order(x, order=order)) if isinstance(x, types.CupySpMatrix) else cu.asarray(x, order=order)
    return x.get(order="A") if to_cpu_memory else x


def sparse_order(x: types.spmatrix | types.sparray | types.CupySpMatrix | types.CSDataset, /, *, order: Literal["K", "A", "C", "F"]) -> Literal["C", "F"]:
    if TYPE_CHECKING:
        from scipy.sparse._base import _spbase

        assert isinstance(x, _spbase | types.CSDataset)

    if order in {"K", "A"}:
        order = "F" if x.format == "csc" else "C"
    return cast("Literal['C', 'F']", order)
