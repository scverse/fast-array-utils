# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from functools import partial, singledispatch
from typing import TYPE_CHECKING, cast

import numpy as np

from .. import types
from ..typing import CpuArray, DiskArray, GpuArray  # noqa: TC001


if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray


# fallback’s arg0 type has to include types of registered functions
@singledispatch
def to_dense_(
    x: CpuArray
    | GpuArray
    | DiskArray
    | types.DaskArray
    | types.sparray
    | types.spmatrix
    | types.CupySpMatrix,
    /,
    *,
    to_cpu_memory: bool = False,
) -> NDArray[Any] | types.CupyArray | types.DaskArray:
    del to_cpu_memory  # it already is
    return np.asarray(x)


@to_dense_.register(types.spmatrix | types.sparray)  # type: ignore[call-overload,misc]
def _to_dense_cs(
    x: types.spmatrix | types.sparray, /, *, to_cpu_memory: bool = False
) -> NDArray[Any]:
    from . import scipy

    del to_cpu_memory  # it already is
    return scipy.to_dense(x)


@to_dense_.register(types.DaskArray)
def _to_dense_dask(
    x: types.DaskArray, /, *, to_cpu_memory: bool = False
) -> NDArray[Any] | types.DaskArray:
    from . import to_dense

    x = x.map_blocks(partial(to_dense, to_cpu_memory=to_cpu_memory))
    return x.compute() if to_cpu_memory else x  # type: ignore[return-value]


@to_dense_.register(types.CSDataset)
def _to_dense_ooc(x: types.CSDataset, /, *, to_cpu_memory: bool = False) -> NDArray[Any]:
    from . import to_dense

    if not to_cpu_memory:
        msg = "to_cpu_memory must be True if x is an CS{R,C}Dataset"
        raise ValueError(msg)
    # TODO(flying-sheep): why is to_memory of type Any?  # noqa: TD003
    return to_dense(cast("types.CSBase", x.to_memory()))


@to_dense_.register(types.CupyArray | types.CupySpMatrix)  # type: ignore[call-overload,misc]
def _to_dense_cupy(
    x: GpuArray, /, *, to_cpu_memory: bool = False
) -> NDArray[Any] | types.CupyArray:
    x = x.toarray() if isinstance(x, types.CupySpMatrix) else x
    return x.get() if to_cpu_memory else x
