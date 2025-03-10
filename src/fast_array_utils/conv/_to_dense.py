# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from functools import singledispatch
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
    x: CpuArray | GpuArray | DiskArray | types.DaskArray, /, *, to_memory: bool = False
) -> NDArray[Any] | types.CupyArray | types.DaskArray:
    del to_memory  # it already is
    return np.asarray(x)


@to_dense_.register(types.CSBase)  # type: ignore[call-overload,misc]
def _to_dense_cs(x: types.CSBase, /, *, to_memory: bool = False) -> NDArray[Any]:
    from . import scipy

    del to_memory  # it already is
    return scipy.to_dense(x)


@to_dense_.register(types.DaskArray)
def _to_dense_dask(
    x: types.DaskArray, /, *, to_memory: bool = False
) -> NDArray[Any] | types.DaskArray:
    import dask.array as da

    from . import to_dense

    x = da.map_blocks(to_dense, x)
    return x.compute() if to_memory else x  # type: ignore[return-value]


@to_dense_.register(types.CSDataset)
def _to_dense_ooc(x: types.CSDataset, /, *, to_memory: bool = False) -> NDArray[Any]:
    from . import to_dense

    if not to_memory:
        msg = "to_memory must be True if x is an CS{R,C}Dataset"
        raise ValueError(msg)
    # TODO(flying-sheep): why is to_memory of type Any?  # noqa: TD003
    return to_dense(cast("types.CSBase", x.to_memory()))


@to_dense_.register(GpuArray)  # type: ignore[call-overload,misc]
def _to_dense_cupy(x: GpuArray, /, *, to_memory: bool = False) -> NDArray[Any] | types.CupyArray:
    x = x.toarray() if isinstance(x, types.CupySparseMatrix) else x
    return x.get() if to_memory else x
