# SPDX-License-Identifier: MPL-2.0
"""Conversion utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

from ..typing import CpuArray, DiskArray, GpuArray  # noqa: TC001
from ._to_dense import to_dense_


if TYPE_CHECKING:
    from typing import Any, Literal

    from numpy.typing import NDArray

    from .. import types


__all__ = ["to_dense"]


@overload
def to_dense(
    x: CpuArray | DiskArray | types.sparray | types.spmatrix | types.CSDataset, /, *, order: Literal["K", "A", "C", "F"] = "K", to_cpu_memory: bool = False
) -> NDArray[Any]: ...


@overload
def to_dense(x: types.DaskArray, /, *, order: Literal["K", "A", "C", "F"] = "K", to_cpu_memory: Literal[False] = False) -> types.DaskArray: ...
@overload
def to_dense(x: types.DaskArray, /, *, order: Literal["K", "A", "C", "F"] = "K", to_cpu_memory: Literal[True]) -> NDArray[Any]: ...


@overload
def to_dense(x: GpuArray | types.CupySpMatrix, /, *, order: Literal["K", "A", "C", "F"] = "K", to_cpu_memory: Literal[False] = False) -> types.CupyArray: ...
@overload
def to_dense(x: GpuArray | types.CupySpMatrix, /, *, order: Literal["K", "A", "C", "F"] = "K", to_cpu_memory: Literal[True]) -> NDArray[Any]: ...


def to_dense(
    x: CpuArray | GpuArray | DiskArray | types.CSDataset | types.DaskArray | types.sparray | types.spmatrix | types.CupySpMatrix,
    /,
    *,
    order: Literal["K", "A", "C", "F"] = "K",
    to_cpu_memory: bool = False,
) -> NDArray[Any] | types.DaskArray | types.CupyArray:
    r"""Convert x to a dense array.

    If ``to_cpu_memory`` is :data:`False`, :class:`dask.array.Array`\ s and
    :class:`cupy.ndarray`\ s/:class:`cupyx.scipy.sparse.spmatrix` instances
    stay out-of-core and in GPU memory, respecively.

    Parameters
    ----------
    x
        Input object to be converted.
    order
        The order of the output array: ``C`` (row-major) or ``F`` (column-major). ``K`` and ``A`` derive the order from ``x``.

        The default matches numpy, and therefore diverges from the ``scipy.sparse`` matrices’
        :meth:`~scipy.sparse.csr_array.toarray`\ ’s default behavior
        of always returning a ``C``-contiguous array.
        Instead, CSC matrices become F-contiguous arrays when ``order="K"`` (the default).

        Dask :class:`~dask.array.Array`\ s concatenation behavior will result in ``order``
        having no effect on the :func:`dask.compute` / ``to_cpu_memory=True`` result.
    to_cpu_memory
        Also load data into memory (resulting in a :class:`numpy.ndarray`).

    Returns
    -------
    Dense form of ``x``

    """
    return to_dense_(x, order=order, to_cpu_memory=to_cpu_memory)
