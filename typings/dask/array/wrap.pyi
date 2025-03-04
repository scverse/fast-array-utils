# SPDX-License-Identifier: MPL-2.0

from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from .core import Array, _Chunks

_Order: TypeAlias = Literal["C", "F", "A", "K"]

def full(
    a: ArrayLike | None = None,
    fill_value: float | np.number[Any] | None = None,
    *,
    order: _Order = "K",
    subok: bool = True,
    shape: tuple[int, ...] | None = None,
    device: Literal["cpu"] = "cpu",
    chunks: _Chunks | None = None,
    name: str | None = None,
) -> Array: ...
def empty(
    a: ArrayLike | None = None,
    *,
    dtype: DTypeLike | None = None,
    order: _Order = "K",
    subok: bool = True,
    shape: tuple[int, ...] | None = None,
    device: Literal["cpu"] = "cpu",
    chunks: _Chunks | None = None,
    name: str | None = None,
) -> Array: ...
def ones(
    a: ArrayLike | None = None,
    *,
    dtype: DTypeLike | None = None,
    order: _Order = "K",
    subok: bool = True,
    shape: tuple[int, ...] | None = None,
    device: Literal["cpu"] = "cpu",
    chunks: _Chunks | None = None,
    name: str | None = None,
) -> Array: ...
def zeros(
    a: ArrayLike | None = None,
    *,
    dtype: DTypeLike | None = None,
    order: _Order = "K",
    subok: bool = True,
    shape: tuple[int, ...] | None = None,
    device: Literal["cpu"] = "cpu",
    chunks: _Chunks | None = None,
    name: str | None = None,
) -> Array: ...
