# SPDX-License-Identifier: MPL-2.0
from collections.abc import Callable, Iterable
from typing import Literal, SupportsIndex, overload

from .core.types import *

type __Signature = str | Type
type _Signature = str | Type | tuple[__Signature, ...]

# https://numba.readthedocs.io/en/stable/reference/jit-compilation.html#numba.jit
@overload
def njit[F: Callable[..., object]](f: F) -> F: ...
@overload
def njit[F: Callable[..., object]](
    signature: _Signature | list[_Signature] | None = None,
    *,
    nopython: bool = True,
    nogil: bool = False,
    cache: bool = False,
    forceobj: bool = False,
    parallel: bool = False,
    error_model: Literal["python", "numpy"] = "python",
    fastmath: bool = False,
    locals: dict[str, object] = {},
    boundscheck: bool = False,
) -> Callable[[F], F]: ...
@overload
def prange(stop: SupportsIndex, /) -> Iterable[int]: ...
@overload
def prange(start: SupportsIndex, stop: SupportsIndex, step: SupportsIndex = ..., /) -> Iterable[int]: ...
def get_num_threads() -> int: ...
