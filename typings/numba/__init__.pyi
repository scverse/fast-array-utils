# SPDX-License-Identifier: MPL-2.0
from collections.abc import Callable, Iterable
from typing import Literal, SupportsIndex, TypeAlias, TypeVar, overload

from .core.types import *

_F = TypeVar("_F", bound=Callable[..., object])

_Signature: TypeAlias = str | Type | tuple[_Signature, ...]

# https://numba.readthedocs.io/en/stable/reference/jit-compilation.html#numba.jit
@overload
def njit(f: _F) -> _F: ...
@overload
def njit(
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
) -> Callable[[_F], _F]: ...
@overload
def prange(stop: SupportsIndex, /) -> Iterable[int]: ...
@overload
def prange(start: SupportsIndex, stop: SupportsIndex, step: SupportsIndex = ..., /) -> Iterable[int]: ...
def get_num_threads() -> int: ...
