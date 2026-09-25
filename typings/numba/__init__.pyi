# SPDX-License-Identifier: MPL-2.0
from collections.abc import Iterable
from typing import SupportsIndex, overload

from .core import config as config
from .core.decorators import njit as njit

@overload
def prange(stop: SupportsIndex, /) -> Iterable[int]: ...
@overload
def prange(start: SupportsIndex, stop: SupportsIndex, step: SupportsIndex = ..., /) -> Iterable[int]: ...
def get_num_threads() -> int: ...
