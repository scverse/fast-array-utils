# SPDX-License-Identifier: MPL-2.0
from collections.abc import Callable
from typing import Literal

from .core import Array, _Chunk

type _Depth = int | tuple[int, ...] | dict[int, _Depth]
type _Boundary = Literal["reflect", "periodic", "nearest", "none"] | int
type _Boundaries = _Boundary | tuple[_Boundary, ...] | dict[int, _Boundary]

def map_overlap(
    func: Callable[[_Chunk], _Chunk],
    *args: Array,
    depth: _Depth | list[_Depth] | None = None,
    boundary: _Boundaries | list[_Boundaries] | None = None,
    trim: bool = True,
    align_arrays: bool = True,
    allow_rechunk: bool = True,
    **kwargs: object,
) -> Array: ...
