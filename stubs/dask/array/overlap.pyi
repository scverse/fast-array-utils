from collections.abc import Callable
from typing import Literal, TypeAlias

from .core import Array

_Depth: TypeAlias = int | tuple[int, ...] | dict[int, _Depth]
_Boundary: TypeAlias = Literal["reflect", "periodic", "nearest", "none"] | int
_Boundaries: TypeAlias = _Boundary | tuple[_Boundary, ...] | dict[int, _Boundary]

def map_overlap(
    func: Callable[[Array], Array],
    *args: Array,
    depth: _Depth | list[_Depth] | None = None,
    boundary: _Boundaries | list[_Boundaries] | None = None,
    trim: bool = True,
    align_arrays: bool = True,
    allow_rechunk: bool = True,
    **kwargs: object,
) -> Array: ...
