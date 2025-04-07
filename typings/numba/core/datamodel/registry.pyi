from collections.abc import Callable
from typing import TypeVar

from numba.core.types import Type

_F = TypeVar("_F", bound=Callable[..., object])

def register_default(typecls: type[Type]) -> Callable[[_F], _F]: ...
