# SPDX-License-Identifier: MPL-2.0
from collections.abc import Callable
from typing import TypeVar

from ..types import Type

_F = TypeVar("_F", bound=Callable[..., object])

def register_default(typecls: type[Type]) -> Callable[[_F], _F]: ...
