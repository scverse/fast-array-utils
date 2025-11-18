# SPDX-License-Identifier: MPL-2.0
from collections.abc import Callable

from ..types import Type

def register_default[F: Callable[..., object]](typecls: type[Type]) -> Callable[[F], F]: ...
