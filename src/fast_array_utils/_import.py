# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from dataclasses import dataclass, field
from types import UnionType
from typing import TYPE_CHECKING, Generic, ParamSpec, TypeVar, cast, overload


if TYPE_CHECKING:
    from collections.abc import Callable

P = ParamSpec("P")
R = TypeVar("R")


__all__ = ["import_by_qualname", "lazy_singledispatch"]


def import_by_qualname(qualname: str) -> object:
    from importlib import import_module

    mod_path, obj_path = qualname.split(":")

    mod = import_module(mod_path)

    # get object
    obj = mod
    for name in obj_path.split("."):
        try:
            obj = getattr(obj, name)
        except AttributeError as e:
            msg = f"Could not import {'.'.join(obj_path)} from {'.'.join(mod_path)} "
            raise ImportError(msg) from e
    return obj


@dataclass
class lazy_singledispatch(Generic[P, R]):  # noqa: N801
    fallback: Callable[P, R]

    _lazy: dict[tuple[str, str], Callable[..., R]] = field(init=False, default_factory=dict)
    _eager: dict[type | UnionType, Callable[..., R]] = field(init=False, default_factory=dict)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        for typ_, fn in self._eager.items():
            if isinstance(args[0], typ_):
                return fn(*args, **kwargs)
        for (import_qualname, host_mod_name), fn in self._lazy.items():
            for cls in type(args[0]).mro():
                if cls.__module__.startswith(host_mod_name):  # can be deeper
                    cls_reg = cast(type, import_by_qualname(import_qualname))
                    if isinstance(args[0], cls_reg):
                        return fn(*args, **kwargs)
        return self.fallback(*args, **kwargs)

    @overload
    def register(
        self, qualname_or_type: str, /, host_mod_name: str | None = None
    ) -> Callable[[Callable[..., R]], lazy_singledispatch[P, R]]: ...
    @overload
    def register(
        self, qualname_or_type: type | UnionType, /, host_mod_name: None = None
    ) -> Callable[[Callable[..., R]], lazy_singledispatch[P, R]]: ...

    def register(
        self, qualname_or_type: str | type | UnionType, /, host_mod_name: str | None = None
    ) -> Callable[[Callable[..., R]], lazy_singledispatch[P, R]]:
        def decorator(fn: Callable[..., R]) -> lazy_singledispatch[P, R]:
            match qualname_or_type, host_mod_name:
                case str(), _:
                    hmn = qualname_or_type.split(":")[0] if host_mod_name is None else host_mod_name
                    self._lazy[(qualname_or_type, hmn)] = fn
                case type() | UnionType(), None:
                    self._eager[qualname_or_type] = fn
                case _:
                    msg = f"name_or_type {qualname_or_type!r} must be a str, type, or UnionType"
                    raise TypeError(msg)
            return self

        return decorator
