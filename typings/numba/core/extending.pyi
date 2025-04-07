import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Concatenate, ParamSpec, Protocol, TypeVar

from llvmlite.ir import Constant, Instruction, IRBuilder, Value  # type: ignore[import-untyped]
from numba.core.base import BaseContext  # type: ignore[import-untyped]
from numba.core.datamodel import models
from numba.core.datamodel import register_default as register_model
from numba.core.pythonapi import PythonAPI
from numba.core.types import Type
from numba.core.typing.templates import Signature  # type: ignore[import-untyped]
from numba.core.typing.typeof import typeof_impl

__all__ = [
    "NativeValue",
    "TypingContext",
    "box",
    "intrinsic",
    "make_attribute_wrapper",
    "models",
    "overload",
    "overload_attribute",
    "overload_method",
    "register_model",
    "typeof_impl",
    "unbox",
]

_F = TypeVar("_F", bound=Callable[..., object])
_P = ParamSpec("_P")
_R = TypeVar("_R")
_T = TypeVar("_T", bound=Type)

TypingContext = object

class _Context(Protocol):
    # https://numba.readthedocs.io/en/stable/extending/low-level.html#boxing-and-unboxing
    context: BaseContext
    builder: IRBuilder
    pyapi: PythonAPI

class BoxContext(_Context, Protocol):
    env_manager: object

    def box(self, typ: Type, val: Value) -> object: ...

class UnboxContext(_Context, Protocol):
    def unbox(self, typ: Type, obj: object) -> NativeValue: ...

@dataclass
class NativeValue:
    value: Value
    is_error: Constant | Instruction = ...
    cleanup: Instruction | None = None

def box(
    typeclass: type[_T],
) -> Callable[
    [Callable[[_T, NativeValue, BoxContext], _R]],
    Callable[[_T, NativeValue, BoxContext], _R],
]: ...
def unbox(
    typeclass: type[_T],
) -> Callable[
    [Callable[[_T, Any, UnboxContext], NativeValue]],
    Callable[[_T, Any, UnboxContext], NativeValue],
]: ...
@typing.overload
def intrinsic(
    func: Callable[Concatenate[TypingContext, _P], tuple[Signature, Callable[..., NativeValue]]],
    /,
) -> Callable[_P, object]: ...
@typing.overload
def intrinsic(
    *,
    prefer_literal: bool = False,
    **kwargs: object,
) -> Callable[
    [Callable[Concatenate[TypingContext, _P], tuple[Signature, Callable[..., NativeValue]]]],
    Callable[_P, object],
]: ...
def make_attribute_wrapper(typeclass: type[Type], struct_attr: str, python_attr: str) -> None: ...
def overload(f: Callable[..., object]) -> Callable[[_F], _F]: ...
def overload_method(typecls: type[Type], name: str) -> Callable[[_F], _F]: ...
def overload_attribute(typecls: type[Type], name: str) -> Callable[[_F], _F]: ...
