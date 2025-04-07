import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import Concatenate, ParamSpec, Protocol, TypeVar

from llvmlite.ir import Constant, Instruction, IRBuilder, Value
from numba.core.base import BaseContext
from numba.core.datamodel import models
from numba.core.datamodel import register_default as register_model
from numba.core.pythonapi import PythonAPI
from numba.core.types import Type
from numba.core.typing.templates import Signature
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

    def box(self, typ: Type, val: NativeValue) -> object: ...

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
    [Callable[[_T, NativeValue, BoxContext[_T]], _R]],
    Callable[[_T, NativeValue, BoxContext[_T]], _R],
]: ...
def unbox(
    typeclass: type[_T],
) -> Callable[
    [Callable[[_T, object, UnboxContext[_T]], NativeValue]],
    Callable[[_T, object, UnboxContext[_T]], NativeValue],
]: ...
@typing.overload
def intrinsic(
    func: Callable[Concatenate[TypingContext, _P], tuple[Signature, Callable[..., NativeValue]]],
    /,
) -> Callable[_P, _R]: ...
@typing.overload
def intrinsic(
    *,
    prefer_literal: bool = False,
    **kwargs: object,
) -> Callable[
    [Callable[Concatenate[TypingContext, _P], tuple[Signature, Callable[..., NativeValue]]]],
    Callable[_P, _R],
]: ...
def make_attribute_wrapper(typecls: type[Type], attr1: str, attr2: str) -> None: ...
def overload(f: Callable[..., object]) -> Callable[[_F], _F]: ...
def overload_method(typecls: type[Type], name: str) -> Callable[[_F], _F]: ...
def overload_attribute(typecls: type[Type], name: str) -> Callable[[_F], _F]: ...
