# SPDX-License-Identifier: MPL-2.0
import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import Concatenate, Protocol

from llvmlite.ir import IRBuilder, Value
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

TypingContext = object

class _Context(Protocol):
    # https://numba.readthedocs.io/en/stable/extending/low-level.html#boxing-and-unboxing
    context: BaseContext
    builder: IRBuilder
    pyapi: PythonAPI

class BoxContext(_Context, Protocol):
    env_manager: object

    def box(self, typ: Type, val: Value) -> Value: ...

class UnboxContext(_Context, Protocol):
    def unbox(self, typ: Type, obj: Value) -> NativeValue: ...

@dataclass
class NativeValue:
    value: Value
    is_error: Value = ...
    cleanup: Value | None = None

def box[T: Type, R](
    typeclass: type[T],
) -> Callable[
    [Callable[[T, NativeValue, BoxContext], R]],
    Callable[[T, NativeValue, BoxContext], R],
]: ...
def unbox[T: Type](
    typeclass: type[T],
) -> Callable[
    [Callable[[T, Value, UnboxContext], NativeValue]],
    Callable[[T, Value, UnboxContext], NativeValue],
]: ...
@typing.overload
def intrinsic[**P](
    func: Callable[Concatenate[TypingContext, P], tuple[Signature, Callable[..., NativeValue]]],
    /,
) -> Callable[P, object]: ...
@typing.overload
def intrinsic[**P](
    *,
    prefer_literal: bool = False,
    **kwargs: object,
) -> Callable[
    [Callable[Concatenate[TypingContext, P], tuple[Signature, Callable[..., NativeValue]]]],
    Callable[P, object],
]: ...
def make_attribute_wrapper(typeclass: type[Type], struct_attr: str, python_attr: str) -> None: ...
def overload[F: Callable[..., object]](f: Callable[..., object]) -> Callable[[F], F]: ...
def overload_method[F: Callable[..., object]](typecls: type[Type], name: str) -> Callable[[F], F]: ...
def overload_attribute[F: Callable[..., object]](typecls: type[Type], name: str) -> Callable[[F], F]: ...
