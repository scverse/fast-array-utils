# SPDX-License-Identifier: MPL-2.0
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager

from llvmlite.ir import Constant, Function, FunctionType, IRBuilder, Module, Value
from llvmlite.ir import Type as LLType
from numba.core.base import BaseContext
from numba.core.extending import NativeValue
from numba.core.types import Type

true_bit: Constant
false_bit: Constant
true_byte: Constant
false_byte: Constant

class _StructProxy:
    def __init__(
        self,
        context: BaseContext,
        builder: IRBuilder,
        value: NativeValue | Value | None = None,
        ref: object = None,
    ) -> None: ...
    def _getvalue(self) -> Value: ...
    def __setattr__(self, name: str, value: Value) -> None: ...
    def __getattr__(self, name: str) -> Value: ...

def create_struct_proxy(typ: Type) -> type[_StructProxy]: ...
def alloca_once_value(builder: IRBuilder, value: Value, name: str = ..., zfill: bool = ...) -> Value: ...
def alloca_once(builder: IRBuilder, ty: LLType, size: Value | None = None, name: str = "", zfill: bool = False) -> Value: ...
def get_or_insert_function(module: Module, fnty: FunctionType, name: str) -> Function: ...
def get_item_pointer(
    context: BaseContext, builder: IRBuilder, aryty: Type, ary: _StructProxy, inds: Sequence[Value], wraparound: bool = False, boundscheck: bool = False
) -> Value: ...

class _Loop:
    index: Value
    do_break: Callable[[], None]

def for_range(builder: IRBuilder, count: Value, start: Value | None = None, intp: LLType | None = None) -> AbstractContextManager[_Loop]: ...
def for_range_slice(
    builder: IRBuilder, start: Value, stop: Value, step: Value, intp: LLType | None = None, inc: bool = True
) -> AbstractContextManager[tuple[Value, Value]]: ...
