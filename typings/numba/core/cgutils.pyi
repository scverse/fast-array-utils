# SPDX-License-Identifier: MPL-2.0
from llvmlite.ir import Constant, IRBuilder, Value
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
        value: NativeValue | None = None,
        ref: object = None,
    ) -> None: ...
    def _getvalue(self) -> Value: ...
    def __setattr__(self, name: str, value: Value) -> None: ...
    def __getattr__(self, name: str) -> Value: ...

def create_struct_proxy(typ: Type) -> type[_StructProxy]: ...
def alloca_once_value(builder: IRBuilder, value: Value, name: str = ..., zfill: bool = ...) -> Value: ...
