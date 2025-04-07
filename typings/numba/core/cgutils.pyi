from llvmlite.ir import Constant, IRBuilder, Value  # type: ignore[import-untyped]
from numba.core.base import BaseContext  # type: ignore[import-untyped]
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
def alloca_once_value(
    builder: object, value: object, name: str = ..., zfill: bool = ...
) -> object: ...
