from llvmlite.ir import IRBuilder, Value  # type: ignore[import-untyped]
from numba.core.base import BaseContext
from numba.core.extending import NativeValue
from numba.core.types import Type

def impl_ret_borrowed(
    context: BaseContext, builder: IRBuilder, typ: Type, value: Value
) -> NativeValue: ...
