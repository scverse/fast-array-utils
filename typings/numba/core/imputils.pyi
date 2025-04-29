# SPDX-License-Identifier: MPL-2.0
from llvmlite.ir import IRBuilder, Value
from numba.core.base import BaseContext
from numba.core.extending import NativeValue
from numba.core.types import Type

def impl_ret_borrowed(context: BaseContext, builder: IRBuilder, typ: Type, value: Value) -> NativeValue: ...
