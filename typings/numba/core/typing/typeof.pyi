# SPDX-License-Identifier: MPL-2.0
from enum import Enum
from functools import singledispatch

from numba.core.types import Type

class Purpose(Enum):
    argument = 1
    constant = 2

class _TypeofContext:
    purpose: Purpose

@singledispatch
def typeof_impl(val: object, c: _TypeofContext) -> Type: ...
