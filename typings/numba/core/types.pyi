# SPDX-License-Identifier: MPL-2.0
# See <https://numba.readthedocs.io/en/stable/reference/types.html#numba-types>

from typing import Literal, Self

from numba.core.typing.templates import Signature

class Type:
    def __init__(self, name: str) -> None: ...
    def __call__(self, *args: object, **kwds: object) -> Signature: ...

class Boolean(Type): ...
class Integer(Type): ...
class Float(Type): ...
class Complex(Type): ...
class NoneType(Type): ...

boolean: Boolean
uint8: Integer
byte = uint8
uint16: Integer
uint32: Integer
uint64: Integer
int8: Integer
char = int8
int16: Integer
int32: Integer
int64: Integer
intc: Integer
uintc: Integer
intp: Integer
uintp: Integer
ssize_t: Integer
size_t: Integer
float32: Float
float64: Float
double = float64
complex64: Complex
complex128: Complex
void: NoneType

# composite
class IterableType(Type): ...
class Sized(Type): ...

class ArrayCompatible(Type):
    ndim: int
    layout: Buffer.Layout
    dtype: Type

class Buffer(IterableType, ArrayCompatible):
    type Layout = Literal["C", "F", "CS", "FS", "A"]
    LAYOUTS: frozenset[Layout]
    def copy(
        self,
        dtype: Type | type[Type] | None = None,
        ndim: int | None = None,
        layout: Layout | None = None,
    ) -> Self: ...

class Array(Buffer):
    def __init__(
        self,
        dtype: Type | type[Type],
        ndim: int,
        layout: Buffer.Layout,
        readonly: bool = False,
        name: str | None = None,
        aligned: bool = True,
    ) -> None: ...

class Dummy(Type): ...
class Opaque(Dummy): ...
class DTypeSpec(Type): ...

class DType(DTypeSpec, Opaque):
    def __init__(self, dtype: Type) -> None: ...

class ConstSized(Sized):
    def __len__(self) -> int: ...

class Hashable(Type): ...
class BaseTuple(ConstSized, Hashable): ...
class BaseAnonymousTuple(BaseTuple): ...
class Container(Sized, IterableType): ...
class Sequence(Container): ...

class UniTuple(BaseAnonymousTuple, Sequence):
    def __init__(self, dtype: Type | type[Type], count: int) -> None: ...
