# SPDX-License-Identifier: MPL-2.0
# See <https://numba.readthedocs.io/en/stable/reference/types.html#numba-types>

class Type: ...
class boolean(Type): ...
class uint8(Type): ...

byte = uint8

class uint16(Type): ...
class uint32(Type): ...
class uint64(Type): ...
class int8(Type): ...

char = int8

class int16(Type): ...
class int32(Type): ...
class int64(Type): ...
class intc(Type): ...
class uintc(Type): ...
class intp(Type): ...
class uintp(Type): ...
class ssize_t(Type): ...
class size_t(Type): ...
class float32(Type): ...
class float64(Type): ...

double = float64

class complex64(Type): ...
class complex128(Type): ...
