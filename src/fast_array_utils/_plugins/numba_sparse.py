# SPDX-License-Identifier: MPL-2.0
"""Numba support for sparse arrays and matrices."""

# taken from https://github.com/numba/numba-scipy/blob/release0.4/numba_scipy/sparse.py
# See https://numba.pydata.org/numba-doc/dev/extending/
from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numba.core.types as nbtypes
import numpy as np
from numba.core import cgutils
from numba.core.imputils import impl_ret_borrowed
from numba.extending import (
    NativeValue,
    box,
    intrinsic,
    make_attribute_wrapper,
    models,
    overload,
    overload_attribute,
    overload_method,
    register_model,
    typeof_impl,
    unbox,
)
from scipy import sparse


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import Any, ClassVar, Literal

    from llvmlite.ir import IRBuilder, Value
    from numba.core.base import BaseContext
    from numba.core.datamodel.manager import DataModelManager
    from numba.core.extending import BoxContext, TypingContext, UnboxContext
    from numba.core.typing.templates import Signature
    from numba.core.typing.typeof import _TypeofContext
    from numpy.typing import NDArray

    from fast_array_utils.types import CSBase


class CSType(nbtypes.Type):
    """A Numba `Type` modeled after the base class `scipy.sparse.compressed._cs_matrix`.

    This is an abstract base class for the actually used, registered types in `TYPES` below.
    It collects information about the type (e.g. field dtypes) for later use in the data model.
    """

    name: ClassVar[Literal["csr_matrix", "csc_matrix", "csr_array", "csc_array"]]
    cls: ClassVar[type[CSBase]]

    @classmethod
    def instance_class(
        cls,
        data: NDArray[np.number[Any]],
        indices: NDArray[np.integer[Any]],
        indptr: NDArray[np.integer[Any]],
        shape: tuple[int, int],  # actually tuple[int, ...] for sparray subclasses
    ) -> CSBase:
        return cls.cls((data, indices, indptr), shape, copy=False)

    def __init__(self, ndim: int, *, dtype: nbtypes.Type, dtype_ind: nbtypes.Type) -> None:
        self.dtype = nbtypes.DType(dtype)
        self.dtype_ind = nbtypes.DType(dtype_ind)
        self.data = nbtypes.Array(dtype, 1, "A")
        self.indices = nbtypes.Array(dtype_ind, 1, "A")
        self.indptr = nbtypes.Array(dtype_ind, 1, "A")
        self.shape = nbtypes.UniTuple(nbtypes.intp, ndim)
        super().__init__(self.name)

    @property
    def key(self) -> tuple[str | nbtypes.Type, ...]:
        return (self.name, self.dtype, self.dtype_ind)


# make data model attributes available in numba functions
for attr in ["data", "indices", "indptr", "shape"]:
    make_attribute_wrapper(CSType, attr, attr)


def make_typeof_fn(typ: type[CSType]) -> Callable[[CSBase, _TypeofContext], CSType]:
    """Create a `typeof` function that maps a scipy matrix/array type to a numba `Type`."""

    def typeof(val: CSBase, c: _TypeofContext) -> CSType:
        if val.indptr.dtype != val.indices.dtype:  # pragma: no cover
            msg = "indptr and indices must have the same dtype"
            raise TypeError(msg)
        data = cast("nbtypes.Array", typeof_impl(val.data, c))
        indptr = cast("nbtypes.Array", typeof_impl(val.indptr, c))
        return typ(val.ndim, dtype=data.dtype, dtype_ind=indptr.dtype)

    return typeof


if TYPE_CHECKING:
    _CSModelBase = models.StructModel[CSType]
else:
    _CSModelBase = models.StructModel


class CSModel(_CSModelBase):
    """Numba data model for compressed sparse matrices.

    This is the class that is used by numba to lower the array types.
    """

    def __init__(self, dmm: DataModelManager, fe_type: CSType) -> None:
        members = [
            ("data", fe_type.data),
            ("indices", fe_type.indices),
            ("indptr", fe_type.indptr),
            ("shape", fe_type.shape),
        ]
        super().__init__(dmm, fe_type, members)


# create all the actual types and data models
CLASSES: Sequence[type[CSBase]] = [
    sparse.csr_matrix,
    sparse.csc_matrix,
    sparse.csr_array,
    sparse.csc_array,
]
TYPES: Sequence[type[CSType]] = [type(f"{cls.__name__}Type", (CSType,), {"cls": cls, "name": cls.__name__}) for cls in CLASSES]
TYPEOF_FUNCS: Mapping[type[CSBase], Callable[[CSBase, _TypeofContext], CSType]] = {typ.cls: make_typeof_fn(typ) for typ in TYPES}
MODELS: Mapping[type[CSType], type[CSModel]] = {typ: type(f"{typ.cls.__name__}Model", (CSModel,), {}) for typ in TYPES}


def unbox_matrix(typ: CSType, obj: Value, c: UnboxContext) -> NativeValue:
    """Convert a Python cs{rc}_{matrix,array} to a Numba value."""
    struct_proxy_cls = cgutils.create_struct_proxy(typ)
    struct_ptr = struct_proxy_cls(c.context, c.builder)

    data = c.pyapi.object_getattr_string(obj, "data")
    indices = c.pyapi.object_getattr_string(obj, "indices")
    indptr = c.pyapi.object_getattr_string(obj, "indptr")
    shape = c.pyapi.object_getattr_string(obj, "shape")

    struct_ptr.data = c.unbox(typ.data, data).value
    struct_ptr.indices = c.unbox(typ.indices, indices).value
    struct_ptr.indptr = c.unbox(typ.indptr, indptr).value
    struct_ptr.shape = c.unbox(typ.shape, shape).value

    c.pyapi.decref(data)
    c.pyapi.decref(indices)
    c.pyapi.decref(indptr)
    c.pyapi.decref(shape)

    is_error_ptr = cgutils.alloca_once_value(c.builder, cgutils.false_bit)
    is_error = c.builder.load(is_error_ptr)

    return NativeValue(struct_ptr._getvalue(), is_error=is_error)  # noqa: SLF001


def box_matrix(typ: CSType, val: NativeValue, c: BoxContext) -> Value:
    """Convert numba value into a Python cs{rc}_{matrix,array}."""
    struct_proxy_cls = cgutils.create_struct_proxy(typ)
    struct_ptr = struct_proxy_cls(c.context, c.builder, value=val)

    data_obj = c.box(typ.data, struct_ptr.data)
    indices_obj = c.box(typ.indices, struct_ptr.indices)
    indptr_obj = c.box(typ.indptr, struct_ptr.indptr)
    shape_obj = c.box(typ.shape, struct_ptr.shape)

    c.pyapi.incref(data_obj)
    c.pyapi.incref(indices_obj)
    c.pyapi.incref(indptr_obj)
    c.pyapi.incref(shape_obj)

    cls_obj = c.pyapi.unserialize(c.pyapi.serialize_object(typ.instance_class))
    obj = c.pyapi.call_function_objargs(cls_obj, (data_obj, indices_obj, indptr_obj, shape_obj))

    c.pyapi.decref(data_obj)
    c.pyapi.decref(indices_obj)
    c.pyapi.decref(indptr_obj)
    c.pyapi.decref(shape_obj)

    return obj


# See https://numba.readthedocs.io/en/stable/extending/overloading-guide.html
@overload(np.shape)
def overload_sparse_shape(x: CSType) -> None | Callable[[CSType], nbtypes.UniTuple]:
    if not isinstance(x, CSType):  # pragma: no cover
        return None

    # nopython code:
    def shape(x: CSType) -> nbtypes.UniTuple:  # pragma: no cover
        return x.shape

    return shape


@overload_attribute(CSType, "ndim")
def overload_sparse_ndim(inst: CSType) -> None | Callable[[CSType], int]:
    if not isinstance(inst, CSType):  # pragma: no cover
        return None

    # nopython code:
    def ndim(inst: CSType) -> int:  # pragma: no cover
        return len(inst.shape)

    return ndim


@intrinsic
def _sparse_copy(
    typingctx: TypingContext,  # noqa: ARG001
    inst: CSType,
    data: nbtypes.Array,  # noqa: ARG001
    indices: nbtypes.Array,  # noqa: ARG001
    indptr: nbtypes.Array,  # noqa: ARG001
    shape: nbtypes.UniTuple,  # noqa: ARG001
) -> tuple[Signature, Callable[..., NativeValue]]:
    def _construct(
        context: BaseContext,
        builder: IRBuilder,
        sig: Signature,
        args: tuple[Value, Value, Value, Value, Value],
    ) -> NativeValue:
        struct_proxy_cls = cgutils.create_struct_proxy(sig.return_type)
        struct = struct_proxy_cls(context, builder)
        _, data, indices, indptr, shape = args
        struct.data = data
        struct.indices = indices
        struct.indptr = indptr
        struct.shape = shape
        return impl_ret_borrowed(
            context,
            builder,
            sig.return_type,
            struct._getvalue(),  # noqa: SLF001
        )

    sig = inst(inst, inst.data, inst.indices, inst.indptr, inst.shape)

    return sig, _construct


@overload_method(CSType, "copy")
def overload_sparse_copy(inst: CSType) -> None | Callable[[CSType], CSType]:
    if not isinstance(inst, CSType):  # pragma: no cover
        return None

    # nopython code:
    def copy(inst: CSType) -> CSType:  # pragma: no cover
        return _sparse_copy(inst, inst.data.copy(), inst.indices.copy(), inst.indptr.copy(), inst.shape)  # type: ignore[return-value]

    return copy


def register() -> None:
    """Register the numba types, data models, and mappings between them and the Python types."""
    for cls, func in TYPEOF_FUNCS.items():
        typeof_impl.register(cls, func)
    for typ, model in MODELS.items():
        register_model(typ)(model)
        unbox(typ)(unbox_matrix)
        box(typ)(box_matrix)
