# SPDX-License-Identifier: MPL-2.0
# taken from https://github.com/numba/numba-scipy/blob/release0.4/numba_scipy/sparse.py
from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from numba.core import cgutils
from numba.core import types as nbtypes
from numba.core.imputils import impl_ret_borrowed
from numba.extending import (
    NativeValue,
    box,
    intrinsic,
    make_attribute_wrapper,
    overload,
    overload_attribute,
    overload_method,
    register_model,
    typeof_impl,
    unbox,
)
from scipy import sparse


if TYPE_CHECKING:
    from numba.core.datamodel import new_models as models
else:
    from numba.extending import models


if TYPE_CHECKING:
    from collections.abc import Callable
    from functools import _SingleDispatchCallable
    from typing import Any, ClassVar, Literal, Protocol

    from llvmlite.ir import IRBuilder
    from numba.core.base import BaseContext
    from numba.core.pythonapi import PythonAPI
    from numba.core.typing.templates import Signature
    from numba.core.typing.typeof import _TypeofContext
    from numpy.typing import DTypeLike, NDArray

    from fast_array_utils.types import CSBase

    class _Context(Protocol):
        # https://numba.readthedocs.io/en/stable/extending/low-level.html#boxing-and-unboxing
        context: BaseContext
        builder: IRBuilder
        pyapi: PythonAPI

    class BoxContext(_Context):
        env_manager: object

        def box(self, typ: nbtypes.Type, val: NativeValue) -> object: ...

    class UnboxContext(_Context):
        def unbox(self, typ: nbtypes.Type, obj: object) -> NativeValue: ...


class CS2DType(nbtypes.Type):
    """A Numba `Type` modeled after the base class `scipy.sparse.compressed._cs_matrix`."""

    name: ClassVar[str]
    cls: ClassVar[type[CSBase]]

    @classmethod
    def instance_class(
        cls,
        data: NDArray[np.number[Any]],
        indices: NDArray[np.integer[Any]],
        indptr: NDArray[np.integer[Any]],
        shape: tuple[int, int],
    ) -> CSBase:
        return cls.cls((data, indices, indptr), shape, copy=False)

    def __init__(self, dtype: DTypeLike) -> None:
        self.dtype = nbtypes.DType(dtype)
        self.data = nbtypes.Array(dtype, 1, "A")
        self.indices = nbtypes.Array(nbtypes.int32, 1, "A")
        self.indptr = nbtypes.Array(nbtypes.int32, 1, "A")
        self.shape = nbtypes.UniTuple(nbtypes.int64, 2)
        super().__init__(self.name)

    @property
    def key(self) -> tuple[str, np.dtype[np.number[Any]]]:
        return (self.name, self.dtype)


make_attribute_wrapper(CS2DType, "data", "data")
make_attribute_wrapper(CS2DType, "indices", "indices")
make_attribute_wrapper(CS2DType, "indptr", "indptr")
make_attribute_wrapper(CS2DType, "shape", "shape")


def make_typeof_fn(typ: type[CS2DType]) -> Callable[[CSBase, _TypeofContext], CS2DType]:
    def typeof(val: CSBase, c: _TypeofContext) -> CS2DType:
        data = typeof_impl(val.data, c)
        return typ(data.dtype)

    return typeof


class CS2DModel(models.StructModel):
    def __init__(self, dmm: object, fe_type: CS2DType) -> None:
        members = [
            ("data", fe_type.data),
            ("indices", fe_type.indices),
            ("indptr", fe_type.indptr),
            ("shape", fe_type.shape),
        ]
        super().__init__(dmm, fe_type, members)


CLASSES = [sparse.csr_matrix, sparse.csc_matrix, sparse.csr_array, sparse.csc_array]
TYPES = [
    type(f"{cls.__name__}Type", (CS2DType,), {"cls": cls, "name": cls.__name__}) for cls in CLASSES
]
TYPEOF_FUNCS = {typ.cls: make_typeof_fn(typ) for typ in TYPES}
MODELS = {typ: type(f"{typ.cls.__name__}Model", (CS2DModel,), {}) for typ in TYPES}


def unbox_matrix(typ: CS2DType, obj: CSBase, c: UnboxContext) -> NativeValue:
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


def box_matrix(typ: CS2DType, val: NativeValue, c: BoxContext) -> CSBase:
    struct_ptr = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)

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


@overload(np.shape)
def overload_sparse_shape(x: nbtypes.Type) -> None | Callable[[CS2DType], nbtypes.UniTuple]:
    if not isinstance(x, CS2DType):
        return None

    def shape(x: CS2DType) -> nbtypes.UniTuple:
        return x.shape

    return shape


@overload_attribute(CS2DType, "ndim")
def overload_sparse_ndim(inst: nbtypes.Type) -> None | Callable[[CS2DType], Literal[2]]:
    if not isinstance(inst, CS2DType):
        return None

    def ndim(_: CS2DType) -> Literal[2]:
        return 2

    return ndim


@intrinsic
def _sparse_copy(
    typingctx: object,  # noqa: ARG001
    inst: CS2DType,
    data: nbtypes.Array,  # noqa: ARG001
    indices: nbtypes.Array,  # noqa: ARG001
    indptr: nbtypes.Array,  # noqa: ARG001
    shape: nbtypes.UniTuple,  # noqa: ARG001
) -> tuple[Signature, Callable[..., NativeValue]]:
    def _construct(
        context: BaseContext,
        builder: IRBuilder,
        sig: Signature,
        args: tuple[CS2DType, nbtypes.Array, nbtypes.Array, nbtypes.Array, nbtypes.UniTuple],
    ) -> NativeValue:
        typ = sig.return_type
        struct_proxy_cls = cgutils.create_struct_proxy(typ)
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


@overload_method(CS2DType, "copy")
def overload_sparse_copy(inst: nbtypes.Type) -> None | Callable[[CS2DType], CS2DType]:
    if not isinstance(inst, CS2DType):
        return None

    def copy(inst: CS2DType) -> CS2DType:
        return _sparse_copy(
            inst, inst.data.copy(), inst.indices.copy(), inst.indptr.copy(), inst.shape
        )

    return copy


def register() -> None:
    for cls, func in TYPEOF_FUNCS.items():
        cast("_SingleDispatchCallable", typeof_impl).register(cls, func)
    for typ, model in MODELS.items():
        register_model(typ)(model)
        unbox(typ)(unbox_matrix)
        box(typ)(box_matrix)
