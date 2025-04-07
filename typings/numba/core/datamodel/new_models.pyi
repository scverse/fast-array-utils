# SPDX-License-Identifier: MPL-2.0
from typing import Generic, TypeVar

from numba.core.types import Type

_T = TypeVar("_T", bound=Type)

class DataModel(Generic[_T]): ...
class CompositeModel(DataModel[_T], Generic[_T]): ...

class StructModel(CompositeModel[_T], Generic[_T]):
    def __init__(self, dmm: object, fe_type: _T, members: object) -> None: ...
