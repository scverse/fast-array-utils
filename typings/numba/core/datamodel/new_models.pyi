# SPDX-License-Identifier: MPL-2.0
from collections.abc import Iterable
from typing import Generic, TypeVar

from ..types import Type
from .manager import DataModelManager

_T = TypeVar("_T", bound=Type)

class DataModel(Generic[_T]): ...
class CompositeModel(DataModel[_T], Generic[_T]): ...

class StructModel(CompositeModel[_T], Generic[_T]):
    def __init__(self, dmm: DataModelManager, fe_type: _T, members: Iterable[tuple[str, Type]]) -> None: ...
