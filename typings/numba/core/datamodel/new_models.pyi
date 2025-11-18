# SPDX-License-Identifier: MPL-2.0
from collections.abc import Iterable

from ..types import Type
from .manager import DataModelManager

class DataModel[T: Type]: ...
class CompositeModel[T: Type](DataModel[T]): ...

class StructModel[T: Type](CompositeModel[T]):
    def __init__(self, dmm: DataModelManager, fe_type: T, members: Iterable[tuple[str, Type]]) -> None: ...
