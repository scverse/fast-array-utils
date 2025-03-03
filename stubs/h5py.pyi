# SPDX-License-Identifier: MPL-2.0
import os
from contextlib import closing
from typing import IO, AnyStr, Literal

import numpy as np
from numpy.typing import ArrayLike

class Empty: ...
class HLObject: ...

class Dataset(HLObject):
    dtype: np.dtype[np.generic]
    shape: tuple[int, ...]

class Group(HLObject): ...

class File(Group, closing[File]):  # not actually a subclass of closing
    def __init__(
        self,
        name: AnyStr | os.PathLike[AnyStr] | IO[bytes],
        mode: Literal["r", "r+", "w", "w-", "x", "a"] = "r",
        *args: object,
        **kw: object,
    ) -> None: ...
    def close(self) -> None: ...
    def create_dataset(
        self,
        name: str,
        shape: tuple[int, ...] | None = None,
        dtype: np.dtype[np.generic] | None = None,
        data: ArrayLike | Empty | None = None,
        **kwds: object,
    ) -> Dataset: ...
