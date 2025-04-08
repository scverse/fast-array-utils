from __future__ import annotations

from functools import cache, wraps
from importlib.metadata import version
from typing import TYPE_CHECKING

from packaging.version import Version

from . import types


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Concatenate, ParamSpec, TypeVar

    _DA = TypeVar("_DA", bound=types.DaskArray)
    _P = ParamSpec("_P")
    _R = TypeVar("_R")


__all__ = ["check_dask_sparray_support"]


@cache
def _dask_supports_sparray() -> bool:
    return Version(version("dask")) >= Version("2025.3")


def check_dask_sparray_support(
    func: Callable[Concatenate[_DA, _P], _R],
) -> Callable[Concatenate[_DA, _P], _R]:
    """Check that Dask isn’t too old when trying to use it with `scipy.sparse.sparray`s."""

    @wraps(func)
    def decorated(arr: _DA, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        if (
            isinstance(arr, types.DaskArray)
            and isinstance(arr._meta, types.sparray)  # noqa: SLF001
            and not _dask_supports_sparray()
        ):
            msg = "dask < 2025.3 does not support `scipy.sparse.sparray`s"
            raise RuntimeError(msg)
        return func(arr, *args, **kwargs)

    return decorated
