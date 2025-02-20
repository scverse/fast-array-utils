# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations


__all__ = ["_import_by_qualname"]


def _import_by_qualname(qualname: str) -> object:
    from importlib import import_module

    parts = qualname.split(".")

    # import the module
    obj = import_module(parts[0])
    for i, name in enumerate(parts[1:]):  # noqa: B007
        try:
            obj = import_module(f"{obj.__name__}.{name}")
        except ModuleNotFoundError:
            break
    else:
        i = len(parts)

    # get object if applicable
    for name in parts[i + 1 :]:
        try:
            obj = getattr(obj, name)
        except AttributeError:
            msg = f"Could not import {name!r} from {'.'.join(parts[:i])} "
            if i + 1 < len(parts):
                msg += f"(trying to get {'.'.join(parts[i + 1 :])!r})"
            raise ImportError(msg) from None
    return obj
