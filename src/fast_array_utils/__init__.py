# SPDX-License-Identifier: MPL-2.0
"""Fast array utils."""

from __future__ import annotations

from . import _patches, conv, stats, types


__all__ = ["conv", "stats", "types"]

_patches.patch_dask()
