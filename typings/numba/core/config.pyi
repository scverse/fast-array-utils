# SPDX-License-Identifier: MPL-2.0
from fast_array_utils.numba import TheadingCategory, ThreadingLayer

THREADING_LAYER: ThreadingLayer | TheadingCategory
THREADING_LAYER_PRIORITY: list[ThreadingLayer]
