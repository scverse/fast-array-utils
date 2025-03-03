# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

import numpy as np


def validate_axis(axis: int | None) -> None:
    if axis is None:
        return
    if not isinstance(axis, int | np.integer):  # pragma: no cover
        msg = "axis must be integer or None."
        raise TypeError(msg)
    if axis not in (0, 1):  # pragma: no cover
        msg = "We only support axis 0 and 1 at the moment"
        raise NotImplementedError(msg)
