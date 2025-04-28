# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

import numpy as np
from numpy.exceptions import AxisError


def validate_axis(ndim: int, axis: int | None) -> None:
    if axis is None:
        return
    if not isinstance(axis, int | np.integer):  # pragma: no cover
        msg = f"axis must be integer or None, not {axis=!r}."
        raise TypeError(msg)
    if axis == 0 and ndim == 1:
        raise AxisError(axis, ndim, "use axis=None for 1D arrays")
    if axis not in range(ndim):
        raise AxisError(axis, ndim)
    if axis not in (0, 1):  # pragma: no cover
        msg = "We only support axis 0 and 1 at the moment"
        raise NotImplementedError(msg)
