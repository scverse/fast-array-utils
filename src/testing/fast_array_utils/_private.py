# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _set_numpy_print() -> None:  # TODO(flying-sheep): #97 remove once we depend on numpy >=2
    if int(np.__version__.split(".", 1)[0]) > 1:
        np.set_printoptions(legacy="1.25")
