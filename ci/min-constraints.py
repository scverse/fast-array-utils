# SPDX-License-Identifier: MPL-2.0
"""Pin every `>=` lower bound of the dependencies and extras (not test deps) to `==`."""

from __future__ import annotations

import tomllib
from pathlib import Path


HERE = Path(__file__).parent

project = tomllib.loads((HERE.parent / "pyproject.toml").read_text())["project"]
deps: list[str] = [*project["dependencies"], *(d for ds in project["optional-dependencies"].values() for d in ds)]
(HERE / "min-constraints.txt").write_text("".join(f"{d.replace('>=', '==')}\n" for d in deps if ">=" in d))
