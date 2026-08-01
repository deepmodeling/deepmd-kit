# SPDX-License-Identifier: LGPL-3.0-or-later
"""Provide project metadata that depends on the selected build configuration."""

import sys
from collections.abc import (
    Mapping,
)
from pathlib import (
    Path,
)
from typing import (
    Any,
)

from .find_pytorch import (
    get_pt_requirement,
)
from .find_tensorflow import (
    get_tf_requirement,
)
from .read_env import (
    get_argument_from_env,
)

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

__all__ = ["dynamic_metadata"]


def __dir__() -> list[str]:
    return __all__


def dynamic_metadata(
    settings: Mapping[str, object],
    _project: Mapping[str, Any],
) -> dict[str, Any]:
    """Return one metadata fragment using the standard v1 provider protocol.

    Each ``[[tool.dynamic-metadata]]`` entry selects a field explicitly. The
    returned mapping is merged into the resolved ``[project]`` table by
    scikit-build-core.
    """
    field = settings.get("field")
    if not isinstance(field, str) or field not in {
        "optional-dependencies",
        "scripts",
    }:
        msg = f"Unsupported dynamic metadata field: {field!r}"
        raise ValueError(msg)

    _, _, find_libpython_requires, extra_scripts, tf_version, pt_version = (
        get_argument_from_env()
    )
    with Path("pyproject.toml").open("rb") as f:
        pyproject = tomllib.load(f)

    if field == "scripts":
        result = {
            **pyproject["tool"]["deepmd_build_backend"]["scripts"],
            **extra_scripts,
        }
    else:
        optional_dependencies = pyproject["tool"]["deepmd_build_backend"][
            "optional-dependencies"
        ]
        optional_dependencies["lmp"].extend(find_libpython_requires)
        optional_dependencies["ipi"].extend(find_libpython_requires)
        result = {
            **optional_dependencies,
            **get_tf_requirement(tf_version),
            **get_pt_requirement(pt_version),
        }

    return {field: result}
