# SPDX-License-Identifier: LGPL-3.0-or-later
import sys
from pathlib import (
    Path,
)
from typing import (
    Dict,
    List,
    Optional,
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


def __dir__() -> List[str]:
    return __all__


def dynamic_metadata(
    field: str,
    settings: Optional[Dict[str, object]] = None,
):
    assert field in ["optional-dependencies", "entry-points", "scripts"]
    _, _, find_libpython_requires, extra_scripts, tf_version, pt_version = (
        get_argument_from_env()
    )
    with Path("pyproject.toml").open("rb") as f:
        pyproject = tomllib.load(f)

    if field == "scripts":
        return {
            **pyproject["tool"]["deepmd_build_backend"]["scripts"],
            **extra_scripts,
        }
    elif field == "optional-dependencies":
        optional_dependencies = pyproject["tool"]["deepmd_build_backend"][
            "optional-dependencies"
        ]
        optional_dependencies["lmp"].extend(find_libpython_requires)
        optional_dependencies["ipi"].extend(find_libpython_requires)
        return {
            **optional_dependencies,
            **get_tf_requirement(tf_version),
            **get_pt_requirement(pt_version),
        }
