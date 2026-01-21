# SPDX-License-Identifier: LGPL-3.0-or-later
import sys
from pathlib import (
    Path,
)

from dependency_groups import (
    resolve,
)

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def read_dependencies_from_dependency_group(group: str) -> tuple[str, ...]:
    """
    Reads dependencies from a dependency group.

    Parameters
    ----------
    group : str
        The name of the dependency group.

    Returns
    -------
    tuple[str, ...]
        A tuple of dependencies in the specified group.
    """
    with Path("pyproject.toml").open("rb") as f:
        pyproject = tomllib.load(f)

    groups = pyproject["dependency-groups"]

    return resolve(groups, group)
