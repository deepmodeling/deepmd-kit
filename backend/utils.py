

from pathlib import Path
import sys
from dependency_groups import resolve

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
    list
        A list of dependencies in the specified group.
    """
    with Path("pyproject.toml").open("rb") as f:
        pyproject = tomllib.load(f)

    groups = pyproject["dependency-groups"]

    return resolve(groups, group)