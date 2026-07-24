# SPDX-License-Identifier: LGPL-3.0-or-later
"""Create permanent GitHub source links from Sphinx AutoAPI metadata."""

from __future__ import (
    annotations,
)

import os
import re
import subprocess
from functools import (
    cache,
)
from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
    Protocol,
)
from urllib.parse import (
    quote,
)

if TYPE_CHECKING:
    from collections.abc import (
        Mapping,
    )

    from sphinx.application import (
        Sphinx,
    )


GITHUB_REPOSITORY = "deepmodeling/deepmd-kit"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_FULL_GIT_HASH = re.compile(r"[0-9a-f]{40,64}")


class SourceLocation(NamedTuple):
    """Location of a documented object within the source repository."""

    path: str
    first_line: int | None
    last_line: int | None


class AutoapiObject(Protocol):
    """Subset of an AutoAPI object needed for static source resolution."""

    id: str
    obj: dict[str, Any]


_source_locations: dict[str, SourceLocation] = {}


@cache
def get_git_commit() -> str | None:
    """Return the immutable commit represented by the documentation build."""
    commit = os.environ.get("READTHEDOCS_GIT_COMMIT_HASH")
    if commit is None:
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=REPOSITORY_ROOT,
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = commit.lower()
    return commit if _FULL_GIT_HASH.fullmatch(commit) else None


def _resolve_original_object(
    documented_object: AutoapiObject,
    all_objects: Mapping[str, AutoapiObject],
) -> AutoapiObject:
    """Follow AutoAPI re-exports to the object that owns the source lines."""
    source_object = documented_object
    visited = set()
    while original_path := source_object.obj.get("original_path"):
        if original_path in visited:
            break
        visited.add(original_path)
        original_object = all_objects.get(original_path)
        if original_object is None:
            break
        source_object = original_object
    return source_object


def _containing_module(
    object_name: str,
    modules: Mapping[str, AutoapiObject],
) -> AutoapiObject | None:
    """Find the nearest AutoAPI module or package containing an object."""
    candidate = object_name
    while candidate:
        if candidate in modules:
            return modules[candidate]
        candidate = candidate.rpartition(".")[0]
    return None


def collect_autoapi_source_locations(app: Sphinx) -> None:
    """Cache source locations after AutoAPI finishes its static analysis.

    AutoAPI runs on ``builder-inited`` with the default priority of 500. This
    callback is registered at priority 600 so its object graph is available
    before Sphinx forks any parallel document-reading workers.
    """
    all_objects = getattr(app.env, "autoapi_all_objects", {})
    modules = {name: obj for name, obj in all_objects.items() if "file_path" in obj.obj}

    _source_locations.clear()
    for documented_name, documented_object in all_objects.items():
        source_object = _resolve_original_object(documented_object, all_objects)
        module = _containing_module(source_object.id, modules)
        if module is None:
            continue

        source_path = Path(module.obj["file_path"]).resolve()
        try:
            relative_path = source_path.relative_to(REPOSITORY_ROOT).as_posix()
        except ValueError:
            # Never create a repository URL for objects imported from outside
            # the checked-out DeePMD-kit source tree.
            continue

        _source_locations[documented_name] = SourceLocation(
            relative_path,
            source_object.obj.get("from_line_no"),
            source_object.obj.get("to_line_no"),
        )


def linkcode_resolve(domain: str, info: dict[str, str]) -> str | None:
    """Resolve a Python object to a commit-pinned GitHub source URL."""
    if domain != "py":
        return None

    module = info.get("module", "")
    fullname = info.get("fullname", "")
    if not module:
        return None

    object_name = (
        fullname if fullname.startswith(f"{module}.") else f"{module}.{fullname}"
    ).rstrip(".")
    location = _source_locations.get(object_name) or _source_locations.get(module)
    commit = get_git_commit()
    if location is None or commit is None:
        return None

    url = (
        f"https://github.com/{GITHUB_REPOSITORY}/blob/{commit}/"
        f"{quote(location.path, safe='/')}"
    )
    if location.first_line is not None:
        url += f"#L{location.first_line}"
        if location.last_line not in (None, location.first_line):
            url += f"-L{location.last_line}"
    return url
