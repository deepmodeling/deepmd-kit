# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend-independent discovery of on-disk data systems."""

from pathlib import (
    Path,
)
from typing import (
    Literal,
)

from deepmd.utils.path import (
    DPOSPath,
    DPPath,
)

DatasetFormat = Literal["deepmd", "lmdb"]


def is_lmdb(systems: str) -> bool:
    """Return whether a path identifies an LMDB dataset."""
    return systems.endswith(".lmdb") or Path(systems, "data.mdb").is_file()


def _inside_lmdb(path: Path, root: Path) -> bool:
    for parent in path.parents:
        if parent == root:
            break
        if is_lmdb(str(parent)):
            return True
    return False


def expand_sys_str(
    root_dir: str | Path,
    *,
    formats: tuple[DatasetFormat, ...] = ("deepmd",),
    patterns: list[str] | None = None,
) -> list[str]:
    """Discover data systems beneath one directory or HDF5 container.

    Parameters
    ----------
    root_dir : str or Path
        A data system, a parent directory, or a DeePMD HDF5 path.
    formats : tuple of str
        Accepted leaf formats. ``deepmd`` recognizes ``type.raw`` in NPY
        directories and HDF5 groups; ``lmdb`` recognizes LMDB directories.
    patterns : list of str, optional
        Recursive glob patterns selecting system paths. A directly supplied
        LMDB dataset is terminal and does not require a glob match.

    Returns
    -------
    list[str]
        Discovered system paths. Unpatterned DeePMD traversal retains its
        established order, with the root system last. Overlapping patterns
        include each matched path once.
    """
    root_name = str(root_dir)
    if is_lmdb(root_name):
        return [root_name] if "lmdb" in formats else []

    root = DPPath(root_name)
    candidates = (
        [*root.rglob("*"), root]
        if patterns is None
        else [path for pattern in patterns for path in root.rglob(pattern)]
    )
    systems: dict[str, None] = {}
    for candidate in candidates:
        name = str(candidate)
        if isinstance(candidate, DPOSPath):
            path = candidate.path
            if _inside_lmdb(path, root.path):
                continue
            if candidate.is_dir() and is_lmdb(name):
                if "lmdb" in formats:
                    systems[name] = None
                continue
        if "deepmd" in formats and (candidate / "type.raw").is_file():
            systems[name] = None
    return list(systems)


def rglob_sys_str(root_dir: str, patterns: list[str]) -> list[str]:
    """Discover DeePMD systems matching recursive glob patterns."""
    return expand_sys_str(root_dir, patterns=patterns)


def process_systems(
    systems: str | list[str],
    patterns: list[str] | None = None,
    *,
    formats: tuple[DatasetFormat, ...] = ("deepmd", "lmdb"),
) -> list[str]:
    """Expand every input path through the same format-aware discovery.

    Explicit input order and repeated input paths are preserved, so an
    input list defines the order used by per-system sampling probabilities.
    Each entry may name a dataset directly or a parent containing datasets.
    """
    paths = [systems] if isinstance(systems, str) else systems
    if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
        raise ValueError(
            f"Invalid systems value {systems!r}: expected str or list[str]."
        )
    return [
        system
        for path in paths
        for system in expand_sys_str(path, formats=formats, patterns=patterns)
    ]
