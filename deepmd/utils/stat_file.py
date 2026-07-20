# SPDX-License-Identifier: LGPL-3.0-or-later
"""Scoped access to persistent training-statistics caches."""

from __future__ import (
    annotations,
)

from collections.abc import (
    Mapping,
)
from contextlib import (
    contextmanager,
)
from dataclasses import (
    dataclass,
)
from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
)

import h5py
import numpy as np
from wcmatch.glob import (
    globfilter,
)

from deepmd.utils.path import (
    DPH5Path,
    DPOSPath,
    DPPath,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterator,
        Sequence,
    )

StatFileMode = Literal["read", "update"]

_HDF5_SUFFIXES = {".h5", ".hdf5"}
_PAIR_TRANSACTION_MARKER = "__deepmd_output_stat_transaction__"


@dataclass(frozen=True)
class StatFileSpec:
    """Describe a statistics cache without opening it.

    Parameters
    ----------
    path
        Cache path. ``None`` disables persistent statistics storage.
    mode
        ``read`` opens an existing cache read-only. ``update`` creates the
        cache when necessary and permits statistics writers to update it.

    Raises
    ------
    ValueError
        If the mode is invalid, the path is empty, or read mode has no path.
    """

    path: str | None
    mode: StatFileMode = "update"

    def __post_init__(self) -> None:
        if self.mode not in {"read", "update"}:
            raise ValueError(
                "`stat_file_mode` must be either 'read' or 'update', "
                f"but received {self.mode!r}."
            )
        if self.path is not None and not self.path.strip():
            raise ValueError("`stat_file` must not be empty.")
        if self.path is None and self.mode == "read":
            raise ValueError("`stat_file_mode='read'` requires `stat_file`.")


def stat_file_specs_by_task(
    spec: StatFileSpec | Mapping[str, StatFileSpec] | None,
    task_names: Sequence[str],
) -> dict[str, StatFileSpec]:
    """Normalize statistics-cache configuration by task.

    Parameters
    ----------
    spec
        One single-task specification, a mapping for multi-task training, or
        ``None`` when persistent caching is disabled.
    task_names
        Ordered task names owned by the trainer.

    Returns
    -------
    dict[str, StatFileSpec]
        One validated specification per task.

    Raises
    ------
    TypeError
        If a single specification is supplied for multiple tasks.
    KeyError
        If a task is absent from a supplied mapping.
    ValueError
        If multiple tasks reference the same physical cache path.
    """
    if spec is None:
        specs = {task: StatFileSpec(None) for task in task_names}
    elif isinstance(spec, Mapping):
        specs = {task: spec[task] for task in task_names}
    else:
        if len(task_names) != 1:
            raise TypeError(
                "Multi-task training requires one statistics-cache "
                "configuration per task."
            )
        specs = {task_names[0]: spec}

    tasks_by_path: dict[Path, list[str]] = {}
    for task, task_spec in specs.items():
        if task_spec.path is None:
            continue
        path = Path(task_spec.path).expanduser().resolve(strict=False)
        tasks_by_path.setdefault(path, []).append(task)
    duplicates = {
        path: tasks for path, tasks in tasks_by_path.items() if len(tasks) > 1
    }
    if duplicates:
        details = "; ".join(
            f"{str(path)!r}: {', '.join(repr(task) for task in tasks)}"
            for path, tasks in duplicates.items()
        )
        raise ValueError(
            "Each training task must use a distinct statistics-cache path; "
            f"duplicate path(s): {details}."
        )
    return specs


@contextmanager
def open_stat_file(
    spec: StatFileSpec,
) -> Iterator[DPPath | None]:
    """Open one statistics cache for a bounded initialization scope.

    Existing HDF5 caches in update mode remain read-only until the first
    write. A cache hit therefore neither acquires a writer lock nor changes
    the file's HDF5 write-status metadata.

    Parameters
    ----------
    spec
        Statistics-cache configuration.

    Yields
    ------
    DPPath or None
        Scoped cache root, or ``None`` when persistent storage is disabled.

    Raises
    ------
    FileNotFoundError
        If read mode targets a cache that does not exist.
    ValueError
        If the target has an unsupported type.
    """
    if spec.path is None:
        yield None
        return

    target = Path(spec.path).expanduser().resolve(strict=False)
    if not target.exists() and spec.mode == "read":
        raise FileNotFoundError(
            f"Statistics cache {str(target)!r} does not exist in read mode."
        )

    if target.is_dir() or (
        not target.exists() and target.suffix.lower() not in _HDF5_SUFFIXES
    ):
        if not target.exists():
            target.mkdir(parents=True, exist_ok=True)
        root: DPPath = DPOSPath(target, mode="r" if spec.mode == "read" else "a")
        yield root
        return

    if target.exists() and not target.is_file():
        raise ValueError(
            f"Statistics cache {str(target)!r} is neither a file nor a directory."
        )
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)

    owner = _H5StatFile(target, spec.mode)
    try:
        yield _H5StatPath(owner, "/")
    finally:
        owner.close()


def load_required_items(
    path: DPPath | None,
    names: Sequence[str],
) -> dict[str, np.ndarray] | None:
    """Load a complete group of statistics datasets.

    Parameters
    ----------
    path
        Statistics-cache root used by the current consumer.
    names
        Dataset names that form one indivisible statistics group.

    Returns
    -------
    dict[str, numpy.ndarray] or None
        Loaded arrays when every dataset exists. ``None`` indicates that the
        group must be recomputed in update mode or that caching is disabled.

    Raises
    ------
    FileNotFoundError
        If a read-only cache is missing one or more required datasets.
    """
    if path is None:
        return None
    missing = [name for name in names if not (path / name).is_file()]
    if missing:
        if getattr(path, "mode", None) == "r":
            missing_items = ", ".join(repr(name) for name in missing)
            raise FileNotFoundError(
                f"Read-only statistics cache {path} is missing required "
                f"item(s): {missing_items}."
            )
        return None
    return {name: (path / name).load_numpy() for name in names}


def load_paired_items(
    path: DPPath | None,
    pairs: Sequence[tuple[str, str]],
) -> dict[str, np.ndarray] | None:
    """Load complete pairs from a legacy statistics cache.

    Read mode requires every requested pair. Update mode preserves the legacy
    convention that both datasets may be absent for an output unavailable in
    the sampled data. An output represented by either dataset is valid only
    when its pair is also present. Statistics writers store every first item
    before any second item, so an interrupted write leaves at least one
    incomplete pair.

    Parameters
    ----------
    path
        Statistics-cache root used by the current consumer.
    pairs
        Dataset-name pairs that represent independently optional outputs.

    Returns
    -------
    dict[str, numpy.ndarray] or None
        Arrays for every represented pair. ``None`` indicates that the cache
        contains no represented pair or must be recomputed in update mode.

    Raises
    ------
    FileNotFoundError
        If a read-only cache is missing any requested dataset.
    """
    if path is None:
        return None

    marker = path / _PAIR_TRANSACTION_MARKER
    if marker.is_file() or marker.is_dir():
        if getattr(path, "mode", None) == "r":
            raise FileNotFoundError(
                f"Read-only statistics cache {path} contains an incomplete "
                "output-statistics transaction."
            )
        return None

    present = {name: (path / name).is_file() for pair in pairs for name in pair}
    if getattr(path, "mode", None) == "r":
        missing = [name for pair in pairs for name in pair if not present[name]]
        if missing:
            missing_items = ", ".join(repr(name) for name in missing)
            raise FileNotFoundError(
                f"Read-only statistics cache {path} is missing required "
                f"item(s): {missing_items}."
            )
        return {name: (path / name).load_numpy() for pair in pairs for name in pair}

    represented = [pair for pair in pairs if any(present[name] for name in pair)]
    missing = [name for pair in represented for name in pair if not present[name]]
    if not represented or missing:
        return None

    return {name: (path / name).load_numpy() for pair in represented for name in pair}


def replace_paired_items(
    path: DPPath,
    pairs: Sequence[tuple[str, str]],
    items: Mapping[str, np.ndarray],
) -> None:
    """Replace optional pairs as one recoverable statistics transaction.

    All requested datasets are removed before the newly computed complete
    pairs are written. A transient marker makes an interrupted replacement
    distinguishable from a valid legacy cache. The marker is removed after
    every dataset has been flushed, so successful caches retain the legacy
    layout.

    Parameters
    ----------
    path
        Writable statistics-cache group that owns the pairs.
    pairs
        Dataset-name pairs requested by the statistics consumer.
    items
        Newly computed datasets. Each requested pair must be either complete
        or entirely absent.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the path is read-only, names are duplicated or reserved, an item
        does not belong to a requested pair, or exactly one item of a pair is
        supplied.
    TypeError
        If the path implementation is unsupported.
    """
    pair_list = list(pairs)
    requested_names = [name for pair in pair_list for name in pair]
    if len(requested_names) != len(set(requested_names)):
        raise ValueError("Statistics pair names must be unique.")
    if _PAIR_TRANSACTION_MARKER in requested_names:
        raise ValueError(
            f"{_PAIR_TRANSACTION_MARKER!r} is reserved for cache transactions."
        )

    unknown_names = set(items).difference(requested_names)
    if unknown_names:
        names = ", ".join(repr(name) for name in sorted(unknown_names))
        raise ValueError(
            f"Statistics items do not belong to a requested pair: {names}."
        )

    complete_pairs: list[tuple[str, str]] = []
    for first, second in pair_list:
        first_present = first in items
        second_present = second in items
        if first_present != second_present:
            raise ValueError(
                f"Statistics pair ({first!r}, {second!r}) must be supplied together."
            )
        if first_present:
            complete_pairs.append((first, second))

    if getattr(path, "mode", None) == "r":
        raise ValueError("Cannot write to a read-only statistics cache.")

    ordered_names = [pair[0] for pair in complete_pairs] + [
        pair[1] for pair in complete_pairs
    ]
    path.mkdir(parents=True, exist_ok=True)
    if isinstance(path, _H5StatPath):
        file = path._owner.file(write=True)
        _replace_h5_items(
            file,
            path._connect_path,
            requested_names,
            [(name, items[name]) for name in ordered_names],
        )
        return
    if isinstance(path, DPH5Path):
        try:
            _replace_h5_items(
                path.root,
                path._connect_path,
                requested_names,
                [(name, items[name]) for name in ordered_names],
            )
        finally:
            DPH5Path._file_keys.cache_clear()
            path._new_keys.clear()
        return
    if isinstance(path, DPOSPath):
        _replace_os_items(
            path,
            requested_names,
            [(name, items[name]) for name in ordered_names],
        )
        return
    raise TypeError(f"Unsupported statistics-cache path type: {type(path).__name__}.")


def _replace_h5_items(
    file: h5py.File,
    connect_path: Callable[[str], str],
    requested_names: Sequence[str],
    ordered_items: Sequence[tuple[str, np.ndarray]],
) -> None:
    """Replace HDF5 datasets while retaining an interruption marker."""
    marker_name = connect_path(_PAIR_TRANSACTION_MARKER)
    if marker_name not in file:
        file.create_dataset(marker_name, data=np.array([1], dtype=np.uint8))
    file.flush()

    for name in requested_names:
        item_name = connect_path(name)
        if item_name in file:
            del file[item_name]
    for name, value in ordered_items:
        file.create_dataset(connect_path(name), data=value)
    file.flush()

    del file[marker_name]
    file.flush()


def _replace_os_items(
    path: DPOSPath,
    requested_names: Sequence[str],
    ordered_items: Sequence[tuple[str, np.ndarray]],
) -> None:
    """Replace directory datasets while retaining an interruption marker."""
    marker = path / _PAIR_TRANSACTION_MARKER
    assert isinstance(marker, DPOSPath)
    if marker.is_dir():
        raise ValueError(f"Statistics transaction marker {marker} is a directory.")
    if not marker.is_file():
        marker.save_numpy(np.array([1], dtype=np.uint8))

    for name in requested_names:
        item = path / name
        assert isinstance(item, DPOSPath)
        item.path.unlink(missing_ok=True)
    for name, value in ordered_items:
        (path / name).save_numpy(value)

    marker.path.unlink()


def run_stat_on_chief(
    action: Callable[[], None],
    *,
    is_chief: bool,
    synchronize_failure: Callable[[bool], bool] | None,
    operation: str,
) -> None:
    """Execute a statistics action on rank 0 and synchronize its outcome.

    Parameters
    ----------
    action
        Statistics operation executed only by the chief process.
    is_chief
        Whether the current process is the chief.
    synchronize_failure
        Backend callback that broadcasts the chief failure flag and returns
        the synchronized value. ``None`` selects single-process execution.
    operation
        Human-readable operation name included in peer-rank errors.

    Returns
    -------
    None

    Raises
    ------
    Exception
        Re-raises the original exception on the chief process.
    RuntimeError
        If the chief reports failure to a peer process.
    """
    completed = False
    try:
        if is_chief:
            action()
        completed = True
    finally:
        local_failure = not completed
        failed = (
            synchronize_failure(local_failure)
            if synchronize_failure is not None
            else local_failure
        )
        if failed and not local_failure:
            raise RuntimeError(f"Rank 0 failed during {operation}; see rank-0 logs.")


class _H5StatFile:
    """Own one HDF5 handle for a single statistics initialization scope."""

    def __init__(self, path: Path, mode: StatFileMode) -> None:
        self.path = path
        self.mode = mode
        self._writable = not path.exists()
        self._file = h5py.File(path, "a" if self._writable else "r")

    def file(self, *, write: bool = False) -> h5py.File:
        """Return the live handle, promoting it for a requested write."""
        if not self._file.id.valid:
            raise RuntimeError("The statistics HDF5 file is closed.")
        if write:
            self._promote_to_writer()
        return self._file

    def flush(self) -> None:
        """Flush pending HDF5 writes."""
        self.file().flush()

    def close(self) -> None:
        """Close the owned handle if it remains open."""
        if self._file.id.valid:
            self._file.close()

    def _promote_to_writer(self) -> None:
        if self.mode == "read":
            raise ValueError("Cannot write to a read-only statistics cache.")
        if self._writable:
            return
        self._file.close()
        self._file = h5py.File(self.path, "r+")
        self._writable = True


class _H5StatPath(DPPath):
    """Provide a non-owning DPPath view over a scoped HDF5 handle."""

    def __init__(self, owner: _H5StatFile, name: str) -> None:
        self._owner = owner
        self._name = name
        self.mode = "r" if owner.mode == "read" else "a"
        self.root_path = str(owner.path)

    def __getnewargs__(self) -> tuple[str, str]:
        raise TypeError("Scoped statistics paths cannot be serialized.")

    def load_numpy(self) -> np.ndarray:
        return self._file[self._name][:]

    def load_txt(self, dtype: np.dtype | None = None, **kwargs: Any) -> np.ndarray:
        array = self.load_numpy()
        return array.astype(dtype) if dtype is not None else array

    def save_numpy(self, arr: np.ndarray) -> None:
        file = self._owner.file(write=True)
        if self._name in file:
            del file[self._name]
        file.create_dataset(self._name, data=arr)
        self._owner.flush()

    def glob(self, pattern: str) -> list[DPPath]:
        file = self._file
        if self._name == "/":
            group = file
        elif self._name not in file or not isinstance(file[self._name], h5py.Group):
            return []
        else:
            group = file[self._name]

        keys: list[str] = []
        group.visit(lambda key: keys.append(self._connect_path(key)))
        return [
            type(self)(self._owner, key)
            for key in globfilter(keys, self._connect_path(pattern))
        ]

    def rglob(self, pattern: str) -> list[DPPath]:
        return self.glob("**/" + pattern)

    def is_file(self) -> bool:
        return self._name in self._file and isinstance(
            self._file[self._name], h5py.Dataset
        )

    def is_dir(self) -> bool:
        if self._name == "/":
            self._owner.file()
            return True
        return self._name in self._file and isinstance(
            self._file[self._name], h5py.Group
        )

    def __truediv__(self, key: str) -> DPPath:
        return type(self)(self._owner, self._connect_path(key))

    def __lt__(self, other: DPPath) -> bool:
        return str(self) < str(other)

    def __str__(self) -> str:
        return f"{self.root_path}#{self._name}"

    @property
    def name(self) -> str:
        return self._name.rsplit("/", 1)[-1]

    def mkdir(self, parents: bool = False, exist_ok: bool = False) -> None:
        if self._owner.mode == "read":
            raise ValueError("Cannot write to a read-only statistics cache.")
        read_file = self._owner.file()
        if self._name in read_file:
            if not isinstance(read_file[self._name], h5py.Group) or not exist_ok:
                raise FileExistsError(f"Statistics path {self} already exists.")
            return

        file = self._owner.file(write=True)
        if parents:
            file.require_group(self._name)
        else:
            file.create_group(self._name)
        self._owner.flush()

    @property
    def _file(self) -> h5py.File:
        return self._owner.file()

    def _connect_path(self, key: str) -> str:
        return f"{self._name.rstrip('/')}/{key.lstrip('/')}"
