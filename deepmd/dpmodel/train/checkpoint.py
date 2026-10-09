# SPDX-License-Identifier: LGPL-3.0-or-later
"""On-disk layout and retention policy of periodic training checkpoints.

A run writes one numbered file per checkpoint and keeps a fixed-size window of
the most recent ones.  Both the naming and the pruning are pure filesystem
concerns, independent of how a backend serializes its state, so they are
described once here and shared by every backend.
"""

from __future__ import (
    annotations,
)

import logging
import os
from math import (
    ceil,
)
from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

from deepmd.common import (
    symlink_prefix_files,
)
from deepmd.loggers import (
    is_node_main_process,
)

if TYPE_CHECKING:
    from collections.abc import (
        Mapping,
    )

log = logging.getLogger(__name__)

__all__ = [
    "CheckpointStore",
    "build_checkpoint_stores",
    "resolve_checkpoint_path",
    "resolve_keep_ckpt_count",
]


def _atomic_write_text(path: Path, text: str) -> None:
    """Replace ``path`` with ``text`` via a temporary sibling and ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class CheckpointStore:
    """Naming, publication and retention of a family of checkpoints.

    Numbered checkpoints are written as ``<directory>/<name>-<step><suffix>``,
    where ``<name>`` is the file name of ``prefix`` and ``<directory>`` is
    ``save_dir`` when given and the directory of ``prefix`` otherwise.
    Publishing a checkpoint points the prefix-named files at it, so a consumer
    that only knows the prefix always reaches the newest checkpoint.

    Nested and absolute ``prefix`` values keep their directory for the
    latest-checkpoint alias and the optional pointer file's default location
    semantics; only the file name of ``prefix`` seeds numbered files under
    ``save_dir``. When ``save_dir`` is unset, numbered files share the prefix
    directory, which is the historical prefix-relative layout.

    Parameters
    ----------
    prefix : str or Path
        The checkpoint prefix, such as ``model.ckpt``. Its directory receives
        the prefix-named symlinks, and its file name seeds the numbered files.
    save_dir : Path, optional
        Directory holding the numbered checkpoints. Defaults to the directory
        of ``prefix``.
    max_keep : int, optional
        Number of most recent numbered checkpoints to retain. Values below one
        retain every checkpoint.
    suffix : str, optional
        File suffix of a checkpoint, including the leading dot.
    pointer_file : str or Path, optional
        File recording the path of the most recently published checkpoint.
        ``None`` publishes symlinks only, which is what a secondary family of
        checkpoints, such as the EMA one, requires so that it does not claim
        the pointer of the primary family.
    """

    def __init__(
        self,
        prefix: str | Path,
        *,
        save_dir: Path | None = None,
        max_keep: int = 5,
        suffix: str = ".pt",
        pointer_file: str | Path | None = None,
    ) -> None:
        self.prefix = Path(prefix)
        self.directory = Path(save_dir) if save_dir is not None else self.prefix.parent
        self.max_keep = int(max_keep)
        self.suffix = suffix
        self.pointer_file = Path(pointer_file) if pointer_file is not None else None

    @property
    def alias_path(self) -> Path:
        """Path of the prefix-named latest-checkpoint alias for this store."""
        return Path(f"{self.prefix}{self.suffix}")

    def prepare(self) -> None:
        """Create the directories receiving the checkpoints and the symlinks."""
        self.directory.mkdir(parents=True, exist_ok=True)
        self.prefix.parent.mkdir(parents=True, exist_ok=True)

    def path_for(self, step: int) -> Path:
        """Return the path of the checkpoint recorded at a step.

        Parameters
        ----------
        step : int
            Training step encoded into the file name.

        Returns
        -------
        Path
            Path of the numbered checkpoint of this store.
        """
        return self.directory / f"{self.prefix.name}-{step}{self.suffix}"

    def step_of(self, path: Path) -> int | None:
        """Return the step encoded in a checkpoint name, or ``None``.

        Only the file name is inspected; see :meth:`holds` for membership of
        this store.

        Parameters
        ----------
        path : Path
            Candidate checkpoint path.

        Returns
        -------
        int or None
            The step of a numbered checkpoint of this store, or ``None`` when
            the name does not follow ``<name>-<step><suffix>``.
        """
        stem_prefix = f"{self.prefix.name}-"
        if path.suffix != self.suffix or not path.name.startswith(stem_prefix):
            return None
        step_text = path.name[len(stem_prefix) : -len(self.suffix)]
        if not step_text.isdigit():
            return None
        return int(step_text)

    def holds(self, path: Path) -> bool:
        """Whether a path is a numbered checkpoint of this store.

        Parameters
        ----------
        path : Path
            Candidate checkpoint path.

        Returns
        -------
        bool
            ``True`` when the path lies in this store's directory and its name
            encodes a step.
        """
        return (
            self.step_of(path) is not None
            and path.parent.resolve() == self.directory.resolve()
        )

    def publish(self, path: Path) -> None:
        """Point the prefix-named files and the pointer file at a checkpoint.

        Parameters
        ----------
        path : Path
            Checkpoint the prefix-named files resolve to from now on.
        """
        self.prefix.parent.mkdir(parents=True, exist_ok=True)
        symlink_prefix_files(str(path.with_suffix("")), str(self.prefix))
        if self.pointer_file is not None:
            _atomic_write_text(self.pointer_file, str(path))

    def resolve_latest(self) -> Path | None:
        """Return the checkpoint currently published by this store, if any.

        Prefers the pointer file when this store owns one, then the
        prefix-named alias. Relative pointer entries are resolved against the
        process working directory, matching how they were recorded.
        """
        if self.pointer_file is not None and self.pointer_file.is_file():
            recorded = Path(self.pointer_file.read_text().strip())
            if recorded.is_file():
                return recorded
        alias = self.alias_path
        if alias.exists():
            return alias.resolve() if alias.is_symlink() else alias
        return None

    def _protected_targets(self, current: Path) -> set[Path]:
        """Paths retention must not delete: the fresh write and the latest alias."""
        protected: set[Path] = set()
        for candidate in (current, self.alias_path):
            try:
                if candidate.exists():
                    protected.add(candidate.resolve())
            except OSError:
                continue
        return protected

    def prune(self, current: Path) -> None:
        """Drop the checkpoints made obsolete by a fresh one.

        Checkpoints numbered above the current step are remnants of a longer
        earlier run over the same directory. They are removed first: leaving
        them in place would let the retention window discard the freshly
        written checkpoint instead, so a rerun in a finished directory would
        keep no result at all. The window then retains the newest ``max_keep``
        checkpoints. The checkpoint just written and the target of the latest
        alias are never removed, nor are unrelated files.

        Parameters
        ----------
        current : Path
            Path of the checkpoint that was just written. A path this store
            does not hold, such as a checkpoint selected by validation, dates
            nothing and claims no slot of the window.
        """
        if self.max_keep < 1:
            return
        current_step = self.step_of(current) if self.holds(current) else None
        protected = self._protected_targets(current)
        retained: list[tuple[int, Path]] = []
        for path in self.directory.glob(f"*{self.suffix}"):
            step = self.step_of(path)
            if step is None or path.is_symlink():
                continue
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if resolved in protected:
                continue
            if current_step is not None and step > current_step:
                path.unlink(missing_ok=True)
            else:
                retained.append((step, path))
        retained.sort(key=lambda item: (item[0], item[1].name))
        # The current checkpoint occupies one slot of the window when this
        # store holds it.
        occupied = 1 if current_step is not None else 0
        excess = max(0, len(retained) + occupied - self.max_keep)
        for _, path in retained[:excess]:
            try:
                if path.resolve() in protected:
                    continue
            except OSError:
                pass
            path.unlink(missing_ok=True)


def resolve_checkpoint_path(
    spec: str | Path,
    *,
    store: CheckpointStore | None = None,
    suffix: str = ".pt",
) -> Path:
    """Resolve an explicit checkpoint, a prefix, or a latest-checkpoint handle.

    Parameters
    ----------
    spec : str or Path
        An existing checkpoint file, a checkpoint prefix (with or without the
        suffix), a directory holding a ``checkpoint`` pointer / prefix alias,
        or the literal ``latest`` when ``store`` is provided.
    store : CheckpointStore, optional
        Store consulted when ``spec`` is ``latest`` or a bare directory that
        should follow that store's pointer and alias conventions.
    suffix : str, optional
        Suffix appended when ``spec`` looks like a prefix rather than a file.

    Returns
    -------
    Path
        Path of an existing checkpoint file.

    Raises
    ------
    FileNotFoundError
        When no checkpoint can be resolved from ``spec``.
    """
    text = str(spec).strip()
    if text == "latest":
        if store is None:
            raise FileNotFoundError(
                "Cannot resolve checkpoint 'latest' without a checkpoint store."
            )
        resolved = store.resolve_latest()
        if resolved is None:
            raise FileNotFoundError(
                "No latest checkpoint is published for the configured store."
            )
        return resolved

    path = Path(text)
    if path.is_file():
        return path

    if path.is_dir():
        if store is not None:
            resolved = store.resolve_latest()
            if resolved is not None and resolved.is_file():
                return resolved
        pointer = path / "checkpoint"
        if pointer.is_file():
            recorded = Path(pointer.read_text().strip())
            candidate = recorded if recorded.is_absolute() else path / recorded
            if candidate.is_file():
                return candidate
            if recorded.is_file():
                return recorded
        for name in ("model.ckpt" + suffix, "model.ckpt.pt"):
            alias = path / name
            if alias.exists():
                return alias.resolve() if alias.is_symlink() else alias
        raise FileNotFoundError(f"Cannot find a checkpoint in directory {path}.")

    suffixed = path if path.name.endswith(suffix) else Path(f"{path}{suffix}")
    if suffixed.exists():
        return suffixed.resolve() if suffixed.is_symlink() else suffixed
    raise FileNotFoundError(f"Cannot find checkpoint {spec!r}.")


def resolve_keep_ckpt_count(
    ckpt_keep_ratio: float | None, num_steps: int, save_freq: int
) -> int | None:
    """Convert a checkpoint-retention ratio into a sliding-window keep count.

    A checkpoint is written every ``save_freq`` steps and once more at the
    final step, so a run of ``num_steps`` produces ``ceil(num_steps /
    save_freq)`` of them in total (the terminal checkpoint is off-cadence when
    ``num_steps`` is not a multiple of ``save_freq``). Keeping the most recent
    ``ceil(ratio * total)`` is equivalent to retaining the final ``ratio``
    fraction of the run by step, without the caller computing the count by
    hand.

    Parameters
    ----------
    ckpt_keep_ratio : float or None
        The fraction of the training run, by step, whose periodic checkpoints
        are retained. ``None`` leaves the keep count unchanged.
    num_steps : int
        The total number of training steps, already resolved (including when
        derived from ``numb_epoch``).
    save_freq : int
        The checkpoint saving frequency in steps. Values below one disable
        periodic saving, leaving the final checkpoint as the only one.

    Returns
    -------
    int or None
        The number of most recent checkpoints to keep (at least one), or
        ``None`` when ``ckpt_keep_ratio`` is not set.
    """
    if ckpt_keep_ratio is None:
        return None
    total_ckpts = max(1, ceil(num_steps / save_freq)) if save_freq > 0 else 1
    return max(1, ceil(ckpt_keep_ratio * total_ckpts))


def build_checkpoint_stores(
    training_params: Mapping[str, Any],
    *,
    num_steps: int,
    ema_prefix: str | Path,
    rank: int = 0,
) -> tuple[CheckpointStore, CheckpointStore]:
    """Build the checkpoint stores of a training run.

    A run keeps two families of checkpoints: the periodic ones, which carry
    the live weights and the state needed to resume, and the EMA ones, which
    carry smoothed weights only. They share a directory and differ in prefix,
    retention and whether they own the pointer file.

    Parameters
    ----------
    training_params : Mapping[str, Any]
        The normalized ``training`` section. ``save_ckpt``, ``save_dir``,
        ``save_freq``, ``max_ckpt_keep``, ``ckpt_keep_ratio`` and
        ``ema_ckpt_keep`` are read from it. When ``ema_ckpt_keep`` is unset,
        the EMA family inherits ``max_ckpt_keep``.
    num_steps : int
        The resolved run length, needed to turn ``ckpt_keep_ratio`` into a
        keep count.
    ema_prefix : str or Path
        Checkpoint prefix of the EMA family, derived by the backend from
        ``save_ckpt``.
    rank : int, optional
        Global rank. Only the chief creates directories. The resolved
        retention is reported by each node's local chief when available.

    Returns
    -------
    tuple[CheckpointStore, CheckpointStore]
        The periodic store and the EMA store.
    """
    save_dir = training_params.get("save_dir")
    save_freq = int(training_params.get("save_freq", 1000))
    max_keep = int(training_params.get("max_ckpt_keep", 5))
    configured_ema_max_keep = training_params.get("ema_ckpt_keep")
    ema_max_keep = (
        max_keep if configured_ema_max_keep is None else int(configured_ema_max_keep)
    )
    ckpt_keep_ratio = training_params.get("ckpt_keep_ratio")

    keep_ckpt_count = resolve_keep_ckpt_count(ckpt_keep_ratio, num_steps, save_freq)
    if keep_ckpt_count is not None:
        max_keep = keep_ckpt_count
        ema_max_keep = keep_ckpt_count
        if is_node_main_process(rank):
            log.info(
                "Resolved checkpoint retention to %d from ckpt_keep_ratio=%s "
                "(num_steps=%d, save_freq=%d).",
                keep_ckpt_count,
                ckpt_keep_ratio,
                num_steps,
                save_freq,
            )

    directory = Path(save_dir) if save_dir else None
    store = CheckpointStore(
        training_params.get("save_ckpt", "model.ckpt"),
        save_dir=directory,
        max_keep=max_keep,
        pointer_file="checkpoint",
    )
    ema_store = CheckpointStore(
        ema_prefix,
        save_dir=directory,
        max_keep=ema_max_keep,
    )
    if rank == 0:
        store.prepare()
    return store, ema_store
