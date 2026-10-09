# SPDX-License-Identifier: LGPL-3.0-or-later
"""Torch checkpoint publication for the pt_expt training backend.

Path layout and retention live in :mod:`deepmd.dpmodel.train.checkpoint`. This
module owns the Torch-specific serialization boundary: atomic ``torch.save``,
rank-gated filesystem side effects, and restart path resolution against the
configured stores.
"""

from __future__ import (
    annotations,
)

import os
from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

import torch

from deepmd.dpmodel.train import (
    CheckpointStore,
    build_checkpoint_stores,
    resolve_checkpoint_path,
)

if TYPE_CHECKING:
    from collections.abc import (
        Mapping,
    )

__all__ = ["TorchCheckpointManager"]


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    """Serialize ``payload`` to ``path`` without replacing it until complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class TorchCheckpointManager:
    """Filesystem owner for pt_expt checkpoint families.

    The trainer assembles checkpoint payloads (including any collective gather
    required under sharding). This manager resolves numbered paths, writes them
    atomically, updates latest aliases, and applies retention. Only the chief
    rank performs filesystem mutations; other ranks are no-ops for write,
    commit, and preparation side effects.
    """

    def __init__(
        self,
        store: CheckpointStore,
        ema_store: CheckpointStore,
        *,
        rank: int = 0,
    ) -> None:
        self.store = store
        self.ema_store = ema_store
        self.rank = int(rank)

    @classmethod
    def from_training_params(
        cls,
        training_params: Mapping[str, Any],
        *,
        num_steps: int,
        ema_prefix: str | Path,
        rank: int = 0,
    ) -> TorchCheckpointManager:
        """Build the manager from a normalized ``training`` section."""
        store, ema_store = build_checkpoint_stores(
            training_params,
            num_steps=num_steps,
            ema_prefix=ema_prefix,
            rank=rank,
        )
        return cls(store, ema_store, rank=rank)

    @property
    def is_chief(self) -> bool:
        """Whether this rank owns checkpoint filesystem side effects."""
        return self.rank == 0

    def store_for(self, *, ema: bool = False) -> CheckpointStore:
        """Return the regular or EMA checkpoint store."""
        return self.ema_store if ema else self.store

    def path_for(self, step: int, *, ema: bool = False) -> Path:
        """Return the numbered path for a periodic checkpoint step."""
        return self.store_for(ema=ema).path_for(step)

    def write(self, path: Path, payload: Mapping[str, Any]) -> None:
        """Atomically serialize ``payload`` to ``path`` on the chief rank."""
        if not self.is_chief:
            return
        _atomic_torch_save(payload, Path(path))

    def commit(self, path: Path, *, ema: bool = False) -> None:
        """Publish the latest alias and apply retention after a successful write."""
        if not self.is_chief:
            return
        store = self.store_for(ema=ema)
        store.publish(Path(path))
        store.prune(Path(path))

    def save_step(
        self,
        step: int,
        payload: Mapping[str, Any],
        *,
        ema: bool = False,
    ) -> Path:
        """Write, publish and prune a periodic checkpoint for ``step``.

        Returns
        -------
        Path
            The numbered path used for this step. Non-chief ranks still return
            the path even though they perform no filesystem work, so callers
            can log a single location.
        """
        path = self.path_for(step, ema=ema)
        self.write(path, payload)
        self.commit(path, ema=ema)
        return path

    def save_at(self, path: Path, payload: Mapping[str, Any]) -> None:
        """Write a checkpoint to an explicit path without publishing or pruning.

        Validation-best checkpoints use an independent namespace and retention
        policy owned by the validator; they must not move the periodic latest
        alias.
        """
        self.write(Path(path), payload)

    def resolve_restart(self, spec: str | Path, *, ema: bool = False) -> Path:
        """Resolve an explicit or latest restart target for one checkpoint family."""
        return resolve_checkpoint_path(spec, store=self.store_for(ema=ema))
