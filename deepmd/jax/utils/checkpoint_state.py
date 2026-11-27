# SPDX-License-Identifier: LGPL-3.0-or-later
"""Portable array loading and empty-array slots in Orbax checkpoints."""

from pathlib import (
    Path,
)
from typing import (
    Any,
)

import numpy as np
import orbax.checkpoint as ocp

from deepmd.jax.env import (
    jax,
)


def _convert_str_to_int_key(item: dict) -> None:
    """Convert Orbax-restored numeric index keys from strings back to ints."""
    for key, value in item.copy().items():
        if isinstance(value, dict):
            _convert_str_to_int_key(value)
        if isinstance(key, str) and key.isdigit():
            item[int(key)] = item.pop(key)


def read_checkpoint_state(path: Path) -> Any:
    """Read checkpoint arrays on the host without restoring their saved mesh.

    Serialization and backend conversion consume NumPy values. Explicit
    array restore types prevent Orbax from resolving saved GPU identities or
    reconstructing an unavailable device mesh. Scalar and textual metadata
    retain their stored types; empty nodes retain their original structure.
    """
    handler = ocp.PyTreeCheckpointHandler()
    with ocp.Checkpointer(handler) as checkpointer:
        metadata = handler.metadata(path)
        restore_args = jax.tree.map(
            lambda value: (
                ocp.RestoreArgs(restore_type=np.ndarray)
                if isinstance(value, ocp.metadata.ArrayMetadata)
                and not isinstance(value, ocp.metadata.ScalarMetadata)
                else ocp.RestoreArgs()
            ),
            metadata,
        )
        return checkpointer.restore(
            path,
            args=ocp.args.PyTreeRestore(restore_args=restore_args),
        )


def pack_checkpoint_state(state: Any) -> Any:
    """Represent zero-size arrays as empty nodes for Orbax storage.

    Empty statistics carry shape and dtype but no numerical state. Their
    constructor slots provide those attributes when restoring the model.
    """
    return jax.tree.map(
        lambda value: None if hasattr(value, "size") and value.size == 0 else value,
        state,
    )


def restore_checkpoint_state(state: Any, template: Any) -> Any:
    """Restore empty-array slots from the model or optimizer state template."""
    return jax.tree.map(
        lambda value, reference: (
            reference
            if value is None and hasattr(reference, "size") and reference.size == 0
            else value
        ),
        state,
        template,
        is_leaf=lambda value: value is None,
    )
