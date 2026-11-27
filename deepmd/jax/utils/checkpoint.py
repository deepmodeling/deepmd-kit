# SPDX-License-Identifier: LGPL-3.0-or-later
"""Portable model, optimizer and EMA state for backend conversion."""

import logging
from collections.abc import (
    Mapping,
)
from pathlib import (
    Path,
)
from typing import (
    Any,
)

import numpy as np

from deepmd.jax.common import (
    as_pure,
    from_flat_state,
    replace_by_pure_dict,
    set_variable_value,
    to_flat_state,
    to_pure_dict,
)
from deepmd.jax.env import (
    jax,
    jnp,
    nnx,
)
from deepmd.jax.optimizer.hybrid_muon import (
    AdamState,
    HybridMuonState,
    MuonState,
    export_hybrid_muon_state,
    import_hybrid_muon_state,
)
from deepmd.jax.utils.checkpoint_state import (
    _convert_str_to_int_key,
    read_checkpoint_state,
    restore_checkpoint_state,
)
from deepmd.utils.dpa4_serialization import (
    canonical_shared_parameter_name,
    jax_to_pt_parameter_name,
)

log = logging.getLogger(__name__)


def _find_hybrid_state(state: Any) -> HybridMuonState:
    matches = [
        item
        for item in jax.tree.leaves(
            state, is_leaf=lambda value: isinstance(value, HybridMuonState)
        )
        if isinstance(item, HybridMuonState)
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one HybridMuon transformation in the optimizer state."
        )
    return matches[0]


def _hybrid_state_from_checkpoint(tree: Any) -> HybridMuonState | None:
    """Find the HybridMuon payload without depending on Optax chain indices."""
    if not isinstance(tree, dict):
        return None
    if "parameters" in tree:
        parameters = tree["parameters"]
        entries = (
            [parameters[key] for key in sorted(parameters, key=int)]
            if isinstance(parameters, dict)
            else parameters
        )
        if entries and all(
            "momentum" in item or "beta1_power" in item for item in entries
        ):
            return HybridMuonState(
                tuple(
                    MuonState(item["momentum"], item["magma_score"])
                    if "momentum" in item
                    else AdamState(
                        item["exp_avg"],
                        item["exp_avg_sq"],
                        item["beta1_power"],
                        item["beta2_power"],
                    )
                    for item in entries
                )
            )
    for child in tree.values():
        result = _hybrid_state_from_checkpoint(child)
        if result is not None:
            return result
    return None


def export_checkpoint_training(
    models: Any,
    checkpoint_path: Path,
    metadata: Mapping[str, Any],
    *,
    step: int,
) -> dict[str, Any]:
    """Read native Orbax training arrays into a named backend-neutral record.

    The record holds the step, the learning rate, the EMA parameters when the
    checkpoint has them and, for HybridMuon, the optimizer state. Adam and
    AdamW moments have no portable layout; a restart from the converted
    checkpoint starts them at zero.
    """
    config = metadata["optimizer"]
    shared_links = metadata.get("shared_links") or {}

    def canonical(name: str) -> str:
        return canonical_shared_parameter_name(
            jax_to_pt_parameter_name(name), shared_links
        )

    result: dict[str, Any] = {
        "step": step,
        "lr": metadata.get("lr", 0.0),
        "shared_links": shared_links,
    }
    if config.get("type") == "HybridMuon":
        state = read_checkpoint_state(checkpoint_path / "training_state")
        hybrid = _hybrid_state_from_checkpoint(state)
        if hybrid is None:
            raise ValueError(f"No HybridMuon state exists in {checkpoint_path}.")
        named = export_hybrid_muon_state(hybrid, nnx.state(models, nnx.Param))
        result["optimizer"] = {
            "type": "HybridMuon",
            "config": dict(config),
            "state": {canonical(name): value for name, value in named.items()},
        }
    else:
        log.warning(
            "The %s optimizer state of %s has no portable layout; a restart "
            "from the converted checkpoint starts its moments at zero.",
            config.get("type", "Adam"),
            checkpoint_path,
        )
    if (checkpoint_path / "ema_state").is_dir():
        average = nnx.clone(models)
        state = read_checkpoint_state(checkpoint_path / "ema_state")
        for branch in state.values():
            _convert_str_to_int_key(branch)
        reference = nnx.state(average)
        state = restore_checkpoint_state(state, to_pure_dict(reference))
        replace_by_pure_dict(reference, state)
        nnx.update(average, reference)
        result["ema"] = {
            "decay": metadata.get("ema_decay", 0.999),
            "parameters": {
                canonical(".".join(map(str, path))): np.asarray(
                    as_pure(variable)
                ).copy()
                for path, variable in to_flat_state(nnx.state(average, nnx.Param))
            },
            "validation_state": metadata.get("ema_validation_state", {}),
        }
    return result


def import_training_state(
    models: Any,
    optimizer: nnx.Optimizer,
    portable: Mapping[str, Any],
    ema_models: Any = None,
) -> dict[str, Any]:
    """Restore named optimizer and EMA arrays into existing JAX model graphs.

    Backend layout differences affect parameter paths only; the weight and
    optimizer array layouts are identical. Shape and route validation occurs
    before replacing the state, including the individual Adam bias clocks.
    """
    shared_links = portable.get("shared_links") or {}

    def canonical(name: str) -> str:
        return canonical_shared_parameter_name(
            jax_to_pt_parameter_name(name), shared_links
        )

    record = portable.get("optimizer")
    if record is not None:
        parameters = nnx.state(models, nnx.Param)
        current = as_pure(optimizer.opt_state)
        template = _find_hybrid_state(current)
        names = export_hybrid_muon_state(template, parameters)
        expected = {canonical(name) for name in names}
        if expected != set(record["state"]):
            raise ValueError(
                "Portable optimizer parameters do not match the target model: "
                f"missing={sorted(expected - set(record['state']))}, "
                f"extra={sorted(set(record['state']) - expected)}."
            )
        named = {name: record["state"][canonical(name)] for name in names}
        restored = import_hybrid_muon_state(named, parameters, template)
        state = jax.tree.map(
            lambda value: restored if isinstance(value, HybridMuonState) else value,
            current,
            is_leaf=lambda value: isinstance(value, HybridMuonState),
        )
        jax.tree.map(
            set_variable_value,
            optimizer.opt_state,
            state,
            is_leaf=lambda value: isinstance(value, nnx.Variable),
        )
        set_variable_value(
            optimizer.step, jnp.asarray(portable.get("step", 0), dtype=jnp.uint32)
        )
    ema = portable.get("ema")
    if ema is None or ema_models is None:
        return {}
    state = nnx.state(ema_models, nnx.Param)
    restored = []
    for path, variable in to_flat_state(state):
        name = canonical(".".join(map(str, path)))
        reference = as_pure(variable)
        value = jnp.array(ema["parameters"][name], dtype=reference.dtype, copy=True)
        if value.shape != reference.shape:
            raise ValueError(
                f"EMA parameter {name} expects shape {reference.shape}, got {value.shape}."
            )
        restored.append((path, variable.replace(value=value)))
    nnx.update(ema_models, from_flat_state(restored))
    return dict(ema.get("validation_state", {}))
