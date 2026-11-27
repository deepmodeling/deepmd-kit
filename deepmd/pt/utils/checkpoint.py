# SPDX-License-Identifier: LGPL-3.0-or-later
"""Named training state for lossless backend checkpoint conversion."""

from collections.abc import (
    Mapping,
)
from typing import (
    Any,
)

import numpy as np
import torch

from deepmd.pt.optimizer.hybrid_muon import (
    HybridMuonOptimizer,
)
from deepmd.pt.train.wrapper import (
    ModelWrapper,
)
from deepmd.utils.dpa4_serialization import (
    canonical_shared_parameter_name,
)


def _canonical_name(name: str, wrapper: ModelWrapper) -> str:
    if name.split(".", 1)[0] in wrapper.model:
        return name
    if not wrapper.multi_task:
        return f"Default.{name}"
    raise ValueError(
        f"Cannot identify the model branch for checkpoint parameter {name!r}."
    )


def infer_shared_links(
    wrapper: ModelWrapper, state: Mapping[str, Any]
) -> dict[str, Any]:
    """Recover component sharing from checkpoint tensor storage aliases.

    PyTorch preserves shared storages when saving a state dictionary. Those
    aliases identify shared modules even in records without explicit sharing
    metadata; numerically equal but independent weights are never conflated.
    """
    links = {}
    scopes = (("descriptor", 0), ("descriptor", 1), ("fitting_net", 0))
    for component, level in scopes:
        groups: dict[tuple, list[str]] = {}
        classes = {}
        for branch, model in wrapper.model.items():
            module = getattr(model.atomic_model, component, None)
            if module is None:
                continue
            names = [name for name, _ in module.named_parameters()]
            if level == 1:
                names = [
                    name
                    for name in names
                    if name.startswith(("type_embedding.", "charge_spin_embedding."))
                ]
            if not names:
                continue
            prefix = f"model.{branch}.atomic_model.{component}."
            signature = []
            for name in names:
                value = state[prefix + name]
                signature.append(
                    (
                        name,
                        value.untyped_storage().data_ptr(),
                        value.storage_offset(),
                        tuple(value.shape),
                        value.stride(),
                        value.dtype,
                    )
                )
            groups.setdefault(tuple(signature), []).append(branch)
            classes[branch] = type(module).__name__
        for branches in groups.values():
            if len(branches) < 2:
                continue
            key = f"{component}_{level}_{branches[0]}"
            links[key] = {
                "type": classes[branches[0]],
                "links": [
                    {
                        "model_key": branch,
                        "shared_type": component,
                        "shared_level": level,
                    }
                    for branch in branches
                ],
            }
    return links


def export_training_state(
    checkpoint: Mapping[str, Any], wrapper: ModelWrapper
) -> dict[str, Any]:
    """Export parameter-ID optimizer records using stable model parameter names."""
    result = dict(wrapper.train_infos)
    shared_links = checkpoint.get("shared_links") or wrapper.shared_links or {}
    if shared_links:
        result["shared_links"] = shared_links
    optimizer = checkpoint.get("optimizer")
    if optimizer is not None:
        result["optimizer"] = _export_optimizer_state(optimizer, wrapper, shared_links)
    ema = checkpoint.get("ema")
    if ema is not None:
        result["ema"] = {
            "decay": ema["decay"],
            "parameters": {
                canonical_shared_parameter_name(
                    _canonical_name(name, wrapper), shared_links
                ): value.detach().cpu().numpy().copy()
                for name, value in ema["model"].items()
            },
            "validation_state": ema.get("validation_state", {}),
        }
    return result


def _export_optimizer_state(
    optimizer: Mapping[str, Any], wrapper: ModelWrapper, shared_links: dict[str, Any]
) -> dict[str, Any]:
    """Resolve PyTorch parameter IDs and Adam clocks into named records."""
    groups = optimizer["param_groups"]
    if len(groups) != 1 or "muon_mode" not in groups[0]:
        raise ValueError(
            "Portable optimizer conversion requires a single HybridMuon parameter group."
        )
    group = groups[0]
    parameters = list(wrapper.named_parameters())
    if len(parameters) != len(group["params"]):
        raise ValueError("Optimizer parameter IDs do not match the checkpoint model.")
    config = {
        key: group[key]
        for key in (
            "momentum",
            "weight_decay",
            "lr_adjust",
            "lr_adjust_coeff",
            "muon_mode",
            "enable_gram",
            "magma_muon",
        )
    }
    config.update(
        {
            "type": "HybridMuon",
            "adam_beta1": group["adam_betas"][0],
            "adam_beta2": group["adam_betas"][1],
        }
    )
    states = {}
    for (name, parameter), index in zip(parameters, group["params"], strict=True):
        if not parameter.requires_grad:
            continue
        current = optimizer["state"].get(index, {}).copy()
        for clock in ("beta1_pow_device", "beta2_pow_device"):
            if "exp_avg" in current and clock in group:
                current[clock] = group[clock]
        canonical = canonical_shared_parameter_name(
            name.removeprefix("model."), shared_links
        )
        states[canonical] = {
            key: value.detach().cpu().numpy().copy()
            if isinstance(value, torch.Tensor)
            else np.asarray(value)
            for key, value in current.items()
        }
    return {"type": "HybridMuon", "config": config, "state": states}


def import_training_state(
    training: Mapping[str, Any], wrapper: ModelWrapper
) -> dict[str, Any]:
    """Build a PyTorch training checkpoint from portable named arrays."""
    wrapper.train_infos.update(
        step=int(training.get("step", 0)), lr=float(training.get("lr", 0.0))
    )
    result: dict[str, Any] = {"model": wrapper.state_dict()}
    shared_links = training.get("shared_links") or {}
    if shared_links:
        result["shared_links"] = shared_links
    record = training.get("optimizer")
    if record is not None:
        config = dict(record["config"])
        config.pop("type", None)
        beta1 = config.pop("adam_beta1", 0.9)
        beta2 = config.pop("adam_beta2", 0.95)
        config.pop("flash_muon", None)
        optimizer = HybridMuonOptimizer(
            wrapper.parameters(),
            lr=wrapper.train_infos["lr"],
            adam_betas=(beta1, beta2),
            flash_muon=False,
            named_parameters=wrapper.named_parameters(),
            adam_patterns=[
                pattern
                for model in wrapper.model.values()
                for pattern in model.adam_route_patterns()
            ],
            **config,
        )
        state = optimizer.state_dict()
        names = {
            canonical_shared_parameter_name(name.removeprefix("model."), shared_links)
            for name, parameter in wrapper.named_parameters()
            if parameter.requires_grad
        }
        if set(record["state"]) != names:
            raise ValueError(
                "Portable optimizer names do not match the target PyTorch model: "
                f"missing={sorted(names - set(record['state']))}, "
                f"extra={sorted(set(record['state']) - names)}."
            )
        for (name, parameter), index in zip(
            wrapper.named_parameters(), state["param_groups"][0]["params"], strict=True
        ):
            canonical = canonical_shared_parameter_name(
                name.removeprefix("model."), shared_links
            )
            current = record["state"].get(canonical, {})
            if current:
                state["state"][index] = {
                    key: torch.as_tensor(
                        np.asarray(value).copy(), device=parameter.device
                    )
                    for key, value in current.items()
                }
        result["optimizer"] = state
    ema = training.get("ema")
    if ema is not None:
        from deepmd.pt_expt.train.ema import (
            ModelEMA,
        )

        model = (
            dict(wrapper.model.items())
            if wrapper.multi_task
            else wrapper.model["Default"]
        )
        average = ModelEMA(model, decay=ema["decay"])
        shadows = {
            canonical_shared_parameter_name(
                _canonical_name(name, wrapper), shared_links
            ): parameter
            for name, parameter in average.shadow_params.items()
        }
        if set(ema["parameters"]) != set(shadows):
            raise ValueError(
                "Portable EMA names do not match the target PyTorch model: "
                f"missing={sorted(set(shadows) - set(ema['parameters']))}, "
                f"extra={sorted(set(ema['parameters']) - set(shadows))}."
            )
        for canonical, parameter in shadows.items():
            value = np.asarray(ema["parameters"][canonical])
            if value.shape != tuple(parameter.shape):
                raise ValueError(
                    f"EMA parameter {canonical} expects shape "
                    f"{tuple(parameter.shape)}, got {value.shape}."
                )
            parameter.copy_(torch.as_tensor(value.copy(), device=parameter.device))
        average.validation_state = ema.get("validation_state", {})
        result["ema"] = average.state_dict()
    return result
