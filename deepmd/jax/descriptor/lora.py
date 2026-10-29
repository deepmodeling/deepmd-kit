# SPDX-License-Identifier: LGPL-3.0-or-later
"""DPA4 LoRA adapters, parameter policy, and portable checkpoint handling."""

from collections.abc import (
    Mapping,
    Sequence,
)
from typing import (
    Any,
)
from zlib import (
    crc32,
)

import numpy as np

from deepmd.dpmodel.descriptor.dpa4 import (
    DescrptDPA4,
)
from deepmd.dpmodel.descriptor.dpa4_nn.activation import (
    GatedActivation,
)
from deepmd.dpmodel.descriptor.dpa4_nn.lora import (
    _OVERRIDE_FREEZE_LEAF_NAMES,
    _UNFREEZE_LEAF_NAMES,
    _UNFREEZE_PER_BLOCK_SUBPATHS,
    _UNFREEZE_SUBMODULE_PATHS,
)
from deepmd.dpmodel.descriptor.dpa4_nn.lora import LoRASO2 as LoRASO2DP
from deepmd.dpmodel.descriptor.dpa4_nn.lora import LoRASO3 as LoRASO3DP
from deepmd.dpmodel.descriptor.dpa4_nn.lora import (
    _leaf_name,
)
from deepmd.dpmodel.descriptor.dpa4_nn.so2 import (
    SO2Linear,
)
from deepmd.dpmodel.descriptor.dpa4_nn.so3 import (
    SO3Linear,
)
from deepmd.dpmodel.fitting.dpa4_ener import (
    _set_trainable_metadata,
)
from deepmd.dpmodel.utils.network import NativeLayer as NativeLayerDP
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.jax.common import (
    ArrayAPIVariable,
    flax_module,
    register_dpmodel_mapping,
    try_convert_module,
)
from deepmd.jax.env import (
    jnp,
    nnx,
)
from deepmd.jax.utils.network import (
    ArrayAPIParam,
)


def _value(value: Any) -> Any:
    return value[...] if isinstance(value, nnx.Variable) else value


def _promote_adapter(module: Any) -> Any:
    """Expose adapters and scalar biases while keeping base matrices frozen."""
    for name in ("A_by_l", "B_by_l", "A_m0", "B_m0", "bias", "bias0"):
        value = getattr(module, name, None)
        if value is not None:
            nnx.Module.__setattr__(
                module, name, ArrayAPIParam(jnp.asarray(_value(value)))
            )
    for name in ("A_m", "B_m"):
        values = getattr(module, name, None)
        if values is not None:
            setattr(
                module,
                name,
                [ArrayAPIParam(jnp.asarray(_value(value))) for value in values],
            )
    if hasattr(module, "weight_m"):
        module.weight_m = [
            ArrayAPIVariable(jnp.asarray(_value(value))) for value in module.weight_m
        ]
    return module


@flax_module
class LoRASO3(LoRASO3DP):
    """Per-degree SO(3) adapters with frozen base matrices."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("trainable", True)
        super().__init__(**kwargs)
        _promote_adapter(self)

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "LoRASO3":
        return _promote_adapter(super().deserialize(data))


@flax_module
class LoRASO2(LoRASO2DP):
    """Per-order SO(2) adapters with frozen base matrices."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("trainable", True)
        super().__init__(**kwargs)
        _promote_adapter(self)

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "LoRASO2":
        return _promote_adapter(super().deserialize(data))


register_dpmodel_mapping(
    LoRASO3DP, lambda module: LoRASO3.deserialize(module.serialize())
)
register_dpmodel_mapping(
    LoRASO2DP, lambda module: LoRASO2.deserialize(module.serialize())
)


def _get_child(root: Any, path: tuple[str | int, ...]) -> Any:
    value = root
    for part in path:
        value = (
            value[part]
            if isinstance(value, (Mapping, Sequence))
            else getattr(value, part)
        )
    return value


def _set_child(root: Any, path: tuple[str | int, ...], value: Any) -> None:
    parent = _get_child(root, path[:-1])
    name = path[-1]
    if isinstance(parent, (Mapping, Sequence)):
        parent[name] = value
    else:
        # NativeLayer's setter derives trainability from a module-wide flag.
        # The fine-tune policy is per parameter, including bias-only training.
        nnx.Module.__setattr__(parent, name, value)


def _path_name(path: tuple[str | int, ...]) -> str:
    return ".".join(str(part) for part in path)


def _under(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def _parameter_nodes(
    nodes: list[tuple[tuple[str | int, ...], Any]],
) -> list[tuple[tuple[str | int, ...], nnx.Variable]]:
    """Identify trainable and frozen model parameters without selecting constants."""
    from deepmd.jax.descriptor.dpa4 import (
        _TRAINABLE_ATTRS,
        _TRAINABLE_LIST_ATTRS,
    )

    parameters = {
        id(value): (path, value)
        for path, value in nodes
        if isinstance(value, nnx.Param)
    }
    for path, module in nodes:
        attributes = getattr(
            module,
            "PARAMETER_ATTRIBUTES",
            _TRAINABLE_ATTRS.get(type(module).__name__, ()),
        )
        list_attributes = getattr(
            module,
            "PARAMETER_LIST_ATTRIBUTES",
            _TRAINABLE_LIST_ATTRS.get(type(module).__name__, ()),
        )
        if isinstance(module, NativeLayerDP):
            attributes = ("w", "b", "idt")
        elif isinstance(module, LoRASO3DP):
            attributes = ("weight", "bias", "A_by_l", "B_by_l")
        elif isinstance(module, LoRASO2DP):
            attributes = ("weight_m0", "bias0", "A_m0", "B_m0")
            list_attributes = ("weight_m", "A_m", "B_m")
        for name in attributes:
            value = getattr(module, name, None)
            if isinstance(value, nnx.Variable):
                parameters.setdefault(id(value), ((*path, name), value))
        for name in list_attributes:
            values = getattr(module, name, None)
            if values is not None:
                for index, value in enumerate(values):
                    parameters.setdefault(id(value), ((*path, name, index), value))
    return list(parameters.values())


def _apply_trainable_policy(model: Any) -> None:
    """Apply the production SeZM fine-tune policy to NNX parameter leaves."""
    nodes = list(nnx.iter_graph(model))
    for path, module in nodes:
        if isinstance(module, nnx.Module) and any(
            part in ("fitting_net", "dens_fitting_net") for part in path
        ):
            _set_trainable_metadata(module, True)
    descriptor_paths = [
        _path_name(path) for path, module in nodes if isinstance(module, DescrptDPA4)
    ]
    gated_paths = [
        _path_name(path)
        for path, module in nodes
        if isinstance(module, GatedActivation)
    ]
    replacements = {}
    for path, parameter in _parameter_nodes(nodes):
        name = _path_name(path)
        canonical = name
        for prefix in descriptor_paths:
            if not prefix or _under(name, prefix):
                suffix = name[len(prefix) :].lstrip(".")
                canonical = f"atomic_model.descriptor.{suffix}"
                break
        leaf = _leaf_name(name)
        trainable = leaf in _UNFREEZE_LEAF_NAMES or "bias" in leaf or leaf == "b"
        trainable = trainable or any(
            _under(canonical, prefix) for prefix in _UNFREEZE_SUBMODULE_PATHS
        )
        trainable = trainable or any(
            part in ("fitting_net", "dens_fitting_net") for part in path
        )
        blocks_prefix = "atomic_model.descriptor.blocks."
        if canonical.startswith(blocks_prefix):
            block_suffix = canonical[len(blocks_prefix) :].split(".", 1)[1]
            trainable = trainable or any(
                _under(block_suffix, prefix) for prefix in _UNFREEZE_PER_BLOCK_SUBPATHS
            )
        if leaf in _OVERRIDE_FREEZE_LEAF_NAMES or any(
            not prefix or _under(name, prefix) for prefix in gated_paths
        ):
            trainable = False
        if trainable != isinstance(parameter, nnx.Param):
            variable_type = ArrayAPIParam if trainable else ArrayAPIVariable
            replacements[id(parameter)] = variable_type(parameter[...])
    _replace_modules(model, replacements)


def _make_adapter(
    module: Any,
    rank: int,
    alpha: float | None,
    seed: int | list[int] | None = None,
) -> Any:
    config = module.serialize()["config"].copy()
    config["trainable"] = True
    config["seed"] = seed
    adapter_class = LoRASO3 if isinstance(module, SO3Linear) else LoRASO2
    adapter = adapter_class(**config, lora_rank=rank, lora_alpha=alpha)
    names = (
        ("weight", "bias") if isinstance(module, SO3Linear) else ("weight_m0", "bias0")
    )
    for name in names:
        value = getattr(module, name)
        if value is not None:
            variable_type = ArrayAPIParam if "bias" in name else ArrayAPIVariable
            nnx.Module.__setattr__(adapter, name, variable_type(_value(value)))
    if isinstance(module, SO2Linear):
        adapter.weight_m = [
            ArrayAPIVariable(_value(value)) for value in module.weight_m
        ]
    return adapter


def has_lora(model: Any) -> bool:
    """Return whether the model graph contains SO(3) or SO(2) adapters."""
    return any(
        isinstance(module, (LoRASO3DP, LoRASO2DP))
        for _, module in nnx.iter_graph(model)
    )


def apply_lora_to_sezm(model: Any, *, rank: int, alpha: float | None = None) -> Any:
    """
    Inject adapters and apply the SeZM fine-tune policy in place.

    Parameters
    ----------
    model : Any
        DPA4 model, descriptor, or shared NNX model graph.
    rank : int
        Positive adapter rank.
    alpha : float, optional
        Scaling numerator; the default equals ``rank``.

    Returns
    -------
    Any
        The same graph with frozen base matrices, trainable adapters, and the
        production SeZM policy for remaining parameters. Shared module aliases
        retain a single adapter object. Repeated application preserves adapters.
    """
    if rank < 1:
        raise ValueError(f"LoRA requires rank >= 1, got {rank}.")
    nodes = list(nnx.iter_graph(model))
    descriptors = [
        (path, module) for path, module in nodes if isinstance(module, DescrptDPA4)
    ]
    replacements = {}
    for path, module in nodes:
        if isinstance(module, (LoRASO3DP, LoRASO2DP)):
            requested_alpha = float(rank) if alpha is None else float(alpha)
            if (module.lora_rank, module.lora_alpha) != (rank, requested_alpha):
                raise ValueError(
                    f"LoRA adapter {_path_name(path)!r} has rank={module.lora_rank}, "
                    f"alpha={module.lora_alpha}; requested rank={rank}, alpha={requested_alpha}. "
                    "Merge the existing adapters before changing their configuration."
                )
            continue
        if isinstance(module, (SO3Linear, SO2Linear)):
            seed = None
            for descriptor_path, descriptor in descriptors:
                if path[: len(descriptor_path)] == descriptor_path:
                    relative_path = _path_name(path[len(descriptor_path) :])
                    seed = child_seed(
                        descriptor.seed, crc32(relative_path.encode("utf-8"))
                    )
                    break
            replacements[id(module)] = _make_adapter(module, rank, alpha, seed)
    _replace_modules(model, replacements)
    _apply_trainable_policy(model)
    return model


def _replace_modules(model: Any, replacements: dict[int, Any]) -> None:
    """Replace every reference to each selected module, including shared aliases."""
    for _, module in list(nnx.iter_graph(model)):
        if isinstance(module, Mapping):
            children = list(module.items())
        elif isinstance(module, Sequence):
            children = list(enumerate(module))
        elif isinstance(module, nnx.Module):
            children = list(vars(module).items())
        else:
            continue
        for name, child in children:
            replacement = replacements.get(id(child))
            if replacement is not None:
                _set_child(module, (name,), replacement)


def add_lora_scaling(model: Any, variables: dict[str, np.ndarray]) -> None:
    """Add the scaling buffers required by the portable PyTorch variable layout."""
    for path, module in nnx.iter_graph(model):
        if isinstance(module, (LoRASO3DP, LoRASO2DP)):
            variables[f"{_path_name(path)}.lora_scaling"] = np.asarray(
                module.lora_scaling
            )


def restore_lora_from_variables(
    model: Any, variables: Mapping[str, np.ndarray]
) -> None:
    """Restore adapters from their flattened portable descriptor variables."""
    adapters = []
    for path, module in list(nnx.iter_graph(model)):
        if not isinstance(module, (SO3Linear, SO2Linear)):
            continue
        prefix = _path_name(path) + "."
        first_key = "A_by_l" if isinstance(module, SO3Linear) else "A_m0"
        if prefix + first_key not in variables:
            partner = "B_by_l" if isinstance(module, SO3Linear) else "B_m0"
            if prefix + partner in variables:
                raise ValueError(f"LoRA checkpoint is missing {prefix + first_key!r}.")
            continue
        first = variables[prefix + first_key]
        rank = first.shape[1] if first_key == "A_by_l" else first.shape[0]
        scaling = float(np.asarray(variables.get(prefix + "lora_scaling", 1.0)))
        adapter = _make_adapter(module, rank, rank * scaling)
        data = adapter.serialize()
        for key in data["@variables"]:
            if prefix + key in variables:
                data["@variables"][key] = variables[prefix + key]
            elif key.startswith(("A_", "B_")):
                raise ValueError(f"LoRA checkpoint is missing {prefix + key!r}.")
        adapters.append((path, type(adapter).deserialize(data)))
    for path, adapter in adapters:
        _set_child(model, path, adapter)
    if adapters:
        _apply_trainable_policy(model)


def merge_lora_into_base(model: Any) -> Any:
    """
    Fold adapters into base matrices and replace modules in place.

    Optimizer and compiled functions referencing the adapter graph must be
    reconstructed after merging. Use ``serialize_merged_model`` to export a
    model without changing the training graph or its optimizer state.
    """
    replacements = {
        id(module): try_convert_module(module.merge_into_base())
        for _, module in nnx.iter_graph(model)
        if isinstance(module, (LoRASO3DP, LoRASO2DP))
    }
    _replace_modules(model, replacements)
    return model


def serialize_merged_model(model: Any) -> dict[str, Any]:
    """Serialize a plain model with adapter deltas folded into its matrices."""
    clone = nnx.clone(model)
    merge_lora_into_base(clone)
    return clone.serialize()
