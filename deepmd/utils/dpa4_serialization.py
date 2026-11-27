# SPDX-License-Identifier: LGPL-3.0-or-later
"""Conversion between composed DPA4 records and the PyTorch SeZM wrapper."""

from copy import (
    deepcopy,
)
from typing import (
    Any,
)

import numpy as np

from deepmd.utils.bridging import (
    DEFAULT_TRUNCATION,
    resolve_bridging_options,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


def is_dpa4_model(data: dict[str, Any]) -> bool:
    """Return whether a serialized model contains a DPA4 descriptor."""
    descriptor = data.get("descriptor", {})
    if str(descriptor.get("type", "")).lower() in {"dpa4", "sezm"}:
        return True
    if "atomic_model" in data:
        return is_dpa4_model(data["atomic_model"])
    return any(is_dpa4_model(value) for value in data.get("models", []))


def normalize_sezm_atomic(data: dict[str, Any]) -> dict[str, Any]:
    """Convert a PyTorch ``sezm_atomic`` record to a standard atomic record.

    The PyTorch atomic record carries the state of the denoising head
    (``dens_fitting``, ``active_mode`` and the ``dens_force_rmsd`` variable)
    and ``@version`` 3. The standard atomic deserializer accepts version 2,
    and the denoising state is the only difference between the two, so the
    version is checked against the known range {2, 3} before it is rewritten.

    Parameters
    ----------
    data : dict[str, Any]
        Serialized PyTorch ``sezm_atomic`` record.

    Returns
    -------
    dict[str, Any]
        The standard atomic record with ``type`` ``"standard"`` and
        ``@version`` 2.

    Raises
    ------
    NotImplementedError
        If the record holds a populated denoising head or a non-energy
        ``active_mode``; only the energy path is portable.
    ValueError
        If the record version lies outside {2, 3}.
    """
    data = data.copy()
    check_version_compatibility(int(data.get("@version", 2)), 3, 2)
    if data.pop("dens_fitting", None) is not None:
        raise NotImplementedError(
            "A PyTorch SeZM/DPA4 record with a `dens` fitting head is not "
            "portable; only the energy path is."
        )
    active_mode = data.pop("active_mode", None)
    if active_mode not in (None, "ener"):
        raise NotImplementedError(
            f"A PyTorch SeZM/DPA4 record in active_mode {active_mode!r} is not "
            "portable; only the energy path is."
        )
    variables = data.get("@variables")
    if isinstance(variables, dict):
        data["@variables"] = {
            key: value
            for key, value in variables.items()
            if key in ("out_bias", "out_std")
        }
    data["@version"] = 2
    data["type"] = "standard"
    return data


def to_pt_dpa4_model(
    data: dict[str, Any], model_definition: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Express a DPA4 plus analytical-potential composition as a SeZM wrapper.

    The PyTorch model owns the analytical term directly, whereas array-API
    backends compose two atomic models. Output biases are additive constants
    and are combined into the learned model's output statistics. Only the
    matching sum composition is accepted; arbitrary ensembles cannot be
    represented by this wrapper.
    """
    if str(data.get("type", "")).lower() in {
        "sezm",
        "dpa4",
        "sezm_native_spin",
        "sezmproperty",
        "sezm_property",
    }:
        return data, model_definition
    spin = data.get("spin")
    backbone_type = data.get("backbone_type", data.get("type"))
    atomic = data
    method = "none"
    truncation = DEFAULT_TRUNCATION
    definition = deepcopy(model_definition)
    if backbone_type in {"linear", "linear_ener"}:
        children = data["models"]
        learned = [child for child in children if is_dpa4_model(child)]
        analytical = [
            child for child in children if child.get("type") == "inner_potential"
        ]
        if (
            len(children) != 2
            or len(learned) != 1
            or len(analytical) != 1
            or data["weights"] != "sum"
        ):
            raise ValueError(
                "PyTorch SeZM conversion requires one DPA4 plus one analytical "
                "potential with sum weights."
            )
        atomic = learned[0].copy()
        method = analytical[0]["mode"].lower()
        truncation = float(analytical[0]["truncation"])
        if any(child["type_map"] != data["type_map"] for child in children):
            raise ValueError(
                "PyTorch SeZM conversion requires identical type maps in the composition."
            )
        variables = atomic["@variables"].copy()
        variables["out_bias"] = sum(
            np.asarray(item["@variables"]["out_bias"])
            for item in (atomic, analytical[0], data)
        )
        atomic["@variables"] = variables
        for name in ("atom_exclude_types", "pair_exclude_types"):
            atomic[name] = data.get(name, [])
        if "models" in definition:
            candidates = [item for item in definition["models"] if "descriptor" in item]
            inner = [
                item
                for item in definition["models"]
                if item.get("type") == "inner_potential"
            ]
            if len(candidates) != 1 or len(inner) != 1:
                raise ValueError(
                    "The SeZM checkpoint definition must contain one learned "
                    "model and one analytical potential."
                )
            options = resolve_bridging_options(inner[0])
            definition = deepcopy(candidates[0])
            definition["type_map"] = data["type_map"]
            definition["bridging_truncation"] = options["truncation"]
            definition["bridging_clamp"] = options["clamp"]
            definition["bridging_support"] = options["support"]
            for name in ("spin", "lora"):
                if name in model_definition:
                    definition[name] = deepcopy(model_definition[name])
    atomic = atomic.copy()
    atomic.pop("spin", None)
    atomic.pop("backbone_type", None)
    atomic.update({"type": "sezm_atomic", "@version": 3})
    definition["type"] = "SeZM"
    definition["bridging_method"] = method
    result = {
        "@class": "Model",
        "@version": 2,
        "type": "sezm_native_spin" if spin is not None else "SeZM",
        "atomic_model": atomic,
        "bridging_method": method,
        "bridging_truncation": truncation,
        "lora": model_definition.get("lora"),
    }
    if atomic["fitting"]["type"] == "property":
        result["type"] = "SeZMProperty"
    if spin is not None:
        result["spin"] = spin
    return result, definition


def jax_to_pt_parameter_name(name: str) -> str:
    """Map a DPA4 NNX parameter path to the equivalent PyTorch path."""
    name = name.replace(".atomic_model.models.0.", ".atomic_model.")
    if name.startswith("atomic_model.models.0."):
        name = name.replace("atomic_model.models.0.", "atomic_model.", 1)
    name = name.replace(
        ".fitting_net.nets._networks.", ".fitting_net.filter_layers.networks."
    )
    parts = name.split(".")
    if parts[-1] in {"w", "b"}:
        parts[-1] = "matrix" if parts[-1] == "w" else "bias"
        if "fitting_net" in parts and "hidden_layers" in parts:
            parts.insert(-1, "linear")
    return ".".join(parts)


def canonical_shared_parameter_name(name: str, shared_links: dict[str, Any]) -> str:
    """Name shared state by its declared base branch in either backend."""
    for item in shared_links.values():
        links = item.get("links", [])
        if not links:
            continue
        base = links[0]
        for link in links[1:]:
            component = link["shared_type"]
            if component not in {"descriptor", "fitting_net"}:
                continue
            prefix = f"{link['model_key']}.atomic_model.{component}."
            if not name.startswith(prefix):
                continue
            suffix = name[len(prefix) :]
            if component == "descriptor" and int(link["shared_level"]) == 1:
                if not suffix.startswith(("type_embedding.", "charge_spin_embedding.")):
                    continue
            name = f"{base['model_key']}.atomic_model.{component}.{suffix}"
    return name
