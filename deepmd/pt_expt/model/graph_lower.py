# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared graph-lower model capability checks."""

from typing import (
    Any,
)


def graph_edge_dtype(model: Any, lower_kind: str) -> str:
    """Return the graph edge-vector dtype encoded by a deployment artifact.

    The dtype itself is the atomic model's capability, so compositions
    answer by aggregation instead of by a reach-in through a single
    ``.descriptor`` (a ``LinearEnergyAtomicModel`` has none). Only the
    lower-kind applicability -- which artifact kinds carry edge geometry at
    all -- is decided here.

    Parameters
    ----------
    model : Any
        Model exposing the atomic-model capability interface.
    lower_kind : str
        Concrete lower-forward schema.

    Returns
    -------
    str
        ``"float32"`` for eligible compressed DPA1 or DPA4C graph lowers,
        otherwise ``"float64"``.
    """
    if lower_kind not in ("graph", "dpa1_canonical", "dpa4c_canonical"):
        return "float64"
    return str(model.atomic_model.graph_edge_dtype())


def model_uses_graph_lower(model: Any) -> bool:
    """Return whether a model's default lower uses ``NeighborGraph``.

    Parameters
    ----------
    model : Any
        Model exposing the atomic-model and descriptor capability interfaces.

    Returns
    -------
    bool
        ``True`` when the model is mixed-type and its descriptor enables the
        graph lower.
    """
    mixed_types = getattr(model, "mixed_types", None)
    if mixed_types is None:
        return False
    try:
        if not bool(mixed_types()):
            return False
    except (AttributeError, NotImplementedError):
        return False

    # ENERGY-output models only, mirroring ``_resolve_graph_method``'s
    # default-flip gate: the compiled-training graph trace is
    # energy-specific (``do_grad_r("energy")``, ``_translate_energy_keys``),
    # so a non-energy model (property/dos/dipole/polar) on this path raises
    # ``KeyError('energy')`` at its first compiled batch.
    try:
        if "energy" not in model.atomic_output_def().keys():
            return False
    except (AttributeError, NotImplementedError):
        return False

    atomic_model = getattr(model, "atomic_model", None)
    if atomic_model is None:
        return False
    return bool(atomic_model.uses_graph_lower())


# Training families whose graph forward does not truncate neighbors. Keep the
# allowlist explicit so new families can opt in after their training path is checked.
_NEIGHBOR_STAT_FREE_DESCRIPTORS = frozenset({"dpa4", "dpa4c", "DPA4", "sezm", "SeZM"})


def model_config_skips_neighbor_stat(config: dict) -> bool:
    """Whether a fresh DPA4-family training model can omit neighbor statistics.

    Unlike the runtime predicate, this operates before constructing a model.
    Checkpoint restoration must be excluded by the caller, since restored buffers may
    disable the graph route.
    """
    from deepmd.utils.bridging import (
        expand_bridging_method,
    )

    config = expand_bridging_method(config)
    model_type = config.get("type", "standard")
    if model_type == "linear_ener":
        children = config.get("models", [])
        return bool(children) and all(
            child.get("type") == "inner_potential"
            or model_config_skips_neighbor_stat(child)
            for child in children
        )
    dpa4_model = model_type in ("dpa4", "DPA4", "sezm", "SeZM")
    if model_type != "standard" and not dpa4_model:
        return False
    if "spin" in config and config["spin"].get("scheme", "deepspin") != "native":
        return False
    if "use_srtab" in config:
        return False
    fitting_type = config.get("fitting_net", {}).get(
        "type", "dpa4_ener" if dpa4_model else "ener"
    )
    if fitting_type not in ("ener", "direct_force_ener", "dpa4_ener", "sezm_ener"):
        return False
    descriptor = config.get("descriptor", {})
    descriptor_type = descriptor.get("type", "dpa4" if dpa4_model else "se_e2_a")
    return descriptor_type in _NEIGHBOR_STAT_FREE_DESCRIPTORS


def resolve_stat_free_selections(config: dict) -> dict:
    """Replace unused automatic capacities with their ordinary schema default.

    Call only after ``model_config_skips_neighbor_stat`` succeeds. No model
    weights or normalization statistics depend on this capacity on that route.
    Keep numeric capacities for explicit dense-lower export compatibility.
    """
    import copy

    from deepmd.dpmodel.utils.update_sel import (
        UpdateSel,
    )
    from deepmd.utils.argcheck import (
        DEFAULT_DPA4_SEL,
    )
    from deepmd.utils.bridging import (
        expand_bridging_method,
    )

    config = copy.deepcopy(expand_bridging_method(config))
    if config.get("type") == "linear_ener":
        config["models"] = [resolve_stat_free_selections(m) for m in config["models"]]
    descriptor = config.get("descriptor", {})
    if isinstance(descriptor.get("sel"), str):
        # Validate the existing auto syntax without opening training data.
        UpdateSel().parse_auto_sel_ratio(descriptor["sel"])
        # The capacity is unused in graph training, including any auto ratio.
        descriptor["sel"] = DEFAULT_DPA4_SEL
    return config
