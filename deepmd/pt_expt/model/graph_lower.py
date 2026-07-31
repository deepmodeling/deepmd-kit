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
        ``"float32"`` for eligible geometrically compressed DPA1 graph
        lowers, otherwise ``"float64"``.
    """
    if lower_kind not in ("graph", "dpa1_canonical"):
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
