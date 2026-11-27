# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared graph-lower model capability checks."""

from typing import (
    Any,
)

from deepmd.pt_expt.utils.canonical_graph import (
    CANONICAL_LOWER_KIND,
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
        ``"float32"`` for the compact canonical lower, whose schema fixes it,
        and for a graph lower of a model that accepts it, otherwise
        ``"float64"``.
    """
    if lower_kind == CANONICAL_LOWER_KIND:
        return "float32"
    if lower_kind != "graph":
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


def canonical_lower_evaluator(model: Any) -> str | None:
    """Return what evaluates a model on the compact canonical lower.

    The fused operators of a compressed DPA4C model and of a compressed DPA1
    model evaluate it (see ``canonical_model_eligible`` of
    :mod:`deepmd.pt_expt.kernels.dpa4c.canonical` and of
    :mod:`deepmd.pt_expt.kernels.cuda.dpa1.canonical`). Otherwise the graph
    forward of the model evaluates it on a ``NeighborGraph`` view of the
    compact graph: a graph-lower energy model whose atomic model evaluates
    the compact canonical graph (see ``supports_canonical_graph``), deployed
    on a CUDA device, that reads no frame or atomic parameter, which the
    compact schema does not carry, and has no long-range module, which only
    the fused DPA4C operators evaluate there.

    Parameters
    ----------
    model : Any
        Model exposing the atomic-model capability interface.

    Returns
    -------
    str or None
        ``"dpa4c"`` or ``"dpa1"`` for fused operators, ``"graph"`` for the
        graph forward, or ``None`` when the compact canonical lower does not
        serve the model.
    """
    from deepmd.pt_expt.kernels.cuda.dpa1.canonical import (
        canonical_model_eligible as dpa1_eligible,
    )
    from deepmd.pt_expt.kernels.dpa4c.canonical import (
        canonical_model_eligible as dpa4c_eligible,
    )
    from deepmd.pt_expt.kernels.utils import (
        backend_device_type,
    )

    if dpa4c_eligible(model):
        return "dpa4c"
    if dpa1_eligible(model):
        return "dpa1"
    if (
        backend_device_type() == "cuda"
        and model_uses_graph_lower(model)
        and bool(model.atomic_model.supports_canonical_graph())
        and model.get_dim_fparam() == 0
        and model.get_dim_aparam() == 0
        and not model.atomic_model.required_topologies()
    ):
        return "graph"
    return None
