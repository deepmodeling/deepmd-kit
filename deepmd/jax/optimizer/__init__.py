# SPDX-License-Identifier: LGPL-3.0-or-later
from collections.abc import (
    Mapping,
)
from copy import (
    deepcopy,
)
from typing import (
    Any,
)

import optax
from jax.sharding import (
    NamedSharding,
)

from .adam import (
    build_adam,
)
from .hybrid_muon import (
    build_hybrid_muon,
    export_hybrid_muon_state,
    import_hybrid_muon_state,
)


def build_optimizer(
    models: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    matrix_sharding: NamedSharding | None = None,
) -> optax.GradientTransformationExtraArgs:
    """Build the optimizer an input ``optimizer`` section configures.

    Parameters
    ----------
    models : Mapping[str, Any]
        Models of the training tasks; their ``adam_route_patterns`` select
        the HybridMuon AdamW routes.
    config : Mapping[str, Any]
        The ``optimizer`` section, with ``type`` Adam, AdamW or HybridMuon.
    matrix_sharding : NamedSharding, optional
        Placement of the HybridMuon orthogonalization batches.

    Returns
    -------
    optax.GradientTransformationExtraArgs
        Transformation taking the learning rate as an extra update argument.

    Raises
    ------
    ValueError
        If the type is unsupported or an option is unknown.
    """
    params = deepcopy(dict(config))
    optimizer_type = params.pop("type", "Adam")
    beta1 = params.pop("adam_beta1", 0.9)
    beta2 = params.pop("adam_beta2", 0.95 if optimizer_type == "HybridMuon" else 0.999)
    if optimizer_type == "HybridMuon":
        # The pure JAX implementation uses XLA for every matrix operation.
        params.pop("flash_muon", None)
        adam_patterns = sorted(
            {
                pattern
                for model in models.values()
                for pattern in model.adam_route_patterns()
            }
        )
        return build_hybrid_muon(
            adam_betas=(beta1, beta2),
            adam_patterns=adam_patterns,
            matrix_sharding=matrix_sharding,
            **params,
        )
    if optimizer_type not in {"Adam", "AdamW"}:
        raise ValueError(f"Unsupported JAX optimizer type: {optimizer_type!r}")
    weight_decay = params.pop(
        "weight_decay", 0.001 if optimizer_type == "AdamW" else 0.0
    )
    epsilon = params.pop("adam_eps", 1e-8)
    if params:
        raise ValueError(f"Unknown {optimizer_type} options: {sorted(params)}")
    return build_adam(
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        weight_decay=weight_decay,
        decoupled=optimizer_type == "AdamW",
    )


__all__ = [
    "build_adam",
    "build_hybrid_muon",
    "build_optimizer",
    "export_hybrid_muon_state",
    "import_hybrid_muon_state",
]
