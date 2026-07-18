# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared graph-lower model capability checks."""

from typing import (
    Any,
)


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

    descriptor = getattr(getattr(model, "atomic_model", None), "descriptor", None)
    uses_graph_lower = getattr(descriptor, "uses_graph_lower", None)
    if uses_graph_lower is None:
        return False
    try:
        return bool(uses_graph_lower())
    except (AttributeError, NotImplementedError):
        return False
