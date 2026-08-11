# SPDX-License-Identifier: LGPL-3.0-or-later
"""Expansion of the ``bridging_method`` sugar into its canonical config form.

A bridged model IS a linear composition: the learned model plus an
analytical inner potential, summed by ``linear_ener``. The canonical
config spelling is therefore::

    "model": {
        "type": "linear_ener", "weights": "sum", "type_map": [...],
        "models": [
            {"type": "dpa4", "descriptor": {...}, "fitting_net": {...}},
            {"type": "inner_potential", "mode": "zbl",
             "r_inner": 0.5, "r_outer": 0.8}
        ]
    }

The concise spelling -- a ``bridging_method`` flag on the ``dpa4`` (or
``standard``) model type -- is the recommended user interface; it is pure
sugar over the canonical form. This module is the ONE owner of that
expansion (issue #5948): every backend's ``get_model`` entry point calls
:func:`expand_bridging_method` before dispatch, so no per-builder flag
handling can drift.
"""

import copy

__all__ = [
    "expand_bridging_method",
]

# Top-level keys that belong to the composition, not to the learned child.
_COMPOSITION_KEYS = (
    "type",
    "type_map",
    "spin",
    "atom_exclude_types",
    "pair_exclude_types",
    "bridging_method",
    "bridging_r_inner",
    "bridging_r_outer",
)


def expand_bridging_method(data: dict) -> dict:
    """Expand the ``bridging_method`` sugar into a ``linear_ener`` config.

    A config without an active ``bridging_method`` is returned unchanged
    (the same object, not a copy). A config with an active method is
    deep-copied and rewritten to the canonical composition form: a
    ``linear_ener`` model with ``weights: "sum"`` over the learned
    sub-model and an ``inner_potential`` sub-model. The exclusion lists
    move to the composition level; a top-level ``spin`` section stays at
    the top level; every other key stays on the learned child.

    For backward compatibility with the legacy pt ``type: "dpa4"``
    builder, ``descriptor.exclude_types`` is promoted to the composition's
    ``pair_exclude_types`` (and the two must match when both are given).
    Hand-written canonical configs get no such promotion.

    Parameters
    ----------
    data : dict
        The model section of a training config.

    Returns
    -------
    dict
        The canonical config; ``data`` itself when no expansion applies.

    Raises
    ------
    ValueError
        If ``bridging_method`` is set on a model type that does not
        support it, or if ``pair_exclude_types`` and
        ``descriptor.exclude_types`` are both given and differ.
    """
    method = str(data.get("bridging_method", "none"))
    if method.lower() in ("none", ""):
        return data
    model_type = str(data.get("type", "standard"))
    if model_type.lower() not in ("standard", "dpa4", "sezm"):
        raise ValueError(
            "`bridging_method` is only supported on the 'standard' and "
            f"'dpa4'/'sezm' model types, but got type {model_type!r}. "
            'Spell the composition explicitly with `type: "linear_ener"` '
            "and an `inner_potential` sub-model instead."
        )
    data = copy.deepcopy(data)
    r_inner = float(data.get("bridging_r_inner", 0.5))
    r_outer = float(data.get("bridging_r_outer", 0.8))

    # Legacy promotion (pt `type: "dpa4"` semantics): a descriptor-scoped
    # exclusion also governs the analytical term of a bridged model.
    descriptor_exclude_types = [
        list(pair) for pair in (data.get("descriptor", {}).get("exclude_types") or [])
    ]
    if "pair_exclude_types" in data:
        pair_exclude_types = [list(pair) for pair in (data["pair_exclude_types"] or [])]
        if descriptor_exclude_types and descriptor_exclude_types != pair_exclude_types:
            raise ValueError(
                "SeZM `pair_exclude_types` and `descriptor.exclude_types` must match "
                "when both are provided."
            )
    else:
        pair_exclude_types = descriptor_exclude_types

    learned = {
        key: value for key, value in data.items() if key not in _COMPOSITION_KEYS
    }
    learned["type"] = model_type
    learned["type_map"] = copy.deepcopy(data["type_map"])
    canonical = {
        "type": "linear_ener",
        "type_map": data["type_map"],
        "weights": "sum",
        "models": [
            learned,
            {
                "type": "inner_potential",
                "mode": method,
                "r_inner": r_inner,
                "r_outer": r_outer,
            },
        ],
        "atom_exclude_types": data.get("atom_exclude_types", []),
        "pair_exclude_types": pair_exclude_types,
    }
    if "spin" in data:
        canonical["spin"] = data["spin"]
    return canonical
