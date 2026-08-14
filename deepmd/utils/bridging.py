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
    "is_bridged_sezm_config",
]

_DPA4_FAMILY_TYPES = ("dpa4", "sezm")


def is_bridged_sezm_config(data: dict) -> bool:
    """Return whether a config is a bridged DPA4/SeZM linear composition.

    True for a ``linear_ener`` config whose children contain an
    ``inner_potential`` sub-model and a DPA4/SeZM-family learned
    sub-model -- the canonical form the ``bridging_method`` sugar expands
    to. Checkpoint consumers that route DPA4/SeZM models specially (e.g.
    the ``.pt2`` freeze path) must recognize this shape too, because the
    persisted model params keep the canonical spelling while the pt
    backend realizes it as a ``SeZMModel``.

    Parameters
    ----------
    data : dict
        The model section of a training config.
    """
    if str(data.get("type", "standard")).lower() != "linear_ener":
        return False
    children = [sub for sub in (data.get("models") or []) if isinstance(sub, dict)]
    if not any(sub.get("type") == "inner_potential" for sub in children):
        return False

    def _is_dpa4_family(sub: dict) -> bool:
        if str(sub.get("type", "standard")).lower() in _DPA4_FAMILY_TYPES:
            return True
        descriptor = sub.get("descriptor")
        return (
            isinstance(descriptor, dict)
            and str(descriptor.get("type", "")).lower() in _DPA4_FAMILY_TYPES
        )

    return any(_is_dpa4_family(sub) for sub in children)


# Routing of the concise-form top-level keys during sugar expansion. Every
# key the `standard`/`dpa4` argcheck schemas declare must appear in exactly
# one tuple below: a schema-coverage test derives the key universe from
# `deepmd.utils.argcheck` and fails when a new key is left unrouted, so
# adding a model key forces an explicit routing decision here.

# Keys that belong to the composition, not to the learned child.
_COMPOSITION_KEYS = (
    "type",
    "type_map",
    "spin",
    "atom_exclude_types",
    "pair_exclude_types",
)
# Keys consumed by the expansion itself; they appear in neither the
# composition nor the learned child.
_CONSUMED_KEYS = (
    "bridging_method",
    "bridging_r_inner",
    "bridging_r_outer",
)
# Training-owned keys: the trainer reads them from the top level of the
# model section, so they stay at the composition level and must never be
# forwarded to a sub-model.
_TRAINER_KEYS = ("lora",)
# Keys that configure the learned model and are forwarded to the learned
# child. This tuple is not consulted at expansion time (the child receives
# every key not routed above); it exists so the schema-coverage test can
# assert that every argcheck key has an explicit routing decision.
_LEARNED_CHILD_KEYS = (
    "descriptor",
    "fitting_net",
    "model_branch_alias",
    "info",
    "use_compile",
    "enable_tf32",
    "data_stat_nbatch",
    "data_stat_protect",
    "data_bias_nsample",
    "use_srtab",
    "smin_alpha",
    "sw_rmin",
    "sw_rmax",
    "preset_out_bias",
    "srtab_add_bias",
    "type_embedding",
    "modifier",
    "compress",
    "finetune_head",
)
_NON_CHILD_KEYS = _COMPOSITION_KEYS + _CONSUMED_KEYS + _TRAINER_KEYS
# The routing tables are pairwise disjoint: a key has exactly one owner.
assert not set(_LEARNED_CHILD_KEYS) & set(_NON_CHILD_KEYS)


def route_canonical_learned_options(composition: dict, learned: dict) -> None:
    """Route learned-model options from a canonical composition to its child.

    A canonical ``linear_ener`` config accepts generic model options (e.g.
    ``data_stat_protect``, ``preset_out_bias``) at the composition top
    level, but the learned child is their one owner: a bridged builder
    reads them from the child config only. This helper copies each
    learned-owned key present at the top level onto ``learned`` (in
    place) when the child does not set it, and raises when both levels
    set different values — a silent drop or a silent override would both
    unpin the ownership contract.

    Parameters
    ----------
    composition : dict
        The canonical ``linear_ener`` model config.
    learned : dict
        The learned child's config; modified in place.

    Raises
    ------
    ValueError
        If a learned-owned key is set at both levels with different
        values.
    """
    for key in _LEARNED_CHILD_KEYS:
        if key not in composition:
            continue
        if key in learned:
            if learned[key] != composition[key]:
                raise ValueError(
                    f"`{key}` is set both on the linear_ener composition "
                    f"({composition[key]!r}) and on its learned child "
                    f"({learned[key]!r}) with different values. The learned "
                    "child owns this option: set it on the child only."
                )
        else:
            learned[key] = copy.deepcopy(composition[key])


def expand_bridging_method(data: dict) -> dict:
    """Expand the ``bridging_method`` sugar into a ``linear_ener`` config.

    A config without an active ``bridging_method`` is returned unchanged
    (the same object, not a copy). A config with an active method is
    deep-copied and rewritten to the canonical composition form: a
    ``linear_ener`` model with ``weights: "sum"`` over the learned
    sub-model and an ``inner_potential`` sub-model. The exclusion lists
    move to the composition level; a top-level ``spin`` section and the
    training-owned keys (``lora``) stay at the top level; every other key
    stays on the learned child.

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

    learned = {key: value for key, value in data.items() if key not in _NON_CHILD_KEYS}
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
    for key in _TRAINER_KEYS:
        if key in data:
            canonical[key] = data[key]
    return canonical
