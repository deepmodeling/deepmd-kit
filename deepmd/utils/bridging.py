# SPDX-License-Identifier: LGPL-3.0-or-later
"""Expansion of the ``bridging_method`` sugar into its canonical config form.

A bridged model IS a linear composition: the learned model plus an
analytical close-contact potential, summed by ``linear_ener``. The canonical
config spelling is therefore::

    "model": {
        "type": "linear_ener", "weights": "sum", "type_map": [...],
        "models": [
            {"type": "dpa4", "descriptor": {...}, "fitting_net": {...}},
            {"type": "inner_potential", "mode": "zbl",
             "truncation": 1.1, "clamp": 0.75, "support": 3}
        ]
    }

The analytical potential ends at ``truncation`` times each pair's covalent
bond length and is added to the learned energy without any switching
function. ``clamp`` and ``support`` configure the learned sibling: its
descriptor reads no pair separation below the data edge of that pair, the
``support``-th smallest of the distinct per-frame shortest separations of
that pair (see
:mod:`deepmd.dpmodel.utils.contact_clamp`), with the read separation frozen
below ``clamp`` times that data edge.

The concise spelling -- a ``bridging_method`` flag on the ``dpa4`` (or
``standard``) model type -- is the recommended user interface; it is pure
sugar over the canonical form. This module is the ONE owner of that
expansion (issue #5948): every backend's ``get_model`` entry point calls
:func:`expand_bridging_method` before dispatch, so no per-builder flag
handling can drift.
"""

import copy
from typing import (
    Any,
)

__all__ = [
    "DEFAULT_CLAMP",
    "DEFAULT_SUPPORT",
    "DEFAULT_TRUNCATION",
    "concise_bridging_options",
    "drop_retired_descriptor_options",
    "drop_retired_window_options",
    "expand_bridging_method",
    "is_bridged_sezm_config",
    "resolve_bridging_options",
    "route_canonical_learned_options",
]

DEFAULT_TRUNCATION = 1.1
"""Radius at which the analytical potential ends, as a fraction of the pair's
covalent bond length."""

DEFAULT_CLAMP = 0.75
"""Fraction of a pair's data edge at and below which the learned part reads a
frozen separation."""

DEFAULT_SUPPORT = 3
"""Rank, counted from the smallest, of the distinct per-frame shortest
separation of a pair that becomes its data edge."""

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
    "bridging_truncation",
    "bridging_clamp",
    "bridging_support",
)
# Model-level options of the bridging window that the contact clamp replaced.
# A bridged model section that sets one was fitted under another composition
# and is rejected by the argument check, which no longer declares them.
_RETIRED_WINDOW_KEYS = (
    "bridging_r_inner",
    "bridging_r_outer",
    "bridging_fraction_inner",
    "bridging_fraction_outer",
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
    # A long-range module belongs to the learned energy; the composition then
    # rejects it beside the bridging term.
    "long_range",
    "model_branch_alias",
    "info",
    "use_compile",
    "enable_tf32",
    "data_stat_nbatch",
    "data_stat_protect",
    "data_stat_full",
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


_NO_DEFAULT = object()
_SCHEMA_DEFAULTS: dict | None = None


def _learned_key_schema_defaults() -> dict:
    """Collect the argcheck defaults of the learned-owned keys (cached).

    Strict normalization injects these defaults on BOTH the composition
    top level and the learned child, erasing the "did the user set this?"
    provenance. The conflict resolution in
    :func:`route_canonical_learned_options` recovers it by comparing a
    value against its schema default: a level holding exactly the default
    is treated as not explicitly configured.

    Returns
    -------
    dict
        Mapping from key name to its argcheck default, for every
        ``_LEARNED_CHILD_KEYS`` entry that declares one.

    Raises
    ------
    RuntimeError
        If a key is declared with two different defaults anywhere in the
        model schema: the recovery above then has no single reference
        value and must not guess.
    """
    global _SCHEMA_DEFAULTS
    if _SCHEMA_DEFAULTS is None:
        from deepmd.utils.argcheck import (  # deferred: heavy import
            model_args,
        )

        defaults: dict = {}

        def _walk(arg: object) -> None:
            for field in getattr(arg, "sub_fields", {}).values():
                if field.name in _LEARNED_CHILD_KEYS and field.optional:
                    if field.name in defaults and defaults[field.name] != (
                        field.default
                    ):
                        raise RuntimeError(
                            f"`{field.name}` is declared with inconsistent "
                            "argcheck defaults; the canonical-route conflict "
                            "resolution relies on a single one."
                        )
                    defaults[field.name] = field.default
                _walk(field)
            for variant in getattr(arg, "sub_variants", {}).values():
                for choice in variant.choice_dict.values():
                    _walk(choice)

        _walk(model_args())
        _SCHEMA_DEFAULTS = defaults
    return _SCHEMA_DEFAULTS


def route_canonical_learned_options(composition: dict, learned: dict) -> None:
    """Route learned-model options from a canonical composition to its child.

    A canonical ``linear_ener`` config accepts generic model options (e.g.
    ``data_stat_protect``, ``preset_out_bias``) at the composition top
    level, but the learned child is their one owner: a bridged builder
    reads them from the child config only. This helper copies each
    learned-owned key present at the top level onto ``learned`` (in
    place) when the child does not set it.

    When the two levels disagree, the argcheck default decides: strict
    normalization injects defaults on both levels, so a level holding
    exactly the schema default is treated as not explicitly configured
    and the other level wins (in particular, a user-set top-level value
    survives the child default injected on the normal CLI path). Only two
    explicitly configured (non-default) values raise — a silent drop or
    a silent override there would unpin the ownership contract. The one
    unrecoverable ambiguity: explicitly setting a level to exactly the
    default value is indistinguishable from not setting it, and loses to
    an explicit non-default on the other level.

    Parameters
    ----------
    composition : dict
        The canonical ``linear_ener`` model config.
    learned : dict
        The learned child's config; modified in place.

    Raises
    ------
    ValueError
        If a learned-owned key is set to two different non-default values
        at the two levels.
    """
    for key in _LEARNED_CHILD_KEYS:
        if key not in composition:
            continue
        if key in learned:
            if learned[key] != composition[key]:
                default = _learned_key_schema_defaults().get(key, _NO_DEFAULT)
                if learned[key] == default:
                    # argcheck-injected child default: the explicit
                    # top-level value wins
                    learned[key] = copy.deepcopy(composition[key])
                elif composition[key] == default:
                    # top-level default: the explicit child value wins
                    pass
                else:
                    raise ValueError(
                        f"`{key}` is set both on the linear_ener composition "
                        f"({composition[key]!r}) and on its learned child "
                        f"({learned[key]!r}) with different values. The "
                        "learned child owns this option: set it on the child "
                        "only."
                    )
        else:
            learned[key] = copy.deepcopy(composition[key])


def concise_bridging_options(data: dict) -> dict:
    """Read the options the concise ``bridging_*`` keys of a model section spell.

    Parameters
    ----------
    data : dict
        A model section carrying the concise bridging keys.

    Returns
    -------
    dict
        The ``truncation``, ``clamp`` and ``support`` fields of an
        ``inner_potential`` child, defaulted.
    """
    return resolve_bridging_options(
        {
            "truncation": data.get("bridging_truncation", DEFAULT_TRUNCATION),
            "clamp": data.get("bridging_clamp", DEFAULT_CLAMP),
            "support": data.get("bridging_support", DEFAULT_SUPPORT),
        }
    )


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
    options = concise_bridging_options(data)

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
                **options,
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


def resolve_bridging_options(inner_cfg: dict) -> dict:
    """
    Read the truncation, the clamp and its support of an ``inner_potential`` child.

    Parameters
    ----------
    inner_cfg : dict
        The ``inner_potential`` sub-model configuration.

    Returns
    -------
    dict
        ``truncation``, the radius at which the analytical potential ends as
        a fraction of each pair's covalent bond length, and ``clamp``, the
        fraction of a pair's data edge at and below which the learned part
        reads a frozen separation, or ``None`` for a learned part that reads
        every separation unchanged, and ``support``, the rank, counted from the
        smallest, of the distinct per-frame shortest separation of a pair that
        becomes its data edge.
    """
    clamp = inner_cfg.get("clamp", DEFAULT_CLAMP)
    return {
        "truncation": float(inner_cfg.get("truncation", DEFAULT_TRUNCATION)),
        "clamp": None if clamp is None else float(clamp),
        "support": int(inner_cfg.get("support", DEFAULT_SUPPORT)),
    }


#: Descriptor options of the bridging window that the contact clamp replaced,
#: in the spellings records carry: radii in Å, pair-relative fractions and the
#: scale they are read on.
_RETIRED_DESCRIPTOR_OPTIONS = (
    "inner_clamp_r_inner",
    "inner_clamp_r_outer",
    "inner_clamp_f_inner",
    "inner_clamp_f_outer",
    "inner_clamp_scale",
)


def drop_retired_window_options(model: Any) -> Any:
    """Drop the retired window options from the unbridged models of a section.

    The bridging window was replaced by the contact clamp and the truncated
    analytical potential, so the argument check no longer declares its
    options. An unbridged model never read them, but the stored configuration
    of an unbridged model from an earlier release still holds them with their
    defaults, and such a model stays usable. A bridged model that sets one was
    fitted under another composition and is left for the argument check to
    reject. The children of a composition without an ``inner_potential`` child
    are filtered in the same way.

    Parameters
    ----------
    model : Any
        The ``model`` section of an input file, single-task or multi-task; a
        value that is not a mapping is returned unchanged.

    Returns
    -------
    Any
        The section without the options; ``model`` is not modified.
    """
    if not isinstance(model, dict):
        return model
    if str(model.get("bridging_method", "none")).lower() in ("none", ""):
        model = {
            key: value
            for key, value in model.items()
            if key not in _RETIRED_WINDOW_KEYS
        }
    if isinstance(model.get("model_dict"), dict):
        model = {
            **model,
            "model_dict": {
                key: drop_retired_window_options(branch)
                for key, branch in model["model_dict"].items()
            },
        }
    children = model.get("models")
    if isinstance(children, list) and not any(
        isinstance(child, dict) and child.get("type") == "inner_potential"
        for child in children
    ):
        model = {
            **model,
            "models": [drop_retired_window_options(child) for child in children],
        }
    return model


def drop_retired_descriptor_options(config: dict) -> None:
    """Drop the retired bridging window options from a descriptor record.

    A descriptor record of an earlier release spells the window in its
    constructor options, empty for an unbridged descriptor, which never read
    them. The options are removed in place so that the record loads through
    the current constructor.

    Parameters
    ----------
    config : dict
        Constructor arguments read back from a serialized descriptor.
    """
    for key in _RETIRED_DESCRIPTOR_OPTIONS:
        config.pop(key, None)
