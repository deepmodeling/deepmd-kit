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
             "fraction_inner": 0.26, "fraction_outer": 0.80}
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

import numpy as np

__all__ = [
    "check_bridging_record_version",
    "check_window_inside_cutoff",
    "concise_bridging_window",
    "expand_bridging_method",
    "is_bridged_sezm_config",
    "migrate_inner_clamp_keys",
    "resolve_bridging_window",
    "window_midpoint",
]

BRIDGING_RECORD_VERSION = 1.3
"""Descriptor format version that introduced the current window mechanics."""

DEFAULT_FRACTION_INNER = 0.26
"""Inner window radius as a fraction of the pair's covalent bond length."""

DEFAULT_FRACTION_OUTER = 0.80
"""Outer window radius as a fraction of the pair's covalent bond length."""

_DPA4_FAMILY_TYPES = ("dpa4", "sezm")


def window_midpoint(f_inner: float, f_outer: float) -> float:
    """
    Midpoint of a bridging window, in the units of its two radii.

    The switch of the window is exactly half open there, the training-frame
    filter of a bridged model keeps pairs down to it, and the DPA4C clamp holds
    that same value below the inner radius. Reading the point from one place
    keeps the descriptors and the data filter agreeing on where the training
    data end.

    Parameters
    ----------
    f_inner : float
        Inner radius of the window.
    f_outer : float
        Outer radius of the window.

    Returns
    -------
    float
        The midpoint ``(f_inner + f_outer) / 2``.
    """
    return 0.5 * (f_inner + f_outer)


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
    "bridging_fraction_inner",
    "bridging_fraction_outer",
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


def concise_bridging_window(data: dict) -> dict:
    """Read the window the concise ``bridging_*`` keys of a model section spell.

    Parameters
    ----------
    data : dict
        A model section carrying the concise bridging keys.

    Returns
    -------
    dict
        The window fields of an ``inner_potential`` child: the two fractions,
        defaulted, and the two explicit radii, ``None`` when unset.
    """
    return {
        "fraction_inner": float(
            data.get("bridging_fraction_inner", DEFAULT_FRACTION_INNER)
        ),
        "fraction_outer": float(
            data.get("bridging_fraction_outer", DEFAULT_FRACTION_OUTER)
        ),
        "r_inner": data.get("bridging_r_inner"),
        "r_outer": data.get("bridging_r_outer"),
    }


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
    window = concise_bridging_window(data)

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
                **window,
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


def resolve_bridging_window(inner_cfg: dict) -> dict:
    """
    Resolve an ``inner_potential`` child's window into descriptor options.

    The window is a pair of fractions of each atom pair's own covalent bond
    length, so one setting describes every element combination. Explicit radii
    in Å replace it with a window that is the same for every pair; both forms
    reach the descriptor as the same three options, the absolute one through a
    unit length scale that leaves the fractions carrying the radii themselves.

    Parameters
    ----------
    inner_cfg : dict
        The ``inner_potential`` sub-model configuration.

    Returns
    -------
    dict
        The ``inner_clamp_f_inner``, ``inner_clamp_f_outer`` and
        ``inner_clamp_scale`` options of the learned sibling's descriptor.

    Raises
    ------
    ValueError
        If only one of ``r_inner`` and ``r_outer`` is given.
    """
    r_inner = inner_cfg.get("r_inner")
    r_outer = inner_cfg.get("r_outer")
    if (r_inner is None) != (r_outer is None):
        raise ValueError(
            "An explicit bridging window needs both `r_inner` and `r_outer`; "
            "leave both unset to size the window by the covalent bond length "
            "of each atom pair."
        )
    if r_inner is None:
        return {
            "inner_clamp_f_inner": float(
                inner_cfg.get("fraction_inner", DEFAULT_FRACTION_INNER)
            ),
            "inner_clamp_f_outer": float(
                inner_cfg.get("fraction_outer", DEFAULT_FRACTION_OUTER)
            ),
            "inner_clamp_scale": "covalent",
        }
    return {
        "inner_clamp_f_inner": float(r_inner),
        "inner_clamp_f_outer": float(r_outer),
        "inner_clamp_scale": "absolute",
    }


def migrate_inner_clamp_keys(config: dict) -> None:
    """Bring a descriptor configuration's window options up to date in place.

    Records predating the pair-relative window spell it as ``inner_clamp_r_inner``
    and ``inner_clamp_r_outer``, two radii in Å. The absolute scale is that same
    window measured against a unit pair length, so the radii carry over as the
    fractions themselves and the record needs no other adjustment.

    Parameters
    ----------
    config : dict
        The constructor arguments read back from a serialized descriptor.
    """
    r_inner = config.pop("inner_clamp_r_inner", None)
    r_outer = config.pop("inner_clamp_r_outer", None)
    if r_inner is None and r_outer is None:
        return
    # A record that states only one radius is carried through as it stands, so
    # the constructor reports the half-given window rather than this function.
    config["inner_clamp_f_inner"] = None if r_inner is None else float(r_inner)
    config["inner_clamp_f_outer"] = None if r_outer is None else float(r_outer)
    config["inner_clamp_scale"] = "absolute"


def check_window_inside_cutoff(
    contact_radius: np.ndarray, f_outer: float, rcut: float
) -> None:
    """Refuse a bridging window that reaches the cutoff for some pair.

    The switch and the clamp of a pair return to the identity at the outer
    radius. A pair whose outer radius lies at or beyond the cutoff leaves the
    neighbor list before that happens, so its gate never reopens and the
    displayed distance of its edges is read past the cutoff.

    Parameters
    ----------
    contact_radius : np.ndarray
        Per-type length scales in Å with shape ``(ntypes + 1,)``, the trailing
        entry belonging to the padding type.
    f_outer : float
        Outer radius of the window as a fraction of the pair length scale.
    rcut : float
        Cutoff radius in Å.

    Raises
    ------
    ValueError
        If the outer radius of the largest pair reaches ``rcut``.
    """
    outer = f_outer * 2.0 * float(np.max(contact_radius[:-1]))
    if outer >= rcut:
        raise ValueError(
            f"The bridging window of the largest pair ends at {outer:.3f} Å, at or "
            f"beyond the cutoff {rcut} Å; enlarge `rcut` or shrink the window."
        )


def check_bridging_record_version(config: dict, version: float) -> None:
    """Refuse a bridged descriptor record written before the current window.

    Records before :data:`BRIDGING_RECORD_VERSION` froze the clamp at the inner
    radius, muted a closing pair through a full product of switch amplitudes on
    the messages and carried no readout gate, so their weights were trained
    under a different function of the geometry than the one this code builds.
    An unbridged record of any version is unaffected.

    Parameters
    ----------
    config : dict
        The constructor arguments read back from a serialized descriptor, with
        the window keys already in their current spelling.
    version : float
        The ``@version`` the record was written at.

    Raises
    ------
    ValueError
        If the record is bridged and older than the current window mechanics.
    """
    if (
        version < BRIDGING_RECORD_VERSION
        and config.get("inner_clamp_f_inner") is not None
    ):
        raise ValueError(
            f"This bridged DPA4 record was written at format version {version}, "
            "before the current bridging window (clamp freeze point, leave-one-out "
            "source gate and readout gate); its weights were trained under a "
            "different function of the geometry. Retrain the bridged model."
        )
