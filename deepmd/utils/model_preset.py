# SPDX-License-Identifier: LGPL-3.0-or-later
"""Named model presets for the DPA4 family.

A preset names a released model architecture. Writing ``"preset":
"<family>-<grade>-<version>"`` in the ``model`` section of an input file
fills in the four architecture-defining regions of the model configuration:
``type``, ``type_map``, ``descriptor`` and ``fitting_net``. Entries written
explicitly next to the preset take precedence: ``type`` and ``type_map`` as a
whole, ``descriptor`` and ``fitting_net`` key by key, so a cutoff radius or a
fitting width can be overridden, and options that are not part of an
architecture (``use_amp``, ``seed``, ``sel``, charge and spin conditioning,
...) are supplied alongside the preset.

The tables below are organised per family. Shared descriptor and fitting
options are written once, every grade lists only its scaling knobs, and every
version lists the options it changed together with the grades it ships. A new
version therefore adds one entry to the family's version table, and a new
grade adds one entry to its grade table. Existing versions are never edited.

In the multi-task layout a preset next to ``model_dict`` is the base of every
branch and of the ``shared_dict`` entries the branches reference as
``descriptor`` or ``fitting_net``, so a shared descriptor is written as just
its run-specific keys.

Presets are expanded before any other processing of the model configuration
(multi-task sharing, fine-tuning rules, argument checking), see
:func:`expand_model_preset`.
"""

import logging
from copy import (
    deepcopy,
)
from typing import (
    Any,
)

from deepmd.utils.argcheck import (
    descrpt_args_plugin,
    fitting_args_plugin,
)
from deepmd.utils.model_preset_data import (
    MODEL_PRESETS,
    PERIODIC_TABLE,
)

log = logging.getLogger(__name__)

__all__ = [
    "MODEL_PRESETS",
    "PERIODIC_TABLE",
    "expand_model_preset",
    "get_model_preset",
]

# Regions a preset may define. An expanded configuration lists the regions its
# preset defines first, in this order, followed by the remaining explicit
# entries in their original order.
_PRESET_REGIONS = ("type", "type_map", "descriptor", "fitting_net")

# Argument-schema plugin that resolves the legacy key aliases of a region's
# component, keyed by the region name.
_REGION_ARGS_PLUGIN = {
    "descriptor": descrpt_args_plugin,
    "fitting_net": fitting_args_plugin,
}


def get_model_preset(name: str) -> dict[str, Any]:
    """
    Return a copy of the model regions defined by a named preset.

    Parameters
    ----------
    name : str
        The preset name, ``<family>-<grade>-<version>``, matched
        case-insensitively.

    Returns
    -------
    dict[str, Any]
        A deep copy of the preset's ``type`` (when set), ``type_map``,
        ``descriptor`` and ``fitting_net``.

    Raises
    ------
    ValueError
        If ``name`` is not a string or names no preset.
    """
    if not isinstance(name, str):
        raise ValueError(f"model preset must be a string naming a preset, got {name!r}")
    key = name.lower()
    if key not in MODEL_PRESETS:
        raise ValueError(
            f"Unknown model preset {name!r}. Available presets: "
            + ", ".join(MODEL_PRESETS)
        )
    return deepcopy(MODEL_PRESETS[key])


def _canonicalize_aliases(
    region: str, component_type: Any, explicit: dict[str, Any]
) -> dict[str, Any]:
    """Resolve legacy key aliases in ``explicit`` to their canonical name.

    Without this, an override that names a preset-defined key by a legacy
    alias (for example ``so2_layers`` for ``mixing_layers``) would sit next to
    the preset's canonical key instead of replacing it, and the argument check
    would then reject the alias as an unknown key.
    """
    plugin = _REGION_ARGS_PLUGIN.get(region)
    if plugin is None or not isinstance(component_type, str):
        return explicit
    try:
        schema = plugin.get_argument(component_type)
    except KeyError:
        return explicit
    return schema.normalize_value(explicit, do_default=False, do_alias=True)


def _merge_region(
    region: str, preset_value: Any, explicit: Any, overrides: list[str]
) -> Any:
    """Combine one preset region with its explicit counterpart.

    ``descriptor`` and ``fitting_net`` are merged key by key: explicit keys
    replace preset keys, other keys supplement the preset, and lists inside
    are replaced as a whole. ``type`` and ``type_map``, and any other explicit
    value given for ``descriptor`` or ``fitting_net`` (a multi-task shared-dict
    reference), replace the region entirely. Entries that change a preset
    value are appended to ``overrides``; a shared-dict reference is wiring,
    not an override, and is not reported.
    """
    if region in ("descriptor", "fitting_net") and isinstance(explicit, dict):
        explicit = _canonicalize_aliases(region, preset_value.get("type"), explicit)
        overrides.extend(
            f"{region}.{key}"
            for key, value in explicit.items()
            if key in preset_value and value != preset_value[key]
        )
        return {**preset_value, **explicit}
    if isinstance(explicit, str) and region != "type":
        return explicit
    if explicit != preset_value:
        overrides.append(region)
    return explicit


def _log_expansion(name: str, overrides: list[str], scope: str = "") -> None:
    log.info(
        "Expanded model preset %r%s%s.",
        name,
        scope,
        f" with explicit overrides: {', '.join(overrides)}" if overrides else "",
    )


def _expand_single(model_config: dict[str, Any]) -> dict[str, Any]:
    """Expand the preset of one single-task model or one multi-task branch."""
    if "preset" not in model_config:
        return model_config
    name = model_config["preset"]
    preset = get_model_preset(name)
    expanded: dict[str, Any] = {}
    overrides: list[str] = []
    for region in _PRESET_REGIONS:
        if region not in preset:
            continue
        if region not in model_config:
            expanded[region] = preset[region]
        else:
            expanded[region] = _merge_region(
                region, preset[region], model_config[region], overrides
            )
    for key, value in model_config.items():
        if key != "preset" and key not in expanded:
            expanded[key] = value
    _log_expansion(name, overrides)
    return expanded


def _expand_shared_dict(
    name: str, shared_dict: dict[str, Any], branches: dict[str, Any]
) -> dict[str, Any]:
    """Merge the shared entries referenced as ``descriptor`` or ``fitting_net``
    over the corresponding regions of the preset ``name``.
    """
    roles: dict[str, str] = {}
    for branch in branches.values():
        if not isinstance(branch, dict):
            continue
        for region in ("descriptor", "fitting_net"):
            reference = branch.get(region)
            if isinstance(reference, str):
                roles[reference.split(":")[0]] = region
    preset = get_model_preset(name)
    expanded: dict[str, Any] = {}
    overrides: list[str] = []
    merged: list[str] = []
    for key, entry in shared_dict.items():
        if key in roles and isinstance(entry, dict):
            region = roles[key]
            entry = _merge_region(region, preset[region], entry, overrides)
            merged.append(key)
        expanded[key] = entry
    if merged:
        _log_expansion(name, overrides, f" for shared entries {', '.join(merged)}")
    return expanded


def expand_model_preset(model_config: dict[str, Any]) -> dict[str, Any]:
    """
    Expand the ``preset`` entries of a model configuration.

    The single-task ``model`` section and every branch of a multi-task
    ``model_dict`` may carry a ``preset``. In the multi-task layout a
    ``preset`` next to ``model_dict`` is the default for every branch that
    has none of its own and the base of the ``shared_dict`` entries that the
    branches reference as ``descriptor`` or ``fitting_net``; ``type``,
    ``type_map``, ``descriptor`` and ``fitting_net`` written next to
    ``model_dict`` are branch defaults in the same way as the model-wide
    options of the PyTorch backend, so they take part in the merge of every
    branch that expands a preset. The preset supplies ``type``, ``type_map``,
    ``descriptor`` and ``fitting_net``;
    entries written explicitly take precedence key by key inside
    ``descriptor`` and ``fitting_net`` and as a whole for ``type`` and
    ``type_map``. The ``preset`` key itself is removed, so the result is a
    plain model configuration and the function is idempotent. A ``model_dict``
    or a branch that is not a mapping is left to the argument check.

    Parameters
    ----------
    model_config : dict[str, Any]
        The ``model`` section of an input file. It is not modified; the
        explicit values it holds are reused in the result rather than copied.

    Returns
    -------
    dict[str, Any]
        The expanded model configuration, or ``model_config`` itself when it
        carries no preset.

    Raises
    ------
    ValueError
        If a preset name is not a string or is unknown.
    """
    if "model_dict" not in model_config:
        return _expand_single(model_config)
    branches = model_config["model_dict"]
    has_default = "preset" in model_config
    if not isinstance(branches, dict) or (
        not has_default
        and not any(
            isinstance(branch, dict) and "preset" in branch
            for branch in branches.values()
        )
    ):
        return model_config
    # Branch defaults: the default preset and the region entries written next
    # to ``model_dict``. Branch entries replace them as whole values.
    defaults = {
        key: model_config[key]
        for key in ("preset", *_PRESET_REGIONS)
        if key in model_config
    }
    # Every branch that receives the top-level default, or carries its own
    # preset, is merged with the defaults before its own preset is expanded;
    # a branch that reaches neither path is passed through untouched. The
    # shared-dict role scan below reads these merged branches, so an entry
    # inherited only through a top-level default (a shared-dict reference
    # among them) is still recognised as referenced.
    merged_branches = {
        branch_name: (
            {**defaults, **branch}
            if isinstance(branch, dict) and (has_default or "preset" in branch)
            else branch
        )
        for branch_name, branch in branches.items()
    }
    expanded = {
        key: value
        for key, value in model_config.items()
        # The regions distributed to every branch above no longer belong at
        # the top level; a downstream backend without its own model-wide
        # cascade would otherwise reject them as unknown top-level keys.
        if key != "preset" and not (has_default and key in _PRESET_REGIONS)
    }
    if has_default and isinstance(model_config.get("shared_dict"), dict):
        expanded["shared_dict"] = _expand_shared_dict(
            model_config["preset"], model_config["shared_dict"], merged_branches
        )
    expanded["model_dict"] = {
        branch_name: (_expand_single(branch) if isinstance(branch, dict) else branch)
        for branch_name, branch in merged_branches.items()
    }
    return expanded
