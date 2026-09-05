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

log = logging.getLogger(__name__)

__all__ = [
    "MODEL_PRESETS",
    "PERIODIC_TABLE",
    "expand_model_preset",
    "get_model_preset",
]

# fmt: off
PERIODIC_TABLE: tuple[str, ...] = (
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al",
    "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe",
    "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm",
    "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
    "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)
"""The 118 element symbols in atomic-number order, the ``type_map`` of every preset."""
# fmt: on

# Regions a preset may define. An expanded configuration lists the regions its
# preset defines first, in this order, followed by the remaining explicit
# entries in their original order.
_PRESET_REGIONS = ("type", "type_map", "descriptor", "fitting_net")

# === DPA4 (SeZM) ===
# Descriptor and fitting options shared by every DPA4 grade and version.
_DPA4_DESCRIPTOR: dict[str, Any] = {
    "type": "dpa4",
    "rcut": 6.0,
    "n_radial": 16,
    "use_env_seed": True,
    "mmax": 1,
    "radial_so2_mode": "degree_channel",
    "focus_dim": 0,
    "n_atten_head": 1,
    "message_node_so3": True,
    "ffn_neurons": 0,
    "ffn_so3_grid": True,
    "grid_mlp": False,
    "grid_branch": [0, 0, 1],
    "ffn_blocks": 1,
    "so3_readout": "mlp",
    "precision": "float32",
}
_DPA4_FITTING: dict[str, Any] = {
    "type": "dpa4_ener",
    "neuron": [0],
    "precision": "float32",
}
# Scaling knobs of each grade.
_DPA4_GRADES: dict[str, dict[str, dict[str, Any]]] = {
    "nano": {
        "descriptor": {
            "channels": 32,
            "lmax": 1,
            "n_blocks": 2,
            "mixing_layers": 3,
            "radial_so2_mode": "none",
            "n_focus": 1,
        },
    },
    "mini": {
        "descriptor": {
            "channels": 32,
            "lmax": 2,
            "n_blocks": 2,
            "mixing_layers": 3,
            "radial_so2_rank": 1,
            "n_focus": 1,
        },
    },
    "neo": {
        "descriptor": {
            "channels": 32,
            "lmax": 3,
            "n_blocks": 2,
            "mixing_layers": 3,
            "radial_so2_rank": 1,
            "n_focus": 2,
        },
    },
    "air": {
        "descriptor": {
            "channels": 64,
            "lmax": 3,
            "n_blocks": 3,
            "mixing_layers": 4,
            "radial_so2_rank": 1,
            "n_focus": 1,
        },
    },
    "plus": {
        "descriptor": {
            "channels": 64,
            "lmax": 4,
            "n_blocks": 4,
            "mixing_layers": 4,
            "radial_so2_rank": 2,
            "n_focus": 1,
        },
    },
    "pro": {
        "descriptor": {
            "channels": 64,
            "lmax": 5,
            "n_blocks": 6,
            "mixing_layers": 4,
            "radial_so2_rank": 2,
            "n_focus": 2,
            "so3_readout": "none",
        },
    },
    "max": {
        "descriptor": {
            "channels": 96,
            "lmax": 6,
            "n_blocks": 8,
            "mixing_layers": 4,
            "radial_so2_rank": 4,
            "n_focus": 2,
            "so3_readout": "none",
        },
    },
    "ultra": {
        "descriptor": {
            "channels": 128,
            "lmax": 6,
            "n_blocks": 10,
            "mixing_layers": 4,
            "radial_so2_rank": 4,
            "n_focus": 3,
            "message_node_so3": False,
            "ffn_so3_grid": False,
            "grid_branch": 0,
        },
    },
}
# Options that changed with each version, and the grades the version ships.
_DPA4_VERSIONS: dict[str, dict[str, Any]] = {
    # Channel RMSNorm on every cutoff-vanishing branch; post-norm after the
    # SO(2) branch and pre-norm before the FFN branch.
    "v20260820": {
        "descriptor": {
            "edge_norm": True,
            "sandwich_norm": [False, True, True, False],
        },
        "grades": ("nano", "mini", "neo", "air", "plus", "pro"),
    },
    # Radial-site RMSNorm removed; pre-norm before both the SO(2) and the FFN
    # branch.
    "v20260901": {
        "descriptor": {
            "edge_norm": [False, True, True],
            "sandwich_norm": [True, False, True, False],
        },
        "grades": ("nano", "mini", "neo", "air", "plus", "pro", "max", "ultra"),
    },
}

# === DPA4C ===
# DPA4C is a descriptor of the standard model, so no model ``type`` is set.
_DPA4C_DESCRIPTOR: dict[str, Any] = {
    "type": "dpa4c",
    "rcut": 6.0,
    "precision": "float32",
}
_DPA4C_FITTING: dict[str, Any] = {
    "type": "ener",
    "resnet_dt": False,
    "activation_function": "silu",
    "precision": "float32",
}
_DPA4C_GRADES: dict[str, dict[str, dict[str, Any]]] = {
    "nano": {
        "descriptor": {"channels": 8, "lmax": 2, "radial_modes": 0},
        "fitting_net": {"neuron": [96, 96, 96]},
    },
    "mini": {
        "descriptor": {"channels": 32, "lmax": 2, "radial_modes": 0},
        "fitting_net": {"neuron": [192, 192, 192]},
    },
    "neo": {
        "descriptor": {"channels": 64, "lmax": 2, "radial_modes": 0},
        "fitting_net": {"neuron": [256, 256, 256]},
    },
    "air": {
        "descriptor": {"channels": 64, "lmax": 3, "radial_modes": 4},
        "fitting_net": {"neuron": [256, 256, 256]},
    },
    "plus": {
        "descriptor": {"channels": 128, "lmax": 3, "radial_modes": 4},
        "fitting_net": {"neuron": [384, 384, 384]},
    },
}
_DPA4C_VERSIONS: dict[str, dict[str, Any]] = {
    "v20260901": {"grades": ("nano", "mini", "neo", "air", "plus")},
}


def _build_family(
    family: str,
    model_options: dict[str, Any],
    descriptor: dict[str, Any],
    fitting_net: dict[str, Any],
    grades: dict[str, dict[str, dict[str, Any]]],
    versions: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Compose the presets ``<family>-<grade>-<version>`` of one family.

    ``model_options`` holds the model-level entries shared by every preset of
    the family (the model ``type`` for DPA4, nothing for DPA4C). ``descriptor``
    and ``fitting_net`` hold the options shared by every grade and version.
    Each grade in ``grades`` adds its own ``descriptor`` and ``fitting_net``
    options, and each version in ``versions`` adds the ``descriptor`` options
    it changed and names the grades it ships.
    """
    presets: dict[str, dict[str, Any]] = {}
    for version, spec in versions.items():
        for grade in spec["grades"]:
            presets[f"{family}-{grade}-{version}"] = {
                **model_options,
                "type_map": list(PERIODIC_TABLE),
                "descriptor": {
                    **descriptor,
                    **spec.get("descriptor", {}),
                    **grades[grade].get("descriptor", {}),
                },
                "fitting_net": {**fitting_net, **grades[grade].get("fitting_net", {})},
            }
    return presets


MODEL_PRESETS: dict[str, dict[str, Any]] = {
    **_build_family(
        "dpa4",
        {"type": "dpa4"},
        _DPA4_DESCRIPTOR,
        _DPA4_FITTING,
        _DPA4_GRADES,
        _DPA4_VERSIONS,
    ),
    **_build_family(
        "dpa4c", {}, _DPA4C_DESCRIPTOR, _DPA4C_FITTING, _DPA4C_GRADES, _DPA4C_VERSIONS
    ),
}
"""All presets keyed by name; each value holds the regions the preset defines."""


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


def _merge_region(
    region: str, preset_value: Any, explicit: Any, overrides: list[str]
) -> Any:
    """Combine one preset region with its explicit counterpart.

    A mapping is merged key by key: explicit keys replace preset keys, other
    keys supplement the preset, and lists inside are replaced as a whole. Any
    other explicit value (a scalar, a list, a multi-task shared-dict reference)
    replaces the region entirely. Entries that change a preset value are
    appended to ``overrides``; a shared-dict reference is wiring, not an
    override, and is not reported.
    """
    if isinstance(explicit, dict):
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
            entry = _merge_region(key, preset[roles[key]], entry, overrides)
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
    expanded = {key: value for key, value in model_config.items() if key != "preset"}
    if has_default and isinstance(model_config.get("shared_dict"), dict):
        expanded["shared_dict"] = _expand_shared_dict(
            model_config["preset"], model_config["shared_dict"], branches
        )
    expanded["model_dict"] = {}
    for branch_name, branch in branches.items():
        if isinstance(branch, dict) and (has_default or "preset" in branch):
            branch = _expand_single({**defaults, **branch})
        expanded["model_dict"][branch_name] = branch
    return expanded
