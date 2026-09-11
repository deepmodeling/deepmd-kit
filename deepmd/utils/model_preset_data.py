# SPDX-License-Identifier: LGPL-3.0-or-later
"""Declarative architecture tables for named DPA4-family model presets."""

from typing import (
    Any,
)

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
    # One C^3 envelope on the messages with no envelope on the radial basis,
    # and fixed Gaussian centres in place of trainable Bessel frequencies.
    "v20260911": {
        "descriptor": {
            "edge_norm": [False, True, True],
            "sandwich_norm": [True, False, True, False],
            "env_exp": 5,
            "basis_type": "gaussian/fix",
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
    # Fixed Gaussian centres in place of trainable Bessel frequencies.
    "v20260911": {
        "descriptor": {"basis_type": "gaussian/fix"},
        "grades": ("nano", "mini", "neo", "air", "plus"),
    },
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
