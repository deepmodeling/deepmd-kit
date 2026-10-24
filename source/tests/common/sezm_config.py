# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend-independent SeZM configurations for export tests."""


def _tiny_sezm_model_params() -> dict:
    """Minimal fp64 SeZM config for self-contained export tests.

    ``precision="float64"`` is what unlocks the ``rtol=1e-10, atol=1e-10``
    parity pt_expt enforces; fp32 accumulation alone drifts in the 1e-6
    range. All other knobs keep ``make_fx`` tracing small.
    """
    return {
        "type": "SeZM",
        "type_map": ["A", "B"],
        "descriptor": {
            "type": "SeZM",
            "sel": [2, 2],
            "rcut": 3.0,
            "channels": 4,
            "n_focus": 1,
            "n_radial": 3,
            "radial_mlp": [6],
            "use_env_seed": True,
            "l_schedule": [1, 0],
            "mmax": 1,
            "so2_norm": False,
            "so2_layers": 1,
            "n_atten_head": 1,
            "sandwich_norm": [True, False, True, False],
            "ffn_neurons": 8,
            "ffn_blocks": 1,
            "s2_activation": [False, True],
            "mlp_bias": False,
            "layer_scale": False,
            "use_amp": False,
            "activation_function": "silu",
            "glu_activation": True,
            "precision": "float64",
            "seed": 7,
        },
        "fitting_net": {
            "neuron": [8],
            "activation_function": "silu",
            "precision": "float64",
            "seed": 7,
        },
        "use_compile": False,
    }


def _tiny_sezm_native_spin_model_params() -> dict:
    """Minimal fp64 native-spin SeZM config for freeze routing tests."""
    params = _tiny_sezm_model_params()
    params["type_map"] = ["O", "H"]
    params["spin"] = {
        "use_spin": [True, False],
        "scheme": "native",
    }
    return params
