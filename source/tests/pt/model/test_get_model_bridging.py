# SPDX-License-Identifier: LGPL-3.0-or-later
"""pt realization of the canonical bridging composition (issue #5948).

The canonical config spelling is ``type: "linear_ener"`` with an
``inner_potential`` sub-model. The pt backend implements bridging inside
``SeZMModel``, so its linear builder maps the canonical form onto the
``SeZMModel`` constructor arguments; the legacy ``bridging_method`` flag
is sugar expanded by the shared normalizer at the ``get_model`` entry.
Both spellings must therefore build the same model.
"""

import copy

import pytest

from deepmd.pt.model.model import (
    SeZMModel,
    get_model,
    get_standard_model,
)


def _descriptor() -> dict:
    return {
        "type": "dpa4",
        "rcut": 4.0,
        "rcut_smth": 0.5,
        "sel": 20,
        "n_dim": 8,
        "e_dim": 8,
        "precision": "float64",
        "seed": 7,
    }


def _fitting() -> dict:
    return {
        "type": "dpa4_ener",
        "neuron": [4, 4],
        "precision": "float64",
        "seed": 7,
    }


def _sugar_config() -> dict:
    return {
        "type": "dpa4",
        "type_map": ["Ni", "O"],
        "descriptor": _descriptor(),
        "fitting_net": _fitting(),
        "bridging_method": "ZBL",
        "bridging_r_inner": 0.8,
        "bridging_r_outer": 1.2,
    }


def _canonical_config() -> dict:
    return {
        "type": "linear_ener",
        "weights": "sum",
        "type_map": ["Ni", "O"],
        "models": [
            {
                "type": "dpa4",
                "descriptor": _descriptor(),
                "fitting_net": _fitting(),
            },
            {
                "type": "inner_potential",
                "mode": "ZBL",
                "r_inner": 0.8,
                "r_outer": 1.2,
            },
        ],
    }


def test_canonical_builds_sezm_model() -> None:
    model = get_model(_canonical_config())
    assert isinstance(model, SeZMModel)
    assert model.bridging_method == "ZBL"
    assert model.bridging_r_inner == 0.8
    assert model.bridging_r_outer == 1.2


def test_canonical_matches_sugar_serialize() -> None:
    """Same seeds, both spellings: the serialized models must agree."""
    import numpy as np

    d_sugar = get_model(_sugar_config()).serialize()
    d_canon = get_model(_canonical_config()).serialize()

    def _strip_arrays(obj):
        if isinstance(obj, dict):
            return {k: _strip_arrays(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_strip_arrays(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return ("ndarray", obj.shape)
        if hasattr(obj, "detach"):  # torch tensor
            return ("tensor", tuple(obj.shape))
        return obj

    assert _strip_arrays(d_canon) == _strip_arrays(d_sugar)


def test_standard_builder_rejects_the_flag() -> None:
    """Fail fast: the pt standard builder used to silently DROP the
    bridging term.
    """
    cfg = _sugar_config()
    cfg["type"] = "standard"
    with pytest.raises(ValueError, match="bridging_method"):
        get_standard_model(cfg)


def test_get_model_expands_the_flag_on_standard_type() -> None:
    """Through the dispatcher the flag is sugar on any supported type."""
    cfg = _sugar_config()
    cfg["type"] = "standard"
    model = get_model(cfg)
    assert isinstance(model, SeZMModel)
    assert model.bridging_method == "ZBL"


def test_canonical_requires_sum_weights() -> None:
    cfg = _canonical_config()
    cfg["weights"] = "mean"
    with pytest.raises(ValueError, match="sum"):
        get_model(cfg)


def test_canonical_rejects_non_dpa4_learned_sibling() -> None:
    cfg = _canonical_config()
    cfg["models"][0]["descriptor"] = {
        "type": "se_e2_a",
        "rcut": 4.0,
        "rcut_smth": 0.5,
        "sel": [20, 20],
        "neuron": [4, 8],
    }
    with pytest.raises(NotImplementedError, match="DPA4/SeZM"):
        get_model(cfg)


def test_canonical_rejects_two_inner_children() -> None:
    cfg = _canonical_config()
    cfg["models"].append(copy.deepcopy(cfg["models"][1]))
    with pytest.raises(ValueError, match="at most one"):
        get_model(cfg)


def test_plain_linear_ener_is_unaffected() -> None:
    """A linear_ener composition without an inner_potential child keeps
    the pre-existing builder path.
    """
    sub = {
        "descriptor": {
            "type": "se_atten",
            "rcut": 4.0,
            "rcut_smth": 0.5,
            "sel": 20,
            "neuron": [4, 8],
            "attn_layer": 0,
            "seed": 1,
        },
        "fitting_net": {"neuron": [5, 5], "seed": 1},
    }
    cfg = {
        "type": "linear_ener",
        "weights": "mean",
        "type_map": ["Ni", "O"],
        "models": [copy.deepcopy(sub), copy.deepcopy(sub)],
    }
    model = get_model(cfg)
    assert not isinstance(model, SeZMModel)
