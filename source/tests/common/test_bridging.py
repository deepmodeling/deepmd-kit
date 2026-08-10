# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for the ``bridging_method`` sugar expansion (issue #5948).

``expand_bridging_method`` is the ONE owner of the sugar: it rewrites a
flag-form config into the canonical ``linear_ener`` composition over the
learned model and an ``inner_potential`` sub-model. These tests pin the
key routing, the legacy exclusion promotion, and the rejections.
"""

import copy

import pytest

from deepmd.utils.bridging import (
    expand_bridging_method,
)


def _flag_config() -> dict:
    return {
        "type": "dpa4",
        "type_map": ["Ni", "O"],
        "descriptor": {"type": "dpa4", "rcut": 4.0, "sel": 8},
        "fitting_net": {"type": "dpa4_ener", "neuron": [8, 8]},
        "bridging_method": "ZBL",
        "bridging_r_inner": 0.8,
        "bridging_r_outer": 1.2,
    }


@pytest.mark.parametrize(
    "method",
    [
        None,  # key absent
        "none",  # lower-case disable spelling
        "None",  # argcheck default spelling
        "",  # empty string disables too
    ],
)
def test_inactive_flag_returns_config_unchanged(method) -> None:
    data = _flag_config()
    if method is None:
        del data["bridging_method"]
    else:
        data["bridging_method"] = method
    assert expand_bridging_method(data) is data


def test_expansion_shape() -> None:
    out = expand_bridging_method(_flag_config())
    assert out["type"] == "linear_ener"
    assert out["weights"] == "sum"
    assert out["type_map"] == ["Ni", "O"]
    learned, inner = out["models"]
    assert learned["type"] == "dpa4"
    assert learned["descriptor"]["type"] == "dpa4"
    assert learned["fitting_net"]["type"] == "dpa4_ener"
    assert inner == {
        "type": "inner_potential",
        "mode": "ZBL",
        "r_inner": 0.8,
        "r_outer": 1.2,
    }
    # the flag keys must not leak into the canonical config
    for key in ("bridging_method", "bridging_r_inner", "bridging_r_outer"):
        assert key not in out
        assert key not in learned


def test_default_radii() -> None:
    data = _flag_config()
    del data["bridging_r_inner"]
    del data["bridging_r_outer"]
    inner = expand_bridging_method(data)["models"][1]
    assert inner["r_inner"] == 0.5
    assert inner["r_outer"] == 0.8


def test_input_is_not_mutated() -> None:
    data = _flag_config()
    ref = copy.deepcopy(data)
    expand_bridging_method(data)
    assert data == ref


def test_spin_stays_top_level() -> None:
    data = _flag_config()
    data["spin"] = {"scheme": "native", "use_spin": [True, False]}
    out = expand_bridging_method(data)
    assert out["spin"] == {"scheme": "native", "use_spin": [True, False]}
    assert "spin" not in out["models"][0]


def test_other_model_keys_stay_on_learned_child() -> None:
    data = _flag_config()
    data["data_stat_protect"] = 1e-3
    data["preset_out_bias"] = {"energy": [None, 1.0]}
    out = expand_bridging_method(data)
    learned = out["models"][0]
    assert learned["data_stat_protect"] == 1e-3
    assert learned["preset_out_bias"] == {"energy": [None, 1.0]}
    assert "data_stat_protect" not in out
    assert "preset_out_bias" not in out


def test_exclusions_move_to_composition_level() -> None:
    data = _flag_config()
    data["pair_exclude_types"] = [[0, 1]]
    data["atom_exclude_types"] = [1]
    out = expand_bridging_method(data)
    assert out["pair_exclude_types"] == [[0, 1]]
    assert out["atom_exclude_types"] == [1]
    learned = out["models"][0]
    assert "pair_exclude_types" not in learned
    assert "atom_exclude_types" not in learned


def test_descriptor_exclude_types_promotion() -> None:
    """Legacy pt semantics: a descriptor-scoped exclusion on a bridged
    model also governs the analytical term.
    """
    data = _flag_config()
    data["descriptor"]["exclude_types"] = [[0, 1]]
    out = expand_bridging_method(data)
    assert out["pair_exclude_types"] == [[0, 1]]


def test_descriptor_exclude_types_mismatch_raises() -> None:
    data = _flag_config()
    data["descriptor"]["exclude_types"] = [[0, 1]]
    data["pair_exclude_types"] = [[0, 0]]
    with pytest.raises(ValueError, match="must match"):
        expand_bridging_method(data)


def test_matching_exclusions_pass() -> None:
    data = _flag_config()
    data["descriptor"]["exclude_types"] = [[0, 1]]
    data["pair_exclude_types"] = [[0, 1]]
    out = expand_bridging_method(data)
    assert out["pair_exclude_types"] == [[0, 1]]


@pytest.mark.parametrize(
    "model_type",
    [
        "linear_ener",  # composition types must spell inner_potential directly
        "frozen",  # unrelated model type
    ],
)
def test_unsupported_model_type_raises(model_type: str) -> None:
    data = _flag_config()
    data["type"] = model_type
    with pytest.raises(ValueError, match="linear_ener"):
        expand_bridging_method(data)


def test_standard_type_is_supported() -> None:
    data = _flag_config()
    data["type"] = "standard"
    out = expand_bridging_method(data)
    assert out["models"][0]["type"] == "standard"
