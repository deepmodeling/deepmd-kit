# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for the ``bridging_method`` sugar expansion (issue #5948).

``expand_bridging_method`` is the ONE owner of the sugar: it rewrites a
flag-form config into the canonical ``linear_ener`` composition over the
learned model and an ``inner_potential`` sub-model. These tests pin the
key routing, the legacy exclusion promotion, and the rejections.
"""

import copy
import json
from pathlib import (
    Path,
)

import numpy as np
import pytest

from deepmd.dpmodel.descriptor.dpa4_nn import (
    BridgingSwitch,
)
from deepmd.utils.bridging import (
    DEFAULT_FRACTION_INNER,
    DEFAULT_FRACTION_OUTER,
    expand_bridging_method,
    resolve_bridging_window,
)
from deepmd.utils.data import (
    min_pair_dist_requirement,
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
        "fraction_inner": 0.26,
        "fraction_outer": 0.80,
        "r_inner": 0.8,
        "r_outer": 1.2,
    }
    # the flag keys must not leak into the canonical config
    for key in (
        "bridging_method",
        "bridging_fraction_inner",
        "bridging_fraction_outer",
        "bridging_r_inner",
        "bridging_r_outer",
    ):
        assert key not in out
        assert key not in learned


def test_window_defaults_to_the_covalent_rule() -> None:
    """Without explicit radii the window is a fraction of each pair's own size."""
    data = _flag_config()
    del data["bridging_r_inner"]
    del data["bridging_r_outer"]
    inner = expand_bridging_method(data)["models"][1]
    assert inner["r_inner"] is None
    assert inner["r_outer"] is None
    assert resolve_bridging_window(inner) == {
        "inner_clamp_f_inner": 0.26,
        "inner_clamp_f_outer": 0.80,
        "inner_clamp_scale": "covalent",
    }


def test_explicit_radii_override_the_covalent_rule() -> None:
    """Radii in Å apply the same window to every pair, through a unit scale."""
    inner = expand_bridging_method(_flag_config())["models"][1]
    assert resolve_bridging_window(inner) == {
        "inner_clamp_f_inner": 0.8,
        "inner_clamp_f_outer": 1.2,
        "inner_clamp_scale": "absolute",
    }


def test_half_a_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="both `r_inner` and `r_outer`"):
        resolve_bridging_window({"r_outer": 1.2})


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


def test_lora_stays_top_level() -> None:
    """`lora` is training-owned: the pt trainer reads it from the top
    level of the model section, so the expansion must keep it there and
    never forward it to the learned child (which the pt bridge builder
    rejects).
    """
    data = _flag_config()
    data["lora"] = {"rank": 2, "alpha": None}
    out = expand_bridging_method(data)
    assert out["lora"] == {"rank": 2, "alpha": None}
    assert "lora" not in out["models"][0]


def test_routing_covers_the_argcheck_schema() -> None:
    """Every key the `standard`/`dpa4` argcheck schemas declare must have
    an explicit routing decision in the expansion. Adding a model key to
    argcheck without deciding its routing fails here instead of silently
    landing on the learned child (how top-level `lora` once broke).
    """
    from deepmd.utils.argcheck import (
        model_args,
        sezm_model_args,
        standard_model_args,
    )
    from deepmd.utils.bridging import (
        _COMPOSITION_KEYS,
        _CONSUMED_KEYS,
        _LEARNED_CHILD_KEYS,
        _TRAINER_KEYS,
    )

    schema_keys = {"type"}
    schema_keys |= set(model_args(exclude_hybrid=True).sub_fields)
    schema_keys |= set(standard_model_args().sub_fields)
    schema_keys |= set(sezm_model_args().sub_fields)

    routing = [
        set(_COMPOSITION_KEYS),
        set(_CONSUMED_KEYS),
        set(_TRAINER_KEYS),
        set(_LEARNED_CHILD_KEYS),
    ]
    routed = set().union(*routing)
    assert sum(len(s) for s in routing) == len(routed), (
        "a key is routed to more than one destination"
    )
    assert schema_keys == routed, (
        f"unrouted argcheck keys: {sorted(schema_keys - routed)}; "
        f"routed keys absent from the schema: {sorted(routed - schema_keys)}. "
        "Decide the routing in deepmd.utils.bridging and update the "
        "corresponding tuple."
    )


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


class TestIsBridgedSezmConfig:
    """The canonical-shape predicate used by pt checkpoint consumers."""

    def _canonical(self) -> dict:
        return {
            "type": "linear_ener",
            "weights": "sum",
            "type_map": ["Ni", "O"],
            "models": [
                {"type": "dpa4", "descriptor": {"type": "dpa4"}},
                {"type": "inner_potential", "mode": "zbl"},
            ],
        }

    def test_canonical_shape_is_recognized(self) -> None:
        from deepmd.utils.bridging import (
            is_bridged_sezm_config,
        )

        assert is_bridged_sezm_config(self._canonical())

    def test_descriptor_type_alone_is_recognized(self) -> None:
        from deepmd.utils.bridging import (
            is_bridged_sezm_config,
        )

        cfg = self._canonical()
        cfg["models"][0] = {"type": "standard", "descriptor": {"type": "dpa4"}}
        assert is_bridged_sezm_config(cfg)

    def test_non_linear_type_is_not(self) -> None:
        from deepmd.utils.bridging import (
            is_bridged_sezm_config,
        )

        cfg = self._canonical()
        cfg["type"] = "dpa4"
        assert not is_bridged_sezm_config(cfg)

    def test_linear_without_inner_child_is_not(self) -> None:
        from deepmd.utils.bridging import (
            is_bridged_sezm_config,
        )

        cfg = self._canonical()
        cfg["models"] = [cfg["models"][0]]
        assert not is_bridged_sezm_config(cfg)

    def test_non_dpa4_learned_child_is_not(self) -> None:
        from deepmd.utils.bridging import (
            is_bridged_sezm_config,
        )

        cfg = self._canonical()
        cfg["models"][0] = {"type": "standard", "descriptor": {"type": "se_e2_a"}}
        assert not is_bridged_sezm_config(cfg)


def test_route_canonical_learned_options_copies_and_conflicts() -> None:
    """The learned child owns the generic model options: a top-level value
    is copied onto the child when absent, accepted when equal, and
    rejected when the two levels differ.
    """
    from deepmd.utils.bridging import (
        route_canonical_learned_options,
    )

    composition = {
        "type": "linear_ener",  # composition-owned: never routed
        "type_map": ["Ni", "O"],  # composition-owned: never routed
        "data_stat_protect": 0.123,  # learned-owned: routed
        "preset_out_bias": {"energy": [1.0, 2.0]},  # learned-owned: routed
    }
    learned = {"descriptor": {"type": "dpa4"}}
    route_canonical_learned_options(composition, learned)
    assert learned["data_stat_protect"] == 0.123
    assert learned["preset_out_bias"] == {"energy": [1.0, 2.0]}
    assert learned["preset_out_bias"] is not composition["preset_out_bias"]
    assert "type_map" not in learned

    # equal values at both levels pass
    route_canonical_learned_options(composition, learned)

    learned["data_stat_protect"] = 0.456
    with pytest.raises(ValueError, match="data_stat_protect"):
        route_canonical_learned_options(composition, learned)


def test_routing_helper_handles_every_learned_key_uniformly() -> None:
    """Walk the WHOLE learned-key table through the canonical-route helper:
    each key copies down when the child lacks it and conflicts when the two
    levels differ. Pins uniformity, so a future per-key special case in the
    helper cannot land untested.
    """
    from deepmd.utils.bridging import (
        _LEARNED_CHILD_KEYS,
        route_canonical_learned_options,
    )

    for key in _LEARNED_CHILD_KEYS:
        learned = {}
        route_canonical_learned_options({key: "sentinel-a"}, learned)
        assert learned[key] == "sentinel-a", key
        with pytest.raises(ValueError, match=key):
            route_canonical_learned_options({key: "sentinel-a"}, {key: "sentinel-b"})


def test_routing_resolves_argcheck_default_conflicts() -> None:
    """Strict normalization injects schema defaults on BOTH levels, erasing
    the set-by-user provenance; the helper recovers it by comparing against
    the argcheck default, so only two explicit non-default values conflict.
    """
    from deepmd.utils.bridging import (
        route_canonical_learned_options,
    )

    # child holds the injected default -> the explicit top-level value wins
    learned = {"data_stat_protect": 0.01}
    route_canonical_learned_options({"data_stat_protect": 0.123}, learned)
    assert learned["data_stat_protect"] == 0.123
    # the top level holds the injected default -> the explicit child wins
    learned = {"data_stat_protect": 0.123}
    route_canonical_learned_options({"data_stat_protect": 0.01}, learned)
    assert learned["data_stat_protect"] == 0.123
    # two explicit non-default values are a REAL conflict
    with pytest.raises(ValueError, match="data_stat_protect"):
        route_canonical_learned_options(
            {"data_stat_protect": 0.2}, {"data_stat_protect": 0.3}
        )


# The frame filter of a bridged model is the midpoint of its window, so the two
# decisions it rests on -- which models count as bridged, and which radius the
# window yields -- are pinned here against the shipped examples.

_EXAMPLES = Path(__file__).parent.parent.parent.parent / "examples" / "water"
_BRIDGED_INPUTS = (
    _EXAMPLES / "dpa4" / "input-zbl.json",
    _EXAMPLES / "dpa4c" / "input-zbl.json",
)


@pytest.mark.parametrize("path", _BRIDGED_INPUTS, ids=lambda p: p.parent.name)
def test_a_shipped_bridged_example_derives_its_own_filter(path) -> None:
    """Every learned family the composition accepts owns a frame filter."""
    config = json.loads(path.read_text())
    training_data = config["training"]["training_data"]
    assert "min_pair_dist" not in training_data, (
        "a bridged model derives the filter from its window, so the input must "
        "not carry a second copy of it"
    )
    requirement = min_pair_dist_requirement(config["model"], 0.0)
    assert requirement is not None
    assert requirement.key == "pair_margin"
    assert requirement.default == pytest.approx(
        0.5 * (DEFAULT_FRACTION_INNER + DEFAULT_FRACTION_OUTER)
    )
    assert requirement.length_scale == "covalent"


@pytest.mark.parametrize("path", _BRIDGED_INPUTS, ids=lambda p: p.parent.name)
def test_a_shipped_bridged_example_refuses_an_explicit_filter(path) -> None:
    config = json.loads(path.read_text())
    with pytest.raises(ValueError, match="must not be set on a bridged model"):
        min_pair_dist_requirement(config["model"], 0.5)


def test_an_explicit_window_filters_at_its_midpoint() -> None:
    """Radii given in Å yield a filter in Å, halfway between the two."""
    requirement = min_pair_dist_requirement(_flag_config(), 0.0)
    assert requirement is not None
    assert requirement.default == pytest.approx(1.0)
    assert requirement.length_scale == "absolute"


@pytest.mark.parametrize(
    "fractions",
    [(DEFAULT_FRACTION_INNER, DEFAULT_FRACTION_OUTER), (0.1, 0.9), (0.5, 0.8)],
)
def test_the_bridging_switch_is_half_open_at_the_filter_radius(fractions) -> None:
    """The filter keeps exactly the pairs whose learned amplitude is at least one half."""
    f_inner, f_outer = fractions
    config = _flag_config()
    del config["bridging_r_inner"], config["bridging_r_outer"]
    config["bridging_fraction_inner"] = f_inner
    config["bridging_fraction_outer"] = f_outer
    requirement = min_pair_dist_requirement(config, 0.0)
    assert requirement is not None
    contact = np.array([1.7])
    opening = BridgingSwitch(f_inner, f_outer).call(
        requirement.default * contact, contact
    )
    np.testing.assert_allclose(opening, 0.5, rtol=0.0, atol=1.0e-14)


def test_an_unbridged_model_without_a_floor_registers_no_filter() -> None:
    """Whether the filter runs and whether it is registered is one decision."""
    config = _flag_config()
    for key in ("bridging_method", "bridging_r_inner", "bridging_r_outer"):
        del config[key]
    assert min_pair_dist_requirement(config, 0.0) is None
    requirement = min_pair_dist_requirement(config, 0.5)
    assert requirement is not None
    assert requirement.default == 0.5
    assert requirement.length_scale == "absolute"
