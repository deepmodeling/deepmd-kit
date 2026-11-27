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

import pytest
from dargs.dargs import (
    ArgumentKeyError,
)

from deepmd.utils.argcheck import (
    normalize,
)
from deepmd.utils.bridging import (
    DEFAULT_CLAMP,
    DEFAULT_SUPPORT,
    DEFAULT_TRUNCATION,
    drop_retired_descriptor_options,
    drop_retired_window_options,
    expand_bridging_method,
    resolve_bridging_options,
)
from deepmd.utils.compat import (
    update_deepmd_input,
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
        "bridging_truncation": 1.05,
        "bridging_clamp": 0.7,
        "bridging_support": 2,
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
        "truncation": 1.05,
        "clamp": 0.7,
        "support": 2,
    }
    # the flag keys must not leak into the canonical config
    for key in (
        "bridging_method",
        "bridging_truncation",
        "bridging_clamp",
        "bridging_support",
    ):
        assert key not in out
        assert key not in learned


def test_options_default_to_the_shared_constants() -> None:
    data = _flag_config()
    del data["bridging_truncation"]
    del data["bridging_clamp"]
    del data["bridging_support"]
    inner = expand_bridging_method(data)["models"][1]
    assert inner["truncation"] == DEFAULT_TRUNCATION
    assert inner["clamp"] == DEFAULT_CLAMP
    assert inner["support"] == DEFAULT_SUPPORT == 3
    assert resolve_bridging_options({"type": "inner_potential"}) == {
        "truncation": DEFAULT_TRUNCATION,
        "clamp": DEFAULT_CLAMP,
        "support": DEFAULT_SUPPORT,
    }


def test_a_null_clamp_leaves_the_learned_part_unclamped() -> None:
    data = _flag_config()
    data["bridging_clamp"] = None
    inner = expand_bridging_method(data)["models"][1]
    assert resolve_bridging_options(inner)["clamp"] is None


@pytest.mark.parametrize(
    "window",
    [
        {
            "inner_clamp_f_inner": None,
            "inner_clamp_f_outer": None,
            "inner_clamp_scale": "covalent",
        },
        {"inner_clamp_r_inner": None, "inner_clamp_r_outer": None},
    ],
)
def test_a_descriptor_record_drops_the_retired_window(window: dict) -> None:
    config = {"rcut": 4.0, **window}
    drop_retired_descriptor_options(config)
    assert config == {"rcut": 4.0}


@pytest.mark.parametrize("setting", ["inner_clamp_f_inner", "inner_clamp_r_inner"])
def test_active_descriptor_window_requires_retraining(setting: str) -> None:
    config = {setting: 0.0}
    with pytest.raises(ValueError, match="requires retraining"):
        drop_retired_descriptor_options(config)
    assert config == {setting: 0.0}


_RETIRED_MODEL_WINDOW = {
    "bridging_fraction_inner": 0.26,
    "bridging_fraction_outer": 0.8,
    "bridging_r_inner": None,
    "bridging_r_outer": None,
}


def test_an_unbridged_section_drops_the_retired_window() -> None:
    data = {**_flag_config(), "bridging_method": "none", **_RETIRED_MODEL_WINDOW}
    ref = copy.deepcopy(data)
    out = drop_retired_window_options(data)
    assert out == {k: v for k, v in ref.items() if k not in _RETIRED_MODEL_WINDOW}
    assert data == ref


def test_a_bridged_section_keeps_the_retired_window_for_the_argument_check() -> None:
    data = {**_flag_config(), **_RETIRED_MODEL_WINDOW}
    assert drop_retired_window_options(data) is data


def test_every_branch_of_a_multi_task_section_is_filtered_on_its_own() -> None:
    plain = {"type": "dpa4", "bridging_method": "None", **_RETIRED_MODEL_WINDOW}
    bridged = {**_flag_config(), **_RETIRED_MODEL_WINDOW}
    out = drop_retired_window_options({"model_dict": {"a": plain, "b": bridged}})
    assert not set(_RETIRED_MODEL_WINDOW) & set(out["model_dict"]["a"])
    assert out["model_dict"]["b"] is bridged
    assert plain["bridging_fraction_inner"] == 0.26


def test_the_children_of_an_unbridged_composition_are_filtered() -> None:
    child = {"type": "dpa4", "bridging_method": "None", **_RETIRED_MODEL_WINDOW}
    sibling = {"type": "dpa4", "bridging_method": "None"}
    composition = {"type": "linear_ener", "models": [child, sibling]}
    out = drop_retired_window_options(composition)
    assert not set(_RETIRED_MODEL_WINDOW) & set(out["models"][0])
    assert out["models"][1] == sibling
    assert child["bridging_fraction_inner"] == 0.26


def test_the_children_of_a_bridged_composition_are_left_for_the_argument_check() -> (
    None
):
    learned = {"type": "dpa4", "bridging_method": "None", **_RETIRED_MODEL_WINDOW}
    inner = {"type": "inner_potential", "mode": "ZBL"}
    composition = {"type": "linear_ener", "models": [learned, inner]}
    assert drop_retired_window_options(composition) == composition


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


# A bridged model trains on every frame of its data: the contact clamp is the
# identity on all of them, so no frame filter derives from the bridging term.

_EXAMPLES = Path(__file__).parent.parent.parent.parent / "examples" / "water"
_BRIDGED_INPUTS = (
    _EXAMPLES / "dpa4" / "input-zbl.json",
    _EXAMPLES / "dpa4c" / "input-zbl.json",
)


@pytest.mark.parametrize("path", _BRIDGED_INPUTS, ids=lambda p: p.parent.name)
def test_a_shipped_bridged_example_expands_with_the_defaults(path) -> None:
    config = json.loads(path.read_text())
    assert "min_pair_dist" not in config["training"]["training_data"]
    inner = expand_bridging_method(config["model"])["models"][1]
    assert resolve_bridging_options(inner) == {
        "truncation": DEFAULT_TRUNCATION,
        "clamp": DEFAULT_CLAMP,
        "support": DEFAULT_SUPPORT,
    }


def _normalized_model(config: dict) -> dict:
    return normalize(update_deepmd_input(config, warning=False))["model"]


def test_an_unbridged_input_with_the_retired_window_still_normalizes() -> None:
    """The model section stored by an earlier release still holds the window."""
    config = json.loads((_EXAMPLES / "dpa4" / "input.json").read_text())
    config["model"].update(_RETIRED_MODEL_WINDOW)
    assert not set(_RETIRED_MODEL_WINDOW) & set(_normalized_model(config))


def test_a_bridged_input_with_the_retired_window_fails_at_normalization() -> None:
    config = json.loads(_BRIDGED_INPUTS[0].read_text())
    config["model"].update(_RETIRED_MODEL_WINDOW)
    with pytest.raises(ArgumentKeyError, match="bridging_fraction_inner"):
        _normalized_model(config)


def test_min_pair_dist_is_a_distance_in_angstrom() -> None:
    assert min_pair_dist_requirement(0.0) is None
    requirement = min_pair_dist_requirement(0.5)
    assert requirement is not None
    assert requirement.key == "pair_margin"
    assert requirement.default == 0.5
    assert requirement.source_policy == "derived"
