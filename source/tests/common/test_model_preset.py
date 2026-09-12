# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for named model presets and their expansion."""

import copy
import logging

import pytest
from dargs.dargs import (
    ArgumentKeyError,
)

from deepmd.dpmodel.utils.multi_task import (
    preprocess_shared_params,
)
from deepmd.utils.argcheck import (
    normalize,
)
from deepmd.utils.compat import (
    update_deepmd_input,
)
from deepmd.utils.model_preset import (
    MODEL_PRESETS,
    PERIODIC_TABLE,
    expand_model_preset,
    get_model_preset,
)

TRAINING = {"training_data": {"systems": ["fake"]}, "numb_steps": 10}


def _normalize_model(model: dict) -> dict:
    config = {"model": copy.deepcopy(model), "training": copy.deepcopy(TRAINING)}
    return normalize(config)["model"]


@pytest.mark.parametrize("name", sorted(MODEL_PRESETS))
def test_every_preset_expands_to_a_valid_model(name: str) -> None:
    model = expand_model_preset({"preset": name})
    assert "preset" not in model
    assert model["type_map"] == list(PERIODIC_TABLE)
    assert len(model["type_map"]) == 118
    normalized = _normalize_model(model)
    if name.startswith("dpa4c-"):
        assert normalized["type"] == "standard"
        assert normalized["descriptor"]["type"] == "dpa4c"
        assert normalized["fitting_net"]["type"] == "ener"
    else:
        assert normalized["type"] == "dpa4"
        assert normalized["descriptor"]["type"] == "dpa4"
        assert normalized["fitting_net"]["type"] == "dpa4_ener"


def test_explicit_entries_override_and_supplement(caplog) -> None:
    model = {
        "preset": "dpa4-nano-v20260911",
        "type_map": ["O", "H"],
        "descriptor": {"rcut": 5.0, "use_amp": True, "seed": 1},
        "fitting_net": {"seed": 1},
        "use_compile": False,
    }
    original = copy.deepcopy(model)
    with caplog.at_level(logging.INFO, logger="deepmd.utils.model_preset"):
        expanded = expand_model_preset(model)
    assert model == original
    preset = get_model_preset("dpa4-nano-v20260911")

    assert "preset" not in expanded
    assert expanded["type"] == "dpa4"
    assert expanded["type_map"] == ["O", "H"]
    assert expanded["descriptor"]["rcut"] == 5.0
    assert expanded["descriptor"]["use_amp"] is True
    assert expanded["descriptor"]["seed"] == 1
    for key, value in preset["descriptor"].items():
        if key != "rcut":
            assert expanded["descriptor"][key] == value
    assert expanded["fitting_net"] == {**preset["fitting_net"], "seed": 1}
    assert expanded["use_compile"] is False
    _normalize_model(expanded)

    # Only entries that change a preset value are reported as overrides.
    assert "type_map" in caplog.text
    assert "descriptor.rcut" in caplog.text
    assert "use_amp" not in caplog.text


def test_expansion_is_idempotent_and_a_noop_without_preset() -> None:
    plain = {
        "type_map": ["O"],
        "descriptor": {"type": "se_e2_a", "sel": [10]},
        "fitting_net": {"neuron": [4]},
    }
    assert expand_model_preset(plain) is plain
    multi = {"shared_dict": {}, "model_dict": {"a": copy.deepcopy(plain)}}
    assert expand_model_preset(multi) is multi
    expanded = expand_model_preset({"preset": "dpa4c-neo-v20260911"})
    assert expand_model_preset(expanded) is expanded


def test_preset_name_is_case_insensitive() -> None:
    assert expand_model_preset({"preset": "DPA4-Neo-v20260911"}) == expand_model_preset(
        {"preset": "dpa4-neo-v20260911"}
    )


def test_unknown_or_malformed_preset_raises() -> None:
    with pytest.raises(ValueError, match="Unknown model preset"):
        expand_model_preset({"preset": "dpa4-huge-v20260911"})
    with pytest.raises(ValueError, match="must be a string"):
        expand_model_preset({"preset": 3})


def test_multi_task_branches_expand_with_shared_references() -> None:
    model = {
        "preset": "dpa4-mini-v20260911",
        "shared_dict": {
            "type_map": ["O", "H"],
            "descriptor": {"type": "dpa4", "rcut": 6.0},
        },
        "model_dict": {
            "water_1": {
                "type_map": "type_map",
                "descriptor": "descriptor",
                "fitting_net": {"seed": 2},
            },
            "water_2": {
                "preset": "dpa4-neo-v20260911",
                "type_map": "type_map",
            },
        },
    }
    expanded = expand_model_preset(model)
    assert "preset" not in expanded
    # The shared descriptor is the mini descriptor with the explicit keys on top.
    mini = get_model_preset("dpa4-mini-v20260911")
    assert expanded["shared_dict"]["descriptor"] == {**mini["descriptor"], "rcut": 6.0}
    assert expanded["shared_dict"]["type_map"] == ["O", "H"]

    water_1 = expanded["model_dict"]["water_1"]
    assert "preset" not in water_1
    assert water_1["type"] == "dpa4"
    # Shared-dict references are kept for the multi-task preprocessing.
    assert water_1["type_map"] == "type_map"
    assert water_1["descriptor"] == "descriptor"
    assert water_1["fitting_net"] == {**mini["fitting_net"], "seed": 2}

    water_2 = expanded["model_dict"]["water_2"]
    assert "preset" not in water_2
    assert water_2["type_map"] == "type_map"
    assert water_2["descriptor"]["lmax"] == 3
    assert water_2["descriptor"]["n_focus"] == 2


def test_preset_catalog() -> None:
    catalog = {
        ("dpa4", "v20260820"): ("nano", "mini", "neo", "air", "plus", "pro"),
        ("dpa4", "v20260911"): (
            "nano",
            "mini",
            "neo",
            "air",
            "plus",
            "pro",
            "max",
            "ultra",
        ),
        ("dpa4c", "v20260901"): ("nano", "mini", "neo", "air", "plus"),
        ("dpa4c", "v20260911"): ("nano", "mini", "neo", "air", "plus"),
    }
    assert set(MODEL_PRESETS) == {
        f"{family}-{grade}-{version}"
        for (family, version), grades in catalog.items()
        for grade in grades
    }


@pytest.mark.parametrize(
    ("family", "version", "options"),
    [
        (
            "dpa4",
            "v20260820",
            {"edge_norm": True, "sandwich_norm": [False, True, True, False]},
        ),
        (
            "dpa4",
            "v20260911",
            {
                "edge_norm": [False, True, True],
                "sandwich_norm": [True, False, True, False],
                "env_exp": 5,
                "basis_type": "gaussian/fix",
            },
        ),
        ("dpa4c", "v20260901", {"basis_type": "bessel"}),
        ("dpa4c", "v20260911", {"basis_type": "gaussian/fix"}),
    ],
)
def test_version_descriptor_options(family: str, version: str, options: dict) -> None:
    for name in MODEL_PRESETS:
        if name.startswith(f"{family}-") and name.endswith(f"-{version}"):
            descriptor = _normalize_model(get_model_preset(name))["descriptor"]
            assert {key: descriptor[key] for key in options} == options


def test_presets_carry_no_runtime_options() -> None:
    for name, preset in MODEL_PRESETS.items():
        for region in ("descriptor", "fitting_net"):
            for key in ("use_amp", "seed", "sel", "trainable"):
                assert key not in preset[region], (name, region, key)
        for key in ("add_chg_spin_ebd", "default_chg_spin", "so2_layers"):
            assert key not in preset["descriptor"], (name, key)


def test_periodic_table_matches_econf_type_map() -> None:
    from deepmd.utils.econf_embd import (
        type_map,
    )

    assert list(PERIODIC_TABLE) == type_map


def test_leftover_preset_fails_argument_check() -> None:
    config = {
        "model": {"preset": "dpa4-nano-v20260911"},
        "training": copy.deepcopy(TRAINING),
    }
    with pytest.raises(ArgumentKeyError, match="preset"):
        normalize(config)


def test_multi_task_preset_passes_shared_param_preprocessing() -> None:
    """The expansion runs before multi-task preprocessing and argument check."""
    model = {
        "preset": "dpa4-nano-v20260911",
        "shared_dict": {
            "type_map": ["O", "H"],
            "descriptor": {"type": "dpa4", "rcut": 6.0},
        },
        "model_dict": {
            "water_1": {
                "type_map": "type_map",
                "descriptor": "descriptor",
                "fitting_net": {"seed": 1},
            },
            "water_2": {
                "type_map": "type_map",
                "descriptor": "descriptor",
                "fitting_net": {"seed": 2},
            },
        },
    }
    processed, shared_links = preprocess_shared_params(
        expand_model_preset(model), lambda item_key, params: dict
    )
    assert set(shared_links) == {"descriptor"}
    nano = get_model_preset("dpa4-nano-v20260911")
    for branch in processed["model_dict"].values():
        assert branch["type"] == "dpa4"
        assert branch["type_map"] == ["O", "H"]
        assert branch["descriptor"] == nano["descriptor"]
        assert branch["fitting_net"]["neuron"] == [0]
    config = {
        "model": processed,
        "loss_dict": {"water_1": {"type": "ener"}, "water_2": {"type": "ener"}},
        "training": {
            "model_prob": {"water_1": 0.5, "water_2": 0.5},
            "data_dict": {
                "water_1": {"training_data": {"systems": ["fake"]}},
                "water_2": {"training_data": {"systems": ["fake"]}},
            },
            "numb_steps": 10,
        },
    }
    normalized = normalize(config, multi_task=True)
    assert set(normalized["model"]["model_dict"]) == {"water_1", "water_2"}


def test_multi_task_top_level_regions_are_branch_defaults() -> None:
    model = {
        "preset": "dpa4-nano-v20260911",
        "descriptor": {"rcut": 7.0},
        "fitting_net": {"seed": 3},
        "shared_dict": {"type_map": ["O", "H"]},
        "model_dict": {
            "water_1": {"type_map": "type_map"},
            "water_2": {
                "preset": "dpa4-mini-v20260911",
                "type_map": "type_map",
                "descriptor": {"rcut": 5.0},
            },
        },
    }
    expanded = expand_model_preset(model)
    water_1 = expanded["model_dict"]["water_1"]
    assert water_1["descriptor"]["rcut"] == 7.0
    assert water_1["descriptor"]["lmax"] == 1
    nano = get_model_preset("dpa4-nano-v20260911")
    assert water_1["fitting_net"] == {**nano["fitting_net"], "seed": 3}
    # A branch entry replaces the top-level default as a whole.
    water_2 = expanded["model_dict"]["water_2"]
    assert water_2["descriptor"]["rcut"] == 5.0
    assert water_2["descriptor"]["lmax"] == 2
    assert water_2["fitting_net"]["seed"] == 3
    # The top-level region is consumed once distributed to every branch, so a
    # downstream backend without its own model-wide cascade (dpmodel-based
    # entrypoints) does not see it as an unknown top-level key.
    assert "descriptor" not in expanded
    assert "preset" not in expanded
    processed, _ = preprocess_shared_params(
        expanded, lambda item_key, params: dict, cascade_defaults=False
    )
    assert "descriptor" not in processed


def test_shared_dict_entries_take_the_top_level_preset_as_base() -> None:
    model = {
        "preset": "dpa4-nano-v20260911",
        "shared_dict": {
            "type_map": ["O", "H"],
            "descriptor": {"use_amp": True, "seed": 42},
            "shared_fit": {"dim_case_embd": 2, "seed": 42},
            "unreferenced": {"rcut": 5.0},
        },
        "model_dict": {
            "water_1": {
                "type_map": "type_map",
                "descriptor": "descriptor:1",
                "fitting_net": "shared_fit",
            },
            "water_2": {
                "preset": "dpa4-mini-v20260911",
                "type_map": "type_map",
                "descriptor": "descriptor",
            },
        },
    }
    expanded = expand_model_preset(model)
    nano = get_model_preset("dpa4-nano-v20260911")
    shared = expanded["shared_dict"]
    # Referenced entries: preset region plus the run-specific keys; a shared
    # level suffix in the reference does not change the entry it names.
    assert shared["descriptor"] == {**nano["descriptor"], "use_amp": True, "seed": 42}
    assert shared["shared_fit"] == {
        **nano["fitting_net"],
        "dim_case_embd": 2,
        "seed": 42,
    }
    # Entries nobody references and the type map are left as written.
    assert shared["unreferenced"] == {"rcut": 5.0}
    assert shared["type_map"] == ["O", "H"]
    # A branch-level preset shapes only that branch's own regions.
    water_2 = expanded["model_dict"]["water_2"]
    assert water_2["descriptor"] == "descriptor"
    assert water_2["fitting_net"]["neuron"] == [0]
    # Without a top-level preset the shared dictionary is untouched.
    branch_only = {
        "shared_dict": {"type_map": ["O"], "descriptor": {"rcut": 5.0}},
        "model_dict": {
            "a": {
                "preset": "dpa4-nano-v20260911",
                "type_map": "type_map",
                "descriptor": "descriptor",
            }
        },
    }
    assert expand_model_preset(branch_only)["shared_dict"] == branch_only["shared_dict"]


def test_explicit_alias_replaces_the_preset_canonical_key() -> None:
    """A legacy key alias for a preset-defined key overrides it in place,
    instead of sitting next to it and failing the strict argument check.
    """
    model = expand_model_preset(
        {
            "preset": "dpa4-nano-v20260911",
            "type_map": ["O", "H"],
            "descriptor": {"so2_layers": 5},
        }
    )
    assert model["descriptor"]["mixing_layers"] == 5
    assert "so2_layers" not in model["descriptor"]
    _normalize_model(model)


def test_explicit_dict_for_a_whole_value_region_does_not_crash() -> None:
    """`type_map` (and `type`) are always replaced as a whole; a malformed
    dict there must not raise before the argument check reports it.
    """
    expanded = expand_model_preset(
        {"preset": "dpa4-nano-v20260911", "type_map": {"O": 0, "H": 1}}
    )
    assert expanded["type_map"] == {"O": 0, "H": 1}
    with pytest.raises(Exception):
        _normalize_model(expanded)


def test_shared_dict_role_is_recognised_through_a_branch_default() -> None:
    """A `descriptor`/`fitting_net` reference inherited by a branch only
    through the top-level default must still be recognised as referenced.
    """
    nano = get_model_preset("dpa4-nano-v20260911")
    model = {
        "preset": "dpa4-nano-v20260911",
        "descriptor": "desc",
        "shared_dict": {"type_map": ["O", "H"], "desc": {"seed": 42}},
        "model_dict": {"water_1": {"type_map": "type_map"}},
    }
    expanded = expand_model_preset(model)
    assert expanded["shared_dict"]["desc"] == {**nano["descriptor"], "seed": 42}
    assert expanded["model_dict"]["water_1"]["descriptor"] == "desc"


def test_malformed_multi_task_layout_is_left_to_argcheck() -> None:
    malformed = {"preset": "dpa4-nano-v20260911", "model_dict": "water"}
    assert expand_model_preset(malformed) is malformed
    branch_not_mapping = {
        "preset": "dpa4-nano-v20260911",
        "model_dict": {"water": "not a mapping"},
    }
    assert expand_model_preset(branch_not_mapping)["model_dict"] == {
        "water": "not a mapping"
    }


def test_update_deepmd_input_expands_presets() -> None:
    jdata = {
        "model": {"preset": "dpa4c-mini-v20260911"},
        "training": copy.deepcopy(TRAINING),
    }
    out = update_deepmd_input(jdata, warning=False)
    assert "preset" not in out["model"]
    assert out["model"]["descriptor"]["channels"] == 32
    assert out["model"]["fitting_net"]["neuron"] == [192, 192, 192]
