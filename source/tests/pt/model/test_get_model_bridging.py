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


def test_canonical_rejects_lora_on_child() -> None:
    """The pt trainer reads `lora` from the top level only; a child-level
    `lora` must fail fast instead of silently training without adapters.
    """
    cfg = _canonical_config()
    cfg["models"][0]["lora"] = {"rank": 2}
    with pytest.raises(NotImplementedError, match="lora"):
        get_model(cfg)


def test_canonical_rejects_mismatched_child_type_map() -> None:
    """An explicit child type_map that differs from the composition's must
    not be silently overwritten.
    """
    cfg = _canonical_config()
    cfg["models"][0]["type_map"] = ["O", "Ni"]
    with pytest.raises(NotImplementedError, match="type_map"):
        get_model(cfg)


def test_nested_bridging_flag_on_child_raises() -> None:
    """A `bridging_method` flag on a linear child must not be dropped."""
    cfg = _canonical_config()
    cfg["models"] = [cfg["models"][0]]
    cfg["models"][0]["bridging_method"] = "ZBL"
    with pytest.raises(ValueError, match="sub-model"):
        get_model(cfg)


def test_deep_eval_recognizes_canonical_params() -> None:
    """`_is_sezm_model_params` must route the canonical bridged spelling
    like the flag spelling (both realize a SeZMModel).
    """
    from deepmd.pt.infer.deep_eval import (
        _is_sezm_model_params,
    )

    assert _is_sezm_model_params(_canonical_config())
    assert not _is_sezm_model_params(
        {
            "type": "linear_ener",
            "models": [
                {"descriptor": {"type": "se_atten"}},
                {"descriptor": {"type": "se_atten"}},
            ],
        }
    )


def test_is_sezm_checkpoint_recognizes_canonical_params(tmp_path) -> None:
    """The `.pt2` freeze router must recognize a checkpoint whose persisted
    model params keep the canonical bridged spelling.
    """
    import torch

    from deepmd.pt.entrypoints.freeze_pt2 import (
        is_sezm_checkpoint,
    )

    ckpt = str(tmp_path / "canonical.pt")
    torch.save({"model": {"_extra_state": {"model_params": _canonical_config()}}}, ckpt)
    assert is_sezm_checkpoint(ckpt)
    ckpt2 = str(tmp_path / "multitask.pt")
    torch.save(
        {
            "model": {
                "_extra_state": {
                    "model_params": {"model_dict": {"branch": _canonical_config()}}
                }
            }
        },
        ckpt2,
    )
    assert is_sezm_checkpoint(ckpt2)


def test_sugar_with_top_level_lora_builds() -> None:
    """The concise dpa4+bridging form with trainer-owned top-level `lora`
    must keep building: the expansion routes `lora` to the composition
    level, so the bridge builder never sees it on the learned child.
    """
    from deepmd.pt.train.training import (
        get_model_for_wrapper,
    )

    cfg = _sugar_config()
    cfg["lora"] = {"rank": 2, "alpha": None}
    model = get_model_for_wrapper(copy.deepcopy(cfg))
    assert isinstance(model, SeZMModel)
    # The trainer injects the adapters later by reading the top level of
    # its own (unexpanded) config; expansion must not have mutated it.
    assert cfg["lora"] == {"rank": 2, "alpha": None}


def test_update_sel_normalized_config_skips_inner_potential_child(
    monkeypatch,
) -> None:
    """The default CLI path hands `update_sel` a NORMALIZED config, where
    argcheck always inserts `shared_dict: {}`. Both the update loop and
    the shared-config reconstruction loop must skip the analytical child.
    """
    from deepmd.pt.model.model import (
        LinearEnergyModel,
    )
    from deepmd.pt.model.model.dp_model import (
        DPModelCommon,
    )
    from deepmd.utils.argcheck import (
        model_args,
    )

    seen = []

    def _fake_update_sel(train_data, type_map, sub):
        seen.append(copy.deepcopy(sub))
        return sub, 0.9

    monkeypatch.setattr(DPModelCommon, "update_sel", staticmethod(_fake_update_sel))
    cfg = model_args().normalize_value(_canonical_config(), trim_pattern="_*")
    assert cfg["shared_dict"] == {}  # inserted by normalization
    updated, min_dist = LinearEnergyModel.update_sel(None, cfg["type_map"], cfg)
    assert min_dist == 0.9
    assert len(seen) == 1  # only the learned child
    assert "descriptor" in seen[0]
    assert updated["models"][1]["type"] == "inner_potential"


def test_update_sel_skips_inner_potential_child(monkeypatch) -> None:
    """Neighbor-stat selection must skip the analytical child instead of
    crashing on its missing descriptor.
    """
    from deepmd.pt.model.model import (
        LinearEnergyModel,
    )
    from deepmd.pt.model.model.dp_model import (
        DPModelCommon,
    )

    seen = []

    def _fake_update_sel(train_data, type_map, sub):
        seen.append(copy.deepcopy(sub))
        return sub, 0.9

    monkeypatch.setattr(DPModelCommon, "update_sel", staticmethod(_fake_update_sel))
    cfg = _canonical_config()
    updated, min_dist = LinearEnergyModel.update_sel(None, cfg["type_map"], cfg)
    assert min_dist == 0.9
    assert len(seen) == 1  # only the learned child
    assert "descriptor" in seen[0]
    assert updated["models"][1]["type"] == "inner_potential"


def test_canonical_top_level_option_routes_to_learned_child() -> None:
    """Generic learned-model options at the canonical top level reach the
    learned child (the child is their one owner).
    """
    cfg = _canonical_config()
    cfg["data_stat_protect"] = 0.123
    model = get_model(cfg)
    assert model.atomic_model.data_stat_protect == 0.123


def test_canonical_conflicting_top_level_option_raises() -> None:
    """A learned-owned option set differently at both levels must fail
    loudly instead of one value silently winning.
    """
    cfg = _canonical_config()
    cfg["data_stat_protect"] = 0.123
    cfg["models"][0]["data_stat_protect"] = 0.456
    with pytest.raises(ValueError, match="data_stat_protect"):
        get_model(cfg)


@pytest.mark.parametrize(
    "scheme",
    [
        "native",  # spin as an equivariant descriptor feature
        "deepspin",  # classical virtual-atom representation
    ],
)
def test_canonical_spin_rejects_mismatched_pair_exclusions(scheme: str) -> None:
    """Both spin routes must fail fast on a pair-exclusion mismatch like
    the no-spin route, not silently overwrite the descriptor's exclusions.
    """
    cfg = _canonical_config()
    cfg["pair_exclude_types"] = [[0, 0]]
    cfg["models"][0]["descriptor"]["exclude_types"] = [[0, 1]]
    cfg["spin"] = {
        "scheme": scheme,
        "use_spin": [True, False],
        "virtual_scale": 0.3,
    }
    with pytest.raises(ValueError, match="must match"):
        get_model(cfg)
