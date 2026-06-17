# SPDX-License-Identifier: LGPL-3.0-or-later
import pytest

from dpa_adapt.config.manager import (
    MFTConfigManager,
)
from dpa_adapt.mft import (
    MFTFineTuner,
)


class FakeTuner:
    pretrained = "/share/DPA-3.1-3M.pt"
    aux_branch = "MP_traj_v024_alldata_mixu"
    aux_prob = 0.5
    type_map = ["Cu", "O"]
    fitting_net_params = {"type": "ener", "neuron": [240, 240, 240]}
    downstream_task_type = "ener"
    learning_rate = 1e-3
    stop_lr = 1e-5
    max_steps = 1000
    batch_size = "auto:32"
    seed = 42
    output_dir = "/tmp/mft_test"
    save_freq = 500
    disp_freq = 100
    train_data = "/data/downstream"
    aux_data = "/data/aux"
    valid_data = None


def test_build_has_model_dict():
    config = MFTConfigManager(FakeTuner()).build()
    assert "model_dict" in config["model"]
    assert "shared_dict" in config["model"]


def test_aux_branch_key_present():
    config = MFTConfigManager(FakeTuner()).build()
    assert "MP_traj_v024_alldata_mixu" in config["model"]["model_dict"]
    assert "DOWNSTREAM" in config["model"]["model_dict"]


def test_finetune_head_correct():
    config = MFTConfigManager(FakeTuner()).build()
    downstream = config["model"]["model_dict"]["DOWNSTREAM"]
    assert downstream["finetune_head"] == "MP_traj_v024_alldata_mixu"


def test_model_prob_values():
    config = MFTConfigManager(FakeTuner()).build()
    prob = config["training"]["model_prob"]
    assert prob["MP_traj_v024_alldata_mixu"] == 0.5
    assert prob["DOWNSTREAM"] == 1.0


def test_data_dict_paths():
    config = MFTConfigManager(FakeTuner()).build()
    dd = config["training"]["data_dict"]
    assert dd["MP_traj_v024_alldata_mixu"]["training_data"]["systems"] == ["/data/aux"]
    assert dd["DOWNSTREAM"]["training_data"]["systems"] == ["/data/downstream"]


def test_aux_fitting_net_is_ener():
    config = MFTConfigManager(FakeTuner()).build()
    fn = config["model"]["model_dict"]["MP_traj_v024_alldata_mixu"]["fitting_net"]
    assert fn["type"] == "ener"


def test_build_cmd_flags():
    cm = MFTConfigManager(FakeTuner())
    cmd = cm.build_cmd("input.json")
    assert "--use-pretrain-script" not in cmd
    assert "--model-branch" not in cmd
    assert "--finetune /share/DPA-3.1-3M.pt" in cmd
    assert "--skip-neighbor-stat" in cmd


def test_descriptor_has_repflow_params():
    config = MFTConfigManager(FakeTuner()).build()
    desc = config["model"]["shared_dict"]["dpa3_descriptor"]
    assert desc["type"] == "dpa3"
    assert "repflow" in desc
    rf = desc["repflow"]
    assert rf["n_dim"] == 128
    assert rf["e_dim"] == 64
    assert rf["a_dim"] == 32
    assert rf["nlayers"] == 16
    assert rf["e_rcut"] == 6.0
    assert rf["a_rcut"] == 4.0
    assert desc["activation_function"] == "custom_silu:3.0"
    assert desc["precision"] == "float32"


def test_systems_accepts_list():
    t = FakeTuner()
    t.train_data = ["/data/d1", "/data/d2"]
    t.aux_data = ["/data/a1", "/data/a2", "/data/a3"]
    config = MFTConfigManager(t).build()
    dd = config["training"]["data_dict"]
    assert dd["DOWNSTREAM"]["training_data"]["systems"] == ["/data/d1", "/data/d2"]
    assert dd["MP_traj_v024_alldata_mixu"]["training_data"]["systems"] == [
        "/data/a1",
        "/data/a2",
        "/data/a3",
    ]


def test_type_map_in_shared_dict():
    config = MFTConfigManager(FakeTuner()).build()
    shared = config["model"]["shared_dict"]
    assert "type_map" in shared
    assert isinstance(shared["type_map"], list)
    assert shared["type_map"] == ["Cu", "O"]


def test_branch_type_map_is_string():
    config = MFTConfigManager(FakeTuner()).build()
    md = config["model"]["model_dict"]
    assert md["MP_traj_v024_alldata_mixu"]["type_map"] == "type_map"
    assert md["DOWNSTREAM"]["type_map"] == "type_map"


def test_data_dict_has_training_data():
    config = MFTConfigManager(FakeTuner()).build()
    dd = config["training"]["data_dict"]
    assert "training_data" in dd["MP_traj_v024_alldata_mixu"]
    assert "training_data" in dd["DOWNSTREAM"]


def test_no_validation_data_in_training():
    config = MFTConfigManager(FakeTuner()).build()
    assert "validation_data" not in config["training"]


def test_fitting_net_params_used():
    config = MFTConfigManager(FakeTuner()).build()
    md = config["model"]["model_dict"]
    assert md["MP_traj_v024_alldata_mixu"]["fitting_net"] == {
        "type": "ener",
        "neuron": [240, 240, 240],
    }
    assert md["DOWNSTREAM"]["fitting_net"] == {
        "type": "ener",
        "neuron": [240, 240, 240],
    }


def test_fitting_net_default_when_none():
    t = FakeTuner()
    t.fitting_net_params = None
    config = MFTConfigManager(t).build()
    md = config["model"]["model_dict"]
    assert md["MP_traj_v024_alldata_mixu"]["fitting_net"] == {"type": "ener"}
    assert md["DOWNSTREAM"]["fitting_net"] == {"type": "ener"}


# --- MFTFineTuner.__init__ auto-reading fitting_net from checkpoint ----------


def _fake_sd(branches):
    """Build a minimal state_dict mirroring the real checkpoint layout."""
    return {
        "model": {
            "_extra_state": {
                "model_params": {
                    "model_dict": {
                        name: {"fitting_net": fn} for name, fn in branches.items()
                    }
                }
            }
        }
    }


def test_explicit_fitting_net_params_skips_ckpt_load(monkeypatch):
    """Backward compat: when user supplies fitting_net_params, the
    checkpoint is not touched and the user's value is kept verbatim.
    """
    import torch

    def _explode(*args, **kwargs):
        raise AssertionError(
            "torch.load must not be called when fitting_net_params is provided"
        )

    monkeypatch.setattr(torch, "load", _explode)

    custom = {"type": "ener", "neuron": [123, 456], "resnet_dt": True}
    t = MFTFineTuner(
        pretrained="/does/not/exist.pt",
        aux_branch="Domains_Alloy",
        fitting_net_params=custom,
    )
    assert t.fitting_net_params == custom


def test_fitting_net_params_auto_read_from_ckpt(monkeypatch):
    """When fitting_net_params is omitted, MFTFineTuner pulls it out of the
    checkpoint at the documented nested path.
    """
    import torch

    expected = {"type": "ener", "neuron": [240, 240, 240], "resnet_dt": True}
    fake = _fake_sd(
        {
            "Domains_Alloy": expected,
            "MP_traj_v024_alldata_mixu": {"type": "ener", "neuron": [120, 120]},
        }
    )
    monkeypatch.setattr(torch, "load", lambda *a, **kw: fake)

    t = MFTFineTuner(
        pretrained="/does/not/exist.pt",
        aux_branch="Domains_Alloy",
    )
    assert t.fitting_net_params == expected


class TestAutoTypeMap:
    """When type_map is not provided, MFTFineTuner auto-detects it from the
    checkpoint and validates data type_maps.
    """

    def _fake_ckpt_sd(self, type_map=None):
        """Minimal DPA-3.1-3M-like state_dict with a shared type_map."""
        if type_map is None:
            type_map = ["H", "He", "Li", "Be", "B", "C", "N", "O"]
        return {
            "model": {
                "_extra_state": {
                    "model_params": {
                        "shared_dict": {
                            "dpa3_descriptor": {"type": "dpa3"},
                            "type_map": type_map,
                        },
                        "model_dict": {
                            "Domains_Alloy": {
                                "fitting_net": {"type": "ener"},
                            },
                        },
                    }
                }
            }
        }

    def test_validate_and_resolve_sets_type_map(self, monkeypatch, tmp_path):
        """_validate_and_resolve_type_map reads checkpoint type_map."""
        import torch

        monkeypatch.setattr(
            torch,
            "load",
            lambda *a, **kw: self._fake_ckpt_sd(),
        )

        t = MFTFineTuner(
            pretrained="/fake.pt",
            aux_branch="Domains_Alloy",
        )
        assert t.type_map is None

        t._validate_and_resolve_type_map(str(tmp_path), str(tmp_path))
        assert t.type_map == ["H", "He", "Li", "Be", "B", "C", "N", "O"]

    def test_config_has_nonempty_type_map(self, monkeypatch):
        """Generated mft_input.json must have a non-empty global type_map
        when the user does not pass one explicitly.
        """
        import torch

        monkeypatch.setattr(
            torch,
            "load",
            lambda *a, **kw: self._fake_ckpt_sd(),
        )

        t = MFTFineTuner(
            pretrained="/fake.pt",
            aux_branch="Domains_Alloy",
        )
        t.train_data = "/data/downstream"
        t.aux_data = "/data/aux"
        t._validate_and_resolve_type_map(t.train_data, t.aux_data)

        config = MFTConfigManager(t).build()
        shared = config["model"]["shared_dict"]
        assert "type_map" in shared
        assert isinstance(shared["type_map"], list)
        assert len(shared["type_map"]) == 8
        assert shared["type_map"][0] == "H"
        # Must NOT be empty — empty [] causes CUDA gather out-of-bounds
        assert shared["type_map"] != []

    def test_explicit_type_map_still_respected(self, monkeypatch):
        """When user passes type_map explicitly, it is used verbatim."""
        import torch

        monkeypatch.setattr(
            torch,
            "load",
            lambda *a, **kw: self._fake_ckpt_sd(),
        )

        t = MFTFineTuner(
            pretrained="/fake.pt",
            aux_branch="Domains_Alloy",
            type_map=["Cu", "O"],
        )
        t.train_data = "/data/downstream"
        t.aux_data = "/data/aux"

        config = MFTConfigManager(t).build()
        shared = config["model"]["shared_dict"]
        assert shared["type_map"] == ["Cu", "O"]

    def test_data_type_map_validated_against_checkpoint(self, monkeypatch, tmp_path):
        """If data type_map.raw contains elements not in the checkpoint,
        _validate_and_resolve_type_map raises ValueError.
        """
        import numpy as np
        import torch

        monkeypatch.setattr(
            torch,
            "load",
            lambda *a, **kw: self._fake_ckpt_sd(),
        )

        t = MFTFineTuner(
            pretrained="/fake.pt",
            aux_branch="Domains_Alloy",
        )

        # Create a system with an unsupported element
        sysdir = tmp_path / "sys"
        sysdir.mkdir()
        (sysdir / "type.raw").write_text("0\n1\n")
        (sysdir / "type_map.raw").write_text("Pu\nU\n")
        sd = sysdir / "set.000"
        sd.mkdir()
        np.save(sd / "coord.npy", np.zeros((1, 6)))
        np.save(sd / "box.npy", np.eye(3).reshape(1, 9))

        with pytest.raises(ValueError, match="Pu"):
            t._validate_and_resolve_type_map(str(sysdir), str(tmp_path))


def test_unknown_aux_branch_raises_with_branch_list(monkeypatch):
    """If aux_branch is not in the checkpoint, the error names the bad
    branch and lists what IS available.  With lazy loading the error is
    raised on first access to ``fitting_net_params``, not at construction.
    """
    import torch

    fake = _fake_sd(
        {
            "Domains_Alloy": {"type": "ener"},
            "MP_traj_v024_alldata_mixu": {"type": "ener"},
            "Omat24": {"type": "ener"},
        }
    )
    monkeypatch.setattr(torch, "load", lambda *a, **kw: fake)

    t = MFTFineTuner(
        pretrained="/does/not/exist.pt",
        aux_branch="NotARealBranch",
    )
    with pytest.raises(ValueError) as exc_info:
        _ = t.fitting_net_params  # triggers lazy load
    msg = str(exc_info.value)
    assert "NotARealBranch" in msg
    assert "Domains_Alloy" in msg
    assert "MP_traj_v024_alldata_mixu" in msg
    assert "Omat24" in msg
