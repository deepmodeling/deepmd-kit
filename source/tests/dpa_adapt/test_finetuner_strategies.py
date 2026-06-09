# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for DPAFineTuner training-paradigm strategies
(linear_probe / finetune).

Mock ``dp --pt train`` via ``subprocess.run``; verify:
- Correct DPATrainer params per strategy
- Auto type_map inference (non-empty, checkpoint-derived)
- Config structure (input.json)
"""

from __future__ import (
    annotations,
)

import json
import os
from pathlib import (
    Path,
)
from unittest.mock import (
    patch,
)

import pytest

from deepmd.dpa_adapt.finetuner import (
    DPAFineTuner,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FULL_TYPE_MAP = ["H", "He", "Li", "Be", "B", "C", "N", "O"]  # 8 el. subset


def _fake_ckpt_sd(type_map=None):
    """Minimal DPA-3.1-3M-like state_dict."""
    if type_map is None:
        type_map = list(_FULL_TYPE_MAP)
    descriptor = {
        "type": "dpa3",
        "repflow": {
            "n_dim": 128,
            "e_dim": 64,
            "a_dim": 32,
            "nlayers": 16,
            "e_rcut": 6.0,
            "e_rcut_smth": 5.3,
            "e_sel": 1200,
            "a_rcut": 4.0,
            "a_rcut_smth": 3.5,
            "a_sel": 300,
            "axis_neuron": 4,
            "skip_stat": True,
            "a_compress_rate": 1,
            "a_compress_e_rate": 2,
            "a_compress_use_split": True,
            "update_angle": True,
            "smooth_edge_update": True,
            "use_dynamic_sel": True,
            "sel_reduce_factor": 10.0,
            "update_style": "res_residual",
            "update_residual": 0.1,
            "update_residual_init": "const",
            "n_multi_edge_message": 1,
            "optim_update": True,
            "use_exp_switch": True,
        },
        "activation_function": "custom_silu:3.0",
        "precision": "float32",
        "use_tebd_bias": False,
        "concat_output_tebd": False,
        "exclude_types": [],
        "env_protection": 0.0,
        "trainable": True,
        "use_econf_tebd": False,
    }
    return {
        "model": {
            "_extra_state": {
                "model_params": {
                    "shared_dict": {
                        "dpa3_descriptor": descriptor,
                        "type_map": type_map,
                    },
                    # model_dict must be non-empty for read_checkpoint_type_map
                    # to enter the multi-task branch and scan shared_dict.
                    "model_dict": {
                        "SPICE2": {"fitting_net": {"type": "ener"}},
                    },
                }
            }
        }
    }


def _make_system_dirs(tmp_path, formulas=("CompA", "CompB"), n=3):
    """Create minimal system dirs with type_map.raw, set.000/coord.npy,
    and set.000/overpotential.npy.
    """
    import numpy as np

    systems = []
    for formula in formulas:
        for i in range(n):
            sysdir = tmp_path / formula / str(i)
            sysdir.mkdir(parents=True)
            (sysdir / "type_map.raw").write_text("H\nO\n")
            (sysdir / "type.raw").write_text("0\n1\n")
            sdir = sysdir / "set.000"
            sdir.mkdir()
            np.save(sdir / "coord.npy", np.zeros((2, 6)))
            np.save(sdir / "box.npy", np.tile(np.eye(3).ravel(), (2, 1)))
            np.save(sdir / "overpotential.npy", np.ones((2, 1)))
            systems.append(str(sysdir))
    return systems


def _make_system_dirs(tmp_path, formulas=("CompA", "CompB"), n=3):
    """Create minimal system dirs with type_map.raw, set.000/coord.npy,
    and set.000/overpotential.npy.
    """
    import numpy as np

    systems = []
    for formula in formulas:
        for i in range(n):
            sysdir = tmp_path / formula / str(i)
            sysdir.mkdir(parents=True)
            (sysdir / "type_map.raw").write_text("H\nO\n")
            (sysdir / "type.raw").write_text("0\n1\n")
            sdir = sysdir / "set.000"
            sdir.mkdir()
            np.save(sdir / "coord.npy", np.zeros((2, 6)))
            np.save(sdir / "box.npy", np.tile(np.eye(3).ravel(), (2, 1)))
            np.save(sdir / "overpotential.npy", np.ones((2, 1)))
            systems.append(str(sysdir))
    return systems


def _mock_dp_train(ckpt_dir):
    """Return a ``subprocess.run`` side-effect that writes a fake ckpt."""

    def _run(cmd, *args, **kwargs):
        os.makedirs(ckpt_dir, exist_ok=True)
        # Determine max_steps from config
        for a in cmd if isinstance(cmd, list) else []:
            if a.endswith(".json"):
                with open(a) as f:
                    cfg = json.load(f)
                step = cfg["training"]["numb_steps"]
                (Path(ckpt_dir) / f"model.ckpt-{step}.pt").write_bytes(b"")
                break

        class R:
            returncode = 0

        return R()

    return _run


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStrategyValidation:
    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="strategy"):
            DPAFineTuner(strategy="nonexistent")

    def test_default_is_frozen_sklearn(self):
        m = DPAFineTuner()
        assert m.strategy == "frozen_sklearn"


class TestAutoTypeMap:
    """Auto type_map inference for training paradigms."""

    def test_resolve_type_maps_from_checkpoint(self, monkeypatch, tmp_path):
        """LP/FT: type_map from checkpoint (8 elements)."""
        import torch

        monkeypatch.setattr(torch, "load", lambda *a, **kw: _fake_ckpt_sd())

        systems = _make_system_dirs(tmp_path)
        m = DPAFineTuner(
            pretrained="/fake.pt",
            strategy="linear_probe",
            init_branch="SPICE2",
        )
        tm = m._resolve_type_maps(systems)
        assert tm == _FULL_TYPE_MAP
        assert len(tm) == 8
        assert tm != []

    def test_no_type_map_raw_is_ok(self, monkeypatch, tmp_path):
        """LP/FT: missing type_map.raw should not crash (checkpoint fallback)."""
        import torch

        monkeypatch.setattr(torch, "load", lambda *a, **kw: _fake_ckpt_sd())

        import numpy as np

        systems = []
        for i in range(2):
            sysdir = tmp_path / f"sys_{i}"
            sysdir.mkdir(parents=True)
            sdir = sysdir / "set.000"
            sdir.mkdir()
            np.save(sdir / "coord.npy", np.zeros((2, 6)))
            np.save(sdir / "box.npy", np.tile(np.eye(3).ravel(), (2, 1)))
            np.save(sdir / "overpotential.npy", np.ones((2, 1)))
            systems.append(str(sysdir))

        m = DPAFineTuner(
            pretrained="/fake.pt",
            strategy="finetune",
        )
        tm = m._resolve_type_maps(systems)
        assert tm == _FULL_TYPE_MAP  # still reads from checkpoint


class TestTrainingParadigms:
    """End-to-end: each strategy builds correct config, type_map auto-inferred,
    dp train mocked to write a fake checkpoint.
    """

    @pytest.fixture(autouse=True)
    def _mock_torch(self, monkeypatch, tmp_path):
        import torch

        monkeypatch.setattr(torch, "load", lambda *a, **kw: _fake_ckpt_sd())
        # DPATrainer.__init__ checks os.path.isfile(pretrained); create a
        # real file so the check passes.
        self._ckpt = tmp_path / "fake.pt"
        self._ckpt.write_bytes(b"")

    @pytest.mark.parametrize(
        "strategy,expect_freeze,expect_tm_len",
        [
            ("linear_probe", True, 8),
            ("finetune", False, 8),
        ],
    )
    def test_config_type_map_nonempty(
        self,
        tmp_path,
        strategy,
        expect_freeze,
        expect_tm_len,
    ):
        """input.json must have non-empty type_map (not []) for each strategy."""
        out_dir = tmp_path / "out"
        systems = _make_system_dirs(tmp_path)
        valid_systems = _make_system_dirs(tmp_path, formulas=("CompC",), n=2)

        m = DPAFineTuner(
            pretrained=str(self._ckpt),
            strategy=strategy,
            property_name="overpotential",
            task_dim=1,
            intensive=True,
            max_steps=20,
            output_dir=str(out_dir),
        )

        with patch("subprocess.run", side_effect=_mock_dp_train(str(out_dir))):
            ckpt = m._fit_training(
                systems, valid_systems, m._resolve_type_maps(systems)
            )

        assert ckpt is not None
        assert "model.ckpt-20.pt" in ckpt

        # Check the generated input.json
        input_json = out_dir / "input.json"
        assert input_json.is_file(), f"input.json not found in {out_dir}"
        cfg = json.loads(input_json.read_text())
        tm = cfg["model"]["type_map"]
        assert isinstance(tm, list), f"type_map is not a list: {tm!r}"
        assert len(tm) == expect_tm_len, (
            f"{strategy}: type_map should be {expect_tm_len} elements, "
            f"got {len(tm)}: {tm}"
        )
        assert tm != [], "type_map is empty — would cause CUDA gather out-of-bounds"

    @pytest.mark.parametrize("strategy", ["linear_probe", "finetune"])
    def test_strategy_to_trainer_params(self, tmp_path, strategy):
        """Each strategy produces correct DPATrainer freeze_backbone / pretrained."""
        out_dir = tmp_path / "out"
        systems = _make_system_dirs(tmp_path)
        valid_systems = _make_system_dirs(tmp_path, formulas=("CompC",), n=2)

        m = DPAFineTuner(
            pretrained=str(self._ckpt),
            strategy=strategy,
            property_name="gap",
            task_dim=1,
            intensive=True,
            max_steps=20,
            output_dir=str(out_dir),
            init_branch="SPICE2",
        )

        with patch("subprocess.run", side_effect=_mock_dp_train(str(out_dir))):
            m._fit_training(systems, valid_systems, list(_FULL_TYPE_MAP))

        cfg = json.loads((out_dir / "input.json").read_text())

        # Check fitting_net params were propagated
        fn = cfg["model"]["fitting_net"]
        assert fn["property_name"] == "gap"
        assert fn["task_dim"] == 1
        assert fn["intensive"] is True

        # LP must freeze backbone
        if strategy == "linear_probe":
            assert cfg["model"]["descriptor"]["trainable"] is False
        else:
            assert cfg["model"]["descriptor"]["trainable"] is True

    def test_fit_dispatch_calls_training_path(self, tmp_path):
        """fit() with a training strategy calls _fit_training, not sklearn."""
        out_dir = tmp_path / "out"
        systems = _make_system_dirs(tmp_path)
        valid_systems = _make_system_dirs(tmp_path, formulas=("CompC",), n=2)

        m = DPAFineTuner(
            pretrained=str(self._ckpt),
            strategy="finetune",
            property_name="overpotential",
            max_steps=20,
            output_dir=str(out_dir),
        )

        with patch("subprocess.run", side_effect=_mock_dp_train(str(out_dir))):
            m.fit(train_data=systems, valid_data=valid_systems)

        assert m._fitted is True
        assert (out_dir / "input.json").is_file()
        cfg = json.loads((out_dir / "input.json").read_text())
        assert len(cfg["model"]["type_map"]) == 8


def _mock_load_descriptor_model_cache_test(self):
    self._checkpoint_type_map = ["H", "O"]
    return None


class TestFitDescriptorCache:
    """_fit_sklearn() caches extracted descriptors via desc_cache."""

    def test_fit_uses_cache(self, tmp_path, monkeypatch):
        """Second fit() on same data hits the cache — extraction called once."""
        import numpy as np

        # Isolate cache to a temp directory.
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

        # Create pretrained checkpoint file (cache key uses its mtime).
        ckpt = tmp_path / "fake.pt"
        ckpt.write_text("mock")

        # Create a minimal deepmd/npy system.
        root = tmp_path / "sys"
        root.mkdir()
        (root / "type.raw").write_text("0\n1\n")
        (root / "type_map.raw").write_text("H\nO\n")
        sd = root / "set.000"
        sd.mkdir()
        np.save(sd / "coord.npy", np.zeros((3, 6)))
        np.save(sd / "box.npy", np.tile(np.eye(3).ravel(), (3, 1)))
        np.save(sd / "energy.npy", np.arange(3, dtype=float))

        call_count = 0

        def _fake_extract(self, systems):
            nonlocal call_count
            call_count += 1
            n_frames = sum(s.data["coords"].shape[0] for s in systems)
            return np.random.default_rng(42).random((n_frames, 32))

        with (
            patch.object(
                DPAFineTuner,
                "_load_descriptor_model",
                _mock_load_descriptor_model_cache_test,
            ),
            patch.object(DPAFineTuner, "_extract_features", _fake_extract),
        ):
            m = DPAFineTuner(pretrained=str(ckpt), predictor="ridge")
            m.fit(str(root), target_key="energy")

            m2 = DPAFineTuner(pretrained=str(ckpt), predictor="ridge")
            m2.fit(str(root), target_key="energy")

        assert call_count == 1, f"Expected 1 extraction call, got {call_count}"
