# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for ConditionManager and conditions integration — no real DPA checkpoint needed."""

import pickle
import sys
from pathlib import (
    Path,
)
from unittest.mock import (
    MagicMock,
    patch,
)

import numpy as np
import pytest

# ---- mock torch (same pattern as test_predictor.py) ----


def _pickle_save(obj, path, **kwargs):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _pickle_load(path, **kwargs):
    with open(path, "rb") as f:
        return pickle.load(f)


_mock_torch = MagicMock()
_mock_torch.save = _pickle_save
_mock_torch.load = _pickle_load
_mock_torch.cuda.is_available.return_value = False

sys.modules.setdefault("torch", _mock_torch)

from deepmd.dpa_adapt import (
    DPAFineTuner,
    DPAPredictor,
)
from deepmd.dpa_adapt.conditions import (
    ConditionManager,
    DPAConditionError,
)

# ---- helpers ----


def _make_npy_system(root: Path, n_frames: int = 3, n_atoms: int = 2) -> None:
    (root / "type.raw").write_text("0\n1\n")
    (root / "type_map.raw").write_text("Cu\nO\n")
    set_dir = root / "set.000"
    set_dir.mkdir()
    np.save(set_dir / "coord.npy", np.zeros((n_frames, n_atoms * 3)))
    np.save(set_dir / "box.npy", np.eye(3).reshape(1, 9).repeat(n_frames, 0))
    np.save(set_dir / "energy.npy", np.arange(n_frames, dtype=float))


FEAT_DIM = 8


def _mock_extract_features(self, systems):
    n_frames = sum(s.data["coords"].shape[0] for s in systems)
    rng = np.random.default_rng(0)
    return rng.random((n_frames, FEAT_DIM))


def _mock_load_descriptor_model(self):
    self._checkpoint_type_map = ["Cu", "O"]
    return None


# ======================================================================
# ConditionManager tests
# ======================================================================


class TestConditionManager:
    def test_fit_transform_single_key(self):
        cm = ConditionManager()
        cond = {"T": np.array([300.0, 400.0, 500.0])}
        X = cm.fit_transform(cond)
        assert X.shape == (3, 1)

    def test_fit_transform_multi_key(self):
        cm = ConditionManager()
        cond = {
            "T": np.array([300.0, 400.0, 500.0]),
            "P": np.array([1.0, 2.0, 3.0]),
        }
        X = cm.fit_transform(cond)
        assert X.shape == (3, 2)

    def test_transform_normalizes_correctly(self):
        cm = ConditionManager()
        cond = {"T": np.array([300.0, 400.0, 500.0])}
        X = cm.fit_transform(cond)
        assert abs(X.mean()) < 1e-6
        assert abs(X.std(ddof=0) - 1.0) < 1e-6

    def test_save_load_roundtrip(self, tmp_path):
        cm = ConditionManager()
        cond = {"T": np.array([300.0, 400.0, 500.0])}
        cm.fit(cond)
        expected = cm.transform(cond)

        path = str(tmp_path / "cm.pkl")
        cm.save(path)
        cm2 = ConditionManager.load(path)
        result = cm2.transform(cond)
        np.testing.assert_array_equal(result, expected)

    def test_transform_before_fit_raises(self):
        cm = ConditionManager()
        with pytest.raises(DPAConditionError, match="before fit"):
            cm.transform({"T": np.array([1.0])})

    def test_transform_missing_key_raises(self):
        cm = ConditionManager()
        cm.fit({"T": np.array([1.0, 2.0])})
        with pytest.raises(DPAConditionError, match="missing from transform"):
            cm.transform({"other": np.array([1.0, 2.0])})


# ======================================================================
# DPAFineTuner with conditions
# ======================================================================


class TestFineTunerWithConditions:
    def test_fit_with_conditions_changes_feature_dim(self, tmp_path):
        system = tmp_path / "sys"
        system.mkdir()
        _make_npy_system(system, n_frames=4)

        with (
            patch.object(
                DPAFineTuner, "_load_descriptor_model", _mock_load_descriptor_model
            ),
            patch.object(DPAFineTuner, "_extract_features", _mock_extract_features),
        ):
            ft = DPAFineTuner(pretrained="fake.pt", predictor="linear")
            cond = {"T": np.array([300.0, 400.0, 500.0, 600.0])}
            ft.fit(str(system), target_key="energy", conditions=cond)

        # The pipeline's first step (StandardScaler) reveals the input dim
        scaler = ft.predictor.named_steps["standardscaler"]
        assert scaler.n_features_in_ == FEAT_DIM + 1

    def test_predict_missing_conditions_raises(self, tmp_path):
        system = tmp_path / "sys"
        system.mkdir()
        _make_npy_system(system, n_frames=4)

        with (
            patch.object(
                DPAFineTuner, "_load_descriptor_model", _mock_load_descriptor_model
            ),
            patch.object(DPAFineTuner, "_extract_features", _mock_extract_features),
        ):
            ft = DPAFineTuner(pretrained="fake.pt", predictor="linear")
            cond = {"T": np.array([300.0, 400.0, 500.0, 600.0])}
            ft.fit(str(system), target_key="energy", conditions=cond)

            with pytest.raises(DPAConditionError, match="fit with conditions"):
                ft.predict(str(system))

    def test_predict_unexpected_conditions_raises(self, tmp_path):
        system = tmp_path / "sys"
        system.mkdir()
        _make_npy_system(system, n_frames=4)

        with (
            patch.object(
                DPAFineTuner, "_load_descriptor_model", _mock_load_descriptor_model
            ),
            patch.object(DPAFineTuner, "_extract_features", _mock_extract_features),
        ):
            ft = DPAFineTuner(pretrained="fake.pt", predictor="linear")
            ft.fit(str(system), target_key="energy")

            with pytest.raises(DPAConditionError, match="fit without conditions"):
                ft.predict(
                    str(system), conditions={"T": np.array([1.0, 2.0, 3.0, 4.0])}
                )

    def test_freeze_load_with_conditions(self, tmp_path):
        system = tmp_path / "sys"
        system.mkdir()
        _make_npy_system(system, n_frames=4)

        with (
            patch.object(
                DPAFineTuner, "_load_descriptor_model", _mock_load_descriptor_model
            ),
            patch.object(DPAFineTuner, "_extract_features", _mock_extract_features),
        ):
            ft = DPAFineTuner(pretrained="fake.pt", predictor="linear")
            cond = {"T": np.array([300.0, 400.0, 500.0, 600.0])}
            ft.fit(str(system), target_key="energy", conditions=cond)

            frozen = ft.freeze(str(tmp_path / "model.pth"))

            pred = DPAPredictor(frozen)
            result = pred.predict(str(system), conditions=cond)

        assert result.predictions.shape == (4, 1)


# ======================================================================
# DPAFineTuner without conditions (backward compat)
# ======================================================================


class TestFineTunerNoConditions:
    def test_fit_predict_no_conditions_unchanged(self, tmp_path):
        system = tmp_path / "sys"
        system.mkdir()
        _make_npy_system(system, n_frames=4)

        with (
            patch.object(
                DPAFineTuner, "_load_descriptor_model", _mock_load_descriptor_model
            ),
            patch.object(DPAFineTuner, "_extract_features", _mock_extract_features),
        ):
            ft = DPAFineTuner(pretrained="fake.pt", predictor="linear")
            ft.fit(str(system), target_key="energy")

            result = ft.predict(str(system))

        assert result.predictions.shape == (4, 1)
