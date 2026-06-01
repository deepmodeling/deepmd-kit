"""Tests for DPAPredictor — no real DPA checkpoint or torch required.

A mock torch module is injected into sys.modules so that torch.save /
torch.load are backed by pickle.  All DPA descriptor calls are also mocked.
"""
import pickle
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Build a minimal mock torch module backed by pickle
# ---------------------------------------------------------------------------

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

# Inject before any dpa_tools import so the lazy `import torch` lines inside
# freeze() / DPAPredictor.__init__ pick up the mock.
sys.modules.setdefault("torch", _mock_torch)

from deepmd.dpa_tools import DPAFineTuner, DPAPredictor  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_npy_system(root: Path, n_frames: int = 3, n_atoms: int = 2) -> None:
    """Create a minimal deepmd/npy system directory for testing."""
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
    return np.random.default_rng(0).random((n_frames, FEAT_DIM))


def _mock_load_descriptor_model(self):
    self._checkpoint_type_map = ["Cu", "O"]
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPredictRoundtrip:
    """Freeze a Ridge on mock features, reload with DPAPredictor, check shape."""

    def test_predict_roundtrip(self, tmp_path):
        system = tmp_path / "sys"
        system.mkdir()
        _make_npy_system(system, n_frames=4)

        with (
            patch.object(DPAFineTuner, "_load_descriptor_model", _mock_load_descriptor_model),
            patch.object(DPAFineTuner, "_extract_features", _mock_extract_features),
        ):
            ft = DPAFineTuner(pretrained="fake.pt", predictor="linear")
            ft.fit(str(system), target_key="energy")
            frozen = ft.freeze(str(tmp_path / "model.pth"))

            pred = DPAPredictor(frozen)
            result = pred.predict(str(system))

        assert hasattr(result, "predictions")
        assert result.predictions.shape == (4, 1)


class TestEvaluateReturnsMetrics:
    """evaluate() must return mae/rmse/r2/predictions/labels with consistent shapes."""

    def test_evaluate_returns_metrics(self, tmp_path):
        system = tmp_path / "sys"
        system.mkdir()
        _make_npy_system(system, n_frames=5)

        with (
            patch.object(DPAFineTuner, "_load_descriptor_model", _mock_load_descriptor_model),
            patch.object(DPAFineTuner, "_extract_features", _mock_extract_features),
        ):
            ft = DPAFineTuner(pretrained="fake.pt", predictor="linear")
            ft.fit(str(system), target_key="energy")
            frozen = ft.freeze(str(tmp_path / "model.pth"))

            pred = DPAPredictor(frozen)
            result = pred.evaluate(str(system))

        for key in ("mae", "rmse", "r2", "predictions", "labels"):
            assert hasattr(result, key), f"Missing key: {key}"

        assert result.predictions.shape == result.labels.shape
        assert result.predictions.shape[0] == 5
        assert isinstance(result.mae, float)
        assert isinstance(result.rmse, float)


class TestFreezeBundleHasModelBranch:
    """freeze() bundle must include model_branch (guards the §1 bug fix)."""

    def test_freeze_bundle_has_model_branch(self, tmp_path):
        system = tmp_path / "sys"
        system.mkdir()
        _make_npy_system(system, n_frames=3)

        with (
            patch.object(DPAFineTuner, "_load_descriptor_model", _mock_load_descriptor_model),
            patch.object(DPAFineTuner, "_extract_features", _mock_extract_features),
        ):
            ft = DPAFineTuner(
                pretrained="fake.pt",
                model_branch="Omat24",
                predictor="linear",
            )
            ft.fit(str(system), target_key="energy")
            frozen = ft.freeze(str(tmp_path / "model.pth"))

        with open(frozen, "rb") as f:
            bundle = pickle.load(f)

        assert "model_branch" in bundle, "Bundle is missing 'model_branch' key"
        assert bundle["model_branch"] == "Omat24"


# ---------------------------------------------------------------------------
# Committee helpers
# ---------------------------------------------------------------------------

def _make_mlp_bundle(tmp_path, n_frames=20):
    """Create a frozen bundle with an MLPRegressor (uses random_state)."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pipeline = make_pipeline(StandardScaler(), MLPRegressor(
        hidden_layer_sizes=(10, 5),
        max_iter=300,
        random_state=42,
        early_stopping=False,
    ))

    bundle = {
        "predictor":          pipeline,
        "target_key":         "energy",
        "type_map":           ["Cu", "O"],
        "task_dim":           1,
        "pretrained":         "fake.pt",
        "pooling":            "mean",
        "model_branch":       None,
        "condition_manager":  None,
    }
    path = str(tmp_path / "mlp_model.pth")
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    return path


def _make_rf_bundle(tmp_path, n_frames=20):
    """Create a frozen bundle with a pre-fitted RandomForestRegressor."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pipeline = make_pipeline(StandardScaler(), RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    ))
    # Pre-fit on synthetic data so that tree estimators are available.
    rng = np.random.default_rng(0)
    X = rng.random((n_frames, FEAT_DIM))
    y = rng.random(n_frames)
    pipeline.fit(X, y)

    bundle = {
        "predictor":          pipeline,
        "target_key":         "energy",
        "type_map":           ["Cu", "O"],
        "task_dim":           1,
        "pretrained":         "fake.pt",
        "pooling":            "mean",
        "model_branch":       None,
        "condition_manager":  None,
    }
    path = str(tmp_path / "rf_model.pth")
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    return path


# ---------------------------------------------------------------------------
# Committee tests
# ---------------------------------------------------------------------------

class TestCommitteeFitPredict:
    """n_committee > 1 trains ensemble and returns mean+std."""

    def test_committee_fit_predict(self, tmp_path):
        system = tmp_path / "sys"
        system.mkdir()
        _make_npy_system(system, n_frames=20)
        bundle_path = _make_mlp_bundle(tmp_path, n_frames=20)

        with (
            patch.object(DPAFineTuner, "_load_descriptor_model", _mock_load_descriptor_model),
            patch.object(DPAFineTuner, "_extract_features", _mock_extract_features),
        ):
            pred = DPAPredictor(bundle_path, n_committee=5)
            pred.fit(str(system), target_key="energy")
            result = pred.predict(str(system), return_uncertainty=True)

        assert hasattr(result, "predictions")
        assert hasattr(result, "uncertainty")
        assert result.predictions.shape == (20, 1)
        assert result.uncertainty.shape == (20, 1)
        assert np.all(result.uncertainty >= 0)
        assert np.any(result.uncertainty > 0), "Committee std should be > 0 for some samples"


class TestCommitteeThreshold:
    """After fit, uncertainty_threshold_ is set."""

    def test_committee_threshold_set(self, tmp_path):
        system = tmp_path / "sys"
        system.mkdir()
        _make_npy_system(system, n_frames=20)
        bundle_path = _make_mlp_bundle(tmp_path, n_frames=20)

        with (
            patch.object(DPAFineTuner, "_load_descriptor_model", _mock_load_descriptor_model),
            patch.object(DPAFineTuner, "_extract_features", _mock_extract_features),
        ):
            pred = DPAPredictor(bundle_path, n_committee=5)
            pred.fit(str(system), target_key="energy")

        assert hasattr(pred, "uncertainty_threshold_")
        assert isinstance(pred.uncertainty_threshold_, float)
        assert pred.uncertainty_threshold_ > 0


class TestCommitteeN1BackwardCompat:
    """n_committee=1 must behave identically to the current single-estimator behaviour."""

    def test_committee_n1_backward_compat(self, tmp_path):
        system = tmp_path / "sys"
        system.mkdir()
        _make_npy_system(system, n_frames=4)

        with (
            patch.object(DPAFineTuner, "_load_descriptor_model", _mock_load_descriptor_model),
            patch.object(DPAFineTuner, "_extract_features", _mock_extract_features),
        ):
            ft = DPAFineTuner(pretrained="fake.pt", predictor="linear")
            ft.fit(str(system), target_key="energy")
            frozen = ft.freeze(str(tmp_path / "model.pth"))

            pred = DPAPredictor(frozen, n_committee=1)
            result = pred.predict(str(system))

        assert hasattr(result, "predictions")
        assert result.predictions.shape == (4, 1)


class TestReturnUncertaintyFalse:
    """Default return_uncertainty=False returns DotDict (not a tuple)."""

    def test_return_uncertainty_false(self, tmp_path):
        system = tmp_path / "sys"
        system.mkdir()
        _make_npy_system(system, n_frames=20)
        bundle_path = _make_mlp_bundle(tmp_path, n_frames=20)

        with (
            patch.object(DPAFineTuner, "_load_descriptor_model", _mock_load_descriptor_model),
            patch.object(DPAFineTuner, "_extract_features", _mock_extract_features),
        ):
            pred = DPAPredictor(bundle_path, n_committee=5)
            pred.fit(str(system), target_key="energy")
            result = pred.predict(str(system))  # default return_uncertainty=False

        assert not isinstance(result, tuple)
        assert hasattr(result, "predictions")
        assert not hasattr(result, "uncertainty"), (
            "uncertainty should not be present when return_uncertainty=False"
        )


class TestRfUncertainty:
    """RF natively supports uncertainty via per-tree std."""

    def test_rf_uncertainty(self, tmp_path):
        system = tmp_path / "sys"
        system.mkdir()
        _make_npy_system(system, n_frames=20)
        bundle_path = _make_rf_bundle(tmp_path, n_frames=20)

        with (
            patch.object(DPAFineTuner, "_load_descriptor_model", _mock_load_descriptor_model),
            patch.object(DPAFineTuner, "_extract_features", _mock_extract_features),
        ):
            pred = DPAPredictor(bundle_path)
            result = pred.predict(str(system), return_uncertainty=True)

        assert hasattr(result, "predictions")
        assert hasattr(result, "uncertainty")
        assert result.predictions.shape == (20, 1)
        assert result.uncertainty.shape == (20, 1)
        assert np.all(result.uncertainty >= 0)
        assert np.any(result.uncertainty > 0), (
            "RF tree-level std should be > 0 for some samples"
        )


class TestRidgeUncertaintyRaises:
    """Ridge cannot produce uncertainty — calling return_uncertainty=True must raise."""

    def test_ridge_uncertainty_raises(self, tmp_path):
        system = tmp_path / "sys"
        system.mkdir()
        _make_npy_system(system, n_frames=4)

        with (
            patch.object(DPAFineTuner, "_load_descriptor_model", _mock_load_descriptor_model),
            patch.object(DPAFineTuner, "_extract_features", _mock_extract_features),
        ):
            ft = DPAFineTuner(pretrained="fake.pt", predictor="linear")
            ft.fit(str(system), target_key="energy")
            frozen = ft.freeze(str(tmp_path / "model.pth"))

            pred = DPAPredictor(frozen)
            with pytest.raises(ValueError, match="Ridge regression"):
                pred.predict(str(system), return_uncertainty=True)
