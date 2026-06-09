# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for DPAPredictor — no real DPA checkpoint or torch required.

A mock torch module is injected into sys.modules so that torch.save /
torch.load are backed by pickle.  All DPA descriptor calls are also mocked.
"""

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

# ---------------------------------------------------------------------------
# Use real torch serialization when available; otherwise fall back to a minimal
# pickle-backed mock so these tests can still run without a torch install.
# ---------------------------------------------------------------------------


def _pickle_save(obj, path, **kwargs):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _pickle_load(path, **kwargs):
    with open(path, "rb") as f:
        return pickle.load(f)


try:
    import torch as _torch_for_test
except Exception:
    _mock_torch = MagicMock()
    _mock_torch.save = _pickle_save
    _mock_torch.load = _pickle_load
    _mock_torch.cuda.is_available.return_value = False
    # Prevent scipy._lib.array_api_compat.is_torch_array from crashing
    # (it tries issubclass(cls, torch.Tensor); we make Tensor a real class).
    _mock_torch.Tensor = type("Tensor", (), {})
    _torch_for_test = _mock_torch

    # Inject before any dpa_adapt import so the lazy `import torch` lines inside
    # freeze() / DPAPredictor.__init__ pick up the mock.
    sys.modules.setdefault("torch", _mock_torch)
else:
    _torch_for_test.set_default_device(None)

from deepmd.dpa_adapt import (
    DPAFineTuner,
    DPAPredictor,
)

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
            patch.object(
                DPAFineTuner, "_load_descriptor_model", _mock_load_descriptor_model
            ),
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
            patch.object(
                DPAFineTuner, "_load_descriptor_model", _mock_load_descriptor_model
            ),
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
            patch.object(
                DPAFineTuner, "_load_descriptor_model", _mock_load_descriptor_model
            ),
            patch.object(DPAFineTuner, "_extract_features", _mock_extract_features),
        ):
            ft = DPAFineTuner(
                pretrained="fake.pt",
                model_branch="Omat24",
                predictor="linear",
            )
            ft.fit(str(system), target_key="energy")
            frozen = ft.freeze(str(tmp_path / "model.pth"))

        from deepmd.dpa_adapt._backend import (
            load_torch_file,
        )

        bundle = load_torch_file(frozen)

        assert "model_branch" in bundle, "Bundle is missing 'model_branch' key"
        assert bundle["model_branch"] == "Omat24"


# ---------------------------------------------------------------------------
# Committee helpers
# ---------------------------------------------------------------------------


def _make_mlp_bundle(tmp_path, n_frames=20):
    """Create a frozen bundle with an MLPRegressor (uses random_state)."""
    from sklearn.neural_network import (
        MLPRegressor,
    )
    from sklearn.pipeline import (
        make_pipeline,
    )
    from sklearn.preprocessing import (
        StandardScaler,
    )

    pipeline = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(10, 5),
            max_iter=300,
            random_state=42,
            early_stopping=False,
        ),
    )

    from deepmd.dpa_adapt._backend import (
        load_torch_file,
    )

    bundle = {
        "predictor": pipeline,
        "target_key": "energy",
        "type_map": ["Cu", "O"],
        "task_dim": 1,
        "pretrained": "fake.pt",
        "pooling": "mean",
        "model_branch": None,
        "condition_manager": None,
    }
    path = str(tmp_path / "mlp_model.pth")
    _torch_for_test.save(bundle, path)
    assert load_torch_file(path)["target_key"] == "energy"
    return path


def _make_rf_bundle(tmp_path, n_frames=20):
    """Create a frozen bundle with a pre-fitted RandomForestRegressor."""
    from sklearn.ensemble import (
        RandomForestRegressor,
    )
    from sklearn.pipeline import (
        make_pipeline,
    )
    from sklearn.preprocessing import (
        StandardScaler,
    )

    pipeline = make_pipeline(
        StandardScaler(),
        RandomForestRegressor(
            n_estimators=100,
            random_state=42,
        ),
    )
    # Pre-fit on synthetic data so that tree estimators are available.
    rng = np.random.default_rng(0)
    X = rng.random((n_frames, FEAT_DIM))
    y = rng.random(n_frames)
    pipeline.fit(X, y)

    from deepmd.dpa_adapt._backend import (
        load_torch_file,
    )

    bundle = {
        "predictor": pipeline,
        "target_key": "energy",
        "type_map": ["Cu", "O"],
        "task_dim": 1,
        "pretrained": "fake.pt",
        "pooling": "mean",
        "model_branch": None,
        "condition_manager": None,
    }
    path = str(tmp_path / "rf_model.pth")
    _torch_for_test.save(bundle, path)
    assert load_torch_file(path)["target_key"] == "energy"
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
            patch.object(
                DPAFineTuner, "_load_descriptor_model", _mock_load_descriptor_model
            ),
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
        assert np.any(result.uncertainty > 0), (
            "Committee std should be > 0 for some samples"
        )


class TestCommitteeThreshold:
    """After fit, uncertainty_threshold_ is set."""

    def test_committee_threshold_set(self, tmp_path):
        system = tmp_path / "sys"
        system.mkdir()
        _make_npy_system(system, n_frames=20)
        bundle_path = _make_mlp_bundle(tmp_path, n_frames=20)

        with (
            patch.object(
                DPAFineTuner, "_load_descriptor_model", _mock_load_descriptor_model
            ),
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
            patch.object(
                DPAFineTuner, "_load_descriptor_model", _mock_load_descriptor_model
            ),
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
            patch.object(
                DPAFineTuner, "_load_descriptor_model", _mock_load_descriptor_model
            ),
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
            patch.object(
                DPAFineTuner, "_load_descriptor_model", _mock_load_descriptor_model
            ),
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
            patch.object(
                DPAFineTuner, "_load_descriptor_model", _mock_load_descriptor_model
            ),
            patch.object(DPAFineTuner, "_extract_features", _mock_extract_features),
        ):
            ft = DPAFineTuner(pretrained="fake.pt", predictor="linear")
            ft.fit(str(system), target_key="energy")
            frozen = ft.freeze(str(tmp_path / "model.pth"))

            pred = DPAPredictor(frozen)
            with pytest.raises(ValueError, match="Ridge regression"):
                pred.predict(str(system), return_uncertainty=True)


# ---------------------------------------------------------------------------
# Multi-property tests
# ---------------------------------------------------------------------------


def _make_multi_npy_system(root: Path, n_frames: int = 5, n_atoms: int = 2) -> None:
    """Create a minimal system with homo.npy and lumo.npy label files."""
    (root / "type.raw").write_text("0\n1\n")
    (root / "type_map.raw").write_text("Cu\nO\n")
    set_dir = root / "set.000"
    set_dir.mkdir()
    np.save(set_dir / "coord.npy", np.zeros((n_frames, n_atoms * 3)))
    np.save(set_dir / "box.npy", np.eye(3).reshape(1, 9).repeat(n_frames, 0))
    np.save(set_dir / "homo.npy", -np.arange(n_frames, dtype=float) - 0.1)
    np.save(set_dir / "lumo.npy", np.arange(n_frames, dtype=float) + 0.1)


class TestMultiPropertyFit:
    """fit() with list[str] target_key must produce multi-output predictions."""

    @pytest.mark.parametrize("predictor_type", ["ridge", "rf", "mlp"])
    def test_multi_output_all_predictors(self, tmp_path, predictor_type):
        # MLP needs enough samples to split a validation set (10% of n_frames).
        n = 50 if predictor_type == "mlp" else 5
        system = tmp_path / "sys"
        system.mkdir()
        _make_multi_npy_system(system, n_frames=n)

        with (
            patch.object(
                DPAFineTuner, "_load_descriptor_model", _mock_load_descriptor_model
            ),
            patch.object(DPAFineTuner, "_extract_features", _mock_extract_features),
        ):
            ft = DPAFineTuner(pretrained="fake.pt", predictor=predictor_type)
            ft.fit(str(system), target_key=["homo", "lumo"])

            assert ft._task_dim == 2
            assert ft._fitted is True

            result = ft.predict(str(system))
            assert result.predictions.shape == (n, 2), (
                f"{predictor_type}: expected ({n},2), got {result.predictions.shape}"
            )


class TestMultiPropertyEvaluate:
    """evaluate() with list target_key returns per-property metrics dict."""

    def test_evaluate_returns_per_property_dict(self, tmp_path):
        system = tmp_path / "sys"
        system.mkdir()
        _make_multi_npy_system(system, n_frames=5)

        with (
            patch.object(
                DPAFineTuner, "_load_descriptor_model", _mock_load_descriptor_model
            ),
            patch.object(DPAFineTuner, "_extract_features", _mock_extract_features),
        ):
            ft = DPAFineTuner(pretrained="fake.pt", predictor="ridge")
            ft.fit(str(system), target_key=["homo", "lumo"])
            result = ft.evaluate(str(system))

        assert isinstance(result.mae, dict), (
            f"Expected dict mae, got {type(result.mae)}"
        )
        assert isinstance(result.rmse, dict)
        assert isinstance(result.r2, dict)
        assert set(result.mae.keys()) == {"homo", "lumo"}
        assert set(result.rmse.keys()) == {"homo", "lumo"}
        assert set(result.r2.keys()) == {"homo", "lumo"}
        assert all(isinstance(v, float) for v in result.mae.values())
        assert result.predictions.shape == result.labels.shape
        assert result.predictions.shape[0] == 5

    def test_single_property_still_returns_float(self, tmp_path):
        """Backward compat: single str target_key returns flat floats, not dict."""
        system = tmp_path / "sys"
        system.mkdir()
        _make_npy_system(system, n_frames=5)

        with (
            patch.object(
                DPAFineTuner, "_load_descriptor_model", _mock_load_descriptor_model
            ),
            patch.object(DPAFineTuner, "_extract_features", _mock_extract_features),
        ):
            ft = DPAFineTuner(pretrained="fake.pt", predictor="ridge")
            ft.fit(str(system), target_key="energy")
            result = ft.evaluate(str(system))

        assert isinstance(result.mae, float), (
            f"Expected float mae, got {type(result.mae)}"
        )
        assert isinstance(result.rmse, float)
        assert isinstance(result.r2, float)


class TestMultiPropertyFreezeRoundtrip:
    """freeze/load round-trip preserves list target_key and multi-output."""

    def test_freeze_load_roundtrip_list_target_key(self, tmp_path):
        system = tmp_path / "sys"
        system.mkdir()
        _make_multi_npy_system(system, n_frames=5)

        with (
            patch.object(
                DPAFineTuner, "_load_descriptor_model", _mock_load_descriptor_model
            ),
            patch.object(DPAFineTuner, "_extract_features", _mock_extract_features),
        ):
            ft = DPAFineTuner(pretrained="fake.pt", predictor="ridge")
            ft.fit(str(system), target_key=["homo", "lumo"])
            frozen = ft.freeze(str(tmp_path / "model.pth"))

            pred = DPAPredictor(frozen)
            result = pred.predict(str(system))

        assert result.predictions.shape == (5, 2)
        assert pred._target_key == ["homo", "lumo"]
        assert pred._task_dim == 2

    def test_freeze_load_roundtrip_evaluate_per_property(self, tmp_path):
        system = tmp_path / "sys"
        system.mkdir()
        _make_multi_npy_system(system, n_frames=50)

        with (
            patch.object(
                DPAFineTuner, "_load_descriptor_model", _mock_load_descriptor_model
            ),
            patch.object(DPAFineTuner, "_extract_features", _mock_extract_features),
        ):
            ft = DPAFineTuner(pretrained="fake.pt", predictor="mlp")
            ft.fit(str(system), target_key=["homo", "lumo"])
            frozen = ft.freeze(str(tmp_path / "model.pth"))

            pred = DPAPredictor(frozen)
            metrics = pred.evaluate(str(system))

        assert isinstance(metrics.mae, dict)
        assert set(metrics.mae.keys()) == {"homo", "lumo"}
        assert metrics.predictions.shape == (50, 2)
