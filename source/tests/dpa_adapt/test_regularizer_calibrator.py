# SPDX-License-Identifier: LGPL-3.0-or-later

from __future__ import (
    annotations,
)

from typing import (
    TYPE_CHECKING,
)
from unittest.mock import (
    patch,
)

if TYPE_CHECKING:
    from pathlib import (
        Path,
    )

import numpy as np
import pytest

from dpa_adapt import (
    Calibrator,
    DPAFineTuner,
    DPAPredictor,
    Regularizer,
)
from source.tests.dpa_adapt.test_predictor import (
    _make_npy_system,
    _mock_extract_features,
    _mock_load_descriptor_model,
)


def test_regularizer_rejects_mft_aux_fields() -> None:
    with pytest.raises(ValueError, match="strategy='mft'"):
        Regularizer.from_config({"aux_data": "mp_traj/dpdata"})


def test_regularizer_nonzero_fails_loudly_until_backend_exists(tmp_path: Path) -> None:
    system = tmp_path / "sys"
    system.mkdir()
    _make_npy_system(system)
    model = DPAFineTuner(
        pretrained="fake.pt",
        predictor="linear",
        regularizer={"descriptor_anchor": 0.01},
    )

    with pytest.raises(NotImplementedError, match="descriptor-anchor"):
        model.fit(str(system), target_key="energy")


def test_calibrator_prediction_only_ridge() -> None:
    calibrator = Calibrator(method="ridge", alpha=0.0)
    raw = np.array([[1.0], [2.0], [3.0], [4.0]])
    labels = 2.0 * raw + 1.0

    calibrator.fit_from_arrays(raw, labels)
    pred = calibrator.predict_from_arrays(np.array([[5.0], [6.0]]))

    np.testing.assert_allclose(pred, np.array([[11.0], [13.0]]), atol=1e-10)


def test_calibrator_uses_fparam_and_group_stats(tmp_path: Path) -> None:
    system = tmp_path / "sys"
    system.mkdir()
    _make_npy_system(system, n_frames=4)
    fparam = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
    np.save(system / "set.000" / "fparam.npy", fparam)

    raw = np.array([[1.0], [1.5], [2.0], [2.5]])
    labels = raw + fparam[:, :1] * 0.5 + 2.0

    calibrator = Calibrator(
        method="ridge",
        alpha=0.0,
        features=["prediction", "fparam", "group_stats"],
    )
    calibrator.fit_from_arrays(raw, labels, data=str(system))
    pred = calibrator.predict_from_arrays(raw, data=str(system))

    assert "fparam_0" in calibrator.feature_names_
    assert "weight_eff_n" in calibrator.feature_names_
    np.testing.assert_allclose(pred, labels, atol=1e-10)


def test_calibrator_aligns_group_stats_to_group_predictions(tmp_path: Path) -> None:
    system = tmp_path / "grouped"
    system.mkdir()
    _make_npy_system(system, n_frames=4)
    set_dir = system / "set.000"
    np.save(set_dir / "group_id.npy", np.array([0, 0, 1, 1]))
    np.save(set_dir / "weight.npy", np.array([0.25, 0.75, 1.0, 3.0]))
    np.save(set_dir / "pool_mask.npy", np.ones((4, 2)))
    np.save(
        set_dir / "fparam.npy",
        np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
    )

    raw = np.array([[2.0], [4.0]])
    labels = np.array([[10.0], [20.0]])
    calibrator = Calibrator(
        method="ridge",
        alpha=0.0,
        features=["prediction", "fparam", "group_stats"],
    )

    calibrator.fit_from_arrays(raw, labels, data=str(system))
    pred = calibrator.predict_from_arrays(raw, data=str(system))

    assert pred.shape == (2, 1)
    assert "n_frames" in calibrator.feature_names_
    assert "weight_l2" in calibrator.feature_names_


def test_finetuner_calibrate_applies_and_freezes(tmp_path: Path) -> None:
    system = tmp_path / "sys"
    system.mkdir()
    _make_npy_system(system, n_frames=5)

    with (
        patch.object(
            DPAFineTuner, "_load_descriptor_model", _mock_load_descriptor_model
        ),
        patch.object(DPAFineTuner, "_extract_features", _mock_extract_features),
    ):
        model = DPAFineTuner(pretrained="fake.pt", predictor="linear")
        model.fit(str(system), target_key="energy")
        raw = model.predict(str(system), calibrated=False).predictions
        model.calibrate(str(system), method="ridge", alpha=0.0)
        calibrated = model.predict(str(system)).predictions

        labels = np.arange(5).reshape(-1, 1)
        raw_rmse = float(np.sqrt(np.mean((raw - labels) ** 2)))
        calibrated_rmse = float(np.sqrt(np.mean((calibrated - labels) ** 2)))

        assert model.calibrator is not None
        assert calibrated_rmse <= raw_rmse
        np.testing.assert_allclose(
            model.predict(str(system)).raw_predictions,
            raw,
        )

        frozen = model.freeze(str(tmp_path / "model.pth"))
        pred = DPAPredictor(frozen)
        loaded = pred.predict(str(system))

    assert hasattr(loaded, "raw_predictions")
    assert loaded.predictions.shape == (5, 1)
