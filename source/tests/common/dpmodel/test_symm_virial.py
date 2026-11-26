# SPDX-License-Identifier: LGPL-3.0-or-later
"""Symmetric virial supervision and evaluation share a tensor convention."""

from pathlib import (
    Path,
)
from types import (
    SimpleNamespace,
)

import numpy as np
import pytest

from deepmd.dpmodel.loss.ener import (
    EnergyLoss,
)
from deepmd.dpmodel.loss.ener_spin import (
    EnergySpinLoss,
)
from deepmd.infer.model_test.base import (
    ChunkContext,
)
from deepmd.infer.model_test.ener import (
    EnerTester,
)
from deepmd.main import (
    parse_args,
)
from deepmd.utils.eval_metrics import (
    compute_energy_type_metrics,
    compute_full_validation_energy_metrics,
    compute_full_validation_spin_metrics,
)


def _loss(spin: bool, symm: bool, loss_func: str = "mse") -> EnergyLoss:
    kwargs = {
        "starter_learning_rate": 1.0,
        "start_pref_e": 0.0,
        "limit_pref_e": 0.0,
        "start_pref_v": 1.0,
        "limit_pref_v": 1.0,
        "loss_func": loss_func,
        "symm_virial": symm,
    }
    if spin:
        return EnergySpinLoss(**kwargs)
    return EnergyLoss(**kwargs, start_pref_f=0.0, limit_pref_f=0.0)


@pytest.mark.parametrize("spin", [False, True])
@pytest.mark.parametrize("loss_func", ["mse", "mae"])
def test_antisymmetric_virial_is_optional(spin: bool, loss_func: str) -> None:
    prediction = np.array([[1.0, 5.0, 0.0, -1.0, 2.0, 0.0, 0.0, 0.0, 3.0]])
    label = np.array([[1.0, 2.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 3.0]])
    original = prediction.copy()
    model = {"energy": np.zeros((1, 1)), "virial": prediction}
    labels = {"virial": label, "find_virial": 1.0}
    for enabled in (False, True):
        loss_fn = _loss(spin, enabled, loss_func)
        value, metrics = loss_fn(1.0, 2, model, labels)
        assert (value == 0) == enabled
        restored = type(loss_fn).deserialize(loss_fn.serialize())
        np.testing.assert_equal(restored(1.0, 2, model, labels)[0], value)
        assert restored.symm_virial == enabled
        assert (metrics[f"{'rmse' if loss_func == 'mse' else 'mae'}_v"] == 0) == enabled
    np.testing.assert_array_equal(prediction, original)
    assert "symm_virial" not in _loss(spin, False).serialize()


def test_symmetric_metrics_use_nine_components() -> None:
    prediction = {"energy": np.zeros((1, 1)), "virial": np.eye(3).reshape(1, 9)}
    labels = {
        "find_virial": 1.0,
        "virial": np.zeros((1, 9)),
        "box": (2 * np.eye(3)).reshape(1, 9),
    }
    result = compute_energy_type_metrics(prediction, labels, 2, True, symm_virial=True)
    assert result.virial.weight == 9
    assert result.virial.mae == pytest.approx(1 / 3)
    assert result.virial.rmse == pytest.approx(np.sqrt(1 / 3))
    assert result.stress.mae == pytest.approx(1 / 24)
    for compute in (
        compute_full_validation_energy_metrics,
        compute_full_validation_spin_metrics,
    ):
        full = compute(prediction, labels, 2, True, symm_virial=True)
        assert full["rmse_v_per_atom"][0] == pytest.approx(result.virial_per_atom.rmse)
        assert full["mae_s"][0] == pytest.approx(result.stress.mae)


class _VirialEvaluator:
    has_efield = has_spin = has_hessian = False

    def get_dim_fparam(self) -> int:
        return 0

    def get_dim_aparam(self) -> int:
        return 0

    def has_chg_spin_ebd(self) -> bool:
        return False

    def eval(
        self, coord: np.ndarray, box: np.ndarray, atype: np.ndarray, **kwargs: object
    ) -> tuple:
        virial = np.array([[0.0, 3.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        return np.zeros((1, 1)), np.zeros((1, 6)), virial


def test_test_metrics_and_details_share_projection(tmp_path: Path) -> None:
    tester = EnerTester(_VirialEvaluator(), atomic=False, symm_virial=True)
    labels = {
        "coord": np.zeros((1, 6)),
        "type": np.zeros((1, 2), dtype=np.int32),
        "box": (2 * np.eye(3)).reshape(1, 9),
        "energy": np.zeros((1, 1)),
        "force": np.zeros((1, 6)),
        "virial": np.zeros((1, 9)),
        "find_energy": 1.0,
        "find_force": 1.0,
        "find_virial": 1.0,
    }
    context = ChunkContext("test", str(tmp_path / "detail"), False, 0)
    result = tester.evaluate_chunk(
        SimpleNamespace(pbc=True, mixed_type=False), labels, context
    )
    assert result["mae_v"][0] == pytest.approx(2 / 9)
    values = np.loadtxt(tmp_path / "detail.v.out").reshape(1, -1)
    np.testing.assert_allclose(
        values[:, 9:].reshape(3, 3), [[0, 1, 0], [1, 0, 0], [0, 0, 0]]
    )
    assert "symm_v=1" in (tmp_path / "detail.v.out").read_text()
    assert "symm_v=1" in (tmp_path / "detail.s.out").read_text()
    np.testing.assert_array_equal(labels["virial"], np.zeros((1, 9)))
    assert parse_args(
        ["test", "-m", "model.pt2", "-s", "data", "--symm-virial"]
    ).symm_virial
    assert not parse_args(["test", "-m", "model.pt2", "-s", "data"]).symm_virial


@pytest.mark.parametrize("spin", [False, True])
def test_loss_schema_defaults_to_full_virial(spin: bool) -> None:
    from dargs import (
        Argument,
    )

    from deepmd.utils.argcheck import (
        loss_ener,
        loss_ener_spin,
    )

    schema = Argument("loss", dict, (loss_ener_spin if spin else loss_ener)())
    assert schema.normalize_value({})["symm_virial"] is False
    enabled = schema.normalize_value({"symm_virial": True})
    schema.check_value(enabled, strict=True)
    assert enabled["symm_virial"] is True
