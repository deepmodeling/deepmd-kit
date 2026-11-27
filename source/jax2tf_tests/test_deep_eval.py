# SPDX-License-Identifier: LGPL-3.0-or-later
from unittest.mock import (
    patch,
)

import numpy as np
import pytest

from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    ModelOutputDef,
)
from deepmd.jax.infer.deep_eval import (
    DeepEval,
)


class _NoChargeSpinModel:
    def __init__(self) -> None:
        self.kwargs = None

    def get_rcut(self) -> float:
        return 4.0

    def get_type_map(self) -> list[str]:
        return ["H"]

    def get_dim_fparam(self) -> int:
        return 0

    def get_dim_aparam(self) -> int:
        return 0

    def has_default_fparam(self) -> bool:
        return False

    def has_chg_spin_ebd(self) -> bool:
        return False

    def __call__(self, coord, atype, **kwargs):
        del coord, atype
        self.kwargs = kwargs
        return {}


class _ChargeSpinModel(_NoChargeSpinModel):
    def has_chg_spin_ebd(self) -> bool:
        return True


def _saved_model_evaluator(model: _NoChargeSpinModel) -> DeepEval:
    with patch("deepmd.jax.jax2tf.tfmodel.TFModelWrapper", return_value=model):
        return DeepEval(
            "test.savedmodel",
            ModelOutputDef(FittingOutputDef([])),
            auto_batch_size=False,
        )


def test_eval_model_warns_and_ignores_charge_spin_without_embedding() -> None:
    model = _NoChargeSpinModel()
    deep_eval = _saved_model_evaluator(model)

    with pytest.warns(UserWarning, match="will be ignored"):
        deep_eval._eval_model(
            np.zeros((1, 6)),
            None,
            np.array([0, 0], dtype=np.int32),
            None,
            None,
            np.array([[1.0, 2.0]]),
            [],
        )

    assert model.kwargs is not None
    assert "charge_spin" not in model.kwargs


def test_eval_model_forwards_charge_spin_with_embedding() -> None:
    model = _ChargeSpinModel()
    deep_eval = _saved_model_evaluator(model)

    deep_eval._eval_model(
        np.zeros((1, 6)),
        None,
        np.array([0, 0], dtype=np.int32),
        None,
        None,
        np.array([[1.0, 2.0]]),
        [],
    )

    assert model.kwargs is not None
    np.testing.assert_allclose(np.asarray(model.kwargs["charge_spin"]), [[1.0, 2.0]])


def test_eval_model_forwards_none_charge_spin_with_embedding() -> None:
    model = _ChargeSpinModel()
    deep_eval = _saved_model_evaluator(model)

    deep_eval._eval_model(
        np.zeros((1, 6)),
        None,
        np.array([0, 0], dtype=np.int32),
        None,
        None,
        None,
        [],
    )

    assert model.kwargs is not None
    assert model.kwargs["charge_spin"] is None
