# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
import pytest

from deepmd.jax.infer.deep_eval import (
    DeepEval,
)


class _NoChargeSpinModel:
    def get_dim_fparam(self) -> int:
        return 0

    def get_dim_aparam(self) -> int:
        return 0

    def has_default_fparam(self) -> bool:
        return False

    def has_chg_spin_ebd(self) -> bool:
        return False


def test_eval_model_rejects_charge_spin_without_embedding() -> None:
    deep_eval = object.__new__(DeepEval)
    deep_eval.dp = _NoChargeSpinModel()

    with pytest.raises(ValueError, match="does not support"):
        deep_eval._eval_model(
            np.zeros((1, 6)),
            None,
            np.array([0, 0], dtype=np.int32),
            None,
            None,
            np.array([[1.0, 2.0]]),
            [],
        )
