# SPDX-License-Identifier: LGPL-3.0-or-later
import pytest

from deepmd.tf.loss import (
    EnerSpinLoss,
    EnerStdLoss,
)


@pytest.mark.parametrize("loss_type", [EnerStdLoss, EnerSpinLoss])
def test_virial_comparison(loss_type: type[EnerStdLoss] | type[EnerSpinLoss]) -> None:
    """TF1 accepts the full-virial default but cannot train on symmetric virials."""
    full = loss_type(starter_learning_rate=1.0, symm_virial=False)
    assert full.serialize() == loss_type(starter_learning_rate=1.0).serialize()
    with pytest.raises(NotImplementedError, match="symm_virial=True"):
        loss_type(starter_learning_rate=1.0, symm_virial=True)
