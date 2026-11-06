# SPDX-License-Identifier: LGPL-3.0-or-later
"""Training data ownership during TensorFlow setup failures."""

import importlib
from unittest.mock import (
    Mock,
)

import pytest


@pytest.mark.timeout(60)
@pytest.mark.parametrize("failure", ["validation", "summary", "epoch", "build"])
def test_training_setup_closes_acquired_data(monkeypatch, failure):
    """Preflight errors release every data system acquired before the failure."""
    module = importlib.import_module("deepmd.tf.entrypoints.train")
    trainer = Mock(data_requirements=[])
    trainer.model.get_type_map.return_value = ["H"]
    trainer.model.get_rcut.return_value = 6.0
    monkeypatch.setattr(module, "DPTrainer", Mock(return_value=trainer))
    train_data = Mock(type_map=["H"])
    valid_data = Mock(type_map=["H"])
    config = {
        "model": {},
        "training": {
            "training_data": {},
            "validation_data": {},
            "numb_steps": 1,
        },
    }
    if failure == "validation":
        acquired = [train_data, ValueError("validation failed")]
    else:
        acquired = [train_data, valid_data]
    if failure == "summary":
        train_data.print_summary.side_effect = ValueError("summary failed")
    elif failure == "epoch":
        config["training"].pop("numb_steps")
        config["training"]["numb_epoch"] = 0
    elif failure == "build":
        trainer.build.side_effect = ValueError("build failed")
    monkeypatch.setattr(module, "get_data", Mock(side_effect=acquired))

    with pytest.raises(ValueError):
        module._do_work(config, Mock(my_rank=0, is_chief=False))

    train_data.close.assert_called_once_with()
    if failure in ("epoch", "build"):
        valid_data.close.assert_called_once_with()
    else:
        valid_data.close.assert_not_called()
