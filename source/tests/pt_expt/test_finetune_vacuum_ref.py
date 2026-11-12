# SPDX-License-Identifier: LGPL-3.0-or-later
"""The reference condition of a fine-tuned model comes from the target task."""

import tempfile
import unittest
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)

import numpy as np
import torch

from deepmd.pt_expt.model.get_model import (
    get_model,
)
from deepmd.pt_expt.train.wrapper import (
    ModelWrapper,
)
from deepmd.pt_expt.utils.env import (
    DEVICE,
)
from deepmd.pt_expt.utils.finetune import (
    get_finetune_rules,
)
from deepmd.utils.argcheck import (
    normalize,
)

TYPE_MAP = ["O", "H"]


def model_config(preset: dict | None, default=(0, 1)) -> dict:
    descriptor = {
        "type": "dpa4",
        "sel": 20,
        "rcut": 4.0,
        "channels": 16,
        "n_radial": 8,
        "lmax": 2,
        "mmax": 1,
        "n_blocks": 2,
        "precision": "float64",
        "seed": 1,
        "add_chg_spin_ebd": True,
    }
    if default is not None:
        descriptor["default_chg_spin"] = list(default)
    model = {
        "type": "dpa4",
        "type_map": TYPE_MAP,
        "descriptor": descriptor,
        "fitting_net": {
            "type": "dpa4_ener",
            "neuron": [16],
            "precision": "float64",
            "seed": 1,
            "vacuum_ref": True,
        },
    }
    if preset is not None:
        model["preset_out_bias"] = preset
    return normalize(
        {
            "model": model,
            "learning_rate": {"start_lr": 1e-3},
            "loss": {},
            "training": {"training_data": {"systems": ["x"]}, "numb_steps": 1},
        }
    )["model"]


def reference(model) -> np.ndarray:
    """The condition every type is referenced to, or None without a reference."""
    conditions = model.atomic_model.vacuum_charge_spin
    return None if conditions is None else conditions.cpu().numpy()


class TestFinetuneVacuumReference(unittest.TestCase):
    """The reference of the target task, not of the pretrained model, holds."""

    def setUp(self) -> None:
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.pretrained = self.tmp / "pretrained.pt"
        # The pretrained task takes (0, 1) in every frame and names no
        # condition, so its reference is the default condition.
        params = model_config({"energy": {"O": -3.0, "H": 0.5}})
        model = get_model(deepcopy(params)).to(DEVICE)
        torch.save(
            {"model": ModelWrapper(model, model_params=params).state_dict()},
            self.pretrained,
        )
        self.pretrained_reference = np.array([[0.0, 1.0], [0.0, 1.0]])

    def finetuned(self, target: dict, change_model_params: bool = True):
        config, _ = get_finetune_rules(
            str(self.pretrained), target, change_model_params=change_model_params
        )
        return get_model(config)

    def test_preset_names_the_condition(self) -> None:
        """A target table that names the state uses it, whatever the default."""
        model = self.finetuned(
            model_config(
                {
                    "energy": {
                        "O": {"value": -3.0, "charge": 0, "multiplicity": 3},
                        "H": {"value": 0.5, "charge": 0, "multiplicity": 2},
                    }
                },
                default=(1, 2),
            )
        )
        np.testing.assert_array_equal(reference(model), [[0.0, 3.0], [0.0, 2.0]])

    def test_preset_without_a_condition_takes_the_target_default(self) -> None:
        """A target table without a condition takes the target default."""
        model = self.finetuned(
            model_config({"energy": {"O": -3.0, "H": 0.5}}, default=(1, 2))
        )
        np.testing.assert_array_equal(reference(model), [[1.0, 2.0]] * 2)
        # the resolved condition belongs to the model from then on
        stored = model.atomic_model.preset_out_bias["@charge_spin"]
        self.assertEqual(stored, [[1.0, 2.0], [1.0, 2.0]])

    def test_preset_without_a_condition_inherits_the_pretrained_default(self) -> None:
        """Without a target default the pretrained one holds."""
        model = self.finetuned(
            model_config({"energy": {"O": -3.0, "H": 0.5}}, default=None)
        )
        np.testing.assert_array_equal(reference(model), self.pretrained_reference)

    def test_training_without_a_preset_has_no_reference(self) -> None:
        """A target without a preset keeps the plain network output."""
        model = self.finetuned(model_config(None))
        self.assertIsNone(reference(model))
        self.assertFalse(model.atomic_model.fitting_net.vacuum_ref)


if __name__ == "__main__":
    unittest.main()
