# SPDX-License-Identifier: LGPL-3.0-or-later
import argparse
import os
import shutil
import tempfile
import unittest
from copy import (
    deepcopy,
)

import torch

from deepmd.pt_expt.entrypoints.main import (
    freeze,
    main,
)
from deepmd.pt_expt.model.get_model import (
    get_model,
)
from deepmd.pt_expt.train.wrapper import (
    ModelWrapper,
)

model_se_e2_a = {
    "type_map": ["O", "H", "B"],
    "descriptor": {
        "type": "se_e2_a",
        "sel": [46, 92, 4],
        "rcut_smth": 0.50,
        "rcut": 4.00,
        "neuron": [25, 50, 100],
        "resnet_dt": False,
        "axis_neuron": 16,
        "seed": 1,
    },
    "fitting_net": {
        "neuron": [24, 24, 24],
        "resnet_dt": True,
        "seed": 1,
    },
    "data_stat_nbatch": 20,
}


class TestDPFreezePtExpt(unittest.TestCase):
    """Test dp freeze for the pt_expt backend."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp()

        # Build a model and save a fake checkpoint
        model_params = deepcopy(model_se_e2_a)
        model = get_model(model_params)
        wrapper = ModelWrapper(model, model_params=model_params)
        state_dict = wrapper.state_dict()
        cls.ckpt_file = os.path.join(cls.tmpdir, "model.pt")
        torch.save({"model": state_dict}, cls.ckpt_file)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmpdir)

    def test_freeze_pte(self) -> None:
        """Freeze to .pte and verify the file is created."""
        output = os.path.join(self.tmpdir, "frozen_model.pte")
        freeze(model=self.ckpt_file, output=output)
        self.assertTrue(os.path.exists(output))

    def test_freeze_main_dispatcher(self) -> None:
        """Test main() CLI dispatcher with freeze command."""
        output_file = os.path.join(self.tmpdir, "frozen_via_main.pte")
        flags = argparse.Namespace(
            command="freeze",
            checkpoint_folder=self.ckpt_file,
            output=output_file,
            head=None,
            log_level=2,  # WARNING
            log_path=None,
        )
        main(flags)
        self.assertTrue(os.path.exists(output_file))

    def test_freeze_default_suffix(self) -> None:
        """Test that main() defaults output suffix to .pte."""
        output_file = os.path.join(self.tmpdir, "frozen_default_suffix.pth")
        flags = argparse.Namespace(
            command="freeze",
            checkpoint_folder=self.ckpt_file,
            output=output_file,
            head=None,
            log_level=2,  # WARNING
            log_path=None,
        )
        main(flags)
        expected = os.path.join(self.tmpdir, "frozen_default_suffix.pte")
        self.assertTrue(os.path.exists(expected))


if __name__ == "__main__":
    unittest.main()
