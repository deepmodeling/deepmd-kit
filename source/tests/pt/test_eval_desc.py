# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import os
import shutil
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

from deepmd.entrypoints.eval_desc import (
    eval_desc,
)
from deepmd.pt.entrypoints.main import (
    get_trainer,
)

from .model.test_permutation import (
    model_se_e2_a,
)


class DPEvalDesc:
    def test_dp_eval_desc_1_frame(self) -> None:
        trainer = get_trainer(deepcopy(self.config))
        with torch.device("cpu"):
            input_dict, label_dict, _ = trainer.get_data(is_train=False)
        has_spin = getattr(trainer.model, "has_spin", False)
        if callable(has_spin):
            has_spin = has_spin()
        if not has_spin:
            input_dict.pop("spin", None)
        input_dict["do_atomic_virial"] = True
        result = trainer.model(**input_dict)
        model = torch.jit.script(trainer.model)
        tmp_model = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
        torch.jit.save(model, tmp_model.name)

        # Test eval_desc
        eval_desc(
            model=tmp_model.name,
            system=self.config["training"]["validation_data"]["systems"][0],
            datafile=None,
            output=self.output_dir,
        )
        os.unlink(tmp_model.name)

        # Check that descriptor file was created
        system_name = os.path.basename(
            self.config["training"]["validation_data"]["systems"][0].rstrip("/")
        )
        desc_file = os.path.join(self.output_dir, f"{system_name}.npy")
        self.assertTrue(os.path.exists(desc_file))

        # Load and validate descriptor
        descriptors = np.load(desc_file)
        self.assertIsInstance(descriptors, np.ndarray)
        # Descriptors should be 3D: (nframes, natoms, ndesc)
        self.assertEqual(len(descriptors.shape), 3)  # Should be 3D array
        self.assertGreater(descriptors.shape[0], 0)  # Should have frames
        self.assertGreater(descriptors.shape[1], 0)  # Should have atoms
        self.assertGreater(descriptors.shape[2], 0)  # Should have descriptor dimensions

    def tearDown(self) -> None:
        for f in os.listdir("."):
            if f.startswith("model") and f.endswith(".pt"):
                os.remove(f)
            if f in ["lcurve.out", self.input_json]:
                os.remove(f)
            if f in ["stat_files"]:
                shutil.rmtree(f)
        # Clean up output directory
        if hasattr(self, "output_dir") and os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)


class TestDPEvalDescSeA(DPEvalDesc, unittest.TestCase):
    def setUp(self) -> None:
        self.output_dir = "test_eval_desc_output"
        input_json = str(Path(__file__).parent / "water" / "se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        data_file = [str(Path(__file__).parent / "water" / "data" / "single")]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file
        self.config["model"] = deepcopy(model_se_e2_a)
        self.input_json = "test_eval_desc.json"
        with open(self.input_json, "w") as fp:
            json.dump(self.config, fp, indent=4)


if __name__ == "__main__":
    unittest.main()
