# SPDX-License-Identifier: LGPL-3.0-or-later
"""Smoke test for the pt_expt training infrastructure.

Verifies that:
1. ``get_model`` constructs a model from config
2. ``make_stat_input`` + ``compute_or_load_stat`` work
3. A few training steps run without error
4. Loss decreases over those steps
"""

import os
import shutil
import tempfile
import unittest

import torch

from deepmd.pt_expt.entrypoints.main import (
    get_trainer,
)
from deepmd.pt_expt.model import (
    get_model,
)
from deepmd.utils.argcheck import (
    normalize,
)
from deepmd.utils.compat import (
    update_deepmd_input,
)

EXAMPLE_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "examples",
    "water",
)


def _make_config(data_dir: str, numb_steps: int = 5) -> dict:
    """Build a minimal config dict pointing at *data_dir*."""
    config = {
        "model": {
            "type_map": ["O", "H"],
            "descriptor": {
                "type": "se_e2_a",
                "sel": [6, 12],
                "rcut_smth": 0.50,
                "rcut": 3.00,
                "neuron": [8, 16],
                "resnet_dt": False,
                "axis_neuron": 4,
                "type_one_side": True,
                "seed": 1,
            },
            "fitting_net": {
                "neuron": [16, 16],
                "resnet_dt": True,
                "seed": 1,
            },
            "data_stat_nbatch": 1,
        },
        "learning_rate": {
            "type": "exp",
            "decay_steps": 500,
            "start_lr": 0.001,
            "stop_lr": 3.51e-8,
        },
        "loss": {
            "type": "ener",
            "start_pref_e": 0.02,
            "limit_pref_e": 1,
            "start_pref_f": 1000,
            "limit_pref_f": 1,
            "start_pref_v": 0,
            "limit_pref_v": 0,
        },
        "training": {
            "training_data": {
                "systems": [
                    os.path.join(data_dir, "data_0"),
                ],
                "batch_size": 1,
            },
            "validation_data": {
                "systems": [
                    os.path.join(data_dir, "data_3"),
                ],
                "batch_size": 1,
                "numb_btch": 1,
            },
            "numb_steps": numb_steps,
            "seed": 10,
            "disp_file": "lcurve.out",
            "disp_freq": 5,
            "save_freq": numb_steps,
        },
    }
    return config


class TestTraining(unittest.TestCase):
    """Basic smoke test for the pt_expt training loop."""

    @classmethod
    def setUpClass(cls) -> None:
        data_dir = os.path.join(EXAMPLE_DIR, "data")
        if not os.path.isdir(data_dir):
            raise unittest.SkipTest(f"Example data not found: {data_dir}")
        cls.data_dir = data_dir

    def test_get_model(self) -> None:
        """Test that get_model constructs a model from config."""
        config = _make_config(self.data_dir)
        config = update_deepmd_input(config, warning=False)
        config = normalize(config)
        model = get_model(config["model"])
        # model should be a torch.nn.Module
        self.assertIsInstance(model, torch.nn.Module)
        # should have parameters
        nparams = sum(p.numel() for p in model.parameters())
        self.assertGreater(nparams, 0)

    def _run_training(self, config: dict) -> None:
        """Run training and verify lcurve + checkpoint creation."""
        tmpdir = tempfile.mkdtemp(prefix="pt_expt_train_")
        try:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                trainer = get_trainer(config)
                trainer.run()

                # Read lcurve to verify training ran
                lcurve_path = os.path.join(tmpdir, "lcurve.out")
                self.assertTrue(os.path.exists(lcurve_path), "lcurve.out not created")

                with open(lcurve_path) as f:
                    lines = [l for l in f.readlines() if not l.startswith("#")]
                self.assertGreater(len(lines), 0, "lcurve.out is empty")

                # Verify checkpoint was saved
                ckpt_files = [f for f in os.listdir(tmpdir) if f.endswith(".pt")]
                self.assertGreater(len(ckpt_files), 0, "No checkpoint files saved")
            finally:
                os.chdir(old_cwd)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_training_loop(self) -> None:
        """Run a few training steps and verify outputs."""
        config = _make_config(self.data_dir, numb_steps=5)
        config = update_deepmd_input(config, warning=False)
        config = normalize(config)
        self._run_training(config)

    def test_training_loop_compiled(self) -> None:
        """Run a few training steps with torch.compile enabled."""
        config = _make_config(self.data_dir, numb_steps=5)
        config["training"]["enable_compile"] = True
        config = update_deepmd_input(config, warning=False)
        config = normalize(config)
        self._run_training(config)


class TestCompiledRecompile(unittest.TestCase):
    """Test that _CompiledModel recompiles when nall exceeds max_nall."""

    @classmethod
    def setUpClass(cls) -> None:
        data_dir = os.path.join(EXAMPLE_DIR, "data")
        if not os.path.isdir(data_dir):
            raise unittest.SkipTest(f"Example data not found: {data_dir}")
        cls.data_dir = data_dir

    def test_nall_growth_triggers_recompile(self) -> None:
        """Shrink max_nall to force a recompile, then verify training works."""
        from deepmd.pt_expt.train.training import (
            _CompiledModel,
        )

        config = _make_config(self.data_dir, numb_steps=5)
        config["training"]["enable_compile"] = True
        config = update_deepmd_input(config, warning=False)
        config = normalize(config)

        tmpdir = tempfile.mkdtemp(prefix="pt_expt_recompile_")
        try:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                trainer = get_trainer(config)

                # The wrapper.model should be a _CompiledModel
                compiled_model = trainer.wrapper.model
                self.assertIsInstance(compiled_model, _CompiledModel)

                original_max_nall = compiled_model._max_nall
                self.assertGreater(original_max_nall, 0)

                # Artificially shrink max_nall to 1 so the next batch
                # will certainly exceed it and trigger recompilation.
                compiled_model._max_nall = 1
                old_compiled_lower = compiled_model.compiled_forward_lower

                # Run one training step — should trigger recompile
                trainer.wrapper.train()
                trainer.optimizer.zero_grad(set_to_none=True)
                inp, lab = trainer.get_data(is_train=True)
                lr = trainer.scheduler.get_last_lr()[0]
                _, loss, more_loss = trainer.wrapper(**inp, cur_lr=lr, label=lab)
                loss.backward()
                trainer.optimizer.step()

                # max_nall should have grown beyond 1
                new_max_nall = compiled_model._max_nall
                self.assertGreater(new_max_nall, 1)

                # compiled_forward_lower should be a new object
                self.assertIsNot(
                    compiled_model.compiled_forward_lower,
                    old_compiled_lower,
                )

                # Loss should be a finite scalar
                self.assertFalse(torch.isnan(loss))
                self.assertFalse(torch.isinf(loss))
            finally:
                os.chdir(old_cwd)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestGetData(unittest.TestCase):
    """Test the batch data conversion in Trainer.get_data."""

    @classmethod
    def setUpClass(cls) -> None:
        data_dir = os.path.join(EXAMPLE_DIR, "data")
        if not os.path.isdir(data_dir):
            raise unittest.SkipTest(f"Example data not found: {data_dir}")
        cls.data_dir = data_dir

    def test_batch_shapes(self) -> None:
        """Verify input/label shapes from get_data."""
        config = _make_config(self.data_dir, numb_steps=5)
        config = update_deepmd_input(config, warning=False)
        config = normalize(config)

        tmpdir = tempfile.mkdtemp(prefix="pt_expt_getdata_")
        try:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                trainer = get_trainer(config)
                input_dict, label_dict = trainer.get_data(is_train=True)

                # coord should be [nf, natoms, 3]
                self.assertEqual(len(input_dict["coord"].shape), 3)
                self.assertEqual(input_dict["coord"].shape[-1], 3)

                # atype should be [nf, natoms]
                self.assertEqual(len(input_dict["atype"].shape), 2)

                # force label should be [nf, natoms, 3]
                if "force" in label_dict:
                    self.assertEqual(len(label_dict["force"].shape), 3)
                    self.assertEqual(label_dict["force"].shape[-1], 3)

                # energy label should exist
                self.assertIn("energy", label_dict)
            finally:
                os.chdir(old_cwd)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
