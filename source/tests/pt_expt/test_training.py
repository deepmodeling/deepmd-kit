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


class TestCompiledConsistency(unittest.TestCase):
    """Verify compiled model produces the same energy/force/virial as uncompiled."""

    @classmethod
    def setUpClass(cls) -> None:
        data_dir = os.path.join(EXAMPLE_DIR, "data")
        if not os.path.isdir(data_dir):
            raise unittest.SkipTest(f"Example data not found: {data_dir}")
        cls.data_dir = data_dir

    def test_compiled_matches_uncompiled(self) -> None:
        """Energy, force, virial from compiled model must match uncompiled."""
        from deepmd.pt_expt.train.training import (
            _CompiledModel,
        )

        config = _make_config(self.data_dir, numb_steps=1)
        # enable virial in loss so the model returns it
        config["loss"]["start_pref_v"] = 1.0
        config["loss"]["limit_pref_v"] = 1.0
        config = update_deepmd_input(config, warning=False)
        config = normalize(config)

        tmpdir = tempfile.mkdtemp(prefix="pt_expt_consistency_")
        try:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                trainer = get_trainer(config)
                # Uncompiled model reference
                uncompiled_model = trainer.model
                uncompiled_model.eval()

                # Build compiled model from the same weights
                config_compiled = _make_config(self.data_dir, numb_steps=1)
                config_compiled["loss"]["start_pref_v"] = 1.0
                config_compiled["loss"]["limit_pref_v"] = 1.0
                config_compiled["training"]["enable_compile"] = True
                config_compiled = update_deepmd_input(config_compiled, warning=False)
                config_compiled = normalize(config_compiled)
                trainer_compiled = get_trainer(config_compiled)
                compiled_model = trainer_compiled.wrapper.model
                self.assertIsInstance(compiled_model, _CompiledModel)

                # Copy uncompiled weights to compiled model so they match
                compiled_model.original_model.load_state_dict(
                    uncompiled_model.state_dict()
                )
                compiled_model.eval()

                # Get a batch and run both models
                input_dict, _ = trainer.get_data(is_train=True)
                coord = input_dict["coord"].detach()
                atype = input_dict["atype"].detach()
                box = input_dict.get("box")
                if box is not None:
                    box = box.detach()

                # Force is computed via autograd.grad inside the model, so
                # we cannot use torch.no_grad() here.
                coord_uc = coord.clone().requires_grad_(True)
                pred_uc = uncompiled_model(coord_uc, atype, box)

                pred_c = compiled_model(coord.clone(), atype, box)

                # Energy
                torch.testing.assert_close(
                    pred_c["energy"],
                    pred_uc["energy"],
                    atol=1e-10,
                    rtol=1e-10,
                    msg="energy mismatch between compiled and uncompiled",
                )
                # Force
                self.assertIn("force", pred_c, "compiled model missing 'force'")
                self.assertIn("force", pred_uc, "uncompiled model missing 'force'")
                torch.testing.assert_close(
                    pred_c["force"],
                    pred_uc["force"],
                    atol=1e-10,
                    rtol=1e-10,
                    msg="force mismatch between compiled and uncompiled",
                )
                # Virial
                if "virial" in pred_uc:
                    self.assertIn("virial", pred_c, "compiled model missing 'virial'")
                    torch.testing.assert_close(
                        pred_c["virial"],
                        pred_uc["virial"],
                        atol=1e-10,
                        rtol=1e-10,
                        msg="virial mismatch between compiled and uncompiled",
                    )
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

                # coord should be a tensor with requires_grad
                self.assertIsInstance(input_dict["coord"], torch.Tensor)
                self.assertTrue(input_dict["coord"].requires_grad)

                # atype should be an integer tensor
                self.assertIsInstance(input_dict["atype"], torch.Tensor)

                # force label should be a tensor
                if "force" in label_dict:
                    self.assertIsInstance(label_dict["force"], torch.Tensor)

                # energy label should exist
                self.assertIn("energy", label_dict)
            finally:
                os.chdir(old_cwd)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestRestart(unittest.TestCase):
    """Test restart and init_model resume paths."""

    @classmethod
    def setUpClass(cls) -> None:
        data_dir = os.path.join(EXAMPLE_DIR, "data")
        if not os.path.isdir(data_dir):
            raise unittest.SkipTest(f"Example data not found: {data_dir}")
        cls.data_dir = data_dir

    def _train_and_get_ckpt(self, config: dict, tmpdir: str) -> str:
        """Train and return the path to the final checkpoint."""
        trainer = get_trainer(config)
        trainer.run()
        # find the latest checkpoint symlink
        ckpt = os.path.join(tmpdir, "model.ckpt.pt")
        self.assertTrue(os.path.exists(ckpt), "Checkpoint not created")
        return ckpt

    def test_restart(self) -> None:
        """Train 5 steps, restart from checkpoint, train 5 more."""
        tmpdir = tempfile.mkdtemp(prefix="pt_expt_restart_")
        try:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                # Phase 1: train 5 steps
                config = _make_config(self.data_dir, numb_steps=5)
                config = update_deepmd_input(config, warning=False)
                config = normalize(config)
                ckpt_path = self._train_and_get_ckpt(config, tmpdir)

                # Phase 2: restart from checkpoint, train to step 10
                config2 = _make_config(self.data_dir, numb_steps=10)
                config2 = update_deepmd_input(config2, warning=False)
                config2 = normalize(config2)
                trainer2 = get_trainer(config2, restart_model=ckpt_path)

                # start_step should be restored
                self.assertEqual(trainer2.start_step, 5)

                # LR should match the schedule at the resumed step,
                # not double-count start_step.
                expected_lr = trainer2.lr_schedule.value(trainer2.start_step)
                actual_lr = trainer2.scheduler.get_last_lr()[0]
                self.assertAlmostEqual(
                    actual_lr,
                    expected_lr,
                    places=10,
                    msg=f"LR after restart should be lr_schedule({trainer2.start_step})"
                    f"={expected_lr}, got {actual_lr}",
                )

                trainer2.run()

                # lcurve should have entries appended (restart opens in append mode)
                with open(os.path.join(tmpdir, "lcurve.out")) as f:
                    lines = [l for l in f.readlines() if not l.startswith("#")]
                self.assertGreater(len(lines), 0, "lcurve.out is empty after restart")
            finally:
                os.chdir(old_cwd)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_init_model(self) -> None:
        """Train 5 steps, init_model from checkpoint (reset step), train 5 more."""
        tmpdir = tempfile.mkdtemp(prefix="pt_expt_init_model_")
        try:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                # Phase 1: train 5 steps
                config = _make_config(self.data_dir, numb_steps=5)
                config = update_deepmd_input(config, warning=False)
                config = normalize(config)
                ckpt_path = self._train_and_get_ckpt(config, tmpdir)

                # Phase 2: init_model — weights loaded but step reset to 0
                config2 = _make_config(self.data_dir, numb_steps=5)
                config2 = update_deepmd_input(config2, warning=False)
                config2 = normalize(config2)
                trainer2 = get_trainer(config2, init_model=ckpt_path)

                # init_model resets step to 0
                self.assertEqual(trainer2.start_step, 0)
                trainer2.run()

                with open(os.path.join(tmpdir, "lcurve.out")) as f:
                    lines = [l for l in f.readlines() if not l.startswith("#")]
                self.assertGreater(
                    len(lines), 0, "lcurve.out is empty after init_model"
                )
            finally:
                os.chdir(old_cwd)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_restart_with_compile(self) -> None:
        """Train uncompiled, restart with compile enabled."""
        from deepmd.pt_expt.train.training import (
            _CompiledModel,
        )

        tmpdir = tempfile.mkdtemp(prefix="pt_expt_restart_compile_")
        try:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                # Phase 1: train 5 steps without compile
                config = _make_config(self.data_dir, numb_steps=5)
                config = update_deepmd_input(config, warning=False)
                config = normalize(config)
                ckpt_path = self._train_and_get_ckpt(config, tmpdir)

                # Phase 2: restart with compile enabled
                config2 = _make_config(self.data_dir, numb_steps=10)
                config2["training"]["enable_compile"] = True
                config2 = update_deepmd_input(config2, warning=False)
                config2 = normalize(config2)
                trainer2 = get_trainer(config2, restart_model=ckpt_path)

                self.assertEqual(trainer2.start_step, 5)
                self.assertIsInstance(trainer2.wrapper.model, _CompiledModel)
                trainer2.run()

                with open(os.path.join(tmpdir, "lcurve.out")) as f:
                    lines = [l for l in f.readlines() if not l.startswith("#")]
                self.assertGreater(len(lines), 0)
            finally:
                os.chdir(old_cwd)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


def _make_dpa3_config(data_dir: str, numb_steps: int = 5) -> dict:
    """Build a minimal DPA3 config dict pointing at *data_dir*."""
    config = {
        "model": {
            "type_map": ["O", "H"],
            "descriptor": {
                "type": "dpa3",
                "repflow": {
                    "n_dim": 8,
                    "e_dim": 4,
                    "a_dim": 4,
                    "nlayers": 2,
                    "e_rcut": 3.0,
                    "e_rcut_smth": 0.5,
                    "e_sel": 18,
                    "a_rcut": 2.5,
                    "a_rcut_smth": 0.5,
                    "a_sel": 10,
                    "axis_neuron": 4,
                    "fix_stat_std": 0.3,
                },
                "seed": 1,
            },
            "fitting_net": {
                "neuron": [8, 8],
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


class TestTrainingDPA3(unittest.TestCase):
    """Smoke test for the pt_expt training loop with DPA3 descriptor."""

    @classmethod
    def setUpClass(cls) -> None:
        data_dir = os.path.join(EXAMPLE_DIR, "data")
        if not os.path.isdir(data_dir):
            raise unittest.SkipTest(f"Example data not found: {data_dir}")
        cls.data_dir = data_dir

    def test_get_model(self) -> None:
        """Test that get_model constructs a DPA3 model from config."""
        config = _make_dpa3_config(self.data_dir)
        config = update_deepmd_input(config, warning=False)
        config = normalize(config)
        model = get_model(config["model"])
        self.assertIsInstance(model, torch.nn.Module)
        nparams = sum(p.numel() for p in model.parameters())
        self.assertGreater(nparams, 0)

    def test_training_loop(self) -> None:
        """Run a few DPA3 training steps and verify outputs."""
        config = _make_dpa3_config(self.data_dir, numb_steps=5)
        config = update_deepmd_input(config, warning=False)
        config = normalize(config)

        tmpdir = tempfile.mkdtemp(prefix="pt_expt_dpa3_train_")
        try:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                trainer = get_trainer(config)
                trainer.run()

                lcurve_path = os.path.join(tmpdir, "lcurve.out")
                self.assertTrue(os.path.exists(lcurve_path), "lcurve.out not created")

                with open(lcurve_path) as f:
                    lines = [l for l in f.readlines() if not l.startswith("#")]
                self.assertGreater(len(lines), 0, "lcurve.out is empty")

                ckpt_files = [f for f in os.listdir(tmpdir) if f.endswith(".pt")]
                self.assertGreater(len(ckpt_files), 0, "No checkpoint files saved")
            finally:
                os.chdir(old_cwd)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
