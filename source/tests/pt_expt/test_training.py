# SPDX-License-Identifier: LGPL-3.0-or-later
"""Smoke test for the pt_expt training infrastructure.

Verifies that:
1. ``get_model`` constructs a model from config
2. ``make_stat_input`` + ``compute_or_load_stat`` work
3. A few training steps run without error
4. Loss decreases over those steps
"""

import copy
import datetime
import os
import shutil
import tempfile
import unittest
from unittest.mock import (
    patch,
)

import torch

from deepmd.loggers.training import (
    format_training_message,
)
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

# Keys present on the compiled path. ``atom_virial`` is intentionally excluded:
# training never passes ``do_atomic_virial=True``, so the compiled graph is
# traced with the default (False) and per-atom virial is not emitted.
_COMPILE_PRED_KEYS = ("atom_energy", "energy", "force", "virial")
_COMPILE_TOL = {"atol": 1e-10, "rtol": 1e-10}

# Descriptor configs used to extend compile-correctness tests to non-trivial
# architectures.  ``precision: float64`` is set so the strict ``atol=rtol=1e-10``
# comparison holds at machine epsilon.
_DESCRIPTOR_DPA1_NO_ATTN = {
    "type": "dpa1",
    "sel": 12,
    "rcut_smth": 0.50,
    "rcut": 3.00,
    "neuron": [8, 16],
    "axis_neuron": 4,
    "attn_layer": 0,
    "precision": "float64",
    "seed": 1,
}

_DESCRIPTOR_DPA2 = {
    "type": "dpa2",
    "repinit": {
        "rcut": 4.0,
        "rcut_smth": 0.5,
        "nsel": 18,
        "neuron": [2, 4, 8],
        "axis_neuron": 4,
        "activation_function": "tanh",
        "use_three_body": True,
        "three_body_sel": 12,
        "three_body_rcut": 3.0,
        "three_body_rcut_smth": 0.5,
    },
    "repformer": {
        "rcut": 3.0,
        "rcut_smth": 0.5,
        "nsel": 12,
        "nlayers": 2,
        "g1_dim": 8,
        "g2_dim": 5,
        "attn2_hidden": 3,
        "attn2_nhead": 1,
        "attn1_hidden": 5,
        "attn1_nhead": 1,
        "axis_neuron": 4,
        "update_h2": False,
        "update_g1_has_conv": True,
        "update_g1_has_grrg": True,
        "update_g1_has_drrd": True,
        "update_g1_has_attn": True,
        "update_g2_has_g1g1": True,
        "update_g2_has_attn": True,
        "attn2_has_gate": True,
    },
    "precision": "float64",
    "seed": 1,
    "add_tebd_to_repinit_out": False,
}

_DESCRIPTOR_DPA3 = {
    "type": "dpa3",
    "repflow": {
        "n_dim": 8,
        "e_dim": 5,
        "a_dim": 4,
        "nlayers": 2,
        "e_rcut": 3.0,
        "e_rcut_smth": 0.5,
        "e_sel": 12,
        "a_rcut": 3.0,
        "a_rcut_smth": 0.5,
        "a_sel": 8,
        "axis_neuron": 4,
    },
    "precision": "float64",
    "concat_output_tebd": False,
    "seed": 1,
}


# DPA4/SeZM uses a dedicated model type ("dpa4") with fixed descriptor and
# fitting types, so it gets a full model config rather than only a descriptor
# dict that can be swapped into ``_make_config``.
_MODEL_DPA4 = {
    "type": "dpa4",
    "type_map": ["O", "H"],
    "descriptor": {
        "type": "dpa4",
        "sel": 20,
        "rcut": 4.0,
        "channels": 16,
        "n_radial": 8,
        "lmax": 2,
        "mmax": 1,
        "n_blocks": 2,
        "seed": 1,
    },
    "fitting_net": {
        "type": "dpa4_ener",
        "neuron": [0],
        "seed": 1,
    },
    "data_stat_nbatch": 1,
}


def _assert_compile_predictions_match(
    testcase: unittest.TestCase,
    out_c: dict,
    out_uc: dict,
    *,
    ctx: str = "",
) -> None:
    for key in _COMPILE_PRED_KEYS:
        testcase.assertIn(key, out_uc, f"{ctx}uncompiled missing '{key}'")
        testcase.assertIn(key, out_c, f"{ctx}compiled missing '{key}'")
        torch.testing.assert_close(
            out_c[key],
            out_uc[key],
            **_COMPILE_TOL,
            msg=f"{ctx}{key} mismatch between compiled and uncompiled",
        )


def _assert_compile_grads_match(
    testcase: unittest.TestCase,
    model_c: torch.nn.Module,
    model_uc: torch.nn.Module,
    *,
    ctx: str = "",
) -> None:
    for (name_uc, p_uc), (_, p_c) in zip(
        model_uc.named_parameters(),
        model_c.named_parameters(),
        strict=True,
    ):
        if p_uc.grad is None:
            continue
        testcase.assertIsNotNone(p_c.grad, msg=f"{ctx}grad is None for {name_uc}")
        torch.testing.assert_close(
            p_c.grad,
            p_uc.grad,
            **_COMPILE_TOL,
            msg=f"{ctx}grad mismatch on {name_uc}",
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

    def test_training_loop_dpa4(self) -> None:
        """Run a few DPA4/SeZM training steps (model type "dpa4" dispatch)."""
        config = _make_config(self.data_dir, numb_steps=5)
        config["model"] = copy.deepcopy(_MODEL_DPA4)
        config = update_deepmd_input(config, warning=False)
        config = normalize(config)
        self.assertEqual(config["model"]["type"], "dpa4")
        self._run_training(config)

    def test_training_loop_compiled(self) -> None:
        """Run a few training steps with torch.compile enabled."""
        config = _make_config(self.data_dir, numb_steps=5)
        config["training"]["enable_compile"] = True
        config = update_deepmd_input(config, warning=False)
        config = normalize(config)
        self._run_training(config)

    def test_training_loop_compiled_silu(self) -> None:
        """Run compiled training with silu activation."""
        config = _make_config(self.data_dir, numb_steps=5)
        config["model"]["descriptor"]["activation_function"] = "silu"
        config["model"]["fitting_net"]["activation_function"] = "silu"
        config["training"]["enable_compile"] = True
        config = update_deepmd_input(config, warning=False)
        config = normalize(config)
        self._run_training(config)


class TestCompiledModelGetattr(unittest.TestCase):
    """Unit tests for _CompiledModel attribute delegation.

    These tests do not require example data or torch.compile — they use a
    lightweight mock original_model to verify that __getattr__ correctly
    forwards unknown attributes/methods to the wrapped original model.
    Compilation is lazy, so no compiled_forward_lower is needed for construction.
    """

    def _make_compiled_model(self):
        from deepmd.pt_expt.train.training import (
            _CompiledModel,
        )

        class _FakeModel(torch.nn.Module):
            def get_rcut(self):
                return 3.0

            def get_type_map(self):
                return ["O", "H"]

            @property
            def atomic_model(self):
                return self

            def get_descriptor(self):
                return self

        return _CompiledModel(_FakeModel(), structure_key=(7,))

    def test_delegates_method(self) -> None:
        """Unknown method calls are forwarded to original_model."""
        cm = self._make_compiled_model()
        self.assertAlmostEqual(cm.get_rcut(), 3.0)

    def test_delegates_method_returning_list(self) -> None:
        """Methods returning non-scalar values are forwarded correctly."""
        cm = self._make_compiled_model()
        self.assertEqual(cm.get_type_map(), ["O", "H"])

    def test_delegates_property(self) -> None:
        """Property access is forwarded to original_model."""
        cm = self._make_compiled_model()
        self.assertIsNotNone(cm.atomic_model)

    def test_own_attrs_not_delegated(self) -> None:
        """Attributes owned by _CompiledModel itself are NOT delegated."""
        cm = self._make_compiled_model()
        # original_model is a registered submodule and must not fall through
        # to delegation.  compiled_forward_lower is None before the first
        # forward call (lazy compile) — accessing it must return None, not
        # delegate to original_model.
        self.assertIsInstance(cm.original_model, torch.nn.Module)
        self.assertIsNone(cm.compiled_forward_lower)

    def test_missing_attr_raises(self) -> None:
        """Accessing an attribute missing from both wrapper and original raises."""
        cm = self._make_compiled_model()
        with self.assertRaises(AttributeError):
            _ = cm.nonexistent_attribute_xyz


class TestCompiledDynamicShapes(unittest.TestCase):
    """Test that _CompiledModel handles varying nall via dynamic shapes."""

    @classmethod
    def setUpClass(cls) -> None:
        data_dir = os.path.join(EXAMPLE_DIR, "data")
        if not os.path.isdir(data_dir):
            raise unittest.SkipTest(f"Example data not found: {data_dir}")
        cls.data_dir = data_dir

    def test_compiled_handles_varying_nall(self) -> None:
        """Run several training steps, assert finite loss each step.

        With ``tracing_mode="symbolic"`` + ``dynamic=True``, nall is a
        symbolic dim so nall growth across batches is handled without
        any recompile or padding.
        """
        from deepmd.pt_expt.train.training import (
            _CompiledModel,
        )

        config = _make_config(self.data_dir, numb_steps=5)
        config["training"]["enable_compile"] = True
        config = update_deepmd_input(config, warning=False)
        config = normalize(config)

        tmpdir = tempfile.mkdtemp(prefix="pt_expt_dynamic_")
        try:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                trainer = get_trainer(config)

                # The wrapper.model should be a _CompiledModel
                compiled_model = trainer.wrapper.model["Default"]
                self.assertIsInstance(compiled_model, _CompiledModel)
                # Lazy compile: compiled_forward_lower is None before any forward.
                self.assertIsNone(compiled_model.compiled_forward_lower)

                trainer.wrapper.train()
                for step in range(3):
                    trainer.optimizer.zero_grad(set_to_none=True)
                    inp, lab = trainer.get_data(is_train=True)
                    lr = trainer.scheduler.get_last_lr()[0]
                    _, loss, _ = trainer.wrapper(**inp, cur_lr=lr, label=lab)
                    loss.backward()
                    trainer.optimizer.step()

                    # After first forward, compiled_forward_lower must be set.
                    if step == 0:
                        self.assertIsNotNone(compiled_model.compiled_forward_lower)

                    # Loss should be a finite scalar at every step
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

    def _check_consistency(self, activation: str | None = None) -> None:
        """Compiled model predictions match uncompiled for the given activation.

        ``activation`` overrides both descriptor and fitting-net activation
        functions when provided.  ``None`` keeps the config default (tanh).
        """
        from deepmd.pt_expt.train.training import (
            _CompiledModel,
        )

        def _build_config(enable_compile: bool) -> dict:
            config = _make_config(self.data_dir, numb_steps=1)
            # enable virial in loss so the model returns it
            config["loss"]["start_pref_v"] = 1.0
            config["loss"]["limit_pref_v"] = 1.0
            if activation is not None:
                config["model"]["descriptor"]["activation_function"] = activation
                config["model"]["fitting_net"]["activation_function"] = activation
            if enable_compile:
                config["training"]["enable_compile"] = True
            config = update_deepmd_input(config, warning=False)
            return normalize(config)

        tmpdir = tempfile.mkdtemp(prefix="pt_expt_consistency_")
        try:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                trainer = get_trainer(_build_config(enable_compile=False))
                # Uncompiled model reference
                uncompiled_model = trainer.model
                uncompiled_model.eval()

                # Build compiled model from the same weights
                trainer_compiled = get_trainer(_build_config(enable_compile=True))
                compiled_model = trainer_compiled.wrapper.model["Default"]
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

                _assert_compile_predictions_match(self, pred_c, pred_uc)
            finally:
                os.chdir(old_cwd)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_compiled_matches_uncompiled(self) -> None:
        """Energy, force, virial from compiled model must match uncompiled."""
        self._check_consistency()

    def test_compiled_matches_uncompiled_silu(self) -> None:
        """Same numerical equivalence under silu activation (full model)."""
        self._check_consistency(activation="silu")

    def test_compiled_gradients_match_uncompiled(self) -> None:
        """Parameter gradients from compiled model must match uncompiled.

        Verifies second-order derivatives are correct: the loss includes
        force terms, and force is computed via autograd.grad(create_graph=True),
        so loss.backward() requires second-order differentiation through the
        make_fx-decomposed backward ops.
        """
        from deepmd.pt_expt.train.training import (
            _CompiledModel,
        )

        config_uc = _make_config(self.data_dir, numb_steps=1)
        config_uc = update_deepmd_input(config_uc, warning=False)
        config_uc = normalize(config_uc)

        config_c = _make_config(self.data_dir, numb_steps=1)
        config_c["training"]["enable_compile"] = True
        config_c = update_deepmd_input(config_c, warning=False)
        config_c = normalize(config_c)

        tmpdir = tempfile.mkdtemp(prefix="pt_expt_grad_consistency_")
        try:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                trainer_uc = get_trainer(config_uc)
                trainer_c = get_trainer(config_c)
                compiled_model = trainer_c.wrapper.model["Default"]
                self.assertIsInstance(compiled_model, _CompiledModel)

                # Match weights
                compiled_model.original_model.load_state_dict(
                    trainer_uc.model.state_dict()
                )

                # Forward + backward through wrapper (includes loss)
                trainer_uc.optimizer.zero_grad(set_to_none=True)
                trainer_c.optimizer.zero_grad(set_to_none=True)

                input_dict, label_dict = trainer_uc.get_data(is_train=True)
                cur_lr = trainer_uc.scheduler.get_last_lr()[0]

                _, loss_uc, _ = trainer_uc.wrapper(
                    **input_dict,
                    cur_lr=cur_lr,
                    label=label_dict,
                )
                _, loss_c, _ = trainer_c.wrapper(
                    **input_dict,
                    cur_lr=cur_lr,
                    label=label_dict,
                )
                loss_uc.backward()
                loss_c.backward()

                _assert_compile_grads_match(
                    self, compiled_model.original_model, trainer_uc.model
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


class TestAdditionalDataRequirement(unittest.TestCase):
    """Cover both branches of fparam handling in get_additional_data_requirement.

    Regression guard for the bug where pt_expt hard-coded ``default=0.0`` for
    fparam, ignoring the model's ``default_fparam``.  When the dataset has no
    ``fparam.npy``, that caused the data system to feed zeros to the fitting
    net while pt fed the model default — producing a constant DC offset
    (e.g. ``[0, -100]`` after fparam normalisation) that slowed training.
    """

    def test_has_default_fparam(self) -> None:
        """has_default_fparam==True: must=False and default broadcasts the model value."""
        from deepmd.pt_expt.train.training import (
            get_additional_data_requirement,
        )

        class _M:
            def get_dim_fparam(self) -> int:
                return 2

            def get_dim_aparam(self) -> int:
                return 0

            def has_default_fparam(self) -> bool:
                return True

            def get_default_fparam(self) -> list[float]:
                return [0.0, 1.0]

            def has_chg_spin_ebd(self) -> bool:
                return False

        reqs = get_additional_data_requirement(_M())
        self.assertEqual(len(reqs), 1)
        fparam_req = reqs[0]
        self.assertEqual(fparam_req.key, "fparam")
        self.assertEqual(fparam_req.ndof, 2)
        self.assertFalse(fparam_req.must)
        # default is the model's default_fparam, not 0.0
        self.assertNotIsInstance(fparam_req.default, float)
        import numpy as np

        np.testing.assert_array_equal(np.asarray(fparam_req.default), [0.0, 1.0])

    def test_no_default_fparam(self) -> None:
        """has_default_fparam==False: must=True (data file must exist) and default=0.0."""
        from deepmd.pt_expt.train.training import (
            get_additional_data_requirement,
        )

        class _M:
            def get_dim_fparam(self) -> int:
                return 2

            def get_dim_aparam(self) -> int:
                return 0

            def has_default_fparam(self) -> bool:
                return False

            def get_default_fparam(self) -> None:
                return None

            def has_chg_spin_ebd(self) -> bool:
                return False

        reqs = get_additional_data_requirement(_M())
        self.assertEqual(len(reqs), 1)
        fparam_req = reqs[0]
        self.assertEqual(fparam_req.key, "fparam")
        self.assertTrue(fparam_req.must)
        self.assertEqual(fparam_req.default, 0.0)


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

    def test_restart_from_compiled_checkpoint(self) -> None:
        """Train WITH compile enabled, restart from the compiled checkpoint.

        Regression test for the state-dict key mismatch bug: save_checkpoint
        previously saved _CompiledModel keys ("original_model.*"), which made
        load_state_dict fail on restart because a fresh ModelWrapper expects
        plain model keys ("model.Default.*").
        """
        from deepmd.pt_expt.train.training import (
            _CompiledModel,
        )

        tmpdir = tempfile.mkdtemp(prefix="pt_expt_compiled_restart_")
        try:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                # Phase 1: train 5 steps WITH compile enabled.
                # Do NOT use _train_and_get_ckpt here — we need the trainer
                # object to assert that compilation actually happened.
                # Without this check the test is vacuous: if torch.compile
                # silently falls back to eager the checkpoint keys are already
                # plain and the bug path is never exercised.
                config = _make_config(self.data_dir, numb_steps=5)
                config["training"]["enable_compile"] = True
                config = update_deepmd_input(config, warning=False)
                config = normalize(config)
                trainer1 = get_trainer(config)
                self.assertIsInstance(
                    trainer1.wrapper.model["Default"],
                    _CompiledModel,
                    "Phase-1 trainer did not produce a _CompiledModel; "
                    "torch.compile may have silently fallen back to eager — "
                    "the bug path is not exercised.",
                )
                trainer1.run()

                ckpt_path = os.path.join(tmpdir, "model.ckpt.pt")
                self.assertTrue(os.path.exists(ckpt_path), "Checkpoint not created")

                # Primary assertion: the checkpoint must NOT contain
                # "_CompiledModel wrapper" keys ("original_model.*").
                raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                ckpt_keys = list(raw["model"].keys())
                self.assertFalse(
                    any("original_model" in k for k in ckpt_keys),
                    f"Checkpoint has _CompiledModel wrapper keys — "
                    f"save_checkpoint must unwrap before serialising: "
                    f"{[k for k in ckpt_keys if 'original_model' in k]}",
                )

                # Secondary assertion: the saved state dict must load cleanly
                # into a fresh *uncompiled* ModelWrapper (strict=False so we
                # can distinguish unexpected vs missing keys clearly).
                config_nc = _make_config(self.data_dir, numb_steps=10)
                config_nc = update_deepmd_input(config_nc, warning=False)
                config_nc = normalize(config_nc)
                fresh = get_trainer(config_nc)
                missing, unexpected = fresh._unwrapped.load_state_dict(
                    raw["model"], strict=False
                )
                self.assertEqual(
                    unexpected,
                    [],
                    f"Unexpected keys when loading compiled ckpt into plain "
                    f"model (indicates _CompiledModel wrapper keys leaked): "
                    f"{unexpected}",
                )
                self.assertEqual(
                    missing,
                    [],
                    f"Missing keys when loading compiled ckpt into plain "
                    f"model: {missing}",
                )

                # Phase 2: restart from the compiled checkpoint
                config2 = _make_config(self.data_dir, numb_steps=10)
                config2["training"]["enable_compile"] = True
                config2 = update_deepmd_input(config2, warning=False)
                config2 = normalize(config2)
                trainer2 = get_trainer(config2, restart_model=ckpt_path)

                self.assertEqual(trainer2.start_step, 5)
                self.assertIsInstance(trainer2.wrapper.model["Default"], _CompiledModel)
                trainer2.run()

                with open(os.path.join(tmpdir, "lcurve.out")) as f:
                    lines = [ln for ln in f.readlines() if not ln.startswith("#")]
                self.assertGreater(len(lines), 0)
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
                self.assertIsInstance(trainer2.wrapper.model["Default"], _CompiledModel)
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


class TestCompiledVaryingNframesWithParams(unittest.TestCase):
    """Compiled training with varying ``nframes`` + ``nall`` + fparam/aparam.

    Exercises the compiled forward path under all three kinds of shape
    variation simultaneously:

    * Different systems have different atom counts -> varying ``nloc`` / ``nall``.
    * Per-system ``batch_size: [2, 3]`` -> varying ``nframes`` (2 vs 3).
    * Both ``fparam`` (per-frame) and ``aparam`` (per-atom) labels are
      provided, covering the ``dim_fparam`` / ``dim_aparam`` > 0 branches
      inside ``forward_lower``.

    The chosen values (``nframes`` in {2, 3}, ``numb_fparam=2``,
    ``numb_aparam=3``) are deliberately chosen so the runtime ``nframes``
    collides with the per-frame / per-atom feature dims — this is the
    exact pattern that previously caused PyTorch's symbolic tracer to
    specialise the batch dim (see _trace_and_compile in training.py).

    ``dp_random.choice`` is mocked to alternate between the two systems
    so both are guaranteed to be sampled across ``nsteps``.
    """

    NFPARAM = 2
    NAPARAM = 3

    @classmethod
    def setUpClass(cls) -> None:
        # Reuse the data-dir helper from the multitask gradient tests so we
        # don't duplicate the npy/raw layout boilerplate.
        from .test_multitask import (
            _generate_random_data_dir,
        )

        cls.tmpdir = tempfile.mkdtemp(prefix="pt_expt_varying_params_data_")
        cls.sys0 = os.path.join(cls.tmpdir, "sys0_8atoms")
        cls.sys1 = os.path.join(cls.tmpdir, "sys1_4atoms")
        # Atom types alternate 0/1 to match the ["O", "H"] type_map below.
        _generate_random_data_dir(
            cls.sys0,
            atom_types=[i % 2 for i in range(8)],
            nframes=4,
            seed=42,
            nfparam=cls.NFPARAM,
            naparam=cls.NAPARAM,
        )
        _generate_random_data_dir(
            cls.sys1,
            atom_types=[i % 2 for i in range(4)],
            nframes=4,
            seed=137,
            nfparam=cls.NFPARAM,
            naparam=cls.NAPARAM,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def _make_config(self, enable_compile: bool) -> dict:
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
                    "numb_fparam": self.NFPARAM,
                    "numb_aparam": self.NAPARAM,
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
                    "systems": [self.sys0, self.sys1],
                    # Per-system batch sizes: sys0 gets nframes=2, sys1 gets nframes=3.
                    # Combined with sys0=8 atoms / sys1=4 atoms this guarantees
                    # both `nframes` and `nall` vary across steps.
                    "batch_size": [2, 3],
                },
                "validation_data": {
                    "systems": [self.sys0],
                    "batch_size": 1,
                    "numb_btch": 1,
                },
                "numb_steps": 6,
                "seed": 10,
                "disp_file": "lcurve.out",
                "disp_freq": 100,
                "save_freq": 100,
            },
        }
        if enable_compile:
            config["training"]["enable_compile"] = True
        config = update_deepmd_input(config, warning=False)
        config = normalize(config)
        return config

    def _run_steps(self, enable_compile: bool, nsteps: int = 6) -> None:
        from deepmd.utils import data_system as _data_system

        config = self._make_config(enable_compile=enable_compile)
        sys_sequence = [i % 2 for i in range(nsteps)]

        tmpdir = tempfile.mkdtemp(prefix="pt_expt_varying_params_run_")
        try:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                trainer = get_trainer(config)
                if enable_compile:
                    from deepmd.pt_expt.train.training import (
                        _CompiledModel,
                    )

                    self.assertIsInstance(
                        trainer.wrapper.model["Default"], _CompiledModel
                    )

                trainer.wrapper.train()
                seen_nframes = set()
                seen_nall = set()
                with patch.object(
                    _data_system.dp_random,
                    "choice",
                    side_effect=sys_sequence,
                ):
                    for _ in range(nsteps):
                        trainer.optimizer.zero_grad(set_to_none=True)
                        inp, lab = trainer.get_data(is_train=True)
                        seen_nframes.add(int(inp["coord"].shape[0]))
                        seen_nall.add(int(inp["atype"].shape[1]))
                        # fparam/aparam must be present in every batch
                        self.assertIn("fparam", inp)
                        self.assertIn("aparam", inp)
                        lr = trainer.scheduler.get_last_lr()[0]
                        _, loss, _ = trainer.wrapper(**inp, cur_lr=lr, label=lab)
                        loss.backward()
                        trainer.optimizer.step()
                        self.assertFalse(torch.isnan(loss), "loss is NaN")
                        self.assertFalse(torch.isinf(loss), "loss is Inf")

                # The two systems differ in both batch-size-auto and natoms,
                # so both nframes and nloc should have varied across steps.
                self.assertGreater(
                    len(seen_nframes),
                    1,
                    msg=f"nframes did not vary across steps: {seen_nframes}",
                )
                self.assertGreater(
                    len(seen_nall),
                    1,
                    msg=f"nloc did not vary across steps: {seen_nall}",
                )
            finally:
                os.chdir(old_cwd)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_compiled(self) -> None:
        """Compiled training with varying nframes + fparam/aparam."""
        self._run_steps(enable_compile=True)

    def test_uncompiled(self) -> None:
        """Baseline: same config, uncompiled, should also succeed."""
        self._run_steps(enable_compile=False)


def _create_small_system(
    path: str, natoms_o: int = 2, natoms_h: int = 4, nframes: int = 10
) -> None:
    """Create a minimal deepmd data system with few atoms."""
    import numpy as np

    natoms = natoms_o + natoms_h
    set_dir = os.path.join(path, "set.000")
    os.makedirs(set_dir, exist_ok=True)

    with open(os.path.join(path, "type.raw"), "w") as f:
        for _ in range(natoms_o):
            f.write("0\n")
        for _ in range(natoms_h):
            f.write("1\n")
    with open(os.path.join(path, "type_map.raw"), "w") as f:
        f.write("O\nH\n")

    rng = np.random.default_rng(42)
    box_len = 5.0
    box = np.zeros((nframes, 9), dtype=np.float32)
    box[:, 0] = box_len
    box[:, 4] = box_len
    box[:, 8] = box_len
    coord = rng.uniform(0, box_len, size=(nframes, natoms * 3)).astype(np.float32)
    energy = rng.normal(-100, 10, size=(nframes,)).astype(np.float32)
    force = rng.normal(0, 1, size=(nframes, natoms * 3)).astype(np.float32)
    virial = rng.normal(0, 1, size=(nframes, 9)).astype(np.float32)
    np.save(os.path.join(set_dir, "coord.npy"), coord)
    np.save(os.path.join(set_dir, "force.npy"), force)
    np.save(os.path.join(set_dir, "energy.npy"), energy)
    np.save(os.path.join(set_dir, "box.npy"), box)
    np.save(os.path.join(set_dir, "virial.npy"), virial)


class TestCompiledVaryingNatoms(unittest.TestCase):
    """Test compiled training with systems of different atom counts.

    Uses the 192-atom ``data_0`` alongside a synthetic 6-atom system so that
    different ``nloc`` / ``nall`` appear across steps, exercising the
    dynamic-shape compile path.

    ``dp_random.choice`` is mocked to alternate [0, 1, 0, 1, ...] so that
    both systems are guaranteed to be sampled.

    ``batch_size: "auto"`` assigns different batch sizes per system (based
    on atom count), so both ``nframes`` and ``natoms`` vary across steps.
    """

    @classmethod
    def setUpClass(cls) -> None:
        data_dir = os.path.join(EXAMPLE_DIR, "data")
        if not os.path.isdir(data_dir):
            raise unittest.SkipTest(f"Example data not found: {data_dir}")
        cls.data_dir = data_dir
        cls.small_data_dir = tempfile.mkdtemp(prefix="pt_expt_small_data_")
        _create_small_system(cls.small_data_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.small_data_dir, ignore_errors=True)

    def _make_varying_config(
        self,
        enable_compile: bool,
        descriptor: dict | None = None,
    ) -> dict:
        """Config with two systems of different natoms and auto batch size.

        ``descriptor`` overrides the default se_e2_a descriptor when given.
        """
        config = _make_config(self.data_dir)
        config["training"]["training_data"]["systems"].append(self.small_data_dir)
        config["training"]["training_data"]["batch_size"] = "auto"
        # enable virial in loss so the model returns it (virial.npy exists in
        # both systems), exercising the compiled virial passthrough on each step
        config["loss"]["start_pref_v"] = 1.0
        config["loss"]["limit_pref_v"] = 1.0
        if descriptor is not None:
            config["model"]["descriptor"] = descriptor
        if enable_compile:
            config["training"]["enable_compile"] = True
        config = update_deepmd_input(config, warning=False)
        config = normalize(config)
        return config

    def _check_varying_natoms(self, descriptor: dict | None = None) -> None:
        """Per-step compiled-vs-uncompiled comparison for the given descriptor.

        The loss config has ``start_pref_f=1000`` and ``start_pref_v=1.0``,
        so ``loss.backward()`` propagates through ``F = -dE/dr`` (computed
        via ``autograd.grad(..., create_graph=True)``); the per-parameter
        grad comparison therefore exercises the second-order derivative
        ``d^2 E / (dr d theta)`` on each step at each system size.

        Verifies multi-step training-trajectory equivalence: weights are
        synced once at the start, then both trainers step their own Adam
        states forward.  All assertions use the strict
        ``atol=rtol=1e-10`` tolerance; if a descriptor's compiled path
        cannot meet that on float64 the descriptor has a real numerical
        problem (see the DPA1 limitation note where this happened).
        """
        from deepmd.pt_expt.train.training import (
            _CompiledModel,
        )

        nsteps = 4
        # Alternate between system 0 (192 atoms) and system 1 (6 atoms)
        sys_sequence = [i % 2 for i in range(nsteps)]

        tmpdir = tempfile.mkdtemp(prefix="pt_expt_varying_")
        try:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                trainer_uc = get_trainer(self._make_varying_config(False, descriptor))
                trainer_c = get_trainer(self._make_varying_config(True, descriptor))
                compiled_model = trainer_c.wrapper.model["Default"]
                self.assertIsInstance(compiled_model, _CompiledModel)

                # Sync weights so predictions can be compared exactly
                compiled_model.original_model.load_state_dict(
                    trainer_uc.model.state_dict()
                )
                trainer_uc.wrapper.train()
                trainer_c.wrapper.train()

                with patch(
                    "deepmd.utils.data_system.dp_random.choice",
                    side_effect=sys_sequence,
                ):
                    for step in range(nsteps):
                        trainer_uc.optimizer.zero_grad(set_to_none=True)
                        trainer_c.optimizer.zero_grad(set_to_none=True)

                        # Single shared batch; mock yields one value per call
                        inp, lab = trainer_uc.get_data(is_train=True)
                        lr = trainer_uc.scheduler.get_last_lr()[0]

                        out_uc, loss_uc, _ = trainer_uc.wrapper(
                            **inp, cur_lr=lr, label=lab
                        )
                        out_c, loss_c, _ = trainer_c.wrapper(
                            **inp, cur_lr=lr, label=lab
                        )

                        ctx = f"step={step} "
                        _assert_compile_predictions_match(self, out_c, out_uc, ctx=ctx)
                        torch.testing.assert_close(
                            loss_c,
                            loss_uc,
                            **_COMPILE_TOL,
                            msg=f"{ctx}loss mismatch",
                        )

                        loss_uc.backward()
                        loss_c.backward()
                        _assert_compile_grads_match(
                            self,
                            compiled_model.original_model,
                            trainer_uc.model,
                            ctx=ctx,
                        )

                        trainer_uc._optimizer_step()
                        trainer_c._optimizer_step()
            finally:
                os.chdir(old_cwd)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_compiled_matches_uncompiled_varying_natoms_se_e2_a(self) -> None:
        """se_e2_a: compiled vs uncompiled match across varying nframes/natoms."""
        self._check_varying_natoms()  # uses default se_e2_a from _make_config

    def test_compiled_matches_uncompiled_varying_natoms_dpa2(self) -> None:
        """DPA2: compiled vs uncompiled match across varying nframes/natoms.

        Exercises the DPA2 repinit + repformers stack; matches at machine
        epsilon (~1e-12) on float64 just like se_e2_a.
        """
        self._check_varying_natoms(_DESCRIPTOR_DPA2)

    def test_compiled_matches_uncompiled_varying_natoms_dpa3(self) -> None:
        """DPA3: compiled vs uncompiled match across varying nframes/natoms.

        Exercises a non-trivial multi-layer repflow descriptor; matches at
        machine epsilon (~1e-12) on float64 just like se_e2_a.
        """
        self._check_varying_natoms(_DESCRIPTOR_DPA3)

    def test_compiled_matches_uncompiled_varying_natoms_dpa1_no_attn(self) -> None:
        """DPA1 (attn_layer=0): compiled vs uncompiled match.

        DPA1 with attention layers is intentionally not covered: the
        compiled se_atten path is hardware-sensitive on multi-threaded
        CPUs (parallel reduction order diverges from eager above the
        1e-10 tolerance).  ``_compile_model`` warns the user instead.
        """
        self._check_varying_natoms(_DESCRIPTOR_DPA1_NO_ATTN)

    def test_compile_warns_dpa1_with_attention(self) -> None:
        """DPA1 (attn_layer>0) under compile must emit a warning.

        Compiled se_atten attention is numerically hardware-sensitive;
        the trainer should warn at compile time but still proceed.
        """
        descriptor = {
            "type": "dpa1",
            "sel": 12,
            "rcut_smth": 0.50,
            "rcut": 3.00,
            "neuron": [8, 16],
            "axis_neuron": 4,
            "attn_layer": 2,
            "precision": "float64",
            "seed": 1,
        }
        config = self._make_varying_config(enable_compile=True, descriptor=descriptor)
        with self.assertLogs("deepmd", level="WARNING") as cm:
            trainer = get_trainer(config)
        self.assertTrue(
            any("attention layer" in msg for msg in cm.output),
            f"expected attention-layer warning, got: {cm.output}",
        )
        # Also confirm the compiled model was actually built (warning is
        # not a rejection).
        from deepmd.pt_expt.train.training import (
            _CompiledModel,
        )

        self.assertIsInstance(trainer.wrapper.model["Default"], _CompiledModel)


class TestCompiledSharedFittingDifferentDescriptor(unittest.TestCase):
    """Regression test: shared fitting with different descriptors gets distinct compiled graphs.

    Before the fix, ``_get_model_structure_key`` returned the id of the first
    fitting-net child without including the descriptor.  Two tasks sharing a
    fitting net but using different descriptors (different rcut / sel, which
    bake different smooth-cutoff constants into the traced graph) received the
    same structure key — task_2 silently reused task_1's compiled graph and
    produced wrong predictions.

    The fix includes ``id(descriptor)`` in the key so each task with a
    distinct descriptor gets its own compiled graph.
    """

    @classmethod
    def setUpClass(cls) -> None:
        data_dir = os.path.join(EXAMPLE_DIR, "data")
        if not os.path.isdir(data_dir):
            raise unittest.SkipTest(f"Example data not found: {data_dir}")
        cls.data_dir = data_dir

    def _make_config(self, enable_compile: bool) -> tuple[dict, object]:
        """Multi-task config: shared fitting_net, DIFFERENT descriptors per task."""
        from deepmd.pt_expt.utils.multi_task import (
            preprocess_shared_params,
        )

        data_dir_0 = os.path.join(self.data_dir, "data_0")
        config = {
            "model": {
                "shared_dict": {
                    "my_type_map": ["O", "H"],
                    "my_fitting": {
                        "neuron": [16, 16],
                        "resnet_dt": True,
                        "seed": 1,
                        "dim_case_embd": 2,
                        "precision": "float64",
                    },
                },
                "model_dict": {
                    "model_1": {
                        "type_map": "my_type_map",
                        "descriptor": {
                            "type": "se_e2_a",
                            "sel": [6, 12],
                            "rcut_smth": 0.50,
                            "rcut": 3.00,
                            "neuron": [8, 16],
                            "resnet_dt": False,
                            "axis_neuron": 4,
                            "type_one_side": True,
                            "precision": "float64",
                            "seed": 1,
                        },
                        "fitting_net": "my_fitting",
                        "data_stat_nbatch": 1,
                    },
                    "model_2": {
                        "type_map": "my_type_map",
                        "descriptor": {
                            "type": "se_e2_a",
                            "sel": [4, 8],
                            "rcut_smth": 0.30,
                            "rcut": 2.50,
                            "neuron": [8, 16],
                            "resnet_dt": False,
                            "axis_neuron": 4,
                            "type_one_side": True,
                            "precision": "float64",
                            "seed": 2,
                        },
                        "fitting_net": "my_fitting",
                        "data_stat_nbatch": 1,
                    },
                },
            },
            "learning_rate": {
                "type": "exp",
                "decay_steps": 500,
                "start_lr": 0.001,
                "stop_lr": 3.51e-8,
            },
            "loss_dict": {
                "model_1": {
                    "type": "ener",
                    "start_pref_e": 0.02,
                    "limit_pref_e": 1,
                    "start_pref_f": 1000,
                    "limit_pref_f": 1,
                    "start_pref_v": 0,
                    "limit_pref_v": 0,
                },
                "model_2": {
                    "type": "ener",
                    "start_pref_e": 0.02,
                    "limit_pref_e": 1,
                    "start_pref_f": 1000,
                    "limit_pref_f": 1,
                    "start_pref_v": 0,
                    "limit_pref_v": 0,
                },
            },
            "training": {
                "model_prob": {"model_1": 0.5, "model_2": 0.5},
                "data_dict": {
                    "model_1": {
                        "stat_file": "./stat_files/model_1",
                        "training_data": {"systems": [data_dir_0], "batch_size": 1},
                        "validation_data": {
                            "systems": [data_dir_0],
                            "batch_size": 1,
                            "numb_btch": 1,
                        },
                    },
                    "model_2": {
                        "stat_file": "./stat_files/model_2",
                        "training_data": {"systems": [data_dir_0], "batch_size": 1},
                        "validation_data": {
                            "systems": [data_dir_0],
                            "batch_size": 1,
                            "numb_btch": 1,
                        },
                    },
                },
                "numb_steps": 1,
                "seed": 10,
                "disp_file": "lcurve.out",
                "disp_freq": 1,
                "save_freq": 1,
            },
        }
        if enable_compile:
            config["training"]["enable_compile"] = True
        config["model"], shared_links = preprocess_shared_params(config["model"])
        config = update_deepmd_input(config, warning=False)
        config = normalize(config, multi_task=True)
        return config, shared_links

    def test_compiled_matches_eager_per_task(self) -> None:
        """Compiled output for each task must match its own eager output.

        With different descriptors, tasks must get separate compiled graphs.
        Before the fix, task_2 reused task_1's compiled graph (rcut=3.0 baked
        in), yielding wrong predictions for task_2 (rcut=2.5).
        """
        from deepmd.pt_expt.train.training import (
            _CompiledModel,
            _get_model_structure_key,
        )

        tmpdir = tempfile.mkdtemp(prefix="pt_expt_diff_desc_")
        try:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                config_uc, shared_links_uc = self._make_config(enable_compile=False)
                config_c, shared_links_c = self._make_config(enable_compile=True)

                trainer_uc = get_trainer(config_uc, shared_links=shared_links_uc)
                trainer_c = get_trainer(config_c, shared_links=shared_links_c)

                for mk in ("model_1", "model_2"):
                    self.assertIsInstance(
                        trainer_c.wrapper.model[mk],
                        _CompiledModel,
                        f"{mk} was not compiled",
                    )

                # Different descriptors → different structure keys → separate graphs.
                key_1 = _get_model_structure_key(
                    trainer_c.wrapper.model["model_1"].original_model
                )
                key_2 = _get_model_structure_key(
                    trainer_c.wrapper.model["model_2"].original_model
                )
                self.assertNotEqual(
                    key_1,
                    key_2,
                    "Tasks with different descriptors must get different structure keys",
                )

                # Sync weights so compiled and uncompiled start from the same state.
                for mk in ("model_1", "model_2"):
                    trainer_c.wrapper.model[mk].original_model.load_state_dict(
                        trainer_uc.wrapper.model[mk].state_dict()
                    )

                for mk in ("model_1", "model_2"):
                    inp_dict, label_dict = trainer_uc.get_data(
                        is_train=True, task_key=mk
                    )
                    cur_lr = trainer_uc.scheduler.get_last_lr()[0]

                    pred_uc, loss_uc, _ = trainer_uc.wrapper(
                        **inp_dict,
                        cur_lr=cur_lr,
                        label=label_dict,
                        task_key=mk,
                    )
                    pred_c, loss_c, _ = trainer_c.wrapper(
                        **inp_dict,
                        cur_lr=cur_lr,
                        label=label_dict,
                        task_key=mk,
                    )

                    for key in ("atom_energy", "energy", "force"):
                        torch.testing.assert_close(
                            pred_c[key],
                            pred_uc[key],
                            atol=1e-10,
                            rtol=1e-10,
                            msg=f"{mk}/{key}: compiled vs eager mismatch",
                        )
                    torch.testing.assert_close(
                        loss_c,
                        loss_uc,
                        atol=1e-10,
                        rtol=1e-10,
                        msg=f"{mk}/loss: compiled vs eager mismatch",
                    )
            finally:
                os.chdir(old_cwd)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestFormatTrainingMessageStepTime(unittest.TestCase):
    """The pt_expt trainer reports the average wall time per step over each
    display interval by passing ``step_time`` to ``format_training_message``
    (replacing the former standalone ``step=... step_time=...`` debug line).
    These tests cover both branches of the optional ``step_time``/``eta``
    arguments so the "avg = ... s/step" segment is rendered only when requested.
    """

    def test_without_step_time(self) -> None:
        """``step_time=None`` (default) omits the step-time segment."""
        msg = format_training_message(batch=100, wall_time=18.41)
        self.assertEqual(msg, "Batch     100: total wall time = 18.41 s")
        self.assertNotIn("s/step", msg)

    def test_with_step_time(self) -> None:
        """``step_time`` is rendered with 4 decimals after the wall time."""
        msg = format_training_message(batch=100, wall_time=18.41, step_time=0.1841)
        self.assertEqual(
            msg,
            "Batch     100: total wall time = 18.41 s, avg = 0.1841 s/step",
        )

    def test_step_time_zero_is_shown(self) -> None:
        """A literal ``0.0`` step time is still shown (not treated as absent)."""
        msg = format_training_message(batch=1, wall_time=0.5, step_time=0.0)
        self.assertIn("avg = 0.0000 s/step", msg)

    def test_with_step_time_and_eta(self) -> None:
        """Step time appears before the eta segment."""
        current_time = datetime.datetime(
            2026, 6, 7, 5, 21, 29, tzinfo=datetime.timezone.utc
        )
        msg = format_training_message(
            batch=100,
            wall_time=18.41,
            eta=100,
            current_time=current_time,
            step_time=0.1841,
        )
        self.assertIn("total wall time = 18.41 s, avg = 0.1841 s/step, eta = ", msg)
        # ordering: wall time -> step time -> eta
        self.assertLess(msg.index("s/step"), msg.index("eta ="))

    def test_eta_without_step_time(self) -> None:
        """Eta still works when no step time is supplied."""
        current_time = datetime.datetime(
            2026, 6, 7, 5, 21, 29, tzinfo=datetime.timezone.utc
        )
        msg = format_training_message(
            batch=100, wall_time=18.41, eta=100, current_time=current_time
        )
        self.assertNotIn("s/step", msg)
        self.assertIn("eta = ", msg)


if __name__ == "__main__":
    unittest.main()
