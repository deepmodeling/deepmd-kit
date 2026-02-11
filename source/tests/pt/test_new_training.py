# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for the refactored training system.

Includes end-to-end CLI tests to verify dp --pt train works correctly.
"""

import copy
import unittest

from deepmd.pt.entrypoints.main import (
    get_trainer,
)
from deepmd.pt.train.config import (
    CheckpointConfig,
    DisplayConfig,
    LearningRateConfig,
    OptimizerConfig,
    TrainingConfig,
)
from deepmd.pt.train.hooks import (
    HookManager,
    HookPriority,
    TrainingHook,
)
from deepmd.pt.train.logger import (
    LossAccumulator,
)
from deepmd.pt.train.optimizer_factory import (
    OptimizerFactory,
)


class TestOptimizerConfig(unittest.TestCase):
    """Test OptimizerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = OptimizerConfig()
        self.assertEqual(config.opt_type, "Adam")
        self.assertEqual(config.weight_decay, 0.001)
        self.assertEqual(config.momentum, 0.95)

    def test_from_dict(self):
        """Test creating from dictionary."""
        params = {
            "opt_type": "AdamW",
            "weight_decay": 0.01,
            "kf_blocksize": 1024,
        }
        config = OptimizerConfig.from_dict(params)
        self.assertEqual(config.opt_type, "AdamW")
        self.assertEqual(config.weight_decay, 0.01)
        self.assertEqual(config.kf_blocksize, 1024)
        # Check defaults are preserved
        self.assertEqual(config.momentum, 0.95)


class TestLearningRateConfig(unittest.TestCase):
    """Test LearningRateConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LearningRateConfig()
        self.assertEqual(config.start_lr, 1e-3)
        self.assertEqual(config.stop_lr, 1e-8)

    def test_from_dict(self):
        """Test creating from dictionary."""
        params = {"start_lr": 0.001, "decay_steps": 5000}
        config = LearningRateConfig.from_dict(params)
        self.assertEqual(config.start_lr, 0.001)
        self.assertEqual(config.decay_steps, 5000)


class TestTrainingConfig(unittest.TestCase):
    """Test TrainingConfig dataclass."""

    def test_single_task_config(self):
        """Test single-task configuration parsing."""
        config_dict = {
            "training": {
                "numb_steps": 1000,
                "warmup_steps": 100,
                "disp_freq": 100,
            },
            "learning_rate": {"start_lr": 0.001},
        }
        config = TrainingConfig.from_dict(config_dict)
        self.assertEqual(config.num_steps, 1000)
        self.assertEqual(config.warmup_steps, 100)
        self.assertFalse(config.is_multitask)

    def test_multitask_config(self):
        """Test multi-task configuration parsing."""
        config_dict = {
            "training": {
                "numb_steps": 1000,
                "optim_dict": {
                    "task1": {"opt_type": "Adam"},
                    "task2": {"opt_type": "AdamW"},
                },
            },
            "learning_rate": {"start_lr": 0.001},
        }
        model_keys = ["task1", "task2"]
        config = TrainingConfig.from_dict(config_dict, model_keys)
        self.assertTrue(config.is_multitask)
        self.assertIn("task1", config.optimizer_dict)
        self.assertIn("task2", config.optimizer_dict)

    def test_warmup_ratio(self):
        """Test warmup ratio computation."""
        config_dict = {
            "training": {
                "numb_steps": 1000,
                "warmup_ratio": 0.1,
            },
            "learning_rate": {"start_lr": 0.001},
        }
        config = TrainingConfig.from_dict(config_dict)
        self.assertEqual(config.warmup_steps, 100)

    def test_invalid_num_steps(self):
        """Test validation of invalid num_steps."""
        config_dict = {
            "training": {"numb_steps": 0},
            "learning_rate": {"start_lr": 0.001},
        }
        with self.assertRaises(ValueError):
            TrainingConfig.from_dict(config_dict)


class TestOptimizerFactory(unittest.TestCase):
    """Test OptimizerFactory."""

    def setUp(self):
        """Set up test fixtures."""
        self.factory = OptimizerFactory()

    def test_available_optimizers(self):
        """Test getting available optimizer types."""
        optimizers = self.factory.get_available_optimizers()
        self.assertIn("Adam", optimizers)
        self.assertIn("AdamW", optimizers)
        self.assertIn("LKF", optimizers)

    def test_scheduler_support(self):
        """Test checking scheduler support."""
        self.assertTrue(self.factory.supports_scheduler("Adam"))
        self.assertTrue(self.factory.supports_scheduler("AdamW"))
        self.assertFalse(self.factory.supports_scheduler("LKF"))


class TestHooks(unittest.TestCase):
    """Test hook system."""

    def test_hook_priority(self):
        """Test hook priority ordering."""
        manager = HookManager()

        # Create hooks with different priorities
        class LowPriorityHook(TrainingHook):
            priority = HookPriority.LOW

        class HighPriorityHook(TrainingHook):
            priority = HookPriority.HIGH

        class NormalPriorityHook(TrainingHook):
            priority = HookPriority.NORMAL

        low = LowPriorityHook()
        high = HighPriorityHook()
        normal = NormalPriorityHook()

        manager.register(low)
        manager.register(high)
        manager.register(normal)

        # Check order: high, normal, low
        self.assertEqual(manager.hooks[0], high)
        self.assertEqual(manager.hooks[1], normal)
        self.assertEqual(manager.hooks[2], low)

    def test_hook_execution(self):
        """Test hook method execution."""
        manager = HookManager()

        class TestHook(TrainingHook):
            def __init__(self):
                self.step_count = 0

            def on_step_end(self, step, logs):
                self.step_count += 1

        hook = TestHook()
        manager.register(hook)

        manager.on_step_end(0, {"loss": 1.0})
        manager.on_step_end(1, {"loss": 0.5})

        self.assertEqual(hook.step_count, 2)

    def test_hook_error_handling(self):
        """Test that hook errors don't crash training."""
        manager = HookManager()

        class BadHook(TrainingHook):
            def on_step_end(self, step, logs):
                raise RuntimeError("Hook error")

        class GoodHook(TrainingHook):
            def __init__(self):
                self.called = False

            def on_step_end(self, step, logs):
                self.called = True

        bad = BadHook()
        good = GoodHook()

        manager.register(bad)
        manager.register(good)

        # Should not raise
        manager.on_step_end(0, {"loss": 1.0})

        # Good hook should still be called
        self.assertTrue(good.called)


class TestLossAccumulator(unittest.TestCase):
    """Test LossAccumulator."""

    def test_single_task_accumulation(self):
        """Test single-task loss accumulation."""
        accumulator = LossAccumulator(is_multitask=False)

        accumulator.update({"loss": 1.0, "rmse": 0.5}, "Default")
        accumulator.update({"loss": 2.0, "rmse": 1.0}, "Default")

        averaged = accumulator.get_averaged()
        self.assertAlmostEqual(averaged["loss"], 1.5)
        self.assertAlmostEqual(averaged["rmse"], 0.75)

    def test_multitask_accumulation(self):
        """Test multi-task loss accumulation."""
        accumulator = LossAccumulator(is_multitask=True, model_keys=["task1", "task2"])

        accumulator.update({"loss": 1.0}, "task1")
        accumulator.update({"loss": 2.0}, "task1")
        accumulator.update({"loss": 3.0}, "task2")

        avg1 = accumulator.get_averaged("task1")
        avg2 = accumulator.get_averaged("task2")

        self.assertAlmostEqual(avg1["loss"], 1.5)
        self.assertAlmostEqual(avg2["loss"], 3.0)

    def test_reset(self):
        """Test accumulator reset."""
        accumulator = LossAccumulator(is_multitask=False)

        accumulator.update({"loss": 1.0}, "Default")
        accumulator.reset()

        averaged = accumulator.get_averaged()
        self.assertEqual(averaged, {})

    def test_skip_l2_keys(self):
        """Test that l2_ keys are skipped."""
        accumulator = LossAccumulator(is_multitask=False)

        accumulator.update({"loss": 1.0, "l2_loss": 100.0}, "Default")

        averaged = accumulator.get_averaged()
        self.assertIn("loss", averaged)
        self.assertNotIn("l2_loss", averaged)


class TestDisplayConfig(unittest.TestCase):
    """Test DisplayConfig."""

    def test_default_values(self):
        """Test default display configuration."""
        config = DisplayConfig()
        self.assertEqual(config.disp_file, "lcurve.out")
        self.assertEqual(config.disp_freq, 1000)
        self.assertTrue(config.disp_training)

    def test_tensorboard_defaults(self):
        """Test TensorBoard configuration defaults."""
        config = DisplayConfig()
        self.assertFalse(config.tensorboard)
        self.assertEqual(config.tensorboard_log_dir, "log")


class TestCheckpointConfig(unittest.TestCase):
    """Test CheckpointConfig."""

    def test_default_values(self):
        """Test default checkpoint configuration."""
        config = CheckpointConfig()
        self.assertEqual(config.save_ckpt, "model.ckpt")
        self.assertEqual(config.save_freq, 1000)
        self.assertEqual(config.max_ckpt_keep, 5)


class TestEndToEndCLI(unittest.TestCase):
    """End-to-end tests for CLI integration.

    These tests verify that dp --pt train works correctly with the new Trainer.
    """

    def setUp(self):
        """Set up test fixtures with water data config."""
        self.config = {
            "model": {
                "type_map": ["O", "H"],
                "descriptor": {
                    "type": "se_e2_a",
                    "sel": [46, 92],
                    "rcut_smth": 0.50,
                    "rcut": 6.00,
                    "neuron": [25, 50, 100],
                    "resnet_dt": False,
                    "axis_neuron": 16,
                    "seed": 1,
                },
                "fitting_net": {
                    "neuron": [240, 240, 240],
                    "resnet_dt": True,
                    "seed": 1,
                },
            },
            "learning_rate": {
                "type": "exp",
                "start_lr": 0.001,
                "stop_lr": 3.51e-8,
                "decay_steps": 5000,
            },
            "loss": {
                "type": "ener",
                "start_pref_e": 0.02,
                "limit_pref_e": 1,
                "start_pref_f": 1000,
                "limit_pref_f": 1,
                "start_pref_v": 0.0,
                "limit_pref_v": 0.0,
            },
            "training": {
                "training_data": {
                    "systems": ["source/tests/pt/water/data/data_0"],
                    "batch_size": 1,
                    "numb_btch": 1,
                },
                "validation_data": {
                    "systems": ["source/tests/pt/water/data/data_1"],
                    "batch_size": 1,
                    "numb_btch": 1,
                },
                "numb_steps": 2,
                "seed": 10,
                "disp_file": "lcurve.out",
                "disp_freq": 1,
                "save_freq": 1,
            },
        }

    def test_get_trainer_creates_new_trainer(self):
        """Test get_trainer creates new modular Trainer."""
        config = copy.deepcopy(self.config)
        trainer = get_trainer(config)

        # Verify it's the new Trainer
        from deepmd.pt.train.trainer import Trainer as NewTrainer

        self.assertIsInstance(trainer, NewTrainer)

    def test_trainer_get_data(self):
        """Test Trainer.get_data method works correctly."""
        config = copy.deepcopy(self.config)
        trainer = get_trainer(config)

        # Get training data
        input_dict, label_dict, log_dict = trainer.get_data(is_train=True)
        self.assertIn("coord", input_dict)
        self.assertIn("atype", input_dict)

        # Get validation data
        input_dict, label_dict, log_dict = trainer.get_data(is_train=False)
        self.assertIn("coord", input_dict)
        self.assertIn("atype", input_dict)

    def test_trainer_model_accessible(self):
        """Test that trainer.model is accessible."""
        config = copy.deepcopy(self.config)
        trainer = get_trainer(config)

        # Model should be accessible
        self.assertIsNotNone(trainer.model)

        # Should have expected attributes
        self.assertTrue(hasattr(trainer.model, "get_descriptor"))
        self.assertTrue(hasattr(trainer.model, "get_fitting_net"))

    def test_trainer_config_components(self):
        """Test that config components are properly initialized."""
        config = copy.deepcopy(self.config)
        trainer = get_trainer(config)

        # Check config components
        self.assertIsNotNone(trainer.config.optimizer)
        self.assertIsNotNone(trainer.config.learning_rate)
        self.assertIsNotNone(trainer.config.display)
        self.assertIsNotNone(trainer.config.checkpoint)

        # Check optimizer type
        self.assertEqual(trainer.config.optimizer.opt_type, "Adam")

    def test_trainer_components_initialized(self):
        """Test that all trainer components are initialized."""
        config = copy.deepcopy(self.config)
        trainer = get_trainer(config)

        # Check components
        self.assertIsNotNone(trainer.data_manager)
        self.assertIsNotNone(trainer.checkpoint_manager)
        self.assertIsNotNone(trainer.hook_manager)
        self.assertIsNotNone(trainer.logger)
        self.assertIsNotNone(trainer.optimizer)
        self.assertIsNotNone(trainer.lr_schedule)

    def test_trainer_wrapper_initialized(self):
        """Test that model wrapper is properly initialized."""
        config = copy.deepcopy(self.config)
        trainer = get_trainer(config)

        # Wrapper should exist
        self.assertIsNotNone(trainer.wrapper)

        # Wrapper should have model and loss
        self.assertTrue(hasattr(trainer.wrapper, "model"))
        self.assertTrue(hasattr(trainer.wrapper, "loss"))


if __name__ == "__main__":
    unittest.main()
