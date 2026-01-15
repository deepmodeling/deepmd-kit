# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for TensorFlow learning rate schedule wrapper.

This module tests the TF-specific wrapper logic only.
Core learning rate algorithms are tested in dpmodel tests.
"""

import unittest

import numpy as np

from deepmd.dpmodel.utils.learning_rate import (
    LearningRateExp,
)
from deepmd.tf.env import (
    tf,
)
from deepmd.tf.utils.learning_rate import (
    LearningRateSchedule,
)


class TestLearningRateScheduleValidation(unittest.TestCase):
    """Test TF wrapper validation and error handling."""

    def test_missing_start_lr(self) -> None:
        """Test that missing start_lr raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            LearningRateSchedule({"type": "exp", "stop_lr": 1e-5})
        self.assertIn("start_lr", str(cm.exception))

    def test_value_before_build(self) -> None:
        """Test that calling value() before build() raises RuntimeError."""
        lr_schedule = LearningRateSchedule({"start_lr": 1e-3})
        with self.assertRaises(RuntimeError) as cm:
            lr_schedule.value(100)
        self.assertIn("not built", str(cm.exception))

    def test_base_lr_before_build(self) -> None:
        """Test that accessing base_lr before build() raises RuntimeError."""
        lr_schedule = LearningRateSchedule({"start_lr": 1e-3})
        with self.assertRaises(RuntimeError) as cm:
            _ = lr_schedule.base_lr
        self.assertIn("not built", str(cm.exception))


class TestLearningRateScheduleBuild(unittest.TestCase):
    """Test TF tensor building and integration."""

    def test_build_returns_tensor(self) -> None:
        """Test that build() returns a float32 TF tensor."""
        lr_schedule = LearningRateSchedule({"start_lr": 1e-3, "stop_lr": 1e-5})
        global_step = tf.constant(0, dtype=tf.int64)
        lr_tensor = lr_schedule.build(global_step, stop_steps=10000)

        self.assertIsInstance(lr_tensor, tf.Tensor)
        self.assertEqual(lr_tensor.dtype, tf.float32)

    def test_default_type_exp(self) -> None:
        """Test that default type is 'exp' when not specified."""
        lr_schedule = LearningRateSchedule({"start_lr": 1e-3, "stop_lr": 1e-5})
        global_step = tf.constant(0, dtype=tf.int64)
        lr_schedule.build(global_step, stop_steps=10000)

        self.assertIsInstance(lr_schedule.base_lr, LearningRateExp)

    def test_tensor_value_matches_base_lr(self) -> None:
        """Test that TF tensor value matches BaseLR.value()."""
        lr_schedule = LearningRateSchedule(
            {
                "start_lr": 1e-3,
                "stop_lr": 1e-5,
                "type": "exp",
                "decay_steps": 1000,
            }
        )
        test_step = 5000
        global_step = tf.constant(test_step, dtype=tf.int64)
        lr_schedule.build(global_step, stop_steps=10000)

        # Use value() method which works in both graph and eager mode
        # This indirectly verifies tensor computation matches BaseLR
        tensor_value = lr_schedule.value(test_step)
        base_lr_value = lr_schedule.base_lr.value(test_step)

        np.testing.assert_allclose(tensor_value, base_lr_value, rtol=1e-10)

    def test_start_lr_accessor(self) -> None:
        """Test start_lr() accessor returns correct value."""
        lr_schedule = LearningRateSchedule({"start_lr": 1e-3})
        self.assertEqual(lr_schedule.start_lr(), 1e-3)

    def test_value_after_build(self) -> None:
        """Test value() works correctly after build()."""
        lr_schedule = LearningRateSchedule(
            {
                "start_lr": 1e-3,
                "stop_lr": 1e-5,
                "type": "exp",
                "decay_steps": 1000,
            }
        )
        global_step = tf.constant(0, dtype=tf.int64)
        lr_schedule.build(global_step, stop_steps=10000)

        # value() should work after build
        lr_value = lr_schedule.value(5000)
        expected = lr_schedule.base_lr.value(5000)

        np.testing.assert_allclose(lr_value, expected, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
