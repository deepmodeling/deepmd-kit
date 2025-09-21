# SPDX-License-Identifier: LGPL-3.0-or-later
"""Integration test to verify NaN detection during training.

This test creates a mock training scenario where total loss becomes NaN
and verifies that the training stops with appropriate error message.
"""

import unittest
from unittest.mock import (
    patch,
)

from deepmd.utils.nan_detector import (
    LossNaNError,
    check_total_loss_nan,
)


class TestNaNDetectionIntegration(unittest.TestCase):
    """Integration tests for NaN detection during training."""

    def test_training_stops_on_nan_loss(self):
        """Test that training stops when NaN is detected in total loss."""
        # Normal total loss should pass
        try:
            check_total_loss_nan(100, 0.1)
        except Exception as e:
            self.fail(f"Normal total loss should not raise exception: {e}")

        # NaN total loss should raise
        with self.assertRaises(LossNaNError) as context:
            check_total_loss_nan(100, float("nan"))

        exception = context.exception
        self.assertEqual(exception.step, 100)
        self.assertIn("NaN detected in total loss", str(exception))

    @patch("deepmd.utils.nan_detector.log")
    def test_logging_on_nan_detection(self, mock_log):
        """Test that NaN detection logs appropriate error messages."""
        with self.assertRaises(LossNaNError):
            check_total_loss_nan(200, float("nan"))

        # Verify that error was logged
        mock_log.error.assert_called_once()
        logged_message = mock_log.error.call_args[0][0]
        self.assertIn("NaN detected in total loss at step 200", logged_message)

    def test_training_simulation_with_checkpoint_prevention(self):
        """Simulate the training checkpoint scenario to ensure NaN prevents saving."""
        # Simulate the training flow: check total loss, then save checkpoint
        step_id = 1000
        total_loss = float("nan")

        # This should raise LossNaNError, preventing any subsequent checkpoint saving
        with self.assertRaises(LossNaNError) as context:
            check_total_loss_nan(step_id, total_loss)

        # Verify the error contains expected information
        exception = context.exception
        self.assertIn("Training stopped to prevent wasting time", str(exception))
        self.assertIn("corrupted parameters", str(exception))

    def test_realistic_training_scenario(self):
        """Test a more realistic training scenario with decreasing then NaN loss."""
        # Simulate normal training progression
        normal_steps = [
            (1, 1.0),  # Initial high loss
            (10, 0.5),  # Loss decreasing
            (20, 0.25),  # Loss continuing to decrease
            (50, 0.1),  # Good progress
        ]

        # All normal steps should pass
        for step, loss_val in normal_steps:
            try:
                check_total_loss_nan(step, loss_val)
            except Exception as e:
                self.fail(
                    f"Normal training step {step} should not raise exception: {e}"
                )

        # But when loss becomes NaN, training should stop
        with self.assertRaises(LossNaNError) as context:
            check_total_loss_nan(100, float("nan"))

        exception = context.exception
        self.assertEqual(exception.step, 100)
        self.assertIn("Training stopped", str(exception))


if __name__ == "__main__":
    unittest.main()
