# SPDX-License-Identifier: LGPL-3.0-or-later
"""Integration test to verify NaN detection during training.

This test creates a mock training scenario where loss becomes NaN
and verifies that the training stops with appropriate error message.
"""

import unittest
from unittest.mock import (
    patch,
)

from deepmd.utils.nan_detector import (
    LossNaNError,
    check_loss_nan,
)


class TestNaNDetectionIntegration(unittest.TestCase):
    """Integration tests for NaN detection during training."""

    def test_training_stops_on_nan_loss(self):
        """Test that training stops when NaN is detected in loss values."""
        # Simulate a training scenario where loss becomes NaN
        train_results = {
            "energy_loss": 0.1,
            "force_loss": float("nan"),  # This should trigger the detection
            "virial_loss": 0.05,
        }

        valid_results = {
            "energy_loss": 0.12,
            "force_loss": 0.08,
            "virial_loss": 0.06,
        }

        # The NaN in train_results should be detected
        with self.assertRaises(LossNaNError) as context:
            check_loss_nan(100, train_results)

        exception = context.exception
        self.assertEqual(exception.step, 100)
        self.assertIn("force_loss=nan", str(exception))

        # Valid results without NaN should pass
        try:
            check_loss_nan(100, valid_results)
        except Exception as e:
            self.fail(f"Valid results should not raise exception: {e}")

    def test_multi_task_nan_detection(self):
        """Test NaN detection in multi-task training scenario."""
        # Simulate multi-task training results
        multi_task_results = {
            "task1": {
                "energy_loss": 0.1,
                "force_loss": 0.05,
            },
            "task2": {
                "energy_loss": float("nan"),  # NaN in task2
                "force_loss": 0.03,
            },
            "task3": {
                "energy_loss": 0.08,
                "force_loss": 0.04,
            },
        }

        # Check each task separately (as done in the actual training code)
        # task1 and task3 should pass
        try:
            check_loss_nan(50, multi_task_results["task1"])
            check_loss_nan(50, multi_task_results["task3"])
        except Exception as e:
            self.fail(f"Normal tasks should not raise exception: {e}")

        # task2 should fail due to NaN
        with self.assertRaises(LossNaNError) as context:
            check_loss_nan(50, multi_task_results["task2"])

        exception = context.exception
        self.assertEqual(exception.step, 50)
        self.assertIn("energy_loss=nan", str(exception))

    @patch("deepmd.utils.nan_detector.log")
    def test_logging_on_nan_detection(self, mock_log):
        """Test that NaN detection logs appropriate error messages."""
        nan_losses = {
            "energy": 0.5,
            "force": float("nan"),
        }

        with self.assertRaises(LossNaNError):
            check_loss_nan(200, nan_losses)

        # Verify that error was logged
        mock_log.error.assert_called_once()
        logged_message = mock_log.error.call_args[0][0]
        self.assertIn("NaN detected in force at step 200", logged_message)

    def test_training_simulation_with_checkpoint_prevention(self):
        """Simulate the training checkpoint scenario to ensure NaN prevents saving."""

        def mock_save_checkpoint():
            """Mock function that should not be called when NaN is detected."""
            raise AssertionError("Checkpoint should not be saved when NaN is detected!")

        # Simulate the training flow: check loss, then save checkpoint
        step_id = 1000
        loss_results = {
            "total_loss": float("nan"),
            "energy_loss": 0.1,
            "force_loss": 0.05,
        }

        # This should raise LossNaNError before checkpoint saving
        with self.assertRaises(LossNaNError):
            check_loss_nan(step_id, loss_results)
            # This line should never be reached
            mock_save_checkpoint()

        # Verify the error contains expected information
        try:
            check_loss_nan(step_id, loss_results)
        except LossNaNError as e:
            self.assertIn("Training stopped to prevent wasting time", str(e))
            self.assertIn("corrupted parameters", str(e))


if __name__ == "__main__":
    unittest.main()
