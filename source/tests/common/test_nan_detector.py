# SPDX-License-Identifier: LGPL-3.0-or-later
"""Test cases for NaN detection utility."""

import math
import unittest

import numpy as np

from deepmd.utils.nan_detector import (
    LossNaNError,
    check_loss_nan,
    check_single_loss_nan,
)


class TestNaNDetector(unittest.TestCase):
    """Test the NaN detection utility functions."""

    def test_normal_values_pass(self):
        """Test that normal loss values don't trigger NaN detection."""
        # Test with various normal values
        normal_losses = {
            "energy": 0.5,
            "force": 1.0,
            "virial": 0.001,
            "zero": 0.0,
            "negative": -0.5,
        }

        # Should not raise any exception
        try:
            check_loss_nan(100, normal_losses)
        except Exception as e:
            self.fail(f"Normal values should not raise exception: {e}")

    def test_nan_detection_raises_exception(self):
        """Test that NaN values trigger the proper exception."""
        # Test with NaN values
        nan_losses = {
            "energy": 0.5,
            "force": float("nan"),
            "virial": 1.0,
        }

        with self.assertRaises(LossNaNError) as context:
            check_loss_nan(200, nan_losses)

        exception = context.exception
        self.assertEqual(exception.step, 200)
        self.assertIn("force", str(exception))
        self.assertIn("NaN detected in loss at training step 200", str(exception))

    def test_single_loss_nan_detection(self):
        """Test single loss NaN detection."""
        # Normal value should pass
        try:
            check_single_loss_nan(50, "test_loss", 0.5)
        except Exception as e:
            self.fail(f"Normal single loss should not raise exception: {e}")

        # NaN value should raise
        with self.assertRaises(LossNaNError) as context:
            check_single_loss_nan(50, "test_loss", float("nan"))

        exception = context.exception
        self.assertEqual(exception.step, 50)
        self.assertIn("test_loss", str(exception))

    def test_various_nan_representations(self):
        """Test detection of various NaN representations."""
        nan_values = [
            float("nan"),
            np.nan,
            math.nan,
        ]

        for i, nan_val in enumerate(nan_values):
            with self.assertRaises(LossNaNError):
                check_single_loss_nan(i, f"loss_{i}", nan_val)

    def test_tensor_like_objects(self):
        """Test that tensor-like objects work with NaN detection."""

        # Mock tensor-like object with item() method
        class MockTensor:
            def __init__(self, value):
                self._value = value

            def item(self):
                return self._value

        # Normal tensor should pass
        normal_tensor = MockTensor(0.5)
        try:
            check_single_loss_nan(10, "tensor_loss", normal_tensor)
        except Exception as e:
            self.fail(f"Normal tensor should not raise exception: {e}")

        # NaN tensor should raise
        nan_tensor = MockTensor(float("nan"))
        with self.assertRaises(LossNaNError):
            check_single_loss_nan(10, "tensor_loss", nan_tensor)

    def test_error_message_format(self):
        """Test that error messages contain useful information."""
        nan_losses = {
            "energy": 0.5,
            "force": float("nan"),
            "virial": float("nan"),
        }

        with self.assertRaises(LossNaNError) as context:
            check_loss_nan(123, nan_losses)

        error_msg = str(context.exception)

        # Check key information is in the message
        self.assertIn("step 123", error_msg)
        self.assertIn("force=nan", error_msg)
        self.assertIn("virial=nan", error_msg)
        self.assertIn("Training stopped", error_msg)
        self.assertIn("learning rate too high", error_msg)

    def test_mixed_loss_dict(self):
        """Test loss dictionary with mix of normal and NaN values."""
        mixed_losses = {
            "energy": 0.5,
            "force": float("nan"),
            "virial": 1.0,
            "dipole": float("nan"),
        }

        with self.assertRaises(LossNaNError) as context:
            check_loss_nan(99, mixed_losses)

        exception = context.exception
        # Should detect both NaN values
        error_msg = str(exception)
        self.assertIn("force=nan", error_msg)
        self.assertIn("dipole=nan", error_msg)
        # Should not mention normal values
        self.assertNotIn("energy=0.5", error_msg)
        self.assertNotIn("virial=1.0", error_msg)

    def test_edge_cases(self):
        """Test edge cases for NaN detection."""
        # Empty dict should pass
        try:
            check_loss_nan(1, {})
        except Exception as e:
            self.fail(f"Empty dict should not raise exception: {e}")

        # None values should not trigger NaN detection
        try:
            check_loss_nan(1, {"test": None})
        except Exception as e:
            self.fail(f"None values should not raise exception: {e}")

        # Infinity should not trigger NaN detection (separate issue)
        try:
            check_loss_nan(1, {"test": float("inf")})
        except Exception as e:
            self.fail(f"Infinity should not raise NaN exception: {e}")


if __name__ == "__main__":
    unittest.main()
