# SPDX-License-Identifier: LGPL-3.0-or-later
"""Test cases for NaN detection utility."""

import math
import unittest

import numpy as np

from deepmd.utils.nan_detector import (
    LossNaNError,
    check_total_loss_nan,
)


class TestNaNDetector(unittest.TestCase):
    """Test the NaN detection utility functions."""

    def test_normal_values_pass(self):
        """Test that normal loss values don't trigger NaN detection."""
        # Test with various normal values
        normal_losses = [0.5, 1.0, 0.001, 0.0, -0.5]

        # Should not raise any exception
        for i, loss_val in enumerate(normal_losses):
            try:
                check_total_loss_nan(100 + i, loss_val)
            except Exception as e:
                self.fail(f"Normal values should not raise exception: {e}")

    def test_nan_detection_raises_exception(self):
        """Test that NaN values trigger the proper exception."""
        # Test with NaN value
        with self.assertRaises(LossNaNError) as context:
            check_total_loss_nan(200, float("nan"))

        exception = context.exception
        self.assertEqual(exception.step, 200)
        self.assertTrue(math.isnan(exception.total_loss))
        self.assertIn("NaN detected in total loss at training step 200", str(exception))

    def test_various_nan_representations(self):
        """Test detection of various NaN representations."""
        nan_values = [
            float("nan"),
            np.nan,
            math.nan,
        ]

        for i, nan_val in enumerate(nan_values):
            with self.assertRaises(LossNaNError):
                check_total_loss_nan(i, nan_val)

    def test_error_message_format(self):
        """Test that error messages contain useful information."""
        with self.assertRaises(LossNaNError) as context:
            check_total_loss_nan(123, float("nan"))

        error_msg = str(context.exception)

        # Check key information is in the message
        self.assertIn("step 123", error_msg)
        self.assertIn("Training stopped", error_msg)
        self.assertIn("learning rate too high", error_msg)

    def test_edge_cases(self):
        """Test edge cases for NaN detection."""
        # Infinity should not trigger NaN detection (separate issue)
        try:
            check_total_loss_nan(1, float("inf"))
            check_total_loss_nan(2, float("-inf"))
        except Exception as e:
            self.fail(f"Infinity should not raise NaN exception: {e}")

    def test_numeric_types(self):
        """Test that various numeric types work correctly."""
        # Various numeric types that should pass
        test_values = [
            0.5,  # float
            1,  # int
            np.float32(0.3),  # NumPy float32
            np.float64(0.7),  # NumPy float64
        ]

        for i, val in enumerate(test_values):
            try:
                check_total_loss_nan(10 + i, float(val))
            except Exception as e:
                self.fail(f"Numeric type {type(val)} should not raise exception: {e}")

    def test_inheritance_from_runtime_error(self):
        """Test that LossNaNError inherits from RuntimeError."""
        self.assertTrue(issubclass(LossNaNError, RuntimeError))

        try:
            check_total_loss_nan(999, float("nan"))
        except LossNaNError as e:
            self.assertIsInstance(e, RuntimeError)
        except Exception:
            self.fail("Should raise LossNaNError which inherits from RuntimeError")


if __name__ == "__main__":
    unittest.main()
