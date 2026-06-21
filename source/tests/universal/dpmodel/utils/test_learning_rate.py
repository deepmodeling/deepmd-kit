# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.utils.learning_rate import (
    LearningRateCosine,
    LearningRateExp,
    LearningRateWSD,
)


class TestLearningRateExpBasic(unittest.TestCase):
    """Test basic exponential decay learning rate functionality."""

    def test_basic_decay(self) -> None:
        """Test basic exponential decay without warmup."""
        lr = LearningRateExp(
            start_lr=1e-3,
            stop_lr=1e-5,
            num_steps=10000,
            decay_steps=5000,
        )
        np.testing.assert_allclose(lr.value(0), 1e-3, rtol=1e-10)
        np.testing.assert_allclose(lr.value(10000), 1e-5, rtol=1e-5)

    def test_stop_lr_ratio(self) -> None:
        """Test stop_lr_ratio parameter."""
        lr = LearningRateExp(
            start_lr=1e-3,
            stop_lr_ratio=0.01,
            num_steps=10000,
            decay_steps=5000,
        )
        np.testing.assert_allclose(lr.stop_lr, 1e-5, rtol=1e-10)
        np.testing.assert_allclose(lr.value(10000), 1e-5, rtol=1e-5)

    def test_decay_rate_override(self) -> None:
        """Test explicit decay_rate parameter."""
        lr = LearningRateExp(
            start_lr=1e-3,
            stop_lr=1e-5,
            num_steps=10000,
            decay_steps=1000,
            decay_rate=0.9,
        )
        self.assertEqual(lr.decay_rate, 0.9)
        np.testing.assert_allclose(lr.value(1000), 1e-3 * 0.9, rtol=1e-10)

    def test_rejects_nonpositive_or_nonfinite_start_lr(self) -> None:
        """Test invalid start_lr values are rejected by the base schedule."""
        for start_lr in (0.0, -1e-3, np.inf, np.nan):
            with self.subTest(start_lr=start_lr):
                with self.assertRaisesRegex(ValueError, "start_lr"):
                    LearningRateExp(
                        start_lr=start_lr,
                        stop_lr=1e-5,
                        num_steps=10000,
                        decay_steps=5000,
                    )


class TestLearningRateCosineBasic(unittest.TestCase):
    """Test basic cosine annealing learning rate functionality."""

    def test_basic_cosine(self) -> None:
        """Test basic cosine annealing without warmup."""
        lr = LearningRateCosine(
            start_lr=1e-3,
            stop_lr=1e-5,
            num_steps=10000,
        )
        np.testing.assert_allclose(lr.value(0), 1e-3, rtol=1e-10)
        np.testing.assert_allclose(lr.value(10000), 1e-5, rtol=1e-10)
        np.testing.assert_allclose(lr.value(5000), (1e-3 + 1e-5) / 2, rtol=1e-5)

    def test_stop_lr_ratio(self) -> None:
        """Test stop_lr_ratio parameter."""
        lr = LearningRateCosine(
            start_lr=1e-3,
            stop_lr_ratio=0.01,
            num_steps=10000,
        )
        np.testing.assert_allclose(lr.stop_lr, 1e-5, rtol=1e-10)


class TestLearningRateWSDBasic(unittest.TestCase):
    """Test warmup-stable-decay learning rate functionality."""

    def test_basic_wsd_inverse_linear(self) -> None:
        """Test plateau and inverse-linear decay without warmup."""
        lr = LearningRateWSD(
            start_lr=1e-3,
            stop_lr=1e-5,
            num_steps=10000,
            decay_phase_ratio=0.1,
        )
        expected_mid = 1.0 / (0.5 / 1e-5 + 0.5 / 1e-3)

        self.assertEqual(lr.stable_steps, 9000)
        np.testing.assert_allclose(lr.value(0), 1e-3, rtol=1e-10)
        np.testing.assert_allclose(lr.value(9000), 1e-3, rtol=1e-10)
        np.testing.assert_allclose(lr.value(9500), expected_mid, rtol=1e-10)
        np.testing.assert_allclose(lr.value(10000), 1e-5, rtol=1e-10)

    def test_basic_wsd_cosine(self) -> None:
        """Test plateau and cosine decay without warmup."""
        lr = LearningRateWSD(
            start_lr=1e-3,
            stop_lr=1e-5,
            num_steps=10000,
            decay_phase_ratio=0.1,
            decay_type="cosine",
        )
        expected_mid = 1e-5 + (1e-3 - 1e-5) * 0.5

        self.assertEqual(lr.stable_steps, 9000)
        np.testing.assert_allclose(lr.value(0), 1e-3, rtol=1e-10)
        np.testing.assert_allclose(lr.value(9000), 1e-3, rtol=1e-10)
        np.testing.assert_allclose(lr.value(9500), expected_mid, rtol=1e-10)
        np.testing.assert_allclose(lr.value(10000), 1e-5, rtol=1e-10)

    def test_basic_wsd_linear(self) -> None:
        """Test plateau and linear decay without warmup."""
        lr = LearningRateWSD(
            start_lr=1e-3,
            stop_lr=1e-5,
            num_steps=10000,
            decay_phase_ratio=0.1,
            decay_type="linear",
        )
        expected_mid = 1e-3 + (1e-5 - 1e-3) * 0.5

        self.assertEqual(lr.stable_steps, 9000)
        np.testing.assert_allclose(lr.value(0), 1e-3, rtol=1e-10)
        np.testing.assert_allclose(lr.value(9000), 1e-3, rtol=1e-10)
        np.testing.assert_allclose(lr.value(9500), expected_mid, rtol=1e-10)
        np.testing.assert_allclose(lr.value(10000), 1e-5, rtol=1e-10)

    def test_stop_lr_ratio(self) -> None:
        """Test stop_lr_ratio parameter for WSD."""
        lr = LearningRateWSD(
            start_lr=1e-3,
            stop_lr_ratio=0.01,
            num_steps=10000,
            decay_phase_ratio=0.1,
        )
        np.testing.assert_allclose(lr.stop_lr, 1e-5, rtol=1e-10)
        np.testing.assert_allclose(lr.value(10000), 1e-5, rtol=1e-10)

    def test_invalid_decay_phase_ratio(self) -> None:
        """Test invalid WSD decay_phase_ratio values."""
        with self.assertRaises(ValueError):
            LearningRateWSD(
                start_lr=1e-3,
                stop_lr=1e-5,
                num_steps=10000,
                decay_phase_ratio=0.0,
            )
        with self.assertRaises(ValueError):
            LearningRateWSD(
                start_lr=1e-3,
                stop_lr=1e-5,
                num_steps=10000,
                decay_phase_ratio=1.1,
            )
        with self.assertRaises(ValueError):
            LearningRateWSD(
                start_lr=1e-3,
                stop_lr=1e-5,
                num_steps=10000,
                decay_type="bad_mode",
            )

    def test_decay_phase_exceeds_post_warmup_steps(self) -> None:
        """Test WSD clamps decay_phase_steps to post-warmup steps when ratio is too large."""
        lr = LearningRateWSD(
            start_lr=1e-3,
            stop_lr=1e-5,
            num_steps=10,
            warmup_steps=9,
            decay_phase_ratio=0.2,
        )
        # decay_num_steps = 1, so decay_phase_steps should be clamped to 1
        self.assertEqual(lr.decay_phase_steps, 1)
        self.assertEqual(lr.stable_steps, 0)


class TestLearningRateWarmup(unittest.TestCase):
    """Test learning rate warmup functionality."""

    def test_warmup_steps_exp(self) -> None:
        """Test warmup with exponential decay."""
        lr = LearningRateExp(
            start_lr=1e-3,
            stop_lr=1e-5,
            num_steps=10000,
            decay_steps=1000,
            warmup_steps=1000,
        )
        self.assertEqual(lr.decay_num_steps, 9000)
        np.testing.assert_allclose(lr.value(0), 0.0, rtol=1e-10)
        np.testing.assert_allclose(lr.value(500), 0.5e-3, rtol=1e-10)
        np.testing.assert_allclose(lr.value(1000), 1e-3, rtol=1e-10)
        # Step 2000: 1000 steps into decay phase (1 decay period with decay_steps=1000)
        # lr = start_lr * decay_rate^1 = 1e-3 * exp(log(0.01)/9) ≈ 5.995e-4
        np.testing.assert_allclose(
            to_numpy_array(lr.value(2000)), 1e-3 * np.exp(np.log(0.01) / 9), rtol=1e-5
        )

    def test_warmup_steps_cosine(self) -> None:
        """Test warmup with cosine annealing."""
        lr = LearningRateCosine(
            start_lr=1e-3,
            stop_lr=1e-5,
            num_steps=10000,
            warmup_steps=1000,
        )
        self.assertEqual(lr.decay_num_steps, 9000)
        np.testing.assert_allclose(lr.value(0), 0.0, rtol=1e-10)
        np.testing.assert_allclose(lr.value(1000), 1e-3, rtol=1e-10)
        np.testing.assert_allclose(lr.value(10000), 1e-5, rtol=1e-10)

    def test_warmup_steps_wsd(self) -> None:
        """Test warmup with default inverse-linear WSD schedule."""
        lr = LearningRateWSD(
            start_lr=1e-3,
            stop_lr=1e-5,
            num_steps=10000,
            warmup_steps=1000,
            decay_phase_ratio=0.1,
        )
        self.assertEqual(lr.decay_num_steps, 9000)
        self.assertEqual(lr.decay_phase_steps, 1000)
        self.assertEqual(lr.stable_steps, 8000)
        np.testing.assert_allclose(lr.value(0), 0.0, rtol=1e-10)
        np.testing.assert_allclose(lr.value(500), 0.5e-3, rtol=1e-10)
        np.testing.assert_allclose(lr.value(1000), 1e-3, rtol=1e-10)
        np.testing.assert_allclose(lr.value(9000), 1e-3, rtol=1e-10)
        np.testing.assert_allclose(lr.value(10000), 1e-5, rtol=1e-10)

    def test_warmup_steps_wsd_linear(self) -> None:
        """Test warmup with linear WSD decay."""
        lr = LearningRateWSD(
            start_lr=1e-3,
            stop_lr=1e-5,
            num_steps=10000,
            warmup_steps=1000,
            decay_phase_ratio=0.1,
            decay_type="linear",
        )
        expected_mid = 1e-3 + (1e-5 - 1e-3) * 0.5

        np.testing.assert_allclose(lr.value(0), 0.0, rtol=1e-10)
        np.testing.assert_allclose(lr.value(1000), 1e-3, rtol=1e-10)
        np.testing.assert_allclose(lr.value(9500), expected_mid, rtol=1e-10)
        np.testing.assert_allclose(lr.value(10000), 1e-5, rtol=1e-10)

    def test_warmup_ratio(self) -> None:
        """Test warmup_ratio parameter."""
        lr = LearningRateExp(
            start_lr=1e-3,
            stop_lr=1e-5,
            num_steps=10000,
            decay_steps=1000,
            warmup_ratio=0.1,
        )
        self.assertEqual(lr.warmup_steps, 1000)
        self.assertEqual(lr.decay_num_steps, 9000)

    def test_warmup_start_factor(self) -> None:
        """Test warmup_start_factor parameter."""
        lr = LearningRateExp(
            start_lr=1e-3,
            stop_lr=1e-5,
            num_steps=10000,
            decay_steps=1000,
            warmup_steps=1000,
            warmup_start_factor=0.1,
        )
        np.testing.assert_allclose(lr.value(0), 0.1e-3, rtol=1e-10)
        np.testing.assert_allclose(lr.value(1000), 1e-3, rtol=1e-10)

    def test_no_warmup(self) -> None:
        """Test that warmup_steps=0 works correctly."""
        lr = LearningRateExp(
            start_lr=1e-3,
            stop_lr=1e-5,
            num_steps=10000,
            decay_steps=5000,
            warmup_steps=0,
        )
        self.assertEqual(lr.warmup_steps, 0)
        self.assertEqual(lr.decay_num_steps, 10000)
        np.testing.assert_allclose(lr.value(0), 1e-3, rtol=1e-10)


class TestLearningRateArrayInput(unittest.TestCase):
    """Test learning rate with array inputs for JIT compatibility."""

    def test_array_input_exp(self) -> None:
        """Test exponential decay with array input."""
        lr = LearningRateExp(
            start_lr=1e-3,
            stop_lr=1e-5,
            num_steps=10000,
            decay_steps=5000,
            warmup_steps=1000,
        )
        steps = np.array([0, 500, 1000, 5000, 10000])
        lrs = lr.value(steps)
        self.assertEqual(lrs.shape, (5,))
        np.testing.assert_allclose(lrs[0], 0.0, rtol=1e-10)
        np.testing.assert_allclose(lrs[2], 1e-3, rtol=1e-10)

    def test_array_input_cosine(self) -> None:
        """Test cosine annealing with array input."""
        lr = LearningRateCosine(
            start_lr=1e-3,
            stop_lr=1e-5,
            num_steps=10000,
            warmup_steps=1000,
        )
        steps = np.array([0, 1000, 5500, 10000])
        lrs = lr.value(steps)
        self.assertEqual(lrs.shape, (4,))
        np.testing.assert_allclose(lrs[0], 0.0, rtol=1e-10)
        np.testing.assert_allclose(lrs[1], 1e-3, rtol=1e-10)
        np.testing.assert_allclose(lrs[3], 1e-5, rtol=1e-10)

    def test_array_input_wsd(self) -> None:
        """Test inverse-linear warmup-stable-decay with array input."""
        lr = LearningRateWSD(
            start_lr=1e-3,
            stop_lr=1e-5,
            num_steps=10000,
            decay_phase_ratio=0.1,
        )
        steps = np.array([0, 9000, 9500, 10000])
        lrs = lr.value(steps)
        expected_mid = 1.0 / (0.5 / 1e-5 + 0.5 / 1e-3)

        self.assertEqual(lrs.shape, (4,))
        np.testing.assert_allclose(lrs[0], 1e-3, rtol=1e-10)
        np.testing.assert_allclose(lrs[1], 1e-3, rtol=1e-10)
        np.testing.assert_allclose(lrs[2], expected_mid, rtol=1e-10)
        np.testing.assert_allclose(lrs[3], 1e-5, rtol=1e-10)

    def test_array_input_wsd_cosine(self) -> None:
        """Test cosine warmup-stable-decay with array input."""
        lr = LearningRateWSD(
            start_lr=1e-3,
            stop_lr=1e-5,
            num_steps=10000,
            decay_phase_ratio=0.1,
            decay_type="cosine",
        )
        steps = np.array([0, 9000, 9500, 10000])
        lrs = lr.value(steps)
        expected_mid = 1e-5 + (1e-3 - 1e-5) * 0.5

        self.assertEqual(lrs.shape, (4,))
        np.testing.assert_allclose(lrs[0], 1e-3, rtol=1e-10)
        np.testing.assert_allclose(lrs[1], 1e-3, rtol=1e-10)
        np.testing.assert_allclose(lrs[2], expected_mid, rtol=1e-10)
        np.testing.assert_allclose(lrs[3], 1e-5, rtol=1e-10)


class TestLearningRateBeyondStopSteps(unittest.TestCase):
    """Test learning rate behavior beyond num_steps."""

    def test_exp_beyond_num_steps(self) -> None:
        """Test exponential decay clamps to stop_lr."""
        lr = LearningRateExp(
            start_lr=1e-3,
            stop_lr=1e-5,
            num_steps=10000,
            decay_steps=1000,
        )
        np.testing.assert_allclose(lr.value(20000), 1e-5, rtol=1e-10)

    def test_cosine_beyond_num_steps(self) -> None:
        """Test cosine annealing returns stop_lr beyond decay phase."""
        lr = LearningRateCosine(
            start_lr=1e-3,
            stop_lr=1e-5,
            num_steps=10000,
        )
        np.testing.assert_allclose(lr.value(20000), 1e-5, rtol=1e-10)

    def test_wsd_beyond_num_steps(self) -> None:
        """Test WSD returns stop_lr beyond decay phase."""
        lr = LearningRateWSD(
            start_lr=1e-3,
            stop_lr=1e-5,
            num_steps=10000,
            decay_phase_ratio=0.1,
        )
        np.testing.assert_allclose(lr.value(20000), 1e-5, rtol=1e-10)
