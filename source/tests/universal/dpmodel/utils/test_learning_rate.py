# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.utils.learning_rate import (
    LearningRateCosine,
    LearningRateExp,
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
