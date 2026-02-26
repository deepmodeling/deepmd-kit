# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

from deepmd.pt.utils.learning_rate import (
    LearningRateCosine,
    LearningRateExp,
)
from deepmd.tf.utils.learning_rate import (
    LearningRateSchedule,
)


class TestLearningRate(unittest.TestCase):
    def setUp(self) -> None:
        self.start_lr = 0.001
        self.stop_lr = 3.51e-8
        # decay_steps will be auto-adjusted if >= stop_steps
        self.decay_steps = np.arange(400, 501, 100)
        self.stop_steps = np.arange(500, 1600, 500)

    def test_consistency(self) -> None:
        for decay_step in self.decay_steps:
            for stop_step in self.stop_steps:
                self.decay_step = decay_step
                self.stop_step = stop_step
                self.judge_it()
                self.decay_rate_pt()

    def judge_it(self) -> None:
        base_lr = LearningRateSchedule(
            {
                "type": "exp",
                "start_lr": self.start_lr,
                "stop_lr": self.stop_lr,
                "decay_steps": self.decay_step,
            }
        )
        g = tf.Graph()
        with g.as_default():
            global_step = tf.placeholder(shape=[], dtype=tf.int32)
            t_lr = base_lr.build(global_step, self.stop_step)

        my_lr = LearningRateExp(
            start_lr=self.start_lr,
            stop_lr=self.stop_lr,
            decay_steps=self.decay_step,
            num_steps=self.stop_step,
        )
        with tf.Session(graph=g) as sess:
            base_vals = [
                sess.run(t_lr, feed_dict={global_step: step_id})
                for step_id in range(self.stop_step)
                if step_id % self.decay_step != 0
            ]
        my_vals = [
            my_lr.value(step_id)
            for step_id in range(self.stop_step)
            if step_id % self.decay_step != 0
        ]
        self.assertTrue(np.allclose(base_vals, my_vals))
        tf.reset_default_graph()

    def decay_rate_pt(self) -> None:
        my_lr = LearningRateExp(
            start_lr=self.start_lr,
            stop_lr=self.stop_lr,
            decay_steps=self.decay_step,
            num_steps=self.stop_step,
        )

        # Use the auto-adjusted decay_steps from my_lr for consistency
        actual_decay_steps = my_lr.decay_steps
        decay_rate = np.exp(
            np.log(self.stop_lr / self.start_lr) / (self.stop_step / actual_decay_steps)
        )
        my_lr_decay = LearningRateExp(
            start_lr=self.start_lr,
            stop_lr=1e-10,
            decay_steps=actual_decay_steps,
            num_steps=self.stop_step,
            decay_rate=decay_rate,
        )
        min_lr = 1e-5
        my_lr_decay_trunc = LearningRateExp(
            start_lr=self.start_lr,
            stop_lr=min_lr,
            decay_steps=actual_decay_steps,
            num_steps=self.stop_step,
            decay_rate=decay_rate,
        )
        my_vals = [
            my_lr.value(step_id)
            for step_id in range(self.stop_step)
            if step_id % actual_decay_steps != 0
        ]
        my_vals_decay = [
            my_lr_decay.value(step_id)
            for step_id in range(self.stop_step)
            if step_id % actual_decay_steps != 0
        ]
        my_vals_decay_trunc = [
            my_lr_decay_trunc.value(step_id)
            for step_id in range(self.stop_step)
            if step_id % actual_decay_steps != 0
        ]
        self.assertTrue(np.allclose(my_vals_decay, my_vals))
        self.assertTrue(
            np.allclose(my_vals_decay_trunc, np.clip(my_vals, a_min=min_lr, a_max=None))
        )


class TestLearningRateCosine(unittest.TestCase):
    def test_basic_curve(self) -> None:
        start_lr = 1.0
        stop_lr = 0.1
        stop_steps = 10
        lr = LearningRateCosine(
            start_lr=start_lr,
            stop_lr=stop_lr,
            num_steps=stop_steps,
        )

        self.assertTrue(np.allclose(lr.value(0), start_lr))
        self.assertTrue(np.allclose(lr.value(stop_steps), stop_lr))
        self.assertTrue(np.allclose(lr.value(stop_steps + 5), stop_lr))

        mid_step = stop_steps // 2
        expected_mid = stop_lr + (start_lr - stop_lr) * 0.5
        self.assertTrue(np.allclose(lr.value(mid_step), expected_mid))


if __name__ == "__main__":
    unittest.main()
