# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

from deepmd.pt.utils.learning_rate import (
    LearningRateExp,
)
from deepmd.tf.utils import (
    learning_rate,
)


class TestLearningRate(unittest.TestCase):
    def setUp(self) -> None:
        self.start_lr = 0.001
        self.stop_lr = 3.51e-8
        self.decay_steps = np.arange(400, 601, 100)
        self.stop_steps = np.arange(500, 1600, 500)

    def test_consistency(self) -> None:
        for decay_step in self.decay_steps:
            for stop_step in self.stop_steps:
                self.decay_step = decay_step
                self.stop_step = stop_step
                self.judge_it()
                self.decay_rate_pt()

    def judge_it(self) -> None:
        base_lr = learning_rate.LearningRateExp(
            self.start_lr, self.stop_lr, self.decay_step
        )
        g = tf.Graph()
        with g.as_default():
            global_step = tf.placeholder(shape=[], dtype=tf.int32)
            t_lr = base_lr.build(global_step, self.stop_step)

        my_lr = LearningRateExp(
            self.start_lr, self.stop_lr, self.decay_step, self.stop_step
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
            self.start_lr, self.stop_lr, self.decay_step, self.stop_step
        )

        default_ds = 100 if self.stop_step // 10 > 100 else self.stop_step // 100 + 1
        if self.decay_step >= self.stop_step:
            self.decay_step = default_ds
        decay_rate = np.exp(
            np.log(self.stop_lr / self.start_lr) / (self.stop_step / self.decay_step)
        )
        my_lr_decay = LearningRateExp(
            self.start_lr,
            1e-10,
            self.decay_step,
            self.stop_step,
            decay_rate=decay_rate,
        )
        min_lr = 1e-5
        my_lr_decay_trunc = LearningRateExp(
            self.start_lr,
            min_lr,
            self.decay_step,
            self.stop_step,
            decay_rate=decay_rate,
        )
        my_vals = [
            my_lr.value(step_id)
            for step_id in range(self.stop_step)
            if step_id % self.decay_step != 0
        ]
        my_vals_decay = [
            my_lr_decay.value(step_id)
            for step_id in range(self.stop_step)
            if step_id % self.decay_step != 0
        ]
        my_vals_decay_trunc = [
            my_lr_decay_trunc.value(step_id)
            for step_id in range(self.stop_step)
            if step_id % self.decay_step != 0
        ]
        self.assertTrue(np.allclose(my_vals_decay, my_vals))
        self.assertTrue(
            np.allclose(my_vals_decay_trunc, np.clip(my_vals, a_min=min_lr, a_max=None))
        )


if __name__ == "__main__":
    unittest.main()
