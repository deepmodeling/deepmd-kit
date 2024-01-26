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
    def setUp(self):
        self.start_lr = 0.001
        self.stop_lr = 3.51e-8
        self.decay_steps = np.arange(400, 601, 100)
        self.stop_steps = np.arange(500, 1600, 500)

    def test_consistency(self):
        for decay_step in self.decay_steps:
            for stop_step in self.stop_steps:
                self.decay_step = decay_step
                self.stop_step = stop_step
                self.judge_it()

    def judge_it(self):
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


if __name__ == "__main__":
    unittest.main()
