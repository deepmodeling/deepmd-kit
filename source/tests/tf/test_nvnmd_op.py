# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import unittest
from typing import (
    ClassVar,
)

import numpy as np

from deepmd.tf.env import (
    op_module,
    tf,
)


class TestOpAddFltNvnmd(tf.test.TestCase):
    def setUp(self) -> None:
        config = tf.ConfigProto()
        if int(os.environ.get("DP_AUTO_PARALLELIZATION", 0)):
            config.graph_options.rewrite_options.custom_optimizers.add().name = (
                "dpparallel"
            )
        self.sess = self.cached_session(config=config).__enter__()

    def test_op(self) -> None:
        # graph
        t_x = tf.placeholder(tf.float64, [None, 1], "t_x")
        t_y = op_module.add_flt_nvnmd(t_x, t_x)
        # feed_dic
        x = np.reshape(np.arange(0, 8**2) / 3.0, [-1, 1])
        feed_dict = {t_x: x}
        # get value and test
        self.sess.run(tf.global_variables_initializer())
        y_pred = self.sess.run(t_y, feed_dict=feed_dict)
        y_test = np.array(
            [
                0.0,
                0.66666651,
                1.33333302,
                2.0,
                2.66666603,
                3.33333206,
                4.0,
                4.66666412,
                5.33333206,
                6.0,
                6.66666412,
                7.33333206,
                8.0,
                8.66666412,
                9.33332825,
                10.0,
                10.66666412,
                11.33332825,
                12.0,
                12.66666412,
                13.33332825,
                14.0,
                14.66666412,
                15.33332825,
                16.0,
                16.66665649,
                17.33332825,
                18.0,
                18.66665649,
                19.33332825,
                20.0,
                20.66665649,
                21.33332825,
                22.0,
                22.66665649,
                23.33332825,
                24.0,
                24.66665649,
                25.33332825,
                26.0,
                26.66665649,
                27.33332825,
                28.0,
                28.66665649,
                29.33332825,
                30.0,
                30.66665649,
                31.33332825,
                32.0,
                32.66665649,
                33.33331299,
                34.0,
                34.66665649,
                35.33331299,
                36.0,
                36.66665649,
                37.33331299,
                38.0,
                38.66665649,
                39.33331299,
                40.0,
                40.66665649,
                41.33331299,
                42.0,
            ]
        )
        y_pred = np.reshape(y_pred, [-1])
        y_test = np.reshape(y_test, [-1])
        np.testing.assert_almost_equal(y_test, y_pred, 5)
        tf.reset_default_graph()


class TestOpCopyFltNvnmd(tf.test.TestCase):
    def setUp(self) -> None:
        config = tf.ConfigProto()
        if int(os.environ.get("DP_AUTO_PARALLELIZATION", 0)):
            config.graph_options.rewrite_options.custom_optimizers.add().name = (
                "dpparallel"
            )
        self.sess = self.cached_session(config=config).__enter__()

    def test_op(self) -> None:
        # graph
        t_x = tf.placeholder(tf.float64, [None, 1], "t_x")
        t_y, t_y2 = op_module.copy_flt_nvnmd(t_x)
        # feed_dic
        x = np.reshape(np.arange(0, 8) / 3.0, [-1, 1])
        feed_dict = {t_x: x}
        # get value and test
        self.sess.run(tf.global_variables_initializer())
        y_pred, y_pred2 = self.sess.run([t_y, t_y2], feed_dict=feed_dict)
        y_test = np.array(
            [0.0, 0.33333325, 0.66666651, 1.0, 1.33333302, 1.66666603, 2.0, 2.33333206]
        )
        y_pred = np.reshape(y_pred, [-1])
        y_pred2 = np.reshape(y_pred2, [-1])
        y_test = np.reshape(y_test, [-1])
        np.testing.assert_almost_equal(y_test, y_pred, 5)
        np.testing.assert_almost_equal(y_test, y_pred2, 5)
        tf.reset_default_graph()


class TestOpDotmulFltNvnmd(tf.test.TestCase):
    def setUp(self) -> None:
        config = tf.ConfigProto()
        if int(os.environ.get("DP_AUTO_PARALLELIZATION", 0)):
            config.graph_options.rewrite_options.custom_optimizers.add().name = (
                "dpparallel"
            )
        self.sess = self.cached_session(config=config).__enter__()

    def test_op(self) -> None:
        # graph
        t_x = tf.placeholder(tf.float64, [None, 4], "t_x")
        t_y = op_module.dotmul_flt_nvnmd(t_x, t_x)
        # feed_dic
        x = np.reshape(np.arange(0, 8) / 3.0, [-1, 4])
        feed_dict = {t_x: x}
        # get value and test
        self.sess.run(tf.global_variables_initializer())
        y_pred = self.sess.run(t_y, feed_dict=feed_dict)
        y_test = np.array([1.55555, 13.99998])
        y_pred = np.reshape(y_pred, [-1])
        y_test = np.reshape(y_test, [-1])
        np.testing.assert_almost_equal(y_test, y_pred, 5)
        tf.reset_default_graph()


class TestOpFltNvnmd(tf.test.TestCase):
    def setUp(self) -> None:
        config = tf.ConfigProto()
        if int(os.environ.get("DP_AUTO_PARALLELIZATION", 0)):
            config.graph_options.rewrite_options.custom_optimizers.add().name = (
                "dpparallel"
            )
        self.sess = self.cached_session(config=config).__enter__()

    def test_op(self) -> None:
        # graph
        t_x = tf.placeholder(tf.float64, [None, 4], "t_x")
        t_y = op_module.flt_nvnmd(t_x)
        # feed_dic
        x = np.reshape(np.arange(0, 8) / 3.0, [-1, 4])
        feed_dict = {t_x: x}
        # get value and test
        self.sess.run(tf.global_variables_initializer())
        y_pred = self.sess.run(t_y, feed_dict=feed_dict)
        y_test = np.array([0.0, 0.33333, 0.66667, 1.0, 1.33333, 1.66667, 2.0, 2.33333])
        y_pred = np.reshape(y_pred, [-1])
        y_test = np.reshape(y_test, [-1])
        np.testing.assert_almost_equal(y_test, y_pred, 5)
        tf.reset_default_graph()


class TestOpMatmulFitnetNvnmd(tf.test.TestCase):
    def setUp(self) -> None:
        config = tf.ConfigProto()
        if int(os.environ.get("DP_AUTO_PARALLELIZATION", 0)):
            config.graph_options.rewrite_options.custom_optimizers.add().name = (
                "dpparallel"
            )
        self.sess = self.cached_session(config=config).__enter__()

    def test_op(self) -> None:
        # graph
        t_x = tf.placeholder(tf.float64, [None, 4], "t_x")
        t_w = tf.placeholder(tf.float64, [4, 4], "t_w")
        t_y = op_module.matmul_fitnet_nvnmd(t_x, t_w, 23, 19, 1)
        # feed_dic
        x = np.reshape(np.arange(0, 16) / 3.0, [-1, 4])
        feed_dict = {t_x: x, t_w: x}
        # get value and test
        self.sess.run(tf.global_variables_initializer())
        y_pred = self.sess.run(t_y, feed_dict=feed_dict)
        y_test = np.array(
            [
                6.22222,
                6.88888,
                7.55555,
                8.22222,
                16.88887,
                19.33331,
                21.77776,
                24.22221,
                27.55553,
                31.77774,
                35.99997,
                40.2222,
                38.22219,
                44.22217,
                50.22218,
                56.22219,
            ]
        )
        y_pred = np.reshape(y_pred, [-1])
        y_test = np.reshape(y_test, [-1])
        np.testing.assert_almost_equal(y_test, y_pred, 5)
        tf.reset_default_graph()


class TestOpMatmulFltNvnmd(tf.test.TestCase):
    def setUp(self) -> None:
        config = tf.ConfigProto()
        if int(os.environ.get("DP_AUTO_PARALLELIZATION", 0)):
            config.graph_options.rewrite_options.custom_optimizers.add().name = (
                "dpparallel"
            )
        self.sess = self.cached_session(config=config).__enter__()

    def test_op(self) -> None:
        # graph
        t_x = tf.placeholder(tf.float64, [None, 4], "t_x")
        t_w = tf.placeholder(tf.float64, [4, 4], "t_w")
        t_y = op_module.matmul_flt_nvnmd(t_x, t_w, 1, 1)
        # feed_dic
        x = np.reshape(np.arange(0, 16) / 3.0, [-1, 4])
        feed_dict = {t_x: x, t_w: x}
        # get value and test
        self.sess.run(tf.global_variables_initializer())
        y_pred = self.sess.run(t_y, feed_dict=feed_dict)
        y_test = np.array(
            [
                6.22222,
                6.88888,
                7.55555,
                8.22221,
                16.88887,
                19.33331,
                21.77776,
                24.2222,
                27.55553,
                31.77774,
                35.99997,
                40.2222,
                38.22217,
                44.22217,
                50.22217,
                56.22217,
            ]
        )
        y_pred = np.reshape(y_pred, [-1])
        y_test = np.reshape(y_test, [-1])
        np.testing.assert_almost_equal(y_test, y_pred, 5)
        tf.reset_default_graph()


class TestOpMatmulFlt2fixNvnmd(tf.test.TestCase):
    def setUp(self) -> None:
        config = tf.ConfigProto()
        if int(os.environ.get("DP_AUTO_PARALLELIZATION", 0)):
            config.graph_options.rewrite_options.custom_optimizers.add().name = (
                "dpparallel"
            )
        self.sess = self.cached_session(config=config).__enter__()

    def test_op(self) -> None:
        # graph
        t_x = tf.placeholder(tf.float64, [None, 4], "t_x")
        t_w = tf.placeholder(tf.float64, [4, 4], "t_w")
        t_y = op_module.matmul_flt2fix_nvnmd(t_x, t_w, 23)
        # feed_dic
        x = np.reshape(np.arange(0, 16) / 3.0, [-1, 4])
        feed_dict = {t_x: x, t_w: x}
        # get value and test
        self.sess.run(tf.global_variables_initializer())
        y_pred = self.sess.run(t_y, feed_dict=feed_dict)
        y_test = np.array(
            [
                6.22222,
                6.88888,
                7.55555,
                8.22221,
                16.88887,
                19.33331,
                21.77776,
                24.2222,
                27.55554,
                31.77776,
                35.99997,
                40.2222,
                38.2222,
                44.2222,
                50.22217,
                56.2222,
            ]
        )
        y_pred = np.reshape(y_pred, [-1])
        y_test = np.reshape(y_test, [-1])
        np.testing.assert_almost_equal(y_test, y_pred, 5)
        tf.reset_default_graph()


class TestOpMulFltNvnmd(tf.test.TestCase):
    def setUp(self) -> None:
        config = tf.ConfigProto()
        if int(os.environ.get("DP_AUTO_PARALLELIZATION", 0)):
            config.graph_options.rewrite_options.custom_optimizers.add().name = (
                "dpparallel"
            )
        self.sess = self.cached_session(config=config).__enter__()

    def test_op(self) -> None:
        # graph
        t_x = tf.placeholder(tf.float64, [None, 4], "t_x")
        t_w = tf.placeholder(tf.float64, [4, 4], "t_w")
        t_y = op_module.mul_flt_nvnmd(t_x, t_w)
        # feed_dic
        x = np.reshape(np.arange(0, 16) / 3.0, [-1, 4])
        feed_dict = {t_x: x, t_w: x}
        # get value and test
        self.sess.run(tf.global_variables_initializer())
        y_pred = self.sess.run(t_y, feed_dict=feed_dict)
        y_test = np.array(
            [
                0.0,
                0.11111,
                0.44444,
                1.0,
                1.77778,
                2.77777,
                4.0,
                5.44444,
                7.11111,
                9.0,
                11.1111,
                13.44444,
                16.0,
                18.77776,
                21.77774,
                25.0,
            ]
        )
        y_pred = np.reshape(y_pred, [-1])
        y_test = np.reshape(y_test, [-1])
        np.testing.assert_almost_equal(y_test, y_pred, 5)
        tf.reset_default_graph()


class TestOpQuantizeNvnmd(tf.test.TestCase):
    def setUp(self) -> None:
        config = tf.ConfigProto()
        if int(os.environ.get("DP_AUTO_PARALLELIZATION", 0)):
            config.graph_options.rewrite_options.custom_optimizers.add().name = (
                "dpparallel"
            )
        self.sess = self.cached_session(config=config).__enter__()

    def test_op(self) -> None:
        # graph
        t_x = tf.placeholder(tf.float64, [None, 4], "t_x")
        t_y = op_module.quantize_nvnmd(t_x, 0, 23, 23, -1)
        # feed_dic
        x = np.reshape(np.arange(0, 8) / 3.0, [-1, 4])
        feed_dict = {t_x: x}
        # get value and test
        self.sess.run(tf.global_variables_initializer())
        y_pred = self.sess.run(t_y, feed_dict=feed_dict)
        y_test = np.array([0.0, 0.33333, 0.66667, 1.0, 1.33333, 1.66667, 2.0, 2.33333])
        y_pred = np.reshape(y_pred, [-1])
        y_test = np.reshape(y_test, [-1])
        np.testing.assert_almost_equal(y_test, y_pred, 5)
        tf.reset_default_graph()


class TestOpTanh4FltNvnmd(tf.test.TestCase):
    def setUp(self) -> None:
        config = tf.ConfigProto()
        if int(os.environ.get("DP_AUTO_PARALLELIZATION", 0)):
            config.graph_options.rewrite_options.custom_optimizers.add().name = (
                "dpparallel"
            )
        self.sess = self.cached_session(config=config).__enter__()

    def test_op(self) -> None:
        # graph
        t_x = tf.placeholder(tf.float64, [None, 4], "t_x")
        t_y = op_module.tanh4_flt_nvnmd(t_x)
        # feed_dic
        x = np.reshape(np.arange(0, 8) / 3.0, [-1, 4])
        feed_dict = {t_x: x}
        # get value and test
        self.sess.run(tf.global_variables_initializer())
        y_pred = self.sess.run(t_y, feed_dict=feed_dict)
        y_test = np.array([0.0, 0.32485, 0.60494, 0.8125, 0.93827, 0.99151, 1.0, 1.0])
        y_pred = np.reshape(y_pred, [-1])
        y_test = np.reshape(y_test, [-1])
        np.testing.assert_almost_equal(y_test, y_pred, 5)
        tf.reset_default_graph()


class TestOpMapFltNvnmd(tf.test.TestCase):
    """Verify the mapping op's input contract and out-of-range behavior."""

    # Two rows of two output channels; every polynomial is the constant d, so
    # the mapped value identifies the selected table row.
    table_values: ClassVar[list[list[float]]] = [
        [0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 11.0],
        [0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 21.0],
    ]
    table_grad_values: ClassVar[list[list[float]]] = [
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0],
        [0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0],
    ]

    def test_out_of_range_values_clamp_to_the_table_edge(self) -> None:
        """Below/above the table evaluates the first/last row, as mapt.py does."""
        sample_count = 4096
        x = tf.placeholder(tf.float64, [sample_count, 1])
        table = tf.constant(self.table_values, dtype=tf.float64)
        table_grad = tf.constant(self.table_grad_values, dtype=tf.float64)
        table_info = tf.constant([0.0, 2.0, 1.0, 0.0, 2.0], dtype=tf.float64)

        mapped = op_module.map_flt_nvnmd(x, table, table_grad, table_info)
        gradient = tf.gradients(tf.reduce_sum(mapped), x)[0]

        warm_x = np.tile([[0.25], [1.25]], (sample_count // 2, 1))
        test_x = np.full((sample_count, 1), 0.25)
        test_x[:6, 0] = [np.nan, -1.0, 0.25, 1.25, 2.0, 3.0]
        with self.cached_session() as sess:
            # Reuse a nonzero, same-sized allocation so a skipped write cannot
            # pass merely because fresh pages happen to be zero.
            sess.run(mapped, feed_dict={x: warm_x})
            actual, actual_gradient = sess.run(
                [mapped, gradient], feed_dict={x: test_x}
            )

        # NaN and -1 clamp to row 0, while 2.0 and 3.0 clamp to row 1.
        expected = np.tile([[[10.0, 11.0]]], (sample_count, 1, 1))
        expected[:6, 0] = [
            [10.0, 11.0],
            [10.0, 11.0],
            [10.0, 11.0],
            [20.0, 21.0],
            [20.0, 21.0],
            [20.0, 21.0],
        ]
        expected_gradient = np.full((sample_count, 1), 3.0)
        expected_gradient[:6, 0] = [3.0, 3.0, 3.0, 7.0, 7.0, 7.0]

        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual_gradient, expected_gradient)

    def test_first_matching_interval_wins(self) -> None:
        """Nested fine/coarse intervals resolve like MapTable.mapping()."""
        # Fine interval [0, 1] over row 0 only; coarse interval [0, 4] over
        # row 1 only. 0.5 is inside both and must take the fine one.
        table_info = tf.constant(
            [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 4.0, 4.0, 1.0, 2.0],
            dtype=tf.float64,
        )
        x = tf.constant([[0.5], [2.0], [9.0]], dtype=tf.float64)
        mapped = op_module.map_flt_nvnmd(
            x,
            tf.constant(self.table_values, dtype=tf.float64),
            tf.constant(self.table_grad_values, dtype=tf.float64),
            table_info,
        )
        with self.cached_session() as sess:
            actual = sess.run(mapped)

        np.testing.assert_array_equal(
            actual,
            [[[10.0, 11.0]], [[20.0, 21.0]], [[20.0, 21.0]]],
        )

    def _run_invalid(self, message: str, **overrides: tf.Tensor) -> None:
        """Run the op with one input replaced and require it to be rejected."""
        inputs = {
            "x": tf.zeros([1, 1], dtype=tf.float64),
            "table": tf.zeros([1, 4], dtype=tf.float64),
            "table_grad": tf.zeros([1, 4], dtype=tf.float64),
            "table_info": tf.constant([0.0, 1.0, 1.0, 0.0, 1.0], dtype=tf.float64),
        }
        inputs.update(overrides)
        mapped = op_module.map_flt_nvnmd(**inputs)
        with self.cached_session() as sess:
            with self.assertRaisesRegex(tf.errors.InvalidArgumentError, message):
                sess.run(mapped)

    def test_accepts_the_minimal_valid_inputs(self) -> None:
        """The shared fixture must pass, so each rejection isolates one cause."""
        mapped = op_module.map_flt_nvnmd(
            tf.zeros([1, 1], dtype=tf.float64),
            tf.zeros([1, 4], dtype=tf.float64),
            tf.zeros([1, 4], dtype=tf.float64),
            tf.constant([0.0, 1.0, 1.0, 0.0, 1.0], dtype=tf.float64),
        )
        with self.cached_session() as sess:
            np.testing.assert_array_equal(sess.run(mapped), np.zeros([1, 1, 1]))

    def test_rejects_wrong_x_rank(self) -> None:
        self._run_invalid("Dim of x should be 2", x=tf.zeros([1], dtype=tf.float64))

    def test_rejects_wrong_table_rank(self) -> None:
        self._run_invalid(
            "Dim of table should be 2",
            table=tf.zeros([1, 1, 4], dtype=tf.float64),
            table_grad=tf.zeros([1, 1, 4], dtype=tf.float64),
        )

    def test_rejects_mismatched_table_gradient(self) -> None:
        self._run_invalid(
            "table_grad shape should match table",
            table_grad=tf.zeros([2, 4], dtype=tf.float64),
        )

    def test_rejects_wrong_table_info_rank(self) -> None:
        self._run_invalid(
            "Dim of table_info should be 1",
            table_info=tf.constant([[0.0, 1.0, 1.0, 0.0, 1.0]], dtype=tf.float64),
        )

    def test_rejects_table_width_not_multiple_of_four(self) -> None:
        self._run_invalid(
            "table width should be a positive multiple of 4",
            table=tf.zeros([1, 5], dtype=tf.float64),
            table_grad=tf.zeros([1, 5], dtype=tf.float64),
        )

    def test_rejects_table_info_length_not_multiple_of_five(self) -> None:
        self._run_invalid(
            "table_info length should be a positive multiple of 5",
            table_info=tf.constant([0.0, 1.0, 1.0, 0.0], dtype=tf.float64),
        )

    def test_rejects_each_invalid_interval_field(self) -> None:
        """Cover every sub-condition of the per-interval table_info check."""
        # x0 x1 dx N0 N1, one broken field per case.
        cases = {
            "x0 > x1": [1.0, 0.0, 1.0, 0.0, 1.0],
            "dx == 0": [0.0, 1.0, 0.0, 0.0, 1.0],
            "dx < 0": [0.0, 1.0, -1.0, 0.0, 1.0],
            "N0 < 0": [0.0, 1.0, 1.0, -1.0, 1.0],
            "N1 == N0": [0.0, 1.0, 1.0, 0.0, 0.0],
            "N1 < N0": [0.0, 1.0, 1.0, 1.0, 0.0],
            "N1 > rows": [0.0, 1.0, 1.0, 0.0, 2.0],
        }
        for name, info in cases.items():
            with self.subTest(case=name):
                self._run_invalid(
                    "invalid interval in table_info",
                    table_info=tf.constant(info, dtype=tf.float64),
                )


if __name__ == "__main__":
    unittest.main()
