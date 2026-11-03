# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

from deepmd.tf.env import (
    op_module,
    tf,
)


@unittest.skipUnless(
    tf.test.is_gpu_available(), reason="GPU tabulation validation requires a GPU"
)
class TestTabulateGpuSizeValidation(unittest.TestCase):
    """Ensure invalid tabulation widths are rejected before GPU launch."""

    error_message = "last_layer_size must be between 1 and 1024"

    @staticmethod
    def _build_ops(last_layer_size: int) -> dict[str, object]:
        """Build all TensorFlow tabulation variants with a common width."""
        dtype = tf.float64
        # Tensor dimensions must remain valid so a negative operator attribute
        # reaches the custom GPU kernel's validation.
        tensor_width = max(0, last_layer_size)
        table_width = max(1, 6 * last_layer_size)
        with tf.device("/CPU:0"):
            table_info = tf.constant([0.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=dtype)

        with tf.device("/GPU:0"):
            table = tf.zeros([1, table_width], dtype=dtype)
            em_x = tf.zeros([1, 1], dtype=dtype)

            em_a = tf.zeros([1, 1, 4], dtype=dtype)
            descriptor_a = tf.zeros([1, 4, tensor_width], dtype=dtype)
            dy_a = tf.zeros_like(descriptor_a)
            dz_dem_x_a = tf.zeros_like(em_x)
            dz_dem_a = tf.zeros_like(em_a)
            two_embed = tf.zeros([1, tensor_width], dtype=dtype)
            dz_dtwo = tf.zeros_like(two_embed)

            em_t = tf.zeros([1, 1, 1], dtype=dtype)
            descriptor_t = tf.zeros([1, tensor_width], dtype=dtype)
            dy_t = tf.zeros_like(descriptor_t)
            dz_dem_x_t = tf.zeros_like(em_x)
            dz_dem_t = tf.zeros_like(em_t)

            em_r = tf.zeros([1, 1], dtype=dtype)
            descriptor_r = tf.zeros([1, 1, tensor_width], dtype=dtype)
            dy_r = tf.zeros_like(descriptor_r)
            dz_dem_r = tf.zeros_like(em_r)

            return {
                "se_a_forward": op_module.tabulate_fusion_se_a(
                    table,
                    table_info,
                    em_x,
                    em_a,
                    last_layer_size=last_layer_size,
                ),
                "se_a_grad": op_module.tabulate_fusion_se_a_grad(
                    table, table_info, em_x, em_a, dy_a, descriptor_a
                ),
                "se_a_grad_grad": op_module.tabulate_fusion_se_a_grad_grad(
                    table,
                    table_info,
                    em_x,
                    em_a,
                    dz_dem_x_a,
                    dz_dem_a,
                    descriptor_a,
                ),
                "se_atten_forward": op_module.tabulate_fusion_se_atten(
                    table,
                    table_info,
                    em_x,
                    em_a,
                    two_embed,
                    last_layer_size=last_layer_size,
                ),
                "se_atten_grad": op_module.tabulate_fusion_se_atten_grad(
                    table,
                    table_info,
                    em_x,
                    em_a,
                    two_embed,
                    dy_a,
                    descriptor_a,
                ),
                "se_atten_grad_grad": (
                    op_module.tabulate_fusion_se_atten_grad_grad(
                        table,
                        table_info,
                        em_x,
                        em_a,
                        two_embed,
                        dz_dem_x_a,
                        dz_dem_a,
                        dz_dtwo,
                        descriptor_a,
                    )
                ),
                "se_t_forward": op_module.tabulate_fusion_se_t(
                    table,
                    table_info,
                    em_x,
                    em_t,
                    last_layer_size=last_layer_size,
                ),
                "se_t_grad": op_module.tabulate_fusion_se_t_grad(
                    table, table_info, em_x, em_t, dy_t, descriptor_t
                ),
                "se_t_grad_grad": op_module.tabulate_fusion_se_t_grad_grad(
                    table,
                    table_info,
                    em_x,
                    em_t,
                    dz_dem_x_t,
                    dz_dem_t,
                    descriptor_t,
                ),
                "se_r_forward": op_module.tabulate_fusion_se_r(
                    table,
                    table_info,
                    em_r,
                    last_layer_size=last_layer_size,
                ),
                "se_r_grad": op_module.tabulate_fusion_se_r_grad(
                    table, table_info, em_r, dy_r, descriptor_r
                ),
                "se_r_grad_grad": op_module.tabulate_fusion_se_r_grad_grad(
                    table, table_info, em_r, dz_dem_r, descriptor_r
                ),
            }

    def _assert_invalid(self, last_layer_size: int, names: tuple[str, ...]) -> None:
        graph = tf.Graph()
        with graph.as_default():
            ops = self._build_ops(last_layer_size)

        self._assert_invalid_ops(
            graph, {name: ops[name] for name in names}, last_layer_size
        )

    def _assert_invalid_ops(
        self, graph: tf.Graph, ops: dict[str, object], last_layer_size: int
    ) -> None:
        config = tf.ConfigProto(allow_soft_placement=False)
        # Constant folding can execute an all-constant custom op on the CPU
        # before placement, bypassing the GPU-only launch validation.
        config.graph_options.optimizer_options.opt_level = -1
        config.graph_options.rewrite_options.disable_meta_optimizer = True
        config.gpu_options.allow_growth = True
        with tf.Session(graph=graph, config=config) as sess:
            for name, op in ops.items():
                with self.subTest(op=name, last_layer_size=last_layer_size):
                    with self.assertRaisesRegex(
                        tf.errors.InvalidArgumentError, self.error_message
                    ):
                        sess.run(op)

    def test_rejects_oversized_width_for_all_gpu_paths(self) -> None:
        """Cover every forward, first-gradient, and grad-grad GPU wrapper."""
        names = tuple(
            f"{descriptor}_{stage}"
            for descriptor in ("se_a", "se_atten", "se_t", "se_r")
            for stage in ("forward", "grad", "grad_grad")
        )
        self._assert_invalid(1025, names)

    def test_rejects_zero_width(self) -> None:
        """Cover attribute-derived and descriptor-derived zero widths."""
        self._assert_invalid(0, ("se_a_forward", "se_t_grad"))

    def test_rejects_negative_forward_width(self) -> None:
        """Reach GPU validation with a negative last_layer_size attribute."""
        self._assert_invalid(
            -1,
            ("se_a_forward", "se_atten_forward", "se_t_forward", "se_r_forward"),
        )

    def test_rejects_descriptor_width_before_integer_narrowing(self) -> None:
        """Validate the TensorShape's int64 width before converting to native int."""
        # This width wraps to 1 if converted to a 32-bit int before validation.
        # A zero atom dimension keeps every descriptor and gradient empty.
        last_layer_size = 2**32 + 1
        graph = tf.Graph()
        with graph.as_default():
            dtype = tf.float64
            with tf.device("/CPU:0"):
                table_info = tf.constant([0.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=dtype)
            with tf.device("/GPU:0"):
                table = tf.zeros([1, 6], dtype=dtype)
                em_x = tf.zeros([0, 1], dtype=dtype)
                em = tf.zeros([0, 1, 4], dtype=dtype)
                descriptor = tf.zeros([0, 4, last_layer_size], dtype=dtype)
                dy = tf.zeros_like(descriptor)
                op = op_module.tabulate_fusion_se_a_grad(
                    table, table_info, em_x, em, dy, descriptor
                )
        self._assert_invalid_ops(graph, {"se_a_grad": op}, last_layer_size)


if __name__ == "__main__":
    unittest.main()
