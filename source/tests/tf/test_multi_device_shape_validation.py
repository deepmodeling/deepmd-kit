# SPDX-License-Identifier: LGPL-3.0-or-later
"""Regression tests for flattened TensorFlow custom-op input dimensions."""

import unittest

from deepmd.tf.env import (
    GLOBAL_TF_FLOAT_PRECISION,
    op_grads_module,
    op_module,
    tf,
)


class TestMultiDeviceShapeValidation(tf.test.TestCase):
    """Ensure malformed flattened widths fail before native kernel dispatch."""

    def setUp(self) -> None:
        self.sess = self.cached_session().__enter__()
        self.nloc = 2
        self.nnei = 1
        self.ndescrpt = 4
        self.natoms = tf.constant([self.nloc, self.nloc, 1], dtype=tf.int32)

    def _floats(self, width: int):
        """Create one frame of flattened floating-point custom-op input."""
        return tf.zeros([1, width], dtype=GLOBAL_TF_FLOAT_PRECISION)

    def _nlist(self, width: int):
        """Create one frame of flattened neighbor indices."""
        return tf.zeros([1, width], dtype=tf.int32)

    def test_negative_nloc_is_rejected(self) -> None:
        natoms = tf.constant([-1, 0, 0], dtype=tf.int32)

        with self.assertRaisesRegex(
            tf.errors.InvalidArgumentError,
            r"number of local atoms should be non-negative",
        ):
            self.sess.run(
                op_module.prod_force_se_a(
                    self._floats(0),
                    self._floats(0),
                    self._nlist(0),
                    natoms,
                    n_a_sel=0,
                    n_r_sel=0,
                )
            )

    def test_zero_nloc_rejects_nonempty_flattened_width(self) -> None:
        natoms = tf.constant([0, 0, 0], dtype=tf.int32)

        with self.assertRaisesRegex(
            tf.errors.InvalidArgumentError,
            r"net deriv width should be zero when nloc is zero",
        ):
            self.sess.run(
                op_module.prod_force_se_a(
                    self._floats(1),
                    self._floats(0),
                    self._nlist(0),
                    natoms,
                    n_a_sel=0,
                    n_r_sel=0,
                )
            )

    def test_zero_nloc_accepts_empty_flattened_widths(self) -> None:
        natoms = tf.constant([0, 0, 0], dtype=tf.int32)

        result = self.sess.run(
            op_module.prod_force_se_a(
                self._floats(0),
                self._floats(0),
                self._nlist(0),
                natoms,
                n_a_sel=0,
                n_r_sel=0,
            )
        )

        self.assertEqual(result.shape, (1, 0))

    def test_prod_force_rejects_partial_net_deriv_atom(self) -> None:
        # The old integer division truncated 9 / 2 to four descriptors and
        # allowed the extra value to survive until raw pointer dispatch.
        with self.assertRaisesRegex(
            tf.errors.InvalidArgumentError,
            r"net deriv width 9 should be divisible by nloc 2",
        ):
            self.sess.run(
                op_module.prod_force_se_a(
                    self._floats(self.nloc * self.ndescrpt + 1),
                    self._floats(self.nloc * self.ndescrpt * 3),
                    self._nlist(self.nloc * self.nnei),
                    self.natoms,
                    n_a_sel=self.nnei,
                    n_r_sel=0,
                )
            )

    def test_prod_force_rejects_in_deriv_width_mismatch(self) -> None:
        with self.assertRaisesRegex(
            tf.errors.InvalidArgumentError, r"number of descriptors should match"
        ):
            self.sess.run(
                op_module.prod_force_se_a(
                    self._floats(self.nloc * self.ndescrpt),
                    self._floats(self.nloc * self.ndescrpt * 3 - 1),
                    self._nlist(self.nloc * self.nnei),
                    self.natoms,
                    n_a_sel=self.nnei,
                    n_r_sel=0,
                )
            )

    def test_prod_force_r_rejects_descriptor_stride_mismatch(self) -> None:
        with self.assertRaisesRegex(
            tf.errors.InvalidArgumentError,
            r"descriptor width should equal neighbor width",
        ):
            self.sess.run(
                op_module.prod_force_se_r(
                    self._floats(self.nloc * (self.nnei + 1)),
                    self._floats(self.nloc * (self.nnei + 1) * 3),
                    self._nlist(self.nloc * self.nnei),
                    self.natoms,
                )
            )

    def test_forward_ops_reject_nall_smaller_than_nloc(self) -> None:
        natoms = tf.constant([self.nloc, self.nloc - 1, 1], dtype=tf.int32)
        operations = (
            op_module.prod_force_se_a(
                self._floats(self.nloc * self.ndescrpt),
                self._floats(self.nloc * self.ndescrpt * 3),
                self._nlist(self.nloc * self.nnei),
                natoms,
                n_a_sel=self.nnei,
                n_r_sel=0,
            ),
            op_module.prod_force_se_r(
                self._floats(self.nloc * self.nnei),
                self._floats(self.nloc * self.nnei * 3),
                self._nlist(self.nloc * self.nnei),
                natoms,
            ),
            op_module.prod_virial_se_a(
                self._floats(self.nloc * self.ndescrpt),
                self._floats(self.nloc * self.ndescrpt * 3),
                self._floats(self.nloc * self.nnei * 3),
                self._nlist(self.nloc * self.nnei),
                natoms,
                n_a_sel=self.nnei,
                n_r_sel=0,
            ),
            op_module.prod_virial_se_r(
                self._floats(self.nloc * self.nnei),
                self._floats(self.nloc * self.nnei * 3),
                self._floats(self.nloc * self.nnei * 3),
                self._nlist(self.nloc * self.nnei),
                natoms,
            ),
        )
        for operation in operations:
            with (
                self.subTest(operation=operation),
                self.assertRaisesRegex(
                    tf.errors.InvalidArgumentError,
                    r"number of all atoms should be at least nloc",
                ),
            ):
                self.sess.run(operation)

    def test_prod_virial_rejects_partial_net_deriv_atom(self) -> None:
        with self.assertRaisesRegex(
            tf.errors.InvalidArgumentError,
            r"net deriv width 9 should be divisible by nloc 2",
        ):
            self.sess.run(
                op_module.prod_virial_se_a(
                    self._floats(self.nloc * self.ndescrpt + 1),
                    self._floats(self.nloc * self.ndescrpt * 3),
                    self._floats(self.nloc * self.nnei * 3),
                    self._nlist(self.nloc * self.nnei),
                    self.natoms,
                    n_a_sel=self.nnei,
                    n_r_sel=0,
                )
            )

    def test_prod_virial_rejects_descriptor_stride_mismatch(self) -> None:
        operations = (
            op_module.prod_virial_se_a(
                self._floats(self.nloc * self.ndescrpt * 2),
                self._floats(self.nloc * self.ndescrpt * 2 * 3),
                self._floats(self.nloc * self.nnei * 3),
                self._nlist(self.nloc * self.nnei),
                self.natoms,
                n_a_sel=self.nnei,
                n_r_sel=0,
            ),
            op_module.prod_virial_se_r(
                self._floats(self.nloc * (self.nnei + 1)),
                self._floats(self.nloc * (self.nnei + 1) * 3),
                self._floats(self.nloc * self.nnei * 3),
                self._nlist(self.nloc * self.nnei),
                self.natoms,
            ),
        )
        messages = (
            r"descriptor width should be four times neighbor width",
            r"descriptor width should equal neighbor width",
        )
        for operation, message in zip(operations, messages, strict=True):
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(tf.errors.InvalidArgumentError, message),
            ):
                self.sess.run(operation)

    def test_prod_force_grad_rejects_partial_nlist_atom(self) -> None:
        # Fixed-width placeholders in the original tests rejected this feed
        # before the custom op ran, leaving its release-build checks untested.
        with self.assertRaisesRegex(
            tf.errors.InvalidArgumentError,
            r"nlist width 3 should be divisible by nloc 2",
        ):
            self.sess.run(
                op_grads_module.prod_force_se_a_grad(
                    self._floats(self.nloc * 3),
                    self._floats(self.nloc * self.ndescrpt),
                    self._floats(self.nloc * self.ndescrpt * 3),
                    self._nlist(self.nloc * self.nnei + 1),
                    self.natoms,
                    n_a_sel=self.nnei,
                    n_r_sel=0,
                )
            )

    def test_prod_force_r_grad_rejects_partial_nlist_atom(self) -> None:
        with self.assertRaisesRegex(
            tf.errors.InvalidArgumentError,
            r"nlist width 3 should be divisible by nloc 2",
        ):
            self.sess.run(
                op_grads_module.prod_force_se_r_grad(
                    self._floats(self.nloc * 3),
                    self._floats(self.nloc * self.nnei),
                    self._floats(self.nloc * self.nnei * 3),
                    self._nlist(self.nloc * self.nnei + 1),
                    self.natoms,
                )
            )

    def test_prod_virial_grad_rejects_descriptor_stride_mismatch(self) -> None:
        mismatched_ndescrpt = self.ndescrpt * 2
        with self.assertRaisesRegex(
            tf.errors.InvalidArgumentError,
            r"descriptor width should be four times neighbor width",
        ):
            self.sess.run(
                op_grads_module.prod_virial_se_a_grad(
                    self._floats(9),
                    self._floats(self.nloc * mismatched_ndescrpt),
                    self._floats(self.nloc * mismatched_ndescrpt * 3),
                    self._floats(self.nloc * self.nnei * 3),
                    self._nlist(self.nloc * self.nnei),
                    self.natoms,
                    n_a_sel=self.nnei,
                    n_r_sel=0,
                )
            )

    def test_prod_virial_grad_rejects_rij_width_mismatch(self) -> None:
        with self.assertRaisesRegex(
            tf.errors.InvalidArgumentError, r"dim of rij should be  nnei \* 3"
        ):
            self.sess.run(
                op_grads_module.prod_virial_se_a_grad(
                    self._floats(9),
                    self._floats(self.nloc * self.ndescrpt),
                    self._floats(self.nloc * self.ndescrpt * 3),
                    self._floats(self.nloc * self.nnei * 3 - 1),
                    self._nlist(self.nloc * self.nnei),
                    self.natoms,
                    n_a_sel=self.nnei,
                    n_r_sel=0,
                )
            )

    def test_prod_virial_r_grad_rejects_partial_net_deriv_atom(self) -> None:
        with self.assertRaisesRegex(
            tf.errors.InvalidArgumentError,
            r"net deriv width 3 should be divisible by nloc 2",
        ):
            self.sess.run(
                op_grads_module.prod_virial_se_r_grad(
                    self._floats(9),
                    self._floats(self.nloc * self.nnei + 1),
                    self._floats(self.nloc * self.nnei * 3),
                    self._floats(self.nloc * self.nnei * 3),
                    self._nlist(self.nloc * self.nnei),
                    self.natoms,
                )
            )

    @unittest.skipUnless(tf.test.is_gpu_available(), "GPU is required")
    def test_gpu_zero_neighbor_work_returns_zero_outputs(self) -> None:
        natoms = tf.constant([1, 1, 1], dtype=tf.int32)
        with tf.device("/GPU:0"):
            empty_floats = self._floats(0)
            empty_nlist = self._nlist(0)
            force_grad = self._floats(3)
            virial_grad = self._floats(9)
            force_a = op_module.prod_force_se_a(
                empty_floats,
                empty_floats,
                empty_nlist,
                natoms,
                n_a_sel=0,
                n_r_sel=0,
            )
            force_r = op_module.prod_force_se_r(
                empty_floats, empty_floats, empty_nlist, natoms
            )
            force_grad_a = op_grads_module.prod_force_se_a_grad(
                force_grad,
                empty_floats,
                empty_floats,
                empty_nlist,
                natoms,
                n_a_sel=0,
                n_r_sel=0,
            )
            force_grad_r = op_grads_module.prod_force_se_r_grad(
                force_grad, empty_floats, empty_floats, empty_nlist, natoms
            )
            virial_a = op_module.prod_virial_se_a(
                empty_floats,
                empty_floats,
                empty_floats,
                empty_nlist,
                natoms,
                n_a_sel=0,
                n_r_sel=0,
            )
            virial_r = op_module.prod_virial_se_r(
                empty_floats,
                empty_floats,
                empty_floats,
                empty_nlist,
                natoms,
            )
            virial_grad_a = op_grads_module.prod_virial_se_a_grad(
                virial_grad,
                empty_floats,
                empty_floats,
                empty_floats,
                empty_nlist,
                natoms,
                n_a_sel=0,
                n_r_sel=0,
            )
            virial_grad_r = op_grads_module.prod_virial_se_r_grad(
                virial_grad,
                empty_floats,
                empty_floats,
                empty_floats,
                empty_nlist,
                natoms,
            )

        tensors = (
            force_a,
            force_r,
            force_grad_a,
            force_grad_r,
            *virial_a,
            *virial_r,
            virial_grad_a,
            virial_grad_r,
        )
        for tensor in tensors:
            self.assertIn("GPU:0", tensor.device)
            result = self.sess.run(tensor)
            self.assertAllEqual(result, result * 0)


if __name__ == "__main__":
    tf.test.main()
