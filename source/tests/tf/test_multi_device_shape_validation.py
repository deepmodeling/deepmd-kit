# SPDX-License-Identifier: LGPL-3.0-or-later
"""Regression tests for flattened TensorFlow custom-op input dimensions."""

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


if __name__ == "__main__":
    tf.test.main()
