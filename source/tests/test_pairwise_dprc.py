"""Test pairwise DPRc features."""
import numpy as np

from deepmd.env import (
    op_module,
    tf,
)
from deepmd.utils.sess import (
    run_sess,
)


class TestPairwiseOP(tf.test.TestCase):
    """Test dprc_pairwise_idx OP."""

    def test_op_single_frame(self):
        """Test dprc_pairwise_idx OP with a single frame."""
        # same as C++ tests
        idxs = np.array([[1, 1, 1, 0, 0, 2, 2, 2, 3, 3, 0, 1]], dtype=int)
        natoms = np.array([10, 12, 10], dtype=int)
        with self.cached_session() as sess:
            t_idxs = tf.convert_to_tensor(idxs, dtype=tf.int32)
            t_natoms = tf.convert_to_tensor(natoms, dtype=tf.int32)
            t_outputs = op_module.dprc_pairwise_idx(t_idxs, t_natoms)
            (
                forward_qm_map,
                backward_qm_map,
                forward_qmmm_map,
                backward_qmmm_map,
                natoms_qm,
                natoms_qmmm,
                qmmm_frame_idx,
            ) = run_sess(sess, t_outputs)
        np.testing.assert_array_equal(forward_qm_map, np.array([[3, 4, 10]], dtype=int))
        np.testing.assert_array_equal(
            backward_qm_map,
            np.array([[-1, -1, -1, 0, 1, -1, -1, -1, -1, -1, 2, -1]], dtype=int),
        )
        np.testing.assert_array_equal(
            forward_qmmm_map,
            np.array(
                [
                    [3, 4, 0, 1, 2, 10, 11],
                    [3, 4, 5, 6, 7, 10, -1],
                    [3, 4, 8, 9, -1, 10, -1],
                ],
                dtype=int,
            ),
        )
        np.testing.assert_array_equal(
            backward_qmmm_map,
            np.array(
                [
                    [2, 3, 4, 0, 1, -1, -1, -1, -1, -1, 5, 6],
                    [-1, -1, -1, 0, 1, 2, 3, 4, -1, -1, 5, -1],
                    [-1, -1, -1, 0, 1, -1, -1, -1, 2, 3, 5, -1],
                ],
                dtype=int,
            ),
        )
        np.testing.assert_array_equal(natoms_qm, np.array([2, 3, 2], dtype=int))
        np.testing.assert_array_equal(natoms_qmmm, np.array([5, 7, 5], dtype=int))
        np.testing.assert_array_equal(qmmm_frame_idx, np.array([0, 0, 0], dtype=int))
