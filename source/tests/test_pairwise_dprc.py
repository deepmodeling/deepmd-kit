# SPDX-License-Identifier: LGPL-3.0-or-later
"""Test pairwise DPRc features."""
import unittest

import dpdata
import numpy as np
from common import (
    tests_path,
)
from pkg_resources import (
    parse_version,
)

from deepmd.common import (
    j_loader,
    j_must_have,
)
from deepmd.env import (
    GLOBAL_ENER_FLOAT_PRECISION,
    GLOBAL_TF_FLOAT_PRECISION,
    op_module,
    tf,
)
from deepmd.model.model import (
    Model,
)
from deepmd.model.pairwise_dprc import (
    gather_placeholder,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
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


@unittest.skipIf(
    parse_version(tf.__version__) < parse_version("1.15"),
    f"The current tf version {tf.__version__} is too low to run the new testing model.",
)
class TestPairwiseModel(tf.test.TestCase):
    def test_gather_placeholder(self):
        coord = np.arange(12 * 3, dtype=np.float64).reshape(1, 12, 3)
        idxs = np.array([[1, 1, 1, 0, 0, 2, 2, 2, 3, 3, 0, 1]], dtype=int)
        natoms = np.array([10, 12, 10], dtype=int)
        with self.cached_session() as sess:
            t_idxs = tf.convert_to_tensor(idxs, dtype=tf.int32)
            t_natoms = tf.convert_to_tensor(natoms, dtype=tf.int32)
            t_coord = tf.convert_to_tensor(coord, dtype=tf.float32)
            (
                t_forward_qm_map,
                t_backward_qm_map,
                t_forward_qmmm_map,
                t_backward_qmmm_map,
                t_natoms_qm,
                t_natoms_qmmm,
                t_qmmm_frame_idx,
            ) = op_module.dprc_pairwise_idx(t_idxs, t_natoms)

            t_coord_qm = gather_placeholder(t_coord, t_forward_qm_map)
            t_coord_qmmm = gather_placeholder(
                tf.gather(t_coord, t_qmmm_frame_idx), t_forward_qmmm_map
            )

            coord_qm, coord_qmmm = run_sess(sess, [t_coord_qm, t_coord_qmmm])

        np.testing.assert_array_equal(
            coord_qm,
            np.array(
                [
                    [
                        [9, 10, 11],
                        [12, 13, 14],
                        [30, 31, 32],
                    ]
                ],
                dtype=np.float64,
            ),
        )
        np.testing.assert_array_equal(
            coord_qmmm,
            np.array(
                [
                    [
                        [9, 10, 11],
                        [12, 13, 14],
                        [0, 1, 2],
                        [3, 4, 5],
                        [6, 7, 8],
                        [30, 31, 32],
                        [33, 34, 35],
                    ],
                    [
                        [9, 10, 11],
                        [12, 13, 14],
                        [15, 16, 17],
                        [18, 19, 20],
                        [21, 22, 23],
                        [30, 31, 32],
                        [0, 0, 0],
                    ],
                    [
                        [9, 10, 11],
                        [12, 13, 14],
                        [24, 25, 26],
                        [27, 28, 29],
                        [0, 0, 0],
                        [30, 31, 32],
                        [0, 0, 0],
                    ],
                ],
                dtype=np.float64,
            ),
        )

    def test_model_ener(self):
        jfile = tests_path / "pairwise_dprc.json"
        jdata = j_loader(jfile)
        model = Model(**jdata["model"])

        sys = dpdata.LabeledSystem()
        sys.data["atom_names"] = ["C", "N", "O", "H", "OW", "HW"]
        sys.data["coords"] = np.array(
            [
                2.48693,
                -0.12642,
                0.45320,
                3.86292,
                -0.00082,
                0.07286,
                4.19135,
                0.35148,
                -1.21253,
                3.35886,
                0.58875,
                -2.08423,
                5.67422,
                0.44076,
                -1.45160,
                2.40712,
                -0.32538,
                1.52137,
                2.04219,
                -0.93912,
                -0.12445,
                1.98680,
                0.81574,
                0.21261,
                4.57186,
                -0.33026,
                0.71127,
                6.24532,
                0.18814,
                -0.55212,
                5.92647,
                1.46447,
                -1.74069,
                5.95030,
                -0.25321,
                -2.24804,
                -0.32794,
                1.50468,
                0.83176,
                0.23662,
                2.24068,
                1.13166,
                -0.24528,
                1.59132,
                -0.14907,
                -0.50371,
                -1.24800,
                -0.05601,
                -0.28305,
                -1.84629,
                0.67555,
                -0.68673,
                -0.40535,
                0.41384,
                0.38397,
                0.80987,
                -1.90358,
                1.30191,
                0.68503,
                -2.22909,
                0.11626,
                -0.11276,
                -1.70506,
            ]
        ).reshape(1, 21, 3)
        sys.data["atom_types"] = np.array(
            [0, 1, 0, 2, 0, 3, 3, 3, 3, 3, 3, 3, 4, 5, 5, 4, 5, 5, 4, 5, 5]
        )
        sys.data["cells"] = np.array([np.eye(3) * 30])
        nframes = 1
        natoms = 21
        sys.data["coords"] = sys.data["coords"].reshape([nframes, natoms, 3])
        sys.data["cells"] = sys.data["cells"].reshape([nframes, 3, 3])
        sys.data["energies"] = np.ones(
            [
                nframes,
            ]
        )
        sys.data["forces"] = np.zeros([nframes, natoms, 3])
        sys.data["nopbc"] = True
        sys.to_deepmd_npy("system", prec=np.float64)
        idxs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        np.save("system/set.000/aparam.npy", idxs)

        systems = j_must_have(jdata, "systems")
        set_pfx = j_must_have(jdata, "set_prefix")
        batch_size = 1
        test_size = 1
        rcut = model.get_rcut()

        data = DeepmdDataSystem(systems, batch_size, test_size, rcut)
        data.add("energy", 1, atomic=False, must=True, high_prec=True)
        data.add("aparam", 1, atomic=True, must=True, high_prec=True)
        test_data = data.get_test()

        t_energy = tf.placeholder(GLOBAL_ENER_FLOAT_PRECISION, [None], name="t_energy")
        t_coord = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name="i_coord")
        t_type = tf.placeholder(tf.int32, [None], name="i_type")
        t_natoms = tf.placeholder(tf.int32, [model.get_ntypes() + 2], name="i_natoms")
        t_box = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None, 9], name="i_box")
        t_mesh = tf.placeholder(tf.int32, [None], name="i_mesh")
        is_training = tf.placeholder(tf.bool)
        t_aparam = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name="i_aparam")
        input_dict = {}
        input_dict["aparam"] = t_aparam

        model.data_stat(data)
        model_pred = model.build(
            t_coord,
            t_type,
            t_natoms,
            t_box,
            t_mesh,
            input_dict,
            suffix="se_a_atom_ener_0",
            reuse=False,
        )
        energy = model_pred["energy"]
        force = model_pred["force"]
        virial = model_pred["virial"]

        test_types = np.array(
            [
                [
                    0,
                    0,
                    0,
                    1,
                    2,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ],
                [0, 0, 0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 4, -1, -1, 5, 5, -1, -1, -1, -1],
                [0, 0, 0, 1, 2, 3, 3, 3, 3, 3, 3, 3, -1, 4, -1, -1, -1, 5, 5, -1, -1],
                [0, 0, 0, 1, 2, 3, 3, 3, 3, 3, 3, 3, -1, -1, 4, -1, -1, -1, -1, 5, 5],
                [0, 0, 0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 5],
            ]
        )
        # aparam: [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 2. 3. 1. 1. 2. 2. 3. 3.]]
        feed_dict_test = {
            t_energy: np.reshape(np.tile(test_data["energy"], 5), [-1]),
            t_coord: np.reshape(np.tile(test_data["coord"], 5), [-1]),
            t_box: np.reshape(np.tile(test_data["box"], 5), (5, 9)),
            t_type: np.reshape(test_types, [-1]),
            t_natoms: [21, 21, 21, 0, 0, 0, 0, 0],
            t_mesh: test_data["default_mesh"],
            t_aparam: np.reshape(np.tile(test_data["aparam"], 5), [-1]),
            is_training: False,
        }
        sess = self.test_session().__enter__()
        sess.run(tf.global_variables_initializer())
        [e, f, v] = sess.run([energy, force, virial], feed_dict=feed_dict_test)

        # the model is pairwise!
        self.assertAllClose(e[1] + e[2] + e[3] - 3 * e[0], e[4] - e[0])
        self.assertAllClose(f[1] + f[2] + f[3] - 3 * f[0], f[4] - f[0])
