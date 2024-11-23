# SPDX-License-Identifier: LGPL-3.0-or-later
"""Test pairwise DPRc features."""

import json
import unittest

import dpdata
import numpy as np
from packaging.version import parse as parse_version

from deepmd.tf import (
    DeepPotential,
)
from deepmd.tf.common import (
    j_loader,
)
from deepmd.tf.env import (
    GLOBAL_ENER_FLOAT_PRECISION,
    GLOBAL_NP_FLOAT_PRECISION,
    GLOBAL_TF_FLOAT_PRECISION,
    op_module,
    tf,
)
from deepmd.tf.model.model import (
    Model,
)
from deepmd.tf.model.pairwise_dprc import (
    gather_placeholder,
)
from deepmd.tf.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.tf.utils.sess import (
    run_sess,
)
from deepmd.utils.data import (
    DataRequirementItem,
)

from .common import (
    run_dp,
    tests_path,
)

if GLOBAL_NP_FLOAT_PRECISION == np.float32:
    default_places = 4
else:
    default_places = 10


class TestPairwiseOP(tf.test.TestCase):
    """Test dprc_pairwise_idx OP."""

    def test_op_single_frame(self) -> None:
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


class TestConvertForwardMapOP(tf.test.TestCase):
    """Test convert_forward_map OP."""

    def test_convert_forward_map(self) -> None:
        forward_qmmm_map = np.array(
            [
                [3, 4, 0, 1, 2, 10, 11],
                [3, 4, 5, 6, 7, 10, -1],
                [3, 4, 8, 9, -1, 10, -1],
            ],
            dtype=int,
        )
        natoms_qmmm = np.array([5, 7, 5], dtype=int)
        natoms = np.array([10, 12, 10], dtype=int)
        with self.cached_session() as sess:
            (
                forward_qmmm_map,
                backward_qmmm_map,
                natoms_qmmm,
                mesh_qmmm,
            ) = run_sess(
                sess,
                op_module.convert_forward_map(forward_qmmm_map, natoms_qmmm, natoms),
            )
        np.testing.assert_array_equal(
            forward_qmmm_map,
            np.array([[3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 3, 4, 8, 9, 10, 11, 10, 10]]),
        )
        np.testing.assert_array_equal(
            backward_qmmm_map,
            np.array(
                [
                    [2, 3, 4, 0, 1, -1, -1, -1, -1, -1, 14, 15],
                    [-1, -1, -1, 5, 6, 7, 8, 9, -1, -1, 16, -1],
                    [-1, -1, -1, 10, 11, -1, -1, -1, 12, 13, 17, -1],
                ]
            ),
        )
        np.testing.assert_array_equal(natoms_qmmm, np.array([14, 18, 14], dtype=int))
        np.testing.assert_array_equal(
            mesh_qmmm,
            np.array(
                [
                    14,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    6,
                    6,
                    6,
                    6,
                    6,
                    5,
                    5,
                    5,
                    5,
                    5,
                    4,
                    4,
                    4,
                    4,
                    1,
                    2,
                    3,
                    4,
                    14,
                    15,
                    0,
                    2,
                    3,
                    4,
                    14,
                    15,
                    0,
                    1,
                    3,
                    4,
                    14,
                    15,
                    0,
                    1,
                    2,
                    4,
                    14,
                    15,
                    0,
                    1,
                    2,
                    3,
                    14,
                    15,
                    6,
                    7,
                    8,
                    9,
                    16,
                    5,
                    7,
                    8,
                    9,
                    16,
                    5,
                    6,
                    8,
                    9,
                    16,
                    5,
                    6,
                    7,
                    9,
                    16,
                    5,
                    6,
                    7,
                    8,
                    16,
                    11,
                    12,
                    13,
                    17,
                    10,
                    12,
                    13,
                    17,
                    10,
                    11,
                    13,
                    17,
                    10,
                    11,
                    12,
                    17,
                ]
            ),
        )


@unittest.skipIf(
    parse_version(tf.__version__) < parse_version("1.15"),
    f"The current tf version {tf.__version__} is too low to run the new testing model.",
)
class TestPairwiseModel(tf.test.TestCase):
    def test_gather_placeholder(self) -> None:
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

    def test_model_ener(self) -> None:
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

        systems = jdata["training"]["training_data"]["systems"]
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
        # model.merge_frames = False
        model_pred = model.build(
            t_coord,
            t_type,
            t_natoms,
            t_box,
            t_mesh,
            input_dict,
            suffix="pairwise_dprc_0",
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
        sess = self.cached_session().__enter__()
        sess.run(tf.global_variables_initializer())
        [e, f, v] = sess.run([energy, force, virial], feed_dict=feed_dict_test)

        # the model is pairwise!
        self.assertAllClose(e[1] + e[2] + e[3] - 3 * e[0], e[4] - e[0])
        self.assertAllClose(f[1] + f[2] + f[3] - 3 * f[0], f[4] - f[0])
        self.assertAllClose(e[0], 4.82969, 1e-6)
        self.assertAllClose(f[0, 0], -0.104339, 1e-6)

        # test input requirement for the model
        self.assertCountEqual(
            model.input_requirement,
            [DataRequirementItem("aparam", 1, atomic=True, must=True, high_prec=False)],
        )

    def test_nloc(self) -> None:
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

        systems = jdata["training"]["training_data"]["systems"]
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
                [0, 0, 0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 5],
            ]
        )
        nloc1 = 17
        # aparam: [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 2. 3. 1. 1. 2. 2. 3. 3.]]
        feed_dict_test = {
            t_energy: np.reshape(test_data["energy"], [-1]),
            t_coord: np.reshape(test_data["coord"], [-1]),
            t_box: np.reshape(test_data["box"], (1, 9)),
            t_type: np.reshape(test_types, [-1]),
            t_natoms: [nloc1, 21, nloc1, 0, 0, 0, 0, 0],
            t_mesh: test_data["default_mesh"],
            t_aparam: np.reshape(test_data["aparam"], [-1]),
            is_training: False,
        }
        sess = self.cached_session().__enter__()
        sess.run(tf.global_variables_initializer())
        [e1, f1, v1] = sess.run([energy, force, virial], feed_dict=feed_dict_test)

        idx_map = np.concatenate([np.arange(nloc1, 21), np.arange(nloc1)])
        idx_map_inv = np.argsort(idx_map)
        feed_dict_test = {
            t_energy: np.reshape(test_data["energy"], [-1]),
            t_coord: np.reshape(np.reshape(test_data["coord"], [-1, 3])[idx_map], [-1]),
            t_box: np.reshape(test_data["box"], (1, 9)),
            t_type: np.reshape(test_types, [-1])[idx_map],
            t_natoms: [21 - nloc1, 21, 21 - nloc1, 0, 0, 0, 0, 0],
            t_mesh: test_data["default_mesh"],
            t_aparam: np.reshape(test_data["aparam"], [-1])[idx_map],
            is_training: False,
        }
        [e2, f2, v2] = sess.run([energy, force, virial], feed_dict=feed_dict_test)
        f2 = np.reshape(np.reshape(f2, [-1, 3])[idx_map_inv], f2.shape)

        feed_dict_test = {
            t_energy: np.reshape(test_data["energy"], [-1]),
            t_coord: np.reshape(test_data["coord"], [-1]),
            t_box: np.reshape(test_data["box"], (1, 9)),
            t_type: np.reshape(test_types, [-1]),
            t_natoms: [21, 21, 21, 0, 0, 0, 0, 0],
            t_mesh: test_data["default_mesh"],
            t_aparam: np.reshape(test_data["aparam"], [-1]),
            is_training: False,
        }
        [e3, f3, v3] = sess.run([energy, force, virial], feed_dict=feed_dict_test)

        np.testing.assert_allclose(e1 + e2, e3, 6)
        np.testing.assert_allclose(f1 + f2, f3, 6)
        np.testing.assert_allclose(v1 + v2, v3, 6)


def _init_models():
    system = dpdata.LabeledSystem()
    system.data["atom_names"] = ["C", "N", "O", "H", "OW", "HW"]
    system.data["coords"] = np.array(
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
    system.data["atom_types"] = np.array(
        [0, 1, 0, 2, 0, 3, 3, 3, 3, 3, 3, 3, 4, 5, 5, 4, 5, 5, 4, 5, 5]
    )
    system.data["cells"] = np.array([np.eye(3) * 30])
    nframes = 1
    natoms = 21
    system.data["coords"] = system.data["coords"].reshape([nframes, natoms, 3])
    system.data["cells"] = system.data["cells"].reshape([nframes, 3, 3])
    system.data["energies"] = np.ones(
        [
            nframes,
        ]
    )
    system.data["forces"] = np.zeros([nframes, natoms, 3])
    system.data["nopbc"] = True
    system.to_deepmd_npy(str(tests_path / "pairwise_system"), prec=np.float64)
    idxs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    np.save(str(tests_path / "pairwise_system/set.000/aparam.npy"), idxs)

    data_file = str(tests_path / "pairwise_system")
    frozen_model = str(tests_path / "dp-original-pairwise-dprc.pb")
    compressed_model = str(tests_path / "dp-compressed-pairwise-dprc.pb")
    INPUT = str(tests_path / "input.json")
    jdata = j_loader(str(tests_path / "pairwise_dprc.json"))
    jdata["training"]["training_data"]["systems"] = data_file
    with open(INPUT, "w") as fp:
        json.dump(jdata, fp, indent=4)

    ret = run_dp("dp train " + INPUT)
    np.testing.assert_equal(ret, 0, "DP train failed!")
    ret = run_dp("dp freeze -o " + frozen_model)
    np.testing.assert_equal(ret, 0, "DP freeze failed!")
    ret = run_dp("dp compress " + " -i " + frozen_model + " -o " + compressed_model)
    np.testing.assert_equal(ret, 0, "DP model compression failed!")
    return INPUT, frozen_model, compressed_model


@unittest.skipIf(
    parse_version(tf.__version__) < parse_version("1.15"),
    f"The current tf version {tf.__version__} is too low to run the new testing model.",
)
class TestPairwiseCompress(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        INPUT, FROZEN_MODEL, COMPRESSED_MODEL = _init_models()
        cls.dp_original = DeepPotential(FROZEN_MODEL)
        cls.dp_compressed = DeepPotential(COMPRESSED_MODEL)

    def setUp(self) -> None:
        self.coords = np.array(
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
        self.atype = [0, 1, 0, 2, 0, 3, 3, 3, 3, 3, 3, 3, 4, 5, 5, 4, 5, 5, 4, 5, 5]
        self.box = None
        self.idxs = np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
        ).astype(np.float64)
        # self.idxs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0]).astype(np.float64)
        self.type_map = ["C", "N", "O", "H", "OW", "HW"]

    def test_attrs(self) -> None:
        self.assertEqual(self.dp_original.get_ntypes(), len(self.type_map))
        self.assertAlmostEqual(self.dp_original.get_rcut(), 6.0, places=default_places)
        self.assertEqual(self.dp_original.get_type_map(), self.type_map)
        self.assertEqual(self.dp_original.get_dim_fparam(), 0)
        self.assertEqual(self.dp_original.get_dim_aparam(), 1)

        self.assertEqual(self.dp_compressed.get_ntypes(), len(self.type_map))
        self.assertAlmostEqual(
            self.dp_compressed.get_rcut(), 6.0, places=default_places
        )
        self.assertEqual(self.dp_compressed.get_type_map(), self.type_map)
        self.assertEqual(self.dp_compressed.get_dim_fparam(), 0)
        self.assertEqual(self.dp_compressed.get_dim_aparam(), 1)

    def test_1frame(self) -> None:
        ee0, ff0, vv0 = self.dp_original.eval(
            self.coords,
            self.box,
            self.atype,
            atomic=False,
            aparam=self.idxs,
        )
        ee1, ff1, vv1 = self.dp_compressed.eval(
            self.coords,
            self.box,
            self.atype,
            atomic=False,
            aparam=self.idxs,
        )
        # check shape of the returns
        nframes = 1
        natoms = len(self.atype)
        self.assertEqual(ee0.shape, (nframes, 1))
        self.assertEqual(ff0.shape, (nframes, natoms, 3))
        self.assertEqual(vv0.shape, (nframes, 9))
        self.assertEqual(ee1.shape, (nframes, 1))
        self.assertEqual(ff1.shape, (nframes, natoms, 3))
        self.assertEqual(vv1.shape, (nframes, 9))
        # check values
        np.testing.assert_almost_equal(ff0, ff1, default_places)
        np.testing.assert_almost_equal(ee0, ee1, default_places)
        np.testing.assert_almost_equal(vv0, vv1, default_places)

    def test_1frame_atm(self) -> None:
        ee0, ff0, vv0, ae0, av0 = self.dp_original.eval(
            self.coords,
            self.box,
            self.atype,
            atomic=True,
            aparam=self.idxs,
        )
        ee1, ff1, vv1, ae1, av1 = self.dp_compressed.eval(
            self.coords,
            self.box,
            self.atype,
            atomic=True,
            aparam=self.idxs,
        )
        # check shape of the returns
        nframes = 1
        natoms = len(self.atype)
        self.assertEqual(ee0.shape, (nframes, 1))
        self.assertEqual(ff0.shape, (nframes, natoms, 3))
        self.assertEqual(vv0.shape, (nframes, 9))
        self.assertEqual(ae0.shape, (nframes, natoms, 1))
        self.assertEqual(av0.shape, (nframes, natoms, 9))
        self.assertEqual(ee1.shape, (nframes, 1))
        self.assertEqual(ff1.shape, (nframes, natoms, 3))
        self.assertEqual(vv1.shape, (nframes, 9))
        self.assertEqual(ae1.shape, (nframes, natoms, 1))
        self.assertEqual(av1.shape, (nframes, natoms, 9))
        # check values
        np.testing.assert_almost_equal(ff0, ff1, default_places)
        np.testing.assert_almost_equal(ae0, ae1, default_places)
        np.testing.assert_almost_equal(av0, av1, default_places)
        np.testing.assert_almost_equal(ee0, ee1, default_places)
        np.testing.assert_almost_equal(vv0, vv1, default_places)

    def test_2frame_atm(self) -> None:
        coords2 = np.concatenate((self.coords, self.coords))
        box2 = None
        ee0, ff0, vv0, ae0, av0 = self.dp_original.eval(
            coords2,
            box2,
            self.atype,
            atomic=True,
            aparam=self.idxs,
        )
        ee1, ff1, vv1, ae1, av1 = self.dp_compressed.eval(
            coords2,
            box2,
            self.atype,
            atomic=True,
            aparam=self.idxs,
        )
        # check shape of the returns
        nframes = 2
        natoms = len(self.atype)
        self.assertEqual(ee0.shape, (nframes, 1))
        self.assertEqual(ff0.shape, (nframes, natoms, 3))
        self.assertEqual(vv0.shape, (nframes, 9))
        self.assertEqual(ae0.shape, (nframes, natoms, 1))
        self.assertEqual(av0.shape, (nframes, natoms, 9))
        self.assertEqual(ee1.shape, (nframes, 1))
        self.assertEqual(ff1.shape, (nframes, natoms, 3))
        self.assertEqual(vv1.shape, (nframes, 9))
        self.assertEqual(ae1.shape, (nframes, natoms, 1))
        self.assertEqual(av1.shape, (nframes, natoms, 9))

        # check values
        np.testing.assert_almost_equal(ff0, ff1, default_places)
        np.testing.assert_almost_equal(ae0, ae1, default_places)
        np.testing.assert_almost_equal(av0, av1, default_places)
        np.testing.assert_almost_equal(ee0, ee1, default_places)
        np.testing.assert_almost_equal(vv0, vv1, default_places)
