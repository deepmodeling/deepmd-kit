# SPDX-License-Identifier: LGPL-3.0-or-later
import os

import numpy as np

from deepmd.tf.descriptor import (
    DescrptSeAMask,
)
from deepmd.tf.env import (
    tf,
)
from deepmd.tf.infer import (
    DeepPot,
)
from deepmd.tf.utils.convert import (
    convert_pbtxt_to_pb,
)

from .common import (
    DataSystem,
    infer_path,
    j_loader,
    tests_path,
)

GLOBAL_ENER_FLOAT_PRECISION = tf.float64
GLOBAL_TF_FLOAT_PRECISION = tf.float64
GLOBAL_NP_FLOAT_PRECISION = np.float64


class TestModel(tf.test.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        convert_pbtxt_to_pb(
            str(infer_path / os.path.join("dp4mask.pbtxt")),
            str(infer_path / os.path.join("dp4mask.pb")),
        )
        cls.dp = DeepPot(str(infer_path / os.path.join("dp4mask.pb")))

    def test_dp_mask_model(self) -> None:
        dcoord = np.array(
            [
                3.345,
                -1.468,
                23.683,
                3.341,
                2.091,
                21.996,
                5.376,
                2.906,
                21.89,
                2.524,
                -1.325,
                21.655,
                5.469,
                -3.046,
                24.72,
                2.552,
                1.316,
                19.441,
                -2.18,
                -2.923,
                20.749,
                6.579,
                -3.339,
                14.201,
                3.63,
                -1.121,
                22.442,
                4.663,
                1.875,
                22.303,
                6.041,
                -0.865,
                20.327,
                2.092,
                -1.886,
                23.721,
                4.012,
                -1.461,
                24.553,
                7.598,
                -1.759,
                23.271,
                2.547,
                1.397,
                22.242,
                4.514,
                -0.114,
                18.982,
                6.496,
                0.258,
                24.107,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                5.26,
                -0.067,
                22.228,
            ]
        )
        datype = np.array(
            [
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
                0,
                0,
                0,
                0,
                1,
            ]
        )
        aparam = np.array(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1]
        )
        dbox = np.array(
            [20.0, 0, 0, 0, 20.0, 0, 0, 0, 20.0]
        )  # Not used in parctice. For interface compatibility.

        expected_f = np.array(
            [
                14.77904,
                0.64907,
                -5.1460799999999995,
                10.56518,
                47.33218,
                0.2805199999999992,
                2.505610000000001,
                -23.18689,
                25.59182,
                -14.72715,
                -11.144870000000001,
                13.46593,
                15.51214,
                4.37472,
                10.38931,
                -0.91083,
                11.84497,
                -7.1584200000000004,
                7.21265,
                4.815799999999999,
                0.82408,
                -4.26441,
                9.55675,
                11.71466,
                -10.992710000000002,
                -22.47362,
                0.6877,
                -4.03442,
                -19.13836,
                -6.72912,
                -20.3236,
                61.629020000000004,
                111.67949999999999,
                -49.76299,
                -21.67873,
                23.19499,
                -6.11966,
                2.25341,
                -18.46156,
                -2.22062,
                7.679200000000001,
                14.4782,
                8.973709999999999,
                3.34746,
                -4.26533,
                3.2023200000000003,
                -4.97059,
                4.00849,
                44.05685,
                2.49927,
                63.78073,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                105.67309,
                -1.08829,
                -64.25365,
            ]
        )

        atom_pref = np.array(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1]
        )
        atom_pref = np.repeat(atom_pref, 3)

        ee, ff, vv = self.dp.eval(dcoord, dbox, datype, aparam=aparam)
        ff = ff.reshape(expected_f.shape)

        diff_ff = np.multiply(np.square(ff - expected_f), atom_pref)
        normalized_diff_ff = np.sqrt(np.sum(diff_ff) / np.sum(atom_pref))

        assert normalized_diff_ff < 100

    def test_descriptor_se_a_mask(self) -> None:
        jfile = "zinc_se_a_mask.json"
        jdata = j_loader(jfile)

        jdata["training"]["training_data"]["systems"] = [
            str(tests_path / "data_dp_mask")
        ]
        jdata["training"]["validation_data"]["systems"] = [
            str(tests_path / "data_dp_mask")
        ]
        systems = jdata["training"]["validation_data"]["systems"]
        set_pfx = "set"
        batch_size = 2
        test_size = 1
        rcut = 20.0  # For DataSystem interface compatibility, not used in this test.
        sel = jdata["model"]["descriptor"]["sel"]
        ntypes = len(sel)
        total_atom_num = np.cumsum(sel)[-1]

        data = DataSystem(systems, set_pfx, batch_size, test_size, rcut, run_opt=None)

        test_data = data.get_test()
        numb_test = 1

        assert jdata["model"]["descriptor"]["type"] == "se_a_mask", (
            "Wrong descriptor type"
        )
        descrpt = DescrptSeAMask(**jdata["model"]["descriptor"], uniform_seed=True)

        t_coord = tf.placeholder(
            GLOBAL_TF_FLOAT_PRECISION, [None, None], name="i_coord"
        )
        t_type = tf.placeholder(tf.int32, [None, None], name="i_type")
        t_natoms = tf.placeholder(tf.int32, [ntypes + 2], name="i_natoms")
        t_box = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None, 9], name="i_box")
        t_mesh = tf.placeholder(tf.int32, [None], name="i_mesh")
        is_training = tf.placeholder(tf.bool)
        t_aparam = tf.placeholder(tf.int32, [None, None], name="i_aparam")

        dout = descrpt.build(
            t_coord,
            t_type,
            t_natoms,
            t_box,
            t_mesh,
            input_dict={"aparam": t_aparam},
            reuse=False,
        )

        # Manually set the aparam to be all zeros. So that all particles are masked as virtual atoms.
        # This is to test the correctness of the mask.
        test_data["aparam"] = np.zeros([numb_test, total_atom_num], dtype=np.int32)
        feed_dict_test = {
            t_coord: test_data["coord"][:numb_test, :],
            t_box: test_data["box"][:numb_test, :],
            t_type: test_data["type"][:numb_test, :],
            t_natoms: test_data["natoms_vec"],
            t_mesh: test_data["default_mesh"],
            t_aparam: test_data["aparam"][:numb_test, :],
            is_training: False,
        }
        sess = self.cached_session().__enter__()
        sess.run(tf.global_variables_initializer())
        [op_dout] = sess.run([dout], feed_dict=feed_dict_test)
        op_dout = op_dout.reshape([-1])

        ref_dout = np.zeros(op_dout.shape, dtype=float)

        places = 10
        np.testing.assert_almost_equal(op_dout, ref_dout, places)
