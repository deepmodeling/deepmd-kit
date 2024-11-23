# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np

from deepmd.tf.descriptor import (
    Descriptor,
)
from deepmd.tf.env import (
    tf,
)

from .common import (
    DataSystem,
    gen_data,
    j_loader,
)

GLOBAL_ENER_FLOAT_PRECISION = tf.float64
GLOBAL_TF_FLOAT_PRECISION = tf.float64
GLOBAL_NP_FLOAT_PRECISION = np.float64


class TestModel(tf.test.TestCase):
    def setUp(self) -> None:
        gen_data(nframes=2)

    def test_descriptor_one_side_exclude_types(self) -> None:
        """When we enable type_one_side, the descriptor should be the same
        for different types, when its environments are the same.

        Here we generates two data. The only difference is the type:
        (1) 0 1 1 1 1 1
        (2) 1 1 1 1 1 1

        When type_one_side is true, the first atom should have the same descriptor.
        Otherwise, it should be different (with random initial variables). We test
        both situation.
        """
        jfile = "water_se_a.json"
        jdata = j_loader(jfile)

        systems = jdata["systems"]
        set_pfx = "set"
        batch_size = jdata["batch_size"]
        test_size = jdata["numb_test"]
        batch_size = 1
        test_size = 1
        rcut = jdata["model"]["descriptor"]["rcut"]
        sel = jdata["model"]["descriptor"]["sel"]
        ntypes = len(sel)

        data = DataSystem(systems, set_pfx, batch_size, test_size, rcut, run_opt=None)

        test_data = data.get_test()
        numb_test = 1

        # set parameters
        jdata["model"]["descriptor"]["neuron"] = [5, 5, 5]
        jdata["model"]["descriptor"]["axis_neuron"] = 2
        jdata["model"]["descriptor"]["type_one_side"] = True
        jdata["model"]["descriptor"]["exclude_types"] = [[0, 0]]

        t_prop_c = tf.placeholder(tf.float32, [5], name="t_prop_c")
        t_coord = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name="i_coord")
        t_type = tf.placeholder(tf.int32, [None], name="i_type")
        t_natoms = tf.placeholder(tf.int32, [ntypes + 2], name="i_natoms")
        t_box = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None, 9], name="i_box")
        t_mesh = tf.placeholder(tf.int32, [None], name="i_mesh")
        is_training = tf.placeholder(tf.bool)

        # successful
        descrpt = Descriptor(**jdata["model"]["descriptor"])
        dout = descrpt.build(
            t_coord,
            t_type,
            t_natoms,
            t_box,
            t_mesh,
            {},
            reuse=False,
            suffix="_se_a_1side_exclude_types",
        )
        # failed
        descrpt_failed = Descriptor(
            **{**jdata["model"]["descriptor"], "type_one_side": False}
        )
        dout_failed = descrpt_failed.build(
            t_coord,
            t_type,
            t_natoms,
            t_box,
            t_mesh,
            {},
            reuse=False,
            suffix="_se_a_1side_exclude_types_failed",
        )

        feed_dict_test1 = {
            t_prop_c: test_data["prop_c"],
            t_coord: np.reshape(test_data["coord"][:numb_test, :], [-1]),
            t_box: test_data["box"][:numb_test, :],
            t_type: np.reshape(test_data["type"][:numb_test, :], [-1]),
            t_natoms: test_data["natoms_vec"],
            t_mesh: test_data["default_mesh"],
            is_training: False,
        }
        feed_dict_test2 = feed_dict_test1.copy()
        # original type: 0 0 1 1 1 1
        # current: 0 1 1 1 1 1
        # current: 1 1 1 1 1 1
        new_natoms1 = test_data["natoms_vec"].copy()
        new_natoms1[2] = 1
        new_natoms1[3] = 5
        new_type1 = test_data["type"].copy()
        new_type1[:numb_test, 0] = 0
        new_type1[:numb_test, 1:6] = 1
        new_natoms2 = test_data["natoms_vec"].copy()
        new_natoms2[2] = 0
        new_natoms2[3] = 6
        new_type2 = test_data["type"].copy()
        new_type2[:numb_test] = 1
        feed_dict_test1[t_type] = np.reshape(new_type1[:numb_test, :], [-1])
        feed_dict_test1[t_natoms] = new_natoms1
        feed_dict_test2[t_type] = np.reshape(new_type2[:numb_test, :], [-1])
        feed_dict_test2[t_natoms] = new_natoms2

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            [model_dout1] = sess.run([dout], feed_dict=feed_dict_test1)
            [model_dout2] = sess.run([dout], feed_dict=feed_dict_test2)
            [model_dout1_failed] = sess.run([dout_failed], feed_dict=feed_dict_test1)
            [model_dout2_failed] = sess.run([dout_failed], feed_dict=feed_dict_test2)
        model_dout1 = model_dout1.reshape([6, -1])
        model_dout2 = model_dout2.reshape([6, -1])
        model_dout1_failed = model_dout1_failed.reshape([6, -1])
        model_dout2_failed = model_dout2_failed.reshape([6, -1])

        np.testing.assert_almost_equal(model_dout1[0], model_dout2[0], 10)
        with self.assertRaises(AssertionError):
            np.testing.assert_almost_equal(
                model_dout1_failed[0], model_dout2_failed[0], 10
            )

    def test_se_r_one_side_exclude_types(self) -> None:
        """se_r."""
        jfile = "water_se_r.json"
        jdata = j_loader(jfile)

        systems = jdata["systems"]
        set_pfx = "set"
        batch_size = jdata["batch_size"]
        test_size = jdata["numb_test"]
        batch_size = 1
        test_size = 1
        rcut = jdata["model"]["descriptor"]["rcut"]
        sel = jdata["model"]["descriptor"]["sel"]
        ntypes = len(sel)

        data = DataSystem(systems, set_pfx, batch_size, test_size, rcut, run_opt=None)

        test_data = data.get_test()
        numb_test = 1

        # set parameters
        jdata["model"]["descriptor"]["neuron"] = [5, 5, 5]
        jdata["model"]["descriptor"]["type_one_side"] = True
        jdata["model"]["descriptor"]["exclude_types"] = [[0, 0]]

        t_prop_c = tf.placeholder(tf.float32, [5], name="t_prop_c")
        t_coord = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name="i_coord")
        t_type = tf.placeholder(tf.int32, [None], name="i_type")
        t_natoms = tf.placeholder(tf.int32, [ntypes + 2], name="i_natoms")
        t_box = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None, 9], name="i_box")
        t_mesh = tf.placeholder(tf.int32, [None], name="i_mesh")
        is_training = tf.placeholder(tf.bool)

        # successful
        descrpt = Descriptor(**jdata["model"]["descriptor"])
        dout = descrpt.build(
            t_coord,
            t_type,
            t_natoms,
            t_box,
            t_mesh,
            {},
            reuse=False,
            suffix="_se_r_1side_exclude_types",
        )
        # failed
        descrpt_failed = Descriptor(
            **{**jdata["model"]["descriptor"], "type_one_side": False}
        )
        dout_failed = descrpt_failed.build(
            t_coord,
            t_type,
            t_natoms,
            t_box,
            t_mesh,
            {},
            reuse=False,
            suffix="_se_r_1side_exclude_types_failed",
        )

        feed_dict_test1 = {
            t_prop_c: test_data["prop_c"],
            t_coord: np.reshape(test_data["coord"][:numb_test, :], [-1]),
            t_box: test_data["box"][:numb_test, :],
            t_type: np.reshape(test_data["type"][:numb_test, :], [-1]),
            t_natoms: test_data["natoms_vec"],
            t_mesh: test_data["default_mesh"],
            is_training: False,
        }
        feed_dict_test2 = feed_dict_test1.copy()
        # original type: 0 0 1 1 1 1
        # current: 0 1 1 1 1 1
        # current: 1 1 1 1 1 1
        new_natoms1 = test_data["natoms_vec"].copy()
        new_natoms1[2] = 1
        new_natoms1[3] = 5
        new_type1 = test_data["type"].copy()
        new_type1[:numb_test, 0] = 0
        new_type1[:numb_test, 1:6] = 1
        new_natoms2 = test_data["natoms_vec"].copy()
        new_natoms2[2] = 0
        new_natoms2[3] = 6
        new_type2 = test_data["type"].copy()
        new_type2[:numb_test] = 1
        feed_dict_test1[t_type] = np.reshape(new_type1[:numb_test, :], [-1])
        feed_dict_test1[t_natoms] = new_natoms1
        feed_dict_test2[t_type] = np.reshape(new_type2[:numb_test, :], [-1])
        feed_dict_test2[t_natoms] = new_natoms2

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            [model_dout1] = sess.run([dout], feed_dict=feed_dict_test1)
            [model_dout2] = sess.run([dout], feed_dict=feed_dict_test2)
            [model_dout1_failed] = sess.run([dout_failed], feed_dict=feed_dict_test1)
            [model_dout2_failed] = sess.run([dout_failed], feed_dict=feed_dict_test2)
        model_dout1 = model_dout1.reshape([6, -1])
        model_dout2 = model_dout2.reshape([6, -1])
        model_dout1_failed = model_dout1_failed.reshape([6, -1])
        model_dout2_failed = model_dout2_failed.reshape([6, -1])

        np.testing.assert_almost_equal(model_dout1[0], model_dout2[0], 10)
        with self.assertRaises(AssertionError):
            np.testing.assert_almost_equal(
                model_dout1_failed[0], model_dout2_failed[0], 10
            )
