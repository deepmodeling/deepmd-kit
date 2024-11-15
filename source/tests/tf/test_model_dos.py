# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np

from deepmd.tf.descriptor import (
    DescrptSeA,
)
from deepmd.tf.env import (
    tf,
)
from deepmd.tf.fit import (
    DOSFitting,
)
from deepmd.tf.model import (
    DOSModel,
)

from .common import (
    DataSystem,
    del_data,
    gen_data,
    j_loader,
)

GLOBAL_ENER_FLOAT_PRECISION = tf.float64
GLOBAL_TF_FLOAT_PRECISION = tf.float64
GLOBAL_NP_FLOAT_PRECISION = np.float64


class TestModel(tf.test.TestCase):
    def setUp(self) -> None:
        gen_data()

    def tearDown(self) -> None:
        del_data()

    def test_model(self) -> None:
        jfile = "train_dos.json"
        jdata = j_loader(jfile)

        systems = jdata["training"]["systems"]
        set_pfx = "set"
        batch_size = jdata["training"]["batch_size"]
        test_size = jdata["training"]["numb_test"]
        batch_size = 1
        test_size = 1
        rcut = jdata["model"]["descriptor"]["rcut"]

        data = DataSystem(systems, set_pfx, batch_size, test_size, rcut, run_opt=None)

        test_data = data.get_test()
        numb_test = 1
        numb_dos = 100
        natoms = test_data["type"].shape[1]
        test_data["atom_dos"] = np.zeros([numb_test, natoms * numb_dos])
        test_data["dos"] = np.zeros([numb_test, numb_dos])

        jdata["model"]["fitting_net"]["numb_dos"] = numb_dos
        jdata["model"]["descriptor"]["neuron"] = [5, 5, 5]
        jdata["model"]["descriptor"]["axis_neuron"] = 2

        jdata["model"]["descriptor"].pop("type", None)
        descrpt = DescrptSeA(**jdata["model"]["descriptor"], uniform_seed=True)

        jdata["model"]["fitting_net"].pop("type", None)
        jdata["model"]["fitting_net"]["ntypes"] = descrpt.get_ntypes()
        jdata["model"]["fitting_net"]["dim_descrpt"] = descrpt.get_dim_out()
        fitting = DOSFitting(**jdata["model"]["fitting_net"], uniform_seed=True)
        model = DOSModel(descrpt, fitting)

        input_data = {
            "coord": [test_data["coord"]],
            "box": [test_data["box"]],
            "type": [test_data["type"]],
            "natoms_vec": [test_data["natoms_vec"]],
            "default_mesh": [test_data["default_mesh"]],
        }
        model._compute_input_stat(input_data)

        t_prop_c = tf.placeholder(tf.float32, [5], name="t_prop_c")
        t_atom_dos = tf.placeholder(
            GLOBAL_TF_FLOAT_PRECISION, [None], name="t_atom_dos"
        )
        t_dos = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name="t_dos")
        t_coord = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name="i_coord")
        t_type = tf.placeholder(tf.int32, [None], name="i_type")
        t_natoms = tf.placeholder(tf.int32, [model.ntypes + 2], name="i_natoms")
        t_box = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None, 9], name="i_box")
        t_mesh = tf.placeholder(tf.int32, [None], name="i_mesh")
        is_training = tf.placeholder(tf.bool)
        t_fparam = None

        model_pred = model.build(
            t_coord,
            t_type,
            t_natoms,
            t_box,
            t_mesh,
            t_fparam,
            suffix="se_a_dos",
            reuse=False,
        )
        dos = model_pred["dos"]
        atom_dos = model_pred["atom_dos"]

        feed_dict_test = {
            t_prop_c: test_data["prop_c"],
            t_dos: np.reshape(test_data["dos"][:numb_test, :], [-1]),
            t_atom_dos: np.reshape(test_data["atom_dos"][:numb_test, :], [-1]),
            t_coord: np.reshape(test_data["coord"][:numb_test, :], [-1]),
            t_box: test_data["box"][:numb_test, :],
            t_type: np.reshape(test_data["type"][:numb_test, :], [-1]),
            t_natoms: test_data["natoms_vec"],
            t_mesh: test_data["default_mesh"],
            is_training: False,
        }

        sess = self.cached_session().__enter__()
        sess.run(tf.global_variables_initializer())
        [pred_dos, pred_atom_dos] = sess.run([dos, atom_dos], feed_dict=feed_dict_test)

        ref_dos = np.array(
            [
                -1.98049388,
                -4.58033899,
                -6.95508968,
                -0.79619016,
                15.58478599,
                2.7636959,
                -2.99147438,
                -6.94430794,
                -1.77877141,
                -4.5000298,
                -3.12026893,
                -8.42191319,
                3.8991195,
                4.85271854,
                8.30541908,
                -1.0435944,
                -4.42713079,
                19.70011955,
                -6.53945284,
                0.85064846,
                4.36868488,
                4.77303801,
                3.00829128,
                0.70043584,
                -7.69047143,
                -0.0647043,
                4.56830405,
                -8.67154404,
                -4.64015279,
                -7.62202078,
                -8.97078455,
                -5.19685985,
                -1.66080276,
                -6.03225716,
                -4.06780949,
                -0.53046979,
                8.3543131,
                -1.84893576,
                2.42669245,
                -4.26357086,
                -11.33995527,
                10.98529887,
                -10.70000829,
                -4.50179402,
                -1.34978505,
                -8.83091676,
                -11.85324773,
                -3.6305035,
                2.89933807,
                4.65750153,
                1.25464578,
                -5.06196944,
                10.05305042,
                -1.83868447,
                -11.57017913,
                -2.03900316,
                -3.37235187,
                -1.37010554,
                -2.93769471,
                0.11905709,
                6.99367431,
                3.48640865,
                -4.16242817,
                4.44778342,
                -0.98405367,
                1.81581506,
                -5.31481686,
                8.72426364,
                4.78954098,
                7.67879332,
                -5.00417706,
                0.79717914,
                -3.20581567,
                -2.96034568,
                6.31165294,
                2.9891188,
                -12.2013139,
                -13.67496037,
                4.77102881,
                2.71353286,
                6.83849229,
                -3.50400312,
                1.3839428,
                -5.07550528,
                -8.5623218,
                17.64081151,
                6.46051807,
                2.89067584,
                14.23057359,
                17.85941763,
                -6.46129295,
                -3.43602528,
                -3.13520203,
                4.45313732,
                -5.23012576,
                -2.65929557,
                -0.66191939,
                4.47530191,
                9.33992973,
                -6.29808733,
            ]
        )

        ref_ados_1 = np.array(
            [
                -0.33019322,
                -0.76332506,
                -1.15916671,
                -0.13280604,
                2.59680457,
                0.46049936,
                -0.49890317,
                -1.15747878,
                -0.2964021,
                -0.74953328,
                -0.51982728,
                -1.40236941,
                0.64964525,
                0.8084967,
                1.38371838,
                -0.17366078,
                -0.7374573,
                3.28274006,
                -1.09001574,
                0.14173437,
                0.7269307,
                0.79545851,
                0.50059876,
                0.1165872,
                -1.28106632,
                -0.01107711,
                0.76139868,
                -1.44547292,
                -0.77352498,
                -1.26982082,
                -1.49597963,
                -0.86647985,
                -0.27728806,
                -1.00542829,
                -0.67794229,
                -0.08898442,
                1.39205396,
                -0.30789099,
                0.40393006,
                -0.70982912,
                -1.88961087,
                1.830906,
                -1.78326071,
                -0.75013615,
                -0.22537904,
                -1.47257916,
                -1.9756803,
                -0.60493323,
                0.48350014,
                0.77676571,
                0.20885468,
                -0.84351691,
                1.67501205,
                -0.30662021,
                -1.92884376,
                -0.34021625,
                -0.56212664,
                -0.22884438,
                -0.4891038,
                0.0199886,
                1.16506594,
                0.58068956,
                -0.69376438,
                0.74156043,
                -0.16360848,
                0.30303168,
                -0.88639571,
                1.453683,
                0.79818052,
                1.2796414,
                -0.8335433,
                0.13359098,
                -0.53425462,
                -0.4939294,
                1.05247266,
                0.49770575,
                -2.03320073,
                -2.27918678,
                0.79462598,
                0.45187804,
                1.13925239,
                -0.58410808,
                0.23092918,
                -0.84611213,
                -1.42726499,
                2.93985879,
                1.07635712,
                0.48092082,
                2.37197063,
                2.97647126,
                -1.07670667,
                -0.57300341,
                -0.52316403,
                0.74274268,
                -0.87188274,
                -0.44279998,
                -0.11060956,
                0.74619435,
                1.55646754,
                -1.05043903,
            ]
        )

        places = 4
        np.testing.assert_almost_equal(pred_dos, ref_dos, places)
        np.testing.assert_almost_equal(np.sum(pred_atom_dos, axis=0), ref_dos, places)
        np.testing.assert_almost_equal(pred_atom_dos[0], ref_ados_1, places)
