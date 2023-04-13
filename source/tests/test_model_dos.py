import numpy as np
from common import (
    DataSystem,
    del_data,
    gen_data,
    j_loader,
)

from deepmd.common import (
    j_must_have,
)
from deepmd.descriptor import (
    DescrptSeA,
)
from deepmd.env import (
    tf,
)
from deepmd.fit import (
    DOSFitting,
)
from deepmd.model import (
    DOSModel,
)

GLOBAL_ENER_FLOAT_PRECISION = tf.float64
GLOBAL_TF_FLOAT_PRECISION = tf.float64
GLOBAL_NP_FLOAT_PRECISION = np.float64


class TestModel(tf.test.TestCase):
    def setUp(self):
        gen_data()

    def tearDown(self):
        del_data()

    def test_model(self):
        jfile = "train_dos.json"
        jdata = j_loader(jfile)

        systems = j_must_have(jdata["training"], "systems")
        set_pfx = j_must_have(jdata["training"], "set_prefix")
        batch_size = j_must_have(jdata["training"], "batch_size")
        test_size = j_must_have(jdata["training"], "numb_test")
        batch_size = 1
        test_size = 1
        stop_batch = j_must_have(jdata["training"], "stop_batch")
        rcut = j_must_have(jdata["model"]["descriptor"], "rcut")

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
        jdata["model"]["fitting_net"]["descrpt"] = descrpt
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

        sess = self.test_session().__enter__()
        sess.run(tf.global_variables_initializer())
        [pred_dos, pred_atom_dos] = sess.run([dos, atom_dos], feed_dict=feed_dict_test)

        ref_dos = np.array(
            [
                -2.98834333,
                -0.63166985,
                -3.37199568,
                -1.88397887,
                0.87560992,
                4.85426159,
                -1.22677731,
                -0.60918118,
                8.80472675,
                -1.12006829,
                -3.72653765,
                -3.03698828,
                3.50906891,
                5.55140795,
                -3.34920924,
                -4.43507641,
                -6.1729281,
                -8.34865917,
                0.14371788,
                -4.38078479,
                -6.43141133,
                4.07791938,
                7.14102837,
                -0.52347718,
                0.82663796,
                -1.64225631,
                -4.63088421,
                3.3910594,
                -9.09682274,
                1.61104204,
                4.45900773,
                -2.44688559,
                -2.83298183,
                -2.00733658,
                7.33444256,
                7.09187373,
                -1.97065392,
                0.01623084,
                -7.48861264,
                -1.17790161,
                2.77126775,
                -2.55552037,
                3.3518257,
                -0.09316856,
                -1.94521413,
                0.50089251,
                -2.75763233,
                -1.94382637,
                1.30562041,
                5.08351043,
                -1.90604837,
                -0.80030045,
                -4.87093267,
                4.18009666,
                -2.9011435,
                2.58497143,
                4.47495176,
                -0.9639419,
                8.15692179,
                0.48758731,
                -0.62264663,
                -1.70677258,
                -5.51641378,
                3.98621565,
                0.57749944,
                2.9658081,
                -4.10467591,
                -7.14827888,
                0.02838605,
                -2.48630333,
                -4.82178216,
                -0.7444178,
                2.48224802,
                -1.54683936,
                0.46969412,
                -0.0960347,
                -2.08290541,
                6.357031,
                -3.49716615,
                3.28959028,
                7.83932727,
                1.51457023,
                -4.14575033,
                0.02007839,
                4.20953773,
                3.66456664,
                -4.67441496,
                -0.13296372,
                -3.77145766,
                1.49368976,
                -2.53627817,
                -3.14188618,
                0.24991722,
                0.8770123,
                0.16635733,
                -3.15391098,
                -3.7733242,
                -2.25134676,
                1.00975552,
                1.38717682,
            ]
        )

        ref_ados_1 = np.array(
            [
                -0.33019322,
                -0.76332506,
                -0.32665648,
                -0.76601747,
                -1.16441856,
                -0.13627609,
                -1.15916671,
                -0.13280604,
                2.60139518,
                0.44470952,
                -0.48316771,
                -1.15926141,
                2.59680457,
                0.46049936,
                -0.29459777,
                -0.76433726,
                -0.52091744,
                -1.39903065,
                -0.49890317,
                -1.15747878,
                0.66585524,
                0.81804842,
                1.38592217,
                -0.18025826,
                -0.2964021,
                -0.74953328,
                -0.7427461,
                3.27935087,
                -1.09340192,
                0.1462458,
                -0.51982728,
                -1.40236941,
                0.73902497,
                0.79969456,
                0.50726592,
                0.11403234,
                0.64964525,
                0.8084967,
                -1.27543102,
                -0.00571457,
                0.7748912,
                -1.42492251,
                1.38371838,
                -0.17366078,
                -0.76119888,
                -1.26083707,
                -1.48263244,
                -0.85698727,
                -0.7374573,
                3.28274006,
                -0.27029769,
                -1.00478711,
                -0.67481511,
                -0.07978058,
                -1.09001574,
                0.14173437,
                1.4092343,
                -0.31785424,
                0.40551362,
                -0.71900495,
                0.7269307,
                0.79545851,
                -1.88407155,
                1.83983772,
                -1.78413438,
                -0.74852344,
                0.50059876,
                0.1165872,
                -0.2139368,
                -1.44989426,
                -1.96651281,
                -0.6031689,
                -1.28106632,
                -0.01107711,
                0.48796663,
                0.76500912,
                0.21308153,
                -0.85297893,
                0.76139868,
                -1.44547292,
                1.68105021,
                -0.30655702,
                -1.93123,
                -0.34294737,
                -0.77352498,
                -1.26982082,
                -0.5562998,
                -0.22048683,
                -0.48641512,
                0.01124872,
                -1.49597963,
                -0.86647985,
                1.17310075,
                0.59402879,
                -0.705076,
                0.72991794,
                -0.27728806,
                -1.00542829,
                -0.16289102,
                0.29464248,
            ]
        )

        places = 4
        np.testing.assert_almost_equal(pred_dos, ref_dos, places)
        np.testing.assert_almost_equal(np.sum(pred_atom_dos, axis=0), ref_dos, places)
        np.testing.assert_almost_equal(pred_atom_dos[0], ref_ados_1, places)
