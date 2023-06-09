import inspect
import unittest

import numpy as np
from common import (
    DataSystem,
    gen_data,
    j_loader,
)
from packaging.version import parse as parse_version

from deepmd.common import (
    j_must_have,
)
from deepmd.descriptor import (
    DescrptSeAtten,
)
from deepmd.env import (
    tf,
)
from deepmd.utils.type_embed import (
    TypeEmbedNet,
)

GLOBAL_ENER_FLOAT_PRECISION = tf.float64
GLOBAL_TF_FLOAT_PRECISION = tf.float64
GLOBAL_NP_FLOAT_PRECISION = np.float64


@unittest.skipIf(
    parse_version(tf.__version__) < parse_version("1.15"),
    f"The current tf version {tf.__version__} is too low to run the new testing model.",
)
class TestModel(tf.test.TestCase):
    def setUp(self):
        gen_data(nframes=2)
        self.filename = __file__

    def test_descriptor_two_sides(self):
        jfile = "water_se_atten.json"
        jdata = j_loader(jfile)

        systems = j_must_have(jdata, "systems")
        set_pfx = j_must_have(jdata, "set_prefix")
        batch_size = j_must_have(jdata, "batch_size")
        test_size = j_must_have(jdata, "numb_test")
        batch_size = 2
        test_size = 1
        stop_batch = j_must_have(jdata, "stop_batch")
        rcut = j_must_have(jdata["model"]["descriptor"], "rcut")
        sel = j_must_have(jdata["model"]["descriptor"], "sel")
        ntypes = len(jdata["model"]["type_map"])

        data = DataSystem(systems, set_pfx, batch_size, test_size, rcut, run_opt=None)

        test_data = data.get_test()
        numb_test = 1

        # set parameters
        jdata["model"]["descriptor"]["neuron"] = [5, 5, 5]
        jdata["model"]["descriptor"]["axis_neuron"] = 2
        typeebd_param = {
            "neuron": [5],
            "resnet_dt": False,
            "seed": 1,
        }

        # init models
        typeebd = TypeEmbedNet(
            neuron=typeebd_param["neuron"],
            activation_function=None,
            resnet_dt=typeebd_param["resnet_dt"],
            seed=typeebd_param["seed"],
            uniform_seed=True,
            padding=True,
        )

        jdata["model"]["descriptor"].pop("type", None)
        jdata["model"]["descriptor"]["ntypes"] = ntypes
        descrpt = DescrptSeAtten(**jdata["model"]["descriptor"], uniform_seed=True)

        # model._compute_dstats([test_data['coord']], [test_data['box']], [test_data['type']], [test_data['natoms_vec']], [test_data['default_mesh']])
        input_data = {
            "coord": [test_data["coord"]],
            "box": [test_data["box"]],
            "type": [test_data["type"]],
            "natoms_vec": [test_data["natoms_vec"]],
            "default_mesh": [test_data["default_mesh"]],
        }
        descrpt.bias_atom_e = data.compute_energy_shift()

        t_prop_c = tf.placeholder(tf.float32, [5], name="t_prop_c")
        t_energy = tf.placeholder(GLOBAL_ENER_FLOAT_PRECISION, [None], name="t_energy")
        t_force = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name="t_force")
        t_virial = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name="t_virial")
        t_atom_ener = tf.placeholder(
            GLOBAL_TF_FLOAT_PRECISION, [None], name="t_atom_ener"
        )
        t_coord = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name="i_coord")
        t_type = tf.placeholder(tf.int32, [None], name="i_type")
        t_natoms = tf.placeholder(tf.int32, [ntypes + 2], name="i_natoms")
        t_box = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None, 9], name="i_box")
        t_mesh = tf.placeholder(tf.int32, [None], name="i_mesh")
        is_training = tf.placeholder(tf.bool)
        t_fparam = None

        type_embedding = typeebd.build(
            ntypes,
            suffix=self.filename
            + "-"
            + inspect.stack()[0][3]
            + "_se_atten_type_des_ebd_2sdies",
        )

        dout = descrpt.build(
            t_coord,
            t_type,
            t_natoms,
            t_box,
            t_mesh,
            {"type_embedding": type_embedding},
            reuse=False,
            suffix=self.filename
            + "-"
            + inspect.stack()[0][3]
            + "_se_atten_type_des_2sides",
        )

        feed_dict_test = {
            t_prop_c: test_data["prop_c"],
            t_energy: test_data["energy"][:numb_test],
            t_force: np.reshape(test_data["force"][:numb_test, :], [-1]),
            t_virial: np.reshape(test_data["virial"][:numb_test, :], [-1]),
            t_atom_ener: np.reshape(test_data["atom_ener"][:numb_test, :], [-1]),
            t_coord: np.reshape(test_data["coord"][:numb_test, :], [-1]),
            t_box: test_data["box"][:numb_test, :],
            t_type: np.reshape(test_data["type"][:numb_test, :], [-1]),
            t_natoms: test_data["natoms_vec"],
            t_mesh: test_data["default_mesh"],
            is_training: False,
        }

        sess = self.test_session().__enter__()
        sess.run(tf.global_variables_initializer())
        [model_dout] = sess.run([dout], feed_dict=feed_dict_test)
        model_dout = model_dout.reshape([-1])
        np.savetxt("two.out", model_dout.reshape([1, -1]), delimiter=",")

        ref_dout = [
            1.3503570575883254e-04,
            -9.3606804794552518e-05,
            -9.3606804794552518e-05,
            6.4931435609575354e-05,
            -3.4432462227712845e-04,
            2.3883309310633266e-04,
            -2.1612770334269806e-04,
            1.4980041766865035e-04,
            5.1902342465554648e-04,
            -3.5995814159000579e-04,
            1.0061650355705337e-04,
            -7.5148260042556979e-05,
            -7.5148260042556979e-05,
            5.6249549384058458e-05,
            -2.7820514647114664e-04,
            2.0819618461713165e-04,
            -1.5698895407951743e-04,
            1.1721016363267746e-04,
            4.0972585703616773e-04,
            -3.0650763759131061e-04,
            7.5599650998659526e-05,
            -5.8808888720672558e-05,
            -5.8808888720672558e-05,
            4.5766209906762655e-05,
            -2.1712714013251668e-04,
            1.6899894453623564e-04,
            -1.2167120597162636e-04,
            9.4648599144861605e-05,
            3.2200758382615601e-04,
            -2.5060486486718734e-04,
            1.1293831101452813e-04,
            -7.9512063028041913e-05,
            -7.9512063028041913e-05,
            5.5979262682797850e-05,
            -2.9058515610909440e-04,
            2.0457554106366365e-04,
            -1.8732839505532627e-04,
            1.3188376232775540e-04,
            4.4448730317793450e-04,
            -3.1292650304617497e-04,
            1.3015885894252541e-04,
            -8.8816609587789126e-05,
            -8.8816609587789126e-05,
            6.0613949400496957e-05,
            -3.2308121544925519e-04,
            2.2046786823295058e-04,
            -2.1781481424814687e-04,
            1.4862599684199924e-04,
            4.9955378034266583e-04,
            -3.4089120488765758e-04,
            1.0160496779809329e-04,
            -7.4538471222199861e-05,
            -7.4538471222199861e-05,
            5.4703671679263269e-05,
            -2.7394267959121653e-04,
            2.0103409637607701e-04,
            -1.6657135958432620e-04,
            1.2219321453198225e-04,
            4.1344754259964935e-04,
            -3.0339251136512270e-04,
        ]

        places = 10
        np.testing.assert_almost_equal(model_dout, ref_dout, places)

    def test_descriptor_one_side(self):
        jfile = "water_se_atten.json"
        jdata = j_loader(jfile)

        systems = j_must_have(jdata, "systems")
        set_pfx = j_must_have(jdata, "set_prefix")
        batch_size = j_must_have(jdata, "batch_size")
        test_size = j_must_have(jdata, "numb_test")
        batch_size = 1
        test_size = 1
        stop_batch = j_must_have(jdata, "stop_batch")
        rcut = j_must_have(jdata["model"]["descriptor"], "rcut")
        sel = j_must_have(jdata["model"]["descriptor"], "sel")
        ntypes = len(jdata["model"]["type_map"])

        data = DataSystem(systems, set_pfx, batch_size, test_size, rcut, run_opt=None)

        test_data = data.get_test()
        numb_test = 1

        # set parameters
        jdata["model"]["descriptor"]["neuron"] = [5, 5, 5]
        jdata["model"]["descriptor"]["axis_neuron"] = 2
        jdata["model"]["descriptor"]["type_one_side"] = True
        typeebd_param = {
            "neuron": [5],
            "resnet_dt": False,
            "seed": 1,
        }

        # init models
        typeebd = TypeEmbedNet(
            neuron=typeebd_param["neuron"],
            activation_function=None,
            resnet_dt=typeebd_param["resnet_dt"],
            seed=typeebd_param["seed"],
            uniform_seed=True,
            padding=True,
        )

        jdata["model"]["descriptor"].pop("type", None)
        jdata["model"]["descriptor"]["ntypes"] = ntypes
        descrpt = DescrptSeAtten(**jdata["model"]["descriptor"], uniform_seed=True)

        # model._compute_dstats([test_data['coord']], [test_data['box']], [test_data['type']], [test_data['natoms_vec']], [test_data['default_mesh']])
        input_data = {
            "coord": [test_data["coord"]],
            "box": [test_data["box"]],
            "type": [test_data["type"]],
            "natoms_vec": [test_data["natoms_vec"]],
            "default_mesh": [test_data["default_mesh"]],
        }
        descrpt.bias_atom_e = data.compute_energy_shift()

        t_prop_c = tf.placeholder(tf.float32, [5], name="t_prop_c")
        t_energy = tf.placeholder(GLOBAL_ENER_FLOAT_PRECISION, [None], name="t_energy")
        t_force = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name="t_force")
        t_virial = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name="t_virial")
        t_atom_ener = tf.placeholder(
            GLOBAL_TF_FLOAT_PRECISION, [None], name="t_atom_ener"
        )
        t_coord = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name="i_coord")
        t_type = tf.placeholder(tf.int32, [None], name="i_type")
        t_natoms = tf.placeholder(tf.int32, [ntypes + 2], name="i_natoms")
        t_box = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None, 9], name="i_box")
        t_mesh = tf.placeholder(tf.int32, [None], name="i_mesh")
        is_training = tf.placeholder(tf.bool)
        t_fparam = None

        type_embedding = typeebd.build(
            ntypes,
            suffix=self.filename
            + "-"
            + inspect.stack()[0][3]
            + "_se_atten_type_des_ebd_1side",
        )

        dout = descrpt.build(
            t_coord,
            t_type,
            t_natoms,
            t_box,
            t_mesh,
            {"type_embedding": type_embedding},
            reuse=False,
            suffix=self.filename
            + "-"
            + inspect.stack()[0][3]
            + "_se_atten_type_des_1side",
        )

        feed_dict_test = {
            t_prop_c: test_data["prop_c"],
            t_energy: test_data["energy"][:numb_test],
            t_force: np.reshape(test_data["force"][:numb_test, :], [-1]),
            t_virial: np.reshape(test_data["virial"][:numb_test, :], [-1]),
            t_atom_ener: np.reshape(test_data["atom_ener"][:numb_test, :], [-1]),
            t_coord: np.reshape(test_data["coord"][:numb_test, :], [-1]),
            t_box: test_data["box"][:numb_test, :],
            t_type: np.reshape(test_data["type"][:numb_test, :], [-1]),
            t_natoms: test_data["natoms_vec"],
            t_mesh: test_data["default_mesh"],
            is_training: False,
        }

        sess = self.test_session().__enter__()
        sess.run(tf.global_variables_initializer())
        [model_dout] = sess.run([dout], feed_dict=feed_dict_test)
        model_dout = model_dout.reshape([-1])
        np.savetxt("one.out", model_dout.reshape([1, -1]), delimiter=",")

        ref_dout = [
            8.9336098555659429e-05,
            -3.8921422089719007e-05,
            -3.8921422089719007e-05,
            1.6975109833017758e-05,
            -2.9184951813034413e-04,
            1.2724836941382651e-04,
            -1.8062533253590169e-04,
            7.8681048972093648e-05,
            4.2206017420030542e-04,
            -1.8398310612921889e-04,
            6.4996467281506633e-05,
            -3.0812041327073575e-05,
            -3.0812041327073575e-05,
            1.4663988013438402e-05,
            -2.3274950984084172e-04,
            1.1059587214865573e-04,
            -1.3043761448464089e-04,
            6.1788865409826698e-05,
            3.2900269837104958e-04,
            -1.5623668424484728e-04,
            5.0697927477465942e-05,
            -2.3511768544350768e-05,
            -2.3511768544350768e-05,
            1.0919808814040025e-05,
            -1.8622373494960208e-04,
            8.6439275444049409e-05,
            -1.0326450661269683e-04,
            4.7880797898768150e-05,
            2.6230208262918372e-04,
            -1.2172811361250681e-04,
            7.8240863239649707e-05,
            -3.2501260967978116e-05,
            -3.2501260967978116e-05,
            1.3502267073810926e-05,
            -2.5360559687597850e-04,
            1.0535336854834091e-04,
            -1.6047265448841568e-04,
            6.6660202062744658e-05,
            3.6833864909272261e-04,
            -1.5301457671691837e-04,
            9.1148582997925288e-05,
            -3.6614945467066073e-05,
            -3.6614945467066073e-05,
            1.4709958908948206e-05,
            -2.8364168092837332e-04,
            1.1394466218003484e-04,
            -1.8721615730559043e-04,
            7.5203967811613109e-05,
            4.1632420070310456e-04,
            -1.6724364343353009e-04,
            6.9506193268190631e-05,
            -3.0228106532898472e-05,
            -3.0228106532898472e-05,
            1.3156705594652870e-05,
            -2.3740975974826574e-04,
            1.0328972070195332e-04,
            -1.4218547815143072e-04,
            6.1827596642872941e-05,
            3.4031715116440432e-04,
            -1.4804591640658066e-04,
        ]

        places = 10
        np.testing.assert_almost_equal(model_dout, ref_dout, places)

    def test_stripped_type_embedding_descriptor_two_sides(self):
        jfile = "water_se_atten.json"
        jdata = j_loader(jfile)

        systems = j_must_have(jdata, "systems")
        set_pfx = j_must_have(jdata, "set_prefix")
        batch_size = j_must_have(jdata, "batch_size")
        test_size = j_must_have(jdata, "numb_test")
        batch_size = 2
        test_size = 1
        stop_batch = j_must_have(jdata, "stop_batch")
        rcut = j_must_have(jdata["model"]["descriptor"], "rcut")
        sel = j_must_have(jdata["model"]["descriptor"], "sel")
        ntypes = len(jdata["model"]["type_map"])

        data = DataSystem(systems, set_pfx, batch_size, test_size, rcut, run_opt=None)

        test_data = data.get_test()
        numb_test = 1

        # set parameters
        jdata["model"]["descriptor"]["neuron"] = [5, 5, 5]
        jdata["model"]["descriptor"]["axis_neuron"] = 2
        typeebd_param = {
            "neuron": [5],
            "resnet_dt": False,
            "seed": 1,
        }
        jdata["model"]["descriptor"]["compressible"] = False
        jdata["model"]["descriptor"]["stripped_type_embedding"] = True

        # init models
        typeebd = TypeEmbedNet(
            neuron=typeebd_param["neuron"],
            activation_function=None,
            resnet_dt=typeebd_param["resnet_dt"],
            seed=typeebd_param["seed"],
            uniform_seed=True,
            padding=True,
        )

        jdata["model"]["descriptor"].pop("type", None)
        jdata["model"]["descriptor"]["ntypes"] = ntypes
        descrpt = DescrptSeAtten(**jdata["model"]["descriptor"], uniform_seed=True)

        # model._compute_dstats([test_data['coord']], [test_data['box']], [test_data['type']], [test_data['natoms_vec']], [test_data['default_mesh']])
        input_data = {
            "coord": [test_data["coord"]],
            "box": [test_data["box"]],
            "type": [test_data["type"]],
            "natoms_vec": [test_data["natoms_vec"]],
            "default_mesh": [test_data["default_mesh"]],
        }
        descrpt.bias_atom_e = data.compute_energy_shift()

        t_prop_c = tf.placeholder(tf.float32, [5], name="t_prop_c")
        t_energy = tf.placeholder(GLOBAL_ENER_FLOAT_PRECISION, [None], name="t_energy")
        t_force = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name="t_force")
        t_virial = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name="t_virial")
        t_atom_ener = tf.placeholder(
            GLOBAL_TF_FLOAT_PRECISION, [None], name="t_atom_ener"
        )
        t_coord = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name="i_coord")
        t_type = tf.placeholder(tf.int32, [None], name="i_type")
        t_natoms = tf.placeholder(tf.int32, [ntypes + 2], name="i_natoms")
        t_box = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None, 9], name="i_box")
        t_mesh = tf.placeholder(tf.int32, [None], name="i_mesh")
        is_training = tf.placeholder(tf.bool)
        t_fparam = None

        type_embedding = typeebd.build(
            ntypes, suffix=self.filename + "-" + inspect.stack()[0][3]
        )

        dout = descrpt.build(
            t_coord,
            t_type,
            t_natoms,
            t_box,
            t_mesh,
            {"type_embedding": type_embedding},
            reuse=False,
            suffix=self.filename + "-" + inspect.stack()[0][3],
        )

        feed_dict_test = {
            t_prop_c: test_data["prop_c"],
            t_energy: test_data["energy"][:numb_test],
            t_force: np.reshape(test_data["force"][:numb_test, :], [-1]),
            t_virial: np.reshape(test_data["virial"][:numb_test, :], [-1]),
            t_atom_ener: np.reshape(test_data["atom_ener"][:numb_test, :], [-1]),
            t_coord: np.reshape(test_data["coord"][:numb_test, :], [-1]),
            t_box: test_data["box"][:numb_test, :],
            t_type: np.reshape(test_data["type"][:numb_test, :], [-1]),
            t_natoms: test_data["natoms_vec"],
            t_mesh: test_data["default_mesh"],
            is_training: False,
        }

        sess = self.test_session().__enter__()
        sess.run(tf.global_variables_initializer())
        [model_dout] = sess.run([dout], feed_dict=feed_dict_test)
        model_dout = model_dout.reshape([-1])
        np.savetxt("two1.out", model_dout.reshape([1, -1]), delimiter=",")

        ref_dout = [
            6.383405295803018563e-06,
            -4.970443805148654732e-05,
            -4.970443805148654732e-05,
            3.915685888463831281e-04,
            -4.475044754350870549e-05,
            3.524944903606048942e-04,
            -3.724785876580149628e-05,
            2.922698210610697768e-04,
            1.253193390649937305e-04,
            -9.866284622165712517e-04,
            2.746706080574298427e-06,
            -2.936129326691491197e-05,
            -2.936129326691491197e-05,
            3.361216906441433726e-04,
            -2.672646322529656884e-05,
            3.054685398038972568e-04,
            -1.987664884734486549e-05,
            2.233066336404597018e-04,
            7.321769925898204405e-05,
            -8.355355708215854091e-04,
            2.577054933601801410e-06,
            -2.596197919081208680e-05,
            -2.596197919081208680e-05,
            2.634209681343336554e-04,
            -2.437422597229959228e-05,
            2.471257526952611570e-04,
            -1.804241933402407212e-05,
            1.824137020231941916e-04,
            6.580156956353395276e-05,
            -6.669984436619770833e-04,
            7.430461358270195101e-06,
            -4.911199551777768784e-05,
            -4.911199551777768784e-05,
            3.246586591299433010e-04,
            -4.545306022962886084e-05,
            3.004976436904362321e-04,
            -3.943764170159576608e-05,
            2.607633588556588491e-04,
            1.265722360907320951e-04,
            -8.368076661582606740e-04,
            1.006529590251595632e-05,
            -5.947201651377735437e-05,
            -5.947201651377735437e-05,
            3.519072130827992253e-04,
            -5.475453207501003249e-05,
            3.240000254870917815e-04,
            -5.001544285922251417e-05,
            2.958951751684798311e-04,
            1.541766955454939312e-04,
            -9.123303972245934293e-04,
            5.302564262473944598e-06,
            -4.087818220772074430e-05,
            -4.087818220772074430e-05,
            3.170373643181760982e-04,
            -3.805426060094593870e-05,
            2.950399422301473447e-04,
            -3.093771463791340635e-05,
            2.394374838886431352e-04,
            1.045675931841061270e-04,
            -8.106366082292457186e-04,
        ]

        places = 10
        np.testing.assert_almost_equal(model_dout, ref_dout, places)

    def test_compressible_descriptor_two_sides(self):
        jfile = "water_se_atten.json"
        jdata = j_loader(jfile)

        systems = j_must_have(jdata, "systems")
        set_pfx = j_must_have(jdata, "set_prefix")
        batch_size = j_must_have(jdata, "batch_size")
        test_size = j_must_have(jdata, "numb_test")
        batch_size = 2
        test_size = 1
        stop_batch = j_must_have(jdata, "stop_batch")
        rcut = j_must_have(jdata["model"]["descriptor"], "rcut")
        sel = j_must_have(jdata["model"]["descriptor"], "sel")
        ntypes = len(jdata["model"]["type_map"])

        data = DataSystem(systems, set_pfx, batch_size, test_size, rcut, run_opt=None)

        test_data = data.get_test()
        numb_test = 1

        # set parameters
        jdata["model"]["descriptor"]["neuron"] = [5, 5, 5]
        jdata["model"]["descriptor"]["axis_neuron"] = 2
        jdata["model"]["descriptor"]["compressible"] = True
        jdata["model"]["descriptor"]["attn_layer"] = 0
        jdata["model"]["descriptor"]["stripped_type_embedding"] = True
        typeebd_param = {
            "neuron": [5],
            "resnet_dt": False,
            "seed": 1,
        }

        # init models
        typeebd = TypeEmbedNet(
            neuron=typeebd_param["neuron"],
            activation_function=None,
            resnet_dt=typeebd_param["resnet_dt"],
            seed=typeebd_param["seed"],
            uniform_seed=True,
            padding=True,
        )

        jdata["model"]["descriptor"].pop("type", None)
        jdata["model"]["descriptor"]["ntypes"] = ntypes
        descrpt = DescrptSeAtten(**jdata["model"]["descriptor"], uniform_seed=True)

        # model._compute_dstats([test_data['coord']], [test_data['box']], [test_data['type']], [test_data['natoms_vec']], [test_data['default_mesh']])
        input_data = {
            "coord": [test_data["coord"]],
            "box": [test_data["box"]],
            "type": [test_data["type"]],
            "natoms_vec": [test_data["natoms_vec"]],
            "default_mesh": [test_data["default_mesh"]],
        }
        descrpt.bias_atom_e = data.compute_energy_shift()

        t_prop_c = tf.placeholder(tf.float32, [5], name="t_prop_c")
        t_energy = tf.placeholder(GLOBAL_ENER_FLOAT_PRECISION, [None], name="t_energy")
        t_force = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name="t_force")
        t_virial = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name="t_virial")
        t_atom_ener = tf.placeholder(
            GLOBAL_TF_FLOAT_PRECISION, [None], name="t_atom_ener"
        )
        t_coord = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name="i_coord")
        t_type = tf.placeholder(tf.int32, [None], name="i_type")
        t_natoms = tf.placeholder(tf.int32, [ntypes + 2], name="i_natoms")
        t_box = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None, 9], name="i_box")
        t_mesh = tf.placeholder(tf.int32, [None], name="i_mesh")
        is_training = tf.placeholder(tf.bool)
        t_fparam = None

        type_embedding = typeebd.build(
            ntypes,
            suffix=self.filename
            + "-"
            + inspect.stack()[0][3]
            + "_se_atten_type_des_ebd_2sdies",
        )

        dout = descrpt.build(
            t_coord,
            t_type,
            t_natoms,
            t_box,
            t_mesh,
            {"type_embedding": type_embedding},
            reuse=False,
            suffix=self.filename
            + "-"
            + inspect.stack()[0][3]
            + "_se_atten_type_des_2sides",
        )

        feed_dict_test = {
            t_prop_c: test_data["prop_c"],
            t_energy: test_data["energy"][:numb_test],
            t_force: np.reshape(test_data["force"][:numb_test, :], [-1]),
            t_virial: np.reshape(test_data["virial"][:numb_test, :], [-1]),
            t_atom_ener: np.reshape(test_data["atom_ener"][:numb_test, :], [-1]),
            t_coord: np.reshape(test_data["coord"][:numb_test, :], [-1]),
            t_box: test_data["box"][:numb_test, :],
            t_type: np.reshape(test_data["type"][:numb_test, :], [-1]),
            t_natoms: test_data["natoms_vec"],
            t_mesh: test_data["default_mesh"],
            is_training: False,
        }

        sess = self.test_session().__enter__()
        sess.run(tf.global_variables_initializer())
        [model_dout] = sess.run([dout], feed_dict=feed_dict_test)
        model_dout = model_dout.reshape([-1])
        np.savetxt("two.out", model_dout.reshape([1, -1]), delimiter=",")

        ref_dout = [
            1.141503478024449673e-02,
            -3.178281748950724847e-05,
            -3.178281748950724847e-05,
            9.254130621715512665e-08,
            9.355933444157461115e-03,
            -2.610048686000205533e-05,
            4.941903103720379602e-03,
            -1.380808118210685460e-05,
            3.126563600452313824e-02,
            -8.708904919620347271e-05,
            9.450853881799477565e-03,
            -2.980684015958201359e-05,
            -2.980684015958201359e-05,
            1.108555048277460046e-07,
            7.781618028607465050e-03,
            -2.477633432327578414e-05,
            4.121729175224964070e-03,
            -1.322807700672256342e-05,
            2.594071094608556283e-02,
            -8.191466287679302258e-05,
            8.687152275749851840e-03,
            7.169964582221861242e-04,
            7.169964582221861242e-04,
            5.920093158205708974e-05,
            6.980509354575046600e-03,
            5.760687878669166736e-04,
            3.535108314119884531e-03,
            2.917691604627718586e-04,
            2.237560827369438829e-02,
            1.845969886863443182e-03,
            1.117485257493494588e-02,
            9.236674604597058238e-04,
            9.236674604597058238e-04,
            7.636846801269864835e-05,
            8.961677239268169004e-03,
            7.406856311644359698e-04,
            4.531907674672685724e-03,
            3.746091075776196411e-04,
            2.878475290103474707e-02,
            2.378376253052099428e-03,
            1.227893341898089741e-02,
            1.013435353901762093e-03,
            1.013435353901762093e-03,
            8.366615519789353964e-05,
            9.853113810276110246e-03,
            8.131104560097670826e-04,
            4.984313719158160738e-03,
            4.113088418105881284e-04,
            3.165040520299260834e-02,
            2.611774507262543468e-03,
            1.073868692816030805e-02,
            8.884306507846149482e-04,
            8.884306507846149482e-04,
            7.351499010825891778e-05,
            8.615254842259736923e-03,
            7.126675675525927581e-04,
            4.359946655901846685e-03,
            3.606316239946267860e-04,
            2.762742703322493970e-02,
            2.285540100738304818e-03,
        ]

        places = 10
        np.testing.assert_almost_equal(model_dout, ref_dout, places)
