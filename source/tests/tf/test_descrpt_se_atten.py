# SPDX-License-Identifier: LGPL-3.0-or-later
import inspect
import unittest

import numpy as np
from packaging.version import parse as parse_version

from deepmd.tf.descriptor import (
    DescrptSeAtten,
)
from deepmd.tf.env import (
    tf,
)
from deepmd.tf.utils.type_embed import (
    TypeEmbedNet,
)

from .common import (
    DataSystem,
    gen_data,
    j_loader,
)

GLOBAL_ENER_FLOAT_PRECISION = tf.float64
GLOBAL_TF_FLOAT_PRECISION = tf.float64
GLOBAL_NP_FLOAT_PRECISION = np.float64


@unittest.skipIf(
    parse_version(tf.__version__) < parse_version("1.15"),
    f"The current tf version {tf.__version__} is too low to run the new testing model.",
)
class TestModel(tf.test.TestCase):
    def setUp(self) -> None:
        gen_data(nframes=2)
        self.filename = __file__

    def test_descriptor_two_sides(self) -> None:
        jfile = "water_se_atten.json"
        jdata = j_loader(jfile)

        systems = jdata["systems"]
        set_pfx = "set"
        batch_size = jdata["batch_size"]
        test_size = jdata["numb_test"]
        batch_size = 2
        test_size = 1
        rcut = jdata["model"]["descriptor"]["rcut"]
        sel = jdata["model"]["descriptor"]["sel"]
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
            ntypes=ntypes,
            neuron=typeebd_param["neuron"],
            activation_function=None,
            resnet_dt=typeebd_param["resnet_dt"],
            seed=typeebd_param["seed"],
            uniform_seed=True,
            use_tebd_bias=True,
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

        sess = self.cached_session().__enter__()
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

    def test_descriptor_one_side(self) -> None:
        jfile = "water_se_atten.json"
        jdata = j_loader(jfile)

        systems = jdata["systems"]
        set_pfx = "set"
        batch_size = jdata["batch_size"]
        test_size = jdata["numb_test"]
        batch_size = 1
        test_size = 1
        rcut = jdata["model"]["descriptor"]["rcut"]
        sel = jdata["model"]["descriptor"]["sel"]
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
            ntypes=ntypes,
            neuron=typeebd_param["neuron"],
            activation_function=None,
            resnet_dt=typeebd_param["resnet_dt"],
            seed=typeebd_param["seed"],
            uniform_seed=True,
            use_tebd_bias=True,
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

        sess = self.cached_session().__enter__()
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

    def test_stripped_type_embedding_descriptor_two_sides(self) -> None:
        jfile = "water_se_atten.json"
        jdata = j_loader(jfile)

        systems = jdata["systems"]
        set_pfx = "set"
        batch_size = jdata["batch_size"]
        test_size = jdata["numb_test"]
        batch_size = 2
        test_size = 1
        rcut = jdata["model"]["descriptor"]["rcut"]
        sel = jdata["model"]["descriptor"]["sel"]
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
        jdata["model"]["descriptor"]["tebd_input_mode"] = "strip"

        # init models
        typeebd = TypeEmbedNet(
            ntypes=ntypes,
            neuron=typeebd_param["neuron"],
            activation_function=None,
            resnet_dt=typeebd_param["resnet_dt"],
            seed=typeebd_param["seed"],
            uniform_seed=True,
            use_tebd_bias=True,
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

        sess = self.cached_session().__enter__()
        sess.run(tf.global_variables_initializer())
        [model_dout] = sess.run([dout], feed_dict=feed_dict_test)
        model_dout = model_dout.reshape([-1])
        np.savetxt("two1.out", model_dout.reshape([1, -1]), delimiter=",")

        ref_dout = [
            2.910296358673981606e-06,
            -3.297689549631518680e-05,
            -3.297689549631518680e-05,
            3.790996417030466402e-04,
            -3.082208958603667925e-05,
            3.544004728264616810e-04,
            -2.397997896082787038e-05,
            2.744923480535521121e-04,
            8.486866768450577558e-05,
            -9.750155670867453753e-04,
            8.680391572974659491e-07,
            -1.596948473518331016e-05,
            -1.596948473518331016e-05,
            3.249686279109944903e-04,
            -1.508338456375446526e-05,
            3.070479490395221158e-04,
            -1.047241469038003787e-05,
            2.085462014454144320e-04,
            4.065724483202033993e-05,
            -8.245932936607477210e-04,
            5.959146184656097397e-07,
            -1.265847984116858078e-05,
            -1.265847984116858078e-05,
            2.713109337202710531e-04,
            -1.163070862097512446e-05,
            2.491582022684395484e-04,
            -8.056716526966370043e-06,
            1.720174894426871476e-04,
            3.174999037064446555e-05,
            -6.798281455902291598e-04,
            3.145148216891492605e-06,
            -3.245585831548520087e-05,
            -3.245585831548520087e-05,
            3.350745140453206166e-04,
            -2.936281422860278914e-05,
            3.031890775924862423e-04,
            -2.408578375619038739e-05,
            2.487530226589902390e-04,
            8.275930808338685728e-05,
            -8.545607559813118157e-04,
            4.745334138737575192e-06,
            -4.149649152356857482e-05,
            -4.149649152356857482e-05,
            3.633282453063247882e-04,
            -3.734652895210441184e-05,
            3.270295126452897193e-04,
            -3.235347865588130865e-05,
            2.832387658145111447e-04,
            1.064511649928167193e-04,
            -9.321000322425568741e-04,
            1.879347284602219830e-06,
            -2.470327295060103235e-05,
            -2.470327295060103235e-05,
            3.269344178119031551e-04,
            -2.248434624179290029e-05,
            2.975826199248595046e-04,
            -1.721291645154368551e-05,
            2.273800448313684436e-04,
            6.252118835933537862e-05,
            -8.271938096175299659e-04,
        ]

        places = 10
        np.testing.assert_almost_equal(model_dout, ref_dout, places)

    def test_compressible_descriptor_two_sides(self) -> None:
        jfile = "water_se_atten.json"
        jdata = j_loader(jfile)

        systems = jdata["systems"]
        set_pfx = "set"
        batch_size = jdata["batch_size"]
        test_size = jdata["numb_test"]
        batch_size = 2
        test_size = 1
        rcut = jdata["model"]["descriptor"]["rcut"]
        sel = jdata["model"]["descriptor"]["sel"]
        ntypes = len(jdata["model"]["type_map"])

        data = DataSystem(systems, set_pfx, batch_size, test_size, rcut, run_opt=None)

        test_data = data.get_test()
        numb_test = 1

        # set parameters
        jdata["model"]["descriptor"]["neuron"] = [5, 5, 5]
        jdata["model"]["descriptor"]["axis_neuron"] = 2
        jdata["model"]["descriptor"]["attn_layer"] = 0
        jdata["model"]["descriptor"]["tebd_input_mode"] = "strip"
        typeebd_param = {
            "neuron": [5],
            "resnet_dt": False,
            "seed": 1,
        }

        # init models
        typeebd = TypeEmbedNet(
            ntypes=ntypes,
            neuron=typeebd_param["neuron"],
            activation_function=None,
            resnet_dt=typeebd_param["resnet_dt"],
            seed=typeebd_param["seed"],
            uniform_seed=True,
            use_tebd_bias=True,
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

        sess = self.cached_session().__enter__()
        sess.run(tf.global_variables_initializer())
        [model_dout] = sess.run([dout], feed_dict=feed_dict_test)
        model_dout = model_dout.reshape([-1])
        np.savetxt("two.out", model_dout.reshape([1, -1]), delimiter=",")

        ref_dout = [
            1.036073419051481218e-02,
            7.240082713918804831e-04,
            7.240082713918804831e-04,
            5.059763982689874189e-05,
            8.861222417326152997e-03,
            6.192258716986104783e-04,
            5.684670353835866163e-03,
            3.972355266104098072e-04,
            2.972080556074847488e-02,
            2.076940570592187858e-03,
            8.618769799976173929e-03,
            6.012533410070171639e-04,
            6.012533410070171639e-04,
            4.196935946091339792e-05,
            7.372555367686711193e-03,
            5.143434970398183797e-04,
            4.737331445281250247e-03,
            3.304321425798863437e-04,
            2.472045260422556581e-02,
            1.724891497741334358e-03,
            7.501652728125289375e-03,
            6.589020340101068521e-04,
            6.589020340101068521e-04,
            5.792892984552734919e-05,
            6.670726906383729442e-03,
            5.860573142386985013e-04,
            4.019558129868144349e-03,
            3.531475436354094741e-04,
            2.075417763310022021e-02,
            1.824442459657951146e-03,
            9.633741334492003025e-03,
            8.463229941979812576e-04,
            8.463229941979812576e-04,
            7.437495215274456432e-05,
            8.566452651264443857e-03,
            7.526427265583468876e-04,
            5.159465444394889175e-03,
            4.533298301373441018e-04,
            2.667538316932921080e-02,
            2.344288082726328319e-03,
            1.059332370946120330e-02,
            9.300091136049074697e-04,
            9.300091136049074697e-04,
            8.164809027640537134e-05,
            9.420348275312082423e-03,
            8.270372110426749569e-04,
            5.675669673060779359e-03,
            4.982872107808511419e-04,
            2.934228206409428968e-02,
            2.576073356437785442e-03,
            9.259830885475134332e-03,
            8.130992022541684528e-04,
            8.130992022541684528e-04,
            7.141532944786595336e-05,
            8.231990685424640450e-03,
            7.228771128684428069e-04,
            4.957665460862610216e-03,
            4.353342880152572089e-04,
            2.560566234978201017e-02,
            2.248802567567107294e-03,
        ]

        places = 10
        np.testing.assert_almost_equal(model_dout, ref_dout, places)
