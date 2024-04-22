# SPDX-License-Identifier: LGPL-3.0-or-later
import inspect
import unittest

import numpy as np
from packaging.version import parse as parse_version

from deepmd.tf.common import (
    j_must_have,
)
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
            ntypes=ntypes,
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

        sess = self.cached_session().__enter__()
        sess.run(tf.global_variables_initializer())
        [model_dout] = sess.run([dout], feed_dict=feed_dict_test)
        model_dout = model_dout.reshape([-1])
        np.savetxt("two.out", model_dout.reshape([1, -1]), delimiter=",")

        ref_dout = [
            1.35077997858830281628e-04,
            -9.36317565146126714985e-05,
            -9.36317565146126714985e-05,
            6.49457155161046269156e-05,
            -3.44426119482271894060e-04,
            2.38892351975707574810e-04,
            -2.16192628113445024177e-04,
            1.49838432021978586618e-04,
            5.19172506251499308108e-04,
            -3.60044742999178198160e-04,
            1.00648981900694042455e-04,
            -7.51687985725674168679e-05,
            -7.51687985725674168679e-05,
            5.62621404496089786633e-05,
            -2.78288905170686305408e-04,
            2.08248552733448707985e-04,
            -1.57037506111419247626e-04,
            1.17240613774749092711e-04,
            4.09846227953978995209e-04,
            -3.06582508385239355716e-04,
            7.56236313388503977959e-05,
            -5.88249954799233110928e-05,
            -5.88249954799233110928e-05,
            4.57767614608878164778e-05,
            -2.17191782618980676941e-04,
            1.69041932410352632298e-04,
            -1.21708419050609283887e-04,
            9.46734475047640323129e-05,
            3.22101565810662901230e-04,
            -2.50667145896081176772e-04,
            1.12972766463605449241e-04,
            -7.95331652304217509748e-05,
            -7.95331652304217509748e-05,
            5.59918979793375151091e-05,
            -2.90669309441163412500e-04,
            2.04626666596480422588e-04,
            -1.87383581443938113499e-04,
            1.31917380775058677711e-04,
            4.44613289651917854839e-04,
            -3.13002780120454830552e-04,
            1.30198051172878586420e-04,
            -8.88399346622230731045e-05,
            -8.88399346622230731045e-05,
            6.06275354032895547767e-05,
            -3.23173886613725041324e-04,
            2.20522620462074609186e-04,
            -2.17878181114203837987e-04,
            1.48663514408247710887e-04,
            4.99693951217273298233e-04,
            -3.40973735611388808521e-04,
            1.01636483586918407768e-04,
            -7.45585238544824841465e-05,
            -7.45585238544824841465e-05,
            5.47161372646580776566e-05,
            -2.74022957033491422908e-04,
            2.01084733576426032218e-04,
            -1.66621218118959135701e-04,
            1.22224760787930633501e-04,
            4.13566215420014648540e-04,
            -3.03467107774532218571e-04,
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
            ntypes=ntypes,
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

        sess = self.cached_session().__enter__()
        sess.run(tf.global_variables_initializer())
        [model_dout] = sess.run([dout], feed_dict=feed_dict_test)
        model_dout = model_dout.reshape([-1])
        np.savetxt("one.out", model_dout.reshape([1, -1]), delimiter=",")

        ref_dout = [
            8.93630739076099766573e-05,
            -3.89301763666544977088e-05,
            -3.89301763666544977088e-05,
            1.69776207161541659875e-05,
            -2.91934413405367434308e-04,
            1.27275579758193970945e-04,
            -1.80678576267614851526e-04,
            7.86981804444128273503e-05,
            4.22180092132026806885e-04,
            -1.84021204552106459797e-04,
            6.50166826308631336630e-05,
            -3.08191630112232239067e-05,
            -3.08191630112232239067e-05,
            1.46662082284045218266e-05,
            -2.32818649311590855893e-04,
            1.10619882905346373389e-04,
            -1.30477133579203922803e-04,
            6.18026466291577325669e-05,
            3.29098263271154821506e-04,
            -1.56269574751685376771e-04,
            5.07138199677916164739e-05,
            -2.35171440781703185510e-05,
            -2.35171440781703185510e-05,
            1.09213797907981395710e-05,
            -1.86279366618262112341e-04,
            8.64577620996147407865e-05,
            -1.03296053419269992513e-04,
            4.78913622480582772448e-05,
            2.62378744147910732392e-04,
            -1.21753360060300813640e-04,
            7.82644227540903690814e-05,
            -3.25084361414888650958e-05,
            -3.25084361414888650958e-05,
            1.35041631983765535098e-05,
            -2.53679234140297192677e-04,
            1.05375493947693795707e-04,
            -1.60519879294703589519e-04,
            6.66744631236456129558e-05,
            3.68443126822399244329e-04,
            -1.53045684128227086913e-04,
            9.11756668850765601567e-05,
            -3.66229408732609030826e-05,
            -3.66229408732609030826e-05,
            1.47120125015788778301e-05,
            -2.83723246380394433092e-04,
            1.13968452838666050924e-04,
            -1.87270570170312914944e-04,
            7.52199008968667767218e-05,
            4.16441090538891684186e-04,
            -1.67277425363850822723e-04,
            6.95274814976590320665e-05,
            -3.02348814013024743688e-05,
            -3.02348814013024743688e-05,
            1.31585743503078956499e-05,
            -2.37479534432029007343e-04,
            1.03311591705779548338e-04,
            -1.42227987950226271961e-04,
            6.18410015482571886070e-05,
            3.40414922285898623351e-04,
            -1.48076286203042110793e-04,
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
        jdata["model"]["descriptor"]["stripped_type_embedding"] = True

        # init models
        typeebd = TypeEmbedNet(
            ntypes=ntypes,
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

        sess = self.cached_session().__enter__()
        sess.run(tf.global_variables_initializer())
        [model_dout] = sess.run([dout], feed_dict=feed_dict_test)
        model_dout = model_dout.reshape([-1])
        np.savetxt("two1.out", model_dout.reshape([1, -1]), delimiter=",")

        ref_dout = [
            2.91097766899578214544e-06,
            -3.29852641315371480153e-05,
            -3.29852641315371480153e-05,
            3.79203396610324763253e-04,
            -3.08296489918391639377e-05,
            3.54494448654088176176e-04,
            -2.39859951795545287153e-05,
            2.74566675797922735754e-04,
            8.48899306339350606405e-05,
            -9.75279256930798588154e-04,
            8.68233546069119197236e-07,
            -1.59734540671145569350e-05,
            -1.59734540671145569350e-05,
            3.25058299172223158675e-04,
            -1.50870029997722798618e-05,
            3.07130006247707560297e-04,
            -1.04749968193353404274e-05,
            2.08603290940140382453e-04,
            4.06672203401530534743e-05,
            -8.24818142292956771496e-04,
            5.96048958156013435895e-07,
            -1.26616643393676577874e-05,
            -1.26616643393676577874e-05,
            2.71386217904519277955e-04,
            -1.16335252819255226156e-05,
            2.49225002219057890918e-04,
            -8.05872731607348350672e-06,
            1.72064906604221990903e-04,
            3.17578679792106490973e-05,
            -6.80014462388431415590e-04,
            3.14589844246059013866e-06,
            -3.24641804781093787271e-05,
            -3.24641804781093787271e-05,
            3.35166446053445504782e-04,
            -2.93700743352437964023e-05,
            3.03269488552582232397e-04,
            -2.40918900326344598056e-05,
            2.48820558204534102165e-04,
            8.27802464035270346319e-05,
            -8.54792312332452379302e-04,
            4.74647063755037437353e-06,
            -4.15071266538516597008e-05,
            -4.15071266538516597008e-05,
            3.63427481731051901445e-04,
            -3.73557622901099313961e-05,
            3.27115874272415044135e-04,
            -3.23616690622182231118e-05,
            2.83315238433851622219e-04,
            1.06478087368629440682e-04,
            -9.32351467783467118162e-04,
            1.87979034371445873837e-06,
            -2.47095892917853045061e-05,
            -2.47095892917853045061e-05,
            3.27024569668371480752e-04,
            -2.24898874228677589208e-05,
            2.97661928194053256209e-04,
            -1.72172753256989610575e-05,
            2.27442187831376464941e-04,
            6.25369616966375661696e-05,
            -8.27419096402015846574e-04,
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
        jdata["model"]["descriptor"]["attn_layer"] = 0
        jdata["model"]["descriptor"]["stripped_type_embedding"] = True
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
