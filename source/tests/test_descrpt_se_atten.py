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

        type_embedding = typeebd.build(ntypes, suffix="_se_atten_type_des_ebd_2sdies")

        dout = descrpt.build(
            t_coord,
            t_type,
            t_natoms,
            t_box,
            t_mesh,
            {"type_embedding": type_embedding},
            reuse=False,
            suffix="_se_atten_type_des_2sides",
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

        ref_dout = [
            9.345601811834877858e-06,
            -5.952429657436736067e-05,
            -5.952429657436736067e-05,
            3.798219339938377500e-04,
            -5.473801266600750138e-05,
            3.492689212282017045e-04,
            -4.701511803776506468e-05,
            2.996014142828410851e-04,
            1.519318254663050438e-04,
            -9.691679729305131179e-04,
            4.639301212229915600e-06,
            -3.880307472438465351e-05,
            -3.880307472438465351e-05,
            3.263510366915401122e-04,
            -3.602511945944678490e-05,
            3.030070692648913970e-04,
            -2.749648518890264080e-05,
            2.306635984680620104e-04,
            9.768537816050416954e-05,
            -8.212186297001088662e-04,
            3.460569880141428627e-06,
            -3.057606512460111250e-05,
            -3.057606512460111250e-05,
            2.716853957153315133e-04,
            -2.822734036378936432e-05,
            2.510848446158910584e-04,
            -2.058670330591998133e-05,
            1.836196876431213907e-04,
            7.592953891416902529e-05,
            -6.758138628497427618e-04,
            9.016224735333271687e-06,
            -5.493669637892903309e-05,
            -5.493669637892903309e-05,
            3.349227130387897179e-04,
            -5.006878854367239126e-05,
            3.053247155435676952e-04,
            -4.307577100091461854e-05,
            2.627594780740223688e-04,
            1.390650311881827983e-04,
            -8.480702102774508640e-04,
            1.194421346644617234e-05,
            -6.584805186418614701e-05,
            -6.584805186418614701e-05,
            3.630189444637618094e-04,
            -5.971894402294766291e-05,
            3.292293648749792449e-04,
            -5.411449195138936510e-05,
            2.983321514371517751e-04,
            1.677372743720770298e-04,
            -9.247324089117067367e-04,
            6.525370698008342040e-06,
            -4.616209490400820508e-05,
            -4.616209490400820508e-05,
            3.272132307784752408e-04,
            -4.228414781396409573e-05,
            2.998297009884601412e-04,
            -3.397721060608091141e-05,
            2.410596980080286660e-04,
            1.158980826260448753e-04,
            -8.219405348709558565e-04
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

        type_embedding = typeebd.build(ntypes, suffix="_se_atten_type_des_ebd_1side")

        dout = descrpt.build(
            t_coord,
            t_type,
            t_natoms,
            t_box,
            t_mesh,
            {"type_embedding": type_embedding},
            reuse=False,
            suffix="_se_atten_type_des_1side",
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

        ref_dout = [
            9.345601811834877858e-06,
            -5.952429657436736067e-05,
            -5.952429657436736067e-05,
            3.798219339938377500e-04,
            -5.473801266600750138e-05,
            3.492689212282017045e-04,
            -4.701511803776506468e-05,
            2.996014142828410851e-04,
            1.519318254663050438e-04,
            -9.691679729305131179e-04,
            4.639301212229915600e-06,
            -3.880307472438465351e-05,
            -3.880307472438465351e-05,
            3.263510366915401122e-04,
            -3.602511945944678490e-05,
            3.030070692648913970e-04,
            -2.749648518890264080e-05,
            2.306635984680620104e-04,
            9.768537816050416954e-05,
            -8.212186297001088662e-04,
            3.460569880141428627e-06,
            -3.057606512460111250e-05,
            -3.057606512460111250e-05,
            2.716853957153315133e-04,
            -2.822734036378936432e-05,
            2.510848446158910584e-04,
            -2.058670330591998133e-05,
            1.836196876431213907e-04,
            7.592953891416902529e-05,
            -6.758138628497427618e-04,
            9.016224735333271687e-06,
            -5.493669637892903309e-05,
            -5.493669637892903309e-05,
            3.349227130387897179e-04,
            -5.006878854367239126e-05,
            3.053247155435676952e-04,
            -4.307577100091461854e-05,
            2.627594780740223688e-04,
            1.390650311881827983e-04,
            -8.480702102774508640e-04,
            1.194421346644617234e-05,
            -6.584805186418614701e-05,
            -6.584805186418614701e-05,
            3.630189444637618094e-04,
            -5.971894402294766291e-05,
            3.292293648749792449e-04,
            -5.411449195138936510e-05,
            2.983321514371517751e-04,
            1.677372743720770298e-04,
            -9.247324089117067367e-04,
            6.525370698008342040e-06,
            -4.616209490400820508e-05,
            -4.616209490400820508e-05,
            3.272132307784752408e-04,
            -4.228414781396409573e-05,
            2.998297009884601412e-04,
            -3.397721060608091141e-05,
            2.410596980080286660e-04,
            1.158980826260448753e-04,
            -8.219405348709558565e-04
        ]

        places = 10
        np.testing.assert_almost_equal(model_dout, ref_dout, places)
