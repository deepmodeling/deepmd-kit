# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
from packaging.version import parse as parse_version

from deepmd.tf.descriptor import (
    DescrptHybrid,
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
class TestHybrid(tf.test.TestCase):
    def setUp(self) -> None:
        gen_data(nframes=2)

    def test_descriptor_hybrid(self) -> None:
        jfile = "water_hybrid.json"
        jdata = j_loader(jfile)

        systems = jdata["systems"]
        set_pfx = "set"
        batch_size = 2
        test_size = 1
        rcut = 6
        ntypes = len(jdata["model"]["type_map"])

        data = DataSystem(systems, set_pfx, batch_size, test_size, rcut, run_opt=None)

        test_data = data.get_test()
        numb_test = 1

        # set parameters
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
        descrpt = DescrptHybrid(**jdata["model"]["descriptor"], uniform_seed=True)

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

        type_embedding = typeebd.build(ntypes, suffix="_hybrid")

        dout = descrpt.build(
            t_coord,
            t_type,
            t_natoms,
            t_box,
            t_mesh,
            {"type_embedding": type_embedding},
            reuse=False,
            suffix="_hybrid",
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

        ref_dout1 = [
            1.34439289e-03,
            9.95335191e-04,
            9.95335191e-04,
            7.37036883e-04,
            -2.40334638e-03,
            -1.77950629e-03,
            -1.78625508e-03,
            -1.32260520e-03,
            2.31395955e-03,
            1.71323873e-03,
            1.12402325e-03,
            8.27402417e-04,
            8.27402417e-04,
            6.09780774e-04,
            -2.00288256e-03,
            -1.47523735e-03,
            -1.48709874e-03,
            -1.09534659e-03,
            1.93208251e-03,
            1.42264043e-03,
            8.80206788e-04,
            6.73128004e-04,
            6.73128004e-04,
            5.15064921e-04,
            -1.60451351e-03,
            -1.22735660e-03,
            -1.22348138e-03,
            -9.35591125e-04,
            1.52943273e-03,
            1.16977053e-03,
            1.12942993e-03,
            8.65695820e-04,
            8.65695820e-04,
            6.63612770e-04,
            -2.06179152e-03,
            -1.58039748e-03,
            -1.57205083e-03,
            -1.20484913e-03,
            1.96339452e-03,
            1.50495265e-03,
            1.24346598e-03,
            9.52189162e-04,
            9.52189162e-04,
            7.29159208e-04,
            -2.26886213e-03,
            -1.73741924e-03,
            -1.72943295e-03,
            -1.32438322e-03,
            2.16107314e-03,
            1.65486075e-03,
            1.08331265e-03,
            8.29328473e-04,
            8.29328473e-04,
            6.35096141e-04,
            -1.97633477e-03,
            -1.51323078e-03,
            -1.50850007e-03,
            -1.15495520e-03,
            1.88273217e-03,
            1.44143407e-03,
        ]
        # below is copied from test_descript_se_atten.py
        ref_dout2 = [
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
        nframes = model_dout.shape[0]
        natoms = model_dout.shape[1]
        ref_dout1 = np.array(ref_dout1).reshape(nframes, natoms, -1)
        ref_dout2 = np.array(ref_dout2).reshape(nframes, natoms, -1)
        ref_dout = np.concatenate([ref_dout1, ref_dout2], axis=2)
        np.testing.assert_almost_equal(model_dout, ref_dout, places)
