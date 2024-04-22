# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
from packaging.version import parse as parse_version

from deepmd.tf.common import (
    j_must_have,
)
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
    def setUp(self):
        gen_data(nframes=2)

    def test_descriptor_hybrid(self):
        jfile = "water_hybrid.json"
        jdata = j_loader(jfile)

        systems = j_must_have(jdata, "systems")
        set_pfx = j_must_have(jdata, "set_prefix")
        batch_size = j_must_have(jdata, "batch_size")
        test_size = j_must_have(jdata, "numb_test")
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
        nframes = model_dout.shape[0]
        natoms = model_dout.shape[1]
        ref_dout1 = np.array(ref_dout1).reshape(nframes, natoms, -1)
        ref_dout2 = np.array(ref_dout2).reshape(nframes, natoms, -1)
        ref_dout = np.concatenate([ref_dout1, ref_dout2], axis=2)
        np.testing.assert_almost_equal(model_dout, ref_dout, places)
