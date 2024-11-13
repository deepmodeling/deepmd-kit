# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np

from deepmd.tf.descriptor.se_a_ebd_v2 import (
    DescrptSeAEbdV2,
)
from deepmd.tf.env import (
    tf,
)
from deepmd.tf.fit import (
    EnerFitting,
)
from deepmd.tf.model import (
    EnerModel,
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


class TestModel(tf.test.TestCase):
    def setUp(self) -> None:
        gen_data()

    def test_model(self) -> None:
        jfile = "water_se_a_ebd.json"
        jdata = j_loader(jfile)

        systems = jdata["systems"]
        set_pfx = "set"
        batch_size = jdata["batch_size"]
        test_size = jdata["numb_test"]
        batch_size = 1
        test_size = 1
        rcut = jdata["model"]["descriptor"]["rcut"]

        data = DataSystem(systems, set_pfx, batch_size, test_size, rcut, run_opt=None)

        test_data = data.get_test()
        numb_test = 1

        jdata["model"]["descriptor"].pop("type", None)
        jdata["model"]["type_embedding"] = {}
        jdata["model"]["type_embedding"]["neuron"] = [1]
        jdata["model"]["type_embedding"]["resnet_dt"] = False
        jdata["model"]["type_embedding"]["seed"] = 1
        typeebd_param = jdata["model"]["type_embedding"]
        typeebd = TypeEmbedNet(
            ntypes=len(jdata["model"]["descriptor"]["sel"]),
            neuron=typeebd_param["neuron"],
            activation_function=None,
            resnet_dt=typeebd_param["resnet_dt"],
            seed=typeebd_param["seed"],
            uniform_seed=True,
            use_tebd_bias=True,
            padding=True,
        )
        descrpt = DescrptSeAEbdV2(
            **jdata["model"]["descriptor"],
        )
        jdata["model"]["fitting_net"]["ntypes"] = descrpt.get_ntypes()
        jdata["model"]["fitting_net"]["dim_descrpt"] = descrpt.get_dim_out()
        jdata["model"]["fitting_net"]["dim_rot_mat_1"] = descrpt.get_dim_rot_mat_1()
        fitting = EnerFitting(
            **jdata["model"]["fitting_net"],
        )
        # fitting = EnerFitting(jdata['model']['fitting_net'], descrpt)
        model = EnerModel(descrpt, fitting, typeebd)

        # model._compute_dstats([test_data['coord']], [test_data['box']], [test_data['type']], [test_data['natoms_vec']], [test_data['default_mesh']])
        input_data = {
            "coord": [test_data["coord"]],
            "box": [test_data["box"]],
            "type": [test_data["type"]],
            "natoms_vec": [test_data["natoms_vec"]],
            "default_mesh": [test_data["default_mesh"]],
        }
        model._compute_input_stat(input_data)
        model.descrpt.bias_atom_e = data.compute_energy_shift()

        t_prop_c = tf.placeholder(tf.float32, [5], name="t_prop_c")
        t_energy = tf.placeholder(GLOBAL_ENER_FLOAT_PRECISION, [None], name="t_energy")
        t_force = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name="t_force")
        t_virial = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name="t_virial")
        t_atom_ener = tf.placeholder(
            GLOBAL_TF_FLOAT_PRECISION, [None], name="t_atom_ener"
        )
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
            suffix="se_a_ebd_v2",
            reuse=False,
        )
        energy = model_pred["energy"]
        force = model_pred["force"]
        virial = model_pred["virial"]
        atom_ener = model_pred["atom_ener"]

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
        [e, f, v] = sess.run([energy, force, virial], feed_dict=feed_dict_test)

        e = e.reshape([-1])
        f = f.reshape([-1])
        v = v.reshape([-1])

        refe = [6.100037044296185e-01]
        reff = [
            8.448651008616304e-02,
            8.613568658155157e-02,
            4.377711655236228e-03,
            9.264613309788312e-02,
            9.351200240060925e-02,
            -6.743918515275118e-03,
            -1.268078358219972e-01,
            4.855965861982662e-02,
            1.361334787979757e-04,
            4.193213089916692e-02,
            -1.324120032345251e-01,
            -4.507320444374342e-03,
            -1.314595297986654e-01,
            4.120567370248839e-02,
            7.896917575801866e-03,
            3.920259153744955e-02,
            -1.370010180699507e-01,
            -1.159523750186610e-03,
        ]
        refv = [
            -0.277134219204478,
            0.088897922530779,
            0.008633318264458,
            0.088897922530779,
            -0.292191560546969,
            -0.005709595520904,
            0.008633318264458,
            -0.005709595520904,
            -0.000682136341924,
        ]
        refe = np.reshape(refe, [-1])
        reff = np.reshape(reff, [-1])
        refv = np.reshape(refv, [-1])

        places = 6
        for ii in range(e.size):
            self.assertAlmostEqual(e[ii], refe[ii], places=places)
        for ii in range(f.size):
            self.assertAlmostEqual(f[ii], reff[ii], places=places)
        for ii in range(v.size):
            self.assertAlmostEqual(v[ii], refv[ii], places=places)
