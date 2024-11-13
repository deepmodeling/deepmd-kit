# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np

from deepmd.tf.descriptor.se_a_ebd import (
    DescrptSeAEbd,
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
        stop_batch = jdata["stop_batch"]
        rcut = jdata["model"]["descriptor"]["rcut"]

        data = DataSystem(systems, set_pfx, batch_size, test_size, rcut, run_opt=None)

        test_data = data.get_test()
        numb_test = 1

        jdata["model"]["descriptor"].pop("type", None)
        descrpt = DescrptSeAEbd(
            **jdata["model"]["descriptor"],
        )
        jdata["model"]["fitting_net"]["ntypes"] = descrpt.get_ntypes()
        jdata["model"]["fitting_net"]["dim_descrpt"] = descrpt.get_dim_out()
        jdata["model"]["fitting_net"]["dim_rot_mat_1"] = descrpt.get_dim_rot_mat_1()
        fitting = EnerFitting(
            **jdata["model"]["fitting_net"],
        )
        # fitting = EnerFitting(jdata['model']['fitting_net'], descrpt)
        model = EnerModel(descrpt, fitting)

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
            suffix="se_a_ebd",
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

        refe = [-4.0809183546731935]
        reff = [
            -0.0009433080242679126,
            0.0024540766182440917,
            7.134654644656728e-05,
            0.0012476799379696184,
            0.002212567421017593,
            -0.0007091290731634111,
            0.009875291762629728,
            -0.007876013249122177,
            -8.78061284553672e-05,
            -0.013879889764531257,
            0.005100427326599536,
            0.00027143866516841334,
            0.003799286895370519,
            -0.007567683893582063,
            0.00024200485149578332,
            -9.906080717069433e-05,
            0.005676625776843024,
            0.00021214513850801415,
        ]
        refv = [
            0.034972620377374586,
            -0.01539771296182217,
            -0.0008639588474713173,
            -0.015397712961822166,
            0.011757303581302702,
            0.0005455501828665077,
            -0.0008639588474713181,
            0.0005455501828665083,
            -6.180425284925767e-07,
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
