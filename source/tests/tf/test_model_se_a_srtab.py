# SPDX-License-Identifier: LGPL-3.0-or-later
import os

import numpy as np

from deepmd.tf.descriptor import (
    DescrptSeA,
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


def _make_tab(ntype) -> None:
    xx = np.arange(0, 9, 0.001)
    yy = 1000 / (xx + 0.5) ** 6
    prt = xx
    ninter = ntype * (ntype + 1) // 2
    for ii in range(ninter):
        prt = np.append(prt, yy)
    prt = np.reshape(prt, [ninter + 1, -1])
    np.savetxt("tab.xvg", prt.T)


class TestModel(tf.test.TestCase):
    def setUp(self) -> None:
        gen_data()
        _make_tab(2)

    def tearDown(self) -> None:
        os.remove("tab.xvg")

    def test_model(self) -> None:
        jfile = "water_se_a_srtab.json"
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
        descrpt = DescrptSeA(**jdata["model"]["descriptor"], uniform_seed=True)
        jdata["model"]["fitting_net"]["ntypes"] = descrpt.get_ntypes()
        jdata["model"]["fitting_net"]["dim_descrpt"] = descrpt.get_dim_out()
        jdata["model"]["fitting_net"]["dim_rot_mat_1"] = descrpt.get_dim_rot_mat_1()
        fitting = EnerFitting(**jdata["model"]["fitting_net"], uniform_seed=True)
        # descrpt = DescrptSeA(jdata['model']['descriptor'])
        # fitting = EnerFitting(jdata['model']['fitting_net'], descrpt)
        model = EnerModel(
            descrpt,
            fitting,
            None,
            jdata["model"].get("type_map"),
            jdata["model"].get("data_stat_nbatch"),
            jdata["model"].get("data_stat_protect"),
            jdata["model"].get("use_srtab"),
            jdata["model"].get("smin_alpha"),
            jdata["model"].get("sw_rmin"),
            jdata["model"].get("sw_rmax"),
        )

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
            suffix="se_a_srtab",
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

        refe = [1.141610882066236599e02]
        reff = [
            -1.493121233165248043e02,
            -1.831419491743885715e02,
            -8.439542992300344437e00,
            -1.811987095947552859e02,
            -1.476380826187439084e02,
            1.264271856742560018e01,
            1.544377958934875323e02,
            -7.816520233903435866e00,
            1.287925245463442225e00,
            -4.000393268449002449e00,
            1.910748885843098890e02,
            7.134789955349889468e00,
            1.826908441979261113e02,
            3.677156386479059513e00,
            -1.122312112141401741e01,
            -2.617413911684622008e00,
            1.438445070562470391e02,
            -1.402769654524568033e00,
        ]
        refv = [
            3.585047655925112622e02,
            -7.569252978336677984e00,
            -1.068382043878426124e01,
            -7.569252978336677096e00,
            3.618439481685132932e02,
            5.448668500896081568e00,
            -1.068382043878426302e01,
            5.448668500896082456e00,
            1.050393462151727686e00,
        ]
        refe = np.reshape(refe, [-1])
        reff = np.reshape(reff, [-1])
        refv = np.reshape(refv, [-1])

        places = 10
        np.testing.assert_almost_equal(e, refe, places)
        np.testing.assert_almost_equal(f, reff, places)
        np.testing.assert_almost_equal(v, refv, places)
