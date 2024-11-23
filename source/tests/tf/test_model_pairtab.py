# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
import scipy.spatial.distance

from deepmd.tf.env import (
    tf,
)
from deepmd.tf.model.model import (
    Model,
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
        jfile = "water.json"
        jdata = j_loader(jfile)
        systems = jdata["systems"]
        set_pfx = "set"
        batch_size = 1
        test_size = 1

        tab_filename = "test_pairtab_tab.txt"
        jdata["model"] = {
            "type": "pairtab",
            "tab_file": tab_filename,
            "rcut": 6,
            "sel": [6],
        }
        rcut = jdata["model"]["rcut"]

        def pair_pot(r: float):
            # LJ, as example
            return 4 * (1 / r**12 - 1 / r**6)

        dx = 1e-4
        d = np.arange(dx, rcut + dx, dx)
        tab = np.array(
            [
                d,
                pair_pot(d),
                pair_pot(d),
                pair_pot(d),
            ]
        ).T
        np.savetxt(tab_filename, tab)

        data = DataSystem(systems, set_pfx, batch_size, test_size, rcut, run_opt=None)

        test_data = data.get_test()
        numb_test = 1

        model = Model(
            **jdata["model"],
        )

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
            suffix="test_pairtab",
            reuse=False,
        )
        energy = model_pred["energy"]
        force = model_pred["force"]
        virial = model_pred["virial"]

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
            t_mesh: [],  # nopbc
            is_training: False,
        }

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            [e, _, _] = sess.run([energy, force, virial], feed_dict=feed_dict_test)

        e = e.reshape([-1])

        coord = test_data["coord"][0, :].reshape(-1, 3)
        distance = scipy.spatial.distance.cdist(coord, coord).ravel()
        refe = [np.sum(pair_pot(distance[np.nonzero(distance)])) / 2]

        refe = np.reshape(refe, [-1])

        places = 10
        np.testing.assert_almost_equal(e, refe, places)
