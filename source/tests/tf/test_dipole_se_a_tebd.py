# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
from packaging.version import parse as parse_version

from deepmd.tf.descriptor import (
    DescrptSeA,
)
from deepmd.tf.env import (
    tf,
)
from deepmd.tf.fit import (
    DipoleFittingSeA,
)
from deepmd.tf.model import (
    DipoleModel,
)
from deepmd.tf.utils.type_embed import (
    TypeEmbedNet,
)

from .common import (
    DataSystem,
    finite_difference,
    gen_data,
    j_loader,
    strerch_box,
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
        gen_data()

    def test_model(self) -> None:
        jfile = "polar_se_a_tebd.json"
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
        jdata["model"]["fitting_net"].pop("type", None)
        jdata["model"]["fitting_net"].pop("fit_diag", None)
        jdata["model"]["fitting_net"]["ntypes"] = descrpt.get_ntypes()
        jdata["model"]["fitting_net"]["dim_descrpt"] = descrpt.get_dim_out()
        jdata["model"]["fitting_net"]["embedding_width"] = descrpt.get_dim_rot_mat_1()
        fitting = DipoleFittingSeA(**jdata["model"]["fitting_net"], uniform_seed=True)
        typeebd_param = jdata["model"]["type_embedding"]
        typeebd = TypeEmbedNet(
            ntypes=descrpt.get_ntypes(),
            neuron=typeebd_param["neuron"],
            resnet_dt=typeebd_param["resnet_dt"],
            seed=typeebd_param["seed"],
            uniform_seed=True,
            use_tebd_bias=True,
        )
        model = DipoleModel(descrpt, fitting, typeebd)

        # model._compute_dstats([test_data['coord']], [test_data['box']], [test_data['type']], [test_data['natoms_vec']], [test_data['default_mesh']])
        input_data = {
            "coord": [test_data["coord"]],
            "box": [test_data["box"]],
            "type": [test_data["type"]],
            "natoms_vec": [test_data["natoms_vec"]],
            "default_mesh": [test_data["default_mesh"]],
            "fparam": [test_data["fparam"]],
        }
        model._compute_input_stat(input_data)

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
        inputs_dict = {}

        model_pred = model.build(
            t_coord,
            t_type,
            t_natoms,
            t_box,
            t_mesh,
            inputs_dict,
            suffix="dipole_se_a_tebd",
            reuse=False,
        )
        dipole = model_pred["dipole"]
        gdipole = model_pred["global_dipole"]
        force = model_pred["force"]
        virial = model_pred["virial"]
        atom_virial = model_pred["atom_virial"]

        feed_dict_test = {
            t_prop_c: test_data["prop_c"],
            t_coord: np.reshape(test_data["coord"][:numb_test, :], [-1]),
            t_box: test_data["box"][:numb_test, :],
            t_type: np.reshape(test_data["type"][:numb_test, :], [-1]),
            t_natoms: test_data["natoms_vec"],
            t_mesh: test_data["default_mesh"],
            is_training: False,
        }

        sess = self.cached_session().__enter__()
        sess.run(tf.global_variables_initializer())
        [p, gp] = sess.run([dipole, gdipole], feed_dict=feed_dict_test)

        p = p.reshape([-1])
        refp = [
            15.759189570481473,
            9.56848733368029,
            0.3494387894045414,
            1.3280752117673629,
            10.285935424492124,
            -0.5847081785394377,
        ]

        places = 10
        np.testing.assert_almost_equal(p, refp, places)

        gp = gp.reshape([-1])
        refgp = np.array(refp).reshape(-1, 3).sum(0)

        places = 9
        np.testing.assert_almost_equal(gp, refgp, places)

        # make sure only one frame is used
        feed_dict_single = {
            t_prop_c: test_data["prop_c"],
            t_coord: np.reshape(test_data["coord"][:1, :], [-1]),
            t_box: test_data["box"][:1, :],
            t_type: np.reshape(test_data["type"][:1, :], [-1]),
            t_natoms: test_data["natoms_vec"],
            t_mesh: test_data["default_mesh"],
            is_training: False,
        }

        [pf, pv, pav] = sess.run(
            [force, virial, atom_virial], feed_dict=feed_dict_single
        )
        pf, pv = pf.reshape(-1), pv.reshape(-1)
        spv = pav.reshape(1, 3, -1, 9).sum(2).reshape(-1)

        base_dict = feed_dict_single.copy()
        coord0 = base_dict.pop(t_coord)
        box0 = base_dict.pop(t_box)

        fdf = -finite_difference(
            lambda coord: sess.run(
                gdipole, feed_dict={**base_dict, t_coord: coord, t_box: box0}
            ).reshape(-1),
            test_data["coord"][:numb_test, :].reshape([-1]),
        ).reshape(-1)
        fdv = -(
            finite_difference(
                lambda box: sess.run(
                    gdipole,
                    feed_dict={
                        **base_dict,
                        t_coord: strerch_box(coord0, box0, box),
                        t_box: box,
                    },
                ).reshape(-1),
                test_data["box"][:numb_test, :],
            )
            .reshape([-1, 3, 3])
            .transpose(0, 2, 1)
            @ box0.reshape(3, 3)
        ).reshape(-1)

        delta = 1e-5
        np.testing.assert_allclose(pf, fdf, delta)
        np.testing.assert_allclose(pv, fdv, delta)
        # make sure atomic virial sum to virial
        places = 10
        np.testing.assert_almost_equal(pv, spv, places)
