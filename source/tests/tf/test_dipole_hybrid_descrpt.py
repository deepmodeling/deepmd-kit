# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np

from deepmd.tf.descriptor.hybrid import (
    DescrptHybrid,
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
        jfile = "polar_se_a.json"
        jdata = j_loader(jfile)

        systems = jdata["systems"]
        set_pfx = "set"
        batch_size = 1
        test_size = 1
        rcut = jdata["model"]["descriptor"]["rcut"]

        data = DataSystem(systems, set_pfx, batch_size, test_size, rcut, run_opt=None)

        test_data = data.get_test()
        numb_test = 1

        descrpt = DescrptHybrid(
            list=[
                {
                    "type": "se_e2_a",
                    "sel": [20, 20],
                    "rcut_smth": 1.8,
                    "rcut": 6.0,
                    "neuron": [2, 4, 8],
                    "resnet_dt": False,
                    "axis_neuron": 8,
                    "precision": "float64",
                    "type_one_side": True,
                    "seed": 1,
                },
                {
                    "type": "se_e2_a",
                    "sel": [20, 20],
                    "rcut_smth": 1.8,
                    "rcut": 6.0,
                    "neuron": [2, 4, 8],
                    "resnet_dt": False,
                    "axis_neuron": 8,
                    "precision": "float64",
                    "type_one_side": True,
                    "seed": 1,
                },
                {
                    "type": "se_e3",
                    "sel": [5, 5],
                    "rcut_smth": 1.8,
                    "rcut": 2.0,
                    "neuron": [2],
                    "resnet_dt": False,
                    "precision": "float64",
                    "seed": 1,
                },
            ]
        )
        jdata["model"]["fitting_net"].pop("type", None)
        jdata["model"]["fitting_net"].pop("fit_diag", None)
        jdata["model"]["fitting_net"]["ntypes"] = descrpt.get_ntypes()
        jdata["model"]["fitting_net"]["dim_descrpt"] = descrpt.get_dim_out()
        jdata["model"]["fitting_net"]["embedding_width"] = descrpt.get_dim_rot_mat_1()
        fitting = DipoleFittingSeA(**jdata["model"]["fitting_net"], uniform_seed=True)
        model = DipoleModel(descrpt, fitting)

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
            suffix="dipole_hybrid",
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
        [p, gp, f, v, av] = sess.run(
            [dipole, gdipole, force, virial, atom_virial], feed_dict=feed_dict_test
        )
