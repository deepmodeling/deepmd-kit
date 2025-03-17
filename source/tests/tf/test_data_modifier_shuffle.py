# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import shutil

import numpy as np

from deepmd.tf.env import (
    GLOBAL_NP_FLOAT_PRECISION,
    tf,
)
from deepmd.tf.infer.deep_dipole import (
    DeepDipole,
)
from deepmd.tf.modifier import (
    DipoleChargeModifier,
)
from deepmd.tf.train.run_options import (
    RunOptions,
)
from deepmd.tf.train.trainer import (
    DPTrainer,
)
from deepmd.tf.utils.data_system import (
    DeepmdDataSystem,
)

from ..seed import (
    GLOBAL_SEED,
)

if GLOBAL_NP_FLOAT_PRECISION == np.float32:
    global_default_fv_hh = 1e-2
    global_default_dw_hh = 1e-2
    global_default_places = 3
else:
    global_default_fv_hh = 1e-6
    global_default_dw_hh = 1e-4
    global_default_places = 5

modifier_datapath = "data_modifier"


class TestDataModifier(tf.test.TestCase):
    def setUp(self) -> None:
        # with tf.variable_scope('load', reuse = False) :
        tf.reset_default_graph()
        self._setUp()

    def tearDown(self) -> None:
        tf.reset_default_graph()
        if os.path.isdir(os.path.join(modifier_datapath, "sys_test_0")):
            shutil.rmtree(os.path.join(modifier_datapath, "sys_test_0"))
        if os.path.isfile(os.path.join(modifier_datapath, "dipole.pb")):
            os.remove(os.path.join(modifier_datapath, "dipole.pb"))

    def _setUp(self) -> None:
        run_opt = RunOptions(
            restart=None, init_model=None, log_path=None, log_level=30, mpi_log="master"
        )
        jdata = self._setUp_jdata()
        self._setUp_data()

        # init model
        model = DPTrainer(jdata, run_opt=run_opt)
        rcut = model.model.get_rcut()

        # init data system
        systems = jdata["training"]["systems"]
        set_pfx = "set"
        batch_size = jdata["training"]["batch_size"]
        test_size = jdata["training"]["numb_test"]
        data = DeepmdDataSystem(
            systems, batch_size, test_size, rcut, set_prefix=set_pfx
        )
        data.add_data_requirements(model.data_requirements)

        # clear the default graph
        tf.reset_default_graph()

        # build the model with stats from the first system
        model.build(data)

        # freeze the graph
        with self.cached_session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            graph = tf.get_default_graph()
            input_graph_def = graph.as_graph_def()
            nodes = "o_dipole,o_rmat,o_rmat_deriv,o_nlist,o_rij,descrpt_attr/rcut,descrpt_attr/ntypes,descrpt_attr/sel,descrpt_attr/ndescrpt,model_attr/tmap,model_attr/sel_type,model_attr/model_type,model_attr/output_dim,model_attr/model_version"
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, input_graph_def, nodes.split(",")
            )
            output_graph = os.path.join(modifier_datapath, "dipole.pb")
            with tf.gfile.GFile(output_graph, "wb") as f:
                f.write(output_graph_def.SerializeToString())

    def _setUp_data(self) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        jdata = self._setUp_jdata()
        # sys0
        self.atom_types0 = np.array([0, 3, 2, 1, 3, 4, 1, 4], dtype=int)
        self.natoms = len(self.atom_types0)
        self.nframes = 1
        scale = 10.0
        self.sel_type = jdata["model"]["fitting_net"]["sel_type"]
        self.nsel = 0
        for ii in self.sel_type:
            self.nsel += np.sum(self.atom_types0 == ii)
        self.coords0 = rng.random([self.nframes, self.natoms * 3]) * scale
        self.dipoles0 = rng.random([self.nframes, self.nsel * 3])
        self.box0 = np.reshape(np.eye(3) * scale, [-1, 9])
        self.box0 = np.tile(self.box0, [self.nframes, 1])
        self._write_sys_data(
            os.path.join(modifier_datapath, "sys_test_0"),
            self.atom_types0,
            self.coords0,
            self.dipoles0,
            self.box0,
        )
        # sys1
        self.idx_map = np.array([6, 7, 1, 0, 5, 2, 4, 3], dtype=int)
        self.sel_idx_map = np.array([3, 0, 2, 1], dtype=int)
        self.atom_types1 = self.atom_types0[self.idx_map]
        self.coords1 = np.reshape(self.coords0, [self.nframes, -1, 3])
        self.coords1 = self.coords1[:, self.idx_map, :]
        self.coords1 = np.reshape(self.coords1, [self.nframes, self.natoms * 3])
        self.dipoles1 = self.dipoles0[:, self.sel_idx_map]
        self.box1 = self.box0
        self.sel_mask0 = np.isin(self.atom_types0, self.sel_type)
        self.sel_mask1 = np.isin(self.atom_types1, self.sel_type)

    def _write_sys_data(self, dirname, atom_types, coords, dipoles, box) -> None:
        os.makedirs(dirname, exist_ok=True)
        os.makedirs(dirname + "/set.0", exist_ok=True)
        np.savetxt(os.path.join(dirname, "type.raw"), atom_types, fmt="%d")
        np.save(os.path.join(dirname, "set.0", "coord.npy"), coords)
        np.save(os.path.join(dirname, "set.0", "atomic_dipole.npy"), dipoles)
        np.save(os.path.join(dirname, "set.0", "box.npy"), box)

    def _setUp_jdata(self):
        aa = {"a": [1, 2, 3]}
        jdata = {
            "model": {
                "type_map": ["A", "B", "C", "D", "E"],
                "descriptor": {
                    "type": "se_e2_a",
                    "sel": [50, 50, 50, 50, 50],
                    "rcut_smth": 3.80,
                    "rcut": 4.00,
                    "neuron": [2, 4],
                    "resnet_dt": False,
                    "axis_neuron": 4,
                    "seed": 1,
                },
                "fitting_net": {
                    "type": "dipole",
                    "sel_type": [1, 3],
                    "neuron": [10],
                    "resnet_dt": True,
                    "seed": 1,
                },
            },
            "loss": {
                "type": "tensor",
                "pref": 1.0,
                "pref_atomic": 1.0,
                "_comment": " that's all",
            },
            "learning_rate": {
                "type": "exp",
                "start_lr": 0.01,
                "stop_lr": 1e-8,
                "decay_steps": 5000,
                "decay_rate": 0.95,
            },
            "training": {
                "systems": ["data_modifier/sys_test_0"],
                "stop_batch": 1000000,
                "batch_size": 1,
                "numb_test": 2,
            },
        }
        return jdata

    def test_z_dipole(self) -> None:
        dd = DeepDipole(os.path.join(modifier_datapath, "dipole.pb"))

        dv0 = dd.eval(self.coords0, self.box0, self.atom_types0)[:, self.sel_mask0]
        dv1 = dd.eval(self.coords1, self.box1, self.atom_types1)[:, self.sel_mask1]

        dv01 = dv0.reshape([self.nframes, -1, 3])
        dv01 = dv01[:, self.sel_idx_map, :]
        dv01 = dv01.reshape([self.nframes, -1])
        dv1 = dv1.reshape([self.nframes, -1])

        np.testing.assert_almost_equal(dv01, dv1, err_msg="dipole dose not match")

    def test_modify(self) -> None:
        dcm = DipoleChargeModifier(
            model_name=os.path.join(modifier_datapath, "dipole.pb"),
            model_charge_map=[-1, -3],
            sys_charge_map=[1, 1, 1, 1, 1],
            ewald_h=1,
            ewald_beta=0.25,
        )
        ve0, vf0, vv0 = dcm.eval(self.coords0, self.box0, self.atom_types0)
        ve1, vf1, vv1 = dcm.eval(self.coords1, self.box1, self.atom_types1)
        vf01 = vf0[:, self.idx_map, :]

        np.testing.assert_almost_equal(ve0, ve1, err_msg="energy should match")
        np.testing.assert_almost_equal(vv0, vv1, err_msg="virial should match")
        np.testing.assert_almost_equal(vf01, vf1, err_msg="force dose not match")
