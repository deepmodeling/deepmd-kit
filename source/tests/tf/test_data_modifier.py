# SPDX-License-Identifier: LGPL-3.0-or-later
import os

import numpy as np

from deepmd.tf.env import (
    GLOBAL_NP_FLOAT_PRECISION,
    tf,
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

from .common import (
    Data,
    j_loader,
    tests_path,
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
INPUT = os.path.join(modifier_datapath, "dipole.json")


class TestDataModifier(tf.test.TestCase):
    def setUp(self) -> None:
        # with tf.variable_scope('load', reuse = False) :
        tf.reset_default_graph()
        self._setUp()

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def _setUp(self) -> None:
        run_opt = RunOptions(
            restart=None, init_model=None, log_path=None, log_level=30, mpi_log="master"
        )
        jdata = j_loader(INPUT)
        # init model
        model = DPTrainer(jdata, run_opt=run_opt)
        rcut = model.model.get_rcut()

        # init data system
        systems = jdata["training"]["systems"]
        # systems[0] = tests_path / systems[0]
        systems = [tests_path / ii for ii in systems]
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
            output_graph = str(
                tests_path / os.path.join(modifier_datapath, "dipole.pb")
            )
            with tf.gfile.GFile(output_graph, "wb") as f:
                f.write(output_graph_def.SerializeToString())

    def test_fv(self) -> None:
        # with tf.variable_scope('load', reuse = False) :
        self._test_fv()

    def _test_fv(self) -> None:
        dcm = DipoleChargeModifier(
            model_name=str(tests_path / os.path.join(modifier_datapath, "dipole.pb")),
            model_charge_map=[-8],
            sys_charge_map=[6, 1],
            ewald_h=1,
            ewald_beta=0.25,
        )
        data = Data()
        coord, box, atype = data.get_data()
        atype = atype[0]
        ve, vf, vv = dcm.eval(coord, box, atype)

        hh = global_default_fv_hh
        hh = 1e-4
        places = global_default_places
        places = 1
        nframes = coord.shape[0]
        ndof = coord.shape[1]
        natoms = ndof // 3
        vf = np.reshape(vf, [nframes, -1])
        for ii in range(ndof):
            coordp = np.copy(coord)
            coordm = np.copy(coord)
            coordp[:, ii] += hh
            coordm[:, ii] -= hh
            ep, _, __ = dcm.eval(coordp, box, atype, eval_fv=False)
            em, _, __ = dcm.eval(coordm, box, atype, eval_fv=False)
            num_f = -(ep - em) / (2.0 * hh)
            np.testing.assert_almost_equal(
                vf[:, ii].ravel(),
                num_f.ravel(),
                places,
                err_msg=f"dof {ii} does not match",
            )

        box3 = np.reshape(box, [nframes, 3, 3])
        rbox3 = np.linalg.inv(box3)
        coord3 = np.reshape(coord, [nframes, natoms, 3])
        rcoord3 = np.matmul(coord3, rbox3)
        num_deriv = np.zeros([nframes, 3, 3])
        for ii in range(3):
            for jj in range(3):
                box3p = np.copy(box3)
                box3m = np.copy(box3)
                box3p[:, ii, jj] = box3[:, ii, jj] + hh
                box3m[:, ii, jj] = box3[:, ii, jj] - hh
                boxp = np.reshape(box3p, [-1, 9])
                boxm = np.reshape(box3m, [-1, 9])
                coord3p = np.matmul(rcoord3, box3p)
                coord3m = np.matmul(rcoord3, box3m)
                coordp = np.reshape(coord3p, [nframes, -1])
                coordm = np.reshape(coord3m, [nframes, -1])
                ep, _, __ = dcm.eval(coordp, boxp, atype, eval_fv=False)
                em, _, __ = dcm.eval(coordm, boxm, atype, eval_fv=False)
                num_deriv[:, ii, jj] = -(ep - em) / (2.0 * hh)
        # box3t = np.transpose(box3, [0,2,1])
        # t_esti = np.matmul(num_deriv, box3t)
        num_deriv = np.transpose(num_deriv, [0, 2, 1])
        t_esti = np.matmul(num_deriv, box3)

        # print(t_esti, '\n', vv.reshape([-1, 3, 3]))
        np.testing.assert_almost_equal(
            t_esti.ravel(), vv.ravel(), places, err_msg="virial component failed"
        )
