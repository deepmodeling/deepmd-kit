# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import unittest

import numpy as np
import paddle
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

import json
from pathlib import (
    Path,
)

from deepmd.pd.model.descriptor import (
    prod_env_mat,
)
from deepmd.pd.utils import (
    dp_random,
    env,
)
from deepmd.pd.utils.dataset import (
    DeepmdDataSetForLoader,
)
from deepmd.pd.utils.env import (
    DEVICE,
    GLOBAL_NP_FLOAT_PRECISION,
    GLOBAL_PD_FLOAT_PRECISION,
)
from deepmd.pd.utils.nlist import (
    extend_input_and_build_neighbor_list,
)
from deepmd.tf.common import (
    expand_sys_str,
)
from deepmd.tf.env import (
    op_module,
)

from ..test_finetune import (
    energy_data_requirement,
)
from .test_embedding_net import (
    get_single_batch,
)

CUR_DIR = os.path.dirname(__file__)


def base_se_a(rcut, rcut_smth, sel, batch, mean, stddev):
    g = tf.Graph()
    with g.as_default():
        coord = tf.placeholder(GLOBAL_NP_FLOAT_PRECISION, [None, None])
        box = tf.placeholder(GLOBAL_NP_FLOAT_PRECISION, [None, None])
        atype = tf.placeholder(tf.int32, [None, None])
        natoms_vec = tf.placeholder(tf.int32, [None])
        default_mesh = tf.placeholder(tf.int32, [None])
        stat_descrpt, descrpt_deriv, rij, nlist = op_module.prod_env_mat_a(
            coord,
            atype,
            natoms_vec,
            box,
            default_mesh,
            tf.constant(mean),
            tf.constant(stddev),
            rcut_a=-1.0,
            rcut_r=rcut,
            rcut_r_smth=rcut_smth,
            sel_a=sel,
            sel_r=[0 for i in sel],
        )

        net_deriv_reshape = tf.ones_like(stat_descrpt)
        force = op_module.prod_force_se_a(
            net_deriv_reshape,
            descrpt_deriv,
            nlist,
            natoms_vec,
            n_a_sel=sum(sel),
            n_r_sel=0,
        )

    with tf.Session(graph=g) as sess:
        y = sess.run(
            [stat_descrpt, force, nlist],
            feed_dict={
                coord: batch["coord"],
                box: batch["box"],
                natoms_vec: batch["natoms"],
                atype: batch["atype"],
                default_mesh: np.array([0, 0, 0, 2, 2, 2]),
            },
        )
    tf.reset_default_graph()
    return y


class TestSeA(unittest.TestCase):
    def setUp(self):
        dp_random.seed(20)
        with open(str(Path(__file__).parent / "water/se_e2_a.json")) as fin:
            content = fin.read()
        config = json.loads(content)
        data_file = [str(Path(__file__).parent / "water/data/data_0")]
        config["training"]["training_data"]["systems"] = data_file
        config["training"]["validation_data"]["systems"] = data_file
        model_config = config["model"]
        self.rcut = model_config["descriptor"]["rcut"]
        self.rcut_smth = model_config["descriptor"]["rcut_smth"]
        self.sel = model_config["descriptor"]["sel"]
        self.bsz = config["training"]["training_data"]["batch_size"]
        self.systems = config["training"]["validation_data"]["systems"]
        if isinstance(self.systems, str):
            self.systems = expand_sys_str(self.systems)
        ds = DeepmdDataSetForLoader(
            self.systems[0],
            model_config["type_map"],
        )
        ds.add_data_requirement(energy_data_requirement)
        self.np_batch, self.pd_batch = get_single_batch(ds)
        self.sec = np.cumsum(self.sel)
        self.ntypes = len(self.sel)
        self.nnei = sum(self.sel)

    def test_consistency(self):
        avg_zero = paddle.zeros(
            [self.ntypes, self.nnei * 4],
            dtype=GLOBAL_PD_FLOAT_PRECISION,
        ).to(device=env.DEVICE)
        std_ones = paddle.ones(
            [self.ntypes, self.nnei * 4],
            dtype=GLOBAL_PD_FLOAT_PRECISION,
        ).to(device=env.DEVICE)
        base_d, base_force, base_nlist = base_se_a(
            rcut=self.rcut,
            rcut_smth=self.rcut_smth,
            sel=self.sel,
            batch=self.np_batch,
            mean=avg_zero.detach().cpu(),
            stddev=std_ones.detach().cpu(),
        )

        pd_coord = self.pd_batch["coord"].to(env.DEVICE)
        atype = self.pd_batch["atype"].to(env.DEVICE)
        pd_coord.stop_gradient = False
        (
            extended_coord,
            extended_atype,
            mapping,
            nlist,
        ) = extend_input_and_build_neighbor_list(
            pd_coord,
            self.pd_batch["atype"].to(env.DEVICE),
            self.rcut,
            self.sel,
            mixed_types=False,
            box=self.pd_batch["box"].to(env.DEVICE),
        )
        my_d, _, _ = prod_env_mat(
            extended_coord,
            nlist,
            atype,
            avg_zero.reshape([-1, self.nnei, 4]).to(DEVICE),
            std_ones.reshape([-1, self.nnei, 4]).to(DEVICE),
            self.rcut,
            self.rcut_smth,
        )
        my_d.sum().backward()
        bsz = pd_coord.shape[0]
        my_force = pd_coord.grad.reshape([bsz, -1, 3]).cpu().detach().numpy()
        base_force = base_force.reshape(bsz, -1, 3)
        base_d = base_d.reshape(bsz, -1, self.nnei, 4)
        my_d = my_d.reshape([bsz, -1, self.nnei, 4]).cpu().detach().numpy()
        base_nlist = base_nlist.reshape(bsz, -1, self.nnei)

        mapping = mapping.cpu()
        my_nlist = nlist.reshape([bsz, -1]).cpu()
        mask = my_nlist == -1
        my_nlist = my_nlist * (~mask).astype(my_nlist.dtype)
        my_nlist = paddle.take_along_axis(mapping, axis=-1, indices=my_nlist)
        my_nlist = my_nlist * (~mask).astype(my_nlist.dtype) - mask.astype(
            my_nlist.dtype
        )
        my_nlist = my_nlist.cpu().reshape([bsz, -1, self.nnei]).numpy()
        self.assertTrue(np.allclose(base_nlist, my_nlist))
        self.assertTrue(np.allclose(np.mean(base_d, axis=2), np.mean(my_d, axis=2)))
        self.assertTrue(np.allclose(np.std(base_d, axis=2), np.std(my_d, axis=2)))
        # descriptors may be different when there are multiple neighbors in the same distance
        self.assertTrue(np.allclose(base_force, -my_force))


if __name__ == "__main__":
    unittest.main()
