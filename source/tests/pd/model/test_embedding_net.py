# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import os
import re
import unittest

import numpy as np
import paddle
import tensorflow.compat.v1 as tf

from deepmd.pd.utils import (
    env,
)

tf.disable_eager_execution()

from pathlib import (
    Path,
)

from deepmd.pd.model.descriptor import (
    DescrptSeA,
)
from deepmd.pd.utils import (
    dp_random,
)
from deepmd.pd.utils.dataset import (
    DeepmdDataSetForLoader,
)
from deepmd.pd.utils.env import (
    DEVICE,
    GLOBAL_NP_FLOAT_PRECISION,
)
from deepmd.pd.utils.nlist import (
    extend_input_and_build_neighbor_list,
)
from deepmd.tf.common import (
    expand_sys_str,
)
from deepmd.tf.descriptor import DescrptSeA as DescrptSeA_tf

from ..test_finetune import (
    energy_data_requirement,
)

CUR_DIR = os.path.dirname(__file__)


def gen_key(worb, depth, elemid):
    return (worb, depth, elemid)


def get_single_batch(dataset, index=None):
    if index is None:
        index = dp_random.choice(np.arange(len(dataset)))
    np_batch = dataset[index]
    pd_batch = {}

    for key in [
        "coord",
        "box",
        "force",
        "force_mag",
        "energy",
        "virial",
        "atype",
        "natoms",
    ]:
        if key in np_batch.keys():
            np_batch[key] = np.expand_dims(np_batch[key], axis=0)
            pd_batch[key] = paddle.to_tensor(np_batch[key]).to(device=env.DEVICE)
            if key in ["coord", "force", "force_mag"]:
                np_batch[key] = np_batch[key].reshape(1, -1)
    np_batch["natoms"] = np_batch["natoms"][0]
    return np_batch, pd_batch


def base_se_a(descriptor, coord, atype, natoms, box):
    g = tf.Graph()
    with g.as_default():
        name_pfx = "d_sea_"
        t_coord = tf.placeholder(
            GLOBAL_NP_FLOAT_PRECISION, [None, None], name=name_pfx + "t_coord"
        )
        t_atype = tf.placeholder(tf.int32, [None, None], name=name_pfx + "t_type")
        t_natoms = tf.placeholder(
            tf.int32, [descriptor.ntypes + 2], name=name_pfx + "t_natoms"
        )
        t_box = tf.placeholder(
            GLOBAL_NP_FLOAT_PRECISION, [None, None], name=name_pfx + "t_box"
        )
        t_default_mesh = tf.placeholder(tf.int32, [None], name=name_pfx + "t_mesh")
        t_embedding = descriptor.build(
            t_coord, t_atype, t_natoms, t_box, t_default_mesh, input_dict={}
        )
        fake_energy = tf.reduce_sum(t_embedding)
        t_force = descriptor.prod_force_virial(fake_energy, t_natoms)[0]
        t_vars = {}
        for var in tf.global_variables():
            ms = re.findall(r"([a-z]+)_(\d)_(\d)", var.name)
            if len(ms) == 1:
                m = ms[0]
                key = gen_key(worb=m[0], depth=int(m[1]), elemid=int(m[2]))
                t_vars[key] = var
        init_op = tf.global_variables_initializer()

    with tf.Session(graph=g) as sess:
        sess.run(init_op)
        embedding, force, values = sess.run(
            [t_embedding, t_force, t_vars],
            feed_dict={
                t_coord: coord,
                t_atype: atype,
                t_natoms: natoms,
                t_box: box,
                t_default_mesh: np.array([0, 0, 0, 2, 2, 2]),
            },
        )
    tf.reset_default_graph()
    return embedding, force, values


class TestSeA(unittest.TestCase):
    def setUp(self):
        dp_random.seed(0)
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
        self.filter_neuron = model_config["descriptor"]["neuron"]
        self.axis_neuron = model_config["descriptor"]["axis_neuron"]
        self.np_batch, self.paddle_batch = get_single_batch(ds)

    def test_consistency(self):
        dp_d = DescrptSeA_tf(
            rcut=self.rcut,
            rcut_smth=self.rcut_smth,
            sel=self.sel,
            neuron=self.filter_neuron,
            axis_neuron=self.axis_neuron,
            seed=1,
        )
        dp_embedding, dp_force, dp_vars = base_se_a(
            descriptor=dp_d,
            coord=self.np_batch["coord"],
            atype=self.np_batch["atype"],
            natoms=self.np_batch["natoms"],
            box=self.np_batch["box"],
        )

        # Reproduced
        descriptor = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
            neuron=self.filter_neuron,
            axis_neuron=self.axis_neuron,
        ).to(DEVICE)
        for name, param in descriptor.named_parameters():
            ms = re.findall(r"(\d)\.layers\.(\d)\.([a-z]+)", name)
            if len(ms) == 1:
                m = ms[0]
                key = gen_key(worb=m[2], depth=int(m[1]) + 1, elemid=int(m[0]))
                var = dp_vars[key]
                with paddle.no_grad():
                    # Keep parameter value consistency between 2 implentations
                    paddle.assign(var, param)

        pd_coord = self.paddle_batch["coord"].to(env.DEVICE)
        pd_coord.stop_gradient = False

        (
            extended_coord,
            extended_atype,
            mapping,
            nlist,
        ) = extend_input_and_build_neighbor_list(
            pd_coord,
            self.paddle_batch["atype"].to(env.DEVICE),
            self.rcut,
            self.sel,
            mixed_types=False,
            box=self.paddle_batch["box"].to(env.DEVICE),
        )
        descriptor_out, _, _, _, _ = descriptor(
            extended_coord,
            extended_atype,
            nlist,
        )
        my_embedding = descriptor_out.cpu().detach().numpy()
        fake_energy = paddle.sum(descriptor_out)
        fake_energy.backward()
        my_force = -pd_coord.grad.cpu().numpy()

        # Check
        np.testing.assert_allclose(dp_embedding, my_embedding)
        dp_force = dp_force.reshape(*my_force.shape)
        np.testing.assert_allclose(dp_force, my_force)


if __name__ == "__main__":
    unittest.main()
