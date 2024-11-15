# SPDX-License-Identifier: LGPL-3.0-or-later
import re
import unittest

import numpy as np
import tensorflow.compat.v1 as tf
import torch

tf.disable_eager_execution()

from deepmd.pt.model.task import (
    EnergyFittingNet,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)
from deepmd.tf.fit.ener import (
    EnerFitting,
)

from ...seed import (
    GLOBAL_SEED,
)


class FakeDescriptor:
    def __init__(self, ntypes, embedding_width) -> None:
        self._ntypes = ntypes
        self._dim_out = embedding_width

    def get_ntypes(self):
        return self._ntypes

    def get_dim_out(self):
        return self._dim_out


def gen_key(type_id, layer_id, w_or_b):
    return (type_id, layer_id, w_or_b)


def base_fitting_net(dp_fn, embedding, natoms, atype):
    g = tf.Graph()
    with g.as_default():
        t_embedding = tf.placeholder(GLOBAL_NP_FLOAT_PRECISION, [None, None])
        t_natoms = tf.placeholder(tf.int32, [None])
        t_atype = tf.placeholder(tf.int32, [None, None])
        t_energy = dp_fn.build(t_embedding, t_natoms, {"atype": t_atype})
        init_op = tf.global_variables_initializer()
        t_vars = {}
        for var in tf.global_variables():
            key = None
            matched = re.match(r"layer_(\d)_type_(\d)/([a-z]+)", var.name)
            if matched:
                key = gen_key(
                    type_id=matched.group(2),
                    layer_id=matched.group(1),
                    w_or_b=matched.group(3),
                )
            else:
                matched = re.match(r"final_layer_type_(\d)/([a-z]+)", var.name)
                if matched:
                    key = gen_key(
                        type_id=matched.group(1), layer_id=-1, w_or_b=matched.group(2)
                    )
            if key is not None:
                t_vars[key] = var

    with tf.Session(graph=g) as sess:
        sess.run(init_op)
        energy, values = sess.run(
            [t_energy, t_vars],
            feed_dict={
                t_embedding: embedding,
                t_natoms: natoms,
                t_atype: atype,
            },
        )
    tf.reset_default_graph()
    return energy, values


class TestFittingNet(unittest.TestCase):
    def setUp(self) -> None:
        nloc = 7
        self.embedding_width = 30
        self.natoms = np.array([nloc, nloc, 2, 5], dtype=np.int32)
        rng = np.random.default_rng(GLOBAL_SEED)
        self.embedding = rng.uniform(size=[4, nloc * self.embedding_width])
        self.ntypes = self.natoms.size - 2
        self.n_neuron = [32, 32, 32]
        self.atype = np.zeros([4, nloc], dtype=np.int32)
        cnt = 0
        for i in range(self.ntypes):
            self.atype[:, cnt : cnt + self.natoms[i + 2]] = i
            cnt += self.natoms[i + 2]

        fake_d = FakeDescriptor(2, 30)
        self.dp_fn = EnerFitting(
            fake_d.get_ntypes(), fake_d.get_dim_out(), self.n_neuron
        )
        self.dp_fn.bias_atom_e = rng.uniform(size=[self.ntypes])

    def test_consistency(self) -> None:
        dp_energy, values = base_fitting_net(
            self.dp_fn, self.embedding, self.natoms, self.atype
        )
        my_fn = EnergyFittingNet(
            self.ntypes,
            self.embedding_width,
            neuron=self.n_neuron,
            bias_atom_e=self.dp_fn.bias_atom_e,
            mixed_types=False,
        ).to(env.DEVICE)
        for name, param in my_fn.named_parameters():
            matched = re.match(
                r"filter_layers\.networks\.(\d).layers\.(\d)\.([a-z]+)", name
            )
            key = None
            if matched:
                if int(matched.group(2)) == len(self.n_neuron):
                    layer_id = -1
                else:
                    layer_id = matched.group(2)
                key = gen_key(
                    type_id=matched.group(1),
                    layer_id=layer_id,
                    w_or_b=matched.group(3),
                )
            assert key is not None
            var = values[key]
            with torch.no_grad():
                # Keep parameter value consistency between 2 implementations
                param.data.copy_(torch.from_numpy(var))
        embedding = torch.from_numpy(self.embedding)
        embedding = embedding.view(4, -1, self.embedding_width)
        atype = torch.from_numpy(self.atype)
        ret = my_fn(embedding.to(env.DEVICE), atype.to(env.DEVICE))
        my_energy = ret["energy"]
        my_energy = my_energy.detach().cpu()
        np.testing.assert_allclose(dp_energy, my_energy.numpy().reshape([-1]))


if __name__ == "__main__":
    unittest.main()
