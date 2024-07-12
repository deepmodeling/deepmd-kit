# SPDX-License-Identifier: LGPL-3.0-or-later
import logging

import numpy as np

from deepmd.tf.env import (
    tf,
)
from deepmd.tf.utils.graph import (
    get_pattern_nodes_from_graph_def,
    get_tensor_by_name_from_graph,
)

log = logging.getLogger(__name__)


def get_type_embedding(self, graph):
    type_embedding = get_tensor_by_name_from_graph(graph, "t_typeebd")
    type_embedding = type_embedding.astype(self.filter_np_precision)
    return type_embedding


def get_two_side_type_embedding(self, graph):
    type_embedding = get_tensor_by_name_from_graph(graph, "t_typeebd")
    type_embedding = type_embedding.astype(self.filter_np_precision)
    type_embedding_shape = type_embedding.shape

    type_embedding_nei = np.tile(
        np.reshape(type_embedding, [1, type_embedding_shape[0], -1]),
        [type_embedding_shape[0], 1, 1],
    )  # (ntypes) * ntypes * Y
    type_embedding_center = np.tile(
        np.reshape(type_embedding, [type_embedding_shape[0], 1, -1]),
        [1, type_embedding_shape[0], 1],
    )  # ntypes * (ntypes) * Y
    two_side_type_embedding = np.concatenate(
        [type_embedding_nei, type_embedding_center], -1
    )  # ntypes * ntypes * (Y+Y)
    two_side_type_embedding = np.reshape(
        two_side_type_embedding, [-1, two_side_type_embedding.shape[-1]]
    )
    return two_side_type_embedding


def get_extra_side_embedding_net_variable(
    self, graph_def, type_side_suffix, varialbe_name, suffix
):
    ret = {}
    for i in range(1, self.layer_size + 1):
        target = get_pattern_nodes_from_graph_def(
            graph_def,
            f"filter_type_all{suffix}/{varialbe_name}_{i}{type_side_suffix}",
        )
        node = target[f"filter_type_all{suffix}/{varialbe_name}_{i}{type_side_suffix}"]
        ret["layer_" + str(i)] = node
    return ret


def _layer_0(self, x, w, b):
    return self.filter_activation_fn(tf.matmul(x, w) + b)


def _layer_1(self, x, w, b):
    t = tf.concat([x, x], axis=1)
    return t, self.filter_activation_fn(tf.matmul(x, w) + b) + t


def make_data(self, xx):
    with tf.Session() as sess:
        for layer in range(self.layer_size):
            if layer == 0:
                if self.filter_neuron[0] == 1:
                    yy = (
                        _layer_0(
                            self,
                            xx,
                            self.matrix["layer_" + str(layer + 1)],
                            self.bias["layer_" + str(layer + 1)],
                        )
                        + xx
                    )
                elif self.filter_neuron[0] == 2:
                    tt, yy = _layer_1(
                        self,
                        xx,
                        self.matrix["layer_" + str(layer + 1)],
                        self.bias["layer_" + str(layer + 1)],
                    )
                else:
                    yy = _layer_0(
                        self,
                        xx,
                        self.matrix["layer_" + str(layer + 1)],
                        self.bias["layer_" + str(layer + 1)],
                    )
            else:
                tt, zz = _layer_1(
                    self,
                    yy,
                    self.matrix["layer_" + str(layer + 1)],
                    self.bias["layer_" + str(layer + 1)],
                )
                yy = zz
        vv = sess.run(zz)
    return vv
