# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from functools import (
    cached_property,
)
from typing import (
    Callable,
)

import numpy as np
from scipy.special import (
    comb,
)

import deepmd
from deepmd.tf.common import (
    ACTIVATION_FN_DICT,
)
from deepmd.tf.descriptor import (
    Descriptor,
)
from deepmd.tf.env import (
    op_module,
    tf,
)
from deepmd.tf.utils.graph import (
    get_embedding_net_nodes_from_graph_def,
    get_tensor_by_name_from_graph,
)
from deepmd.utils.tabulate import (
    BaseTabulate,
)

log = logging.getLogger(__name__)


class DPTabulate(BaseTabulate):
    r"""Class for tabulation.

    Compress a model, which including tabulating the embedding-net.
    The table is composed of fifth-order polynomial coefficients and is assembled from two sub-tables. The first table takes the stride(parameter) as it's uniform stride, while the second table takes 10 * stride as it's uniform stride
    The range of the first table is automatically detected by deepmd-kit, while the second table ranges from the first table's upper boundary(upper) to the extrapolate(parameter) * upper.

    Parameters
    ----------
    descrpt
            Descriptor of the original model
    neuron
            Number of neurons in each hidden layers of the embedding net :math:`\\mathcal{N}`
    graph : tf.Graph
            The graph of the original model
    graph_def : tf.GraphDef
            The graph_def of the original model
    type_one_side
            Try to build N_types tables. Otherwise, building N_types^2 tables
    exclude_types : list[list[int]]
            The excluded pairs of types which have no interaction with each other.
            For example, `[[0, 1]]` means no interaction between type 0 and type 1.
    activation_function
            The activation function in the embedding net. Supported options are {"tanh","gelu"} in common.ACTIVATION_FN_DICT.
    suffix : str, optional
            The suffix of the scope
    """

    def __init__(
        self,
        descrpt: Descriptor,
        neuron: list[int],
        graph: tf.Graph,
        graph_def: tf.GraphDef,
        type_one_side: bool = False,
        exclude_types: list[list[int]] = [],
        activation_fn: Callable[[tf.Tensor], tf.Tensor] = tf.nn.tanh,
        suffix: str = "",
    ) -> None:
        super().__init__(
            descrpt,
            neuron,
            type_one_side,
            exclude_types,
            False,
        )

        self.descrpt_type = self._get_descrpt_type()
        """Constructor."""
        self.graph = graph
        self.graph_def = graph_def
        self.suffix = suffix

        # functype
        if activation_fn == ACTIVATION_FN_DICT["tanh"]:
            self.functype = 1
        elif activation_fn in (
            ACTIVATION_FN_DICT["gelu"],
            ACTIVATION_FN_DICT["gelu_tf"],
        ):
            self.functype = 2
        elif activation_fn == ACTIVATION_FN_DICT["relu"]:
            self.functype = 3
        elif activation_fn == ACTIVATION_FN_DICT["relu6"]:
            self.functype = 4
        elif activation_fn == ACTIVATION_FN_DICT["softplus"]:
            self.functype = 5
        elif activation_fn == ACTIVATION_FN_DICT["sigmoid"]:
            self.functype = 6
        else:
            raise RuntimeError("Unknown activation function type!")
        self.activation_fn = activation_fn

        # self.sess = tf.Session(graph = self.graph)

        self.sub_graph, self.sub_graph_def = self._load_sub_graph()
        self.sub_sess = tf.Session(graph=self.sub_graph)

        if isinstance(self.descrpt, deepmd.tf.descriptor.DescrptSeR):
            self.sel_a = self.descrpt.sel_r
            self.rcut = self.descrpt.rcut
            self.rcut_smth = self.descrpt.rcut_smth
        elif isinstance(self.descrpt, deepmd.tf.descriptor.DescrptSeA):
            self.sel_a = self.descrpt.sel_a
            self.rcut = self.descrpt.rcut_r
            self.rcut_smth = self.descrpt.rcut_r_smth
        elif isinstance(self.descrpt, deepmd.tf.descriptor.DescrptSeT):
            self.sel_a = self.descrpt.sel_a
            self.rcut = self.descrpt.rcut_r
            self.rcut_smth = self.descrpt.rcut_r_smth
        else:
            raise RuntimeError("Unsupported descriptor")

        self.davg = get_tensor_by_name_from_graph(
            self.graph, f"descrpt_attr{self.suffix}/t_avg"
        )
        self.dstd = get_tensor_by_name_from_graph(
            self.graph, f"descrpt_attr{self.suffix}/t_std"
        )
        self.ntypes = self.descrpt.get_ntypes()

        self.embedding_net_nodes = get_embedding_net_nodes_from_graph_def(
            self.graph_def, suffix=self.suffix
        )
        for key in self.embedding_net_nodes.keys():
            assert key.find("bias") > 0 or key.find("matrix") > 0, (
                "currently, only support weight matrix and bias matrix at the tabulation op!"
            )

        # move it to the descriptor class
        # for tt in self.exclude_types:
        #     if (tt[0] not in range(self.ntypes)) or (tt[1] not in range(self.ntypes)):
        #         raise RuntimeError("exclude types" + str(tt) + " must within the number of atomic types " + str(self.ntypes) + "!")
        # if (self.ntypes * self.ntypes - len(self.exclude_types) == 0):
        #     raise RuntimeError("empty embedding-net are not supported in model compression!")
        self.layer_size = self._get_layer_size()
        self.table_size = self._get_table_size()

        self.bias = self._get_bias()
        self.matrix = self._get_matrix()

        self.data_type = self._get_data_type()
        self.last_layer_size = self._get_last_layer_size()

        self.data = {}

        self.upper = {}
        self.lower = {}

    def _load_sub_graph(self):
        sub_graph_def = tf.GraphDef()
        with tf.Graph().as_default() as sub_graph:
            tf.import_graph_def(sub_graph_def, name="")
        return sub_graph, sub_graph_def

    def _get_descrpt_type(self) -> str:
        if isinstance(self.descrpt, deepmd.tf.descriptor.DescrptSeAtten):
            return "Atten"
        elif isinstance(self.descrpt, deepmd.tf.descriptor.DescrptSeAEbdV2):
            return "AEbdV2"
        elif isinstance(self.descrpt, deepmd.tf.descriptor.DescrptSeA):
            return "A"
        elif isinstance(self.descrpt, deepmd.tf.descriptor.DescrptSeT):
            return "T"
        elif isinstance(self.descrpt, deepmd.tf.descriptor.DescrptSeR):
            return "R"
        raise RuntimeError(f"Unsupported descriptor {self.descrpt}")

    def _get_bias(self):
        bias = {}
        for layer in range(1, self.layer_size + 1):
            bias["layer_" + str(layer)] = []
            if isinstance(
                self.descrpt, deepmd.tf.descriptor.DescrptSeAtten
            ) or isinstance(self.descrpt, deepmd.tf.descriptor.DescrptSeAEbdV2):
                node = self.embedding_net_nodes[
                    f"filter_type_all{self.suffix}/bias_{layer}"
                ]
                bias["layer_" + str(layer)].append(tf.make_ndarray(node))
            elif isinstance(self.descrpt, deepmd.tf.descriptor.DescrptSeA):
                if self.type_one_side:
                    for ii in range(0, self.ntypes):
                        if not self._all_excluded(ii):
                            node = self.embedding_net_nodes[
                                f"filter_type_all{self.suffix}/bias_{layer}_{ii}"
                            ]
                            bias["layer_" + str(layer)].append(tf.make_ndarray(node))
                        else:
                            bias["layer_" + str(layer)].append(np.array([]))
                else:
                    for ii in range(0, self.ntypes * self.ntypes):
                        if (
                            ii // self.ntypes,
                            ii % self.ntypes,
                        ) not in self.exclude_types:
                            node = self.embedding_net_nodes[
                                f"filter_type_{ii // self.ntypes}{self.suffix}/bias_{layer}_{ii % self.ntypes}"
                            ]
                            bias["layer_" + str(layer)].append(tf.make_ndarray(node))
                        else:
                            bias["layer_" + str(layer)].append(np.array([]))
            elif isinstance(self.descrpt, deepmd.tf.descriptor.DescrptSeT):
                for ii in range(self.ntypes):
                    for jj in range(ii, self.ntypes):
                        node = self.embedding_net_nodes[
                            f"filter_type_all{self.suffix}/bias_{layer}_{ii}_{jj}"
                        ]
                        bias["layer_" + str(layer)].append(tf.make_ndarray(node))
            elif isinstance(self.descrpt, deepmd.tf.descriptor.DescrptSeR):
                if self.type_one_side:
                    for ii in range(0, self.ntypes):
                        if not self._all_excluded(ii):
                            node = self.embedding_net_nodes[
                                f"filter_type_all{self.suffix}/bias_{layer}_{ii}"
                            ]
                            bias["layer_" + str(layer)].append(tf.make_ndarray(node))
                        else:
                            bias["layer_" + str(layer)].append(np.array([]))
                else:
                    for ii in range(0, self.ntypes * self.ntypes):
                        if (
                            ii // self.ntypes,
                            ii % self.ntypes,
                        ) not in self.exclude_types:
                            node = self.embedding_net_nodes[
                                f"filter_type_{ii // self.ntypes}{self.suffix}/bias_{layer}_{ii % self.ntypes}"
                            ]
                            bias["layer_" + str(layer)].append(tf.make_ndarray(node))
                        else:
                            bias["layer_" + str(layer)].append(np.array([]))
            else:
                raise RuntimeError("Unsupported descriptor")
        return bias

    def _get_matrix(self):
        matrix = {}
        for layer in range(1, self.layer_size + 1):
            matrix["layer_" + str(layer)] = []
            if isinstance(
                self.descrpt, deepmd.tf.descriptor.DescrptSeAtten
            ) or isinstance(self.descrpt, deepmd.tf.descriptor.DescrptSeAEbdV2):
                node = self.embedding_net_nodes[
                    f"filter_type_all{self.suffix}/matrix_{layer}"
                ]
                matrix["layer_" + str(layer)].append(tf.make_ndarray(node))
            elif isinstance(self.descrpt, deepmd.tf.descriptor.DescrptSeA):
                if self.type_one_side:
                    for ii in range(0, self.ntypes):
                        if not self._all_excluded(ii):
                            node = self.embedding_net_nodes[
                                f"filter_type_all{self.suffix}/matrix_{layer}_{ii}"
                            ]
                            matrix["layer_" + str(layer)].append(tf.make_ndarray(node))
                        else:
                            matrix["layer_" + str(layer)].append(np.array([]))
                else:
                    for ii in range(0, self.ntypes * self.ntypes):
                        if (
                            ii // self.ntypes,
                            ii % self.ntypes,
                        ) not in self.exclude_types:
                            node = self.embedding_net_nodes[
                                f"filter_type_{ii // self.ntypes}{self.suffix}/matrix_{layer}_{ii % self.ntypes}"
                            ]
                            matrix["layer_" + str(layer)].append(tf.make_ndarray(node))
                        else:
                            matrix["layer_" + str(layer)].append(np.array([]))
            elif isinstance(self.descrpt, deepmd.tf.descriptor.DescrptSeT):
                for ii in range(self.ntypes):
                    for jj in range(ii, self.ntypes):
                        node = self.embedding_net_nodes[
                            f"filter_type_all{self.suffix}/matrix_{layer}_{ii}_{jj}"
                        ]
                        matrix["layer_" + str(layer)].append(tf.make_ndarray(node))
            elif isinstance(self.descrpt, deepmd.tf.descriptor.DescrptSeR):
                if self.type_one_side:
                    for ii in range(0, self.ntypes):
                        if not self._all_excluded(ii):
                            node = self.embedding_net_nodes[
                                f"filter_type_all{self.suffix}/matrix_{layer}_{ii}"
                            ]
                            matrix["layer_" + str(layer)].append(tf.make_ndarray(node))
                        else:
                            matrix["layer_" + str(layer)].append(np.array([]))
                else:
                    for ii in range(0, self.ntypes * self.ntypes):
                        if (
                            ii // self.ntypes,
                            ii % self.ntypes,
                        ) not in self.exclude_types:
                            node = self.embedding_net_nodes[
                                f"filter_type_{ii // self.ntypes}{self.suffix}/matrix_{layer}_{ii % self.ntypes}"
                            ]
                            matrix["layer_" + str(layer)].append(tf.make_ndarray(node))
                        else:
                            matrix["layer_" + str(layer)].append(np.array([]))
            else:
                raise RuntimeError("Unsupported descriptor")

        return matrix

    # one-by-one executions
    def _make_data(self, xx, idx):
        with self.sub_graph.as_default():
            with self.sub_sess.as_default():
                xx = tf.reshape(xx, [xx.size, -1])
                for layer in range(self.layer_size):
                    if layer == 0:
                        xbar = (
                            tf.matmul(xx, self.matrix["layer_" + str(layer + 1)][idx])
                            + self.bias["layer_" + str(layer + 1)][idx]
                        )
                        if self.neuron[0] == 1:
                            yy = (
                                self._layer_0(
                                    xx,
                                    self.matrix["layer_" + str(layer + 1)][idx],
                                    self.bias["layer_" + str(layer + 1)][idx],
                                )
                                + xx
                            )
                            dy = op_module.unaggregated_dy_dx_s(
                                yy - xx,
                                self.matrix["layer_" + str(layer + 1)][idx],
                                xbar,
                                tf.constant(self.functype),
                            ) + tf.ones([1, 1], yy.dtype)  # pylint: disable=no-explicit-dtype
                            dy2 = op_module.unaggregated_dy2_dx_s(
                                yy - xx,
                                dy,
                                self.matrix["layer_" + str(layer + 1)][idx],
                                xbar,
                                tf.constant(self.functype),
                            )
                        elif self.neuron[0] == 2:
                            tt, yy = self._layer_1(
                                xx,
                                self.matrix["layer_" + str(layer + 1)][idx],
                                self.bias["layer_" + str(layer + 1)][idx],
                            )
                            dy = op_module.unaggregated_dy_dx_s(
                                yy - tt,
                                self.matrix["layer_" + str(layer + 1)][idx],
                                xbar,
                                tf.constant(self.functype),
                            ) + tf.ones([1, 2], yy.dtype)  # pylint: disable=no-explicit-dtype
                            dy2 = op_module.unaggregated_dy2_dx_s(
                                yy - tt,
                                dy,
                                self.matrix["layer_" + str(layer + 1)][idx],
                                xbar,
                                tf.constant(self.functype),
                            )
                        else:
                            yy = self._layer_0(
                                xx,
                                self.matrix["layer_" + str(layer + 1)][idx],
                                self.bias["layer_" + str(layer + 1)][idx],
                            )
                            dy = op_module.unaggregated_dy_dx_s(
                                yy,
                                self.matrix["layer_" + str(layer + 1)][idx],
                                xbar,
                                tf.constant(self.functype),
                            )
                            dy2 = op_module.unaggregated_dy2_dx_s(
                                yy,
                                dy,
                                self.matrix["layer_" + str(layer + 1)][idx],
                                xbar,
                                tf.constant(self.functype),
                            )
                    else:
                        ybar = (
                            tf.matmul(yy, self.matrix["layer_" + str(layer + 1)][idx])
                            + self.bias["layer_" + str(layer + 1)][idx]
                        )
                        if self.neuron[layer] == self.neuron[layer - 1]:
                            zz = (
                                self._layer_0(
                                    yy,
                                    self.matrix["layer_" + str(layer + 1)][idx],
                                    self.bias["layer_" + str(layer + 1)][idx],
                                )
                                + yy
                            )
                            dz = op_module.unaggregated_dy_dx(
                                zz - yy,
                                self.matrix["layer_" + str(layer + 1)][idx],
                                dy,
                                ybar,
                                tf.constant(self.functype),
                            )
                            dy2 = op_module.unaggregated_dy2_dx(
                                zz - yy,
                                self.matrix["layer_" + str(layer + 1)][idx],
                                dy,
                                dy2,
                                ybar,
                                tf.constant(self.functype),
                            )
                        elif self.neuron[layer] == 2 * self.neuron[layer - 1]:
                            tt, zz = self._layer_1(
                                yy,
                                self.matrix["layer_" + str(layer + 1)][idx],
                                self.bias["layer_" + str(layer + 1)][idx],
                            )
                            dz = op_module.unaggregated_dy_dx(
                                zz - tt,
                                self.matrix["layer_" + str(layer + 1)][idx],
                                dy,
                                ybar,
                                tf.constant(self.functype),
                            )
                            dy2 = op_module.unaggregated_dy2_dx(
                                zz - tt,
                                self.matrix["layer_" + str(layer + 1)][idx],
                                dy,
                                dy2,
                                ybar,
                                tf.constant(self.functype),
                            )
                        else:
                            zz = self._layer_0(
                                yy,
                                self.matrix["layer_" + str(layer + 1)][idx],
                                self.bias["layer_" + str(layer + 1)][idx],
                            )
                            dz = op_module.unaggregated_dy_dx(
                                zz,
                                self.matrix["layer_" + str(layer + 1)][idx],
                                dy,
                                ybar,
                                tf.constant(self.functype),
                            )
                            dy2 = op_module.unaggregated_dy2_dx(
                                zz,
                                self.matrix["layer_" + str(layer + 1)][idx],
                                dy,
                                dy2,
                                ybar,
                                tf.constant(self.functype),
                            )
                        dy = dz
                        yy = zz

                vv = zz.eval()
                dd = dy.eval()
                d2 = dy2.eval()
        return vv, dd, d2

    def _layer_0(self, x, w, b):
        return self.activation_fn(tf.matmul(x, w) + b)

    def _layer_1(self, x, w, b):
        t = tf.concat([x, x], axis=1)
        return t, self.activation_fn(tf.matmul(x, w) + b) + t

    def _get_layer_size(self):
        layer_size = 0
        if isinstance(self.descrpt, deepmd.tf.descriptor.DescrptSeAtten) or isinstance(
            self.descrpt, deepmd.tf.descriptor.DescrptSeAEbdV2
        ):
            layer_size = len(self.embedding_net_nodes) // 2
        elif isinstance(self.descrpt, deepmd.tf.descriptor.DescrptSeA):
            layer_size = len(self.embedding_net_nodes) // (
                (self.ntypes * self.ntypes - len(self.exclude_types)) * 2
            )
            if self.type_one_side:
                layer_size = len(self.embedding_net_nodes) // (
                    (self.ntypes - self._n_all_excluded) * 2
                )
        elif isinstance(self.descrpt, deepmd.tf.descriptor.DescrptSeT):
            layer_size = len(self.embedding_net_nodes) // int(
                comb(self.ntypes + 1, 2) * 2
            )
        elif isinstance(self.descrpt, deepmd.tf.descriptor.DescrptSeR):
            layer_size = len(self.embedding_net_nodes) // (
                (self.ntypes * self.ntypes - len(self.exclude_types)) * 2
            )
            if self.type_one_side:
                layer_size = len(self.embedding_net_nodes) // (
                    (self.ntypes - self._n_all_excluded) * 2
                )
        else:
            raise RuntimeError("Unsupported descriptor")
        return layer_size

    @cached_property
    def _n_all_excluded(self) -> int:
        """Then number of types excluding all types."""
        return sum(int(self._all_excluded(ii)) for ii in range(0, self.ntypes))

    def _convert_numpy_to_tensor(self) -> None:
        """Convert self.data from np.ndarray to tf.Tensor."""
        for ii in self.data:
            self.data[ii] = tf.constant(self.data[ii])
