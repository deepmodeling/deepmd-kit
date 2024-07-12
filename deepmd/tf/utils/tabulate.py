# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from functools import (
    lru_cache,
)
from typing import (
    Callable,
    Dict,
    List,
    Tuple,
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

log = logging.getLogger(__name__)


class DPTabulate:
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
    exclude_types : List[List[int]]
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
        neuron: List[int],
        graph: tf.Graph,
        graph_def: tf.GraphDef,
        type_one_side: bool = False,
        exclude_types: List[List[int]] = [],
        activation_fn: Callable[[tf.Tensor], tf.Tensor] = tf.nn.tanh,
        suffix: str = "",
    ) -> None:
        """Constructor."""
        self.descrpt = descrpt
        self.neuron = neuron
        self.graph = graph
        self.graph_def = graph_def
        self.type_one_side = type_one_side
        self.exclude_types = exclude_types
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
            raise RuntimeError("Unknown actication function type!")
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
        self.ntypes = get_tensor_by_name_from_graph(self.graph, "descrpt_attr/ntypes")

        self.embedding_net_nodes = get_embedding_net_nodes_from_graph_def(
            self.graph_def, suffix=self.suffix
        )
        for key in self.embedding_net_nodes.keys():
            assert (
                key.find("bias") > 0 or key.find("matrix") > 0
            ), "currently, only support weight matrix and bias matrix at the tabulation op!"

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

    def build(
        self, min_nbor_dist: float, extrapolate: float, stride0: float, stride1: float
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        r"""Build the tables for model compression.

        Parameters
        ----------
        min_nbor_dist
            The nearest distance between neighbor atoms
        extrapolate
            The scale of model extrapolation
        stride0
            The uniform stride of the first table
        stride1
            The uniform stride of the second table

        Returns
        -------
        lower : dict[str, int]
            The lower boundary of environment matrix by net
        upper : dict[str, int]
            The upper boundary of environment matrix by net
        """
        # tabulate range [lower, upper] with stride0 'stride0'
        lower, upper = self._get_env_mat_range(min_nbor_dist)
        if isinstance(self.descrpt, deepmd.tf.descriptor.DescrptSeAtten) or isinstance(
            self.descrpt, deepmd.tf.descriptor.DescrptSeAEbdV2
        ):
            uu = np.max(upper)
            ll = np.min(lower)
            xx = np.arange(ll, uu, stride0, dtype=self.data_type)
            xx = np.append(
                xx,
                np.arange(uu, extrapolate * uu, stride1, dtype=self.data_type),
            )
            xx = np.append(xx, np.array([extrapolate * uu], dtype=self.data_type))
            nspline = ((uu - ll) / stride0 + (extrapolate * uu - uu) / stride1).astype(
                int
            )
            self._build_lower(
                "filter_net", xx, 0, uu, ll, stride0, stride1, extrapolate, nspline
            )
        elif isinstance(self.descrpt, deepmd.tf.descriptor.DescrptSeA):
            for ii in range(self.table_size):
                if (self.type_one_side and not self._all_excluded(ii)) or (
                    not self.type_one_side
                    and (ii // self.ntypes, ii % self.ntypes) not in self.exclude_types
                ):
                    if self.type_one_side:
                        net = "filter_-1_net_" + str(ii)
                        # upper and lower should consider all types which are not excluded and sel>0
                        idx = [
                            (type_i, ii) not in self.exclude_types
                            and self.sel_a[type_i] > 0
                            for type_i in range(self.ntypes)
                        ]
                        uu = np.max(upper[idx])
                        ll = np.min(lower[idx])
                    else:
                        ielement = ii // self.ntypes
                        net = (
                            "filter_" + str(ielement) + "_net_" + str(ii % self.ntypes)
                        )
                        uu = upper[ielement]
                        ll = lower[ielement]
                    xx = np.arange(ll, uu, stride0, dtype=self.data_type)
                    xx = np.append(
                        xx,
                        np.arange(uu, extrapolate * uu, stride1, dtype=self.data_type),
                    )
                    xx = np.append(
                        xx, np.array([extrapolate * uu], dtype=self.data_type)
                    )
                    nspline = (
                        (uu - ll) / stride0 + (extrapolate * uu - uu) / stride1
                    ).astype(int)
                    self._build_lower(
                        net, xx, ii, uu, ll, stride0, stride1, extrapolate, nspline
                    )
        elif isinstance(self.descrpt, deepmd.tf.descriptor.DescrptSeT):
            xx_all = []
            for ii in range(self.ntypes):
                xx = np.arange(
                    extrapolate * lower[ii], lower[ii], stride1, dtype=self.data_type
                )
                xx = np.append(
                    xx, np.arange(lower[ii], upper[ii], stride0, dtype=self.data_type)
                )
                xx = np.append(
                    xx,
                    np.arange(
                        upper[ii],
                        extrapolate * upper[ii],
                        stride1,
                        dtype=self.data_type,
                    ),
                )
                xx = np.append(
                    xx, np.array([extrapolate * upper[ii]], dtype=self.data_type)
                )
                xx_all.append(xx)
            nspline = (
                (upper - lower) / stride0
                + 2 * ((extrapolate * upper - upper) / stride1)
            ).astype(int)
            idx = 0
            for ii in range(self.ntypes):
                for jj in range(ii, self.ntypes):
                    net = "filter_" + str(ii) + "_net_" + str(jj)
                    self._build_lower(
                        net,
                        xx_all[ii],
                        idx,
                        upper[ii],
                        lower[ii],
                        stride0,
                        stride1,
                        extrapolate,
                        nspline[ii],
                    )
                    idx += 1
        elif isinstance(self.descrpt, deepmd.tf.descriptor.DescrptSeR):
            for ii in range(self.table_size):
                if (self.type_one_side and not self._all_excluded(ii)) or (
                    not self.type_one_side
                    and (ii // self.ntypes, ii % self.ntypes) not in self.exclude_types
                ):
                    if self.type_one_side:
                        net = "filter_-1_net_" + str(ii)
                        # upper and lower should consider all types which are not excluded and sel>0
                        idx = [
                            (type_i, ii) not in self.exclude_types
                            and self.sel_a[type_i] > 0
                            for type_i in range(self.ntypes)
                        ]
                        uu = np.max(upper[idx])
                        ll = np.min(lower[idx])
                    else:
                        ielement = ii // self.ntypes
                        net = (
                            "filter_" + str(ielement) + "_net_" + str(ii % self.ntypes)
                        )
                        uu = upper[ielement]
                        ll = lower[ielement]
                    xx = np.arange(ll, uu, stride0, dtype=self.data_type)
                    xx = np.append(
                        xx,
                        np.arange(uu, extrapolate * uu, stride1, dtype=self.data_type),
                    )
                    xx = np.append(
                        xx, np.array([extrapolate * uu], dtype=self.data_type)
                    )
                    nspline = (
                        (uu - ll) / stride0 + (extrapolate * uu - uu) / stride1
                    ).astype(int)
                    self._build_lower(
                        net, xx, ii, uu, ll, stride0, stride1, extrapolate, nspline
                    )
        else:
            raise RuntimeError("Unsupported descriptor")
        self._convert_numpy_to_tensor()

        return self.lower, self.upper

    def _build_lower(
        self, net, xx, idx, upper, lower, stride0, stride1, extrapolate, nspline
    ):
        vv, dd, d2 = self._make_data(xx, idx)
        self.data[net] = np.zeros(
            [nspline, 6 * self.last_layer_size], dtype=self.data_type
        )

        # tt.shape: [nspline, self.last_layer_size]
        if isinstance(self.descrpt, deepmd.tf.descriptor.DescrptSeA):
            tt = np.full((nspline, self.last_layer_size), stride1)
            tt[: int((upper - lower) / stride0), :] = stride0
        elif isinstance(self.descrpt, deepmd.tf.descriptor.DescrptSeT):
            tt = np.full((nspline, self.last_layer_size), stride1)
            tt[
                int((lower - extrapolate * lower) / stride1) + 1 : (
                    int((lower - extrapolate * lower) / stride1)
                    + int((upper - lower) / stride0)
                ),
                :,
            ] = stride0
        elif isinstance(self.descrpt, deepmd.tf.descriptor.DescrptSeR):
            tt = np.full((nspline, self.last_layer_size), stride1)
            tt[: int((upper - lower) / stride0), :] = stride0
        else:
            raise RuntimeError("Unsupported descriptor")

        # hh.shape: [nspline, self.last_layer_size]
        hh = (
            vv[1 : nspline + 1, : self.last_layer_size]
            - vv[:nspline, : self.last_layer_size]
        )

        self.data[net][:, : 6 * self.last_layer_size : 6] = vv[
            :nspline, : self.last_layer_size
        ]
        self.data[net][:, 1 : 6 * self.last_layer_size : 6] = dd[
            :nspline, : self.last_layer_size
        ]
        self.data[net][:, 2 : 6 * self.last_layer_size : 6] = (
            0.5 * d2[:nspline, : self.last_layer_size]
        )
        self.data[net][:, 3 : 6 * self.last_layer_size : 6] = (
            1 / (2 * tt * tt * tt)
        ) * (
            20 * hh
            - (
                8 * dd[1 : nspline + 1, : self.last_layer_size]
                + 12 * dd[:nspline, : self.last_layer_size]
            )
            * tt
            - (
                3 * d2[:nspline, : self.last_layer_size]
                - d2[1 : nspline + 1, : self.last_layer_size]
            )
            * tt
            * tt
        )
        self.data[net][:, 4 : 6 * self.last_layer_size : 6] = (
            1 / (2 * tt * tt * tt * tt)
        ) * (
            -30 * hh
            + (
                14 * dd[1 : nspline + 1, : self.last_layer_size]
                + 16 * dd[:nspline, : self.last_layer_size]
            )
            * tt
            + (
                3 * d2[:nspline, : self.last_layer_size]
                - 2 * d2[1 : nspline + 1, : self.last_layer_size]
            )
            * tt
            * tt
        )
        self.data[net][:, 5 : 6 * self.last_layer_size : 6] = (
            1 / (2 * tt * tt * tt * tt * tt)
        ) * (
            12 * hh
            - 6
            * (
                dd[1 : nspline + 1, : self.last_layer_size]
                + dd[:nspline, : self.last_layer_size]
            )
            * tt
            + (
                d2[1 : nspline + 1, : self.last_layer_size]
                - d2[:nspline, : self.last_layer_size]
            )
            * tt
            * tt
        )

        self.upper[net] = upper
        self.lower[net] = lower

    def _load_sub_graph(self):
        sub_graph_def = tf.GraphDef()
        with tf.Graph().as_default() as sub_graph:
            tf.import_graph_def(sub_graph_def, name="")
        return sub_graph, sub_graph_def

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
                            ) + tf.ones([1, 1], yy.dtype)
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
                            ) + tf.ones([1, 2], yy.dtype)
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

    # Change the embedding net range to sw / min_nbor_dist
    def _get_env_mat_range(self, min_nbor_dist):
        sw = self._spline5_switch(min_nbor_dist, self.rcut_smth, self.rcut)
        if isinstance(self.descrpt, deepmd.tf.descriptor.DescrptSeA):
            lower = -self.davg[:, 0] / self.dstd[:, 0]
            upper = ((1 / min_nbor_dist) * sw - self.davg[:, 0]) / self.dstd[:, 0]
        elif isinstance(self.descrpt, deepmd.tf.descriptor.DescrptSeT):
            var = np.square(sw / (min_nbor_dist * self.dstd[:, 1:4]))
            lower = np.min(-var, axis=1)
            upper = np.max(var, axis=1)
        elif isinstance(self.descrpt, deepmd.tf.descriptor.DescrptSeR):
            lower = -self.davg[:, 0] / self.dstd[:, 0]
            upper = ((1 / min_nbor_dist) * sw - self.davg[:, 0]) / self.dstd[:, 0]
        else:
            raise RuntimeError("Unsupported descriptor")
        log.info("training data with lower boundary: " + str(lower))
        log.info("training data with upper boundary: " + str(upper))
        # returns element-wise lower and upper
        return np.floor(lower), np.ceil(upper)

    def _spline5_switch(self, xx, rmin, rmax):
        if xx < rmin:
            vv = 1
        elif xx < rmax:
            uu = (xx - rmin) / (rmax - rmin)
            vv = uu * uu * uu * (-6 * uu * uu + 15 * uu - 10) + 1
        else:
            vv = 0
        return vv

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

    @property
    @lru_cache
    def _n_all_excluded(self) -> int:
        """Then number of types excluding all types."""
        return sum(int(self._all_excluded(ii)) for ii in range(0, self.ntypes))

    @lru_cache
    def _all_excluded(self, ii: int) -> bool:
        """Check if type ii excluds all types.

        Parameters
        ----------
        ii : int
            type index

        Returns
        -------
        bool
            if type ii excluds all types
        """
        return all((ii, type_i) in self.exclude_types for type_i in range(self.ntypes))

    def _get_table_size(self):
        table_size = 0
        if isinstance(self.descrpt, deepmd.tf.descriptor.DescrptSeAtten) or isinstance(
            self.descrpt, deepmd.tf.descriptor.DescrptSeAEbdV2
        ):
            table_size = 1
        elif isinstance(self.descrpt, deepmd.tf.descriptor.DescrptSeA):
            table_size = self.ntypes * self.ntypes
            if self.type_one_side:
                table_size = self.ntypes
        elif isinstance(self.descrpt, deepmd.tf.descriptor.DescrptSeT):
            table_size = int(comb(self.ntypes + 1, 2))
        elif isinstance(self.descrpt, deepmd.tf.descriptor.DescrptSeR):
            table_size = self.ntypes * self.ntypes
            if self.type_one_side:
                table_size = self.ntypes
        else:
            raise RuntimeError("Unsupported descriptor")
        return table_size

    def _get_data_type(self):
        for item in self.matrix["layer_" + str(self.layer_size)]:
            if len(item) != 0:
                return type(item[0][0])
        return None

    def _get_last_layer_size(self):
        for item in self.matrix["layer_" + str(self.layer_size)]:
            if len(item) != 0:
                return item.shape[1]
        return 0

    def _convert_numpy_to_tensor(self):
        """Convert self.data from np.ndarray to tf.Tensor."""
        for ii in self.data:
            self.data[ii] = tf.constant(self.data[ii])
