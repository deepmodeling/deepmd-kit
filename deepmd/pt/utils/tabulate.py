# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from functools import (
    lru_cache,
)
from typing import (
    Callable,
)

import numpy as np
import torch
from scipy.special import (
    comb,
)

import deepmd
from deepmd.pt.utils.env import (
    ACTIVATION_FN_DICT,
)

log = logging.getLogger(__name__)

SQRT_2_PI = np.sqrt(2 / np.pi)
GGELU = 0.044715


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
    type_one_side
            Try to build N_types tables. Otherwise, building N_types^2 tables
    exclude_types : List[List[int]]
            The excluded pairs of types which have no interaction with each other.
            For example, `[[0, 1]]` means no interaction between type 0 and type 1.
    activation_function
            The activation function in the embedding net. Supported options are {"tanh","gelu"} in common.ACTIVATION_FN_DICT.
    """

    def __init__(
        self,
        descrpt,
        neuron: list[int],
        type_one_side: bool = False,
        exclude_types: list[list[int]] = [],
        activation_fn: Callable[[torch.Tensor], torch.Tensor] = torch.tanh,
    ) -> None:
        """Constructor."""
        self.descrpt = descrpt
        self.neuron = neuron
        self.type_one_side = type_one_side
        self.exclude_types = exclude_types

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

        if (
            isinstance(self.descrpt, deepmd.pt.model.descriptor.DescrptSeR)
            or isinstance(self.descrpt, deepmd.pt.model.descriptor.DescrptSeA)
            or isinstance(self.descrpt, deepmd.pt.model.descriptor.DescrptSeT)
            or isinstance(self.descrpt, deepmd.pt.model.descriptor.DescrptSeT)
            or isinstance(self.descrpt, deepmd.pt.model.descriptor.DescrptDPA1)
        ):
            self.sel_a = self.descrpt.get_sel()
            self.rcut = self.descrpt.get_rcut()
            self.rcut_smth = self.descrpt.get_rcut_smth()
        else:
            raise RuntimeError("Unsupported descriptor")

        self.davg = self.descrpt.serialize()["@variables"]["davg"]
        self.dstd = self.descrpt.serialize()["@variables"]["dstd"]
        self.ntypes = self.descrpt.get_ntypes()

        self.embedding_net_nodes = self.descrpt.serialize()["embeddings"]["networks"]

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
    ) -> tuple[dict[str, int], dict[str, int]]:
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
        if isinstance(self.descrpt, deepmd.pt.model.descriptor.DescrptDPA1):
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
        elif isinstance(self.descrpt, deepmd.pt.model.descriptor.DescrptSeA):
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
                        uu = np.max(upper[ielement])
                        ll = np.min(lower[ielement])
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
        elif isinstance(self.descrpt, deepmd.pt.model.descriptor.DescrptSeT):
            xx_all = []
            for ii in range(self.ntypes):
                uu = np.max(upper[ii])
                ll = np.min(lower[ii])
                xx = np.arange(extrapolate * ll, ll, stride1, dtype=self.data_type)
                xx = np.append(xx, np.arange(ll, uu, stride0, dtype=self.data_type))
                xx = np.append(
                    xx,
                    np.arange(
                        uu,
                        extrapolate * uu,
                        stride1,
                        dtype=self.data_type,
                    ),
                )
                xx = np.append(xx, np.array([extrapolate * uu], dtype=self.data_type))
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
                        uu,
                        ll,
                        stride0,
                        stride1,
                        extrapolate,
                        nspline[ii][0],
                    )
                    idx += 1
        elif isinstance(self.descrpt, deepmd.pt.model.descriptor.DescrptSeR):
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
        if isinstance(
            self.descrpt, deepmd.pt.model.descriptor.DescrptSeA
        ) or isinstance(self.descrpt, deepmd.pt.model.descriptor.DescrptDPA1):
            tt = np.full((nspline, self.last_layer_size), stride1)  # pylint: disable=no-explicit-dtype
            tt[: int((upper - lower) / stride0), :] = stride0
        elif isinstance(self.descrpt, deepmd.pt.model.descriptor.DescrptSeT):
            tt = np.full((nspline, self.last_layer_size), stride1)  # pylint: disable=no-explicit-dtype
            tt[
                int((lower - extrapolate * lower) / stride1) + 1 : (
                    int((lower - extrapolate * lower) / stride1)
                    + int((upper - lower) / stride0)
                ),
                :,
            ] = stride0
        elif isinstance(self.descrpt, deepmd.pt.model.descriptor.DescrptSeR):
            tt = np.full((nspline, self.last_layer_size), stride1)  # pylint: disable=no-explicit-dtype
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

    def _make_data(self, xx, idx):
        xx = torch.from_numpy(xx)
        xx = xx.view(xx.size(0), -1)
        for layer in range(self.layer_size):
            if layer == 0:
                xbar = torch.matmul(
                    xx, torch.from_numpy(self.matrix["layer_" + str(layer + 1)][idx])
                ) + torch.from_numpy(self.bias["layer_" + str(layer + 1)][idx])
                if self.neuron[0] == 1:
                    yy = (
                        self._layer_0(
                            xx,
                            self.matrix["layer_" + str(layer + 1)][idx],
                            self.bias["layer_" + str(layer + 1)][idx],
                        )
                        + xx
                    )
                    dy = unaggregated_dy_dx_s(
                        yy - xx,
                        self.matrix["layer_" + str(layer + 1)][idx],
                        xbar,
                        self.functype,
                    ) + torch.ones((1, 1), dtype=yy.dtype)  # pylint: disable=no-explicit-device
                    dy2 = unaggregated_dy2_dx_s(
                        yy - xx,
                        dy,
                        self.matrix["layer_" + str(layer + 1)][idx],
                        xbar,
                        self.functype,
                    )
                elif self.neuron[0] == 2:
                    tt, yy = self._layer_1(
                        xx,
                        self.matrix["layer_" + str(layer + 1)][idx],
                        self.bias["layer_" + str(layer + 1)][idx],
                    )
                    dy = unaggregated_dy_dx_s(
                        yy - tt,
                        self.matrix["layer_" + str(layer + 1)][idx],
                        xbar,
                        self.functype,
                    ) + torch.ones((1, 2), dtype=yy.dtype)  # pylint: disable=no-explicit-device
                    dy2 = unaggregated_dy2_dx_s(
                        yy - tt,
                        dy,
                        self.matrix["layer_" + str(layer + 1)][idx],
                        xbar,
                        self.functype,
                    )
                else:
                    yy = self._layer_0(
                        xx,
                        self.matrix["layer_" + str(layer + 1)][idx],
                        self.bias["layer_" + str(layer + 1)][idx],
                    )
                    dy = unaggregated_dy_dx_s(
                        yy,
                        self.matrix["layer_" + str(layer + 1)][idx],
                        xbar,
                        self.functype,
                    )
                    dy2 = unaggregated_dy2_dx_s(
                        yy,
                        dy,
                        self.matrix["layer_" + str(layer + 1)][idx],
                        xbar,
                        self.functype,
                    )
            else:
                ybar = torch.matmul(
                    yy, torch.from_numpy(self.matrix["layer_" + str(layer + 1)][idx])
                ) + torch.from_numpy(self.bias["layer_" + str(layer + 1)][idx])
                if self.neuron[layer] == self.neuron[layer - 1]:
                    zz = (
                        self._layer_0(
                            yy,
                            self.matrix["layer_" + str(layer + 1)][idx],
                            self.bias["layer_" + str(layer + 1)][idx],
                        )
                        + yy
                    )
                    dz = unaggregated_dy_dx(
                        zz - yy,
                        self.matrix["layer_" + str(layer + 1)][idx],
                        dy,
                        ybar,
                        self.functype,
                    )
                    dy2 = unaggregated_dy2_dx(
                        zz - yy,
                        self.matrix["layer_" + str(layer + 1)][idx],
                        dy,
                        dy2,
                        ybar,
                        self.functype,
                    )
                elif self.neuron[layer] == 2 * self.neuron[layer - 1]:
                    tt, zz = self._layer_1(
                        yy,
                        self.matrix["layer_" + str(layer + 1)][idx],
                        self.bias["layer_" + str(layer + 1)][idx],
                    )
                    dz = unaggregated_dy_dx(
                        zz - tt,
                        self.matrix["layer_" + str(layer + 1)][idx],
                        dy,
                        ybar,
                        self.functype,
                    )
                    dy2 = unaggregated_dy2_dx(
                        zz - tt,
                        self.matrix["layer_" + str(layer + 1)][idx],
                        dy,
                        dy2,
                        ybar,
                        self.functype,
                    )
                else:
                    zz = self._layer_0(
                        yy,
                        self.matrix["layer_" + str(layer + 1)][idx],
                        self.bias["layer_" + str(layer + 1)][idx],
                    )
                    dz = unaggregated_dy_dx(
                        zz,
                        self.matrix["layer_" + str(layer + 1)][idx],
                        dy,
                        ybar,
                        self.functype,
                    )
                    dy2 = unaggregated_dy2_dx(
                        zz,
                        self.matrix["layer_" + str(layer + 1)][idx],
                        dy,
                        dy2,
                        ybar,
                        self.functype,
                    )
                dy = dz
                yy = zz

        vv = zz.detach().numpy()
        dd = dy.detach().numpy()
        d2 = dy2.detach().numpy()
        return vv, dd, d2

    def _layer_0(self, x, w, b):
        w = torch.from_numpy(w)
        b = torch.from_numpy(b)
        return self.activation_fn(torch.matmul(x, w) + b)

    def _layer_1(self, x, w, b):
        w = torch.from_numpy(w)
        b = torch.from_numpy(b)
        t = torch.cat([x, x], dim=1)
        return t, self.activation_fn(torch.matmul(x, w) + b) + t

    # Change the embedding net range to sw / min_nbor_dist
    def _get_env_mat_range(self, min_nbor_dist):
        sw = self._spline5_switch(min_nbor_dist, self.rcut_smth, self.rcut)
        if isinstance(
            self.descrpt, deepmd.pt.model.descriptor.DescrptSeA
        ) or isinstance(self.descrpt, deepmd.pt.model.descriptor.DescrptDPA1):
            lower = -self.davg[:, 0] / self.dstd[:, 0]
            upper = ((1 / min_nbor_dist) * sw - self.davg[:, 0]) / self.dstd[:, 0]
        elif isinstance(self.descrpt, deepmd.pt.model.descriptor.DescrptSeT):
            var = np.square(sw / (min_nbor_dist * self.dstd[:, 1:4]))
            lower = np.min(-var, axis=1)
            upper = np.max(var, axis=1)
        elif isinstance(self.descrpt, deepmd.pt.model.descriptor.DescrptSeR):
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
        # get the number of layers in EmbeddingNet
        layer_size = 0
        basic_size = 0
        if self.type_one_side:
            basic_size = len(self.embedding_net_nodes) * len(self.neuron)
        else:
            basic_size = (
                len(self.embedding_net_nodes)
                * len(self.embedding_net_nodes[0])
                * len(self.neuron)
            )
        if isinstance(self.descrpt, deepmd.pt.model.descriptor.DescrptDPA1):
            layer_size = len(self.embedding_net_nodes[0]["layers"])
        elif isinstance(self.descrpt, deepmd.pt.model.descriptor.DescrptSeA):
            layer_size = len(self.embedding_net_nodes[0]["layers"])
            if self.type_one_side:
                layer_size = basic_size // (self.ntypes - self._n_all_excluded)
        elif isinstance(self.descrpt, deepmd.pt.model.descriptor.DescrptSeT):
            layer_size = len(self.embedding_net_nodes[0]["layers"])
            # layer_size = basic_size // int(comb(self.ntypes + 1, 2))
        elif isinstance(self.descrpt, deepmd.pt.model.descriptor.DescrptSeR):
            layer_size = basic_size // (
                self.ntypes * self.ntypes - len(self.exclude_types)
            )
            if self.type_one_side:
                layer_size = basic_size // (self.ntypes - self._n_all_excluded)
        else:
            raise RuntimeError("Unsupported descriptor")
        return layer_size

    def _get_table_size(self):
        table_size = 0
        if isinstance(self.descrpt, deepmd.pt.model.descriptor.DescrptDPA1):
            table_size = 1
        elif isinstance(self.descrpt, deepmd.pt.model.descriptor.DescrptSeA):
            table_size = self.ntypes * self.ntypes
            if self.type_one_side:
                table_size = self.ntypes
        elif isinstance(self.descrpt, deepmd.pt.model.descriptor.DescrptSeT):
            table_size = int(comb(self.ntypes + 1, 2))
        elif isinstance(self.descrpt, deepmd.pt.model.descriptor.DescrptSeR):
            table_size = self.ntypes * self.ntypes
            if self.type_one_side:
                table_size = self.ntypes
        else:
            raise RuntimeError("Unsupported descriptor")
        return table_size

    def _get_bias(self):
        bias = {}
        for layer in range(1, self.layer_size + 1):
            bias["layer_" + str(layer)] = []
            if isinstance(self.descrpt, deepmd.pt.model.descriptor.DescrptDPA1):
                node = self.embedding_net_nodes[0]["layers"][layer - 1]["@variables"][
                    "b"
                ]
                bias["layer_" + str(layer)].append(node)
            elif isinstance(self.descrpt, deepmd.pt.model.descriptor.DescrptSeA):
                if self.type_one_side:
                    for ii in range(0, self.ntypes):
                        if not self._all_excluded(ii):
                            node = self.embedding_net_nodes[ii]["layers"][layer - 1][
                                "@variables"
                            ]["b"]
                            # node = torch.from_numpy(node)
                            bias["layer_" + str(layer)].append(node)
                        else:
                            # bias["layer_" + str(layer)].append(torch.tensor([]))
                            bias["layer_" + str(layer)].append(np.array([]))
                else:
                    for ii in range(0, self.ntypes * self.ntypes):
                        if (
                            ii // self.ntypes,
                            ii % self.ntypes,
                        ) not in self.exclude_types:
                            # node = self.embedding_net_nodes[ii // self.ntypes][ii % self.ntypes]["layers"][layer - 1]["@variables"]["b"]
                            node = self.embedding_net_nodes[
                                (ii % self.ntypes) * self.ntypes + ii // self.ntypes
                            ]["layers"][layer - 1]["@variables"]["b"]
                            # node = torch.from_numpy(node)
                            bias["layer_" + str(layer)].append(node)
                        else:
                            # bias["layer_" + str(layer)].append(torch.tensor([]))
                            bias["layer_" + str(layer)].append(np.array([]))
            elif isinstance(self.descrpt, deepmd.pt.model.descriptor.DescrptSeT):
                for ii in range(self.ntypes):
                    for jj in range(ii, self.ntypes):
                        node = self.embedding_net_nodes[jj * self.ntypes + ii][
                            "layers"
                        ][layer - 1]["@variables"]["b"]
                        # node = torch.from_numpy(node)
                        bias["layer_" + str(layer)].append(node)
            elif isinstance(self.descrpt, deepmd.pt.model.descriptor.DescrptSeR):
                if self.type_one_side:
                    for ii in range(0, self.ntypes):
                        if not self._all_excluded(ii):
                            node = self.embedding_net_nodes[ii]["layers"][layer - 1][
                                "@variables"
                            ]["b"]
                            # node = torch.from_numpy(node)
                            bias["layer_" + str(layer)].append(node)
                        else:
                            # bias["layer_" + str(layer)].append(torch.tensor([]))
                            bias["layer_" + str(layer)].append(np.array([]))
                else:
                    for ii in range(0, self.ntypes * self.ntypes):
                        if (
                            ii // self.ntypes,
                            ii % self.ntypes,
                        ) not in self.exclude_types:
                            node = self.embedding_net_nodes[
                                (ii % self.ntypes) * self.ntypes + ii // self.ntypes
                            ]["layers"][layer - 1]["@variables"]["b"]
                            # node = torch.from_numpy(node)
                            bias["layer_" + str(layer)].append(node)
                        else:
                            # bias["layer_" + str(layer)].append(torch.tensor([]))
                            bias["layer_" + str(layer)].append(np.array([]))
            else:
                raise RuntimeError("Unsupported descriptor")
        return bias

    def _get_matrix(self):
        matrix = {}
        for layer in range(1, self.layer_size + 1):
            matrix["layer_" + str(layer)] = []
            if isinstance(self.descrpt, deepmd.pt.model.descriptor.DescrptDPA1):
                node = self.embedding_net_nodes[0]["layers"][layer - 1]["@variables"][
                    "w"
                ]
                matrix["layer_" + str(layer)].append(node)
            elif isinstance(self.descrpt, deepmd.pt.model.descriptor.DescrptSeA):
                if self.type_one_side:
                    for ii in range(0, self.ntypes):
                        if not self._all_excluded(ii):
                            node = self.embedding_net_nodes[ii]["layers"][layer - 1][
                                "@variables"
                            ]["w"]
                            # node = torch.from_numpy(node)
                            matrix["layer_" + str(layer)].append(node)
                        else:
                            # matrix["layer_" + str(layer)].append(torch.tensor([]))
                            matrix["layer_" + str(layer)].append(np.array([]))
                else:
                    for ii in range(0, self.ntypes * self.ntypes):
                        if (
                            ii // self.ntypes,
                            ii % self.ntypes,
                        ) not in self.exclude_types:
                            # node = self.embedding_net_nodes[ii // self.ntypes][ii % self.ntypes]["layers"][layer - 1]["@variables"]["w"]
                            node = self.embedding_net_nodes[
                                (ii % self.ntypes) * self.ntypes + ii // self.ntypes
                            ]["layers"][layer - 1]["@variables"]["w"]
                            # node = torch.from_numpy(node)
                            matrix["layer_" + str(layer)].append(node)
                        else:
                            # matrix["layer_" + str(layer)].append(torch.tensor([]))
                            matrix["layer_" + str(layer)].append(np.array([]))
            elif isinstance(self.descrpt, deepmd.pt.model.descriptor.DescrptSeT):
                for ii in range(self.ntypes):
                    for jj in range(ii, self.ntypes):
                        node = self.embedding_net_nodes[jj * self.ntypes + ii][
                            "layers"
                        ][layer - 1]["@variables"]["w"]
                        # node = torch.from_numpy(node)
                        matrix["layer_" + str(layer)].append(node)
            elif isinstance(self.descrpt, deepmd.pt.model.descriptor.DescrptSeR):
                if self.type_one_side:
                    for ii in range(0, self.ntypes):
                        if not self._all_excluded(ii):
                            node = self.embedding_net_nodes[ii]["layers"][layer - 1][
                                "@variables"
                            ]["w"]
                            # node = torch.from_numpy(node)
                            matrix["layer_" + str(layer)].append(node)
                        else:
                            # matrix["layer_" + str(layer)].append(torch.tensor([]))
                            matrix["layer_" + str(layer)].append(np.array([]))
                else:
                    for ii in range(0, self.ntypes * self.ntypes):
                        if (
                            ii // self.ntypes,
                            ii % self.ntypes,
                        ) not in self.exclude_types:
                            node = self.embedding_net_nodes[
                                (ii % self.ntypes) * self.ntypes + ii // self.ntypes
                            ]["layers"][layer - 1]["@variables"]["w"]
                            # node = torch.from_numpy(node)
                            matrix["layer_" + str(layer)].append(node)
                        else:
                            # matrix["layer_" + str(layer)].append(torch.tensor([]))
                            matrix["layer_" + str(layer)].append(np.array([]))
            else:
                raise RuntimeError("Unsupported descriptor")

        return matrix

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
        """Convert self.data from np.ndarray to torch.Tensor."""
        for ii in self.data:
            self.data[ii] = torch.tensor(self.data[ii])  # pylint: disable=no-explicit-device, no-explicit-dtype

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


# customized op
def grad(xbar, y, functype):  # functype=tanh, gelu, ..
    if functype == 1:
        return 1 - y * y
    elif functype == 2:
        var = np.tanh(SQRT_2_PI * (xbar + GGELU * xbar**3))
        return (
            0.5 * SQRT_2_PI * xbar * (1 - var**2) * (3 * GGELU * xbar**2 + 1)
            + 0.5 * var
            + 0.5
        )
    elif functype == 3:
        return 0.0 if xbar <= 0 else 1.0
    elif functype == 4:
        return 0.0 if xbar <= 0 or xbar >= 6 else 1.0
    elif functype == 5:
        return 1.0 - 1.0 / (1.0 + np.exp(xbar))
    elif functype == 6:
        return y * (1 - y)
    else:
        return -1.0


def grad_grad(xbar, y, functype):
    if functype == 1:
        return -2 * y * (1 - y * y)
    elif functype == 2:
        var1 = np.tanh(SQRT_2_PI * (xbar + GGELU * xbar**3))
        var2 = SQRT_2_PI * (1 - var1**2) * (3 * GGELU * xbar**2 + 1)
        return (
            3 * GGELU * SQRT_2_PI * xbar**2 * (1 - var1**2)
            - SQRT_2_PI * xbar * var2 * (3 * GGELU * xbar**2 + 1) * var1
            + var2
        )
    elif functype == 3:
        return 0
    elif functype == 4:
        return 0
    elif functype == 5:
        return np.exp(xbar) / ((1 + np.exp(xbar)) * (1 + np.exp(xbar)))
    elif functype == 6:
        return y * (1 - y) * (1 - 2 * y)
    else:
        return -1


def unaggregated_dy_dx_s(
    y: torch.Tensor, w: np.array, xbar: torch.Tensor, functype: int
):
    w = torch.from_numpy(w)
    if y.dim() != 2:
        raise ValueError("Dim of input y should be 2")
    if w.dim() != 2:
        raise ValueError("Dim of input w should be 2")
    if xbar.dim() != 2:
        raise ValueError("Dim of input xbar should be 2")

    length, width = y.shape
    dy_dx = torch.zeros_like(y)
    w = torch.flatten(w)

    for ii in range(length):
        for jj in range(width):
            dy_dx[ii, jj] = grad(xbar[ii, jj], y[ii, jj], functype) * w[jj]

    return dy_dx


def unaggregated_dy2_dx_s(
    y: torch.Tensor, dy: torch.tensor, w: np.array, xbar: torch.Tensor, functype: int
):
    w = torch.from_numpy(w)
    if y.dim() != 2:
        raise ValueError("Dim of input y should be 2")
    if dy.dim() != 2:
        raise ValueError("Dim of input dy should be 2")
    if w.dim() != 2:
        raise ValueError("Dim of input w should be 2")
    if xbar.dim() != 2:
        raise ValueError("Dim of input xbar should be 2")

    length, width = y.shape
    dy2_dx = torch.zeros_like(y)
    w = torch.flatten(w)

    for ii in range(length):
        for jj in range(width):
            dy2_dx[ii, jj] = (
                grad_grad(xbar[ii, jj], y[ii, jj], functype) * w[jj] * w[jj]
            )

    return dy2_dx


def unaggregated_dy_dx(
    z: torch.Tensor, w: np.array, dy_dx: torch.Tensor, ybar: torch.Tensor, functype: int
):
    w = torch.from_numpy(w)
    if z.dim() != 2:
        raise ValueError("z tensor must have 2 dimensions")
    if w.dim() != 2:
        raise ValueError("w tensor must have 2 dimensions")
    if dy_dx.dim() != 2:
        raise ValueError("dy_dx tensor must have 2 dimensions")
    if ybar.dim() != 2:
        raise ValueError("ybar tensor must have 2 dimensions")

    length, width = z.shape
    size = w.shape[0]
    dy_dx = torch.flatten(dy_dx)

    dz_dx = torch.zeros_like(z)

    for kk in range(length):
        for ii in range(width):
            dz_drou = grad(ybar[kk, ii], z[kk, ii], functype)
            accumulator = 0.0
            for jj in range(size):
                accumulator += w[jj, ii] * dy_dx[kk * size + jj]
            dz_drou *= accumulator
            if width == 2 * size or width == size:
                dz_drou += dy_dx[kk * size + ii % size]
            dz_dx[kk, ii] = dz_drou

    return dz_dx


def unaggregated_dy2_dx(
    z: torch.Tensor,
    w: np.array,
    dy_dx: torch.Tensor,
    dy2_dx: torch.Tensor,
    ybar: torch.Tensor,
    functype: int,
):
    w = torch.from_numpy(w)
    if z.dim() != 2:
        raise ValueError("z tensor must have 2 dimensions")
    if w.dim() != 2:
        raise ValueError("w tensor must have 2 dimensions")
    if dy_dx.dim() != 2:
        raise ValueError("dy_dx tensor must have 2 dimensions")
    if dy2_dx.dim() != 2:
        raise ValueError("dy2_dx tensor must have 2 dimensions")
    if ybar.dim() != 2:
        raise ValueError("ybar tensor must have 2 dimensions")

    length, width = z.shape
    size = w.shape[0]
    dy_dx = torch.flatten(dy_dx)
    dy2_dx = torch.flatten(dy2_dx)

    dz2_dx = torch.zeros_like(z)

    for kk in range(length):
        for ii in range(width):
            dz_drou = grad(ybar[kk, ii], z[kk, ii], functype)
            accumulator1 = 0.0
            for jj in range(size):
                accumulator1 += w[jj, ii] * dy2_dx[kk * size + jj]
            dz_drou *= accumulator1
            accumulator2 = 0.0
            for jj in range(size):
                accumulator2 += w[jj, ii] * dy_dx[kk * size + jj]
            dz_drou += (
                grad_grad(ybar[kk, ii], z[kk, ii], functype)
                * accumulator2
                * accumulator2
            )
            if width == 2 * size or width == size:
                dz_drou += dy2_dx[kk * size + ii % size]
            dz2_dx[kk, ii] = dz_drou

    return dz2_dx
