# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from functools import (
    cached_property,
)

import numpy as np
import torch

import deepmd
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.utils import (
    ActivationFn,
)
from deepmd.utils.tabulate import (
    BaseTabulate,
)

log = logging.getLogger(__name__)

SQRT_2_PI = np.sqrt(2 / np.pi)
GGELU = 0.044715


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
    type_one_side
            Try to build N_types tables. Otherwise, building N_types^2 tables
    exclude_types : List[List[int]]
            The excluded pairs of types which have no interaction with each other.
            For example, `[[0, 1]]` means no interaction between type 0 and type 1.
    activation_function
            The activation function in the embedding net. Supported options are {"tanh","gelu"} in common.ActivationFn.
    """

    def __init__(
        self,
        descrpt,
        neuron: list[int],
        type_one_side: bool = False,
        exclude_types: list[list[int]] = [],
        activation_fn: ActivationFn = ActivationFn("tanh"),
    ) -> None:
        super().__init__(
            descrpt,
            neuron,
            type_one_side,
            exclude_types,
            True,
        )
        self.descrpt_type = self._get_descrpt_type()

        supported_descrpt_type = (
            "Atten",
            "A",
            "T",
            "R",
        )

        if self.descrpt_type in supported_descrpt_type:
            self.sel_a = self.descrpt.get_sel()
            self.rcut = self.descrpt.get_rcut()
            self.rcut_smth = self.descrpt.get_rcut_smth()
        else:
            raise RuntimeError("Unsupported descriptor")

        # functype
        activation_map = {
            "tanh": 1,
            "gelu": 2,
            "gelu_tf": 2,
            "relu": 3,
            "relu6": 4,
            "softplus": 5,
            "sigmoid": 6,
        }

        activation = activation_fn.activation
        if activation in activation_map:
            self.functype = activation_map[activation]
        else:
            raise RuntimeError("Unknown activation function type!")

        self.activation_fn = activation_fn
        serialized = self.descrpt.serialize()
        if isinstance(self.descrpt, deepmd.pt.model.descriptor.DescrptDPA2):
            serialized = serialized["repinit_variable"]
        self.davg = serialized["@variables"]["davg"]
        self.dstd = serialized["@variables"]["dstd"]
        self.embedding_net_nodes = serialized["embeddings"]["networks"]

        self.ntypes = self.descrpt.get_ntypes()

        self.layer_size = self._get_layer_size()
        self.table_size = self._get_table_size()

        self.bias = self._get_bias()
        self.matrix = self._get_matrix()

        self.data_type = self._get_data_type()
        self.last_layer_size = self._get_last_layer_size()

    def _make_data(self, xx, idx):
        """Generate tabulation data for the given input.

        Parameters
        ----------
        xx : np.ndarray
            Input values to tabulate
        idx : int
            Index for accessing the correct network parameters

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Values, first derivatives, and second derivatives
        """
        xx = torch.from_numpy(xx).view(-1, 1).to(env.DEVICE)
        for layer in range(self.layer_size):
            if layer == 0:
                xbar = torch.matmul(
                    xx,
                    torch.from_numpy(self.matrix["layer_" + str(layer + 1)][idx]).to(
                        env.DEVICE
                    ),
                ) + torch.from_numpy(self.bias["layer_" + str(layer + 1)][idx]).to(
                    env.DEVICE
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
                    yy,
                    torch.from_numpy(self.matrix["layer_" + str(layer + 1)][idx]).to(
                        env.DEVICE
                    ),
                ) + torch.from_numpy(self.bias["layer_" + str(layer + 1)][idx]).to(
                    env.DEVICE
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

        vv = zz.detach().cpu().numpy().astype(self.data_type)
        dd = dy.detach().cpu().numpy().astype(self.data_type)
        d2 = dy2.detach().cpu().numpy().astype(self.data_type)
        return vv, dd, d2

    def _layer_0(self, x, w, b):
        w = torch.from_numpy(w).to(env.DEVICE)
        b = torch.from_numpy(b).to(env.DEVICE)
        return self.activation_fn(torch.matmul(x, w) + b)

    def _layer_1(self, x, w, b):
        w = torch.from_numpy(w).to(env.DEVICE)
        b = torch.from_numpy(b).to(env.DEVICE)
        t = torch.cat([x, x], dim=1)
        return t, self.activation_fn(torch.matmul(x, w) + b) + t

    def _get_descrpt_type(self) -> str:
        if isinstance(
            self.descrpt,
            (
                deepmd.pt.model.descriptor.DescrptDPA1,
                deepmd.pt.model.descriptor.DescrptDPA2,
            ),
        ):
            return "Atten"
        elif isinstance(self.descrpt, deepmd.pt.model.descriptor.DescrptSeA):
            return "A"
        elif isinstance(self.descrpt, deepmd.pt.model.descriptor.DescrptSeR):
            return "R"
        elif isinstance(self.descrpt, deepmd.pt.model.descriptor.DescrptSeT):
            return "T"
        raise RuntimeError(f"Unsupported descriptor {self.descrpt}")

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
        if self.descrpt_type == "Atten":
            layer_size = len(self.embedding_net_nodes[0]["layers"])
        elif self.descrpt_type == "A":
            layer_size = len(self.embedding_net_nodes[0]["layers"])
            if self.type_one_side:
                layer_size = basic_size // (self.ntypes - self._n_all_excluded)
        elif self.descrpt_type == "T":
            layer_size = len(self.embedding_net_nodes[0]["layers"])
            # layer_size = basic_size // int(comb(self.ntypes + 1, 2))
        elif self.descrpt_type == "R":
            layer_size = basic_size // (
                self.ntypes * self.ntypes - len(self.exclude_types)
            )
            if self.type_one_side:
                layer_size = basic_size // (self.ntypes - self._n_all_excluded)
        else:
            raise RuntimeError("Unsupported descriptor")
        return layer_size

    def _get_network_variable(self, var_name: str) -> dict:
        """Get network variables (weights or biases) for all layers.

        Parameters
        ----------
        var_name : str
            Name of the variable to get ('w' for weights, 'b' for biases)

        Returns
        -------
        dict
            Dictionary mapping layer names to their variables
        """
        result = {}
        for layer in range(1, self.layer_size + 1):
            result["layer_" + str(layer)] = []
            if self.descrpt_type == "Atten":
                node = self.embedding_net_nodes[0]["layers"][layer - 1]["@variables"][
                    var_name
                ]
                result["layer_" + str(layer)].append(node)
            elif self.descrpt_type == "A":
                if self.type_one_side:
                    for ii in range(0, self.ntypes):
                        if not self._all_excluded(ii):
                            node = self.embedding_net_nodes[ii]["layers"][layer - 1][
                                "@variables"
                            ][var_name]
                            result["layer_" + str(layer)].append(node)
                        else:
                            result["layer_" + str(layer)].append(np.array([]))
                else:
                    for ii in range(0, self.ntypes * self.ntypes):
                        if (
                            ii // self.ntypes,
                            ii % self.ntypes,
                        ) not in self.exclude_types:
                            node = self.embedding_net_nodes[
                                (ii % self.ntypes) * self.ntypes + ii // self.ntypes
                            ]["layers"][layer - 1]["@variables"][var_name]
                            result["layer_" + str(layer)].append(node)
                        else:
                            result["layer_" + str(layer)].append(np.array([]))
            elif self.descrpt_type == "T":
                for ii in range(self.ntypes):
                    for jj in range(ii, self.ntypes):
                        node = self.embedding_net_nodes[jj * self.ntypes + ii][
                            "layers"
                        ][layer - 1]["@variables"][var_name]
                        result["layer_" + str(layer)].append(node)
            elif self.descrpt_type == "R":
                if self.type_one_side:
                    for ii in range(0, self.ntypes):
                        if not self._all_excluded(ii):
                            node = self.embedding_net_nodes[ii]["layers"][layer - 1][
                                "@variables"
                            ][var_name]
                            result["layer_" + str(layer)].append(node)
                        else:
                            result["layer_" + str(layer)].append(np.array([]))
                else:
                    for ii in range(0, self.ntypes * self.ntypes):
                        if (
                            ii // self.ntypes,
                            ii % self.ntypes,
                        ) not in self.exclude_types:
                            node = self.embedding_net_nodes[
                                (ii % self.ntypes) * self.ntypes + ii // self.ntypes
                            ]["layers"][layer - 1]["@variables"][var_name]
                            result["layer_" + str(layer)].append(node)
                        else:
                            result["layer_" + str(layer)].append(np.array([]))
            else:
                raise RuntimeError("Unsupported descriptor")
        return result

    def _get_bias(self):
        return self._get_network_variable("b")

    def _get_matrix(self):
        return self._get_network_variable("w")

    def _convert_numpy_to_tensor(self) -> None:
        """Convert self.data from np.ndarray to torch.Tensor."""
        for ii in self.data:
            self.data[ii] = torch.tensor(self.data[ii], device=env.DEVICE)  # pylint: disable=no-explicit-dtype

    @cached_property
    def _n_all_excluded(self) -> int:
        """Then number of types excluding all types."""
        return sum(int(self._all_excluded(ii)) for ii in range(0, self.ntypes))


# customized op
def grad(xbar: torch.Tensor, y: torch.Tensor, functype: int):
    if functype == 1:
        return 1 - y * y

    elif functype == 2:
        var = torch.tanh(SQRT_2_PI * (xbar + GGELU * xbar**3))
        return (
            0.5 * SQRT_2_PI * xbar * (1 - var**2) * (3 * GGELU * xbar**2 + 1)
            + 0.5 * var
            + 0.5
        )

    elif functype == 3:
        return torch.where(xbar > 0, torch.ones_like(xbar), torch.zeros_like(xbar))

    elif functype == 4:
        return torch.where(
            (xbar > 0) & (xbar < 6), torch.ones_like(xbar), torch.zeros_like(xbar)
        )

    elif functype == 5:
        return 1.0 - 1.0 / (1.0 + torch.exp(xbar))

    elif functype == 6:
        return y * (1 - y)

    else:
        raise ValueError(f"Unsupported function type: {functype}")


def grad_grad(xbar: torch.Tensor, y: torch.Tensor, functype: int):
    if functype == 1:
        return -2 * y * (1 - y * y)

    elif functype == 2:
        var1 = torch.tanh(SQRT_2_PI * (xbar + GGELU * xbar**3))
        var2 = SQRT_2_PI * (1 - var1**2) * (3 * GGELU * xbar**2 + 1)
        return (
            3 * GGELU * SQRT_2_PI * xbar**2 * (1 - var1**2)
            - SQRT_2_PI * xbar * var2 * (3 * GGELU * xbar**2 + 1) * var1
            + var2
        )

    elif functype in [3, 4]:
        return torch.zeros_like(xbar)

    elif functype == 5:
        exp_xbar = torch.exp(xbar)
        return exp_xbar / ((1 + exp_xbar) * (1 + exp_xbar))

    elif functype == 6:
        return y * (1 - y) * (1 - 2 * y)

    else:
        return -torch.ones_like(xbar)


def unaggregated_dy_dx_s(
    y: torch.Tensor, w_np: np.ndarray, xbar: torch.Tensor, functype: int
):
    w = torch.from_numpy(w_np).to(env.DEVICE)
    y = y.to(env.DEVICE)
    xbar = xbar.to(env.DEVICE)
    if y.dim() != 2:
        raise ValueError("Dim of input y should be 2")
    if w.dim() != 2:
        raise ValueError("Dim of input w should be 2")
    if xbar.dim() != 2:
        raise ValueError("Dim of input xbar should be 2")

    grad_xbar_y = grad(xbar, y, functype)

    w = torch.flatten(w)[: y.shape[1]].repeat(y.shape[0], 1)

    dy_dx = grad_xbar_y * w

    return dy_dx


def unaggregated_dy2_dx_s(
    y: torch.Tensor,
    dy: torch.Tensor,
    w_np: np.ndarray,
    xbar: torch.Tensor,
    functype: int,
):
    w = torch.from_numpy(w_np).to(env.DEVICE)
    y = y.to(env.DEVICE)
    dy = dy.to(env.DEVICE)
    xbar = xbar.to(env.DEVICE)
    if y.dim() != 2:
        raise ValueError("Dim of input y should be 2")
    if dy.dim() != 2:
        raise ValueError("Dim of input dy should be 2")
    if w.dim() != 2:
        raise ValueError("Dim of input w should be 2")
    if xbar.dim() != 2:
        raise ValueError("Dim of input xbar should be 2")

    grad_grad_result = grad_grad(xbar, y, functype)

    w_flattened = torch.flatten(w)[: y.shape[1]].repeat(y.shape[0], 1)

    dy2_dx = grad_grad_result * w_flattened * w_flattened

    return dy2_dx


def unaggregated_dy_dx(
    z: torch.Tensor,
    w_np: np.ndarray,
    dy_dx: torch.Tensor,
    ybar: torch.Tensor,
    functype: int,
):
    w = torch.from_numpy(w_np).to(env.DEVICE)
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

    grad_ybar_z = grad(ybar, z, functype)

    dy_dx = dy_dx.view(-1)[: (length * size)].view(length, size)

    accumulator = dy_dx @ w

    dz_drou = grad_ybar_z * accumulator

    if width == size:
        dz_drou += dy_dx
    if width == 2 * size:
        dy_dx = torch.cat((dy_dx, dy_dx), dim=1)
        dz_drou += dy_dx

    return dz_drou


def unaggregated_dy2_dx(
    z: torch.Tensor,
    w_np: np.ndarray,
    dy_dx: torch.Tensor,
    dy2_dx: torch.Tensor,
    ybar: torch.Tensor,
    functype: int,
):
    w = torch.from_numpy(w_np).to(env.DEVICE)
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

    grad_ybar_z = grad(ybar, z, functype)
    grad_grad_ybar_z = grad_grad(ybar, z, functype)

    dy2_dx = dy2_dx.view(-1)[: (length * size)].view(length, size)
    dy_dx = dy_dx.view(-1)[: (length * size)].view(length, size)

    accumulator1 = dy2_dx @ w
    accumulator2 = dy_dx @ w

    dz_drou = (
        grad_ybar_z * accumulator1 + grad_grad_ybar_z * accumulator2 * accumulator2
    )

    if width == size:
        dz_drou += dy2_dx
    if width == 2 * size:
        dy2_dx = torch.cat((dy2_dx, dy2_dx), dim=1)
        dz_drou += dy2_dx

    return dz_drou
