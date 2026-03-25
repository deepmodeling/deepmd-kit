# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend-agnostic tabulation math using numpy.

Provides the pure-math functions for model compression tabulation:
activation derivatives, chain-rule derivative propagation, and
embedding-net forward pass. Used by both pt and pt_expt backends.
"""

import logging
from functools import (
    cached_property,
)
from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel.utils.network import (
    get_activation_fn,
)
from deepmd.utils.tabulate import (
    BaseTabulate,
)

log = logging.getLogger(__name__)

SQRT_2_PI = np.sqrt(2 / np.pi)
GGELU = 0.044715

# Mapping from activation function name to integer functype
# used by grad/grad_grad for derivative computation.
ACTIVATION_TO_FUNCTYPE: dict[str, int] = {
    "tanh": 1,
    "gelu": 2,
    "gelu_tf": 2,
    "relu": 3,
    "relu6": 4,
    "softplus": 5,
    "sigmoid": 6,
    "silu": 7,
}


# ---- Activation derivatives (numpy) ----


def grad(xbar: np.ndarray, y: np.ndarray, functype: int) -> np.ndarray:
    """First derivative of the activation function."""
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
        return np.where(xbar > 0, np.ones_like(xbar), np.zeros_like(xbar))
    elif functype == 4:
        return np.where(
            (xbar > 0) & (xbar < 6), np.ones_like(xbar), np.zeros_like(xbar)
        )
    elif functype == 5:
        return 1.0 - 1.0 / (1.0 + np.exp(xbar))
    elif functype == 6:
        return y * (1 - y)
    elif functype == 7:
        sig = 1.0 / (1.0 + np.exp(-xbar))
        return sig + xbar * sig * (1 - sig)
    else:
        raise ValueError(f"Unsupported function type: {functype}")


def grad_grad(xbar: np.ndarray, y: np.ndarray, functype: int) -> np.ndarray:
    """Second derivative of the activation function."""
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
    elif functype in [3, 4]:
        return np.zeros_like(xbar)
    elif functype == 5:
        exp_xbar = np.exp(xbar)
        return exp_xbar / ((1 + exp_xbar) * (1 + exp_xbar))
    elif functype == 6:
        return y * (1 - y) * (1 - 2 * y)
    elif functype == 7:
        sig = 1.0 / (1.0 + np.exp(-xbar))
        d_sig = sig * (1 - sig)
        return 2 * d_sig + xbar * d_sig * (1 - 2 * sig)
    else:
        return -np.ones_like(xbar)


# ---- Chain-rule derivative propagation (numpy) ----


def unaggregated_dy_dx_s(
    y: np.ndarray, w: np.ndarray, xbar: np.ndarray, functype: int
) -> np.ndarray:
    """First derivative for the first layer (scalar input)."""
    if y.ndim != 2:
        raise ValueError("Dim of input y should be 2")
    if w.ndim != 2:
        raise ValueError("Dim of input w should be 2")
    if xbar.ndim != 2:
        raise ValueError("Dim of input xbar should be 2")

    grad_xbar_y = grad(xbar, y, functype)
    w_flat = np.ravel(w)[: y.shape[1]]
    w_rep = np.tile(w_flat, (y.shape[0], 1))
    return grad_xbar_y * w_rep


def unaggregated_dy2_dx_s(
    y: np.ndarray,
    dy: np.ndarray,
    w: np.ndarray,
    xbar: np.ndarray,
    functype: int,
) -> np.ndarray:
    """Second derivative for the first layer (scalar input)."""
    if y.ndim != 2:
        raise ValueError("Dim of input y should be 2")
    if dy.ndim != 2:
        raise ValueError("Dim of input dy should be 2")
    if w.ndim != 2:
        raise ValueError("Dim of input w should be 2")
    if xbar.ndim != 2:
        raise ValueError("Dim of input xbar should be 2")

    gg = grad_grad(xbar, y, functype)
    w_flat = np.ravel(w)[: y.shape[1]]
    w_rep = np.tile(w_flat, (y.shape[0], 1))
    return gg * w_rep * w_rep


def unaggregated_dy_dx(
    z: np.ndarray,
    w: np.ndarray,
    dy_dx: np.ndarray,
    ybar: np.ndarray,
    functype: int,
) -> np.ndarray:
    """First derivative for subsequent layers."""
    if z.ndim != 2:
        raise ValueError("z must have 2 dimensions")
    if w.ndim != 2:
        raise ValueError("w must have 2 dimensions")
    if dy_dx.ndim != 2:
        raise ValueError("dy_dx must have 2 dimensions")
    if ybar.ndim != 2:
        raise ValueError("ybar must have 2 dimensions")

    length, width = z.shape
    size = w.shape[0]

    grad_ybar_z = grad(ybar, z, functype)
    dy_dx = np.ravel(dy_dx)[: length * size].reshape(length, size)
    accumulator = dy_dx @ w
    dz_drou = grad_ybar_z * accumulator

    if width == size:
        dz_drou += dy_dx
    if width == 2 * size:
        dy_dx = np.concatenate((dy_dx, dy_dx), axis=1)
        dz_drou += dy_dx

    return dz_drou


def unaggregated_dy2_dx(
    z: np.ndarray,
    w: np.ndarray,
    dy_dx: np.ndarray,
    dy2_dx: np.ndarray,
    ybar: np.ndarray,
    functype: int,
) -> np.ndarray:
    """Second derivative for subsequent layers."""
    if z.ndim != 2:
        raise ValueError("z must have 2 dimensions")
    if w.ndim != 2:
        raise ValueError("w must have 2 dimensions")
    if dy_dx.ndim != 2:
        raise ValueError("dy_dx must have 2 dimensions")
    if dy2_dx.ndim != 2:
        raise ValueError("dy2_dx must have 2 dimensions")
    if ybar.ndim != 2:
        raise ValueError("ybar must have 2 dimensions")

    length, width = z.shape
    size = w.shape[0]

    grad_ybar_z = grad(ybar, z, functype)
    gg = grad_grad(ybar, z, functype)

    dy2_dx = np.ravel(dy2_dx)[: length * size].reshape(length, size)
    dy_dx = np.ravel(dy_dx)[: length * size].reshape(length, size)

    acc1 = dy2_dx @ w
    acc2 = dy_dx @ w

    dz_drou = grad_ybar_z * acc1 + gg * acc2 * acc2

    if width == size:
        dz_drou += dy2_dx
    if width == 2 * size:
        dy2_dx = np.concatenate((dy2_dx, dy2_dx), axis=1)
        dz_drou += dy2_dx

    return dz_drou


# ---- DPTabulate with numpy math ----


class DPTabulate(BaseTabulate):
    r"""Backend-agnostic tabulation using numpy.

    Compress a model by tabulating the embedding-net. The table is composed
    of fifth-order polynomial coefficients assembled from two sub-tables.

    Parameters
    ----------
    descrpt
        Descriptor of the original model.
    neuron
        Number of neurons in each hidden layer of the embedding net.
    type_one_side
        Try to build N_types tables. Otherwise, building N_types^2 tables.
    exclude_types
        Excluded type pairs with no interaction.
    activation_fn_name
        Name of the activation function (e.g. "tanh", "gelu", "relu").
    """

    def __init__(
        self,
        descrpt: Any,
        neuron: list[int],
        type_one_side: bool = False,
        exclude_types: list[list[int]] = [],
        activation_fn_name: str = "tanh",
    ) -> None:
        super().__init__(
            descrpt,
            neuron,
            type_one_side,
            exclude_types,
            True,  # is_pt flag (for _build_lower numpy int conversion)
        )
        self._activation_fn = get_activation_fn(activation_fn_name)
        activation_fn_name = activation_fn_name.lower()
        if activation_fn_name not in ACTIVATION_TO_FUNCTYPE:
            raise RuntimeError(f"Unknown activation function: {activation_fn_name}")
        self.functype = ACTIVATION_TO_FUNCTYPE[activation_fn_name]

        self.descrpt_type = self._get_descrpt_type()

        supported_descrpt_type = ("Atten", "A", "T", "T_TEBD", "R")
        if self.descrpt_type in supported_descrpt_type:
            self.sel_a = self.descrpt.get_sel()
            self.rcut = self.descrpt.get_rcut()
            self.rcut_smth = self.descrpt.get_rcut_smth()
        else:
            raise RuntimeError("Unsupported descriptor")

        serialized = self.descrpt.serialize()
        # For DPA2, use the repinit sub-block's serialized data
        if self.descrpt_type == "Atten" and "repinit_variable" in serialized:
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

    def _make_data(self, xx: np.ndarray, idx: int) -> Any:
        """Forward pass through embedding net with derivative computation."""
        xx = xx.reshape(-1, 1)
        for layer in range(self.layer_size):
            if layer == 0:
                xbar = (
                    np.matmul(xx, self.matrix["layer_" + str(layer + 1)][idx])
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
                    dy = unaggregated_dy_dx_s(
                        yy - xx,
                        self.matrix["layer_" + str(layer + 1)][idx],
                        xbar,
                        self.functype,
                    ) + np.ones((1, 1), dtype=yy.dtype)
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
                    ) + np.ones((1, 2), dtype=yy.dtype)
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
                ybar = (
                    np.matmul(yy, self.matrix["layer_" + str(layer + 1)][idx])
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

        vv = zz.astype(self.data_type)
        dd = dy.astype(self.data_type)
        d2 = dy2.astype(self.data_type)
        return vv, dd, d2

    def _layer_0(self, x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self._activation_fn(np.matmul(x, w) + b)

    def _layer_1(
        self, x: np.ndarray, w: np.ndarray, b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        t = np.concatenate([x, x], axis=1)
        return t, self._activation_fn(np.matmul(x, w) + b) + t

    def _get_descrpt_type(self) -> str:
        """Determine descriptor type from serialized data."""
        data = self.descrpt.serialize()
        type_str = data.get("type", "")
        type_map = {
            "se_e2_a": "A",
            "se_r": "R",
            "se_e3": "T",
            "se_e3_tebd": "T_TEBD",
            "dpa1": "Atten",
            "se_atten_v2": "Atten",
        }
        descrpt_type = type_map.get(type_str)
        if descrpt_type is None:
            raise RuntimeError(f"Unsupported descriptor type: {type_str}")
        return descrpt_type

    def _get_layer_size(self) -> int:
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
        if self.descrpt_type in ("Atten", "T_TEBD"):
            layer_size = len(self.embedding_net_nodes[0]["layers"])
        elif self.descrpt_type == "A":
            layer_size = len(self.embedding_net_nodes[0]["layers"])
            if self.type_one_side:
                layer_size = basic_size // (self.ntypes - self._n_all_excluded)
        elif self.descrpt_type == "T":
            layer_size = len(self.embedding_net_nodes[0]["layers"])
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
        """Get network variables (weights or biases) for all layers."""
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
            elif self.descrpt_type == "T_TEBD":
                node = self.embedding_net_nodes[0]["layers"][layer - 1]["@variables"][
                    var_name
                ]
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

    def _get_bias(self) -> Any:
        return self._get_network_variable("b")

    def _get_matrix(self) -> Any:
        return self._get_network_variable("w")

    def _convert_numpy_to_tensor(self) -> None:
        """No-op: data stays as numpy arrays."""
        pass

    @cached_property
    def _n_all_excluded(self) -> int:
        """The number of types excluding all types."""
        return sum(int(self._all_excluded(ii)) for ii in range(0, self.ntypes))
