# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend-agnostic tabulation math using the Array API where possible.

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

import array_api_compat
import numpy as np

from deepmd.dpmodel.common import (
    to_numpy_array,
)
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
    "linear": 0,
    "none": 0,
}


# ---- Activation derivatives (Array API compatible) ----


def _stable_sigmoid(xbar: Any) -> Any:
    """Compute sigmoid without overflow for large-magnitude inputs."""
    xp = array_api_compat.array_namespace(xbar)
    positive = xbar >= 0
    exp_neg_abs = xp.exp(xp.where(positive, -xbar, xbar))
    return xp.where(
        positive,
        1.0 / (1.0 + exp_neg_abs),
        exp_neg_abs / (1.0 + exp_neg_abs),
    )


def _repeat_flattened_weight_prefix(w: Any, rows: int, cols: int) -> Any:
    """Repeat the flattened weight prefix row-wise in an Array API way."""
    xp = array_api_compat.array_namespace(w)
    w_flat = xp.reshape(w, (-1,))[:cols]
    w_flat = xp.reshape(w_flat, (1, cols))
    return xp.broadcast_to(w_flat, (rows, cols))


def grad(xbar: Any, y: Any, functype: int) -> Any:
    """First derivative of the activation function."""
    xp = array_api_compat.array_namespace(xbar, y)
    if functype == 0:
        return xp.ones_like(xbar)
    elif functype == 1:
        return 1 - y * y
    elif functype == 2:
        var = xp.tanh(SQRT_2_PI * (xbar + GGELU * xbar**3))
        return (
            0.5 * SQRT_2_PI * xbar * (1 - var**2) * (3 * GGELU * xbar**2 + 1)
            + 0.5 * var
            + 0.5
        )
    elif functype == 3:
        return xp.astype(xbar > 0, xbar.dtype)
    elif functype == 4:
        return xp.astype((xbar > 0) & (xbar < 6), xbar.dtype)
    elif functype == 5:
        return _stable_sigmoid(xbar)
    elif functype == 6:
        return y * (1 - y)
    elif functype == 7:
        sig = _stable_sigmoid(xbar)
        return sig + xbar * sig * (1 - sig)
    else:
        raise ValueError(f"Unsupported function type: {functype}")


def grad_grad(xbar: Any, y: Any, functype: int) -> Any:
    """Second derivative of the activation function."""
    xp = array_api_compat.array_namespace(xbar, y)
    if functype == 0:
        return xp.zeros_like(xbar)
    elif functype == 1:
        return -2 * y * (1 - y * y)
    elif functype == 2:
        var1 = xp.tanh(SQRT_2_PI * (xbar + GGELU * xbar**3))
        var2 = SQRT_2_PI * (1 - var1**2) * (3 * GGELU * xbar**2 + 1)
        return (
            3 * GGELU * SQRT_2_PI * xbar**2 * (1 - var1**2)
            - SQRT_2_PI * xbar * var2 * (3 * GGELU * xbar**2 + 1) * var1
            + var2
        )
    elif functype in [3, 4]:
        return xp.zeros_like(xbar)
    elif functype == 5:
        sig = _stable_sigmoid(xbar)
        return sig * (1 - sig)
    elif functype == 6:
        return y * (1 - y) * (1 - 2 * y)
    elif functype == 7:
        sig = _stable_sigmoid(xbar)
        d_sig = sig * (1 - sig)
        return 2 * d_sig + xbar * d_sig * (1 - 2 * sig)
    else:
        raise ValueError(f"Unsupported function type: {functype}")


# ---- Chain-rule derivative propagation (Array API compatible) ----


def unaggregated_dy_dx_s(y: Any, w: Any, xbar: Any, functype: int) -> Any:
    """First derivative for the first layer (scalar input)."""
    if y.ndim != 2:
        raise ValueError("Dim of input y should be 2")
    if w.ndim != 2:
        raise ValueError("Dim of input w should be 2")
    if xbar.ndim != 2:
        raise ValueError("Dim of input xbar should be 2")

    grad_xbar_y = grad(xbar, y, functype)
    w_rep = _repeat_flattened_weight_prefix(w, y.shape[0], y.shape[1])
    return grad_xbar_y * w_rep


def unaggregated_dy2_dx_s(
    y: Any,
    dy: Any,
    w: Any,
    xbar: Any,
    functype: int,
) -> Any:
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
    w_rep = _repeat_flattened_weight_prefix(w, y.shape[0], y.shape[1])
    return gg * w_rep * w_rep


def unaggregated_dy_dx(
    z: Any,
    w: Any,
    dy_dx: Any,
    ybar: Any,
    functype: int,
) -> Any:
    """First derivative for subsequent layers."""
    xp = array_api_compat.array_namespace(z, w, dy_dx, ybar)
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
    dy_dx = xp.reshape(dy_dx, (-1,))[: length * size]
    dy_dx = xp.reshape(dy_dx, (length, size))
    accumulator = xp.matmul(dy_dx, w)
    dz_drou = grad_ybar_z * accumulator

    if width == size:
        dz_drou += dy_dx
    if width == 2 * size:
        dy_dx = xp.concat((dy_dx, dy_dx), axis=1)
        dz_drou += dy_dx

    return dz_drou


def unaggregated_dy2_dx(
    z: Any,
    w: Any,
    dy_dx: Any,
    dy2_dx: Any,
    ybar: Any,
    functype: int,
) -> Any:
    """Second derivative for subsequent layers."""
    xp = array_api_compat.array_namespace(z, w, dy_dx, dy2_dx, ybar)
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

    dy2_dx = xp.reshape(dy2_dx, (-1,))[: length * size]
    dy2_dx = xp.reshape(dy2_dx, (length, size))
    dy_dx = xp.reshape(dy_dx, (-1,))[: length * size]
    dy_dx = xp.reshape(dy_dx, (length, size))

    acc1 = xp.matmul(dy2_dx, w)
    acc2 = xp.matmul(dy_dx, w)

    dz_drou = grad_ybar_z * acc1 + gg * acc2 * acc2

    if width == size:
        dz_drou += dy2_dx
    if width == 2 * size:
        dy2_dx = xp.concat((dy2_dx, dy2_dx), axis=1)
        dz_drou += dy2_dx

    return dz_drou


# ---- DPTabulate with Array API math ----


class DPTabulate(BaseTabulate):
    r"""Backend-agnostic tabulation using Array API compatible math.

    Compress a model by tabulating the embedding-net. The table is composed
    of fifth-order polynomial coefficients fitted to the embedding-net output
    and its derivatives over intervals of the environment matrix.
    """

    def __init__(
        self,
        descrpt: Any,
        neuron: list[int],
        type_one_side: bool,
        exclude_types: list[list[int]] = [],
        activation_fn: str = "tanh",
        suffix: str = "",
    ) -> None:
        super().__init__(descrpt, neuron, type_one_side, exclude_types)

        self.descrpt_type = self._get_descrpt_type()
        self.neuron = neuron
        self.type_one_side = type_one_side
        self.exclude_types = [tuple(et) for et in exclude_types]
        self.suffix = suffix
        self.activation_fn = activation_fn
        self.functype = ACTIVATION_TO_FUNCTYPE.get(activation_fn, -1)
        if self.functype == -1:
            raise ValueError(f"Unsupported activation function: {activation_fn}")
        self._activation_fn = get_activation_fn(activation_fn)

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

    def _get_math_backend_sample(self) -> Any:
        """Return a sample array choosing the execution backend for math ops."""
        return np.empty((), dtype=self.data_type)

    @cached_property
    def _math_backend_sample(self) -> Any:
        return self._get_math_backend_sample()

    @cached_property
    def _math_backend_device(self) -> Any:
        return array_api_compat.device(self._math_backend_sample)

    def _backend_asarray(self, value: Any) -> Any:
        xp = array_api_compat.array_namespace(self._math_backend_sample)
        return xp.asarray(value, device=self._math_backend_device)

    @cached_property
    def _matrix_backend(self) -> dict[str, list[Any]]:
        matrix = {
            layer: [self._backend_asarray(value) for value in values]
            for layer, values in self.matrix.items()
        }
        self.matrix = None
        return matrix

    @cached_property
    def _bias_backend(self) -> dict[str, list[Any]]:
        bias = {
            layer: [self._backend_asarray(value) for value in values]
            for layer, values in self.bias.items()
        }
        self.bias = None
        return bias

    def _make_data(self, xx: np.ndarray, idx: int) -> Any:
        """Forward pass through embedding net with derivative computation."""
        xp = array_api_compat.array_namespace(self._math_backend_sample)
        xx = xp.reshape(self._backend_asarray(xx), (-1, 1))
        for layer in range(self.layer_size):
            matrix = self._matrix_backend["layer_" + str(layer + 1)][idx]
            bias = self._bias_backend["layer_" + str(layer + 1)][idx]
            if layer == 0:
                xbar = xp.matmul(xx, matrix) + bias
                if self.neuron[0] == 1:
                    yy = self._layer_0(xx, matrix, bias) + xx
                    dy = unaggregated_dy_dx_s(
                        yy - xx,
                        matrix,
                        xbar,
                        self.functype,
                    ) + xp.ones(
                        (1, 1), dtype=yy.dtype, device=array_api_compat.device(yy)
                    )
                    dy2 = unaggregated_dy2_dx_s(
                        yy - xx,
                        dy,
                        matrix,
                        xbar,
                        self.functype,
                    )
                elif self.neuron[0] == 2:
                    tt, yy = self._layer_1(xx, matrix, bias)
                    dy = unaggregated_dy_dx_s(
                        yy - tt,
                        matrix,
                        xbar,
                        self.functype,
                    ) + xp.ones(
                        (1, 2), dtype=yy.dtype, device=array_api_compat.device(yy)
                    )
                    dy2 = unaggregated_dy2_dx_s(
                        yy - tt,
                        dy,
                        matrix,
                        xbar,
                        self.functype,
                    )
                else:
                    yy = self._layer_0(xx, matrix, bias)
                    dy = unaggregated_dy_dx_s(
                        yy,
                        matrix,
                        xbar,
                        self.functype,
                    )
                    dy2 = unaggregated_dy2_dx_s(
                        yy,
                        dy,
                        matrix,
                        xbar,
                        self.functype,
                    )
            else:
                ybar = xp.matmul(yy, matrix) + bias
                if self.neuron[layer] == self.neuron[layer - 1]:
                    zz = self._layer_0(yy, matrix, bias) + yy
                    dz = unaggregated_dy_dx(
                        zz - yy,
                        matrix,
                        dy,
                        ybar,
                        self.functype,
                    )
                    dy2 = unaggregated_dy2_dx(
                        zz - yy,
                        matrix,
                        dy,
                        dy2,
                        ybar,
                        self.functype,
                    )
                elif self.neuron[layer] == 2 * self.neuron[layer - 1]:
                    tt, zz = self._layer_1(yy, matrix, bias)
                    dz = unaggregated_dy_dx(
                        zz - tt,
                        matrix,
                        dy,
                        ybar,
                        self.functype,
                    )
                    dy2 = unaggregated_dy2_dx(
                        zz - tt,
                        matrix,
                        dy,
                        dy2,
                        ybar,
                        self.functype,
                    )
                else:
                    zz = self._layer_0(yy, matrix, bias)
                    dz = unaggregated_dy_dx(
                        zz,
                        matrix,
                        dy,
                        ybar,
                        self.functype,
                    )
                    dy2 = unaggregated_dy2_dx(
                        zz,
                        matrix,
                        dy,
                        dy2,
                        ybar,
                        self.functype,
                    )
                dy = dz
                yy = zz

        vv = to_numpy_array(yy).astype(self.data_type)
        dd = to_numpy_array(dy).astype(self.data_type)
        d2 = to_numpy_array(dy2).astype(self.data_type)
        return vv, dd, d2

    def _layer_0(self, x: Any, w: Any, b: Any) -> Any:
        xp = array_api_compat.array_namespace(x, w, b)
        return self._activation_fn(xp.matmul(x, w) + b)

    def _layer_1(self, x: Any, w: Any, b: Any) -> tuple[Any, Any]:
        xp = array_api_compat.array_namespace(x, w, b)
        t = xp.concat([x, x], axis=1)
        return t, self._activation_fn(xp.matmul(x, w) + b) + t

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
                for ii in range(0, len(self.embedding_net_nodes)):
                    node = self.embedding_net_nodes[ii]["layers"][layer - 1][
                        "@variables"
                    ][var_name]
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
                            node = self.embedding_net_nodes[ii]["layers"][layer - 1][
                                "@variables"
                            ][var_name]
                            result["layer_" + str(layer)].append(node)
                        else:
                            result["layer_" + str(layer)].append(np.array([]))
            else:
                raise RuntimeError("Unsupported descriptor")
        return result

    def _get_matrix(self) -> dict:
        return self._get_network_variable("w")

    def _get_bias(self) -> dict:
        return self._get_network_variable("b")
