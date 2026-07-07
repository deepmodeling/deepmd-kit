# SPDX-License-Identifier: LGPL-3.0-or-later
"""
SO(3)-equivariant linear layers for DPA4/SeZM.

This module defines the channel-only and focus-aware linear maps used by SeZM
SO(3) feature transformations.

This module is the dpmodel (array-API) port of
``deepmd.pt.model.descriptor.sezm_nn.so3``.
"""

from __future__ import (
    annotations,
)

import math
from typing import (
    Any,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
    NativeOP,
)
from deepmd.dpmodel.array_api import (
    xp_asarray_nodetach,
)
from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .indexing import (
    get_so3_dim_of_lmax,
    map_degree_idx,
)
from .utils import (
    init_trunc_normal_fan_in_out,
)


class FocusLinear(NativeOP):
    """
    Per-focus linear projection on the last feature axis.

    Notes
    -----
    Parameters are stored in (in, out) convention to match Muon's rectangular
    correction assumption (rows=fan_in, cols=fan_out):
    - weight: (in_channels, n_focus * out_channels)
    - bias: (n_focus * out_channels,)

    Parameters
    ----------
    in_channels
        Input feature dimension.
    out_channels
        Output feature dimension.
    n_focus
        Number of focus streams.
    precision
        Parameter precision.
    bias
        Whether to use bias.
    trainable
        Whether parameters are trainable.
    seed
        Random seed for initialization.
    init_std
        If given, use normal(0, init_std) instead of default uniform init.
        Useful for gate projections where small initial logits are desired.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        n_focus: int,
        precision: str = DEFAULT_PRECISION,
        bias: bool = True,
        trainable: bool = True,
        seed: int | list[int] | None = None,
        init_std: float | None = None,
    ) -> None:
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_focus = int(n_focus)
        self.precision = precision
        self.trainable = bool(trainable)
        self.use_bias = bool(bias)
        prec = PRECISION_DICT[self.precision.lower()]
        rng = np.random.default_rng(seed)
        shape = (self.in_channels, self.n_focus * self.out_channels)
        if init_std is not None:
            weight = rng.normal(0.0, float(init_std), size=shape)
        else:
            bound = 1.0 / math.sqrt(self.in_channels)
            weight = rng.uniform(-bound, bound, size=shape)
        self.weight = weight.astype(prec)
        if self.use_bias:
            self.bias: np.ndarray | None = np.zeros(
                (self.n_focus * self.out_channels,), dtype=prec
            )
        else:
            self.bias = None

    def call(self, x: Any) -> Any:
        """
        Parameters
        ----------
        x
            Input array with shape (B, F, Cin).

        Returns
        -------
        Array
            Projected array with shape (B, F, Cout).
        """
        xp = array_api_compat.array_namespace(x)
        weight = xp_asarray_nodetach(
            xp, self.weight[...], device=array_api_compat.device(x)
        )
        weight = xp.reshape(weight, (self.in_channels, self.n_focus, self.out_channels))
        # einsum "bfi,ifo->bfo" as a broadcast batched matmul:
        # (B, F, 1, Cin) @ (1, F, Cin, Cout) -> (B, F, 1, Cout)
        weight = xp.permute_dims(weight, (1, 0, 2))  # (F, Cin, Cout)
        out = xp.matmul(x[:, :, None, :], weight[None, ...])[..., 0, :]
        if self.use_bias:
            bias = xp_asarray_nodetach(
                xp, self.bias[...], device=array_api_compat.device(x)
            )
            bias = xp.reshape(bias, (self.n_focus, self.out_channels))
            out = out + bias[None, ...]
        return out

    def serialize(self) -> dict[str, Any]:
        """Serialize the FocusLinear to a dict."""
        variables = {"weight": to_numpy_array(self.weight)}
        if self.use_bias:
            variables["bias"] = to_numpy_array(self.bias)
        return {
            "@class": "FocusLinear",
            "@version": 1,
            "config": {
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "n_focus": self.n_focus,
                "precision": np.dtype(PRECISION_DICT[self.precision]).name,
                "bias": self.use_bias,
                "trainable": self.trainable,
                "seed": None,
            },
            "@variables": variables,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> FocusLinear:
        """Deserialize a FocusLinear from a dict."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "FocusLinear":
            raise ValueError(f"Invalid class for FocusLinear: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(
            in_channels=int(config["in_channels"]),
            out_channels=int(config["out_channels"]),
            n_focus=int(config["n_focus"]),
            precision=str(config["precision"]),
            bias=bool(config["bias"]),
            trainable=bool(config["trainable"]),
            seed=config.get("seed"),
        )
        prec = PRECISION_DICT[obj.precision.lower()]
        obj.weight = np.asarray(variables["weight"], dtype=prec)
        if obj.use_bias:
            obj.bias = np.asarray(variables["bias"], dtype=prec)
        return obj


class ChannelLinear(NativeOP):
    """
    Channel-only linear projection on the last feature axis.

    Notes
    -----
    Parameters are stored in (in, out) convention to match Muon's rectangular
    correction assumption (rows=fan_in, cols=fan_out):
    - weight: (in_channels, out_channels)
    - bias: (out_channels,)

    Parameters
    ----------
    in_channels
        Input feature dimension.
    out_channels
        Output feature dimension.
    precision
        Parameter precision.
    bias
        Whether to use bias.
    trainable
        Whether parameters are trainable.
    seed
        Random seed for initialization.
    init_std
        If given, use normal(0, init_std) instead of default uniform init.
        Useful for gate projections where small initial logits are desired.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        precision: str = DEFAULT_PRECISION,
        bias: bool = True,
        trainable: bool = True,
        seed: int | list[int] | None = None,
        init_std: float | None = None,
    ) -> None:
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.precision = precision
        self.trainable = bool(trainable)
        self.use_bias = bool(bias)
        prec = PRECISION_DICT[self.precision.lower()]
        rng = np.random.default_rng(seed)
        shape = (self.in_channels, self.out_channels)
        if init_std is not None:
            weight = rng.normal(0.0, float(init_std), size=shape)
        else:
            bound = 1.0 / math.sqrt(self.in_channels)
            weight = rng.uniform(-bound, bound, size=shape)
        self.weight = weight.astype(prec)
        if self.use_bias:
            self.bias: np.ndarray | None = np.zeros((self.out_channels,), dtype=prec)
        else:
            self.bias = None

    def call(self, x: Any) -> Any:
        """
        Parameters
        ----------
        x
            Input array with shape ``(..., C_in)``.

        Returns
        -------
        Array
            Projected array with shape ``(..., C_out)``.
        """
        xp = array_api_compat.array_namespace(x)
        # einsum "...i,io->...o" is a plain matmul on the last axis
        device = array_api_compat.device(x)
        out = xp.matmul(x, xp_asarray_nodetach(xp, self.weight[...], device=device))
        if self.use_bias:
            out = out + xp_asarray_nodetach(xp, self.bias[...], device=device)
        return out

    def serialize(self) -> dict[str, Any]:
        """Serialize the ChannelLinear to a dict."""
        variables = {"weight": to_numpy_array(self.weight)}
        if self.use_bias:
            variables["bias"] = to_numpy_array(self.bias)
        return {
            "@class": "ChannelLinear",
            "@version": 1,
            "config": {
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "precision": np.dtype(PRECISION_DICT[self.precision]).name,
                "bias": self.use_bias,
                "trainable": self.trainable,
                "seed": None,
            },
            "@variables": variables,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> ChannelLinear:
        """Deserialize a ChannelLinear from a dict."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "ChannelLinear":
            raise ValueError(f"Invalid class for ChannelLinear: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(
            in_channels=int(config["in_channels"]),
            out_channels=int(config["out_channels"]),
            precision=str(config["precision"]),
            bias=bool(config["bias"]),
            trainable=bool(config["trainable"]),
            seed=config.get("seed"),
        )
        prec = PRECISION_DICT[obj.precision.lower()]
        obj.weight = np.asarray(variables["weight"], dtype=prec)
        if obj.use_bias:
            obj.bias = np.asarray(variables["bias"], dtype=prec)
        return obj


class SO3Linear(NativeOP):
    """
    Focus-aware degree-wise linear self-interaction.

    This vectorized implementation avoids Python loops by using ``torch.einsum``
    and ``index_select``. The key insight is that weights are shared across all
    ``m`` components within each ``l`` block.

    Notes
    -----
    - Weight storage: ``(lmax+1, C_in, F*C_out)``.
    - Bias storage: ``(F*C_out,)``, only applied to ``l=0`` scalar components.
    - Runtime view restores weights to ``(lmax+1, C_in, F, C_out)`` via reshape.
    - ``expand_index`` maps each packed ``(l,m)`` position to its ``l`` value.
    - Einsum ``ndfi,difo->ndfo`` keeps the whole multi-focus path vectorized.
    - In HybridMuon slice mode, each ``(C_in, F*C_out)`` slice gets independent
      NS update with stable rectangular scaling.

    Parameters
    ----------
    lmax
        Maximum spherical harmonic degree.
    in_channels
        Number of input channels per (l, m) coefficient.
    out_channels
        Number of output channels per (l, m) coefficient.
    n_focus
        Number of focus streams.
    precision
        Parameter precision.
    mlp_bias
        Whether to use bias for l=0 (scalar) components.
    trainable
        Whether parameters are trainable.
    seed
        Random seed for weight initialization.
    init_std
        If given, use normal(0, init_std) for all weights instead of default
        trunc-normal fan-in/fan-out init. Use 0.0 for zero initialization.
    """

    def __init__(
        self,
        *,
        lmax: int,
        in_channels: int,
        out_channels: int,
        n_focus: int = 1,
        precision: str = DEFAULT_PRECISION,
        mlp_bias: bool = False,
        trainable: bool = True,
        seed: int | list[int] | None = None,
        init_std: float | None = None,
    ) -> None:
        self.lmax = int(lmax)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_focus = int(n_focus)
        self.precision = precision
        self.trainable = bool(trainable)
        self.ebed_dim = get_so3_dim_of_lmax(self.lmax)
        self.mlp_bias = bool(mlp_bias)
        prec = PRECISION_DICT[self.precision.lower()]

        # === Step 1. Per-l weight matrix with focus folded on output axis ===
        # Storage: (lmax+1, C_in, F*C_out); runtime view: (lmax+1, C_in, F, C_out).
        num_l = self.lmax + 1
        weight = np.empty(
            (num_l, self.in_channels, self.n_focus * self.out_channels),
            dtype=prec,
        )
        if init_std is not None:
            if init_std == 0.0:
                weight[...] = 0.0
            else:
                rng = np.random.default_rng(seed)
                weight[...] = rng.normal(0.0, float(init_std), size=weight.shape)
        else:
            for l_idx in range(num_l):
                init_trunc_normal_fan_in_out(
                    weight[l_idx],
                    child_seed(seed, 1000 + l_idx),
                )
        self.weight = weight

        # === Step 2. Bias only for l=0 (scalar components) ===
        if self.mlp_bias:
            self.bias: np.ndarray | None = np.zeros(
                (self.n_focus * self.out_channels,), dtype=prec
            )
        else:
            self.bias = None

        # === Step 3. Precompute expand_index for weight lookup ===
        self.expand_index = map_degree_idx(self.lmax)

    def call(self, x: Any) -> Any:
        """
        Parameters
        ----------
        x
            Input features with shape (N, D, F, C_in) where D=(lmax+1)^2.

        Returns
        -------
        Array
            Order-wise mixed features with shape (N, D, F, C_out).
        """
        xp = array_api_compat.array_namespace(x)
        # === Step 1. Expand per-l weights to packed coefficient layout ===
        # (L, Cin, F*Cout) -> (L, Cin, F, Cout)
        weight = xp.reshape(
            xp_asarray_nodetach(
                xp, self.weight[...], device=array_api_compat.device(x)
            ),
            (self.lmax + 1, self.in_channels, self.n_focus, self.out_channels),
        )  # (L, Cin, F, Cout)
        # (L, Cin, F, Cout) -> (D, Cin, F, Cout)
        expand_index = xp_asarray_nodetach(
            xp, self.expand_index, device=array_api_compat.device(x)
        )
        weight_expanded = xp.take(weight, expand_index, axis=0)  # (D, Cin, F, Cout)

        # === Step 2. Per-focus, per-degree channel mixing ===
        # einsum "ndfi,difo->ndfo" as a broadcast batched matmul:
        # (N, D, F, 1, Cin) @ (1, D, F, Cin, Cout) -> (N, D, F, 1, Cout)
        weight_expanded = xp.permute_dims(
            weight_expanded, (0, 2, 1, 3)
        )  # (D, F, Cin, Cout)
        out = xp.matmul(x[:, :, :, None, :], weight_expanded[None, ...])[..., 0, :]

        # === Step 3. Add l=0 bias ===
        if self.mlp_bias:
            bias = xp_asarray_nodetach(
                xp, self.bias[...], device=array_api_compat.device(x)
            )
            bias = xp.reshape(bias, (self.n_focus, self.out_channels))
            out = xp.concat(
                [out[:, :1, :, :] + bias[None, None, ...], out[:, 1:, :, :]], axis=1
            )

        return out

    def serialize(self) -> dict[str, Any]:
        """Serialize the SO3Linear to a dict."""
        variables = {"weight": to_numpy_array(self.weight)}
        if self.mlp_bias:
            variables["bias"] = to_numpy_array(self.bias)
        variables["expand_index"] = to_numpy_array(self.expand_index)
        return {
            "@class": "SO3Linear",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "n_focus": self.n_focus,
                "precision": np.dtype(PRECISION_DICT[self.precision]).name,
                "mlp_bias": self.mlp_bias,
                "trainable": self.trainable,
                "seed": None,
            },
            "@variables": variables,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> SO3Linear:
        """Deserialize an SO3Linear from a dict."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "SO3Linear":
            raise ValueError(f"Invalid class for SO3Linear: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(
            lmax=int(config["lmax"]),
            in_channels=int(config["in_channels"]),
            out_channels=int(config["out_channels"]),
            n_focus=int(config["n_focus"]),
            precision=str(config["precision"]),
            mlp_bias=bool(config["mlp_bias"]),
            trainable=bool(config["trainable"]),
            seed=config.get("seed"),
        )
        prec = PRECISION_DICT[obj.precision.lower()]
        obj.expand_index = np.asarray(variables["expand_index"], dtype=np.int64)
        obj.weight = np.asarray(variables["weight"], dtype=prec)
        if obj.mlp_bias:
            obj.bias = np.asarray(variables["bias"], dtype=prec)
        return obj
