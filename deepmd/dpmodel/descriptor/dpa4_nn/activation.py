# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Activation helper modules for DPA4/SeZM.

This module is the dpmodel port of
``deepmd.pt.model.descriptor.sezm_nn.activation``. It contains the
coefficient-space nonlinear operators. Both pt classes are ported:
``GatedActivation`` (used by ``so2``, ``ffn``) and ``SwiGLU`` (used by
``grid_net``, which is consumed by ``ffn``).

Serialization contract: ``GatedActivation`` mirrors the pt ``serialize()``
format exactly (same config and ``@variables`` keys, including the nested
``gate_linear.weight``/``gate_linear.bias`` state-dict names), so pt
``serialize()`` output deserializes directly. ``SwiGLU`` is parameter-free in
pt (no ``serialize()``, no state-dict entries), so no serialization is
implemented for it.
"""

from __future__ import (
    annotations,
)

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
    xp_sigmoid,
)
from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.utils.network import (
    get_activation_fn,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .indexing import (
    build_m_major_l_index,
    map_degree_idx,
)
from .so3 import (
    FocusLinear,
)


class GatedActivation(NativeOP):
    """
    Gated activation for SO(3) equivariant features with per-l independent gates.

    Standard mode (gate=None in call):
        - l=0: Uses the specified activation function
        - l>0: Each degree l has an independent gate derived from the l=0 scalar
          features. The gate for each l is expanded to all m components within
          that l-block.

    GLU mode (gate provided in call, e.g., from split linear output):
        - l=0: x0 * act(g0) (SwiGLU-style when act=silu, GeGLU when act=gelu, etc.)
        - l>0: Uses gate's scalar (g0) to generate sigmoid gates for x's vector
          components. This preserves SO(3) equivariance (scalar gates vector,
          not vector gates vector).

    This module also supports the m-major reduced layout used inside SO(2)
    blocks. If `mmax` is provided, the coefficient axis is assumed to follow
    the truncated m-major order built by `build_m_major_index(lmax, mmax)`;
    otherwise, it is assumed to be the full packed (l, m) layout with
    D=(lmax+1)^2.

    Parameters
    ----------
    lmax : int
        Maximum spherical harmonic degree.
    mmax : int | None
        Maximum order (|m|) for the m-major reduced layout. If None, use the
        full packed layout with D=(lmax+1)^2.
    channels : int
        Number of channels per focus stream.
    n_focus : int
        Number of focus streams.
    precision : str
        Internal compute precision used by the gate projection and sigmoid path.
    activation_function : str
        Activation function for l=0 components (e.g., "silu", "tanh", "gelu").
    mlp_bias : bool
        Whether to use bias in the gate linear layer.
    layout : str
        Tensor layout convention. ``"nfdc"`` means input shape (N, F, D, C);
        ``"ndfc"`` means input shape (N, D, F, C).
    trainable : bool
        Whether parameters are trainable.
    seed : int | list[int] | None
        Random seed for weight initialization.
    """

    def __init__(
        self,
        *,
        lmax: int,
        mmax: int | None = None,
        channels: int,
        n_focus: int = 1,
        precision: str = DEFAULT_PRECISION,
        activation_function: str = "silu",
        mlp_bias: bool = False,
        layout: str = "nfdc",
        trainable: bool = True,
        seed: int | list[int] | None = None,
    ) -> None:
        self.lmax = int(lmax)
        self.mmax = None if mmax is None else int(mmax)
        if self.mmax is not None:
            if self.mmax < 0:
                raise ValueError("`mmax` must be non-negative")
            if self.mmax > self.lmax:
                raise ValueError("`mmax` must be <= `lmax`")
        self.channels = int(channels)
        self.n_focus = int(n_focus)
        self.precision = precision
        self.mlp_bias = bool(mlp_bias)
        self.layout = str(layout).lower()
        if self.layout not in {"nfdc", "ndfc"}:
            raise ValueError("`layout` must be either 'nfdc' or 'ndfc'")
        self.trainable = bool(trainable)
        self.activation_function = str(activation_function)
        prec = PRECISION_DICT[self.precision.lower()]

        # === Build expand_index for mapping per-l gates to all m components ===
        if self.lmax > 0:
            if self.mmax is None:
                expand_index = map_degree_idx(self.lmax)[1:] - 1
            else:
                degree_index = build_m_major_l_index(self.lmax, self.mmax)
                expand_index = degree_index[1:] - 1
            self.gate_linear: FocusLinear | None = FocusLinear(
                in_channels=self.channels,
                out_channels=self.lmax * self.channels,
                n_focus=self.n_focus,
                precision=self.precision,
                bias=self.mlp_bias,
                seed=seed,
                trainable=self.trainable,
            )
            # pt re-initializes the gate weight with normal(0, 0.01) seeded
            # by child_seed(seed, 1) and zeroes the bias (bias is already
            # zero-initialized here).
            rng = np.random.default_rng(child_seed(seed, 1))
            self.gate_linear.weight = rng.normal(
                0.0, 0.01, size=self.gate_linear.weight.shape
            ).astype(prec)
        else:
            # pt uses nn.Identity() here (parameter-free, no state-dict keys);
            # the dpmodel equivalent is no gate module at all.
            expand_index = np.zeros((0,), dtype=np.int64)
            self.gate_linear = None
        self.expand_index = expand_index

    def call(self, x: Any, gate: Any = None) -> Any:
        """
        Apply the gated activation.

        Parameters
        ----------
        x : Array
            Value features. Shape is (N, F, D, C) when ``layout='nfdc'``,
            or (N, D, F, C) when ``layout='ndfc'``.
        gate : Array | None
            Optional gate features with the same layout as ``x``.
            When provided, enables GLU mode:
            - l=0: x0 * act(g0) (e.g., SwiGLU when act=silu)
            - l>0: sigmoid(Linear(g0)) gates x's vector components
            When None (default), uses standard mode where gates are derived
            from x itself.

        Returns
        -------
        Array
            Gated features with the same layout as ``x``.
        """
        xp = array_api_compat.array_namespace(x)
        degree_axis = 1 if self.layout == "ndfc" else 2

        gate_source = x if gate is None else gate
        if degree_axis == 1:
            gate_scalar_source = gate_source[:, 0, :, :]  # (N, F, C)
            g0 = gate_source[:, :1, :, :]
            x0_in = x[:, :1, :, :]
        else:
            gate_scalar_source = gate_source[:, :, 0, :]  # (N, F, C)
            g0 = gate_source[:, :, :1, :]
            x0_in = x[:, :, :1, :]

        scalar_act = get_activation_fn(self.activation_function)
        if gate is not None:
            x0 = x0_in * scalar_act(g0)
        else:
            x0 = scalar_act(x0_in)

        if self.lmax == 0:
            return x0

        gate_weight = xp_asarray_nodetach(
            xp, self.gate_linear.weight[...], device=array_api_compat.device(x)
        )
        input_dtype = gate_scalar_source.dtype
        if input_dtype != gate_weight.dtype:
            gate_scalar_source = xp.astype(gate_scalar_source, gate_weight.dtype)
        gating_scalars = xp_sigmoid(self.gate_linear.call(gate_scalar_source))
        if gating_scalars.dtype != input_dtype:
            gating_scalars = xp.astype(gating_scalars, input_dtype)
        gating_scalars = xp.reshape(
            gating_scalars,
            (x.shape[0], gate_scalar_source.shape[1], self.lmax, self.channels),
        )
        expand_index = xp_asarray_nodetach(
            xp, self.expand_index, device=array_api_compat.device(x)
        )
        gates = xp.take(gating_scalars, expand_index, axis=2)  # (N, F, D-1, C)
        if self.layout == "ndfc":
            gates = xp.permute_dims(gates, (0, 2, 1, 3))  # (N, D-1, F, C)
            xt = x[:, 1:, :, :] * gates
        else:
            xt = x[:, :, 1:, :] * gates
        return xp.concat([x0, xt], axis=degree_axis)

    def serialize(self) -> dict[str, Any]:
        """Serialize the GatedActivation to a dict (pt-compatible format)."""
        variables = {"expand_index": to_numpy_array(self.expand_index)}
        if self.gate_linear is not None:
            variables["gate_linear.weight"] = to_numpy_array(self.gate_linear.weight)
            if self.mlp_bias:
                variables["gate_linear.bias"] = to_numpy_array(self.gate_linear.bias)
        return {
            "@class": "GatedActivation",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "mmax": self.mmax,
                "channels": self.channels,
                "n_focus": self.n_focus,
                "precision": np.dtype(PRECISION_DICT[self.precision]).name,
                "activation_function": self.activation_function,
                "mlp_bias": self.mlp_bias,
                "layout": self.layout,
                "trainable": self.trainable,
                "seed": None,
            },
            "@variables": variables,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> GatedActivation:
        """Deserialize a GatedActivation from a dict."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "GatedActivation":
            raise ValueError(f"Invalid class for GatedActivation: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        mmax = config["mmax"]
        obj = cls(
            lmax=int(config["lmax"]),
            mmax=None if mmax is None else int(mmax),
            channels=int(config["channels"]),
            n_focus=int(config["n_focus"]),
            precision=str(config["precision"]),
            activation_function=str(config["activation_function"]),
            mlp_bias=bool(config["mlp_bias"]),
            layout=str(config["layout"]),
            trainable=bool(config["trainable"]),
            seed=config.get("seed"),
        )
        prec = PRECISION_DICT[obj.precision.lower()]
        expand_index = np.asarray(variables["expand_index"], dtype=np.int64)
        if not np.array_equal(expand_index, obj.expand_index):
            raise ValueError("expand_index does not match the lmax/mmax tables")
        if obj.gate_linear is not None:
            weight = np.asarray(variables["gate_linear.weight"], dtype=prec)
            if weight.shape != obj.gate_linear.weight.shape:
                raise ValueError(
                    f"gate_linear.weight shape {weight.shape} does not match "
                    f"the expected shape {obj.gate_linear.weight.shape}"
                )
            obj.gate_linear.weight = weight
            if obj.mlp_bias:
                obj.gate_linear.bias = np.asarray(
                    variables["gate_linear.bias"], dtype=prec
                ).reshape(obj.gate_linear.bias.shape)
        return obj


class SwiGLU(NativeOP):
    """Point-wise SwiGLU on the last feature axis.

    Parameter-free, matching the pt version (which defines no ``serialize()``
    and contributes no state-dict entries).
    """

    def call(self, inputs: Any) -> Any:
        """
        Apply point-wise SwiGLU.

        Parameters
        ----------
        inputs : Array
            Input array with shape ``(..., 2*C)``; the first half of the last
            axis is the gate, the second half the value.

        Returns
        -------
        Array
            Gated array with shape ``(..., C)``.
        """
        # torch.chunk(inputs, 2, dim=-1): first chunk gets ceil(C/2) entries
        nc = (inputs.shape[-1] + 1) // 2
        gate = inputs[..., :nc]
        value = inputs[..., nc:]
        return gate * xp_sigmoid(gate) * value
