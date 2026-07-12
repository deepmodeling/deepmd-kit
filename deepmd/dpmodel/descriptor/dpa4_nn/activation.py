# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Activation helper modules for DPA4/SeZM.

This module contains coefficient-space nonlinear operators, including
GatedActivation and point-wise SwiGLU. Grid projectors and grid nets live in
dedicated modules so coefficient-space and function-space logic remain separate.

This module is the dpmodel (array-API) port of
``deepmd.pt.model.descriptor.sezm_nn.activation``.
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
    get_xp_precision,
    to_numpy_array,
)
from deepmd.dpmodel.utils.network import (
    Identity,
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
        - l>0: Each degree l has an independent gate derived from the l=0 scalar features.
               The gate for each l is expanded to all m components within that l-block.

    GLU mode (gate provided in call, e.g., from split linear output):
        - l=0: x0 * act(g0) (SwiGLU-style when act=silu, GeGLU when act=gelu, etc.)
        - l>0: Uses gate's scalar (g0) to generate sigmoid gates for x's vector components.
               This preserves SO(3) equivariance (scalar gates vector, not vector gates vector).

    This module also supports the m-major reduced layout used inside SO(2) blocks.
    If `mmax` is provided, the coefficient axis is assumed to follow the truncated
    m-major order built by `build_m_major_index(lmax, mmax)`; otherwise, it is assumed
    to be the full packed (l, m) layout with D=(lmax+1)^2.

    Parameters
    ----------
    lmax
        Maximum spherical harmonic degree.
    mmax
        Maximum order (|m|) for the m-major reduced layout. If None, use the full
        packed layout with D=(lmax+1)^2.
    channels
        Number of channels per focus stream.
    n_focus
        Number of focus streams.
    precision
        Internal compute precision used by the gate projection and sigmoid path.
    activation_function
        Activation function for l=0 components (e.g., "silu", "tanh", "gelu").
    mlp_bias
        Whether to use bias in the gate linear layer.
    layout
        Tensor layout convention. ``"nfdc"`` means input shape (N, F, D, C);
        ``"ndfc"`` means input shape (N, D, F, C); ``"fndc"`` means input shape
        (F, N, D, C), the focus-major layout used by the SO(2) mixing stack.
    trainable
        Whether parameters are trainable.
    seed
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
        if self.layout not in {"nfdc", "ndfc", "fndc"}:
            raise ValueError("`layout` must be one of 'nfdc', 'ndfc', or 'fndc'")

        self.activation_function = str(activation_function)
        self.scalar_act = get_activation_fn(activation_function)

        # === Build expand_index for mapping per-l gates to all m components ===
        if self.lmax > 0:
            if self.mmax is None:
                expand_index = map_degree_idx(self.lmax)[1:] - 1
            else:
                degree_index = build_m_major_l_index(self.lmax, self.mmax)
                expand_index = degree_index[1:] - 1
            self.gate_linear: NativeOP = FocusLinear(
                in_channels=self.channels,
                out_channels=self.lmax * self.channels,
                n_focus=self.n_focus,
                precision=self.precision,
                bias=self.mlp_bias,
                seed=seed,
                trainable=trainable,
            )

            prec = PRECISION_DICT[self.precision.lower()]
            rng = np.random.default_rng(child_seed(seed, 1))
            self.gate_linear.weight = rng.normal(
                0.0, 0.01, size=self.gate_linear.weight.shape
            ).astype(prec)
            if self.gate_linear.bias is not None:
                self.gate_linear.bias = np.zeros(
                    self.gate_linear.bias.shape, dtype=prec
                )
        else:
            expand_index = np.zeros(0, dtype=np.int64)
            self.gate_linear = Identity()
        self.expand_index = expand_index

        self.trainable = bool(trainable)

    def call(self, x: Any, gate: Any = None) -> Any:
        """
        Parameters
        ----------
        x
            Value features. Shape is (N, F, D, C) when ``layout='nfdc'``,
            (N, D, F, C) when ``layout='ndfc'``, or (F, N, D, C) when
            ``layout='fndc'``.
        gate
            Optional gate features with the same layout as ``x``.
            When provided, enables GLU mode:
            - l=0: x0 * act(g0) (e.g., SwiGLU when act=silu)
            - l>0: sigmoid(Linear(g0)) gates x's vector components
            When None (default), uses standard mode where gates are derived from x itself.

        Returns
        -------
        Array
            Gated features with the same layout as ``x``.
        """
        xp = array_api_compat.array_namespace(x)
        # ``ndfc`` carries the degree axis at position 1; ``nfdc`` and the
        # focus-major ``fndc`` carry it at position 2. Every select/narrow/reshape
        # below is expressed against this single degree axis, so the three layouts
        # share one code path apart from the per-focus gate projection.
        degree_axis = 1 if self.layout == "ndfc" else 2

        scalar_idx = tuple(
            0 if ax == degree_axis else slice(None) for ax in range(x.ndim)
        )
        l0_idx = tuple(
            slice(0, 1) if ax == degree_axis else slice(None) for ax in range(x.ndim)
        )
        rest_idx = tuple(
            slice(1, x.shape[degree_axis]) if ax == degree_axis else slice(None)
            for ax in range(x.ndim)
        )

        if gate is not None:
            gate_scalar_source = gate[scalar_idx]
        else:
            gate_scalar_source = x[scalar_idx]

        if gate is not None:
            x0 = x[l0_idx] * self.scalar_act(gate[l0_idx])
        else:
            x0 = self.scalar_act(x[l0_idx])

        if self.lmax == 0:
            return x0

        input_dtype = gate_scalar_source.dtype
        gate_src = xp.astype(gate_scalar_source, get_xp_precision(xp, self.precision))
        if self.layout == "fndc":
            # The scalar source is focus-major (F, N, C). ``FocusLinear`` mixes
            # channels with the focus stream on axis 1, so present it in the shared
            # (N, F, C) convention and restore the focus-major orientation.
            gate_logits = xp.permute_dims(
                self.gate_linear(xp.permute_dims(gate_src, (1, 0, 2))), (1, 0, 2)
            )
        else:
            gate_logits = self.gate_linear(gate_src)
        gating_scalars = xp.astype(xp_sigmoid(gate_logits), input_dtype)
        gating_scalars = xp.reshape(
            gating_scalars,
            (x.shape[0], gate_scalar_source.shape[1], self.lmax, self.channels),
        )
        expand_index = xp_asarray_nodetach(
            xp, self.expand_index, device=array_api_compat.device(x)
        )
        gates = xp.take(gating_scalars, expand_index, axis=2)
        if self.layout == "ndfc":
            gates = xp.permute_dims(gates, (0, 2, 1, 3))
        return xp.concat([x0, x[rest_idx] * gates], axis=degree_axis)

    def serialize(self) -> dict[str, Any]:
        variables = {"expand_index": to_numpy_array(self.expand_index)}
        if self.lmax > 0:
            variables["gate_linear.weight"] = to_numpy_array(self.gate_linear.weight)
            if self.gate_linear.bias is not None:
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
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "GatedActivation":
            raise ValueError(f"Invalid class for GatedActivation: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(**config)
        prec = PRECISION_DICT[obj.precision.lower()]
        obj.expand_index = np.asarray(variables["expand_index"], dtype=np.int64)
        if obj.lmax > 0:
            obj.gate_linear.weight = np.asarray(
                variables["gate_linear.weight"], dtype=prec
            )
            if obj.gate_linear.bias is not None:
                obj.gate_linear.bias = np.asarray(
                    variables["gate_linear.bias"], dtype=prec
                )
        return obj


class SwiGLU(NativeOP):
    """Point-wise SwiGLU on the last feature axis."""

    def call(self, inputs: Any) -> Any:
        # torch.chunk(inputs, 2, dim=-1): the first half takes ceil(C/2) elements.
        nc = (inputs.shape[-1] + 1) // 2
        gate = inputs[..., :nc]
        value = inputs[..., nc:]
        return gate * xp_sigmoid(gate) * value
