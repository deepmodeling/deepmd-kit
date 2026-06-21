# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Activation helper modules for SeZM.

This module contains coefficient-space nonlinear operators, including
GatedActivation and point-wise SwiGLU. Grid projectors and grid nets live in
dedicated modules so coefficient-space and function-space logic remain separate.
"""

from __future__ import (
    annotations,
)

from typing import (
    Any,
)

import torch
import torch.nn as nn

from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
    RESERVED_PRECISION_DICT,
)
from deepmd.pt.utils.utils import (
    ActivationFn,
    get_generator,
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
from .utils import (
    np_safe,
    safe_numpy_to_tensor,
)


class GatedActivation(nn.Module):
    """
    Gated activation for SO(3) equivariant features with per-l independent gates.

    Standard mode (gate=None in forward):
        - l=0: Uses the specified activation function
        - l>0: Each degree l has an independent gate derived from the l=0 scalar features.
               The gate for each l is expanded to all m components within that l-block.

    GLU mode (gate provided in forward, e.g., from split linear output):
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
    dtype
        Internal compute dtype used by the gate projection and sigmoid path.
    activation_function
        Activation function for l=0 components (e.g., "silu", "tanh", "gelu").
    mlp_bias
        Whether to use bias in the gate linear layer.
    layout
        Tensor layout convention. ``"nfdc"`` means input shape (N, F, D, C);
        ``"ndfc"`` means input shape (N, D, F, C).
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
        dtype: torch.dtype,
        activation_function: str = "silu",
        mlp_bias: bool = False,
        layout: str = "nfdc",
        trainable: bool,
        seed: int | list[int] | None = None,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.mmax = None if mmax is None else int(mmax)
        if self.mmax is not None:
            if self.mmax < 0:
                raise ValueError("`mmax` must be non-negative")
            if self.mmax > self.lmax:
                raise ValueError("`mmax` must be <= `lmax`")
        self.channels = int(channels)
        self.n_focus = int(n_focus)
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[dtype]
        self.mlp_bias = bool(mlp_bias)
        self.layout = str(layout).lower()
        if self.layout not in {"nfdc", "ndfc"}:
            raise ValueError("`layout` must be either 'nfdc' or 'ndfc'")

        self.scalar_act = ActivationFn(activation_function)

        # === Build expand_index for mapping per-l gates to all m components ===
        if self.lmax > 0:
            if self.mmax is None:
                expand_index = map_degree_idx(self.lmax, device=self.device)[1:] - 1
            else:
                degree_index = build_m_major_l_index(
                    self.lmax, self.mmax, device=self.device
                )
                expand_index = degree_index[1:] - 1
            self.gate_linear: nn.Module = FocusLinear(
                in_channels=self.channels,
                out_channels=self.lmax * self.channels,
                n_focus=self.n_focus,
                dtype=self.dtype,
                bias=self.mlp_bias,
                seed=seed,
                trainable=trainable,
            )

            gen_gate = get_generator(child_seed(seed, 1))
            nn.init.normal_(
                self.gate_linear.weight, mean=0.0, std=0.01, generator=gen_gate
            )
            if self.gate_linear.bias is not None:
                nn.init.zeros_(self.gate_linear.bias)
        else:
            expand_index = torch.zeros(0, dtype=torch.long, device=self.device)
            self.gate_linear = nn.Identity()
        self.register_buffer("expand_index", expand_index, persistent=True)

        for p in self.parameters():
            p.requires_grad = trainable

    def forward(
        self, x: torch.Tensor, gate: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Value features. Shape is (N, F, D, C) when ``layout='nfdc'``,
            or (N, D, F, C) when ``layout='ndfc'``.
        gate
            Optional gate features with the same layout as ``x``.
            When provided, enables GLU mode:
            - l=0: x0 * act(g0) (e.g., SwiGLU when act=silu)
            - l>0: sigmoid(Linear(g0)) gates x's vector components
            When None (default), uses standard mode where gates are derived from x itself.

        Returns
        -------
        torch.Tensor
            Gated features with the same layout as ``x``.
        """
        degree_axis = 1 if self.layout == "ndfc" else 2

        if gate is not None:
            gate_scalar_source = gate.select(dim=degree_axis, index=0)
        else:
            gate_scalar_source = x.select(dim=degree_axis, index=0)

        if gate is not None:
            x0 = x.narrow(degree_axis, 0, 1) * self.scalar_act(
                gate.narrow(degree_axis, 0, 1)
            )
        else:
            x0 = self.scalar_act(x.narrow(degree_axis, 0, 1))

        if self.lmax == 0:
            return x0

        input_dtype = gate_scalar_source.dtype
        gating_scalars = torch.sigmoid(
            self.gate_linear(gate_scalar_source.to(dtype=self.dtype))
        ).to(dtype=input_dtype)
        gating_scalars = gating_scalars.reshape(
            x.shape[0], gate_scalar_source.shape[1], self.lmax, self.channels
        )
        gates = gating_scalars.index_select(dim=2, index=self.expand_index)
        if self.layout == "ndfc":
            gates = gates.transpose(1, 2)

        out = x.new_empty(x.shape)
        out.narrow(degree_axis, 0, 1).copy_(x0)
        out.narrow(degree_axis, 1, x.shape[degree_axis] - 1).copy_(
            x.narrow(degree_axis, 1, x.shape[degree_axis] - 1) * gates
        )
        return out

    def serialize(self) -> dict[str, Any]:
        trainable = all(p.requires_grad for p in self.parameters())
        state = self.state_dict()
        return {
            "@class": "GatedActivation",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "mmax": self.mmax,
                "channels": self.channels,
                "n_focus": self.n_focus,
                "precision": RESERVED_PRECISION_DICT[self.dtype],
                "activation_function": self.scalar_act.activation,
                "mlp_bias": self.mlp_bias,
                "layout": self.layout,
                "trainable": trainable,
                "seed": None,
            },
            "@variables": {key: np_safe(value) for key, value in state.items()},
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
        precision = config.pop("precision")
        config["dtype"] = PRECISION_DICT[precision]
        obj = cls(**config)
        template = obj.state_dict()
        state = {
            key: safe_numpy_to_tensor(
                value, device=template[key].device, dtype=template[key].dtype
            )
            for key, value in variables.items()
        }
        obj.load_state_dict(state)
        return obj


class SwiGLU(nn.Module):
    """Point-wise SwiGLU on the last feature axis."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gate, value = torch.chunk(inputs, chunks=2, dim=-1)
        return gate * torch.sigmoid(gate) * value
