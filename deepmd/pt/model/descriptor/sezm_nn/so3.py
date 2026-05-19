# SPDX-License-Identifier: LGPL-3.0-or-later
"""
SO(3)-equivariant linear layers for SeZM.

This module defines the channel-only and focus-aware linear maps used by SeZM
SO(3) feature transformations.
"""

from __future__ import (
    annotations,
)

import math
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
    get_generator,
)

from .indexing import (
    get_so3_dim_of_lmax,
    map_degree_idx,
)
from .utils import (
    init_trunc_normal_fan_in_out,
    np_safe,
    safe_numpy_to_tensor,
)


class FocusLinear(nn.Module):
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
    dtype
        Parameter dtype.
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
        dtype: torch.dtype,
        bias: bool = True,
        trainable: bool,
        seed: int | list[int] | None = None,
        init_std: float | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_focus = int(n_focus)
        self.dtype = dtype
        self.device = env.DEVICE
        self.use_bias = bool(bias)
        self.weight = nn.Parameter(
            torch.empty(
                self.in_channels,
                self.n_focus * self.out_channels,
                device=self.device,
                dtype=self.dtype,
            )
        )
        gen = get_generator(seed)
        if init_std is not None:
            nn.init.normal_(self.weight, mean=0.0, std=init_std, generator=gen)
        else:
            bound = 1.0 / math.sqrt(self.in_channels)
            nn.init.uniform_(self.weight, -bound, bound, generator=gen)
        if self.use_bias:
            self.bias: nn.Parameter | None = nn.Parameter(
                torch.zeros(
                    self.n_focus * self.out_channels,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
        else:
            self.bias = None
        for p in self.parameters():
            p.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Input tensor with shape (B, F, Cin).

        Returns
        -------
        torch.Tensor
            Projected tensor with shape (B, F, Cout).
        """
        weight = self.weight.view(self.in_channels, self.n_focus, self.out_channels)
        out = torch.einsum("bfi,ifo->bfo", x, weight)
        if self.use_bias:
            bias = self.bias.view(self.n_focus, self.out_channels)
            out = out + bias.unsqueeze(0)
        return out


class ChannelLinear(nn.Module):
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
    dtype
        Parameter dtype.
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
        dtype: torch.dtype,
        bias: bool = True,
        trainable: bool,
        seed: int | list[int] | None = None,
        init_std: float | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.dtype = dtype
        self.device = env.DEVICE
        self.use_bias = bool(bias)
        self.weight = nn.Parameter(
            torch.empty(
                self.in_channels,
                self.out_channels,
                device=self.device,
                dtype=self.dtype,
            )
        )
        gen = get_generator(seed)
        if init_std is not None:
            nn.init.normal_(self.weight, mean=0.0, std=init_std, generator=gen)
        else:
            bound = 1.0 / math.sqrt(self.in_channels)
            nn.init.uniform_(self.weight, -bound, bound, generator=gen)
        if self.use_bias:
            self.bias: nn.Parameter | None = nn.Parameter(
                torch.zeros(
                    self.out_channels,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
        else:
            self.bias = None
        for p in self.parameters():
            p.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Input tensor with shape ``(..., C_in)``.

        Returns
        -------
        torch.Tensor
            Projected tensor with shape ``(..., C_out)``.
        """
        out = torch.einsum("...i,io->...o", x, self.weight)
        if self.use_bias:
            out = out + self.bias
        return out


class SO3Linear(nn.Module):
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
    dtype
        Parameter dtype.
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
        dtype: torch.dtype,
        mlp_bias: bool = False,
        trainable: bool,
        seed: int | list[int] | None = None,
        init_std: float | None = None,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_focus = int(n_focus)
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[dtype]
        self.ebed_dim = get_so3_dim_of_lmax(self.lmax)
        self.mlp_bias = bool(mlp_bias)

        # === Step 1. Per-l weight matrix with focus folded on output axis ===
        # Storage: (lmax+1, C_in, F*C_out); runtime view: (lmax+1, C_in, F, C_out).
        num_l = self.lmax + 1
        self.weight = nn.Parameter(
            torch.empty(
                num_l,
                self.in_channels,
                self.n_focus * self.out_channels,
                dtype=self.dtype,
                device=self.device,
            )
        )
        if init_std is not None:
            if init_std == 0.0:
                nn.init.zeros_(self.weight)
            else:
                nn.init.normal_(
                    self.weight,
                    mean=0.0,
                    std=init_std,
                    generator=get_generator(seed),
                )
        else:
            for l_idx in range(num_l):
                init_trunc_normal_fan_in_out(
                    self.weight[l_idx],
                    child_seed(seed, 1000 + l_idx),
                )

        # === Step 2. Bias only for l=0 (scalar components) ===
        if self.mlp_bias:
            self.bias: nn.Parameter | None = nn.Parameter(
                torch.zeros(
                    self.n_focus * self.out_channels,
                    dtype=self.dtype,
                    device=self.device,
                )
            )
        else:
            self.bias = None

        # === Step 3. Precompute expand_index for weight lookup ===
        self.register_buffer(
            "expand_index",
            map_degree_idx(self.lmax, device=self.device),
            persistent=True,
        )

        for p in self.parameters():
            p.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Input features with shape (N, D, F, C_in) where D=(lmax+1)^2.

        Returns
        -------
        torch.Tensor
            Order-wise mixed features with shape (N, D, F, C_out).
        """
        # === Step 1. Expand per-l weights to packed coefficient layout ===
        # (L, Cin, F*Cout) -> (L, Cin, F, Cout)
        weight = self.weight.view(
            self.lmax + 1,
            self.in_channels,
            self.n_focus,
            self.out_channels,
        )  # (L, Cin, F, Cout)
        # (L, Cin, F, Cout) -> (D, Cin, F, Cout)
        weight_expanded = torch.index_select(
            weight, dim=0, index=self.expand_index
        )  # (D, Cin, F, Cout)

        # === Step 2. Per-focus, per-degree channel mixing ===
        out = torch.einsum("ndfi,difo->ndfo", x, weight_expanded)

        # === Step 3. Add l=0 bias ===
        if self.mlp_bias:
            bias = self.bias.view(self.n_focus, self.out_channels)
            out[:, 0, :, :] = out[:, 0, :, :] + bias.unsqueeze(0)

        return out

    def serialize(self) -> dict[str, Any]:
        trainable = all(p.requires_grad for p in self.parameters())
        state = self.state_dict()
        return {
            "@class": "SO3Linear",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "n_focus": self.n_focus,
                "precision": RESERVED_PRECISION_DICT[self.dtype],
                "mlp_bias": self.mlp_bias,
                "trainable": trainable,
                "seed": None,
            },
            "@variables": {key: np_safe(value) for key, value in state.items()},
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> SO3Linear:
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "SO3Linear":
            raise ValueError(f"Invalid class for SO3Linear: {data_cls}")
        version = int(data.pop("@version"))
        if version != 1:
            raise ValueError(f"Unsupported SO3Linear version: {version}")
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
