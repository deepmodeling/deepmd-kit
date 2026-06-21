# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Equivariant feed-forward layers for SeZM.

This module defines the full SO(3)-equivariant feed-forward network used
inside SeZM interaction blocks and the descriptor output head.
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
from deepmd.utils.version import (
    check_version_compatibility,
)

from .activation import (
    GatedActivation,
)
from .grid_net import (
    S2GridNet,
    SO3GridNet,
)
from .projection import (
    resolve_s2_grid_resolution,
)
from .so3 import (
    SO3Linear,
)
from .utils import (
    get_promoted_dtype,
    np_safe,
    safe_numpy_to_tensor,
)


class EquivariantFFN(nn.Module):
    """
    Full equivariant FFN operating on all spherical harmonic degrees.

    Default structure (glu_activation=False):
        SO3 linear (in -> hidden) -> GatedActivation -> SO3 linear (hidden -> out)

    Default structure (glu_activation=True):
        SO3 linear (in -> 2*hidden) -> split -> GatedActivation(val, gate) -> SO3 linear (hidden -> out)

    Optional grid-FFN structure (s2_activation=True or ffn_so3_grid=True):
        SO3 linear (in -> hidden)
        -> project packed SO(3) coefficients to the S2 or SO3 grid
        -> grid GLU, polynomial MLP, or scalar-routed attention on hidden features
        -> project grid features back to packed SO(3) coefficients
        -> add scalar LinearSwiGLU branch to l=0
        -> SO3 linear (hidden -> out)

    GatedActivation serves as the unified "activation" for equivariant networks,
    analogous to SiLU in standard MLPs, but respecting SO(3) equivariance:
    - l=0: Uses the specified activation function (or GLU variant when glu_activation=True)
    - l>0: sigmoid gate from l=0 scalar features

    When glu_activation=True, the first linear outputs 2*hidden_channels, then splits into
    value and gate branches. This transforms activations like silu->swiglu, gelu->geglu.
    The split approach is more efficient than two separate linear layers.

    Parameters
    ----------
    lmax
        Maximum degree.
    channels
        Number of channels per (l, m) coefficient.
    hidden_channels
        Hidden dimension for the FFN.
    kmax
        Maximum Wigner-D frame order (|k|) used by the SO3 Wigner-D FFN grid.
    grid_mlp
        If True, select the polynomial grid MLP operation when the
        block-internal FFN grid path is enabled.
    grid_branch
        Number of scalar-routed polynomial product branches used when the
        block-internal FFN grid path is enabled. ``0`` disables this branch
        mixer. Positive values take precedence over ``grid_mlp``.
    dtype
        Parameter dtype.
    s2_activation
        If True, enable the S2 FFN grid path.
    ffn_so3_grid
        If True, enable the SO3 Wigner-D FFN grid path.
    lebedev_quadrature
        If True, use Lebedev quadrature for the S2 projector in this FFN.
    activation_function
        Activation function for l=0 components (e.g., "silu", "tanh", "gelu").
    glu_activation
        If True, use GLU-style gating (e.g., silu -> swiglu, gelu -> geglu).
    mlp_bias
        Whether to use bias in SO3Linear (l=0 bias), GatedActivation
        (gate linear bias), and the scalar point-wise projection when
        ``grid_mlp=True``.
    trainable
        Whether parameters are trainable.
    seed
        Random seed for weight initialization.
    """

    def __init__(
        self,
        *,
        lmax: int,
        channels: int,
        hidden_channels: int,
        kmax: int = 1,
        grid_mlp: bool = False,
        grid_branch: int = 0,
        dtype: torch.dtype,
        s2_activation: bool = False,
        ffn_so3_grid: bool = False,
        lebedev_quadrature: bool = False,
        activation_function: str = "silu",
        glu_activation: bool = True,
        mlp_bias: bool = False,
        trainable: bool,
        seed: int | list[int] | None = None,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)
        self.kmax = int(kmax)
        if self.kmax < 0:
            raise ValueError("`kmax` must be non-negative")
        self.use_grid_mlp = bool(grid_mlp)
        self.grid_branch = int(grid_branch)
        if self.grid_branch < 0:
            raise ValueError("`grid_branch` must be non-negative")
        self.use_grid_branch = self.grid_branch > 0
        self.s2_activation = bool(s2_activation)
        self.ffn_so3_grid = bool(ffn_so3_grid)
        self.lebedev_quadrature = bool(lebedev_quadrature)
        self.s2_grid_method = "lebedev" if self.lebedev_quadrature else "e3nn"
        base_grid = resolve_s2_grid_resolution(
            self.lmax,
            self.lmax,
            method=self.s2_grid_method,
        )
        self.s2_grid_resolution = (
            [max(base_grid), max(base_grid)]
            if self.s2_grid_method == "e3nn"
            else base_grid
        )
        self.activation_function = activation_function
        self.glu_activation = bool(glu_activation)
        self.mlp_bias = bool(mlp_bias)
        self.dtype = dtype
        self.compute_dtype = get_promoted_dtype(self.dtype)
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[dtype]
        self.grid_n_frames = 2 * self.kmax + 1 if self.ffn_so3_grid else 1

        # === Step 0. Split deterministic seeds at the module top-level ===
        seed_so3_in = child_seed(seed, 0)
        seed_act = child_seed(seed, 1)
        seed_so3_out = child_seed(seed, 2)

        # === First SO3Linear for channel mixing ===
        self.use_grid_net = self.s2_activation or self.ffn_so3_grid
        linear1_out_channels = self.hidden_channels
        if self.use_grid_net:
            linear1_out_channels = 2 * self.grid_n_frames * self.hidden_channels
        else:
            linear1_out_channels = (
                2 * self.hidden_channels
                if self.glu_activation
                else self.hidden_channels
            )
        self.so3_linear_1 = SO3Linear(
            lmax=self.lmax,
            in_channels=self.channels,
            out_channels=linear1_out_channels,
            n_focus=1,
            dtype=dtype,
            mlp_bias=self.mlp_bias,
            trainable=trainable,
            seed=seed_so3_in,
        )

        # === Equivariant nonlinearity path ===
        if self.use_grid_net:
            grid_op = (
                "branch"
                if self.use_grid_branch
                else ("mlp" if self.use_grid_mlp else "glu")
            )
            if self.ffn_so3_grid:
                self.act = SO3GridNet(
                    lmax=self.lmax,
                    kmax=self.kmax,
                    channels=self.hidden_channels,
                    n_focus=1,
                    mode="self",
                    op_type=grid_op,
                    dtype=self.compute_dtype,
                    layout="ndfc",
                    grid_branches=max(1, self.grid_branch),
                    mlp_bias=self.mlp_bias,
                    trainable=trainable,
                    seed=seed_act,
                )
            else:
                self.act = S2GridNet(
                    lmax=self.lmax,
                    channels=self.hidden_channels,
                    n_focus=1,
                    mode="self",
                    op_type=grid_op,
                    dtype=self.compute_dtype,
                    layout="ndfc",
                    grid_resolution_list=self.s2_grid_resolution,
                    coefficient_layout="packed",
                    grid_method=self.s2_grid_method,
                    grid_branches=max(1, self.grid_branch),
                    mlp_bias=self.mlp_bias,
                    trainable=trainable,
                    seed=seed_act,
                )
        else:
            self.act = GatedActivation(
                lmax=self.lmax,
                channels=self.hidden_channels,
                dtype=self.compute_dtype,
                activation_function=activation_function,
                mlp_bias=self.mlp_bias,
                layout="ndfc",
                trainable=trainable,
                seed=seed_act,
            )

        # === Second SO3Linear for channel mixing ===
        # Zero-initialized so residual path starts near-identity.
        self.so3_linear_2 = SO3Linear(
            lmax=self.lmax,
            in_channels=self.grid_n_frames * self.hidden_channels,
            out_channels=self.channels,
            n_focus=1,
            dtype=dtype,
            mlp_bias=self.mlp_bias,
            trainable=trainable,
            seed=seed_so3_out,
            init_std=0.0,
        )

        for p in self.parameters():
            p.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Input with shape (N, D, F, C) where D=(lmax+1)^2.

        Returns
        -------
        torch.Tensor
            Output with shape (N, D, F, C).
        """
        # === Step 1. Input up projection ===
        x = self.so3_linear_1(x)

        # === Step 2. Equivariant nonlinearity ===
        if self.use_grid_net:
            x = self.act(x)
        elif self.glu_activation:
            # Split into value and gate branches along channel dimension
            x_val, x_gate = x.chunk(2, dim=-1)
            # Pass gate to GatedActivation for GLU-style gating
            x = self.act(x_val, gate=x_gate)
        else:
            x = self.act(x)

        # === Step 3. Per-degree output projection ===
        x = self.so3_linear_2(x)

        return x

    def serialize(self) -> dict[str, Any]:
        trainable = all(p.requires_grad for p in self.parameters())
        state = self.state_dict()
        return {
            "@class": "EquivariantFFN",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "channels": self.channels,
                "hidden_channels": self.hidden_channels,
                "kmax": self.kmax,
                "grid_mlp": self.use_grid_mlp,
                "grid_branch": self.grid_branch,
                "precision": RESERVED_PRECISION_DICT[self.dtype],
                "s2_activation": self.s2_activation,
                "ffn_so3_grid": self.ffn_so3_grid,
                "lebedev_quadrature": self.lebedev_quadrature,
                "activation_function": self.activation_function,
                "glu_activation": self.glu_activation,
                "mlp_bias": self.mlp_bias,
                "trainable": trainable,
                "seed": None,
            },
            "@variables": {key: np_safe(value) for key, value in state.items()},
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> EquivariantFFN:
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "EquivariantFFN":
            raise ValueError(f"Invalid class for EquivariantFFN: {data_cls}")
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
