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
    S2GridProjector,
    SwiGLU,
    SwiGLUS2Activation,
    resolve_s2_grid_resolution,
)
from .so3 import (
    ChannelLinear,
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

    Optional grid-FFN structure (grid_mlp=True):
        SO3 linear (in -> hidden)
        -> project packed SO(3) coefficients to the S2 grid
        -> packed S2-grid point-wise MLP on hidden features
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
    grid_mlp
        If True, use the optional grid-MLP FFN structure on the block-internal
        FFN path. This path takes precedence over the simpler activation-only
        path inside this module.
    dtype
        Parameter dtype.
    s2_activation
        If True and ``grid_mlp=False``, replace the default GatedActivation path
        with the merged scalar/grid SwiGLU-S2 activation.
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
        grid_mlp: bool = False,
        dtype: torch.dtype,
        s2_activation: bool = False,
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
        self.use_grid_mlp = bool(grid_mlp)
        self.s2_activation = bool(s2_activation)
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

        # === Step 0. Split deterministic seeds at the module top-level ===
        seed_so3_in = child_seed(seed, 0)
        seed_act = child_seed(seed, 1)
        seed_so3_out = child_seed(seed, 2)

        # === First SO3Linear for channel mixing ===
        # Grid-FFN keeps the hidden width and performs the nonlinear expansion
        # inside the scalar/grid point-wise MLPs.
        linear1_out_channels = self.hidden_channels
        if not self.use_grid_mlp:
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
        self.scalar_mlp: nn.Module | None = None
        self.grid_projector: S2GridProjector | None = None
        self.pointwise_grid_mlp: nn.Module | None = None
        if self.use_grid_mlp:
            self.scalar_mlp = nn.Sequential(
                ChannelLinear(
                    in_channels=self.channels,
                    out_channels=2 * self.hidden_channels,
                    dtype=dtype,
                    bias=self.mlp_bias,
                    trainable=trainable,
                    seed=child_seed(seed_act, 0),
                ),
                SwiGLU(),
            )
            self.grid_projector = S2GridProjector(
                lmax=self.lmax,
                mmax=self.lmax,
                dtype=dtype,
                grid_resolution_list=self.s2_grid_resolution,
                coefficient_layout="packed",
                grid_method=self.s2_grid_method,
            )
            self.pointwise_grid_mlp = PointwiseGridMLP(
                channels=self.hidden_channels,
                dtype=dtype,
                trainable=trainable,
                seed=child_seed(seed_act, 1),
            )
            self.act = nn.Identity()
        elif self.s2_activation:
            self.act = SwiGLUS2Activation(
                lmax=self.lmax,
                channels=self.hidden_channels,
                dtype=self.compute_dtype,
                n_focus=1,
                layout="ndfc",
                grid_resolution_list=self.s2_grid_resolution,
                coefficient_layout="packed",
                grid_method=self.s2_grid_method,
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
            in_channels=self.hidden_channels,
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
        x_input = x
        x = self.so3_linear_1(x)

        # === Step 2. Equivariant nonlinearity ===
        if self.use_grid_mlp:
            scalar_outputs = self.scalar_mlp(x_input.select(dim=1, index=0))
            x_flat, shape_info = self._flatten_grid_inputs(x)
            x_grid = self.grid_projector.to_grid(x_flat.to(dtype=self.dtype))
            x_grid = self.pointwise_grid_mlp(x_grid)
            x = self._restore_grid_outputs(
                self.grid_projector.from_grid(x_grid), shape_info
            )
            x[:, 0, :, :].add_(scalar_outputs)
        elif self.s2_activation:
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

    def _flatten_grid_inputs(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int, int]]:
        n_batch, coeff_dim, n_focus, _ = x.shape
        return (
            x.permute(0, 2, 1, 3).reshape(n_batch * n_focus, coeff_dim, x.shape[-1]),
            (n_batch, coeff_dim, n_focus),
        )

    def _restore_grid_outputs(
        self, x: torch.Tensor, shape_info: tuple[int, int, int]
    ) -> torch.Tensor:
        n_batch, coeff_dim, n_focus = shape_info
        return x.reshape(n_batch, n_focus, coeff_dim, self.hidden_channels).permute(
            0, 2, 1, 3
        )

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
                "grid_mlp": self.use_grid_mlp,
                "precision": RESERVED_PRECISION_DICT[self.dtype],
                "s2_activation": self.s2_activation,
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


class PointwiseGridMLP(nn.Module):
    """
    Apply a two-layer point-wise MLP on flattened S2 grid features.

    Parameters
    ----------
    channels
        Hidden feature dimension on the grid.
    dtype
        Parameter dtype.
    trainable
        Whether parameters are trainable.
    seed
        Random seed for weight initialization.
    """

    def __init__(
        self,
        *,
        channels: int,
        dtype: torch.dtype,
        trainable: bool,
        seed: int | list[int] | None = None,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.linear_1 = ChannelLinear(
            in_channels=self.channels,
            out_channels=2 * self.channels,
            dtype=dtype,
            bias=False,
            trainable=trainable,
            seed=child_seed(seed, 0),
        )
        self.act = SwiGLU()
        self.linear_2 = ChannelLinear(
            in_channels=self.channels,
            out_channels=self.channels,
            dtype=dtype,
            bias=False,
            trainable=trainable,
            seed=child_seed(seed, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the point-wise grid MLP."""
        x = self.act(self.linear_1(x))
        return self.linear_2(x)
