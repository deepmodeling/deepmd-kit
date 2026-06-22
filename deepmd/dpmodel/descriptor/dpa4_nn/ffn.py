# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Equivariant feed-forward layers for DPA4/SeZM.

This module is the dpmodel port of ``deepmd.pt.model.descriptor.sezm_nn.ffn``.
It defines the full SO(3)-equivariant feed-forward network used inside SeZM
interaction blocks.
"""

from __future__ import (
    annotations,
)

from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
    NativeOP,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
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
from .so2 import (
    _compute_precision,
)
from .so3 import (
    SO3Linear,
)


class EquivariantFFN(NativeOP):
    """
    Full equivariant FFN operating on all spherical harmonic degrees.

    Default structure (glu_activation=False):
        SO3 linear (in -> hidden) -> GatedActivation -> SO3 linear (hidden -> out)

    Default structure (glu_activation=True):
        SO3 linear (in -> 2*hidden) -> split -> GatedActivation(val, gate) -> SO3 linear (hidden -> out)

    Optional grid-FFN structure (s2_activation=True):
        SO3 linear (in -> 2*hidden)
        -> project packed SO(3) coefficients to the S2 grid
        -> grid GLU or scalar-routed polynomial branch on hidden features
        -> project grid features back to packed SO(3) coefficients
        -> add scalar LinearSwiGLU branch to l=0
        -> SO3 linear (hidden -> out)

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
        If True, select the polynomial grid MLP operation (``op_type='mlp'``)
        when the block-internal FFN grid path is enabled. ``grid_branch`` takes
        precedence when positive.
    grid_branch
        Number of scalar-routed polynomial product branches used when the
        block-internal FFN grid path is enabled. ``0`` disables this branch
        mixer. Positive values take precedence over ``grid_mlp``.
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
        (gate linear bias).
    precision
        Parameter precision.
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
        s2_activation: bool = False,
        ffn_so3_grid: bool = False,
        lebedev_quadrature: bool = False,
        activation_function: str = "silu",
        glu_activation: bool = True,
        mlp_bias: bool = False,
        precision: str = DEFAULT_PRECISION,
        trainable: bool = True,
        seed: int | list[int] | None = None,
    ) -> None:
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
        self.activation_function = str(activation_function)
        self.glu_activation = bool(glu_activation)
        self.mlp_bias = bool(mlp_bias)
        self.precision = precision
        self.compute_precision = _compute_precision(precision)
        self.trainable = bool(trainable)
        self.grid_n_frames = 2 * self.kmax + 1 if self.ffn_so3_grid else 1

        # === Step 0. Split deterministic seeds at the module top-level ===
        seed_so3_in = child_seed(seed, 0)
        seed_act = child_seed(seed, 1)
        seed_so3_out = child_seed(seed, 2)

        # === First SO3Linear for channel mixing ===
        self.use_grid_net = self.s2_activation or self.ffn_so3_grid
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
            precision=self.precision,
            mlp_bias=self.mlp_bias,
            trainable=self.trainable,
            seed=seed_so3_in,
        )

        # === Equivariant nonlinearity path ===
        if self.use_grid_net:
            grid_op = (
                "branch"
                if self.use_grid_branch
                else ("mlp" if self.use_grid_mlp else "glu")
            )
            self.act: NativeOP
            if self.ffn_so3_grid:
                self.act = SO3GridNet(
                    lmax=self.lmax,
                    kmax=self.kmax,
                    channels=self.hidden_channels,
                    n_focus=1,
                    mode="self",
                    op_type=grid_op,
                    precision=self.compute_precision,
                    layout="ndfc",
                    grid_branches=max(1, self.grid_branch),
                    mlp_bias=self.mlp_bias,
                    trainable=self.trainable,
                    seed=seed_act,
                )
            else:
                self.act = S2GridNet(
                    lmax=self.lmax,
                    channels=self.hidden_channels,
                    n_focus=1,
                    mode="self",
                    op_type=grid_op,
                    precision=self.compute_precision,
                    layout="ndfc",
                    grid_resolution_list=self.s2_grid_resolution,
                    coefficient_layout="packed",
                    grid_method=self.s2_grid_method,
                    grid_branches=max(1, self.grid_branch),
                    mlp_bias=self.mlp_bias,
                    trainable=self.trainable,
                    seed=seed_act,
                )
        else:
            self.act = GatedActivation(
                lmax=self.lmax,
                channels=self.hidden_channels,
                precision=self.compute_precision,
                activation_function=self.activation_function,
                mlp_bias=self.mlp_bias,
                layout="ndfc",
                trainable=self.trainable,
                seed=seed_act,
            )

        # === Second SO3Linear for channel mixing ===
        # Zero-initialized so residual path starts near-identity.
        self.so3_linear_2 = SO3Linear(
            lmax=self.lmax,
            in_channels=self.grid_n_frames * self.hidden_channels,
            out_channels=self.channels,
            n_focus=1,
            precision=self.precision,
            mlp_bias=self.mlp_bias,
            trainable=self.trainable,
            seed=seed_so3_out,
            init_std=0.0,
        )

    def call(self, x: Any) -> Any:
        """
        Parameters
        ----------
        x
            Input with shape (N, D, F, C) where D=(lmax+1)^2.

        Returns
        -------
        Array
            Output with shape (N, D, F, C).
        """
        # === Step 1. Input up projection ===
        x = self.so3_linear_1(x)

        # === Step 2. Equivariant nonlinearity ===
        if self.use_grid_net:
            x = self.act(x)
        elif self.glu_activation:
            # Split into value and gate branches along channel dimension
            # (pt uses x.chunk(2, dim=-1); slicing is array-API portable)
            x_val = x[..., : self.hidden_channels]
            x_gate = x[..., self.hidden_channels :]
            # Pass gate to GatedActivation for GLU-style gating
            x = self.act(x_val, gate=x_gate)
        else:
            x = self.act(x)

        # === Step 3. Per-degree output projection ===
        x = self.so3_linear_2(x)

        return x

    def _sub_modules(self) -> list[tuple[str, NativeOP]]:
        """Sub-modules with their pt module names."""
        return [
            ("so3_linear_1", self.so3_linear_1),
            ("act", self.act),
            ("so3_linear_2", self.so3_linear_2),
        ]

    def _variables(self) -> dict[str, Any]:
        """Variables keyed by the pt ``state_dict`` key names."""
        variables: dict[str, Any] = {}
        for prefix, sub in self._sub_modules():
            for key, value in sub.serialize()["@variables"].items():
                variables[f"{prefix}.{key}"] = value
        return variables

    def _load_variables(self, variables: dict[str, Any]) -> None:
        """Load variables keyed by the pt ``state_dict`` key names."""
        variables = dict(variables)
        for attr, sub in self._sub_modules():
            full = f"{attr}."
            sv = {
                key[len(full) :]: value
                for key, value in variables.items()
                if key.startswith(full)
            }
            for key in list(variables):
                if key.startswith(full):
                    del variables[key]
            if not sv:
                raise KeyError(f"Missing variables with prefix: {full}")
            # rebuild the sub-module through its own (shape-checking)
            # deserialize, reusing its serialized config
            data = sub.serialize()
            data["@variables"] = sv
            setattr(self, attr, type(sub).deserialize(data))
        if variables:
            raise KeyError(f"Unknown variables: {sorted(variables)}")

    def serialize(self) -> dict[str, Any]:
        """Serialize the EquivariantFFN to a dict (pt-compatible format)."""
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
                "precision": np.dtype(PRECISION_DICT[self.precision]).name,
                "s2_activation": self.s2_activation,
                "ffn_so3_grid": self.ffn_so3_grid,
                "lebedev_quadrature": self.lebedev_quadrature,
                "activation_function": self.activation_function,
                "glu_activation": self.glu_activation,
                "mlp_bias": self.mlp_bias,
                "trainable": self.trainable,
                "seed": None,
            },
            "@variables": self._variables(),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> EquivariantFFN:
        """Deserialize an EquivariantFFN from a dict."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "EquivariantFFN":
            raise ValueError(f"Invalid class for EquivariantFFN: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        config = dict(data.pop("config"))
        variables = data.pop("@variables")
        config["precision"] = str(config.pop("precision"))
        obj = cls(**config)
        obj._load_variables(variables)
        return obj
