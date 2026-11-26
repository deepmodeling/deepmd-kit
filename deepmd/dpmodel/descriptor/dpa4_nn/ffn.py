# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Equivariant feed-forward layers for DPA4/SeZM.

This module defines the full SO(3)-equivariant feed-forward network used
inside SeZM interaction blocks and the descriptor output head.

This module is the dpmodel (array-API) port of
``deepmd.pt.model.descriptor.sezm_nn.ffn``.
"""

from __future__ import (
    annotations,
)

from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

from deepmd.dpmodel import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
    NativeOP,
)
from deepmd.dpmodel.utils.expert_linear import (
    ExpertContainer,
    linear_with_experts,
    map_args,
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
from .so3 import (
    SO3Linear,
)
from .utils import (
    get_promoted_dtype,
)

if TYPE_CHECKING:
    from collections.abc import (
        Sequence,
    )


class EquivariantFFN(ExpertContainer, NativeOP):
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

    With ``experts`` every learnable map of the network (the two SO3 linear
    maps and the maps inside the nonlinearity) is an expert map (see
    :class:`~deepmd.dpmodel.utils.expert_linear.ExpertWeights`): the route
    passed to :meth:`call` gives every row the increment of the task, the
    increment of its element and the coefficients of the routed experts, so
    every condition selects a complete set of network weights. The network is
    one unit of the route: its rows enter the layout of the route once and
    every map works on that layout.

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
    precision
        Parameter precision.
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
    experts
        Number of experts and of element increments ``(n_experts, n_elements)``
        of every learnable map; None builds a plain network.
    trainable
        Whether parameters are trainable.
    seed
        Random seed for weight initialization.
    """

    def __init__(
        self,
        *,
        lmax: int,
        parity: bool = False,
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
        experts: Sequence[int] | None = None,
        trainable: bool = True,
        seed: int | list[int] | None = None,
    ) -> None:
        self.parity = bool(parity)
        self.lmax = int(lmax)
        self.experts = None if experts is None else [int(n) for n in experts]
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
        self.compute_precision = np.dtype(
            get_promoted_dtype(PRECISION_DICT[precision])
        ).name
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
        self.so3_linear_1 = linear_with_experts(
            SO3Linear,
            self.experts,
            lmax=self.lmax,
            in_channels=self.channels,
            out_channels=linear1_out_channels,
            n_focus=1,
            precision=self.precision,
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
                    parity=self.parity,
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
                    experts=self.experts,
                    trainable=trainable,
                    seed=seed_act,
                )
            else:
                self.act = S2GridNet(
                    parity=self.parity,
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
                    experts=self.experts,
                    trainable=trainable,
                    seed=seed_act,
                )
        else:
            self.act = GatedActivation(
                lmax=self.lmax,
                channels=self.hidden_channels,
                precision=self.compute_precision,
                activation_function=activation_function,
                mlp_bias=self.mlp_bias,
                layout="ndfc",
                experts=self.experts,
                trainable=trainable,
                seed=seed_act,
            )

        # === Second SO3Linear for channel mixing ===
        # Zero-initialized so residual path starts near-identity.
        self.so3_linear_2 = linear_with_experts(
            SO3Linear,
            self.experts,
            lmax=self.lmax,
            in_channels=self.grid_n_frames * self.hidden_channels,
            out_channels=self.channels,
            n_focus=1,
            precision=self.precision,
            mlp_bias=self.mlp_bias,
            trainable=trainable,
            seed=seed_so3_out,
            init_std=0.0,
        )

        self.trainable = bool(trainable)

    def _expert_members(self) -> list[str]:
        return ["so3_linear_1", "act", "so3_linear_2"]

    def call(self, x: Any, route: Any = None) -> Any:
        """
        Parameters
        ----------
        x
            Input with shape (N, D, F, C) where D=(lmax+1)^2.
        route
            Route of the rows of ``x`` for this unit (see
            :class:`~deepmd.dpmodel.utils.expert_route.UnitRoute`), with the
            weights of :meth:`expert_maps`; required with ``experts``.

        Returns
        -------
        Array
            Output with shape (N, D, F, C).
        """
        return self._forward(x, route, scalar_only=False)

    def call_scalar(self, x: Any, route: Any = None) -> Any:
        """Evaluate the FFN for the ``l=0`` output only.

        Parameters
        ----------
        x : Array
            Input with shape ``(N, D, F, C)``.
        route
            Route of the rows of ``x`` for this unit; required with
            ``experts``.

        Returns
        -------
        Array
            Scalar output with shape ``(N, 1, F, C)``.
        """
        return self._forward(x, route, scalar_only=True)

    def _forward(self, x: Any, route: Any, *, scalar_only: bool) -> Any:
        """Run the network on the rows of its route, for all or the scalar output."""
        if route is not None:
            x = route.enter(x)
        hidden = self._activate_hidden(x, route, scalar_only=scalar_only)

        # === Step 3. Per-degree output projection ===
        project = self.so3_linear_2.call_scalar if scalar_only else self.so3_linear_2
        out = project(hidden, *map_args(route, -1))
        return out if route is None else route.leave(out)

    def _activate_hidden(
        self,
        x: Any,
        route: Any = None,
        *,
        scalar_only: bool,
    ) -> Any:
        """Apply the input projection and equivariant nonlinearity."""
        # === Step 1. Input up projection ===
        x = self.so3_linear_1(x, *map_args(route, 0))

        # === Step 2. Equivariant nonlinearity ===
        # The maps of the nonlinearity lie between the two projections.
        act_route = None if route is None else route.maps(1, -1)
        if self.use_grid_net:
            x = (
                self.act.call_scalar(x, route=act_route)
                if scalar_only
                else self.act(x, route=act_route)
            )
        elif self.glu_activation:
            # Split into value and gate branches along channel dimension
            nc = (x.shape[-1] + 1) // 2
            x_val = x[..., :nc]
            x_gate = x[..., nc:]
            # Pass gate to GatedActivation for GLU-style gating
            x = self.act(x_val, gate=x_gate, route=act_route)
        else:
            x = self.act(x, route=act_route)

        if scalar_only and not self.use_grid_net:
            x = x[:, 0:1, :, :]
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
        for attr, sub in self._sub_modules():
            prefix = f"{attr}."
            sub_variables = {
                key[len(prefix) :]: value
                for key, value in variables.items()
                if key.startswith(prefix)
            }
            # Rebuild each sub-module via its own deserialize, reusing the freshly
            # serialized config and injecting the loaded @variables.
            data = sub.serialize()
            data["@variables"] = sub_variables
            setattr(self, attr, type(sub).deserialize(data))

    def serialize(self) -> dict[str, Any]:
        """Serialize the EquivariantFFN to a dict."""
        return {
            "@class": "EquivariantFFN",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "parity": self.parity,
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
                "experts": self.experts,
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
