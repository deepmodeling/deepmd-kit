# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Activation and S2-grid helper modules for SeZM.

This module contains SeZM nonlinear operators, including GatedActivation,
point-wise SwiGLU, and the S2-grid projection helper used by the
S2 activation path.
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
import torch.nn.functional as F
from e3nn.o3 import (
    FromS2Grid,
    ToS2Grid,
    spherical_harmonics,
)

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
    build_l_major_index,
    build_m_major_index,
    build_m_major_l_index,
    map_degree_idx,
)
from .lebedev import (
    LEBEDEV_PRECISION_TO_NPOINTS,
    load_lebedev_rule,
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
        return F.silu(gate) * value


class S2GridProjector(nn.Module):
    """
    Project SO(3) coefficients to/from a flattened S2 grid.

    Parameters
    ----------
    lmax
        Maximum spherical harmonic degree.
    mmax
        Maximum order kept in the coefficient layout. If None, use ``lmax``.
    dtype
        Buffer dtype used by the projection matrices.
    grid_resolution_list
        Two-element resolution list. For ``grid_method='e3nn'`` it is
        ``[R_phi, R_theta]`` and is converted to the ``e3nn``
        ``(lat, long) = (R_theta, R_phi)`` ordering. For
        ``grid_method='lebedev'`` it is ``[precision, n_points]``.
    coefficient_layout
        Coefficient ordering expected by the caller:
        - ``"packed"``: packed ``(l, m)`` order, optionally truncated by ``mmax``.
        - ``"m_major"``: reduced m-major order used inside ``SO2Convolution``.
    grid_method
        S2 quadrature backend. Must be ``"e3nn"`` or ``"lebedev"``.
    """

    def __init__(
        self,
        *,
        lmax: int,
        mmax: int | None = None,
        dtype: torch.dtype,
        grid_resolution_list: list[int] | None = None,
        coefficient_layout: str = "packed",
        grid_method: str = "e3nn",
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.mmax = int(self.lmax if mmax is None else mmax)
        if self.mmax < 0:
            raise ValueError("`mmax` must be non-negative")
        if self.mmax > self.lmax:
            raise ValueError("`mmax` must be <= `lmax`")
        self.dtype = dtype
        self.device = env.DEVICE
        self.coefficient_layout = str(coefficient_layout).lower()
        if self.coefficient_layout not in {"packed", "m_major"}:
            raise ValueError(
                "`coefficient_layout` must be either 'packed' or 'm_major'"
            )
        self.grid_method = str(grid_method).lower()
        if self.grid_method not in {"e3nn", "lebedev"}:
            raise ValueError("`grid_method` must be either 'e3nn' or 'lebedev'")

        self.grid_resolution_list = _normalize_s2_grid_resolution(
            self.lmax,
            self.mmax,
            grid_resolution_list,
            method=self.grid_method,
        )
        if self.grid_method == "e3nn":
            self.phi_resolution, self.theta_resolution = self.grid_resolution_list
            self.lebedev_precision = 0
            self.lebedev_npoints = 0
        else:
            self.phi_resolution = 0
            self.theta_resolution = 0
            self.lebedev_precision, self.lebedev_npoints = self.grid_resolution_list

        coeff_index = self._build_coefficient_index(device=torch.device("cpu"))
        self.coeff_dim = int(coeff_index.numel())
        to_grid_mat, from_grid_mat = self._build_projection_mats(coeff_index)
        to_grid_mat = to_grid_mat.to(device=self.device, dtype=self.dtype)
        from_grid_mat = from_grid_mat.to(device=self.device, dtype=self.dtype)
        self.register_buffer("to_grid_mat", to_grid_mat, persistent=True)
        self.register_buffer("from_grid_mat", from_grid_mat, persistent=True)

    def _build_coefficient_index(self, device: torch.device) -> torch.Tensor:
        if self.coefficient_layout == "m_major":
            return build_m_major_index(self.lmax, self.mmax, device=device)
        if self.mmax == self.lmax:
            return torch.arange((self.lmax + 1) ** 2, device=device, dtype=torch.long)
        return build_l_major_index(self.lmax, self.mmax, device=device)

    def _rescale_truncated_orders(self, mat: torch.Tensor) -> None:
        if self.lmax == self.mmax:
            return
        for l in range(self.lmax + 1):
            if l <= self.mmax:
                continue
            start_idx = l * l
            length = 2 * l + 1
            rescale = math.sqrt(length / float(2 * self.mmax + 1))
            mat[:, :, start_idx : start_idx + length].mul_(rescale)

    def _rescale_truncated_matrix(self, mat: torch.Tensor) -> None:
        if self.lmax == self.mmax:
            return
        for l in range(self.lmax + 1):
            if l <= self.mmax:
                continue
            start_idx = l * l
            length = 2 * l + 1
            rescale = math.sqrt(length / float(2 * self.mmax + 1))
            mat[:, start_idx : start_idx + length].mul_(rescale)

    def _build_projection_mats(
        self, coeff_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.grid_method == "lebedev":
            return self._build_lebedev_projection_mats(coeff_index)
        return self._build_e3nn_projection_mats(coeff_index)

    def _build_e3nn_projection_mats(
        self, coeff_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.device("cpu"):
            to_grid = ToS2Grid(
                self.lmax,
                (self.theta_resolution, self.phi_resolution),
                normalization="component",
                device="cpu",
            )
            to_grid_mat = torch.einsum("mbi,am->bai", to_grid.shb, to_grid.sha).detach()
            self._rescale_truncated_orders(to_grid_mat)

            from_grid = FromS2Grid(
                (self.theta_resolution, self.phi_resolution),
                self.lmax,
                normalization="component",
                device="cpu",
            )
            from_grid_mat = torch.einsum(
                "am,mbi->bai", from_grid.sha, from_grid.shb
            ).detach()
            self._rescale_truncated_orders(from_grid_mat)

        to_grid_mat = to_grid_mat.flatten(0, 1).index_select(1, coeff_index)
        from_grid_mat = (
            from_grid_mat.flatten(0, 1).permute(1, 0).index_select(0, coeff_index)
        )
        return to_grid_mat, from_grid_mat

    def _build_lebedev_projection_mats(
        self, coeff_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.device("cpu"):
            points, weights = load_lebedev_rule(
                self.lebedev_precision,
                dtype=torch.float64,
                device=torch.device("cpu"),
            )
            harmonics = spherical_harmonics(
                list(range(self.lmax + 1)),
                points,
                normalize=True,
                normalization="norm",
            )
            # e3nn's ``norm`` harmonics are ``component / sqrt(2*l+1)``.
            # ``ToS2Grid(..., normalization="component")`` additionally divides
            # every degree block by ``sqrt(lmax+1)``; keep the same convention so
            # the Lebedev backend can replace the e3nn product-grid backend.
            scale = math.sqrt(float(self.lmax + 1))
            degree_factors = harmonics.new_tensor(
                [
                    float(2 * l + 1)
                    for l in range(self.lmax + 1)
                    for _ in range(2 * l + 1)
                ]
            )
            to_grid_mat = harmonics / scale
            # The packaged Lebedev weights sum to one. For ``norm`` harmonics,
            # ``sum_a w_a Y_j(a) Y_k(a) = delta_jk / (2*l+1)``; the
            # degree_factors and ``scale`` invert this normalization.
            from_grid_mat = harmonics * (
                weights[:, None] * scale * degree_factors[None, :]
            )
            self._rescale_truncated_matrix(to_grid_mat)
            self._rescale_truncated_matrix(from_grid_mat)

        to_grid_mat = to_grid_mat.index_select(1, coeff_index)
        from_grid_mat = from_grid_mat.index_select(1, coeff_index).transpose(0, 1)
        return to_grid_mat, from_grid_mat

    def to_grid(self, embedding: torch.Tensor) -> torch.Tensor:
        """Project coefficients ``(N, D, C)`` to a flattened grid ``(N, A, C)``."""
        return torch.einsum("aj,njc->nac", self.to_grid_mat, embedding)

    def from_grid(self, grid: torch.Tensor) -> torch.Tensor:
        """Project a flattened grid ``(N, A, C)`` back to coefficients ``(N, D, C)``."""
        return torch.einsum("ja,nac->njc", self.from_grid_mat, grid)

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "S2GridProjector",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "mmax": self.mmax,
                "precision": RESERVED_PRECISION_DICT[self.dtype],
                "grid_resolution_list": self.grid_resolution_list,
                "coefficient_layout": self.coefficient_layout,
                "grid_method": self.grid_method,
            },
            "@variables": {},
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> S2GridProjector:
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "S2GridProjector":
            raise ValueError(f"Invalid class for S2GridProjector: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        config = data.pop("config")
        data.pop("@variables", None)
        precision = config.pop("precision")
        config["dtype"] = PRECISION_DICT[precision]
        return cls(**config)


class SwiGLUS2Activation(nn.Module):
    """
    Apply the merged scalar/grid SwiGLU-S2 activation to SO(3) coefficients.

    The degree-0 slice provides two scalar paths:

    - a scalar ``SwiGLU`` branch that is merged back into the output ``l=0`` part
    - a learned sigmoid gate that modulates the full output reconstructed from
      the S2 grid path

    The equivariant branch projects the full ``2 * channels`` coefficients to the
    S2 grid, multiplies the two channel halves point-wise on the grid, projects
    back to coefficients, and applies the scalar sigmoid gate.

    Parameters
    ----------
    lmax
        Maximum spherical harmonic degree.
    mmax
        Maximum order kept in the coefficient layout. If None, use ``lmax``.
    channels
        Output channel count after SwiGLU. The input is expected to have
        ``2 * channels`` on the last axis.
    dtype
        Projection buffer dtype.
    n_focus
        Number of focus streams in the input layout.
    layout
        Tensor layout convention:
        - ``"ndfc"`` for ``(N, D, F, C)``
        - ``"nfdc"`` for ``(N, F, D, C)``
    grid_resolution_list
        Two-element list ``[R_phi, R_theta]``.
    coefficient_layout
        Coefficient ordering: ``"packed"`` or ``"m_major"``.
    grid_method
        S2 quadrature backend. Must be ``"e3nn"`` or ``"lebedev"``.
    mlp_bias
        Whether the scalar sigmoid projection uses bias.
    trainable
        Whether parameters are trainable.
    seed
        Random seed for the scalar sigmoid projection.
    """

    def __init__(
        self,
        *,
        lmax: int,
        mmax: int | None = None,
        channels: int,
        dtype: torch.dtype,
        n_focus: int = 1,
        layout: str = "ndfc",
        grid_resolution_list: list[int] | None = None,
        coefficient_layout: str = "packed",
        grid_method: str = "e3nn",
        mlp_bias: bool = False,
        trainable: bool,
        seed: int | list[int] | None = None,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.mmax = int(self.lmax if mmax is None else mmax)
        self.channels = int(channels)
        self.dtype = dtype
        self.n_focus = int(n_focus)
        self.mlp_bias = bool(mlp_bias)
        self.layout = str(layout).lower()
        if self.layout not in {"ndfc", "nfdc"}:
            raise ValueError("`layout` must be either 'ndfc' or 'nfdc'")
        self.coefficient_layout = str(coefficient_layout).lower()
        self.grid_method = str(grid_method).lower()
        self.grid_resolution_list = _normalize_s2_grid_resolution(
            self.lmax,
            self.mmax,
            grid_resolution_list,
            method=self.grid_method,
        )
        self.scalar_act = SwiGLU()
        self.scalar_gate = FocusLinear(
            in_channels=2 * self.channels,
            out_channels=self.channels,
            n_focus=self.n_focus,
            dtype=self.dtype,
            bias=self.mlp_bias,
            trainable=trainable,
            seed=child_seed(seed, 0),
            init_std=0.01,
        )
        self.projector: S2GridProjector | None
        if self.lmax == 0:
            self.projector = None
            self.coeff_dim = 1
        else:
            self.projector = S2GridProjector(
                lmax=self.lmax,
                mmax=self.mmax,
                dtype=self.dtype,
                grid_resolution_list=self.grid_resolution_list,
                coefficient_layout=self.coefficient_layout,
                grid_method=self.grid_method,
            )
            self.coeff_dim = self.projector.coeff_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Input tensor with last dimension ``2 * channels``.

        Returns
        -------
        torch.Tensor
            Activated tensor with the same coefficient layout and ``channels`` on
            the last axis.
        """
        input_dtype = x.dtype
        # Promote before slicing to avoid the TorchInductor AMP compile bug on
        # the scalar SwiGLU branch in PyTorch 2.11.
        scalar_inputs = self._extract_scalar_inputs(x.to(dtype=self.dtype))
        scalar_outputs = self.scalar_act(scalar_inputs)

        if self.projector is None:
            return self._restore_scalar_outputs(scalar_outputs.to(dtype=input_dtype))

        gate_scalars = torch.sigmoid(self.scalar_gate(scalar_inputs))
        x_flat, shape_info = self._flatten_inputs(x)
        x_grid = self.projector.to_grid(x_flat.to(dtype=self.dtype))
        x_grid_1, x_grid_2 = torch.chunk(x_grid, chunks=2, dim=-1)
        out_flat = self.projector.from_grid(x_grid_1 * x_grid_2)
        outputs = self._restore_outputs(out_flat, shape_info)
        outputs = outputs * self._broadcast_scalar_gate(gate_scalars)
        self._merge_scalar_outputs(outputs, scalar_outputs)
        return outputs.to(dtype=input_dtype)

    def _extract_scalar_inputs(self, x: torch.Tensor) -> torch.Tensor:
        if self.layout == "ndfc":
            return x.select(dim=1, index=0)
        return x.select(dim=2, index=0)

    def _broadcast_scalar_gate(self, gate_scalars: torch.Tensor) -> torch.Tensor:
        if self.layout == "ndfc":
            return gate_scalars.unsqueeze(1)
        return gate_scalars.unsqueeze(2)

    def _restore_scalar_outputs(self, scalar_outputs: torch.Tensor) -> torch.Tensor:
        if self.layout == "ndfc":
            return scalar_outputs.unsqueeze(1)
        return scalar_outputs.unsqueeze(2)

    def _flatten_inputs(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int, int]]:
        if self.layout == "ndfc":
            n_batch, coeff_dim, n_focus, _ = x.shape
            return (
                x.permute(0, 2, 1, 3).reshape(
                    n_batch * n_focus, coeff_dim, x.shape[-1]
                ),
                (n_batch, coeff_dim, n_focus),
            )
        n_batch, n_focus, coeff_dim, _ = x.shape
        return (
            x.reshape(n_batch * n_focus, coeff_dim, x.shape[-1]),
            (n_batch, coeff_dim, n_focus),
        )

    def _restore_outputs(
        self, x: torch.Tensor, shape_info: tuple[int, int, int]
    ) -> torch.Tensor:
        n_batch, coeff_dim, n_focus = shape_info
        if self.layout == "ndfc":
            return x.reshape(n_batch, n_focus, coeff_dim, self.channels).permute(
                0, 2, 1, 3
            )
        return x.reshape(n_batch, n_focus, coeff_dim, self.channels)

    def _merge_scalar_outputs(
        self, outputs: torch.Tensor, scalar_outputs: torch.Tensor
    ) -> None:
        if self.layout == "ndfc":
            outputs[:, 0, :, :].add_(scalar_outputs)
        else:
            outputs[:, :, 0, :].add_(scalar_outputs)

    def serialize(self) -> dict[str, Any]:
        trainable = all(p.requires_grad for p in self.parameters())
        state = self.state_dict()
        return {
            "@class": "SwiGLUS2Activation",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "mmax": self.mmax,
                "channels": self.channels,
                "precision": RESERVED_PRECISION_DICT[self.dtype],
                "n_focus": self.n_focus,
                "layout": self.layout,
                "grid_resolution_list": self.grid_resolution_list,
                "coefficient_layout": self.coefficient_layout,
                "grid_method": self.grid_method,
                "mlp_bias": self.mlp_bias,
                "trainable": trainable,
                "seed": None,
            },
            "@variables": {key: np_safe(value) for key, value in state.items()},
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> SwiGLUS2Activation:
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "SwiGLUS2Activation":
            raise ValueError(f"Invalid class for SwiGLUS2Activation: {data_cls}")
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


def resolve_s2_grid_resolution(
    lmax: int,
    mmax: int,
    *,
    method: str = "e3nn",
) -> list[int]:
    """
    Resolve the default S2 grid resolution.

    For ``method='e3nn'``, the automatic default uses even azimuthal sampling
    ``R_phi = 2 * mmax + 4`` and even polar sampling
    ``R_theta = ceil_even(3 * lmax + 2)``.

    For ``method='lebedev'``, the automatic default picks the smallest packaged
    Lebedev rule whose algebraic precision is at least ``3 * lmax`` and returns
    ``[precision, n_points]``.
    """
    method = str(method).lower()
    if method not in {"e3nn", "lebedev"}:
        raise ValueError("`method` must be either 'e3nn' or 'lebedev'")
    if method == "lebedev":
        required_precision = 3 * int(lmax)
        for precision, n_points in LEBEDEV_PRECISION_TO_NPOINTS.items():
            if precision >= required_precision:
                return [precision, n_points]
        raise ValueError(
            f"No packaged Lebedev rule has precision >= {required_precision}"
        )

    phi_resolution = 2 * mmax + 4
    theta_resolution = 3 * lmax + 2
    theta_resolution += theta_resolution % 2
    return [phi_resolution, theta_resolution]


def _normalize_s2_grid_resolution(
    lmax: int,
    mmax: int,
    grid_resolution_list: list[int] | None,
    *,
    method: str,
) -> list[int]:
    """Resolve default grids or validate already-resolved low-level grids."""
    method = str(method).lower()
    if grid_resolution_list is None:
        return resolve_s2_grid_resolution(lmax, mmax, method=method)
    if method == "lebedev":
        if len(grid_resolution_list) != 2:
            raise ValueError(
                "Lebedev `grid_resolution_list` must be [precision, n_points]"
            )
        precision = int(grid_resolution_list[0])
        n_points = int(grid_resolution_list[1])
        expected_n_points = LEBEDEV_PRECISION_TO_NPOINTS.get(precision)
        if expected_n_points != n_points:
            raise ValueError(
                "Lebedev `grid_resolution_list` must match a packaged "
                f"[precision, n_points] pair; got [{precision}, {n_points}]"
            )
        return [precision, n_points]

    if len(grid_resolution_list) != 2:
        raise ValueError("`grid_resolution_list` must contain two integers")
    resolution = [int(grid_resolution_list[0]), int(grid_resolution_list[1])]
    if resolution[0] < 1 or resolution[1] < 1:
        raise ValueError("grid resolutions must be positive")
    return resolution
