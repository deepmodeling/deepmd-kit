# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Grid-space nonlinearities for SeZM coefficient tensors.

A grid net receives coefficient tensors, converts them to quadrature values,
applies one point-wise grid operation, and projects the result back to
coefficients.  The public shapes are:

* ``mode='self'``: one input ``(N, D, F, 2*C)`` or ``(N, F, D, 2*C)``.
* ``mode='cross'``: query and context inputs with separate ``C`` channels.
* grid values: ``(N, G, F, C)`` after S2 or SO3 projection.

The only nonlinear scalar functions are SwiGLU, sigmoid, and softmax on the
``l=0`` scalar branch.  Non-scalar grid values use channel-linear maps and
point-wise products so equivariance is governed by the projector quadrature.
"""

from __future__ import (
    annotations,
)

from typing import (
    Literal,
)

import torch
import torch.nn as nn

from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.utils import (
    get_generator,
)

from .activation import (
    SwiGLU,
)
from .indexing import (
    build_l_major_index,
    build_m_major_l_index,
    map_degree_idx,
)
from .projection import (
    BaseGridProjector,
    S2GridProjector,
    SO3GridProjector,
)
from .so3 import (
    ChannelLinear,
    FocusLinear,
)

GridNetLayout = Literal["ndfc", "nfdc", "flat"]
GridNetMode = Literal["self", "cross"]
GridNetOp = Literal["glu", "mlp", "branch"]


def _build_frame_degree_index(
    *,
    lmax: int,
    mmax: int,
    coefficient_layout: str,
) -> torch.Tensor:
    """Build the per-coefficient degree index used by frame channel mixers."""
    coefficient_layout = str(coefficient_layout).lower()
    if coefficient_layout == "m_major":
        return build_m_major_l_index(lmax, mmax, device=env.DEVICE)
    if coefficient_layout == "packed":
        degree_index = map_degree_idx(lmax, device=env.DEVICE)
        if int(mmax) == int(lmax):
            return degree_index
        coeff_index = build_l_major_index(lmax, mmax, device=env.DEVICE)
        return degree_index.index_select(0, coeff_index)
    raise ValueError("`coefficient_layout` must be either 'packed' or 'm_major'")


class GridMLP(nn.Module):
    """Polynomial point-wise MLP applied independently at every grid point."""

    def __init__(
        self,
        *,
        channels: int,
        mode: GridNetMode,
        dtype: torch.dtype,
        trainable: bool,
        seed: int | list[int] | None = None,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.mode = str(mode).lower()
        if self.mode not in {"self", "cross"}:
            raise ValueError("`mode` must be either 'self' or 'cross'")
        self.input_channels = (
            2 * self.channels if self.mode == "self" else self.channels
        )
        self.hidden_channels = 2 * self.channels
        self.left_proj = ChannelLinear(
            in_channels=self.input_channels,
            out_channels=self.hidden_channels,
            dtype=dtype,
            bias=False,
            trainable=trainable,
            seed=child_seed(seed, 0),
        )
        self.right_proj = ChannelLinear(
            in_channels=self.input_channels,
            out_channels=self.hidden_channels,
            dtype=dtype,
            bias=False,
            trainable=trainable,
            seed=child_seed(seed, 1),
        )
        self.out_proj = ChannelLinear(
            in_channels=self.hidden_channels,
            out_channels=self.channels,
            dtype=dtype,
            bias=False,
            trainable=trainable,
            seed=child_seed(seed, 2),
        )

    def forward(
        self, query_grid: torch.Tensor, context_grid: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply the point-wise polynomial MLP to ``(N, G, F, C)`` grid fields.

        In self mode, both projections see ``concat(query_grid, context_grid)``
        and can form self and cross quadratic channel terms.  In cross mode,
        the query and context roles stay separate:
        ``(W_q query_grid) * (W_c context_grid)``.
        """
        if self.mode == "self":
            grid = torch.cat([query_grid, context_grid], dim=-1)
            left = self.left_proj(grid)
            right = self.right_proj(grid)
        else:
            left = self.left_proj(query_grid)
            right = self.right_proj(context_grid)
        return self.out_proj(left * right)


class GridBranch(nn.Module):
    """
    Scalar-routed polynomial mixer over grid product branches.

    The softmax sees only invariant scalar inputs.  Each branch is a
    quadratic product of grid fields, so rotations only act through the grid
    argument and the operation remains as band-limited as the product path.
    """

    def __init__(
        self,
        *,
        channels: int,
        n_branches: int,
        dtype: torch.dtype,
        trainable: bool,
        seed: int | list[int] | None = None,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.n_branches = int(n_branches)
        if self.n_branches < 1:
            raise ValueError("`n_branches` must be positive")
        self.left_proj = ChannelLinear(
            in_channels=self.channels,
            out_channels=self.n_branches * self.channels,
            dtype=dtype,
            bias=False,
            trainable=trainable,
            seed=child_seed(seed, 0),
        )
        self.right_proj = ChannelLinear(
            in_channels=self.channels,
            out_channels=self.n_branches * self.channels,
            dtype=dtype,
            bias=False,
            trainable=trainable,
            seed=child_seed(seed, 1),
        )
        self.router = ChannelLinear(
            in_channels=2 * self.channels,
            out_channels=self.n_branches,
            dtype=dtype,
            bias=False,
            trainable=trainable,
            seed=child_seed(seed, 2),
        )
        self.out_proj = ChannelLinear(
            in_channels=self.channels,
            out_channels=self.channels,
            dtype=dtype,
            bias=False,
            trainable=trainable,
            seed=child_seed(seed, 3),
        )

    def forward(
        self,
        query_grid: torch.Tensor,
        context_grid: torch.Tensor,
        scalar_pair: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply scalar-routed grid branch mixing.

        Parameters
        ----------
        query_grid
            First grid source with shape ``(N, G, F, C)``.
        context_grid
            Second grid source with shape ``(N, G, F, C)``.
        scalar_pair
            Invariant router source with shape ``(N, F, 2*C)``.
        """
        n_batch, n_grid, n_focus, _ = query_grid.shape
        left = self.left_proj(query_grid)
        right = self.right_proj(context_grid)
        value = (left * right).reshape(
            n_batch,
            n_grid,
            n_focus,
            self.n_branches,
            self.channels,
        )  # (N, G, F, N_branches, C)
        router = torch.softmax(self.router(scalar_pair), dim=-1)  # (N, F, N_branches)
        out = torch.einsum("ngfhc,nfh->ngfc", value, router)  # (N, G, F, C)
        return self.out_proj(out)


class FrameContract(nn.Module):
    """Per-degree frame/channel contraction that preserves the order index."""

    def __init__(
        self,
        *,
        lmax: int,
        mmax: int,
        coefficient_layout: str,
        n_frames: int,
        channels: int,
        dtype: torch.dtype,
        trainable: bool,
        seed: int | list[int] | None = None,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.mmax = int(mmax)
        self.coefficient_layout = str(coefficient_layout).lower()
        self.n_frames = int(n_frames)
        self.channels = int(channels)
        degree_index = _build_frame_degree_index(
            lmax=self.lmax,
            mmax=self.mmax,
            coefficient_layout=self.coefficient_layout,
        )
        self.register_buffer("degree_index", degree_index, persistent=False)
        self.weight = nn.Parameter(
            torch.empty(
                self.lmax + 1,
                self.n_frames * self.channels,
                self.channels,
                dtype=dtype,
                device=env.DEVICE,
            )
        )
        bound = 1.0 / (self.n_frames * self.channels) ** 0.5
        nn.init.uniform_(self.weight, -bound, bound, generator=get_generator(seed))
        for param in self.parameters():
            param.requires_grad = trainable

    def forward(self, coeff: torch.Tensor) -> torch.Tensor:
        """Contract ``(N, D, F, K*C)`` frame coefficients to ``(N, D, F, C)``."""
        weight = self.weight.index_select(0, self.degree_index)
        return torch.einsum("ndfi,dio->ndfo", coeff, weight)


class FrameExpand(nn.Module):
    """Per-degree frame/channel expansion that preserves the order index."""

    def __init__(
        self,
        *,
        lmax: int,
        mmax: int,
        coefficient_layout: str,
        n_frames: int,
        channels: int,
        dtype: torch.dtype,
        trainable: bool,
        seed: int | list[int] | None = None,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.mmax = int(mmax)
        self.coefficient_layout = str(coefficient_layout).lower()
        self.n_frames = int(n_frames)
        self.channels = int(channels)
        degree_index = _build_frame_degree_index(
            lmax=self.lmax,
            mmax=self.mmax,
            coefficient_layout=self.coefficient_layout,
        )
        self.register_buffer("degree_index", degree_index, persistent=False)
        self.weight = nn.Parameter(
            torch.empty(
                self.lmax + 1,
                self.channels,
                self.n_frames * self.channels,
                dtype=dtype,
                device=env.DEVICE,
            )
        )
        bound = 1.0 / self.channels**0.5
        nn.init.uniform_(self.weight, -bound, bound, generator=get_generator(seed))
        for param in self.parameters():
            param.requires_grad = trainable

    def forward(self, coeff: torch.Tensor) -> torch.Tensor:
        """Expand ``(N, D, F, C)`` coefficients to ``(N, D, F, K*C)``."""
        weight = self.weight.index_select(0, self.degree_index)
        return torch.einsum("ndfi,dio->ndfo", coeff, weight)


class BaseGridNet(nn.Module):
    """
    Shared implementation for S2 and SO(3) grid nets.

    ``mode='self'`` expects one input whose last channel axis contains two
    branches.  ``mode='cross'`` expects query and context inputs; the query side
    is the source of attention queries and SwiGLU gates, while the context side
    is the key/value or second product branch.
    """

    def __init__(
        self,
        *,
        projector: BaseGridProjector,
        channels: int,
        n_focus: int,
        mode: GridNetMode,
        op_type: GridNetOp,
        dtype: torch.dtype,
        layout: GridNetLayout,
        mlp_bias: bool,
        trainable: bool,
        grid_branches: int = 1,
        frame_expand: nn.Module | None = None,
        frame_contract: nn.Module | None = None,
        residual_scale_init: float | None = None,
        seed: int | list[int] | None = None,
    ) -> None:
        super().__init__()
        self.projector = projector.to(device=env.DEVICE)
        self.lmax = int(projector.lmax)
        self.channels = int(channels)
        self.n_focus = int(n_focus)
        self.n_frames = int(projector.n_frames)
        self.mode = str(mode).lower()
        if self.mode not in {"self", "cross"}:
            raise ValueError("`mode` must be either 'self' or 'cross'")
        self.op_type = str(op_type).lower()
        if self.op_type not in {"glu", "mlp", "branch"}:
            raise ValueError("`op_type` must be one of 'glu', 'mlp', or 'branch'")
        self.dtype = dtype
        self.layout = str(layout).lower()
        if self.layout not in {"ndfc", "nfdc", "flat"}:
            raise ValueError("`layout` must be one of 'ndfc', 'nfdc', or 'flat'")
        if self.mode == "self" and self.layout == "flat":
            raise ValueError("`layout='flat'` is only supported for cross grid nets")
        self.mlp_bias = bool(mlp_bias)
        self.expanded_channels = self.n_frames * self.channels
        self.frame_expand = frame_expand
        self.frame_contract = frame_contract
        self.query_channels = (
            2 * self.expanded_channels
            if self.mode == "self"
            else (
                self.channels
                if self.frame_expand is not None
                else self.expanded_channels
            )
        )
        self.context_channels = (
            self.channels if self.frame_expand is not None else self.expanded_channels
        )
        self.output_channels = (
            self.channels if self.frame_contract is not None else self.expanded_channels
        )
        self.frame_zero_index = int(getattr(projector, "frame_zero_index", 0))

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
        if self.op_type == "mlp":
            self.grid_op = GridMLP(
                channels=self.channels,
                mode=self.mode,
                dtype=self.dtype,
                trainable=trainable,
                seed=child_seed(seed, 1),
            )
        elif self.op_type == "branch":
            self.grid_op = GridBranch(
                channels=self.channels,
                n_branches=grid_branches,
                dtype=self.dtype,
                trainable=trainable,
                seed=child_seed(seed, 1),
            )
        else:
            self.grid_op = nn.Identity()

        if residual_scale_init is None:
            self.residual_scale = None
        else:
            self.residual_scale = nn.Parameter(
                torch.ones(
                    self.n_focus,
                    self.output_channels,
                    dtype=self.dtype,
                    device=env.DEVICE,
                )
                * float(residual_scale_init),
                requires_grad=trainable,
            )

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the configured grid net and restore the input layout."""
        input_dtype = query.dtype
        query_ndfc, shape_info = self._to_ndfc(query)
        left, right, scalar_pair = self._prepare_pair(query_ndfc, context)
        grid_out = self._apply_grid_op(left, right, scalar_pair)
        coeff_out = self._from_grid(grid_out)
        coeff_out = self._apply_scalar_path(coeff_out, scalar_pair)
        coeff_out = self._contract_frames(coeff_out)
        coeff_out = self._apply_residual_scale(coeff_out)
        return self._restore_layout(coeff_out.to(dtype=input_dtype), shape_info)

    def _prepare_pair(
        self,
        query: torch.Tensor,
        context: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.mode == "self":
            return self._prepare_self_pair(query)
        return self._prepare_cross_pair(query, context)

    def _prepare_self_pair(
        self,
        query: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        left, right = self._split_self_query(query)
        scalar_pair = self._make_scalar_pair(left, right)
        return left, right, scalar_pair

    def _prepare_cross_pair(
        self,
        query: torch.Tensor,
        context: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if context is None:
            raise ValueError("`context` is required when `mode='cross'`")
        context_ndfc, _ = self._to_ndfc(context)
        self._check_last_dim(query, self.context_channels, "query")
        self._check_last_dim(context_ndfc, self.context_channels, "context")
        if self.frame_expand is None:
            scalar_pair = self._make_scalar_pair(query, context_ndfc)
            return query, context_ndfc, scalar_pair

        scalar_pair = torch.cat(
            [
                query[:, 0, :, :],
                context_ndfc[:, 0, :, :],
            ],
            dim=-1,
        ).to(dtype=self.dtype)
        return (
            self.frame_expand(query),
            self.frame_expand(context_ndfc),
            scalar_pair,
        )

    def _apply_grid_op(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        scalar_pair: torch.Tensor,
    ) -> torch.Tensor:
        left_grid = self._to_grid(left.to(dtype=self.dtype))
        right_grid = self._to_grid(right.to(dtype=self.dtype))
        if self.op_type == "glu":
            return left_grid * right_grid
        if self.op_type == "mlp":
            return self.grid_op(left_grid, right_grid)
        return self.grid_op(left_grid, right_grid, scalar_pair)

    def _contract_frames(self, coeff: torch.Tensor) -> torch.Tensor:
        if self.frame_contract is None:
            return coeff
        return self.frame_contract(coeff)

    def _apply_residual_scale(self, coeff: torch.Tensor) -> torch.Tensor:
        if self.residual_scale is None:
            return coeff
        return coeff * self.residual_scale.reshape(
            1,
            1,
            self.n_focus,
            self.output_channels,
        )

    def _apply_scalar_path(
        self,
        coeff: torch.Tensor,
        scalar_pair: torch.Tensor,
    ) -> torch.Tensor:
        scalar_out = self.scalar_act(scalar_pair)
        scalar_gate = torch.sigmoid(self.scalar_gate(scalar_pair))
        n_batch, coeff_dim, n_focus, _ = coeff.shape
        coeff_view = coeff.reshape(
            n_batch,
            coeff_dim,
            n_focus,
            self.n_frames,
            self.channels,
        )
        coeff_view = coeff_view * scalar_gate[:, None, :, None, :]
        coeff_view[:, 0, :, self.frame_zero_index, :].add_(scalar_out)
        return coeff_view.reshape(n_batch, coeff_dim, n_focus, self.expanded_channels)

    def _split_self_query(
        self, query: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._check_last_dim(query, self.query_channels, "query")
        return torch.chunk(query, chunks=2, dim=-1)

    def _make_scalar_pair(
        self, left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat(
            [
                self._extract_scalar(left),
                self._extract_scalar(right),
            ],
            dim=-1,
        ).to(dtype=self.dtype)

    def _extract_scalar(self, coeff: torch.Tensor) -> torch.Tensor:
        n_batch, _, n_focus, _ = coeff.shape
        coeff_view = coeff.reshape(
            n_batch,
            coeff.shape[1],
            n_focus,
            self.n_frames,
            self.channels,
        )
        return coeff_view[:, 0, :, self.frame_zero_index, :]

    def _to_grid(self, coeff: torch.Tensor) -> torch.Tensor:
        n_batch, coeff_dim, n_focus, _ = coeff.shape
        coeff_view = coeff.reshape(
            n_batch,
            coeff_dim,
            n_focus,
            self.n_frames,
            self.channels,
        )
        to_grid = self.projector.to_grid_mat.reshape(
            self.projector.grid_size,
            coeff_dim,
            self.n_frames,
        )
        return torch.einsum("gdk,ndfkc->ngfc", to_grid, coeff_view)

    def _from_grid(self, grid: torch.Tensor) -> torch.Tensor:
        n_batch, _, n_focus, _ = grid.shape
        coeff_dim = self.projector.coeff_dim // self.n_frames
        from_grid = self.projector.from_grid_mat.reshape(
            coeff_dim,
            self.n_frames,
            self.projector.grid_size,
        )
        coeff = torch.einsum("dkg,ngfc->ndfkc", from_grid, grid)
        return coeff.reshape(n_batch, coeff_dim, n_focus, self.expanded_channels)

    def _to_ndfc(self, value: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        if self.layout == "ndfc":
            return value, tuple(value.shape)
        if self.layout == "nfdc":
            return value.transpose(1, 2), tuple(value.shape)
        n_batch, coeff_dim, _ = value.shape
        return (
            value.reshape(n_batch, coeff_dim, self.n_focus, -1),
            tuple(value.shape),
        )

    def _restore_layout(
        self,
        value: torch.Tensor,
        shape_info: tuple[int, ...],
    ) -> torch.Tensor:
        if self.layout == "ndfc":
            return value
        if self.layout == "nfdc":
            return value.transpose(1, 2)
        n_batch, coeff_dim, _ = shape_info
        return value.reshape(n_batch, coeff_dim, -1)

    def _check_last_dim(
        self,
        value: torch.Tensor,
        expected: int,
        name: str,
    ) -> None:
        if value.shape[-1] != expected:
            raise ValueError(
                f"`{name}` last dimension must be {expected}, got {value.shape[-1]}"
            )


class S2GridNet(BaseGridNet):
    """Grid net using an S2 spherical-harmonic projector."""

    def __init__(
        self,
        *,
        lmax: int,
        mmax: int | None = None,
        channels: int,
        n_focus: int = 1,
        mode: GridNetMode,
        op_type: GridNetOp,
        dtype: torch.dtype,
        layout: GridNetLayout,
        grid_resolution_list: list[int] | None = None,
        coefficient_layout: str = "packed",
        grid_method: str = "e3nn",
        grid_branches: int = 1,
        residual_scale_init: float | None = None,
        mlp_bias: bool = False,
        trainable: bool,
        seed: int | list[int] | None = None,
    ) -> None:
        projector = S2GridProjector(
            lmax=lmax,
            mmax=mmax,
            dtype=dtype,
            grid_resolution_list=grid_resolution_list,
            coefficient_layout=coefficient_layout,
            grid_method=grid_method,
        )
        self.grid_resolution_list = projector.grid_resolution_list
        self.grid_method = projector.grid_method
        super().__init__(
            projector=projector,
            channels=channels,
            n_focus=n_focus,
            mode=mode,
            op_type=op_type,
            dtype=dtype,
            layout=layout,
            mlp_bias=mlp_bias,
            trainable=trainable,
            grid_branches=grid_branches,
            residual_scale_init=residual_scale_init,
            seed=seed,
        )


class SO3GridNet(BaseGridNet):
    """Grid net using a Wigner-D SO(3) projector with frame indices."""

    def __init__(
        self,
        *,
        lmax: int,
        mmax: int | None = None,
        kmax: int = 1,
        channels: int,
        n_focus: int = 1,
        mode: GridNetMode,
        op_type: GridNetOp,
        dtype: torch.dtype,
        layout: GridNetLayout,
        lebedev_precision: int | None = None,
        coefficient_layout: str = "packed",
        grid_branches: int = 1,
        residual_scale_init: float | None = None,
        mlp_bias: bool = False,
        trainable: bool,
        seed: int | list[int] | None = None,
    ) -> None:
        projector = SO3GridProjector(
            lmax=lmax,
            mmax=mmax,
            kmax=kmax,
            dtype=dtype,
            lebedev_precision=lebedev_precision,
            coefficient_layout=coefficient_layout,
        )
        self.frames = projector.frame_set
        self.kmax = projector.kmax
        self.lebedev_precision = projector.lebedev_precision
        self.n_gamma = projector.n_gamma
        frame_expand = None
        frame_contract = None
        if mode == "cross":
            frame_expand = FrameExpand(
                lmax=lmax,
                mmax=projector.mmax,
                coefficient_layout=coefficient_layout,
                n_frames=projector.n_frames,
                channels=channels,
                dtype=dtype,
                trainable=trainable,
                seed=child_seed(seed, 4),
            )
            frame_contract = FrameContract(
                lmax=lmax,
                mmax=projector.mmax,
                coefficient_layout=coefficient_layout,
                n_frames=projector.n_frames,
                channels=channels,
                dtype=dtype,
                trainable=trainable,
                seed=child_seed(seed, 5),
            )
        super().__init__(
            projector=projector,
            channels=channels,
            n_focus=n_focus,
            mode=mode,
            op_type=op_type,
            dtype=dtype,
            layout=layout,
            mlp_bias=mlp_bias,
            trainable=trainable,
            grid_branches=grid_branches,
            frame_expand=frame_expand,
            frame_contract=frame_contract,
            residual_scale_init=residual_scale_init,
            seed=seed,
        )
