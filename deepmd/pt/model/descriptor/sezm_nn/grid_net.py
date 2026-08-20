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
    TYPE_CHECKING,
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
from deepmd.pt_expt.kernels.utils import (
    cuda_infer_level,
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

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )

GridNetLayout = Literal["ndfc", "nfdc", "fndc", "flat"]
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


def _build_so3_scalar_product_weight(
    projector: SO3GridProjector,
) -> torch.Tensor:
    """Build the Haar inner-product weight for every ``(l, m, k)`` slot.

    Real Wigner-D coefficients obey
    ``integral D_lmk D_l'm'k' dR = delta_ll' delta_mm' delta_kk' / (2*l+1)``.
    Frame slots with ``abs(k) > l`` are structural zeros in the regular SeZM
    layout and therefore receive zero weight.
    """
    degree_index = _build_frame_degree_index(
        lmax=projector.lmax,
        mmax=projector.mmax,
        coefficient_layout=projector.coefficient_layout,
    ).reshape(-1, 1)
    frame_values = projector.frame_values.reshape(1, -1)
    valid_frame = torch.abs(frame_values) <= degree_index
    degree_weight = torch.reciprocal((2 * degree_index + 1).to(dtype=projector.dtype))
    return valid_frame.to(dtype=projector.dtype) * degree_weight


def _project_frames(
    coeff: torch.Tensor, proj: ChannelLinear, n_frames: int
) -> torch.Tensor:
    """
    Apply a channel-only linear map to each Wigner-D frame independently.

    Parameters
    ----------
    coeff : torch.Tensor
        Frame-packed coefficients with shape ``(N, D, F, n_frames * C_in)``.
    proj : ChannelLinear
        Linear map acting on the per-frame channel axis (``C_in -> C_out``).
    n_frames : int
        Number of Wigner-D frames packed along the trailing axis.

    Returns
    -------
    torch.Tensor
        Projected coefficients with shape ``(N, D, F, n_frames * C_out)``.

    Notes
    -----
    ``to_grid`` and ``from_grid`` are frame-wise linear and commute with any
    channel map, so applying the map at coefficient resolution here is identical
    to applying it on the grid field while touching ``n_frames``-fold fewer rows
    than the ``G``-point grid.
    """
    n_batch, coeff_dim, n_focus, _ = coeff.shape
    projected = proj(coeff.reshape(n_batch, coeff_dim, n_focus, n_frames, -1))
    return projected.reshape(n_batch, coeff_dim, n_focus, -1)


def _project_pair_in_one_transform(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    n_frames: int,
    to_grid: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project two equally shaped coefficient operands in one linear transform."""
    n_batch, coeff_dim, n_focus, _ = left.shape
    frame_shape = (n_batch, coeff_dim, n_focus, n_frames, -1)
    pair = torch.cat(
        [left.reshape(frame_shape), right.reshape(frame_shape)],
        dim=-1,
    ).reshape(n_batch, coeff_dim, n_focus, -1)
    return torch.chunk(to_grid(pair), chunks=2, dim=-1)


def _project_pair(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    to_grid: Callable[[torch.Tensor], torch.Tensor],
    project_pair: Callable[
        [torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
    ]
    | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project two operands through the selected projector composition."""
    if project_pair is not None:
        return project_pair(left, right)
    return to_grid(left), to_grid(right)


class GridProduct(nn.Module):
    """Parameter-free quadratic grid product ``u(g) * v(g)``."""

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        scalar_pair: torch.Tensor,
        *,
        to_grid: Callable[[torch.Tensor], torch.Tensor],
        from_grid: Callable[[torch.Tensor], torch.Tensor],
        pair_grid: Callable[[torch.Tensor, torch.Tensor], torch.Tensor | None]
        | None = None,
        project_pair: Callable[
            [torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
        ]
        | None = None,
        scalar_product: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        | None = None,
    ) -> torch.Tensor:
        """
        Combine two coefficient operands by a point-wise grid product.

        Parameters
        ----------
        left, right : torch.Tensor
            Coefficient operands with shape ``(N, D, F, n_frames * C)``.
        scalar_pair : torch.Tensor
            Invariant routing signal; unused on this path.
        to_grid, from_grid : Callable
            Coefficient/grid projectors supplied by the owning grid net.
        pair_grid, project_pair, scalar_product : Callable, optional
            Optional fused full composition, paired forward projection, and
            direct scalar coefficient contraction.

        Returns
        -------
        torch.Tensor
            Coefficient result. A direct scalar contraction has shape
            ``(N, 1, F, C)``; other paths retain ``n_frames * C`` channels.
        """
        if scalar_product is not None:
            return scalar_product(left, right)
        fused = pair_grid(left, right) if pair_grid is not None else None
        if fused is not None:
            return fused
        left_grid, right_grid = _project_pair(
            left,
            right,
            to_grid=to_grid,
            project_pair=project_pair,
        )
        return from_grid(left_grid * right_grid)


class GridMLP(nn.Module):
    """Polynomial point-wise MLP applied independently at every grid point."""

    def __init__(
        self,
        *,
        channels: int,
        mode: GridNetMode,
        n_frames: int,
        dtype: torch.dtype,
        trainable: bool,
        seed: int | list[int] | None = None,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.mode = str(mode).lower()
        if self.mode not in {"self", "cross"}:
            raise ValueError("`mode` must be either 'self' or 'cross'")
        self.n_frames = int(n_frames)
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
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        scalar_pair: torch.Tensor,
        *,
        to_grid: Callable[[torch.Tensor], torch.Tensor],
        from_grid: Callable[[torch.Tensor], torch.Tensor],
        pair_grid: Callable[[torch.Tensor, torch.Tensor], torch.Tensor | None]
        | None = None,
        project_pair: Callable[
            [torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
        ]
        | None = None,
        scalar_product: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        | None = None,
    ) -> torch.Tensor:
        """
        Apply the polynomial point-wise MLP on coefficient operands.

        In self mode, both projections see the per-frame concatenation of the
        two operands and can form self and cross quadratic channel terms.  In
        cross mode the query and context roles stay separate:
        ``(W_q query) * (W_c context)``.

        Parameters
        ----------
        left, right : torch.Tensor
            Coefficient operands with shape ``(N, D, F, n_frames * C)``.
        scalar_pair : torch.Tensor
            Invariant routing signal; unused on this path.
        to_grid, from_grid : Callable
            Coefficient/grid projectors supplied by the owning grid net.
        pair_grid, project_pair, scalar_product : Callable, optional
            Optional fused full composition, paired forward projection, and
            direct scalar coefficient contraction.

        Returns
        -------
        torch.Tensor
            Coefficient result. A direct scalar contraction has shape
            ``(N, 1, F, C)``; other paths retain ``n_frames * C`` channels.
        """
        # === Step 1. Channel projections at coefficient resolution ===
        left, right = self._project_operands(left, right)

        # === Step 2. Quadratic product on the grid, projected back ===
        if scalar_product is not None:
            coeff = scalar_product(left, right)
        else:
            coeff = pair_grid(left, right) if pair_grid is not None else None
            if coeff is None:
                left_grid, right_grid = _project_pair(
                    left,
                    right,
                    to_grid=to_grid,
                    project_pair=project_pair,
                )
                coeff = from_grid(left_grid * right_grid)
        if scalar_product is not None:
            return self.out_proj(coeff)
        return _project_frames(coeff, self.out_proj, self.n_frames)

    def _project_operands(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the two coefficient-space channel projections."""
        if self.mode == "self":
            shape = (*left.shape[:-1], self.n_frames, -1)
            fused = torch.cat(
                [left.reshape(shape), right.reshape(shape)], dim=-1
            ).reshape(*left.shape[:-1], -1)  # per-frame concat -> (N, D, F, K*2C)
            left = _project_frames(fused, self.left_proj, self.n_frames)
            right = _project_frames(fused, self.right_proj, self.n_frames)
        else:
            left = _project_frames(left, self.left_proj, self.n_frames)
            right = _project_frames(right, self.right_proj, self.n_frames)
        return left, right


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
        n_frames: int,
        dtype: torch.dtype,
        trainable: bool,
        seed: int | list[int] | None = None,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.n_branches = int(n_branches)
        if self.n_branches < 1:
            raise ValueError("`n_branches` must be positive")
        self.n_frames = int(n_frames)
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
        left: torch.Tensor,
        right: torch.Tensor,
        scalar_pair: torch.Tensor,
        *,
        to_grid: Callable[[torch.Tensor], torch.Tensor],
        from_grid: Callable[[torch.Tensor], torch.Tensor],
        pair_grid: Callable[[torch.Tensor, torch.Tensor], torch.Tensor | None]
        | None = None,
        project_pair: Callable[
            [torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
        ]
        | None = None,
        scalar_product: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        | None = None,
    ) -> torch.Tensor:
        """
        Apply scalar-routed grid branch mixing on coefficient operands.

        Parameters
        ----------
        left, right : torch.Tensor
            Coefficient operands with shape ``(N, D, F, n_frames * C)``.
        scalar_pair : torch.Tensor
            Invariant router source with shape ``(N, F, 2*C)``.
        to_grid, from_grid : Callable
            Coefficient/grid projectors supplied by the owning grid net.
        pair_grid, project_pair, scalar_product : Callable, optional
            Optional fused full composition, paired forward projection, and
            direct scalar coefficient contraction.

        Returns
        -------
        torch.Tensor
            Coefficient result. A direct scalar contraction has shape
            ``(N, 1, F, C)``; other paths retain ``n_frames * C`` channels.
        """
        # === Step 1. Branch channel projections at coefficient resolution ===
        left = _project_frames(left, self.left_proj, self.n_frames)
        right = _project_frames(right, self.right_proj, self.n_frames)

        # === Step 2. Quadratic branches on the grid, routed by scalars ===
        # A single branch makes the router softmax identically one, which
        # reduces the routed product to the plain grid product the fused
        # operator evaluates.
        if scalar_product is not None:
            coeff = scalar_product(left, right)
            n_batch, coeff_dim, n_focus, _ = coeff.shape
            value = coeff.reshape(
                n_batch,
                coeff_dim,
                n_focus,
                self.n_branches,
                self.channels,
            )
            router = torch.softmax(self.router(scalar_pair), dim=-1)
            coeff = torch.einsum("ndfhc,nfh->ndfc", value, router)
        else:
            coeff = (
                pair_grid(left, right)
                if self.n_branches == 1 and pair_grid is not None
                else None
            )
        if coeff is None:
            left_grid, right_grid = _project_pair(
                left,
                right,
                to_grid=to_grid,
                project_pair=project_pair,
            )
            value = left_grid * right_grid  # (N, G, F, N_branches * C)
            n_batch, n_grid, n_focus, _ = value.shape
            value = value.reshape(
                n_batch, n_grid, n_focus, self.n_branches, self.channels
            )
            router = torch.softmax(self.router(scalar_pair), dim=-1)  # (N, F, Nb)
            out = torch.einsum("ngfhc,nfh->ngfc", value, router)  # (N, G, F, C)
            coeff = from_grid(out)

        # === Step 3. Project back to coefficients and mix output channels ===
        if scalar_product is not None:
            return self.out_proj(coeff)
        return _project_frames(coeff, self.out_proj, self.n_frames)


def _degree_batched_matmul(coeff: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Contract ``einsum("ndfi,dio->ndfo", coeff, weight)``.

    Batched over the ``(D, F)`` axes, not over ``N`` (and not by collapsing
    ``N*F``, which would materialize a permuted copy of ``coeff``):
    expanding ``weight`` across ``F`` costs ``D*F*i*o`` elements versus
    ``N*D*F*i`` for the coefficient copy -- a factor ``N/o`` more. No
    reshape is involved, so an empty ``N`` batch flows through naturally.
    """
    coeff_df = coeff.permute(1, 2, 0, 3)  # (D, F, N, i)
    out = torch.matmul(coeff_df, weight.unsqueeze(1))  # (D, F, N, o)
    return out.permute(2, 0, 1, 3)  # (N, D, F, o)


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
        return _degree_batched_matmul(coeff, weight)

    def forward_scalar(self, coeff: torch.Tensor) -> torch.Tensor:
        """Contract the single ``l=0`` coefficient with its frame weights.

        Parameters
        ----------
        coeff : torch.Tensor
            Scalar coefficient with shape ``(N, 1, F, K*C)``.

        Returns
        -------
        torch.Tensor
            Contracted scalar with shape ``(N, 1, F, C)``.
        """
        return torch.einsum("ndfi,dio->ndfo", coeff, self.weight[0:1])


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
        return _degree_batched_matmul(coeff, weight)


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
        coefficient_rows = int(projector.coeff_dim) // self.n_frames
        # One wider projection reduces launch overhead for at most 25 coefficient
        # rows. Larger operands retain independent projections to bound the
        # short-lived concatenated tensor in the compiled training graph.
        self._combine_grid_projection = coefficient_rows <= 25
        self.mode = str(mode).lower()
        if self.mode not in {"self", "cross"}:
            raise ValueError("`mode` must be either 'self' or 'cross'")
        self.op_type = str(op_type).lower()
        if self.op_type not in {"glu", "mlp", "branch"}:
            raise ValueError("`op_type` must be one of 'glu', 'mlp', or 'branch'")
        self.dtype = dtype
        self.layout = str(layout).lower()
        if self.layout not in {"ndfc", "nfdc", "fndc", "flat"}:
            raise ValueError(
                "`layout` must be one of 'ndfc', 'nfdc', 'fndc', or 'flat'"
            )
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
        scalar_product_weight = (
            _build_so3_scalar_product_weight(projector)
            if isinstance(projector, SO3GridProjector)
            else None
        )
        self.register_buffer(
            "_scalar_product_weight",
            scalar_product_weight,
            persistent=False,
        )

        # The fused grid pair product needs the grid-to-coefficient projector
        # transposed so both matrices are read row-major by grid point.
        # The operator is instantiated per coefficient-slot count, which this
        # projector fixes, so the choice is made once here rather than per call.
        self._grid_pair_fn = None
        if (
            cuda_infer_level() >= 1
            and self.projector.to_grid_mat.dtype is torch.float32
        ):
            from deepmd.pt_expt.kernels.cuda.dpa4.grid_pair import (
                SUPPORTED_SLOTS,
                grid_pair,
                op_available,
            )

            slots = int(self.projector.to_grid_mat.shape[1])
            if op_available() and slots in SUPPORTED_SLOTS:
                self._grid_pair_fn = grid_pair
        self.register_buffer(
            "_from_grid_t",
            self.projector.from_grid_mat.transpose(0, 1).contiguous(),
            persistent=False,
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
        if self.op_type == "mlp":
            self.grid_op: nn.Module = GridMLP(
                channels=self.channels,
                mode=self.mode,
                n_frames=self.n_frames,
                dtype=self.dtype,
                trainable=trainable,
                seed=child_seed(seed, 1),
            )
        elif self.op_type == "branch":
            self.grid_op = GridBranch(
                channels=self.channels,
                n_branches=grid_branches,
                n_frames=self.n_frames,
                dtype=self.dtype,
                trainable=trainable,
                seed=child_seed(seed, 1),
            )
        else:
            self.grid_op = GridProduct()

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
        return self._forward(query, context, scalar_only=False)

    def forward_scalar(
        self,
        query: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the grid net and return only the scalar coefficient.

        Parameters
        ----------
        query : torch.Tensor
            Query coefficient tensor in the configured layout.
        context : torch.Tensor, optional
            Optional context coefficient tensor for cross mode.

        Returns
        -------
        torch.Tensor
            Grid-net output with the degree axis restricted to ``l=0``.

        Notes
        -----
        The final SeZM readout consumes only ``l=0``. SO(3) Haar orthogonality
        reduces its quadratic grid projection to a weighted coefficient inner
        product. Other projectors restrict the inverse grid projection to the
        scalar row. Accelerated inference keeps the full fused pair projection
        because materializing a scalar-only fallback grid would be slower than
        that fused operator.
        """
        if self._grid_pair_fn is not None and not self.training:
            return self._slice_scalar_layout(self.forward(query, context))
        return self._forward(query, context, scalar_only=True)

    def _forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor | None,
        *,
        scalar_only: bool,
    ) -> torch.Tensor:
        """Run the shared full or scalar-only grid path."""
        # === Step 1. Normalize the input layout and build product operands ===
        input_dtype = query.dtype
        query_ndfc, shape_info = self._to_ndfc(query)
        left, right, scalar_pair = self._prepare_pair(query_ndfc, context)

        # === Step 2. Select the static projection plan and apply the grid op ===
        direct_scalar = scalar_only and self._scalar_product_weight is not None
        coeff_out = self.grid_op(
            left.to(dtype=self.dtype),
            right.to(dtype=self.dtype),
            scalar_pair,
            to_grid=self._to_grid,
            project_pair=(
                self._project_pair_in_one_transform
                if not direct_scalar
                and (scalar_only or (self.training and self._combine_grid_projection))
                else None
            ),
            from_grid=self._from_grid_scalar if scalar_only else self._from_grid,
            pair_grid=None if scalar_only else self._pair_grid,
            scalar_product=self._scalar_so3_product if direct_scalar else None,
        )

        # === Step 3. Apply scalar gating and contract Wigner-D frames ===
        coeff_out = self._apply_scalar_path(
            coeff_out,
            scalar_pair,
            compact_scalar=direct_scalar,
        )
        coeff_out = self._contract_frames(coeff_out, scalar_only=scalar_only)
        coeff_out = self._apply_residual_scale(coeff_out)

        # === Step 4. Restore the caller layout and dtype ===
        return self._restore_layout(
            coeff_out.to(dtype=input_dtype),
            shape_info,
            scalar_only=scalar_only,
        )

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

    def _contract_frames(
        self,
        coeff: torch.Tensor,
        *,
        scalar_only: bool,
    ) -> torch.Tensor:
        if self.frame_contract is None:
            return coeff
        if scalar_only:
            return self.frame_contract.forward_scalar(coeff)
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
        *,
        compact_scalar: bool,
    ) -> torch.Tensor:
        scalar_out = self.scalar_act(scalar_pair)
        scalar_gate = torch.sigmoid(self.scalar_gate(scalar_pair))
        if compact_scalar:
            scalar_coeff = coeff * scalar_gate[:, None, :, :]
            scalar_coeff = scalar_coeff + scalar_out[:, None, :, :]
            return self._pack_scalar_frame(scalar_coeff)
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

    def _pack_scalar_frame(self, scalar: torch.Tensor) -> torch.Tensor:
        """Embed ``(N, 1, F, C)`` scalars in the ``k=0`` slot of ``K*C``."""
        n_batch, _, n_focus, channels = scalar.shape
        before = scalar.new_zeros(
            n_batch,
            1,
            n_focus,
            self.frame_zero_index,
            channels,
        )
        after = scalar.new_zeros(
            n_batch,
            1,
            n_focus,
            self.n_frames - self.frame_zero_index - 1,
            channels,
        )
        coeff = torch.cat(
            [before, scalar[:, :, :, None, :], after],
            dim=3,
        )
        return coeff.reshape(n_batch, 1, n_focus, self.expanded_channels)

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

    def _pair_grid(
        self, left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor | None:
        """
        Evaluate ``from_grid(to_grid(left) * to_grid(right))`` in one operator.

        The grid field is 39 times larger than its coefficient operand at the
        production SO(3) shape, so keeping it off device memory is worth a
        dedicated kernel. Returns ``None`` when the fused operator does not
        serve this shape, and the caller keeps the projector composition.

        Parameters
        ----------
        left, right : torch.Tensor
            Coefficient operands with shape (N, D, F, n_frames * C).
        right : torch.Tensor
            Second coefficient operand, same shape as ``left``.

        Returns
        -------
        torch.Tensor or None
            Coefficient result with shape (N, D, F, n_frames * C).
        """
        if self._grid_pair_fn is None or self.training or left.shape[2] != 1:
            return None
        n_batch, coeff_dim = left.shape[0], left.shape[1]
        flat_p = coeff_dim * self.n_frames
        c_wide = left.shape[3] // self.n_frames
        if c_wide % 32 != 0 or left.shape != right.shape:
            return None
        out = self._grid_pair_fn(
            left.reshape(n_batch, flat_p, c_wide),
            right.reshape(n_batch, flat_p, c_wide),
            self.projector.to_grid_mat,
            self._from_grid_t,
        )
        return out.reshape(n_batch, coeff_dim, 1, self.n_frames * c_wide)

    def _project_pair_in_one_transform(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project scalar-output operands with one shared linear transform."""
        return _project_pair_in_one_transform(
            left,
            right,
            n_frames=self.n_frames,
            to_grid=self._to_grid,
        )

    def _to_grid(self, coeff: torch.Tensor) -> torch.Tensor:
        # The per-frame channel width is inferred so the projector also serves
        # widened operands (e.g. a branch hidden width ``n_branches * C``).
        n_batch, coeff_dim, n_focus, _ = coeff.shape
        coeff_view = coeff.reshape(n_batch, coeff_dim, n_focus, self.n_frames, -1)
        to_grid = self.projector.to_grid_mat.reshape(
            self.projector.grid_size,
            coeff_dim,
            self.n_frames,
        )
        return torch.einsum("gdk,ndfkc->ngfc", to_grid, coeff_view)

    def _from_grid(self, grid: torch.Tensor) -> torch.Tensor:
        # Channel width is inferred to match the (possibly widened) grid field.
        n_batch, _, n_focus, _ = grid.shape
        coeff_dim = self.projector.coeff_dim // self.n_frames
        from_grid = self.projector.from_grid_mat.reshape(
            coeff_dim,
            self.n_frames,
            self.projector.grid_size,
        )
        coeff = torch.einsum("dkg,ngfc->ndfkc", from_grid, grid)
        return coeff.reshape(n_batch, coeff_dim, n_focus, -1)

    def _from_grid_scalar(self, grid: torch.Tensor) -> torch.Tensor:
        """Project a grid field to the ``l=0`` coefficient only."""
        n_batch, _, n_focus, _ = grid.shape
        coeff_dim = self.projector.coeff_dim // self.n_frames
        from_grid = self.projector.from_grid_mat.reshape(
            coeff_dim,
            self.n_frames,
            self.projector.grid_size,
        )[0:1]
        coeff = torch.einsum("dkg,ngfc->ndfkc", from_grid, grid)
        return coeff.reshape(n_batch, 1, n_focus, -1)

    def _scalar_so3_product(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        """Contract a quadratic SO(3) product directly to ``l=0, k=0``."""
        weight = self._scalar_product_weight
        if weight is None:
            raise RuntimeError("SO(3) scalar product weights are unavailable")
        n_batch, coeff_dim, n_focus, _ = left.shape
        left_view = left.reshape(n_batch, coeff_dim, n_focus, self.n_frames, -1)
        right_view = right.reshape_as(left_view)
        scalar = torch.einsum(
            "ndfkc,dk,ndfkc->nfc",
            left_view,
            weight,
            right_view,
        )
        return scalar[:, None, :, :]

    def _to_ndfc(self, value: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        # All grid operations run in the canonical ``(N, D, F, C)`` layout; the
        # ``fndc`` re-orientation folds the focus-major SO(2) mixing layout into the
        # same transpose the ``nfdc`` path performs, so the grid compute below is
        # identical regardless of the caller's layout.
        if self.layout == "ndfc":
            return value, tuple(value.shape)
        if self.layout == "nfdc":
            return value.transpose(1, 2), tuple(value.shape)
        if self.layout == "fndc":
            return value.permute(1, 2, 0, 3), tuple(value.shape)
        n_batch, coeff_dim, _ = value.shape
        return (
            value.reshape(n_batch, coeff_dim, self.n_focus, -1),
            tuple(value.shape),
        )

    def _restore_layout(
        self,
        value: torch.Tensor,
        shape_info: tuple[int, ...],
        *,
        scalar_only: bool = False,
    ) -> torch.Tensor:
        if self.layout == "ndfc":
            return value
        if self.layout == "nfdc":
            return value.transpose(1, 2)
        if self.layout == "fndc":
            return value.permute(2, 0, 1, 3)
        n_batch, input_coeff_dim, _ = shape_info
        coeff_dim = 1 if scalar_only else input_coeff_dim
        return value.reshape(n_batch, coeff_dim, -1)

    def _slice_scalar_layout(self, value: torch.Tensor) -> torch.Tensor:
        """Select the degree axis from a restored full-layout tensor."""
        if self.layout == "ndfc":
            return value[:, 0:1, :, :]
        if self.layout in {"nfdc", "fndc"}:
            return value[:, :, 0:1, :]
        return value[:, 0:1, :]

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
