# SPDX-License-Identifier: LGPL-3.0-or-later
"""Triton custom-op launchers for SeZM SO(2) rotation kernels.

This layer only decides how to launch a resolved dispatch mode. Fallback policy
stays in the public autograd API so the launchers remain focused on Triton
grids, kernel families, and argument packing.
"""

from __future__ import (
    annotations,
)

import torch  # noqa: TC002

from .constants import (
    SEZM_TRITON_AVAILABLE,
    TRITON_BLOCK_CHANNEL,
    TRITON_BLOCK_FULL,
    TRITON_BLOCK_REDUCED,
    TRITON_EDGE_GEOMETRY_RBF_BLOCK_EDGE,
    TRITON_EDGE_GEOMETRY_RBF_BLOCK_RADIAL,
    TRITON_GRID_E_STRIDE,
    TRITON_SMALL_BLOCK_CHANNEL,
    TritonRotationMode,
)
from .dispatch import (
    coerce_rotation_mode,
)


def _require_kernel_mode(
    rotation_mode: int | TritonRotationMode,
) -> TritonRotationMode:
    """Reject eager fallback before entering the Triton launch layer."""
    resolved_mode = coerce_rotation_mode(rotation_mode)
    if resolved_mode == TritonRotationMode.EAGER_REFERENCE:
        raise ValueError("Eager reference mode must be handled before Triton launch.")
    return resolved_mode


if SEZM_TRITON_AVAILABLE:
    from torch.library import (
        triton_op,
        wrap_triton,
    )

    from .kernels_edge_geometry_rbf import (
        edge_geometry_rbf_bwd_accum_kernel,
        edge_geometry_rbf_bwd_coord_kernel,
        edge_geometry_rbf_forward_kernel,
    )
    from .kernels_generic import (
        rotate_back_bwd_dw_kernel,
        rotate_back_bwd_dx_kernel,
        rotate_back_forward_kernel,
        rotate_to_local_bwd_dw_kernel,
        rotate_to_local_bwd_dx_kernel,
        rotate_to_local_forward_kernel,
    )
    from .kernels_small import (
        rotate_back_l1_bwd_dx_kernel,
        rotate_back_l1_forward_kernel,
        rotate_back_l2_bwd_dx_kernel,
        rotate_back_l2_forward_kernel,
        rotate_back_l3_bwd_dx_kernel,
        rotate_back_l3_forward_kernel,
        rotate_back_small_bwd_dw_kernel,
        rotate_to_local_l1_bwd_dx_kernel,
        rotate_to_local_l1_forward_kernel,
        rotate_to_local_l2_bwd_dx_kernel,
        rotate_to_local_l2_forward_kernel,
        rotate_to_local_l3_bwd_dx_kernel,
        rotate_to_local_l3_forward_kernel,
        rotate_to_local_small_bwd_dw_kernel,
    )

    _ROTATE_TO_LOCAL_SMALL_FORWARD = {
        TritonRotationMode.SMALL_LE1: rotate_to_local_l1_forward_kernel,
        TritonRotationMode.SMALL_L2: rotate_to_local_l2_forward_kernel,
        TritonRotationMode.SMALL_L3: rotate_to_local_l3_forward_kernel,
    }
    _ROTATE_TO_LOCAL_SMALL_BWD_DX = {
        TritonRotationMode.SMALL_LE1: rotate_to_local_l1_bwd_dx_kernel,
        TritonRotationMode.SMALL_L2: rotate_to_local_l2_bwd_dx_kernel,
        TritonRotationMode.SMALL_L3: rotate_to_local_l3_bwd_dx_kernel,
    }
    _ROTATE_BACK_SMALL_FORWARD = {
        TritonRotationMode.SMALL_LE1: rotate_back_l1_forward_kernel,
        TritonRotationMode.SMALL_L2: rotate_back_l2_forward_kernel,
        TritonRotationMode.SMALL_L3: rotate_back_l3_forward_kernel,
    }
    _ROTATE_BACK_SMALL_BWD_DX = {
        TritonRotationMode.SMALL_LE1: rotate_back_l1_bwd_dx_kernel,
        TritonRotationMode.SMALL_L2: rotate_back_l2_bwd_dx_kernel,
        TritonRotationMode.SMALL_L3: rotate_back_l3_bwd_dx_kernel,
    }

    def _small_channel_grid(channels: int) -> tuple[int, int]:
        """Return the standard ``(edge, channel)`` grid for small kernels."""
        return (
            TRITON_GRID_E_STRIDE,
            (channels + TRITON_SMALL_BLOCK_CHANNEL - 1) // TRITON_SMALL_BLOCK_CHANNEL,
        )

    def _generic_rotate_to_local_forward_grid(
        reduced_dim: int,
        channels: int,
    ) -> tuple[int, int, int]:
        """Return the standard forward grid for generic rotate-to-local."""
        return (
            TRITON_GRID_E_STRIDE,
            (reduced_dim + TRITON_BLOCK_REDUCED - 1) // TRITON_BLOCK_REDUCED,
            (channels + TRITON_BLOCK_CHANNEL - 1) // TRITON_BLOCK_CHANNEL,
        )

    def _generic_rotate_to_local_bwd_dx_grid(
        dim_full: int,
        channels: int,
    ) -> tuple[int, int, int]:
        """Return the source-gradient grid for generic rotate-to-local."""
        return (
            TRITON_GRID_E_STRIDE,
            (dim_full + TRITON_BLOCK_FULL - 1) // TRITON_BLOCK_FULL,
            (channels + TRITON_BLOCK_CHANNEL - 1) // TRITON_BLOCK_CHANNEL,
        )

    def _generic_rotate_to_local_bwd_dw_grid(
        reduced_dim: int,
        dim_full: int,
    ) -> tuple[int, int, int]:
        """Return the Wigner-gradient grid for generic rotate-to-local."""
        return (
            TRITON_GRID_E_STRIDE,
            (reduced_dim + TRITON_BLOCK_REDUCED - 1) // TRITON_BLOCK_REDUCED,
            (dim_full + TRITON_BLOCK_FULL - 1) // TRITON_BLOCK_FULL,
        )

    def _generic_rotate_back_forward_grid(
        dim_full: int,
        channels: int,
    ) -> tuple[int, int, int]:
        """Return the standard forward grid for generic rotate-back."""
        return (
            TRITON_GRID_E_STRIDE,
            (dim_full + TRITON_BLOCK_FULL - 1) // TRITON_BLOCK_FULL,
            (channels + TRITON_BLOCK_CHANNEL - 1) // TRITON_BLOCK_CHANNEL,
        )

    def _generic_rotate_back_bwd_dx_grid(
        reduced_dim: int,
        channels: int,
    ) -> tuple[int, int, int]:
        """Return the reduced-gradient grid for generic rotate-back."""
        return (
            TRITON_GRID_E_STRIDE,
            (reduced_dim + TRITON_BLOCK_REDUCED - 1) // TRITON_BLOCK_REDUCED,
            (channels + TRITON_BLOCK_CHANNEL - 1) // TRITON_BLOCK_CHANNEL,
        )

    def _generic_rotate_back_bwd_dw_grid(
        dim_full: int,
        reduced_dim: int,
    ) -> tuple[int, int, int]:
        """Return the Wigner-gradient grid for generic rotate-back."""
        return (
            TRITON_GRID_E_STRIDE,
            (dim_full + TRITON_BLOCK_FULL - 1) // TRITON_BLOCK_FULL,
            (reduced_dim + TRITON_BLOCK_REDUCED - 1) // TRITON_BLOCK_REDUCED,
        )

    def _edge_geometry_rbf_grid(num_edges: int, n_radial: int) -> tuple[int, int]:
        """Return the standard grid for the fused edge geometry/RBF chain."""
        return (
            (num_edges + TRITON_EDGE_GEOMETRY_RBF_BLOCK_EDGE - 1)
            // TRITON_EDGE_GEOMETRY_RBF_BLOCK_EDGE,
            (n_radial + TRITON_EDGE_GEOMETRY_RBF_BLOCK_RADIAL - 1)
            // TRITON_EDGE_GEOMETRY_RBF_BLOCK_RADIAL,
        )

    def _edge_geometry_rbf_coord_grid(num_edges: int) -> tuple[int]:
        """Return the edge-only grid for geometry/RBF coordinate gradients."""
        return (
            (num_edges + TRITON_EDGE_GEOMETRY_RBF_BLOCK_EDGE - 1)
            // TRITON_EDGE_GEOMETRY_RBF_BLOCK_EDGE,
        )

    def _launch_rotate_to_local_small_forward(
        *,
        rotation_mode: TritonRotationMode,
        x: torch.Tensor,
        src: torch.Tensor,
        wigner: torch.Tensor,
        coeff_index: torch.Tensor,
        out: torch.Tensor,
        dim_full: int,
    ) -> None:
        """Launch one specialized small-family rotate-to-local forward kernel."""
        reduced_dim = coeff_index.numel()
        channels = x.shape[2]
        kernel = _ROTATE_TO_LOCAL_SMALL_FORWARD[rotation_mode]
        wrap_triton(kernel)[_small_channel_grid(channels)](
            x,
            src,
            wigner,
            coeff_index,
            out,
            src.shape[0],
            reduced_dim,
            dim_full,
            channels,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            wigner.stride(0),
            wigner.stride(1),
            wigner.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            BLOCK_CHANNEL=TRITON_SMALL_BLOCK_CHANNEL,
            GRID_E_STRIDE=TRITON_GRID_E_STRIDE,
            num_warps=1,
        )

    def _launch_rotate_to_local_small_bwd_dx(
        *,
        rotation_mode: TritonRotationMode,
        grad_out: torch.Tensor,
        wigner: torch.Tensor,
        coeff_index: torch.Tensor,
        grad_edge: torch.Tensor,
        dim_full: int,
    ) -> None:
        """Launch one specialized small-family rotate-to-local dx kernel."""
        reduced_dim = coeff_index.numel()
        channels = grad_out.shape[2]
        kernel = _ROTATE_TO_LOCAL_SMALL_BWD_DX[rotation_mode]
        wrap_triton(kernel)[_small_channel_grid(channels)](
            grad_out,
            wigner,
            coeff_index,
            grad_edge,
            grad_out.shape[0],
            reduced_dim,
            dim_full,
            channels,
            grad_out.stride(0),
            grad_out.stride(1),
            grad_out.stride(2),
            wigner.stride(0),
            wigner.stride(1),
            wigner.stride(2),
            grad_edge.stride(0),
            grad_edge.stride(1),
            grad_edge.stride(2),
            BLOCK_CHANNEL=TRITON_SMALL_BLOCK_CHANNEL,
            GRID_E_STRIDE=TRITON_GRID_E_STRIDE,
            num_warps=1,
        )

    def _launch_rotate_back_small_forward(
        *,
        rotation_mode: TritonRotationMode,
        x_local: torch.Tensor,
        wigner: torch.Tensor,
        coeff_index: torch.Tensor,
        out: torch.Tensor,
        dim_full: int,
    ) -> None:
        """Launch one specialized small-family rotate-back forward kernel."""
        reduced_dim = coeff_index.numel()
        channels = x_local.shape[2]
        kernel = _ROTATE_BACK_SMALL_FORWARD[rotation_mode]
        wrap_triton(kernel)[_small_channel_grid(channels)](
            x_local,
            wigner,
            coeff_index,
            out,
            x_local.shape[0],
            reduced_dim,
            dim_full,
            channels,
            x_local.stride(0),
            x_local.stride(1),
            x_local.stride(2),
            wigner.stride(0),
            wigner.stride(1),
            wigner.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            BLOCK_CHANNEL=TRITON_SMALL_BLOCK_CHANNEL,
            GRID_E_STRIDE=TRITON_GRID_E_STRIDE,
            num_warps=1,
        )

    def _launch_rotate_back_small_bwd_dx(
        *,
        rotation_mode: TritonRotationMode,
        grad_out: torch.Tensor,
        wigner: torch.Tensor,
        coeff_index: torch.Tensor,
        grad_x_local: torch.Tensor,
        dim_full: int,
    ) -> None:
        """Launch one specialized small-family rotate-back dx kernel."""
        reduced_dim = coeff_index.numel()
        channels = grad_out.shape[2]
        kernel = _ROTATE_BACK_SMALL_BWD_DX[rotation_mode]
        wrap_triton(kernel)[_small_channel_grid(channels)](
            grad_out,
            wigner,
            coeff_index,
            grad_x_local,
            grad_out.shape[0],
            reduced_dim,
            dim_full,
            channels,
            grad_out.stride(0),
            grad_out.stride(1),
            grad_out.stride(2),
            wigner.stride(0),
            wigner.stride(1),
            wigner.stride(2),
            grad_x_local.stride(0),
            grad_x_local.stride(1),
            grad_x_local.stride(2),
            BLOCK_CHANNEL=TRITON_SMALL_BLOCK_CHANNEL,
            GRID_E_STRIDE=TRITON_GRID_E_STRIDE,
            num_warps=1,
        )

    @triton_op(
        "deepmd::_kernel_sezm_rotate_to_local",
        mutates_args=("out",),
    )
    def _kernel_sezm_rotate_to_local(
        x: torch.Tensor,
        src: torch.Tensor,
        wigner: torch.Tensor,
        coeff_index: torch.Tensor,
        out: torch.Tensor,
        dim_full: int,
        rotation_mode: int,
    ) -> None:
        """Launch the fused Triton forward kernel for ``D_to_m @ x[src]``."""
        mode = _require_kernel_mode(rotation_mode)
        reduced_dim = coeff_index.numel()
        channels = x.shape[2]
        if mode != TritonRotationMode.GENERIC_TILED:
            _launch_rotate_to_local_small_forward(
                rotation_mode=mode,
                x=x,
                src=src,
                wigner=wigner,
                coeff_index=coeff_index,
                out=out,
                dim_full=dim_full,
            )
            return
        wrap_triton(rotate_to_local_forward_kernel)[
            _generic_rotate_to_local_forward_grid(reduced_dim, channels)
        ](
            x,
            src,
            wigner,
            coeff_index,
            out,
            src.shape[0],
            reduced_dim,
            dim_full,
            channels,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            wigner.stride(0),
            wigner.stride(1),
            wigner.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            BLOCK_REDUCED=TRITON_BLOCK_REDUCED,
            BLOCK_FULL=TRITON_BLOCK_FULL,
            BLOCK_CHANNEL=TRITON_BLOCK_CHANNEL,
            GRID_E_STRIDE=TRITON_GRID_E_STRIDE,
            num_warps=1,
        )

    @triton_op(
        "deepmd::_kernel_sezm_rotate_to_local_bwd_dx",
        mutates_args=("grad_edge",),
    )
    def _kernel_sezm_rotate_to_local_bwd_dx(
        grad_out: torch.Tensor,
        wigner: torch.Tensor,
        coeff_index: torch.Tensor,
        grad_edge: torch.Tensor,
        dim_full: int,
        rotation_mode: int,
    ) -> None:
        """Launch the Triton backward kernel for source-feature gradients."""
        mode = _require_kernel_mode(rotation_mode)
        reduced_dim = coeff_index.numel()
        channels = grad_out.shape[2]
        if mode != TritonRotationMode.GENERIC_TILED:
            _launch_rotate_to_local_small_bwd_dx(
                rotation_mode=mode,
                grad_out=grad_out,
                wigner=wigner,
                coeff_index=coeff_index,
                grad_edge=grad_edge,
                dim_full=dim_full,
            )
            return
        wrap_triton(rotate_to_local_bwd_dx_kernel)[
            _generic_rotate_to_local_bwd_dx_grid(dim_full, channels)
        ](
            grad_out,
            wigner,
            coeff_index,
            grad_edge,
            grad_out.shape[0],
            reduced_dim,
            dim_full,
            channels,
            grad_out.stride(0),
            grad_out.stride(1),
            grad_out.stride(2),
            wigner.stride(0),
            wigner.stride(1),
            wigner.stride(2),
            grad_edge.stride(0),
            grad_edge.stride(1),
            grad_edge.stride(2),
            BLOCK_REDUCED=TRITON_BLOCK_REDUCED,
            BLOCK_FULL=TRITON_BLOCK_FULL,
            BLOCK_CHANNEL=TRITON_BLOCK_CHANNEL,
            GRID_E_STRIDE=TRITON_GRID_E_STRIDE,
            num_warps=1,
        )

    @triton_op(
        "deepmd::_kernel_sezm_rotate_to_local_bwd_dw",
        mutates_args=("grad_wigner",),
    )
    def _kernel_sezm_rotate_to_local_bwd_dw(
        grad_out: torch.Tensor,
        x: torch.Tensor,
        src: torch.Tensor,
        coeff_index: torch.Tensor,
        grad_wigner: torch.Tensor,
        dim_full: int,
        rotation_mode: int,
    ) -> None:
        """Launch the Triton backward kernel for Wigner gradients."""
        mode = _require_kernel_mode(rotation_mode)
        reduced_dim = coeff_index.numel()
        channels = grad_out.shape[2]
        if mode != TritonRotationMode.GENERIC_TILED:
            wrap_triton(rotate_to_local_small_bwd_dw_kernel)[(TRITON_GRID_E_STRIDE,)](
                grad_out,
                x,
                src,
                coeff_index,
                grad_wigner,
                grad_out.shape[0],
                reduced_dim,
                dim_full,
                channels,
                grad_out.stride(0),
                grad_out.stride(1),
                grad_out.stride(2),
                x.stride(0),
                x.stride(1),
                x.stride(2),
                grad_wigner.stride(0),
                grad_wigner.stride(1),
                grad_wigner.stride(2),
                BLOCK_CHANNEL=TRITON_SMALL_BLOCK_CHANNEL,
                GRID_E_STRIDE=TRITON_GRID_E_STRIDE,
                num_warps=1,
            )
            return
        wrap_triton(rotate_to_local_bwd_dw_kernel)[
            _generic_rotate_to_local_bwd_dw_grid(reduced_dim, dim_full)
        ](
            grad_out,
            x,
            src,
            coeff_index,
            grad_wigner,
            grad_out.shape[0],
            reduced_dim,
            dim_full,
            channels,
            grad_out.stride(0),
            grad_out.stride(1),
            grad_out.stride(2),
            x.stride(0),
            x.stride(1),
            x.stride(2),
            grad_wigner.stride(0),
            grad_wigner.stride(1),
            grad_wigner.stride(2),
            BLOCK_REDUCED=TRITON_BLOCK_REDUCED,
            BLOCK_FULL=TRITON_BLOCK_FULL,
            BLOCK_CHANNEL=TRITON_BLOCK_CHANNEL,
            GRID_E_STRIDE=TRITON_GRID_E_STRIDE,
            num_warps=1,
        )

    @triton_op(
        "deepmd::_kernel_sezm_rotate_back",
        mutates_args=("out",),
    )
    def _kernel_sezm_rotate_back(
        x_local: torch.Tensor,
        wigner: torch.Tensor,
        coeff_index: torch.Tensor,
        out: torch.Tensor,
        dim_full: int,
        rotation_mode: int,
    ) -> None:
        """Launch the fused Triton forward kernel for ``Dt_from_m @ x_local``."""
        mode = _require_kernel_mode(rotation_mode)
        reduced_dim = coeff_index.numel()
        channels = x_local.shape[2]
        if mode != TritonRotationMode.GENERIC_TILED:
            _launch_rotate_back_small_forward(
                rotation_mode=mode,
                x_local=x_local,
                wigner=wigner,
                coeff_index=coeff_index,
                out=out,
                dim_full=dim_full,
            )
            return
        wrap_triton(rotate_back_forward_kernel)[
            _generic_rotate_back_forward_grid(dim_full, channels)
        ](
            x_local,
            wigner,
            coeff_index,
            out,
            x_local.shape[0],
            reduced_dim,
            dim_full,
            channels,
            x_local.stride(0),
            x_local.stride(1),
            x_local.stride(2),
            wigner.stride(0),
            wigner.stride(1),
            wigner.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            BLOCK_REDUCED=TRITON_BLOCK_REDUCED,
            BLOCK_FULL=TRITON_BLOCK_FULL,
            BLOCK_CHANNEL=TRITON_BLOCK_CHANNEL,
            GRID_E_STRIDE=TRITON_GRID_E_STRIDE,
            num_warps=1,
        )

    @triton_op(
        "deepmd::_kernel_sezm_rotate_back_bwd_dx",
        mutates_args=("grad_x_local",),
    )
    def _kernel_sezm_rotate_back_bwd_dx(
        grad_out: torch.Tensor,
        wigner: torch.Tensor,
        coeff_index: torch.Tensor,
        grad_x_local: torch.Tensor,
        dim_full: int,
        rotation_mode: int,
    ) -> None:
        """Launch the Triton backward kernel for reduced-layout gradients."""
        mode = _require_kernel_mode(rotation_mode)
        reduced_dim = coeff_index.numel()
        channels = grad_out.shape[2]
        if mode != TritonRotationMode.GENERIC_TILED:
            _launch_rotate_back_small_bwd_dx(
                rotation_mode=mode,
                grad_out=grad_out,
                wigner=wigner,
                coeff_index=coeff_index,
                grad_x_local=grad_x_local,
                dim_full=dim_full,
            )
            return
        wrap_triton(rotate_back_bwd_dx_kernel)[
            _generic_rotate_back_bwd_dx_grid(reduced_dim, channels)
        ](
            grad_out,
            wigner,
            coeff_index,
            grad_x_local,
            grad_out.shape[0],
            reduced_dim,
            dim_full,
            channels,
            grad_out.stride(0),
            grad_out.stride(1),
            grad_out.stride(2),
            wigner.stride(0),
            wigner.stride(1),
            wigner.stride(2),
            grad_x_local.stride(0),
            grad_x_local.stride(1),
            grad_x_local.stride(2),
            BLOCK_REDUCED=TRITON_BLOCK_REDUCED,
            BLOCK_FULL=TRITON_BLOCK_FULL,
            BLOCK_CHANNEL=TRITON_BLOCK_CHANNEL,
            GRID_E_STRIDE=TRITON_GRID_E_STRIDE,
            num_warps=1,
        )

    @triton_op(
        "deepmd::_kernel_sezm_rotate_back_bwd_dw",
        mutates_args=("grad_wigner",),
    )
    def _kernel_sezm_rotate_back_bwd_dw(
        grad_out: torch.Tensor,
        x_local: torch.Tensor,
        coeff_index: torch.Tensor,
        grad_wigner: torch.Tensor,
        dim_full: int,
        rotation_mode: int,
    ) -> None:
        """Launch the Triton backward kernel for Wigner gradients."""
        mode = _require_kernel_mode(rotation_mode)
        reduced_dim = coeff_index.numel()
        channels = grad_out.shape[2]
        if mode != TritonRotationMode.GENERIC_TILED:
            wrap_triton(rotate_back_small_bwd_dw_kernel)[(TRITON_GRID_E_STRIDE,)](
                grad_out,
                x_local,
                coeff_index,
                grad_wigner,
                grad_out.shape[0],
                x_local.shape[1],
                dim_full,
                channels,
                grad_out.stride(0),
                grad_out.stride(1),
                grad_out.stride(2),
                x_local.stride(0),
                x_local.stride(1),
                x_local.stride(2),
                grad_wigner.stride(0),
                grad_wigner.stride(1),
                grad_wigner.stride(2),
                BLOCK_CHANNEL=TRITON_SMALL_BLOCK_CHANNEL,
                GRID_E_STRIDE=TRITON_GRID_E_STRIDE,
                num_warps=1,
            )
            return
        wrap_triton(rotate_back_bwd_dw_kernel)[
            _generic_rotate_back_bwd_dw_grid(dim_full, reduced_dim)
        ](
            grad_out,
            x_local,
            grad_wigner,
            grad_out.shape[0],
            x_local.shape[1],
            dim_full,
            channels,
            grad_out.stride(0),
            grad_out.stride(1),
            grad_out.stride(2),
            x_local.stride(0),
            x_local.stride(1),
            x_local.stride(2),
            grad_wigner.stride(0),
            grad_wigner.stride(1),
            grad_wigner.stride(2),
            BLOCK_REDUCED=TRITON_BLOCK_REDUCED,
            BLOCK_FULL=TRITON_BLOCK_FULL,
            BLOCK_CHANNEL=TRITON_BLOCK_CHANNEL,
            GRID_E_STRIDE=TRITON_GRID_E_STRIDE,
            num_warps=1,
        )

    @triton_op(
        "deepmd::_kernel_sezm_edge_geometry_rbf",
        mutates_args=("edge_vec", "edge_len", "edge_env", "edge_rbf"),
    )
    def _kernel_sezm_edge_geometry_rbf(
        coord_flat: torch.Tensor,
        center_coord_index: torch.Tensor,
        neighbor_coord_index: torch.Tensor,
        freqs: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_len: torch.Tensor,
        edge_env: torch.Tensor,
        edge_rbf: torch.Tensor,
        eps: float,
        rcut: float,
        edge_env_a: float,
        edge_env_b: float,
        edge_env_c: float,
        edge_env_d: float,
        edge_env_exponent: int,
        radial_env_a: float,
        radial_env_b: float,
        radial_env_c: float,
        radial_env_d: float,
        radial_env_exponent: int,
        r_inner: float,
        r_outer: float,
        has_inner_clamp: bool,
    ) -> None:
        """Launch the fused edge geometry/RBF forward kernel."""
        wrap_triton(edge_geometry_rbf_forward_kernel)[
            _edge_geometry_rbf_grid(center_coord_index.shape[0], freqs.numel())
        ](
            coord_flat,
            center_coord_index,
            neighbor_coord_index,
            freqs,
            edge_vec,
            edge_len,
            edge_env,
            edge_rbf,
            center_coord_index.shape[0],
            freqs.numel(),
            coord_flat.stride(0),
            coord_flat.stride(1),
            edge_vec.stride(0),
            edge_vec.stride(1),
            edge_rbf.stride(0),
            edge_rbf.stride(1),
            eps,
            rcut,
            edge_env_a,
            edge_env_b,
            edge_env_c,
            edge_env_d,
            radial_env_a,
            radial_env_b,
            radial_env_c,
            radial_env_d,
            r_inner,
            r_outer,
            EDGE_ENV_EXPONENT=int(edge_env_exponent),
            RADIAL_ENV_EXPONENT=int(radial_env_exponent),
            HAS_INNER_CLAMP=bool(has_inner_clamp),
            BLOCK_EDGE=TRITON_EDGE_GEOMETRY_RBF_BLOCK_EDGE,
            BLOCK_RADIAL=TRITON_EDGE_GEOMETRY_RBF_BLOCK_RADIAL,
            num_warps=4,
        )

    @triton_op(
        "deepmd::_kernel_sezm_edge_geometry_rbf_bwd_accum",
        mutates_args=("grad_r_total", "grad_freq"),
    )
    def _kernel_sezm_edge_geometry_rbf_bwd_accum(
        grad_edge_len: torch.Tensor,
        grad_edge_env: torch.Tensor,
        grad_edge_rbf: torch.Tensor,
        coord_flat: torch.Tensor,
        center_coord_index: torch.Tensor,
        neighbor_coord_index: torch.Tensor,
        freqs: torch.Tensor,
        grad_r_total: torch.Tensor,
        grad_freq: torch.Tensor,
        eps: float,
        rcut: float,
        edge_env_a: float,
        edge_env_b: float,
        edge_env_c: float,
        edge_env_d: float,
        edge_env_exponent: int,
        radial_env_a: float,
        radial_env_b: float,
        radial_env_c: float,
        radial_env_d: float,
        radial_env_exponent: int,
        r_inner: float,
        r_outer: float,
        has_inner_clamp: bool,
    ) -> None:
        """Launch the fused edge geometry/RBF accumulation kernel."""
        wrap_triton(edge_geometry_rbf_bwd_accum_kernel)[
            _edge_geometry_rbf_grid(center_coord_index.shape[0], freqs.numel())
        ](
            grad_edge_len,
            grad_edge_env,
            grad_edge_rbf,
            coord_flat,
            center_coord_index,
            neighbor_coord_index,
            freqs,
            grad_r_total,
            grad_freq,
            center_coord_index.shape[0],
            freqs.numel(),
            coord_flat.stride(0),
            coord_flat.stride(1),
            grad_edge_rbf.stride(0),
            grad_edge_rbf.stride(1),
            eps,
            rcut,
            edge_env_a,
            edge_env_b,
            edge_env_c,
            edge_env_d,
            radial_env_a,
            radial_env_b,
            radial_env_c,
            radial_env_d,
            r_inner,
            r_outer,
            EDGE_ENV_EXPONENT=int(edge_env_exponent),
            RADIAL_ENV_EXPONENT=int(radial_env_exponent),
            HAS_INNER_CLAMP=bool(has_inner_clamp),
            BLOCK_EDGE=TRITON_EDGE_GEOMETRY_RBF_BLOCK_EDGE,
            BLOCK_RADIAL=TRITON_EDGE_GEOMETRY_RBF_BLOCK_RADIAL,
            num_warps=4,
        )

    @triton_op(
        "deepmd::_kernel_sezm_edge_geometry_rbf_bwd_coord",
        mutates_args=("grad_coord",),
    )
    def _kernel_sezm_edge_geometry_rbf_bwd_coord(
        grad_edge_vec: torch.Tensor,
        grad_r_total: torch.Tensor,
        coord_flat: torch.Tensor,
        center_coord_index: torch.Tensor,
        neighbor_coord_index: torch.Tensor,
        grad_coord: torch.Tensor,
        eps: float,
        r_inner: float,
        r_outer: float,
        has_inner_clamp: bool,
    ) -> None:
        """Launch the fused edge geometry/RBF coordinate backward kernel."""
        wrap_triton(edge_geometry_rbf_bwd_coord_kernel)[
            _edge_geometry_rbf_coord_grid(center_coord_index.shape[0])
        ](
            grad_edge_vec,
            grad_r_total,
            coord_flat,
            center_coord_index,
            neighbor_coord_index,
            grad_coord,
            center_coord_index.shape[0],
            coord_flat.stride(0),
            coord_flat.stride(1),
            grad_edge_vec.stride(0),
            grad_edge_vec.stride(1),
            grad_coord.stride(0),
            grad_coord.stride(1),
            eps,
            r_inner,
            r_outer,
            HAS_INNER_CLAMP=bool(has_inner_clamp),
            BLOCK_EDGE=TRITON_EDGE_GEOMETRY_RBF_BLOCK_EDGE,
            num_warps=4,
        )
