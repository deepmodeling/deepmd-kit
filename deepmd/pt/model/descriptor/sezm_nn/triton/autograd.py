# SPDX-License-Identifier: LGPL-3.0-or-later
"""Autograd and public API for SeZM Triton kernels."""

from __future__ import (
    annotations,
)

from typing import (
    Any,
)

import torch
from torch import (
    Tensor,
)

from ..utils import (
    safe_norm,
)
from .constants import (
    SEZM_TRITON_AVAILABLE,
    TritonRotationMode,
)
from .dispatch import (
    coerce_rotation_mode,
    resolve_triton_rotation_mode,
)

if SEZM_TRITON_AVAILABLE:
    from . import custom_ops as _custom_ops  # noqa: F401


def _compute_cutoff_envelope_eager(
    *,
    r: Tensor,
    rcut: float,
    a: float,
    b: float,
    c: float,
    d: float,
    exponent: int,
) -> Tensor:
    """Reference eager evaluation of the C^3 cutoff envelope."""
    x = (r / rcut).clamp(min=0.0, max=1.0)
    poly = a + x * (b + x * (c + x * d))
    env = 1.0 + (x ** int(exponent)) * poly
    return env * (x < 1.0).to(dtype=r.dtype)


def _edge_geometry_rbf_eager(
    *,
    coord_flat: Tensor,
    center_coord_index: Tensor,
    neighbor_coord_index: Tensor,
    freqs: Tensor,
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
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Reference eager implementation of the edge geometry/RBF chain."""
    center_pos = coord_flat.index_select(0, center_coord_index)
    neighbor_pos = coord_flat.index_select(0, neighbor_coord_index)
    edge_vec = neighbor_pos - center_pos
    raw_len = safe_norm(edge_vec, float(eps))
    edge_len = raw_len
    if has_inner_clamp:
        delta = float(r_outer - r_inner)
        t = ((edge_len - float(r_inner)) / delta).clamp(0.0, 1.0)
        t2 = t * t
        t4 = t2 * t2
        h = t4 * (20.0 + t * (-45.0 + t * (36.0 - 10.0 * t)))
        clamped = float(r_inner) + delta * h
        edge_len = torch.where(edge_len >= float(r_outer), edge_len, clamped)
        scale = edge_len / raw_len
        edge_vec = edge_vec * scale
    edge_env = _compute_cutoff_envelope_eager(
        r=edge_len,
        rcut=float(rcut),
        a=float(edge_env_a),
        b=float(edge_env_b),
        c=float(edge_env_c),
        d=float(edge_env_d),
        exponent=int(edge_env_exponent),
    )
    radial_env = _compute_cutoff_envelope_eager(
        r=edge_len,
        rcut=float(rcut),
        a=float(radial_env_a),
        b=float(radial_env_b),
        c=float(radial_env_c),
        d=float(radial_env_d),
        exponent=int(radial_env_exponent),
    )
    freqs_row = freqs.view(1, -1)
    phase = edge_len * freqs_row
    edge_rbf = freqs_row * torch.sinc(phase / torch.pi) * radial_env
    return edge_vec, edge_len, edge_env, edge_rbf


def _extract_envelope_params(
    envelope: Any,
) -> tuple[float, float, float, float, float, int]:
    """Extract the polynomial envelope parameters from one SeZM module."""
    return (
        float(envelope.rcut),
        float(envelope.coeff_a),
        float(envelope.coeff_b),
        float(envelope.coeff_c),
        float(envelope.coeff_d),
        int(envelope.p),
    )


def _extract_edge_geometry_rbf_constants(
    *,
    edge_envelope: Any,
    radial_basis: Any,
    inner_clamp: Any,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    int,
    float,
    float,
    float,
    float,
    int,
    float,
    float,
    bool,
]:
    """Extract scalar constants used by the fused geometry/RBF chain."""
    (
        rcut,
        edge_env_a,
        edge_env_b,
        edge_env_c,
        edge_env_d,
        edge_env_exponent,
    ) = _extract_envelope_params(edge_envelope)
    (
        _,
        radial_env_a,
        radial_env_b,
        radial_env_c,
        radial_env_d,
        radial_env_exponent,
    ) = _extract_envelope_params(radial_basis.envelope)
    if inner_clamp is None:
        r_inner = 0.0
        r_outer = 0.0
        has_inner_clamp = False
    else:
        r_inner = float(inner_clamp.r_inner)
        r_outer = float(inner_clamp.r_outer)
        has_inner_clamp = True
    return (
        rcut,
        edge_env_a,
        edge_env_b,
        edge_env_c,
        edge_env_d,
        edge_env_exponent,
        radial_env_a,
        radial_env_b,
        radial_env_c,
        radial_env_d,
        radial_env_exponent,
        r_inner,
        r_outer,
        has_inner_clamp,
    )


def _rotate_to_local_eager(
    *,
    x: Tensor,
    src: Tensor,
    wigner: Tensor,
    coeff_index: Tensor,
    dim_full: int,
) -> Tensor:
    """Reference eager implementation for ``D_to_m @ x[src]``."""
    D_to_m = wigner[:, :dim_full, :dim_full].index_select(1, coeff_index)
    return torch.bmm(D_to_m, x.index_select(0, src))


def _rotate_back_eager(
    *,
    x_local: Tensor,
    wigner: Tensor,
    coeff_index: Tensor,
    dim_full: int,
) -> Tensor:
    """Reference eager implementation for ``Dt_from_m @ x_local``."""
    Dt_from_m = wigner[:, :dim_full, :dim_full].index_select(2, coeff_index)
    return torch.bmm(Dt_from_m, x_local)


def _resolve_rotation_mode_for_call(
    *,
    dim_full: int,
    coeff_index: Tensor,
    rotation_mode: int | TritonRotationMode | None,
) -> TritonRotationMode:
    """Resolve the effective dispatch mode for one public API call."""
    if rotation_mode is None:
        return resolve_triton_rotation_mode(
            dim_full=int(dim_full),
            reduced_dim=int(coeff_index.numel()),
        )
    return coerce_rotation_mode(rotation_mode)


if SEZM_TRITON_AVAILABLE:

    class _RotateToLocalFunction(torch.autograd.Function):
        """Autograd wrapper for the fused ``global -> local reduced`` rotation."""

        @staticmethod
        def forward(
            ctx: Any,
            x: Tensor,
            src: Tensor,
            wigner: Tensor,
            coeff_index: Tensor,
            dim_full: int,
            rotation_mode: int,
        ) -> Tensor:
            reduced_dim = int(coeff_index.numel())
            out = torch.empty(
                src.shape[0],
                reduced_dim,
                x.shape[2],
                dtype=x.dtype,
                device=x.device,
            )
            torch.ops.deepmd._kernel_sezm_rotate_to_local(
                x,
                src,
                wigner,
                coeff_index,
                out,
                dim_full,
                rotation_mode,
            )
            ctx.save_for_backward(x, src, wigner, coeff_index)
            ctx.dim_full = int(dim_full)
            ctx.rotation_mode = int(rotation_mode)
            return out

        @staticmethod
        def backward(
            ctx: Any,
            grad_out: Tensor,
        ) -> tuple[Tensor, None, Tensor, None, None, None]:
            x, src, wigner, coeff_index = ctx.saved_tensors
            dim_full = int(ctx.dim_full)
            rotation_mode = coerce_rotation_mode(int(ctx.rotation_mode))
            grad_out = grad_out.contiguous()
            grad_edge = torch.empty(
                src.shape[0],
                dim_full,
                x.shape[2],
                dtype=grad_out.dtype,
                device=grad_out.device,
            )
            torch.ops.deepmd._kernel_sezm_rotate_to_local_bwd_dx(
                grad_out,
                wigner,
                coeff_index,
                grad_edge,
                dim_full,
                int(rotation_mode),
            )
            grad_x = torch.zeros_like(x)
            grad_x.index_add_(0, src, grad_edge)

            if rotation_mode == TritonRotationMode.GENERIC_TILED:
                grad_rows = torch.empty(
                    src.shape[0],
                    coeff_index.numel(),
                    dim_full,
                    dtype=wigner.dtype,
                    device=grad_out.device,
                )
                torch.ops.deepmd._kernel_sezm_rotate_to_local_bwd_dw(
                    grad_out,
                    x,
                    src,
                    coeff_index,
                    grad_rows,
                    dim_full,
                    int(rotation_mode),
                )
                grad_wigner = torch.zeros_like(wigner)
                grad_wigner[:, coeff_index, :dim_full] = grad_rows
            else:
                grad_wigner = torch.zeros_like(wigner)
                torch.ops.deepmd._kernel_sezm_rotate_to_local_bwd_dw(
                    grad_out,
                    x,
                    src,
                    coeff_index,
                    grad_wigner,
                    dim_full,
                    int(rotation_mode),
                )
            return grad_x, None, grad_wigner, None, None, None

    class _RotateBackFunction(torch.autograd.Function):
        """Autograd wrapper for the fused ``local reduced -> global`` rotation."""

        @staticmethod
        def forward(
            ctx: Any,
            x_local: Tensor,
            wigner: Tensor,
            coeff_index: Tensor,
            dim_full: int,
            rotation_mode: int,
        ) -> Tensor:
            out = torch.empty(
                x_local.shape[0],
                dim_full,
                x_local.shape[2],
                dtype=x_local.dtype,
                device=x_local.device,
            )
            torch.ops.deepmd._kernel_sezm_rotate_back(
                x_local,
                wigner,
                coeff_index,
                out,
                dim_full,
                rotation_mode,
            )
            ctx.save_for_backward(x_local, wigner, coeff_index)
            ctx.dim_full = int(dim_full)
            ctx.rotation_mode = int(rotation_mode)
            return out

        @staticmethod
        def backward(
            ctx: Any,
            grad_out: Tensor,
        ) -> tuple[Tensor, Tensor, None, None, None]:
            x_local, wigner, coeff_index = ctx.saved_tensors
            dim_full = int(ctx.dim_full)
            rotation_mode = coerce_rotation_mode(int(ctx.rotation_mode))
            grad_out = grad_out.contiguous()
            grad_x_local = torch.empty_like(x_local)
            torch.ops.deepmd._kernel_sezm_rotate_back_bwd_dx(
                grad_out,
                wigner,
                coeff_index,
                grad_x_local,
                dim_full,
                int(rotation_mode),
            )

            if rotation_mode == TritonRotationMode.GENERIC_TILED:
                grad_cols = torch.empty(
                    x_local.shape[0],
                    dim_full,
                    coeff_index.numel(),
                    dtype=wigner.dtype,
                    device=grad_out.device,
                )
                torch.ops.deepmd._kernel_sezm_rotate_back_bwd_dw(
                    grad_out,
                    x_local,
                    coeff_index,
                    grad_cols,
                    dim_full,
                    int(rotation_mode),
                )
                grad_wigner = torch.zeros_like(wigner)
                grad_wigner[:, :dim_full, coeff_index] = grad_cols
            else:
                grad_wigner = torch.zeros_like(wigner)
                torch.ops.deepmd._kernel_sezm_rotate_back_bwd_dw(
                    grad_out,
                    x_local,
                    coeff_index,
                    grad_wigner,
                    dim_full,
                    int(rotation_mode),
                )
            return grad_x_local, grad_wigner, None, None, None

    class _EdgeGeometryRBFFunction(torch.autograd.Function):
        """Autograd wrapper for the fused edge geometry/RBF chain."""

        @staticmethod
        def forward(
            ctx: Any,
            coord_flat: Tensor,
            center_coord_index: Tensor,
            neighbor_coord_index: Tensor,
            freqs: Tensor,
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
        ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
            freq_flat = freqs.reshape(-1)
            num_edges = int(center_coord_index.shape[0])
            edge_vec = torch.empty(
                num_edges,
                3,
                dtype=coord_flat.dtype,
                device=coord_flat.device,
            )
            edge_len = torch.empty(
                num_edges,
                dtype=coord_flat.dtype,
                device=coord_flat.device,
            )
            edge_env = torch.empty(
                num_edges,
                dtype=coord_flat.dtype,
                device=coord_flat.device,
            )
            edge_rbf = torch.empty(
                num_edges,
                freq_flat.numel(),
                dtype=coord_flat.dtype,
                device=coord_flat.device,
            )
            torch.ops.deepmd._kernel_sezm_edge_geometry_rbf(
                coord_flat,
                center_coord_index,
                neighbor_coord_index,
                freq_flat,
                edge_vec,
                edge_len,
                edge_env,
                edge_rbf,
                float(eps),
                float(rcut),
                float(edge_env_a),
                float(edge_env_b),
                float(edge_env_c),
                float(edge_env_d),
                int(edge_env_exponent),
                float(radial_env_a),
                float(radial_env_b),
                float(radial_env_c),
                float(radial_env_d),
                int(radial_env_exponent),
                float(r_inner),
                float(r_outer),
                bool(has_inner_clamp),
            )
            ctx.save_for_backward(
                coord_flat,
                center_coord_index,
                neighbor_coord_index,
                freqs,
            )
            ctx.eps = float(eps)
            ctx.rcut = float(rcut)
            ctx.edge_env_a = float(edge_env_a)
            ctx.edge_env_b = float(edge_env_b)
            ctx.edge_env_c = float(edge_env_c)
            ctx.edge_env_d = float(edge_env_d)
            ctx.edge_env_exponent = int(edge_env_exponent)
            ctx.radial_env_a = float(radial_env_a)
            ctx.radial_env_b = float(radial_env_b)
            ctx.radial_env_c = float(radial_env_c)
            ctx.radial_env_d = float(radial_env_d)
            ctx.radial_env_exponent = int(radial_env_exponent)
            ctx.r_inner = float(r_inner)
            ctx.r_outer = float(r_outer)
            ctx.has_inner_clamp = bool(has_inner_clamp)
            return edge_vec, edge_len.unsqueeze(-1), edge_env.unsqueeze(-1), edge_rbf

        @staticmethod
        def backward(
            ctx: Any,
            grad_edge_vec: Tensor | None,
            grad_edge_len: Tensor | None,
            grad_edge_env: Tensor | None,
            grad_edge_rbf: Tensor | None,
        ) -> tuple[
            Tensor,
            None,
            None,
            Tensor,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ]:
            coord_flat, center_coord_index, neighbor_coord_index, freqs = (
                ctx.saved_tensors
            )
            num_edges = int(center_coord_index.shape[0])
            freq_flat = freqs.reshape(-1)

            if grad_edge_vec is None:
                grad_edge_vec = torch.zeros(
                    num_edges,
                    3,
                    dtype=coord_flat.dtype,
                    device=coord_flat.device,
                )
            else:
                grad_edge_vec = grad_edge_vec.contiguous()
            if grad_edge_len is None:
                grad_edge_len = torch.zeros(
                    num_edges,
                    dtype=coord_flat.dtype,
                    device=coord_flat.device,
                )
            else:
                grad_edge_len = grad_edge_len.contiguous().squeeze(-1)
            if grad_edge_env is None:
                grad_edge_env = torch.zeros(
                    num_edges,
                    dtype=coord_flat.dtype,
                    device=coord_flat.device,
                )
            else:
                grad_edge_env = grad_edge_env.contiguous().squeeze(-1)
            if grad_edge_rbf is None:
                grad_edge_rbf = torch.zeros(
                    num_edges,
                    freq_flat.numel(),
                    dtype=coord_flat.dtype,
                    device=coord_flat.device,
                )
            else:
                grad_edge_rbf = grad_edge_rbf.contiguous()

            grad_r_total = torch.zeros(
                num_edges,
                dtype=coord_flat.dtype,
                device=coord_flat.device,
            )
            grad_freq = torch.zeros(
                freq_flat.numel(),
                dtype=freq_flat.dtype,
                device=coord_flat.device,
            )
            torch.ops.deepmd._kernel_sezm_edge_geometry_rbf_bwd_accum(
                grad_edge_len,
                grad_edge_env,
                grad_edge_rbf,
                coord_flat,
                center_coord_index,
                neighbor_coord_index,
                freq_flat,
                grad_r_total,
                grad_freq,
                float(ctx.eps),
                float(ctx.rcut),
                float(ctx.edge_env_a),
                float(ctx.edge_env_b),
                float(ctx.edge_env_c),
                float(ctx.edge_env_d),
                int(ctx.edge_env_exponent),
                float(ctx.radial_env_a),
                float(ctx.radial_env_b),
                float(ctx.radial_env_c),
                float(ctx.radial_env_d),
                int(ctx.radial_env_exponent),
                float(ctx.r_inner),
                float(ctx.r_outer),
                bool(ctx.has_inner_clamp),
            )
            grad_coord = torch.zeros_like(coord_flat)
            torch.ops.deepmd._kernel_sezm_edge_geometry_rbf_bwd_coord(
                grad_edge_vec,
                grad_r_total,
                coord_flat,
                center_coord_index,
                neighbor_coord_index,
                grad_coord,
                float(ctx.eps),
                float(ctx.r_inner),
                float(ctx.r_outer),
                bool(ctx.has_inner_clamp),
            )
            return (
                grad_coord,
                None,
                None,
                grad_freq.view_as(freqs),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )


def rotate_to_local_triton(
    x: Tensor,
    src: Tensor,
    wigner: Tensor,
    coeff_index: Tensor,
    dim_full: int,
    rotation_mode: int | TritonRotationMode | None = None,
) -> Tensor:
    """
    Apply the fused ``global -> local reduced`` rotation.

    Parameters
    ----------
    x
        Node features with shape ``(N, D, C)``.
    src
        Source-node indices with shape ``(E,)``.
    wigner
        Packed Wigner matrices with shape ``(E, D, D)``.
    coeff_index
        Reduced-layout row indices with shape ``(D_m,)``.
    dim_full
        Full packed SO(3) dimension.
    rotation_mode
        Optional pre-resolved dispatch mode.

    Returns
    -------
    Tensor
        Rotated reduced-layout edge features with shape ``(E, D_m, C)``.
    """
    if not SEZM_TRITON_AVAILABLE:
        raise RuntimeError("SeZM Triton kernels are not available in this environment.")
    src = src.contiguous()
    coeff_index = coeff_index.contiguous()
    resolved_mode = _resolve_rotation_mode_for_call(
        dim_full=int(dim_full),
        coeff_index=coeff_index,
        rotation_mode=rotation_mode,
    )
    if resolved_mode == TritonRotationMode.EAGER_REFERENCE:
        return _rotate_to_local_eager(
            x=x,
            src=src,
            wigner=wigner,
            coeff_index=coeff_index,
            dim_full=int(dim_full),
        )
    return _RotateToLocalFunction.apply(
        x,
        src,
        wigner,
        coeff_index,
        int(dim_full),
        int(resolved_mode),
    )


def rotate_back_triton(
    x_local: Tensor,
    wigner: Tensor,
    coeff_index: Tensor,
    dim_full: int,
    rotation_mode: int | TritonRotationMode | None = None,
) -> Tensor:
    """
    Apply the fused ``local reduced -> global`` rotation.

    Parameters
    ----------
    x_local
        Reduced-layout edge features with shape ``(E, D_m, C)``.
    wigner
        Packed Wigner matrices with shape ``(E, D, D)``.
    coeff_index
        Reduced-layout column indices with shape ``(D_m,)``.
    dim_full
        Full packed SO(3) dimension.
    rotation_mode
        Optional pre-resolved dispatch mode.

    Returns
    -------
    Tensor
        Lifted global-layout edge features with shape ``(E, D, C)``.
    """
    if not SEZM_TRITON_AVAILABLE:
        raise RuntimeError("SeZM Triton kernels are not available in this environment.")
    coeff_index = coeff_index.contiguous()
    resolved_mode = _resolve_rotation_mode_for_call(
        dim_full=int(dim_full),
        coeff_index=coeff_index,
        rotation_mode=rotation_mode,
    )
    if resolved_mode == TritonRotationMode.EAGER_REFERENCE:
        return _rotate_back_eager(
            x_local=x_local,
            wigner=wigner,
            coeff_index=coeff_index,
            dim_full=int(dim_full),
        )
    return _RotateBackFunction.apply(
        x_local,
        wigner,
        coeff_index,
        int(dim_full),
        int(resolved_mode),
    )


def edge_geometry_rbf_triton(
    *,
    coord_flat: Tensor,
    center_coord_index: Tensor,
    neighbor_coord_index: Tensor,
    edge_envelope: Any,
    radial_basis: Any,
    eps: float,
    inner_clamp: Any,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Apply the fused edge geometry/RBF chain with eager fallback."""
    (
        rcut,
        edge_env_a,
        edge_env_b,
        edge_env_c,
        edge_env_d,
        edge_env_exponent,
        radial_env_a,
        radial_env_b,
        radial_env_c,
        radial_env_d,
        radial_env_exponent,
        r_inner,
        r_outer,
        has_inner_clamp,
    ) = _extract_edge_geometry_rbf_constants(
        edge_envelope=edge_envelope,
        radial_basis=radial_basis,
        inner_clamp=inner_clamp,
    )
    center_coord_index = center_coord_index.contiguous()
    neighbor_coord_index = neighbor_coord_index.contiguous()
    freqs = radial_basis.adam_freqs.contiguous()
    if (
        center_coord_index.numel() == 0
        or not SEZM_TRITON_AVAILABLE
        or coord_flat.device.type != "cuda"
        or coord_flat.dtype not in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return _edge_geometry_rbf_eager(
            coord_flat=coord_flat,
            center_coord_index=center_coord_index,
            neighbor_coord_index=neighbor_coord_index,
            freqs=freqs,
            eps=float(eps),
            rcut=rcut,
            edge_env_a=edge_env_a,
            edge_env_b=edge_env_b,
            edge_env_c=edge_env_c,
            edge_env_d=edge_env_d,
            edge_env_exponent=edge_env_exponent,
            radial_env_a=radial_env_a,
            radial_env_b=radial_env_b,
            radial_env_c=radial_env_c,
            radial_env_d=radial_env_d,
            radial_env_exponent=radial_env_exponent,
            r_inner=r_inner,
            r_outer=r_outer,
            has_inner_clamp=has_inner_clamp,
        )
    return _EdgeGeometryRBFFunction.apply(
        coord_flat,
        center_coord_index,
        neighbor_coord_index,
        freqs,
        float(eps),
        rcut,
        edge_env_a,
        edge_env_b,
        edge_env_c,
        edge_env_d,
        edge_env_exponent,
        radial_env_a,
        radial_env_b,
        radial_env_c,
        radial_env_d,
        radial_env_exponent,
        r_inner,
        r_outer,
        has_inner_clamp,
    )
