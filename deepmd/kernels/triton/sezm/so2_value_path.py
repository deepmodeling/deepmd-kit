# SPDX-License-Identifier: LGPL-3.0-or-later
# pyright: reportMissingImports=false
# ruff: noqa: ANN001, ANN202, RUF005
"""Fused Triton SO(2) value path for the SeZM/DPA4 descriptor.

The SO(2) value path of :class:`SO2Convolution` -- rotate-to-local, radial
degree mixing, the multi-layer gated SO(2) mixing stack, and the cross-focus
competition -- dominates both the time and the activation memory of a SeZM
inference step.  This module fuses it into two functional operators:

``sezm_triton::so2_rotate_mix``
    One kernel per edge: gathers the source node features, applies the
    block-diagonal Wigner rotation over the structural non-zeros only (kept in
    registers), applies the edge-conditioned radial degree mixing, and stores
    the result directly in the focus-major flat layout ``(F, E, ROW)`` with
    ``ROW = (3 * lmax + 1) * Cf`` that the mixing stack consumes.  The rotated
    pre-mix intermediate is never materialized.  The backward recomputes the
    rotation in registers (nothing is saved besides the operator inputs) and
    reduces the per-edge node gradient with a contention-free CSR segment sum
    (``sezm_triton::segment_sum``) instead of ``index_add_``: at typical
    neighbor counts (~10^2 colliding edges per atom) row-atomic scatters
    serialize and are several times slower.  On narrow hidden widths the
    backward dispatches to an edge-block kernel that replaces the per-edge
    cross-lane ``tl.sum`` chains with batched axis-1 reductions; the win-list
    table :func:`~.tile_configs.rotate_mix_bwd_block_config` decides per
    ``(C_wide, lmax)`` key.

``sezm_triton::so2_mixing_stack``
    The whole mixing stack -- ``n_layers - 1`` gated layers followed by one
    identity layer, with the optional cross-focus competition weight folded
    into the final store -- as a single operator.  Keeping the inter-layer
    activations inside the op (ordinary caching-allocator tensors) instead of
    graph-level intermediates minimizes the compiled graph's activation
    footprint; only the tensors the backward needs surface as outputs (the
    stacked gated-layer pre-activations ``z_all`` and the result itself).
    Gate sigmoids are recomputed in the backward from the saved ``z``.

Per gated layer the stack runs three launches: a pure block GEMM for the
``m = 0`` block, a pointwise kernel evaluating the sigmoid gates from the
``l = 0`` scalar slice and finishing the ``m = 0`` rows, and a ``|m| = 1``
block GEMM with the gate/residual epilogue fused in.  The final identity
layer is two GEMM launches whose epilogue adds the residual, applies the
competition weight, and stores straight into the edge-major ``(E, F, ROW)``
layout the fused attention aggregation consumes -- no reassembly copy.

Layout contract
---------------
The focus-major activation ``(F, E, ROW)`` orders each row m-major:
subtiles ``r = 0..lmax`` hold ``m = 0`` degrees ``l = r``; subtiles
``r = lmax+1..2*lmax`` and ``r = 2*lmax+1..3*lmax`` hold the ``m = -1`` and
``m = +1`` degrees ``l = 1..lmax``.  The sigmoid gate group of subtile
``r > 0`` is ``(r - 1) % lmax`` for the ``m = 0`` rows and
``(r - lmax - 1) % lmax`` for the ``|m| = 1`` rows, matching
:class:`GatedActivation` with one gate group per degree ``l >= 1``.

Weight passing discipline
-------------------------
Per-layer weights are stacked along dim 0 -- ``(n_layers, F, M, M)`` -- and
kernels select a layer through an integer ``layer`` argument.  Slicing the
stack in Python and handing ``select`` views to the Triton higher-order op
must be avoided: Inductor's ``decompose_triton_kernel_wrapper_functional``
re-traces the op body with ``replace_by_example`` and asserts node-for-node
graph equality, which view-typed kernel arguments break (clone insertion
differs between the two traces on PyTorch 2.11).

Numerics
--------
Every ``tl.dot`` runs with ``input_precision="ieee"`` (no TF32), keeping the
potential-energy surface smooth.  fp32 is the supported precision; the
factory refuses non-fp32 weights rather than silently down-casting.  Launch
tile choices never affect results (they change the schedule, not any
reduction order); the swept tables live in :mod:`.tile_configs`.  At
``DP_TRITON_INFER >= 3`` the mixing stack is replaced by the fp16x3
tensor-core operator of :mod:`.so2_stack_fp16x3` on validated shapes, the
one deliberate exception to the exact-fp32 contract.

Wide-channel regime
-------------------
For ``Cf >= GATE_BMM_MIN_FOCUS_DIM`` the per-group ``CP x CP`` register dot
of the gate forward/backward spills (``CP`` is ``Cf`` padded to a power of
two), so the sigmoid projection and the gate-logit contraction run as cuBLAS
batched matmuls inside the op while the Triton kernels keep the pointwise
work.  Non-power-of-two focus widths (e.g. ``Cf = 96``) are supported by the
same padding plus column masks; block GEMM kernels handle any ``Cf`` through
their edge masks, and their K loops stay exact because ``(lmax + 1) * Cf``
and ``2 * lmax * Cf`` remain multiples of the K tile.
"""

from __future__ import (
    annotations,
)

from typing import (
    TYPE_CHECKING,
)

import torch
from torch import (
    Tensor,
)
from torch.library import (
    wrap_triton,
)

from deepmd.pt.model.descriptor.sezm_nn.indexing import (
    build_m_major_index,
)

from .tile_configs import (
    GATE_BMM_MIN_FOCUS_DIM,
    gate_config,
    point_config,
    recompute_config,
    rotate_mix_bwd_block_config,
    rotate_mix_fwd_config,
    stack_fp16x3_configs,
)

if TYPE_CHECKING:
    from deepmd.pt.model.descriptor.sezm_nn.edge_cache import (
        EdgeFeatureCache,
    )
    from deepmd.pt.model.descriptor.sezm_nn.so2 import (
        SO2Convolution,
    )

__all__ = [
    "SO2_VALUE_PATH_TRITON_AVAILABLE",
    "make_triton_value_path",
]

try:
    import triton
    import triton.language as tl

    SO2_VALUE_PATH_TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without triton
    SO2_VALUE_PATH_TRITON_AVAILABLE = False

_SUPPORTED_FOCUS_DIMS = (32, 64, 96, 128)
_MAX_LMAX = 6
_MAX_MIXER_RANK = 4

# Block GEMM tiling: 25 TFLOPS (~58% of the H20 FFMA peak) on the deployed
# block widths, at the measured efficiency ceiling of IEEE-fp32 tl.dot tiling.
# The configuration was confirmed optimal (or within 1%) across the whole
# swept (focus_dim, lmax) family, so it is a constant rather than a table.
_GEMM_CONFIG = (64, 64, 32, 4, 2)  # (BLOCK_M, BLOCK_N, BLOCK_K, warps, stages)
_ROTATE_MIX_BWD_CONFIG = (1, 2)  # per-edge backward (warps, stages)


# ======================================================================
# Eager reference / fallback implementations
# ======================================================================
def _rotate_mix_reference(
    x: Tensor,
    src: Tensor,
    wigner: Tensor,
    kc: Tensor,
    cb: Tensor,
    lmax: int,
    n_focus: int,
    rank: int,
) -> Tensor:
    """Eager ground truth for ``so2_rotate_mix``.

    Rotates the gathered source features into the m-major ``mmax == 1``
    reduced layout, applies the radial degree mixing (rank-``R`` factorized
    kernel, or the degree-wise multiply when ``rank == 0``), and returns the
    focus-major ``(F, E, ROW)`` activation.
    """
    n_edge = src.shape[0]
    c_wide = x.shape[2]
    focus_dim = c_wide // n_focus
    dim = (lmax + 1) ** 2
    n_deg = lmax + 1
    reduced = 3 * lmax + 1
    coeff = build_m_major_index(lmax, 1, device=x.device)
    d_to_m = wigner[:, :dim, :dim].index_select(1, coeff)
    x_local = torch.bmm(d_to_m, x.index_select(0, src))  # (E, reduced, C_wide)
    if rank == 0:
        # kc holds per-degree radial features (E, lmax+1, C_wide); each reduced
        # row is multiplied by the feature of its degree.
        rad = kc.view(n_edge, n_deg, c_wide)
        degree = torch.tensor(
            list(range(n_deg)) + 2 * list(range(1, n_deg)),
            device=x.device,
            dtype=torch.long,
        )
        y = x_local * rad.index_select(1, degree)
    else:
        kc_v = kc.view(n_edge, -1, rank)
        k0 = kc_v[:, : n_deg * n_deg].view(n_edge, n_deg, n_deg, rank)
        k1 = kc_v[:, n_deg * n_deg :].view(n_edge, lmax, lmax, rank)
        cb_v = cb.view(rank, c_wide)
        y = torch.empty_like(x_local)
        y[:, :n_deg] = torch.einsum("eior,eic,rc->eoc", k0, x_local[:, :n_deg], cb_v)
        y[:, n_deg : n_deg + lmax] = torch.einsum(
            "eior,eic,rc->eoc", k1, x_local[:, n_deg : n_deg + lmax], cb_v
        )
        y[:, n_deg + lmax :] = torch.einsum(
            "eior,eic,rc->eoc", k1, x_local[:, n_deg + lmax :], cb_v
        )
    return (
        y.view(n_edge, reduced, n_focus, focus_dim)
        .permute(2, 0, 1, 3)
        .reshape(n_focus, n_edge, reduced * focus_dim)
        .contiguous()
    )


def _rotate_mix_backward_reference(
    grad_u: Tensor,
    x: Tensor,
    src: Tensor,
    wigner: Tensor,
    kc: Tensor,
    cb: Tensor,
    lmax: int,
    n_focus: int,
    rank: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Closed-form eager backward of ``so2_rotate_mix``.

    Returns ``(grad_x_edge, grad_wigner, grad_kc)`` where ``grad_x_edge`` is
    the per-edge source gradient (the caller segment-sums it over ``src``).
    A closed form (not a nested ``autograd.grad``) is required because the
    backward operator is dispatched under ``_AutoDispatchBelowAutograd`` when
    the frozen force graph replays under ``torch.no_grad``.
    """
    n_edge = src.shape[0]
    c_wide = x.shape[2]
    focus_dim = c_wide // n_focus
    dim = (lmax + 1) ** 2
    n_deg = lmax + 1
    reduced = 3 * lmax + 1
    coeff = build_m_major_index(lmax, 1, device=x.device)
    d_to_m = wigner[:, :dim, :dim].index_select(1, coeff)
    x_src = x.index_select(0, src)
    x_local = torch.bmm(d_to_m, x_src)  # (E, reduced, C_wide)

    g_y = (
        grad_u.view(n_focus, n_edge, reduced, focus_dim)
        .permute(1, 2, 0, 3)
        .reshape(n_edge, reduced, c_wide)
    )
    if rank == 0:
        rad = kc.view(n_edge, n_deg, c_wide)
        degree = torch.tensor(
            list(range(n_deg)) + 2 * list(range(1, n_deg)),
            device=x.device,
            dtype=torch.long,
        )
        g_local = g_y * rad.index_select(1, degree)
        prod = g_y * x_local
        grad_kc = prod[:, :n_deg].clone()
        grad_kc[:, 1:] += prod[:, n_deg : n_deg + lmax]
        grad_kc[:, 1:] += prod[:, n_deg + lmax :]
        grad_kc = grad_kc.reshape(kc.shape)
    else:
        kc_v = kc.view(n_edge, -1, rank)
        k0 = kc_v[:, : n_deg * n_deg].view(n_edge, n_deg, n_deg, rank)
        k1 = kc_v[:, n_deg * n_deg :].view(n_edge, lmax, lmax, rank)
        cb_v = cb.view(rank, c_wide)
        g_local = torch.empty_like(g_y)
        g_local[:, :n_deg] = torch.einsum("eior,eoc,rc->eic", k0, g_y[:, :n_deg], cb_v)
        g_local[:, n_deg : n_deg + lmax] = torch.einsum(
            "eior,eoc,rc->eic", k1, g_y[:, n_deg : n_deg + lmax], cb_v
        )
        g_local[:, n_deg + lmax :] = torch.einsum(
            "eior,eoc,rc->eic", k1, g_y[:, n_deg + lmax :], cb_v
        )
        gk0 = torch.einsum("eoc,eic,rc->eior", g_y[:, :n_deg], x_local[:, :n_deg], cb_v)
        gk1 = torch.einsum(
            "eoc,eic,rc->eior",
            g_y[:, n_deg : n_deg + lmax],
            x_local[:, n_deg : n_deg + lmax],
            cb_v,
        ) + torch.einsum(
            "eoc,eic,rc->eior",
            g_y[:, n_deg + lmax :],
            x_local[:, n_deg + lmax :],
            cb_v,
        )
        grad_kc = torch.cat(
            [gk0.reshape(n_edge, -1), gk1.reshape(n_edge, -1)], dim=1
        ).reshape(kc.shape)

    grad_x_edge = torch.bmm(d_to_m.transpose(1, 2), g_local)  # (E, D, C_wide)
    grad_rows = torch.bmm(g_local, x_src.transpose(1, 2))  # (E, reduced, D)
    grad_block = wigner.new_zeros(n_edge, dim, dim)
    grad_block.index_copy_(1, coeff, grad_rows)
    grad_wigner = torch.zeros_like(wigner)
    grad_wigner[:, :dim, :dim] = grad_block
    return grad_x_edge, grad_wigner, grad_kc


def _mixing_stack_reference(
    u0: Tensor,
    alpha: Tensor,
    w0_all: Tensor,
    w1_all: Tensor,
    gw_all: Tensor,
    lmax: int,
    focus_dim: int,
    apply_alpha: bool,
) -> tuple[Tensor, Tensor]:
    """Eager ground truth for ``so2_mixing_stack``.

    Returns the edge-major output ``(E, F, ROW)`` and the stacked gated-layer
    pre-activations ``(n_gated, F, E, ROW)``.
    """
    n_focus, n_edge, row = u0.shape
    m0 = (lmax + 1) * focus_dim
    n_gated = gw_all.shape[0]
    u = u0
    z_saved = []
    for layer in range(n_gated):
        z0 = torch.bmm(u[:, :, :m0], w0_all[layer])
        z1 = torch.bmm(u[:, :, m0:], w1_all[layer])
        z_saved.append(torch.cat([z0, z1], dim=-1))
        z_scalar = z0[:, :, :focus_dim]
        sig = torch.sigmoid(torch.bmm(z_scalar, gw_all[layer]))  # (F, E, lmax*Cf)
        act = torch.cat(
            [
                z_scalar * torch.sigmoid(z_scalar),
                z0[:, :, focus_dim:] * sig,
                z1 * sig.repeat(1, 1, 2),
            ],
            dim=-1,
        )
        u = u + act
    out = u.clone()
    out[:, :, :m0] += torch.bmm(u[:, :, :m0], w0_all[n_gated])
    out[:, :, m0:] += torch.bmm(u[:, :, m0:], w1_all[n_gated])
    if apply_alpha:
        out = out * alpha.transpose(0, 1).unsqueeze(-1).to(out.dtype)
    x_local = out.permute(1, 0, 2).contiguous()
    z_all = (
        torch.stack(z_saved) if n_gated > 0 else u0.new_empty(0, n_focus, n_edge, row)
    )
    return x_local, z_all


def _mixing_stack_backward_reference(
    grad_out: Tensor,
    x_local: Tensor,
    z_all: Tensor,
    alpha: Tensor,
    w0t_all: Tensor,
    w1t_all: Tensor,
    gw_all: Tensor,
    gwt_all: Tensor,
    lmax: int,
    focus_dim: int,
    apply_alpha: bool,
) -> tuple[Tensor, Tensor]:
    """Closed-form eager backward of ``so2_mixing_stack``.

    Returns ``(grad_u0, grad_alpha)``; ``grad_alpha`` is meaningful only when
    ``apply_alpha`` is set (the identity ``grad_alpha = sum(grad * out) /
    alpha`` is exact because the final store is a plain scale).
    """
    n_gated = gw_all.shape[0]
    m0 = (lmax + 1) * focus_dim
    g_edge = grad_out  # (E, F, ROW)
    if apply_alpha:
        grad_alpha = (g_edge * x_local).sum(dim=-1) / alpha.clamp_min(1e-12)
        g_edge = g_edge * alpha.unsqueeze(-1).to(g_edge.dtype)
    else:
        grad_alpha = torch.zeros_like(alpha)
    g = g_edge.permute(1, 0, 2)  # (F, E, ROW)
    g_cur = g.clone()
    g_cur[:, :, :m0] += torch.bmm(g[:, :, :m0], w0t_all[n_gated])
    g_cur[:, :, m0:] += torch.bmm(g[:, :, m0:], w1t_all[n_gated])
    for layer in range(n_gated - 1, -1, -1):
        z = z_all[layer]
        z0, z1 = z[:, :, :m0], z[:, :, m0:]
        z_scalar = z0[:, :, :focus_dim]
        sig = torch.sigmoid(torch.bmm(z_scalar, gw_all[layer]))
        sig2 = sig.repeat(1, 1, 2)
        s0 = torch.sigmoid(z_scalar)
        gz0 = torch.cat(
            [
                g_cur[:, :, :focus_dim] * s0 * (1.0 + z_scalar * (1.0 - s0)),
                g_cur[:, :, focus_dim:m0] * sig,
            ],
            dim=-1,
        )
        gz1 = g_cur[:, :, m0:] * sig2
        g_sig = (g_cur[:, :, focus_dim:m0] * z0[:, :, focus_dim:]).view(*sig.shape) + (
            g_cur[:, :, m0:] * z1
        ).view(sig.shape[0], sig.shape[1], 2, -1).sum(2)
        g_logit = g_sig * sig * (1.0 - sig)
        gz0 = torch.cat(
            [
                gz0[:, :, :focus_dim] + torch.bmm(g_logit, gwt_all[layer]),
                gz0[:, :, focus_dim:],
            ],
            dim=-1,
        )
        g_next = g_cur.clone()
        g_next[:, :, :m0] += torch.bmm(gz0, w0t_all[layer])
        g_next[:, :, m0:] += torch.bmm(gz1, w1t_all[layer])
        g_cur = g_next
    return g_cur, grad_alpha


# ======================================================================
# Triton kernels
# ======================================================================
if SO2_VALUE_PATH_TRITON_AVAILABLE:

    @triton.jit
    def _rotate_mix_fwd_kernel(
        x_ptr,  # (N, D, C_wide), strides (x_sn, x_sd, 1)
        src_ptr,  # (E,)
        w_ptr,  # (E, D, D) block-diagonal Wigner-D, contiguous
        kc_ptr,  # (E, KSZ * RANK) compact kernel, or (E, L+1, CW) when RANK == 0
        cb_ptr,  # (RANK, CW) channel basis (unread when RANK == 0)
        u_ptr,  # (F, E, ROW) focus-major output
        n_edge,
        x_sn,
        x_sd,
        L: tl.constexpr,
        CF: tl.constexpr,
        CW: tl.constexpr,  # true C_wide; BC = next_power_of_2(CW) lanes with mask
        BC: tl.constexpr,
        RANK: tl.constexpr,
    ):
        """One program per edge, channels vectorized.

        Phase 1 rotates the gathered source features over the structural
        block-diagonal non-zeros only, holding the ``3 * L + 1`` reduced rows
        in registers.  Phase 2 applies the low-rank degree mixing
        ``K_eff[i, o, c] = sum_r kc[i, o, r] * cb[r, c]`` (for ``RANK == 1``
        the channel basis factors out of the degree contraction and is applied
        once per output row) and stores focus-major with channel decode
        ``c = f * CF + cf``.  ``RANK == 0`` is the mixer-free variant: each
        reduced row is multiplied by the radial feature of its degree.
        """
        NS0: tl.constexpr = L + 1
        RED: tl.constexpr = 3 * L + 1
        DIM: tl.constexpr = (L + 1) * (L + 1)
        ROW: tl.constexpr = RED * CF

        edge = tl.program_id(0).to(tl.int64)
        chan = tl.arange(0, BC)
        cmask = chan < CW
        src = tl.load(src_ptr + edge).to(tl.int64)
        x_base = x_ptr + src * x_sn
        d_base = w_ptr + edge * DIM * DIM

        # === Phase 1. Rotate to the local frame (registers) ===
        xrows = ()
        for r in tl.static_range(DIM):
            xrows = xrows + (
                tl.load(x_base + r * x_sd + chan, mask=cmask, other=0.0).to(tl.float32),
            )
        rows0 = ()
        rows_m = ()
        rows_p = ()
        for l in tl.static_range(L + 1):
            base = l * l
            r0 = base + l
            acc0 = tl.zeros((BC,), dtype=tl.float32)
            accm = tl.zeros((BC,), dtype=tl.float32)
            accp = tl.zeros((BC,), dtype=tl.float32)
            for j in tl.static_range(2 * l + 1):
                xv = xrows[l * l + j]
                acc0 += tl.load(d_base + r0 * DIM + base + j) * xv
                if l >= 1:
                    accm += tl.load(d_base + (r0 - 1) * DIM + base + j) * xv
                    accp += tl.load(d_base + (r0 + 1) * DIM + base + j) * xv
            rows0 = rows0 + (acc0,)
            if l >= 1:
                rows_m = rows_m + (accm,)
                rows_p = rows_p + (accp,)
        xl = rows0 + rows_m + rows_p

        # === Phase 2. Degree mix (or degree-wise multiply), store focus-major ===
        f_off = (chan // CF).to(tl.int64) * n_edge * ROW + edge * ROW + (chan % CF)
        if RANK == 0:
            rad_base = kc_ptr + edge * NS0 * CW
            for o in tl.static_range(NS0):
                rad = tl.load(rad_base + o * CW + chan, mask=cmask, other=0.0).to(
                    tl.float32
                )
                tl.store(u_ptr + f_off + o * CF, xl[o] * rad, mask=cmask)
            for o in tl.static_range(L):
                rad = tl.load(rad_base + (o + 1) * CW + chan, mask=cmask, other=0.0).to(
                    tl.float32
                )
                tl.store(u_ptr + f_off + (NS0 + o) * CF, xl[NS0 + o] * rad, mask=cmask)
                tl.store(
                    u_ptr + f_off + (NS0 + L + o) * CF,
                    xl[NS0 + L + o] * rad,
                    mask=cmask,
                )
            return
        cb = ()
        for r in tl.static_range(RANK):
            cb = cb + (
                tl.load(cb_ptr + r * CW + chan, mask=cmask, other=0.0).to(tl.float32),
            )
        kc_base = kc_ptr + edge * (NS0 * NS0 + L * L) * RANK
        for o in tl.static_range(NS0):
            acc = tl.zeros((BC,), dtype=tl.float32)
            for i in tl.static_range(NS0):
                if RANK == 1:
                    acc += tl.load(kc_base + i * NS0 + o) * xl[i]
                else:
                    keff = tl.zeros((BC,), dtype=tl.float32)
                    for r in tl.static_range(RANK):
                        keff += tl.load(kc_base + (i * NS0 + o) * RANK + r) * cb[r]
                    acc += keff * xl[i]
            if RANK == 1:
                acc = acc * cb[0]
            tl.store(u_ptr + f_off + o * CF, acc, mask=cmask)
        for o in tl.static_range(L):
            accn = tl.zeros((BC,), dtype=tl.float32)
            accq = tl.zeros((BC,), dtype=tl.float32)
            for i in tl.static_range(L):
                if RANK == 1:
                    k_val = tl.load(kc_base + NS0 * NS0 + i * L + o)
                    accn += k_val * xl[NS0 + i]
                    accq += k_val * xl[NS0 + L + i]
                else:
                    keff = tl.zeros((BC,), dtype=tl.float32)
                    for r in tl.static_range(RANK):
                        keff += (
                            tl.load(kc_base + (NS0 * NS0 + i * L + o) * RANK + r)
                            * cb[r]
                        )
                    accn += keff * xl[NS0 + i]
                    accq += keff * xl[NS0 + L + i]
            if RANK == 1:
                accn = accn * cb[0]
                accq = accq * cb[0]
            tl.store(u_ptr + f_off + (NS0 + o) * CF, accn, mask=cmask)
            tl.store(u_ptr + f_off + (NS0 + L + o) * CF, accq, mask=cmask)

    @triton.jit
    def _rotate_mix_bwd_kernel(
        gu_ptr,  # (F, E, ROW) upstream gradient (focus-major)
        x_ptr,
        src_ptr,
        w_ptr,
        kc_ptr,
        cb_ptr,
        gxe_ptr,  # (E, D, CW) per-edge node gradient (segment-summed by the caller)
        gw_ptr,  # (E, D, D) Wigner gradient (structural non-zeros; pre-zeroed)
        gkc_ptr,  # gradient of kc, same layout as kc
        n_edge,
        x_sn,
        x_sd,
        L: tl.constexpr,
        CF: tl.constexpr,
        CW: tl.constexpr,
        BC: tl.constexpr,
        RANK: tl.constexpr,
    ):
        """Backward of the fused front end (one program per edge).

        The rotated pre-mix rows are recomputed from ``x`` / ``W`` in
        registers (the program reads both anyway), so the forward saves no
        per-edge intermediate.  The node gradient is written densely per edge
        and reduced by a segment sum outside: a direct row-atomic scatter
        serializes on the colliding edges of each atom.  ``RANK == 0``: the
        degree-kernel phase becomes the degree-wise product rule on the radial
        features.
        """
        NS0: tl.constexpr = L + 1
        RED: tl.constexpr = 3 * L + 1
        DIM: tl.constexpr = (L + 1) * (L + 1)
        ROW: tl.constexpr = RED * CF

        edge = tl.program_id(0).to(tl.int64)
        chan = tl.arange(0, BC)
        cmask = chan < CW
        src = tl.load(src_ptr + edge).to(tl.int64)
        cb = ()
        for r in tl.static_range(RANK):
            cb = cb + (
                tl.load(cb_ptr + r * CW + chan, mask=cmask, other=0.0).to(tl.float32),
            )
        x_base = x_ptr + src * x_sn
        d_base = w_ptr + edge * DIM * DIM
        if RANK == 0:
            kc_base = kc_ptr + edge * NS0 * CW
            gkc_base = gkc_ptr + edge * NS0 * CW
        else:
            kc_base = kc_ptr + edge * (NS0 * NS0 + L * L) * RANK
            gkc_base = gkc_ptr + edge * (NS0 * NS0 + L * L) * RANK
        f_off = (chan // CF).to(tl.int64) * n_edge * ROW + edge * ROW + (chan % CF)

        # === Phase 0. Recompute the rotated rows; load the upstream rows ===
        xrows = ()
        for r in tl.static_range(DIM):
            xrows = xrows + (
                tl.load(x_base + r * x_sd + chan, mask=cmask, other=0.0).to(tl.float32),
            )
        rows0 = ()
        rows_m = ()
        rows_p = ()
        for l in tl.static_range(L + 1):
            base = l * l
            r0 = base + l
            acc0 = tl.zeros((BC,), dtype=tl.float32)
            accm = tl.zeros((BC,), dtype=tl.float32)
            accp = tl.zeros((BC,), dtype=tl.float32)
            for j in tl.static_range(2 * l + 1):
                xv = xrows[l * l + j]
                acc0 += tl.load(d_base + r0 * DIM + base + j) * xv
                if l >= 1:
                    accm += tl.load(d_base + (r0 - 1) * DIM + base + j) * xv
                    accp += tl.load(d_base + (r0 + 1) * DIM + base + j) * xv
            rows0 = rows0 + (acc0,)
            if l >= 1:
                rows_m = rows_m + (accm,)
                rows_p = rows_p + (accp,)
        xl = rows0 + rows_m + rows_p
        # For RANK == 1 the channel basis is folded into the upstream rows
        # once; the generic path applies cb inside the contractions.
        gy = ()
        for r in tl.static_range(RED):
            gval = tl.load(gu_ptr + f_off + r * CF, mask=cmask, other=0.0).to(
                tl.float32
            )
            if RANK == 1:
                gval = gval * cb[0]
            gy = gy + (gval,)

        # === Phase 1. Degree-kernel (or radial-feature) gradient ===
        if RANK == 0:
            tl.store(gkc_base + 0 * CW + chan, gy[0] * xl[0], mask=cmask)
            for d in tl.static_range(1, NS0):
                t = (
                    gy[d] * xl[d]
                    + gy[NS0 + d - 1] * xl[NS0 + d - 1]
                    + gy[NS0 + L + d - 1] * xl[NS0 + L + d - 1]
                )
                tl.store(gkc_base + d * CW + chan, t, mask=cmask)
        for i in tl.static_range(NS0 if RANK > 0 else 0):
            for o in tl.static_range(NS0):
                if RANK == 1:
                    tl.store(gkc_base + i * NS0 + o, tl.sum(gy[o] * xl[i]))
                else:
                    t = gy[o] * xl[i]
                    for r in tl.static_range(RANK):
                        tl.store(gkc_base + (i * NS0 + o) * RANK + r, tl.sum(t * cb[r]))
        for i in tl.static_range(L if RANK > 0 else 0):
            for o in tl.static_range(L):
                if RANK == 1:
                    tl.store(
                        gkc_base + NS0 * NS0 + i * L + o,
                        tl.sum(gy[NS0 + o] * xl[NS0 + i])
                        + tl.sum(gy[NS0 + L + o] * xl[NS0 + L + i]),
                    )
                else:
                    t = gy[NS0 + o] * xl[NS0 + i] + gy[NS0 + L + o] * xl[NS0 + L + i]
                    for r in tl.static_range(RANK):
                        tl.store(
                            gkc_base + (NS0 * NS0 + i * L + o) * RANK + r,
                            tl.sum(t * cb[r]),
                        )

        # === Phase 2. Rotation backward with g_local formed on the fly ===
        gd_base = gw_ptr + edge * DIM * DIM
        for l in tl.static_range(L + 1):
            base = l * l
            r0 = base + l
            g0 = tl.zeros((BC,), dtype=tl.float32)
            if RANK == 0:
                rad_l = tl.load(kc_base + l * CW + chan, mask=cmask, other=0.0).to(
                    tl.float32
                )
                g0 = gy[l] * rad_l
            for o in tl.static_range(NS0 if RANK > 0 else 0):
                if RANK == 1:
                    g0 += tl.load(kc_base + l * NS0 + o) * gy[o]
                else:
                    keff = tl.zeros((BC,), dtype=tl.float32)
                    for r in tl.static_range(RANK):
                        keff += tl.load(kc_base + (l * NS0 + o) * RANK + r) * cb[r]
                    g0 += keff * gy[o]
            gm = tl.zeros((BC,), dtype=tl.float32)
            gp = tl.zeros((BC,), dtype=tl.float32)
            if l >= 1:
                if RANK == 0:
                    gm = gy[NS0 + l - 1] * rad_l
                    gp = gy[NS0 + L + l - 1] * rad_l
                for o in tl.static_range(L if RANK > 0 else 0):
                    if RANK == 1:
                        k_val = tl.load(kc_base + NS0 * NS0 + (l - 1) * L + o)
                        gm += k_val * gy[NS0 + o]
                        gp += k_val * gy[NS0 + L + o]
                    else:
                        keff = tl.zeros((BC,), dtype=tl.float32)
                        for r in tl.static_range(RANK):
                            keff += (
                                tl.load(
                                    kc_base + (NS0 * NS0 + (l - 1) * L + o) * RANK + r
                                )
                                * cb[r]
                            )
                        gm += keff * gy[NS0 + o]
                        gp += keff * gy[NS0 + L + o]
            for j in tl.static_range(2 * l + 1):
                col = base + j
                xv = xrows[l * l + j]
                w0 = tl.load(d_base + r0 * DIM + col)
                gx_row = w0 * g0
                tl.store(gd_base + r0 * DIM + col, tl.sum(g0 * xv))
                if l >= 1:
                    wmv = tl.load(d_base + (r0 - 1) * DIM + col)
                    wpv = tl.load(d_base + (r0 + 1) * DIM + col)
                    gx_row += wmv * gm + wpv * gp
                    tl.store(gd_base + (r0 - 1) * DIM + col, tl.sum(gm * xv))
                    tl.store(gd_base + (r0 + 1) * DIM + col, tl.sum(gp * xv))
                tl.store(
                    gxe_ptr + edge * DIM * CW + col * CW + chan, gx_row, mask=cmask
                )

    @triton.jit
    def _rotate_mix_bwd_block_kernel(
        gu_ptr,  # (F, E, ROW) upstream gradient (focus-major)
        x_ptr,  # (N, D, CW) node features
        src_ptr,  # (E,)
        w_ptr,  # (E, D, D) block-diagonal Wigner-D
        kc_ptr,  # (E, KSZ) rank-1 compact kernel, or (E, L+1, CW) when RANK == 0
        cb_ptr,  # (1, CW) channel basis (RANK == 1)
        gxe_ptr,  # (E, D, CW) per-edge node gradient out
        gw_ptr,  # (E, D, D) Wigner gradient out (structural non-zeros; pre-zeroed)
        gkc_ptr,  # gradient of kc out, same layout as kc
        n_edge,
        x_sn,
        x_sd,
        L: tl.constexpr,
        CF: tl.constexpr,
        CW: tl.constexpr,
        CP: tl.constexpr,  # next power of two >= CW (vector lane count)
        RANK: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        """Edge-block variant of the rotate+mix backward.

        The per-edge kernel closes one cross-lane ``tl.sum`` per ``grad_kc``
        entry and per structural Wigner non-zero -- serialized warp
        shuffle-reduction chains that dominate its runtime on narrow hidden
        widths.  This variant processes ``BLOCK_E`` edges per program with
        channels as the vector axis: every reduction becomes one batched
        axis-1 reduction of a ``(BLOCK_E, CP)`` tile, and the per-edge Wigner
        and kernel scalars are loaded as coalesced ``(BLOCK_E,)`` vectors.
        The rotated rows are recomputed in registers, matching the per-edge
        kernel's saved-nothing contract.  Channels are padded to the
        power-of-two lane count ``CP`` with masked lanes (masked lanes issue
        no memory traffic; they only raise register pressure, which the
        launch table absorbs with a smaller ``BLOCK_E``).

        The schedule wins only where the reduction overhead of the per-edge
        kernel dominates; :func:`tile_configs.rotate_mix_bwd_block_config`
        acts as the win list, and ``RANK`` must be at most 1 (the per-focus
        upstream fold applies a single channel basis).
        """
        NS0: tl.constexpr = L + 1
        RED: tl.constexpr = 3 * L + 1
        DIM: tl.constexpr = (L + 1) * (L + 1)
        ROW: tl.constexpr = RED * CF
        KSZ: tl.constexpr = NS0 * NS0 + L * L
        PADDED: tl.constexpr = CP != CW

        pid = tl.program_id(0)
        offs_e = (pid * BLOCK_E + tl.arange(0, BLOCK_E)).to(tl.int64)
        e_mask = offs_e < n_edge
        eq = tl.where(e_mask, offs_e, 0)
        chan = tl.arange(0, CP)
        if PADDED:
            c_mask = chan < CW
            em = e_mask[:, None] & c_mask[None, :]
            chan_c = tl.where(c_mask, chan, 0)
        else:
            em = e_mask[:, None]
            chan_c = chan

        src = tl.load(src_ptr + eq, mask=e_mask, other=0).to(tl.int64)
        x_base = x_ptr + (src * x_sn)[:, None]
        d_base = w_ptr + eq * DIM * DIM
        gd_base = gw_ptr + eq * DIM * DIM
        gxe_base = gxe_ptr + (eq * DIM * CW)[:, None]
        if RANK == 0:
            kc_base = kc_ptr + (eq * NS0 * CW)[:, None]
            gkc_base = gkc_ptr + (eq * NS0 * CW)[:, None]
        else:
            kc_base = kc_ptr + eq * KSZ
            gkc_base = gkc_ptr + eq * KSZ

        # Focus-major upstream offset of channel c = f * CF + cf.
        f_off = (
            gu_ptr
            + ((chan_c // CF).to(tl.int64) * n_edge * ROW + (chan_c % CF))[None, :]
            + (eq * ROW)[:, None]
        )

        # === Phase 0. Upstream rows (channel basis folded once, RANK == 1) ===
        gy = ()
        if RANK == 1:
            cbv = tl.load(cb_ptr + chan, mask=(chan < CW), other=0.0)[None, :]
        for r in tl.static_range(RED):
            gval = tl.load(f_off + r * CF, mask=em, other=0.0)
            if RANK == 1:
                gval = gval * cbv
            gy = gy + (gval,)

        # === Phase 1. Per degree: recompute rotation, kernel grads, gx, gD ===
        for l in tl.static_range(L + 1):
            base = l * l
            r0 = base + l

            xrows = ()
            for j in tl.static_range(2 * l + 1):
                xrows = xrows + (
                    tl.load(
                        x_base + (base + j) * x_sd + chan_c[None, :],
                        mask=em,
                        other=0.0,
                    ),
                )
            xl0 = tl.zeros((BLOCK_E, CP), dtype=tl.float32)
            xlm = tl.zeros((BLOCK_E, CP), dtype=tl.float32)
            xlp = tl.zeros((BLOCK_E, CP), dtype=tl.float32)
            for j in tl.static_range(2 * l + 1):
                xv = xrows[j]
                w0 = tl.load(d_base + r0 * DIM + base + j, mask=e_mask, other=0.0)
                xl0 += w0[:, None] * xv
                if l >= 1:
                    wm = tl.load(
                        d_base + (r0 - 1) * DIM + base + j, mask=e_mask, other=0.0
                    )
                    wp = tl.load(
                        d_base + (r0 + 1) * DIM + base + j, mask=e_mask, other=0.0
                    )
                    xlm += wm[:, None] * xv
                    xlp += wp[:, None] * xv

            # Kernel gradient rows of input degree l.
            if RANK == 0:
                if l == 0:
                    t = gy[0] * xl0
                else:
                    t = gy[l] * xl0 + gy[NS0 + l - 1] * xlm + gy[NS0 + L + l - 1] * xlp
                tl.store(gkc_base + l * CW + chan[None, :], t, mask=em)
            else:
                for o in tl.static_range(NS0):
                    tl.store(
                        gkc_base + l * NS0 + o,
                        tl.sum(gy[o] * xl0, axis=1),
                        mask=e_mask,
                    )
                if l >= 1:
                    for o in tl.static_range(L):
                        tl.store(
                            gkc_base + NS0 * NS0 + (l - 1) * L + o,
                            tl.sum(gy[NS0 + o] * xlm + gy[NS0 + L + o] * xlp, axis=1),
                            mask=e_mask,
                        )

            # Local-frame gradients of the reduced rows of degree l.
            g0 = tl.zeros((BLOCK_E, CP), dtype=tl.float32)
            gm = tl.zeros((BLOCK_E, CP), dtype=tl.float32)
            gp = tl.zeros((BLOCK_E, CP), dtype=tl.float32)
            if RANK == 0:
                rad_l = tl.load(kc_base + l * CW + chan[None, :], mask=em, other=0.0)
                g0 = gy[l] * rad_l
                if l >= 1:
                    gm = gy[NS0 + l - 1] * rad_l
                    gp = gy[NS0 + L + l - 1] * rad_l
            else:
                for o in tl.static_range(NS0):
                    k_val = tl.load(kc_base + l * NS0 + o, mask=e_mask, other=0.0)
                    g0 += k_val[:, None] * gy[o]
                if l >= 1:
                    for o in tl.static_range(L):
                        k_val = tl.load(
                            kc_base + NS0 * NS0 + (l - 1) * L + o,
                            mask=e_mask,
                            other=0.0,
                        )
                        gm += k_val[:, None] * gy[NS0 + o]
                        gp += k_val[:, None] * gy[NS0 + L + o]

            # Rotation backward: node gradient rows and Wigner gradients.
            for j in tl.static_range(2 * l + 1):
                col = base + j
                xv = xrows[j]
                w0 = tl.load(d_base + r0 * DIM + col, mask=e_mask, other=0.0)
                gx_row = w0[:, None] * g0
                tl.store(gd_base + r0 * DIM + col, tl.sum(g0 * xv, axis=1), mask=e_mask)
                if l >= 1:
                    wm = tl.load(d_base + (r0 - 1) * DIM + col, mask=e_mask, other=0.0)
                    wp = tl.load(d_base + (r0 + 1) * DIM + col, mask=e_mask, other=0.0)
                    gx_row += wm[:, None] * gm + wp[:, None] * gp
                    tl.store(
                        gd_base + (r0 - 1) * DIM + col,
                        tl.sum(gm * xv, axis=1),
                        mask=e_mask,
                    )
                    tl.store(
                        gd_base + (r0 + 1) * DIM + col,
                        tl.sum(gp * xv, axis=1),
                        mask=e_mask,
                    )
                tl.store(gxe_base + col * CW + chan[None, :], gx_row, mask=em)

    @triton.jit
    def _segment_sum_kernel(
        rows_ptr,  # (E, P) per-edge rows
        order_ptr,  # (E,) edge ids sorted by segment key
        row_ptr_ptr,  # (N + 1,) CSR offsets into ``order``
        out_ptr,  # (N, P)
        P: tl.constexpr,
        BC: tl.constexpr,
    ):
        """Indirect CSR segment sum: ``out[n] = sum_{i in seg(n)} rows[order[i]]``.

        Replaces the row-atomic scatter / ``index_add_`` of the edge-to-node
        reduction; the contention-free segmented read is several times faster
        than atomics at typical per-atom edge counts.
        """
        node = tl.program_id(0).to(tl.int64)
        chunk = tl.program_id(1)
        cols = chunk * BC + tl.arange(0, BC)
        col_mask = cols < P
        beg = tl.load(row_ptr_ptr + node).to(tl.int64)
        end = tl.load(row_ptr_ptr + node + 1).to(tl.int64)
        acc = tl.zeros((BC,), dtype=tl.float32)
        for i in range(beg, end):
            e = tl.load(order_ptr + i).to(tl.int64)
            acc += tl.load(rows_ptr + e * P + cols, mask=col_mask, other=0.0)
        tl.store(out_ptr + node * P + cols, acc, mask=col_mask)

    @triton.jit
    def _stack_gemm_m0_kernel(
        u_ptr,  # (F, E, ROW) layer input
        w0_ptr,  # (NL, F, M0, M0) stacked weights, layer selected by ``layer``
        alpha_ptr,  # (E, F) competition weight (identity epilogue only)
        v_ptr,  # z_all stack (gated) or the final output (identity epilogue)
        n_edge,
        layer,
        L: tl.constexpr,
        CF: tl.constexpr,
        EPILOGUE: tl.constexpr,  # 0: store raw z; 1: residual (+ alpha) output
        V_EDGE_MAJOR: tl.constexpr,  # v is (E, F, ROW); else focus-major (F, E, ROW)
        APPLY_ALPHA: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """``m = 0`` block GEMM ``z = u[:, :M0] @ W0`` with an optional epilogue.

        Output strides are derived in-kernel from the layout flag on int64
        offsets: a host-side ``n_edge * ROW`` scalar argument would be
        specialized to int32 by the first (small) compilation and overflow
        on systems beyond ~2^31 / ROW edges.
        """
        M0: tl.constexpr = (L + 1) * CF
        ROW: tl.constexpr = (3 * L + 1) * CF
        NT: tl.constexpr = (M0 + BLOCK_N - 1) // BLOCK_N

        pid = tl.program_id(0)
        fid = tl.program_id(1).to(tl.int64)
        n_focus = tl.num_programs(1)
        pid_m = pid // NT
        pid_n = pid % NT

        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        m_mask = offs_m < n_edge
        mm = m_mask[:, None]
        u_row = u_ptr + fid * n_edge * ROW + offs_m * ROW
        offs_k = tl.arange(0, BLOCK_K)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < M0

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        a_ptrs = u_row[:, None] + offs_k[None, :]
        w_ptrs = (
            w0_ptr
            + (layer * n_focus + fid) * M0 * M0
            + offs_k[:, None] * M0
            + offs_n[None, :]
        )
        for _ in range(0, M0, BLOCK_K):
            a = tl.load(a_ptrs, mask=mm, other=0.0)
            w = tl.load(w_ptrs, mask=n_mask[None, :], other=0.0)
            acc = tl.dot(a, w, acc, input_precision="ieee")
            a_ptrs += BLOCK_K
            w_ptrs += BLOCK_K * M0

        if EPILOGUE == 1:
            u_t = tl.load(
                u_row[:, None] + offs_n[None, :], mask=mm & n_mask[None, :], other=0.0
            )
            acc = acc + u_t
            if APPLY_ALPHA:
                alpha = tl.load(
                    alpha_ptr + offs_m * n_focus + fid, mask=m_mask, other=0.0
                )
                acc = acc * alpha[:, None]
            if V_EDGE_MAJOR:
                v_row = v_ptr + fid * ROW + offs_m * (n_focus * ROW)
            else:
                v_row = v_ptr + fid * n_edge * ROW + offs_m * ROW
            tl.store(v_row[:, None] + offs_n[None, :], acc, mask=mm & n_mask[None, :])
        else:
            z_row = v_ptr + (layer * n_focus + fid) * n_edge * ROW + offs_m * ROW
            tl.store(z_row[:, None] + offs_n[None, :], acc, mask=mm & n_mask[None, :])

    @triton.jit
    def _stack_gate_kernel(
        u_ptr,
        z_ptr,  # z_all stack, layer selected by ``layer``
        gw_ptr,  # (NL, F, CF, L*CF) stacked gate projections
        v_ptr,  # (F, E, ROW) layer output, focus-major
        sig_ptr,  # (F, E, L*CF); output when SIG_IN == 0, input when SIG_IN == 1
        n_edge,
        layer,
        L: tl.constexpr,
        CF: tl.constexpr,
        SIG_IN: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        """Gate evaluation and ``m = 0`` finish: ``v = u + act(z)`` on the m0 rows.

        Register tiles are ``CP`` wide (``CF`` padded to a power of two) with a
        column mask, so non-power-of-two focus widths are supported; padded dot
        lanes carry zeros and are never stored.  With ``SIG_IN`` the sigmoid
        projection has already been produced by a cuBLAS bmm (wide-channel
        regime) and this kernel only reads it.
        """
        ROW: tl.constexpr = (3 * L + 1) * CF
        LG: tl.constexpr = L * CF
        CP: tl.constexpr = triton.next_power_of_2(CF)

        pid_m = tl.program_id(0)
        fid = tl.program_id(1).to(tl.int64)
        n_focus = tl.num_programs(1)

        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        m_mask = offs_m < n_edge
        mm = m_mask[:, None]
        nc = tl.arange(0, CP)
        cm = mm & (nc < CF)[None, :]
        wm = ((nc < CF)[:, None]) & ((nc < CF)[None, :])

        u_row = u_ptr + fid * n_edge * ROW + offs_m * ROW
        z_row = z_ptr + (layer * n_focus + fid) * n_edge * ROW + offs_m * ROW
        v_row = v_ptr + fid * n_edge * ROW + offs_m * ROW
        sig_row = sig_ptr + (fid * n_edge + offs_m) * LG

        # l = 0 scalar rows pass through silu.
        z_s = tl.load(z_row[:, None] + nc[None, :], mask=cm, other=0.0)
        u_s = tl.load(u_row[:, None] + nc[None, :], mask=cm, other=0.0)
        tl.store(v_row[:, None] + nc[None, :], u_s + z_s * tl.sigmoid(z_s), mask=cm)

        # Per-group sigmoid gates and the gated m = 0 rows.
        for g in tl.static_range(L):
            if SIG_IN:
                sig_g = tl.load(
                    sig_row[:, None] + (g * CF + nc)[None, :], mask=cm, other=0.0
                )
            else:
                gw_g = tl.load(
                    gw_ptr
                    + (layer * n_focus + fid) * CF * LG
                    + nc[:, None] * LG
                    + (g * CF + nc)[None, :],
                    mask=wm,
                    other=0.0,
                )
                sig_g = tl.sigmoid(tl.dot(z_s, gw_g, input_precision="ieee"))
                tl.store(sig_row[:, None] + (g * CF + nc)[None, :], sig_g, mask=cm)
            z_g = tl.load(
                z_row[:, None] + ((1 + g) * CF + nc)[None, :], mask=cm, other=0.0
            )
            u_g = tl.load(
                u_row[:, None] + ((1 + g) * CF + nc)[None, :], mask=cm, other=0.0
            )
            tl.store(
                v_row[:, None] + ((1 + g) * CF + nc)[None, :],
                u_g + z_g * sig_g,
                mask=cm,
            )

    @triton.jit
    def _stack_gemm_m1_kernel(
        u_ptr,
        w1_ptr,  # (NL, F, M1, M1) stacked weights, layer selected by ``layer``
        sig_ptr,
        alpha_ptr,
        v_ptr,
        z_ptr,  # z_all stack, layer selected by ``layer``
        n_edge,
        layer,
        L: tl.constexpr,
        CF: tl.constexpr,
        HAS_GATE: tl.constexpr,
        V_EDGE_MAJOR: tl.constexpr,  # v is (E, F, ROW); else focus-major (F, E, ROW)
        APPLY_ALPHA: tl.constexpr,
        SAVE_Z: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """``|m| = 1`` block GEMM with the gate / residual / alpha epilogue fused."""
        M0: tl.constexpr = (L + 1) * CF
        M1: tl.constexpr = 2 * L * CF
        ROW: tl.constexpr = (3 * L + 1) * CF
        LG: tl.constexpr = L * CF
        NT: tl.constexpr = (M1 + BLOCK_N - 1) // BLOCK_N

        pid = tl.program_id(0)
        fid = tl.program_id(1).to(tl.int64)
        n_focus = tl.num_programs(1)
        pid_m = pid // NT
        pid_n = pid % NT

        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        m_mask = offs_m < n_edge
        mm = m_mask[:, None]
        u_row = u_ptr + fid * n_edge * ROW + offs_m * ROW
        offs_k = tl.arange(0, BLOCK_K)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < M1

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        a_ptrs = u_row[:, None] + (M0 + offs_k)[None, :]
        w_ptrs = (
            w1_ptr
            + (layer * n_focus + fid) * M1 * M1
            + offs_k[:, None] * M1
            + offs_n[None, :]
        )
        for _ in range(0, M1, BLOCK_K):
            a = tl.load(a_ptrs, mask=mm, other=0.0)
            w = tl.load(w_ptrs, mask=n_mask[None, :], other=0.0)
            acc = tl.dot(a, w, acc, input_precision="ieee")
            a_ptrs += BLOCK_K
            w_ptrs += BLOCK_K * M1

        if SAVE_Z:
            z_row = z_ptr + (layer * n_focus + fid) * n_edge * ROW + offs_m * ROW
            tl.store(
                z_row[:, None] + (M0 + offs_n)[None, :], acc, mask=mm & n_mask[None, :]
            )
        if HAS_GATE:
            # Both |m| = 1 stripes of degree group g share gate group g.
            sig_cols = ((offs_n // CF) % L) * CF + (offs_n % CF)
            sig = tl.load(
                sig_ptr + (fid * n_edge + offs_m)[:, None] * LG + sig_cols[None, :],
                mask=mm & n_mask[None, :],
                other=0.0,
            )
            acc = acc * sig
        u_t = tl.load(
            u_row[:, None] + (M0 + offs_n)[None, :],
            mask=mm & n_mask[None, :],
            other=0.0,
        )
        acc = acc + u_t
        if APPLY_ALPHA:
            alpha = tl.load(alpha_ptr + offs_m * n_focus + fid, mask=m_mask, other=0.0)
            acc = acc * alpha[:, None]
        if V_EDGE_MAJOR:
            v_row = v_ptr + fid * ROW + offs_m * (n_focus * ROW)
        else:
            v_row = v_ptr + fid * n_edge * ROW + offs_m * ROW
        tl.store(
            v_row[:, None] + (M0 + offs_n)[None, :], acc, mask=mm & n_mask[None, :]
        )

    @triton.jit
    def _stack_recompute_kernel(
        z_ptr,  # z_all stack (NL, F, E, ROW), layer selected by ``layer``
        gw_ptr,  # (NL, F, CF, L*CF) stacked gate projections
        sig_ptr,  # (F, E, L*CF) output
        n_edge,
        layer,
        L: tl.constexpr,
        CF: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        """Recompute the gate sigmoids from the saved pre-activation (backward)."""
        ROW: tl.constexpr = (3 * L + 1) * CF
        LG: tl.constexpr = L * CF
        CP: tl.constexpr = triton.next_power_of_2(CF)

        pid_m = tl.program_id(0)
        fid = tl.program_id(1).to(tl.int64)
        n_focus = tl.num_programs(1)

        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        m_mask = offs_m < n_edge
        mm = m_mask[:, None]
        nc = tl.arange(0, CP)
        cm = mm & (nc < CF)[None, :]
        wm = ((nc < CF)[:, None]) & ((nc < CF)[None, :])

        z_row = z_ptr + (layer * n_focus + fid) * n_edge * ROW + offs_m * ROW
        z_s = tl.load(z_row[:, None] + nc[None, :], mask=cm, other=0.0)
        sig_row = sig_ptr + (fid * n_edge + offs_m) * LG
        for g in tl.static_range(L):
            gw_g = tl.load(
                gw_ptr
                + (layer * n_focus + fid) * CF * LG
                + nc[:, None] * LG
                + (g * CF + nc)[None, :],
                mask=wm,
                other=0.0,
            )
            sig_g = tl.sigmoid(tl.dot(z_s, gw_g, input_precision="ieee"))
            tl.store(sig_row[:, None] + (g * CF + nc)[None, :], sig_g, mask=cm)

    @triton.jit
    def _stack_point_bwd_kernel(
        g_ptr,  # (F, E, ROW) upstream gradient of the layer output
        z_ptr,  # z_all stack, layer selected by ``layer``
        sig_ptr,  # (F, E, L*CF) gate sigmoids
        gwt_ptr,  # (NL, F, L*CF, CF) transposed gate projections
        gz_ptr,  # (F, E, ROW) pre-activation gradient output
        gl_ptr,  # (F, E, L*CF) gate-logit gradient output (GLOGIT_OUT only)
        n_edge,
        layer,
        L: tl.constexpr,
        CF: tl.constexpr,
        GLOGIT_OUT: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        """Pointwise part of the gated-layer backward.

        Produces the pre-activation gradient ``gz`` for the value rows and the
        gate-path contribution to the ``l = 0`` scalar rows.  The gate-logit
        contraction back to the scalars is either folded in as a ``CP x CP``
        register dot (small ``CF``) or emitted to ``gl`` for an external
        batched GEMM (wide-channel regime, where the register dot spills).
        """
        ROW: tl.constexpr = (3 * L + 1) * CF
        LG: tl.constexpr = L * CF
        CP: tl.constexpr = triton.next_power_of_2(CF)

        pid_m = tl.program_id(0)
        fid = tl.program_id(1).to(tl.int64)
        n_focus = tl.num_programs(1)

        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        m_mask = offs_m < n_edge
        mm = m_mask[:, None]
        nc = tl.arange(0, CP)
        cm = mm & (nc < CF)[None, :]
        wm = ((nc < CF)[:, None]) & ((nc < CF)[None, :])

        g_row = g_ptr + fid * n_edge * ROW + offs_m * ROW
        z_row = z_ptr + (layer * n_focus + fid) * n_edge * ROW + offs_m * ROW
        gz_row = gz_ptr + fid * n_edge * ROW + offs_m * ROW
        sig_row = sig_ptr + (fid * n_edge + offs_m) * LG
        gl_row = gl_ptr + (fid * n_edge + offs_m) * LG

        # l = 0 value path: silu backward.
        z_s = tl.load(z_row[:, None] + nc[None, :], mask=cm, other=0.0)
        g_s = tl.load(g_row[:, None] + nc[None, :], mask=cm, other=0.0)
        s0 = tl.sigmoid(z_s)
        gz_s = g_s * s0 * (1.0 + z_s * (1.0 - s0))

        for g in tl.static_range(L):
            sig_g = tl.load(
                sig_row[:, None] + (g * CF + nc)[None, :], mask=cm, other=0.0
            )
            gr0 = tl.load(
                g_row[:, None] + ((1 + g) * CF + nc)[None, :], mask=cm, other=0.0
            )
            zr0 = tl.load(
                z_row[:, None] + ((1 + g) * CF + nc)[None, :], mask=cm, other=0.0
            )
            tl.store(
                gz_row[:, None] + ((1 + g) * CF + nc)[None, :], gr0 * sig_g, mask=cm
            )
            rn = (L + 1) + g
            grn = tl.load(g_row[:, None] + (rn * CF + nc)[None, :], mask=cm, other=0.0)
            zrn = tl.load(z_row[:, None] + (rn * CF + nc)[None, :], mask=cm, other=0.0)
            tl.store(gz_row[:, None] + (rn * CF + nc)[None, :], grn * sig_g, mask=cm)
            rp = (2 * L + 1) + g
            grp = tl.load(g_row[:, None] + (rp * CF + nc)[None, :], mask=cm, other=0.0)
            zrp = tl.load(z_row[:, None] + (rp * CF + nc)[None, :], mask=cm, other=0.0)
            tl.store(gz_row[:, None] + (rp * CF + nc)[None, :], grp * sig_g, mask=cm)
            # Gate path: three value rows share gate group g.
            g_sig = gr0 * zr0 + grn * zrn + grp * zrp
            g_logit = g_sig * sig_g * (1.0 - sig_g)
            if GLOGIT_OUT:
                tl.store(gl_row[:, None] + (g * CF + nc)[None, :], g_logit, mask=cm)
            else:
                gwt_g = tl.load(
                    gwt_ptr
                    + (layer * n_focus + fid) * LG * CF
                    + (g * CF + nc)[:, None] * CF
                    + nc[None, :],
                    mask=wm,
                    other=0.0,
                )
                gz_s = tl.dot(g_logit, gwt_g, gz_s, input_precision="ieee")

        tl.store(gz_row[:, None] + nc[None, :], gz_s, mask=cm)

    @triton.jit
    def _stack_gemm_bwd_kernel(
        gz_ptr,  # (F, E, ROW), or the raw upstream gradient when FOLD_ALPHA
        res_ptr,  # (F, E, ROW) residual gradient source; unread if FOLD_ALPHA
        w0t_ptr,  # (NL, F, M0, M0) stacked transposed weights
        w1t_ptr,  # (NL, F, M1, M1) stacked transposed weights
        alpha_ptr,
        gu_ptr,  # (F, E, ROW) layer-input gradient
        n_edge,
        layer,
        L: tl.constexpr,
        CF: tl.constexpr,
        G_EDGE_MAJOR: tl.constexpr,  # gz is (E, F, ROW); else focus-major
        FOLD_ALPHA: tl.constexpr,  # gz = g * alpha on the fly; residual == gz
        RES_IS_GZ: tl.constexpr,  # residual equals gz (final layer, no alpha)
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Backward block GEMM ``g_u = residual + gz @ W^T`` over both blocks.

        The upstream gradient is edge-major only on the final layer, where
        the residual aliases it (``RES_IS_GZ`` or ``FOLD_ALPHA``); an
        explicit residual pointer is always focus-major.  Strides are
        derived in-kernel on int64 offsets (see ``_stack_gemm_m0_kernel``).
        """
        M0: tl.constexpr = (L + 1) * CF
        M1: tl.constexpr = 2 * L * CF
        ROW: tl.constexpr = (3 * L + 1) * CF
        NT0: tl.constexpr = (M0 + BLOCK_N - 1) // BLOCK_N
        NT1: tl.constexpr = (M1 + BLOCK_N - 1) // BLOCK_N
        NT: tl.constexpr = NT0 + NT1

        pid = tl.program_id(0)
        fid = tl.program_id(1).to(tl.int64)
        n_focus = tl.num_programs(1)
        pid_m = pid // NT
        pid_n = pid % NT

        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        m_mask = offs_m < n_edge
        mm = m_mask[:, None]
        offs_k = tl.arange(0, BLOCK_K)

        if G_EDGE_MAJOR:
            gz_row = gz_ptr + fid * ROW + offs_m * (n_focus * ROW)
        else:
            gz_row = gz_ptr + fid * n_edge * ROW + offs_m * ROW
        gu_row = gu_ptr + fid * n_edge * ROW + offs_m * ROW
        if FOLD_ALPHA:
            alpha = tl.load(alpha_ptr + offs_m * n_focus + fid, mask=m_mask, other=0.0)

        if pid_n < NT0:
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            n_mask = offs_n < M0
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            a_ptrs = gz_row[:, None] + offs_k[None, :]
            w_ptrs = (
                w0t_ptr
                + (layer * n_focus + fid) * M0 * M0
                + offs_k[:, None] * M0
                + offs_n[None, :]
            )
            for _ in range(0, M0, BLOCK_K):
                a = tl.load(a_ptrs, mask=mm, other=0.0)
                if FOLD_ALPHA:
                    a = a * alpha[:, None]
                w = tl.load(w_ptrs, mask=n_mask[None, :], other=0.0)
                acc = tl.dot(a, w, acc, input_precision="ieee")
                a_ptrs += BLOCK_K
                w_ptrs += BLOCK_K * M0
            col0 = offs_n
            col_mask = n_mask
        else:
            offs_n = (pid_n - NT0) * BLOCK_N + tl.arange(0, BLOCK_N)
            n_mask = offs_n < M1
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            a_ptrs = gz_row[:, None] + (M0 + offs_k)[None, :]
            w_ptrs = (
                w1t_ptr
                + (layer * n_focus + fid) * M1 * M1
                + offs_k[:, None] * M1
                + offs_n[None, :]
            )
            for _ in range(0, M1, BLOCK_K):
                a = tl.load(a_ptrs, mask=mm, other=0.0)
                if FOLD_ALPHA:
                    a = a * alpha[:, None]
                w = tl.load(w_ptrs, mask=n_mask[None, :], other=0.0)
                acc = tl.dot(a, w, acc, input_precision="ieee")
                a_ptrs += BLOCK_K
                w_ptrs += BLOCK_K * M1
            col0 = M0 + offs_n
            col_mask = n_mask

        if FOLD_ALPHA:
            res = tl.load(
                gz_row[:, None] + col0[None, :], mask=mm & col_mask[None, :], other=0.0
            )
            res = res * alpha[:, None]
        elif RES_IS_GZ:
            res = tl.load(
                gz_row[:, None] + col0[None, :], mask=mm & col_mask[None, :], other=0.0
            )
        else:
            res_row = res_ptr + fid * n_edge * ROW + offs_m * ROW
            res = tl.load(
                res_row[:, None] + col0[None, :], mask=mm & col_mask[None, :], other=0.0
            )
        tl.store(
            gu_row[:, None] + col0[None, :], acc + res, mask=mm & col_mask[None, :]
        )

    @triton.jit
    def _stack_grad_alpha_kernel(
        g_ptr,  # (E, F, ROW) edge-major upstream gradient
        out_ptr,  # (E, F, ROW) forward output
        alpha_ptr,  # (E, F)
        ga_ptr,  # (E, F)
        n_edge,
        L: tl.constexpr,
        CF: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        """Competition-weight gradient from the identity ``grad_alpha =
        sum(grad * out) / alpha`` -- exact because the final store is a plain
        scale, saving the two pre-scale activation copies.
        """
        ROW: tl.constexpr = (3 * L + 1) * CF
        CP: tl.constexpr = triton.next_power_of_2(CF)

        pid_m = tl.program_id(0)
        fid = tl.program_id(1).to(tl.int64)
        n_focus = tl.num_programs(1)

        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        m_mask = offs_m < n_edge
        mm = m_mask[:, None]
        nc = tl.arange(0, CP)
        cm = mm & (nc < CF)[None, :]

        g_row = g_ptr + (offs_m * n_focus + fid) * ROW
        o_row = out_ptr + (offs_m * n_focus + fid) * ROW
        ga = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for r in tl.static_range(3 * L + 1):
            g_r = tl.load(g_row[:, None] + (r * CF + nc)[None, :], mask=cm, other=0.0)
            o_r = tl.load(o_row[:, None] + (r * CF + nc)[None, :], mask=cm, other=0.0)
            ga += tl.sum(g_r * o_r, axis=1)
        alpha = tl.load(alpha_ptr + offs_m * n_focus + fid, mask=m_mask, other=1.0)
        tl.store(
            ga_ptr + offs_m * n_focus + fid,
            ga / tl.maximum(alpha, 1e-12),
            mask=m_mask,
        )


# ======================================================================
# Zero-edge guard and dispatch predicate
# ======================================================================
def _has_no_edges(n_edge) -> bool:
    """True only for eager zero-edge calls; never guards symbolic edge counts."""
    return type(n_edge) is int and n_edge == 0


def _use_triton(tensor: Tensor) -> bool:
    return (
        SO2_VALUE_PATH_TRITON_AVAILABLE
        and tensor.is_cuda
        and tensor.dtype is torch.float32
    )


# ======================================================================
# Operator implementations (Triton on CUDA fp32, eager reference otherwise)
# ======================================================================
def _rotate_mix_impl(
    x: Tensor,
    src: Tensor,
    wigner: Tensor,
    kc: Tensor,
    cb: Tensor,
    lmax: int,
    n_focus: int,
    rank: int,
) -> Tensor:
    if not _use_triton(x):
        return _rotate_mix_reference(x, src, wigner, kc, cb, lmax, n_focus, rank)
    n_edge = src.shape[0]
    c_wide = int(x.shape[2])
    focus_dim = c_wide // int(n_focus)
    row = (3 * int(lmax) + 1) * focus_dim
    u = torch.empty(n_focus, n_edge, row, device=x.device, dtype=x.dtype)
    if _has_no_edges(n_edge):
        return u
    warps, stages = rotate_mix_fwd_config(c_wide, int(lmax))
    wrap_triton(_rotate_mix_fwd_kernel)[(n_edge,)](
        x,
        src,
        wigner,
        kc,
        cb,
        u,
        n_edge,
        x.stride(0),
        x.stride(1),
        L=int(lmax),
        CF=focus_dim,
        CW=c_wide,
        BC=triton.next_power_of_2(c_wide),
        RANK=int(rank),
        num_warps=warps,
        num_stages=stages,
    )
    return u


def _rotate_mix_bwd_impl(
    grad_u: Tensor,
    x: Tensor,
    src: Tensor,
    wigner: Tensor,
    kc: Tensor,
    cb: Tensor,
    lmax: int,
    n_focus: int,
    rank: int,
) -> tuple[Tensor, Tensor, Tensor]:
    if not _use_triton(x):
        return _rotate_mix_backward_reference(
            grad_u, x, src, wigner, kc, cb, lmax, n_focus, rank
        )
    n_edge = src.shape[0]
    c_wide = int(x.shape[2])
    dim = (int(lmax) + 1) ** 2
    grad_x_edge = torch.empty(n_edge, dim, c_wide, device=x.device, dtype=x.dtype)
    grad_wigner = torch.zeros_like(wigner)
    grad_kc = torch.empty_like(kc)
    if _has_no_edges(n_edge):
        return grad_x_edge, grad_wigner, grad_kc
    # The edge-block schedule engages on swept-and-winning (C_wide, lmax)
    # keys (RANK <= 1 -- the block kernel folds a single channel basis);
    # every other shape keeps the per-edge kernel.  The branch resolves at
    # trace time, so exactly one kernel reaches the compiled graph.
    block_cfg = (
        rotate_mix_bwd_block_config(c_wide, int(lmax)) if int(rank) <= 1 else None
    )
    if block_cfg is not None:
        block_e, warps, stages = block_cfg
        wrap_triton(_rotate_mix_bwd_block_kernel)[(triton.cdiv(n_edge, block_e),)](
            grad_u,
            x,
            src,
            wigner,
            kc,
            cb,
            grad_x_edge,
            grad_wigner,
            grad_kc,
            n_edge,
            x.stride(0),
            x.stride(1),
            L=int(lmax),
            CF=c_wide // int(n_focus),
            CW=c_wide,
            CP=triton.next_power_of_2(c_wide),
            RANK=int(rank),
            BLOCK_E=block_e,
            num_warps=warps,
            num_stages=stages,
        )
        return grad_x_edge, grad_wigner, grad_kc
    warps, stages = _ROTATE_MIX_BWD_CONFIG
    wrap_triton(_rotate_mix_bwd_kernel)[(n_edge,)](
        grad_u,
        x,
        src,
        wigner,
        kc,
        cb,
        grad_x_edge,
        grad_wigner,
        grad_kc,
        n_edge,
        x.stride(0),
        x.stride(1),
        L=int(lmax),
        CF=c_wide // int(n_focus),
        CW=c_wide,
        BC=triton.next_power_of_2(c_wide),
        RANK=int(rank),
        num_warps=warps,
        num_stages=stages,
    )
    return grad_x_edge, grad_wigner, grad_kc


def _segment_sum_impl(rows: Tensor, order: Tensor, row_ptr: Tensor) -> Tensor:
    n_rows = rows.shape[0]
    n_seg = row_ptr.shape[0] - 1
    if not _use_triton(rows):
        counts = row_ptr[1:] - row_ptr[:-1]
        seg_of_sorted = torch.repeat_interleave(
            torch.arange(n_seg, device=rows.device, dtype=order.dtype), counts
        )
        out = rows.new_zeros((n_seg, rows.shape[1], rows.shape[2]))
        out.index_add_(0, seg_of_sorted, rows.index_select(0, order))
        return out
    out = torch.empty(
        (n_seg, rows.shape[1], rows.shape[2]), device=rows.device, dtype=rows.dtype
    )
    if _has_no_edges(n_rows):
        return out.zero_()
    per_row = int(rows.shape[1]) * int(rows.shape[2])
    block = 256
    wrap_triton(_segment_sum_kernel)[(n_seg, triton.cdiv(per_row, block))](
        rows,
        order,
        row_ptr,
        out,
        P=per_row,
        BC=block,
        num_warps=4,
        num_stages=2,
    )
    return out


def _mixing_stack_impl(
    u0: Tensor,
    alpha: Tensor,
    w0_all: Tensor,
    w1_all: Tensor,
    gw_all: Tensor,
    lmax: int,
    focus_dim: int,
    apply_alpha: bool,
) -> tuple[Tensor, Tensor]:
    if not _use_triton(u0):
        return _mixing_stack_reference(
            u0, alpha, w0_all, w1_all, gw_all, lmax, focus_dim, apply_alpha
        )
    n_focus, n_edge, row = u0.shape
    lmax = int(lmax)
    focus_dim = int(focus_dim)
    n_gated = gw_all.shape[0]
    z_all = torch.empty(
        (n_gated, n_focus, n_edge, row), device=u0.device, dtype=u0.dtype
    )
    x_local = torch.empty((n_edge, n_focus, row), device=u0.device, dtype=u0.dtype)
    if _has_no_edges(n_edge):
        return x_local, z_all

    block_m, block_n, block_k, warps, stages = _GEMM_CONFIG
    m0 = (lmax + 1) * focus_dim
    m1 = 2 * lmax * focus_dim
    gate_bm, gate_w, gate_s = gate_config(focus_dim, lmax)
    sig_by_bmm = focus_dim >= GATE_BMM_MIN_FOCUS_DIM
    sig = torch.empty(
        (n_focus, n_edge, lmax * focus_dim), device=u0.device, dtype=torch.float32
    )

    u = u0
    for layer in range(n_gated):
        out = torch.empty_like(u)
        wrap_triton(_stack_gemm_m0_kernel)[
            (triton.cdiv(n_edge, block_m) * triton.cdiv(m0, block_n), n_focus)
        ](
            u,
            w0_all,
            u,
            z_all,
            n_edge,
            layer,
            L=lmax,
            CF=focus_dim,
            EPILOGUE=0,
            V_EDGE_MAJOR=False,
            APPLY_ALPHA=False,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=warps,
            num_stages=stages,
        )
        if sig_by_bmm:
            # Wide-channel regime: sigmoid projection as a cuBLAS bmm on the
            # freshly written l = 0 scalar rows of the pre-activation.
            torch.sigmoid(
                torch.bmm(z_all[layer, :, :, :focus_dim], gw_all[layer]), out=sig
            )
        wrap_triton(_stack_gate_kernel)[(triton.cdiv(n_edge, gate_bm), n_focus)](
            u,
            z_all,
            gw_all,
            out,
            sig,
            n_edge,
            layer,
            L=lmax,
            CF=focus_dim,
            SIG_IN=sig_by_bmm,
            BLOCK_M=gate_bm,
            num_warps=gate_w,
            num_stages=gate_s,
        )
        wrap_triton(_stack_gemm_m1_kernel)[
            (triton.cdiv(n_edge, block_m) * triton.cdiv(m1, block_n), n_focus)
        ](
            u,
            w1_all,
            sig,
            u,
            out,
            z_all,
            n_edge,
            layer,
            L=lmax,
            CF=focus_dim,
            HAS_GATE=True,
            V_EDGE_MAJOR=False,
            APPLY_ALPHA=False,
            SAVE_Z=True,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=warps,
            num_stages=stages,
        )
        u = out

    # Final identity layer streams straight into the edge-major output layout.
    wrap_triton(_stack_gemm_m0_kernel)[
        (triton.cdiv(n_edge, block_m) * triton.cdiv(m0, block_n), n_focus)
    ](
        u,
        w0_all,
        alpha,
        x_local,
        n_edge,
        n_gated,
        L=lmax,
        CF=focus_dim,
        EPILOGUE=1,
        V_EDGE_MAJOR=True,
        APPLY_ALPHA=apply_alpha,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=warps,
        num_stages=stages,
    )
    wrap_triton(_stack_gemm_m1_kernel)[
        (triton.cdiv(n_edge, block_m) * triton.cdiv(m1, block_n), n_focus)
    ](
        u,
        w1_all,
        u,
        alpha,
        x_local,
        u,
        n_edge,
        n_gated,
        L=lmax,
        CF=focus_dim,
        HAS_GATE=False,
        V_EDGE_MAJOR=True,
        APPLY_ALPHA=apply_alpha,
        SAVE_Z=False,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=warps,
        num_stages=stages,
    )
    return x_local, z_all


def _mixing_stack_bwd_impl(
    grad_out: Tensor,
    x_local: Tensor,
    z_all: Tensor,
    alpha: Tensor,
    w0t_all: Tensor,
    w1t_all: Tensor,
    gw_all: Tensor,
    gwt_all: Tensor,
    lmax: int,
    focus_dim: int,
    apply_alpha: bool,
) -> tuple[Tensor, Tensor]:
    if not _use_triton(grad_out):
        return _mixing_stack_backward_reference(
            grad_out,
            x_local,
            z_all,
            alpha,
            w0t_all,
            w1t_all,
            gw_all,
            gwt_all,
            lmax,
            focus_dim,
            apply_alpha,
        )
    n_gated, n_focus, n_edge, row = z_all.shape
    lmax = int(lmax)
    focus_dim = int(focus_dim)
    device, dtype = grad_out.device, grad_out.dtype
    grad_alpha = torch.empty((n_edge, n_focus), device=device, dtype=dtype)
    grad_u0 = torch.empty((n_focus, n_edge, row), device=device, dtype=dtype)
    if _has_no_edges(n_edge):
        return grad_u0, grad_alpha

    block_m, block_n, block_k, warps, stages = _GEMM_CONFIG
    m0 = (lmax + 1) * focus_dim
    m1 = 2 * lmax * focus_dim
    n_tiles = triton.cdiv(m0, block_n) + triton.cdiv(m1, block_n)
    point_bm, point_w, point_s = point_config(focus_dim, lmax)

    # === Final layer: g = gz + gz @ W^T with gz = grad [* alpha] on the fly ===
    g_cur = torch.empty((n_focus, n_edge, row), device=device, dtype=dtype)
    wrap_triton(_stack_gemm_bwd_kernel)[
        (triton.cdiv(n_edge, block_m) * n_tiles, n_focus)
    ](
        grad_out,
        grad_out,
        w0t_all,
        w1t_all,
        alpha,
        g_cur,
        n_edge,
        n_gated,
        L=lmax,
        CF=focus_dim,
        G_EDGE_MAJOR=True,
        FOLD_ALPHA=apply_alpha,
        RES_IS_GZ=True,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=warps,
        num_stages=stages,
    )
    if apply_alpha:
        a_bm, a_w, a_s = gate_config(focus_dim, lmax)
        wrap_triton(_stack_grad_alpha_kernel)[(triton.cdiv(n_edge, a_bm), n_focus)](
            grad_out,
            x_local,
            alpha,
            grad_alpha,
            n_edge,
            L=lmax,
            CF=focus_dim,
            BLOCK_M=a_bm,
            num_warps=a_w,
            num_stages=a_s,
        )

    # === Gated layers in reverse; sig / gz buffers are reused across layers ===
    gate_width = lmax * focus_dim
    sig = torch.empty((n_focus, n_edge, gate_width), device=device, dtype=torch.float32)
    gz = torch.empty((n_focus, n_edge, row), device=device, dtype=dtype)
    use_bmm = focus_dim >= GATE_BMM_MIN_FOCUS_DIM
    glogit = (
        torch.empty((n_focus, n_edge, gate_width), device=device, dtype=torch.float32)
        if use_bmm
        else sig
    )
    r_bm, r_w, r_s = recompute_config(focus_dim, lmax)
    for layer in range(n_gated - 1, -1, -1):
        if use_bmm:
            torch.sigmoid(
                torch.bmm(z_all[layer, :, :, :focus_dim], gw_all[layer]), out=sig
            )
        else:
            wrap_triton(_stack_recompute_kernel)[(triton.cdiv(n_edge, r_bm), n_focus)](
                z_all,
                gw_all,
                sig,
                n_edge,
                layer,
                L=lmax,
                CF=focus_dim,
                BLOCK_M=r_bm,
                num_warps=r_w,
                num_stages=r_s,
            )
        wrap_triton(_stack_point_bwd_kernel)[(triton.cdiv(n_edge, point_bm), n_focus)](
            g_cur,
            z_all,
            sig,
            gwt_all,
            gz,
            glogit,
            n_edge,
            layer,
            L=lmax,
            CF=focus_dim,
            GLOGIT_OUT=use_bmm,
            BLOCK_M=point_bm,
            num_warps=point_w,
            num_stages=point_s,
        )
        if use_bmm:
            # Gate-logit contraction back to the scalar rows via cuBLAS.
            gz[:, :, :focus_dim] += torch.bmm(glogit, gwt_all[layer])
        g_next = torch.empty((n_focus, n_edge, row), device=device, dtype=dtype)
        wrap_triton(_stack_gemm_bwd_kernel)[
            (triton.cdiv(n_edge, block_m) * n_tiles, n_focus)
        ](
            gz,
            g_cur,
            w0t_all,
            w1t_all,
            gz,
            g_next,
            n_edge,
            layer,
            L=lmax,
            CF=focus_dim,
            G_EDGE_MAJOR=False,
            FOLD_ALPHA=False,
            RES_IS_GZ=False,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=warps,
            num_stages=stages,
        )
        g_cur = g_next
    return g_cur, grad_alpha


# ======================================================================
# Functional triton_op + fake + autograd registration
# ======================================================================
_rotate_mix_op = torch.library.triton_op(
    "sezm_triton::so2_rotate_mix", mutates_args=()
)(_rotate_mix_impl)
_rotate_mix_bwd_op = torch.library.triton_op(
    "sezm_triton::so2_rotate_mix_bwd", mutates_args=()
)(_rotate_mix_bwd_impl)
_segment_sum_op = torch.library.triton_op("sezm_triton::segment_sum", mutates_args=())(
    _segment_sum_impl
)
_mixing_stack_op = torch.library.triton_op(
    "sezm_triton::so2_mixing_stack", mutates_args=()
)(_mixing_stack_impl)
_mixing_stack_bwd_op = torch.library.triton_op(
    "sezm_triton::so2_mixing_stack_bwd", mutates_args=()
)(_mixing_stack_bwd_impl)


@_rotate_mix_op.register_fake
def _(x, src, wigner, kc, cb, lmax, n_focus, rank):
    focus_dim = x.shape[2] // n_focus
    return x.new_empty((n_focus, src.shape[0], (3 * lmax + 1) * focus_dim))


@_rotate_mix_bwd_op.register_fake
def _(grad_u, x, src, wigner, kc, cb, lmax, n_focus, rank):
    return (
        x.new_empty((src.shape[0], (lmax + 1) ** 2, x.shape[2])),
        torch.empty_like(wigner),
        torch.empty_like(kc),
    )


@_segment_sum_op.register_fake
def _(rows, order, row_ptr):
    return rows.new_empty((row_ptr.shape[0] - 1, rows.shape[1], rows.shape[2]))


@_mixing_stack_op.register_fake
def _(u0, alpha, w0_all, w1_all, gw_all, lmax, focus_dim, apply_alpha):
    n_focus, n_edge, row = u0.shape
    return (
        u0.new_empty((n_edge, n_focus, row)),
        u0.new_empty((gw_all.shape[0], n_focus, n_edge, row)),
    )


@_mixing_stack_bwd_op.register_fake
def _(
    grad_out,
    x_local,
    z_all,
    alpha,
    w0t_all,
    w1t_all,
    gw_all,
    gwt_all,
    lmax,
    focus_dim,
    apply_alpha,
):
    n_gated, n_focus, n_edge, row = z_all.shape
    return (
        z_all.new_empty((n_focus, n_edge, row)),
        z_all.new_empty((n_edge, n_focus)),
    )


def _rotate_mix_setup_context(ctx, inputs, output):
    x, src, wigner, kc, cb, lmax, n_focus, rank = inputs
    ctx.save_for_backward(x, src, wigner, kc, cb)
    ctx.lmax = lmax
    ctx.n_focus = n_focus
    ctx.rank = rank


def _rotate_mix_backward(ctx, grad_u):
    x, src, wigner, kc, cb = ctx.saved_tensors
    grad_x_edge, grad_wigner, grad_kc = _rotate_mix_bwd_op(
        grad_u.contiguous(), x, src, wigner, kc, cb, ctx.lmax, ctx.n_focus, ctx.rank
    )
    # Contention-free segmented reduction of the per-edge node gradient; the
    # integer topology (argsort + CSR offsets) traces as ordinary aten ops.
    order = torch.argsort(src)
    boundaries = torch.arange(x.shape[0] + 1, device=src.device, dtype=src.dtype)
    row_ptr = torch.searchsorted(src.index_select(0, order), boundaries)
    grad_x = _segment_sum_op(grad_x_edge, order, row_ptr)
    return grad_x, None, grad_wigner, grad_kc, None, None, None, None


_rotate_mix_op.register_autograd(
    _rotate_mix_backward, setup_context=_rotate_mix_setup_context
)


def _mixing_stack_setup_context(ctx, inputs, output):
    u0, alpha, w0_all, w1_all, gw_all, lmax, focus_dim, apply_alpha = inputs
    x_local, z_all = output
    ctx.save_for_backward(alpha, x_local, z_all, w0_all, w1_all, gw_all)
    ctx.lmax = lmax
    ctx.focus_dim = focus_dim
    ctx.apply_alpha = apply_alpha


def _mixing_stack_backward(ctx, grad_out, grad_z_unused):
    alpha, x_local, z_all, w0_all, w1_all, gw_all = ctx.saved_tensors
    grad_u0, grad_alpha = _mixing_stack_bwd_op(
        grad_out.contiguous(),
        x_local,
        z_all,
        alpha,
        w0_all.transpose(2, 3).contiguous(),
        w1_all.transpose(2, 3).contiguous(),
        gw_all,
        gw_all.transpose(2, 3).contiguous(),
        ctx.lmax,
        ctx.focus_dim,
        ctx.apply_alpha,
    )
    return (
        grad_u0,
        grad_alpha if ctx.apply_alpha else None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


_mixing_stack_op.register_autograd(
    _mixing_stack_backward, setup_context=_mixing_stack_setup_context
)


# ======================================================================
# Per-convolution entry point
# ======================================================================
class _TritonSO2ValuePath:
    """Per-convolution entry running the SO(2) value path through the fused ops.

    The call contract mirrors the reference ``so2_message(...,
    return_local=True)``: it returns the post-focus-compete local features
    ``(E, F, D_m, Cf)`` and the projected radial features whose ``l = 0``
    slice feeds the attention aggregation.

    The stacked weights are assembled from the live parameters on every call
    and must not be cached across calls: the first call may run inside a
    ``make_fx`` fake-tensor trace, where a cache would capture fake weights,
    and eager weights may change when a checkpoint is loaded after
    construction.  The assembly is a short chain of parameter-only aten ops
    that the compile pipeline constant-folds out of the hot path.

    At ``DP_TRITON_INFER >= 3`` the mixing stack runs through the fp16x3
    tensor-core operator when the ``(focus_dim, lmax)`` key carries a
    validated configuration (see :mod:`.so2_stack_fp16x3`); the selection is
    fixed at construction, so exactly one stack operator reaches the traced
    graph.
    """

    def __init__(self, conv: SO2Convolution) -> None:
        self._conv = conv
        self._stack_op = _mixing_stack_op
        if (
            conv.triton_infer_level >= 3
            and stack_fp16x3_configs(conv.so2_focus_dim, conv.lmax) is not None
        ):
            from .so2_stack_fp16x3 import (
                mixing_stack_fp16x3,
            )

            self._stack_op = mixing_stack_fp16x3

    def _pack_weights(self) -> tuple[Tensor, Tensor, Tensor]:
        """Stack the SO(2) block weights and gate projections per layer.

        Returns ``(w0_all, w1_all, gw_all)`` with shapes
        ``(n_layers, F, M0, M0)``, ``(n_layers, F, M1, M1)`` and
        ``(n_gated, F, Cf, lmax * Cf)``, all in the ``(in, out)`` convention.
        """
        conv = self._conv
        m0 = (conv.lmax + 1) * conv.so2_focus_dim
        w0_list, w1_list, gw_list = [], [], []
        for layer, linear in enumerate(conv.so2_linears):
            weight = (
                linear._build_so2_weight().detach().permute(1, 0, 2).contiguous()
            )  # (F, D_m*Cf, D_m*Cf)
            w0_list.append(weight[:, :m0, :m0])
            w1_list.append(weight[:, m0:, m0:])
            non_linear = conv.non_linearities[layer]
            if type(non_linear).__name__ == "GatedActivation":
                gw_list.append(
                    non_linear.gate_linear.weight.detach()
                    .view(
                        conv.so2_focus_dim,
                        conv.n_focus,
                        conv.lmax * conv.so2_focus_dim,
                    )
                    .permute(1, 0, 2)
                )
        return (
            torch.stack(w0_list).contiguous(),
            torch.stack(w1_list).contiguous(),
            torch.stack(gw_list).contiguous(),
        )

    def __call__(
        self,
        x: Tensor,
        edge_cache: EdgeFeatureCache,
        radial_feat: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute the SO(2) local features and radial features via the fused ops.

        Parameters
        ----------
        x : Tensor
            Node features with shape (N, D, C_wide).
        edge_cache : EdgeFeatureCache
            Precomputed edge cache (provides ``src`` and the Wigner ``D_full``).
        radial_feat : Tensor
            Per-edge radial features with shape (E, lmax+1, C).

        Returns
        -------
        x_local : Tensor
            Post-focus-compete local features with shape (E, F, D_m, Cf).
        rad_feat : Tensor
            Projected radial features with shape (E, lmax+1, C_wide); its
            ``l = 0`` slice is consumed by the attention aggregation.  The
            degree-expanded ``(E, D_m, C_wide)`` layout of the reference path
            is never materialized: the mixer projection and the mixer-free
            multiply read only the ``lmax + 1`` per-degree rows.
        """
        conv = self._conv
        src = edge_cache.src
        w0_all, w1_all, gw_all = self._pack_weights()

        # === Step 1. Radial features and the compact degree kernel ===
        if conv.radial_hidden_proj is not None:
            rad_feat = conv.radial_hidden_proj(radial_feat)  # (E, lmax+1, C_wide)
        else:
            rad_feat = radial_feat
        mixer = conv.radial_degree_mixer
        if mixer is None:
            kc = rad_feat
            cb = rad_feat.new_zeros(1)
            rank = 0
        else:
            kc = torch.matmul(
                rad_feat.reshape(rad_feat.shape[0], -1), mixer.weight
            )  # (E, degree_kernel_size * rank)
            cb = mixer.channel_basis.reshape(-1)
            rank = mixer.rank

        # === Step 2. Fused rotate-to-local + degree mixing (focus-major) ===
        u0 = _rotate_mix_op(
            x.contiguous(),
            src,
            edge_cache.D_full,
            kc.contiguous(),
            cb.contiguous(),
            conv.lmax,
            conv.n_focus,
            rank,
        )

        # === Step 3. Cross-focus competition weight from the l = 0 scalars ===
        apply_alpha = bool(conv.focus_compete and conv.n_focus > 1)
        if apply_alpha:
            # The small (E, F, Cf) copy keeps the softmax backward from
            # retaining a view of the whole focus-major activation.
            gate_src = u0[:, :, : conv.so2_focus_dim].permute(1, 0, 2).contiguous()
            alpha = conv._focus_alpha(gate_src).to(u0.dtype).contiguous()
        else:
            alpha = torch.ones(
                src.shape[0], conv.n_focus, device=u0.device, dtype=u0.dtype
            )

        # === Step 4. Fused mixing stack (identity layer stores edge-major) ===
        x_local, _ = self._stack_op(
            u0,
            alpha,
            w0_all,
            w1_all,
            gw_all,
            conv.lmax,
            conv.so2_focus_dim,
            apply_alpha,
        )
        n_edge = src.shape[0]
        reduced_dim = 3 * conv.lmax + 1
        return (
            x_local.view(n_edge, conv.n_focus, reduced_dim, conv.so2_focus_dim),
            rad_feat,
        )


def _is_supported(conv: SO2Convolution) -> bool:
    """Return whether ``conv`` matches the fused value-path configuration."""
    if (
        conv.mmax != 1
        or not 1 <= conv.lmax <= _MAX_LMAX
        or conv.mixing_layers < 2
        or conv.so2_focus_dim not in _SUPPORTED_FOCUS_DIMS
        or conv.node_wise_grid_product is not None
        or conv.use_so2_attn_res
        or conv.layer_scale
        # Kernels accumulate in fp32; refuse other precisions rather than
        # silently down-casting a double-precision model.
        or conv.so2_linears[0].weight_m0.dtype is not torch.float32
    ):
        return False
    mixer = conv.radial_degree_mixer
    if mixer is not None and (
        mixer.mode != "degree_channel" or not 1 <= mixer.rank <= _MAX_MIXER_RANK
    ):
        return False
    if any(type(norm).__name__ != "Identity" for norm in conv.so2_inter_norms):
        return False
    if any(linear.bias0 is not None for linear in conv.so2_linears):
        return False
    if any(
        linear.in_channels != conv.so2_focus_dim
        or linear.out_channels != conv.so2_focus_dim
        for linear in conv.so2_linears
    ):
        return False
    non_linears = conv.non_linearities
    if any(
        type(non_linears[layer]).__name__ != "GatedActivation"
        or (
            getattr(non_linears[layer].scalar_act, "activation", None)
            or getattr(non_linears[layer], "activation_function", None)
        )
        != "silu"
        for layer in range(conv.mixing_layers - 1)
    ):
        return False
    return type(non_linears[conv.mixing_layers - 1]).__name__ == "Identity"


def make_triton_value_path(conv: SO2Convolution) -> _TritonSO2ValuePath | None:
    """Build the fused Triton value-path entry for a convolution block.

    Parameters
    ----------
    conv : SO2Convolution
        The convolution block to accelerate.

    Returns
    -------
    _TritonSO2ValuePath or None
        The entry callable when Triton is available and ``conv`` matches the
        supported configuration (``mmax == 1``, ``lmax`` 1..6, focus width in
        {32, 64, 96, 128}, gated stack with an identity final layer, radial
        mixer absent or ``degree_channel`` with rank 1..4, fp32 weights);
        otherwise ``None`` and the caller falls back to the reference path.
    """
    if not SO2_VALUE_PATH_TRITON_AVAILABLE or not _is_supported(conv):
        return None
    return _TritonSO2ValuePath(conv)
