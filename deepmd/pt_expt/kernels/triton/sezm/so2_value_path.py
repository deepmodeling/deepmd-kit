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
factory refuses non-fp32 weights rather than silently down-casting.  The
shape-keyed stack GEMM tiles preserve IEEE-fp32 arithmetic and are validated
against the conservative fallback because changing ``BLOCK_K`` may regroup
partial sums.  The swept tables live in :mod:`.tile_configs`.  At
``DP_TRITON_INFER >= 3`` the mixing stack is replaced by the fp16x3 tensor-core
operator of :mod:`.so2_stack_fp16x3` on validated winning shapes, the one
deliberate exception to the exact-fp32 contract.

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
    NamedTuple,
)

import torch
from torch import (
    Tensor,
)
from torch.library import (
    wrap_triton,
)

from .gated_activation import (
    gated_activation_second_order,
    gated_activation_second_order_reference,
)
from .indexing import (
    build_m_major_index,
)
from .second_order import (
    accumulate,
)
from .so2_rotation import (
    _block_to_local_op,
)
from .tile_configs import (
    GATE_BMM_MIN_FOCUS_DIM,
    gate_config,
    point_config,
    point_recompute_config,
    point_train_config,
    recompute_config,
    rotate_mix_bwd_block_config,
    rotate_mix_fwd_config,
    stack_fp16x3_configs,
    stack_fp32_configs,
    stack_m0_gate_config,
)

if TYPE_CHECKING:
    from deepmd.dpmodel.descriptor.dpa4_nn.edge_cache import (
        EdgeCache,
    )
    from deepmd.dpmodel.descriptor.dpa4_nn.so2 import (
        SO2Convolution,
    )

__all__ = [
    "SO2_VALUE_PATH_TRITON_AVAILABLE",
    "fused_gated_activation",
    "make_triton_rotate_mix",
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
) -> tuple[Tensor, Tensor, Tensor]:
    """Eager ground truth for ``so2_mixing_stack``.

    Returns the edge-major output ``(E, F, ROW)``, the stacked gated-layer
    pre-activations ``(n_gated, F, E, ROW)``, and the input of the final
    identity layer ``(F, E, ROW)``, from which the backward recovers every
    gated layer's input.
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
    u_final = u
    out = u.clone()
    out[:, :, :m0] += torch.bmm(u[:, :, :m0], w0_all[n_gated])
    out[:, :, m0:] += torch.bmm(u[:, :, m0:], w1_all[n_gated])
    if apply_alpha:
        out = out * alpha.transpose(0, 1).unsqueeze(-1).to(out.dtype)
    x_local = out.permute(1, 0, 2).contiguous()
    z_all = (
        torch.stack(z_saved) if n_gated > 0 else u0.new_empty(0, n_focus, n_edge, row)
    )
    return x_local, z_all, u_final


def _mixing_stack_backward_reference(
    grad_out: Tensor,
    x_local: Tensor,
    z_all: Tensor,
    u_final: Tensor,
    alpha: Tensor,
    w0t_all: Tensor,
    w1t_all: Tensor,
    gw_all: Tensor,
    gwt_all: Tensor,
    grad_z_upstream: Tensor | None,
    grad_u_upstream: Tensor | None,
    lmax: int,
    focus_dim: int,
    apply_alpha: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Closed-form eager backward of ``so2_mixing_stack``.

    Returns ``(grad_u0, grad_alpha, grad_w0_all, grad_w1_all, grad_gw_all,
    upstream_all, input_all, grad_z_all, grad_logit_all)``. ``grad_alpha`` is
    meaningful only when ``apply_alpha`` is set (the identity ``grad_alpha =
    sum(grad * out) / alpha`` is exact because the final store is a plain
    scale). The weight gradients follow the forward orientation ``z = u W``;
    the final identity layer occupies the last slot of the stacked
    block-weight gradients, matching the layout the forward reads from. The
    last four outputs stack, per gated layer, the gradient entering the
    layer's pointwise backward, the recovered layer input, and the
    pre-activation and gate-logit gradients -- the surfaces the second order
    linearizes around.
    """
    n_gated = gw_all.shape[0]
    m0 = (lmax + 1) * focus_dim
    grad_w0_layers: list[Tensor] = []
    grad_w1_layers: list[Tensor] = []
    grad_gw_layers: list[Tensor] = []
    upstream_layers: list[Tensor] = []
    input_layers: list[Tensor] = []
    grad_z_layers: list[Tensor] = []
    grad_logit_layers: list[Tensor] = []
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
    if grad_u_upstream is not None:
        g_cur = g_cur + grad_u_upstream
    grad_w0_layers.append(torch.bmm(u_final[:, :, :m0].transpose(1, 2), g[:, :, :m0]))
    grad_w1_layers.append(torch.bmm(u_final[:, :, m0:].transpose(1, 2), g[:, :, m0:]))
    u_next = u_final
    for layer in range(n_gated - 1, -1, -1):
        upstream_layers.append(g_cur)
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
        if grad_z_upstream is not None:
            gz0 = gz0 + grad_z_upstream[layer][:, :, :m0]
            gz1 = gz1 + grad_z_upstream[layer][:, :, m0:]
        grad_z_layers.append(torch.cat([gz0, gz1], dim=-1))
        grad_logit_layers.append(g_logit)
        act = torch.cat(
            [
                z_scalar * s0,
                z0[:, :, focus_dim:] * sig,
                z1 * sig2,
            ],
            dim=-1,
        )
        u_next = u_next - act
        input_layers.append(u_next)
        grad_w0_layers.append(torch.bmm(u_next[:, :, :m0].transpose(1, 2), gz0))
        grad_w1_layers.append(torch.bmm(u_next[:, :, m0:].transpose(1, 2), gz1))
        grad_gw_layers.append(torch.bmm(z_scalar.transpose(1, 2), g_logit))
        g_next = g_cur.clone()
        g_next[:, :, :m0] += torch.bmm(gz0, w0t_all[layer])
        g_next[:, :, m0:] += torch.bmm(gz1, w1t_all[layer])
        g_cur = g_next
    n_focus = g_cur.shape[0]
    grad_w0_all = torch.stack(grad_w0_layers[:0:-1] + grad_w0_layers[:1])
    grad_w1_all = torch.stack(grad_w1_layers[:0:-1] + grad_w1_layers[:1])
    grad_gw_all = (
        torch.stack(grad_gw_layers[::-1])
        if grad_gw_layers
        else grad_out.new_empty((0, n_focus, focus_dim, lmax * focus_dim))
    )
    empty_row = grad_out.new_empty((0, *g_cur.shape))
    upstream_all = torch.stack(upstream_layers[::-1]) if upstream_layers else empty_row
    input_all = torch.stack(input_layers[::-1]) if input_layers else empty_row
    grad_z_all = torch.stack(grad_z_layers[::-1]) if grad_z_layers else empty_row
    grad_logit_all = (
        torch.stack(grad_logit_layers[::-1]).float()
        if grad_logit_layers
        else grad_out.new_empty(
            (0, n_focus, g_cur.shape[1], lmax * focus_dim), dtype=torch.float32
        )
    )
    return (
        g_cur,
        grad_alpha,
        grad_w0_all,
        grad_w1_all,
        grad_gw_all,
        upstream_all,
        input_all,
        grad_z_all,
        grad_logit_all,
    )


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
    def _stack_gemm_m0_gate_kernel(
        u_ptr,  # (F, E, ROW) layer input
        w0_ptr,  # (NL, F, M0, M0) stacked m = 0 weights
        gw_ptr,  # (NL, F, CF, L*CF) stacked gate projections
        v_ptr,  # (F, E, ROW) layer output
        z_ptr,  # (NL, F, E, ROW) saved raw pre-activation
        sig_ptr,  # (F, E, L*CF) gate sigmoid output
        n_edge,
        layer,
        L: tl.constexpr,
        CF: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Fused gated ``m = 0`` GEMM, sigmoid projection and residual epilogue.

        One program owns every ``CF``-wide degree group for an edge tile.  The
        scalar pre-activation therefore remains available for the gate dots,
        eliminating the full ``m = 0`` readback and output round trip of the
        separate gate kernel while preserving ``z`` for the force backward.
        """
        M0: tl.constexpr = (L + 1) * CF
        LG: tl.constexpr = L * CF
        ROW: tl.constexpr = (3 * L + 1) * CF
        CP: tl.constexpr = triton.next_power_of_2(CF)

        pid_m = tl.program_id(0)
        fid = tl.program_id(1).to(tl.int64)
        n_focus = tl.num_programs(1)
        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        m_mask = offs_m < n_edge
        mm = m_mask[:, None]
        nc = tl.arange(0, CP)
        c_mask = nc < CF
        cm = mm & c_mask[None, :]
        offs_k = tl.arange(0, BLOCK_K)

        u_row = u_ptr + fid * n_edge * ROW + offs_m * ROW
        weight_base = w0_ptr + (layer * n_focus + fid) * M0 * M0
        accs = ()
        for _ in tl.static_range(L + 1):
            accs = accs + (tl.zeros((BLOCK_M, CP), dtype=tl.float32),)
        for k_base in range(0, M0, BLOCK_K):
            k = k_base + offs_k
            k_mask = k < M0
            a = tl.load(
                u_row[:, None] + k[None, :],
                mask=mm & k_mask[None, :],
                other=0.0,
            )
            next_accs = ()
            for group in tl.static_range(L + 1):
                w = tl.load(
                    weight_base + k[:, None] * M0 + (group * CF + nc)[None, :],
                    mask=k_mask[:, None] & c_mask[None, :],
                    other=0.0,
                )
                next_accs = next_accs + (
                    tl.dot(a, w, accs[group], input_precision="ieee"),
                )
            accs = next_accs

        z_row = z_ptr + (layer * n_focus + fid) * n_edge * ROW + offs_m * ROW
        v_row = v_ptr + fid * n_edge * ROW + offs_m * ROW
        sig_row = sig_ptr + (fid * n_edge + offs_m) * LG
        z_s = accs[0]
        u_s = tl.load(u_row[:, None] + nc[None, :], mask=cm, other=0.0)
        tl.store(z_row[:, None] + nc[None, :], z_s, mask=cm)
        tl.store(v_row[:, None] + nc[None, :], u_s + z_s * tl.sigmoid(z_s), mask=cm)

        weight_gate_base = gw_ptr + (layer * n_focus + fid) * CF * LG
        wm = c_mask[:, None] & c_mask[None, :]
        for group in tl.static_range(L):
            z_group = accs[group + 1]
            gw = tl.load(
                weight_gate_base + nc[:, None] * LG + (group * CF + nc)[None, :],
                mask=wm,
                other=0.0,
            ).to(tl.float32)
            sig = tl.sigmoid(tl.dot(z_s, gw, input_precision="ieee"))
            col = (group + 1) * CF + nc
            u_group = tl.load(u_row[:, None] + col[None, :], mask=cm, other=0.0)
            tl.store(z_row[:, None] + col[None, :], z_group, mask=cm)
            tl.store(
                v_row[:, None] + col[None, :],
                u_group + z_group * sig,
                mask=cm,
            )
            tl.store(sig_row[:, None] + (group * CF + nc)[None, :], sig, mask=cm)

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

        # l = 0 scalar rows pass through silu. Loads widen to fp32 for the
        # sigmoid; stores narrow back to the output dtype.
        z_s = tl.load(z_row[:, None] + nc[None, :], mask=cm, other=0.0).to(tl.float32)
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
                ).to(tl.float32)
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
        z_s = tl.load(z_row[:, None] + nc[None, :], mask=cm, other=0.0).to(tl.float32)
        sig_row = sig_ptr + (fid * n_edge + offs_m) * LG
        for g in tl.static_range(L):
            gw_g = tl.load(
                gw_ptr
                + (layer * n_focus + fid) * CF * LG
                + nc[:, None] * LG
                + (g * CF + nc)[None, :],
                mask=wm,
                other=0.0,
            ).to(tl.float32)
            sig_g = tl.sigmoid(tl.dot(z_s, gw_g, input_precision="ieee"))
            tl.store(sig_row[:, None] + (g * CF + nc)[None, :], sig_g, mask=cm)

    @triton.jit
    def _stack_point_bwd_kernel(
        g_ptr,  # (F, E, ROW) upstream gradient of the layer output
        z_ptr,  # z_all stack, layer selected by ``layer``
        sig_ptr,  # (F, E, L*CF) gate sigmoids
        gwt_ptr,  # (NL, F, L*CF, CF) transposed gate projections
        gz_ptr,  # (F, E, ROW) pre-activation gradient output
        gl_ptr,  # (F, E, L*CF) gate-logit gradient output
        un_ptr,  # (F, E, ROW) layer output, read only when RECOVER_INPUT
        up_ptr,  # (F, E, ROW) layer input, written only when RECOVER_INPUT
        n_edge,
        layer,
        L: tl.constexpr,
        CF: tl.constexpr,
        GLOGIT_OUT: tl.constexpr,
        GLOGIT_STORE: tl.constexpr,
        RECOMPUTE_SIG: tl.constexpr,
        RECOVER_INPUT: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        """Pointwise part of the gated-layer backward.

        Produces the pre-activation gradient ``gz`` for the value rows and the
        gate-path contribution to the ``l = 0`` scalar rows.  The gate-logit
        contraction back to the scalars is either folded in as a ``CP x CP``
        register dot (small ``CF``) or emitted to ``gl`` for an external
        batched GEMM (wide-channel regime, where the register dot spills);
        ``GLOGIT_OUT`` selects between the two.

        ``GLOGIT_STORE`` is independent of that choice: training contracts the
        gate-logit gradient against the pre-activation to form the gate weight's
        gradient, so it must be written out even when the contraction back to
        the scalars was folded into the register dot.

        ``RECOVER_INPUT`` additionally reconstructs the layer's input from its
        output, ``u_l = u_{l+1} - act(z_l)``. The activation is already in
        registers here, so the layer input the weight gradient contracts against
        costs no extra pass and no stored activation in the forward.
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
        un_row = un_ptr + fid * n_edge * ROW + offs_m * ROW
        up_row = up_ptr + fid * n_edge * ROW + offs_m * ROW

        # l = 0 value path: silu backward. Loads are widened to fp32: the
        # sigmoid and the register dots require it, and stores narrow back to
        # the output dtype.
        z_s = tl.load(z_row[:, None] + nc[None, :], mask=cm, other=0.0).to(tl.float32)
        g_s = tl.load(g_row[:, None] + nc[None, :], mask=cm, other=0.0).to(tl.float32)
        s0 = tl.sigmoid(z_s)
        gz_s = g_s * s0 * (1.0 + z_s * (1.0 - s0))
        if RECOVER_INPUT:
            un_s = tl.load(un_row[:, None] + nc[None, :], mask=cm, other=0.0)
            tl.store(up_row[:, None] + nc[None, :], un_s - z_s * s0, mask=cm)

        for g in tl.static_range(L):
            if RECOMPUTE_SIG:
                # The transposed gate weight is already required by the
                # backward contraction.  Reading its transpose here removes
                # the intermediate sigmoid surface and a separate kernel.
                gw_g = tl.load(
                    gwt_ptr
                    + (layer * n_focus + fid) * LG * CF
                    + (g * CF + nc)[None, :] * CF
                    + nc[:, None],
                    mask=wm,
                    other=0.0,
                ).to(tl.float32)
                sig_g = tl.sigmoid(tl.dot(z_s, gw_g, input_precision="ieee"))
            else:
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
            if RECOVER_INPUT:
                # The three gated rows of this group, undone from the output.
                un_r0 = tl.load(
                    un_row[:, None] + ((1 + g) * CF + nc)[None, :], mask=cm, other=0.0
                )
                tl.store(
                    up_row[:, None] + ((1 + g) * CF + nc)[None, :],
                    un_r0 - zr0 * sig_g,
                    mask=cm,
                )
                un_rn = tl.load(
                    un_row[:, None] + (rn * CF + nc)[None, :], mask=cm, other=0.0
                )
                tl.store(
                    up_row[:, None] + (rn * CF + nc)[None, :],
                    un_rn - zrn * sig_g,
                    mask=cm,
                )
                un_rp = tl.load(
                    un_row[:, None] + (rp * CF + nc)[None, :], mask=cm, other=0.0
                )
                tl.store(
                    up_row[:, None] + (rp * CF + nc)[None, :],
                    un_rp - zrp * sig_g,
                    mask=cm,
                )
            # Gate path: three value rows share gate group g.
            g_sig = gr0 * zr0 + grn * zrn + grp * zrp
            g_logit = g_sig * sig_g * (1.0 - sig_g)
            if GLOGIT_STORE:
                tl.store(gl_row[:, None] + (g * CF + nc)[None, :], g_logit, mask=cm)
            if not GLOGIT_OUT:
                gwt_g = tl.load(
                    gwt_ptr
                    + (layer * n_focus + fid) * LG * CF
                    + (g * CF + nc)[:, None] * CF
                    + nc[None, :],
                    mask=wm,
                    other=0.0,
                ).to(tl.float32)
                gz_s = tl.dot(
                    g_logit.to(tl.float32), gwt_g, gz_s, input_precision="ieee"
                )

        tl.store(gz_row[:, None] + nc[None, :], gz_s, mask=cm)

    @triton.jit
    def _gated_act_fwd_kernel(
        z_ptr,  # (F, E, ROW) pre-activation, focus-major
        gw_ptr,  # (F, CF, L*CF) gate projection; unread when SIG_IN
        sig_ptr,  # (F, E, L*CF) precomputed gate sigmoids, read when SIG_IN
        v_ptr,  # (F, E, ROW) activated output
        n_edge,
        L: tl.constexpr,
        CF: tl.constexpr,
        SIG_IN: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        """Standalone gated activation forward: ``v = act(z)``.

        The scalar rows pass through SiLU; each degree group's sigmoid gate
        scales the three value rows (the ``m = 0`` row and the signed
        ``|m| = 1`` pair) that share it.  The gate is either evaluated from
        the scalar rows through a ``CP x CP`` register dot, or, with
        ``SIG_IN``, read from a cuBLAS-produced surface (wide-channel regime,
        where the register dot spills).  The layout and gate-group mapping
        follow the module-level contract of the mixing-stack kernels.
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

        z_row = z_ptr + fid * n_edge * ROW + offs_m * ROW
        v_row = v_ptr + fid * n_edge * ROW + offs_m * ROW
        sig_row = sig_ptr + (fid * n_edge + offs_m) * LG

        # l = 0 scalar rows pass through SiLU. Loads widen to fp32 for the
        # sigmoid; stores narrow back to the output dtype.
        z_s = tl.load(z_row[:, None] + nc[None, :], mask=cm, other=0.0).to(tl.float32)
        tl.store(v_row[:, None] + nc[None, :], z_s * tl.sigmoid(z_s), mask=cm)

        for g in tl.static_range(L):
            if SIG_IN:
                sig_g = tl.load(
                    sig_row[:, None] + (g * CF + nc)[None, :], mask=cm, other=0.0
                )
            else:
                gw_g = tl.load(
                    gw_ptr + fid * CF * LG + nc[:, None] * LG + (g * CF + nc)[None, :],
                    mask=wm,
                    other=0.0,
                ).to(tl.float32)
                sig_g = tl.sigmoid(tl.dot(z_s, gw_g, input_precision="ieee"))
            r0 = (1 + g) * CF
            rn = ((L + 1) + g) * CF
            rp = ((2 * L + 1) + g) * CF
            z_r0 = tl.load(z_row[:, None] + (r0 + nc)[None, :], mask=cm, other=0.0)
            z_rn = tl.load(z_row[:, None] + (rn + nc)[None, :], mask=cm, other=0.0)
            z_rp = tl.load(z_row[:, None] + (rp + nc)[None, :], mask=cm, other=0.0)
            tl.store(v_row[:, None] + (r0 + nc)[None, :], z_r0 * sig_g, mask=cm)
            tl.store(v_row[:, None] + (rn + nc)[None, :], z_rn * sig_g, mask=cm)
            tl.store(v_row[:, None] + (rp + nc)[None, :], z_rp * sig_g, mask=cm)

    @triton.jit
    def _stack_train_traversal_kernel(
        go_ptr,  # (E, F, ROW) edge-major output cotangent
        z_ptr,  # (NL, F, E, ROW) stacked pre-activations
        uf_ptr,  # (F, E, ROW) final identity layer input
        alpha_ptr,  # (E, F) focus competition weight
        w0t_ptr,  # (NL+1, F, M0, M0) transposed block weights
        w1t_ptr,  # (NL+1, F, M1, M1) transposed block weights
        gw_ptr,  # (NL, F, CF, L*CF) gate projections
        gzup_ptr,  # (NL, F, E, ROW) upstream pre-activation gradient, optional
        guup_ptr,  # (F, E, ROW) upstream final-activation gradient, optional
        gu0_ptr,  # out (F, E, ROW) input gradient; doubles as the running head
        ga_ptr,  # out (E, F) competition gradient
        gz_all_ptr,  # out (NL, F, E, ROW) pre-activation gradients
        gq_all_ptr,  # out (NL, F, E, L*CF) gate-logit gradients
        u_all_ptr,  # out (NL, F, E, ROW) recovered layer inputs
        up_all_ptr,  # out (NL, F, E, ROW) per-layer upstream, written when KEEP
        n_edge,
        NL: tl.constexpr,
        L: tl.constexpr,
        CF: tl.constexpr,
        APPLY_ALPHA: tl.constexpr,
        NEED_ALPHA: tl.constexpr,
        HAS_GZUP: tl.constexpr,
        HAS_GUUP: tl.constexpr,
        KEEP: tl.constexpr,
        BLOCK_E: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Whole-stack training backward in one launch.

        One program walks every gated layer for a block of edges. The running
        gradient lives in the ``gu0`` output surface and each layer's
        pre-activation gradient in its ``gz_all`` slice, so the inter-layer
        traffic is block-local and served by the L2 cache; nothing returns to
        the host between layers. The weight gradients are *not* reduced here:
        the traversal stores the per-edge cotangent surfaces they contract
        against, and cuBLAS performs the edge reduction afterwards.

        Phases per layer, matching the multi-kernel formulation exactly:

        1. Pointwise backward of the gated activation (gate sigmoids
           recomputed from the scalar rows), producing ``gz``/``gq`` and the
           recovered input ``u_l = u_{l+1} - act(z_l)``.
        2. Residual contraction ``g += gz @ W^T`` over both block-diagonal
           halves, tiled over the output columns.
        """
        M0: tl.constexpr = (L + 1) * CF
        M1: tl.constexpr = 2 * L * CF
        ROW: tl.constexpr = (3 * L + 1) * CF
        LG: tl.constexpr = L * CF
        CP: tl.constexpr = triton.next_power_of_2(CF)
        NT0: tl.constexpr = (M0 + BLOCK_N - 1) // BLOCK_N
        NT1: tl.constexpr = (M1 + BLOCK_N - 1) // BLOCK_N

        pid_e = tl.program_id(0)
        fid = tl.program_id(1).to(tl.int64)
        n_focus = tl.num_programs(1)

        offs_e = (pid_e * BLOCK_E + tl.arange(0, BLOCK_E)).to(tl.int64)
        e_mask = offs_e < n_edge
        em = e_mask[:, None]
        nc = tl.arange(0, CP)
        cm = em & (nc < CF)[None, :]
        wm = ((nc < CF)[:, None]) & ((nc < CF)[None, :])
        offs_k = tl.arange(0, BLOCK_K)

        go_row = go_ptr + offs_e * (n_focus * ROW) + fid * ROW
        gu_row = gu0_ptr + fid * n_edge * ROW + offs_e * ROW
        uf_row = uf_ptr + fid * n_edge * ROW + offs_e * ROW
        if APPLY_ALPHA:
            av = tl.load(alpha_ptr + offs_e * n_focus + fid, mask=e_mask, other=0.0).to(
                tl.float32
            )

        # === Final identity layer: g = go + go @ W^T, alpha folded on the
        # fly; the competition gradient reduces go against the recomputed
        # unscaled output in the same column sweep. ===
        ga_acc = tl.zeros((BLOCK_E,), dtype=tl.float32)
        w_base0 = w0t_ptr + (NL * n_focus + fid) * M0 * M0
        w_base1 = w1t_ptr + (NL * n_focus + fid) * M1 * M1
        for pid_n in tl.static_range(NT0 + NT1):
            if pid_n < NT0:
                offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
                n_mask = offs_n < M0
                col = offs_n
            else:
                offs_n = (pid_n - NT0) * BLOCK_N + tl.arange(0, BLOCK_N)
                n_mask = offs_n < M1
                col = M0 + offs_n
            acc = tl.zeros((BLOCK_E, BLOCK_N), dtype=tl.float32)
            if NEED_ALPHA:
                y_acc = tl.zeros((BLOCK_E, BLOCK_N), dtype=tl.float32)
            if pid_n < NT0:
                a_ptrs = go_row[:, None] + offs_k[None, :]
                u_ptrs = uf_row[:, None] + offs_k[None, :]
                w_ptrs = w_base0 + offs_k[:, None] * M0 + offs_n[None, :]
                # The forward product u @ W reads the transposed weight along
                # the other axis; its tile lives at the mirrored offsets.
                wf_ptrs = w_base0 + offs_n[:, None] * M0 + offs_k[None, :]
                for k0 in range(0, M0, BLOCK_K):
                    k_mask = (k0 + offs_k) < M0
                    a = tl.load(a_ptrs, mask=em & k_mask[None, :], other=0.0)
                    w = tl.load(
                        w_ptrs,
                        mask=k_mask[:, None] & n_mask[None, :],
                        other=0.0,
                    )
                    if a.dtype == tl.float32:
                        acc = tl.dot(a, w, acc, input_precision="ieee")
                    else:
                        acc = tl.dot(a, w, acc)
                    if NEED_ALPHA:
                        uv = tl.load(u_ptrs, mask=em & k_mask[None, :], other=0.0)
                        wf = tl.load(
                            wf_ptrs,
                            mask=n_mask[:, None] & k_mask[None, :],
                            other=0.0,
                        )
                        if uv.dtype == tl.float32:
                            y_acc = tl.dot(
                                uv, tl.trans(wf), y_acc, input_precision="ieee"
                            )
                        else:
                            y_acc = tl.dot(uv, tl.trans(wf), y_acc)
                    a_ptrs += BLOCK_K
                    u_ptrs += BLOCK_K
                    w_ptrs += BLOCK_K * M0
                    wf_ptrs += BLOCK_K
            else:
                a_ptrs = go_row[:, None] + M0 + offs_k[None, :]
                u_ptrs = uf_row[:, None] + M0 + offs_k[None, :]
                w_ptrs = w_base1 + offs_k[:, None] * M1 + offs_n[None, :]
                wf_ptrs = w_base1 + offs_n[:, None] * M1 + offs_k[None, :]
                for k0 in range(0, M1, BLOCK_K):
                    k_mask = (k0 + offs_k) < M1
                    a = tl.load(a_ptrs, mask=em & k_mask[None, :], other=0.0)
                    w = tl.load(
                        w_ptrs,
                        mask=k_mask[:, None] & n_mask[None, :],
                        other=0.0,
                    )
                    if a.dtype == tl.float32:
                        acc = tl.dot(a, w, acc, input_precision="ieee")
                    else:
                        acc = tl.dot(a, w, acc)
                    if NEED_ALPHA:
                        uv = tl.load(u_ptrs, mask=em & k_mask[None, :], other=0.0)
                        wf = tl.load(
                            wf_ptrs,
                            mask=n_mask[:, None] & k_mask[None, :],
                            other=0.0,
                        )
                        if uv.dtype == tl.float32:
                            y_acc = tl.dot(
                                uv, tl.trans(wf), y_acc, input_precision="ieee"
                            )
                        else:
                            y_acc = tl.dot(uv, tl.trans(wf), y_acc)
                    a_ptrs += BLOCK_K
                    u_ptrs += BLOCK_K
                    w_ptrs += BLOCK_K * M1
                    wf_ptrs += BLOCK_K
            go_tile = tl.load(
                go_row[:, None] + col[None, :], mask=em & n_mask[None, :], other=0.0
            ).to(tl.float32)
            if NEED_ALPHA:
                u_tile = tl.load(
                    uf_row[:, None] + col[None, :],
                    mask=em & n_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                ga_acc += tl.sum(
                    tl.where(n_mask[None, :], go_tile * (u_tile + y_acc), 0.0), 1
                )
            g_tile = go_tile + acc
            if APPLY_ALPHA:
                g_tile = g_tile * av[:, None]
            if HAS_GUUP:
                guup_row = guup_ptr + fid * n_edge * ROW + offs_e * ROW
                g_tile += tl.load(
                    guup_row[:, None] + col[None, :],
                    mask=em & n_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
            tl.store(gu_row[:, None] + col[None, :], g_tile, mask=em & n_mask[None, :])
        if NEED_ALPHA:
            tl.store(ga_ptr + offs_e * n_focus + fid, ga_acc, mask=e_mask)

        # === Gated layers, last to first ===
        for step in tl.static_range(NL):
            layer = NL - 1 - step
            z_row = z_ptr + (layer * n_focus + fid) * n_edge * ROW + offs_e * ROW
            gz_row = gz_all_ptr + (layer * n_focus + fid) * n_edge * ROW + offs_e * ROW
            gq_row = gq_all_ptr + (layer * n_focus + fid) * n_edge * LG + offs_e * LG
            ul_row = u_all_ptr + (layer * n_focus + fid) * n_edge * ROW + offs_e * ROW
            if step == 0:
                un_row = uf_row
            else:
                un_row = (
                    u_all_ptr
                    + ((layer + 1) * n_focus + fid) * n_edge * ROW
                    + offs_e * ROW
                )
            if HAS_GZUP:
                gzup_row = (
                    gzup_ptr + (layer * n_focus + fid) * n_edge * ROW + offs_e * ROW
                )

            # --- Phase 1: pointwise backward, gate dot in registers ---
            if KEEP:
                up_row = (
                    up_all_ptr + (layer * n_focus + fid) * n_edge * ROW + offs_e * ROW
                )
            z_s = tl.load(z_row[:, None] + nc[None, :], mask=cm, other=0.0).to(
                tl.float32
            )
            g_s = tl.load(gu_row[:, None] + nc[None, :], mask=cm, other=0.0).to(
                tl.float32
            )
            if KEEP:
                tl.store(up_row[:, None] + nc[None, :], g_s, mask=cm)
            s0 = tl.sigmoid(z_s)
            gz_s = g_s * s0 * (1.0 + z_s * (1.0 - s0))
            un_s = tl.load(un_row[:, None] + nc[None, :], mask=cm, other=0.0)
            tl.store(ul_row[:, None] + nc[None, :], un_s - z_s * s0, mask=cm)

            for grp in tl.static_range(L):
                gw_g = tl.load(
                    gw_ptr
                    + (layer * n_focus + fid) * CF * LG
                    + nc[:, None] * LG
                    + (grp * CF + nc)[None, :],
                    mask=wm,
                    other=0.0,
                ).to(tl.float32)
                sig_g = tl.sigmoid(tl.dot(z_s, gw_g, input_precision="ieee"))
                r0 = (1 + grp) * CF
                rn = ((L + 1) + grp) * CF
                rp = ((2 * L + 1) + grp) * CF
                gr0 = tl.load(
                    gu_row[:, None] + (r0 + nc)[None, :], mask=cm, other=0.0
                ).to(tl.float32)
                grn = tl.load(
                    gu_row[:, None] + (rn + nc)[None, :], mask=cm, other=0.0
                ).to(tl.float32)
                grp_v = tl.load(
                    gu_row[:, None] + (rp + nc)[None, :], mask=cm, other=0.0
                ).to(tl.float32)
                if KEEP:
                    tl.store(up_row[:, None] + (r0 + nc)[None, :], gr0, mask=cm)
                    tl.store(up_row[:, None] + (rn + nc)[None, :], grn, mask=cm)
                    tl.store(up_row[:, None] + (rp + nc)[None, :], grp_v, mask=cm)
                zr0 = tl.load(
                    z_row[:, None] + (r0 + nc)[None, :], mask=cm, other=0.0
                ).to(tl.float32)
                zrn = tl.load(
                    z_row[:, None] + (rn + nc)[None, :], mask=cm, other=0.0
                ).to(tl.float32)
                zrp = tl.load(
                    z_row[:, None] + (rp + nc)[None, :], mask=cm, other=0.0
                ).to(tl.float32)
                gz_r0 = gr0 * sig_g
                gz_rn = grn * sig_g
                gz_rp = grp_v * sig_g
                if HAS_GZUP:
                    gz_r0 += tl.load(
                        gzup_row[:, None] + (r0 + nc)[None, :], mask=cm, other=0.0
                    ).to(tl.float32)
                    gz_rn += tl.load(
                        gzup_row[:, None] + (rn + nc)[None, :], mask=cm, other=0.0
                    ).to(tl.float32)
                    gz_rp += tl.load(
                        gzup_row[:, None] + (rp + nc)[None, :], mask=cm, other=0.0
                    ).to(tl.float32)
                tl.store(gz_row[:, None] + (r0 + nc)[None, :], gz_r0, mask=cm)
                tl.store(gz_row[:, None] + (rn + nc)[None, :], gz_rn, mask=cm)
                tl.store(gz_row[:, None] + (rp + nc)[None, :], gz_rp, mask=cm)
                # Recover the gated rows of the layer input.
                un_r0 = tl.load(
                    un_row[:, None] + (r0 + nc)[None, :], mask=cm, other=0.0
                )
                un_rn = tl.load(
                    un_row[:, None] + (rn + nc)[None, :], mask=cm, other=0.0
                )
                un_rp = tl.load(
                    un_row[:, None] + (rp + nc)[None, :], mask=cm, other=0.0
                )
                tl.store(
                    ul_row[:, None] + (r0 + nc)[None, :], un_r0 - zr0 * sig_g, mask=cm
                )
                tl.store(
                    ul_row[:, None] + (rn + nc)[None, :], un_rn - zrn * sig_g, mask=cm
                )
                tl.store(
                    ul_row[:, None] + (rp + nc)[None, :], un_rp - zrp * sig_g, mask=cm
                )
                # Gate-logit gradient and its register-dot fold onto the
                # scalar rows.
                g_logit = (gr0 * zr0 + grn * zrn + grp_v * zrp) * sig_g * (1.0 - sig_g)
                tl.store(gq_row[:, None] + (grp * CF + nc)[None, :], g_logit, mask=cm)
                gz_s = tl.dot(g_logit, tl.trans(gw_g), gz_s, input_precision="ieee")
            if HAS_GZUP:
                gz_s += tl.load(gzup_row[:, None] + nc[None, :], mask=cm, other=0.0).to(
                    tl.float32
                )
            tl.store(gz_row[:, None] + nc[None, :], gz_s, mask=cm)
            # Phase 2 re-reads the surface phase 1 just stored through
            # differently shaped pointers; the fence forbids the compiler
            # from reordering those accesses across the phase boundary.
            tl.debug_barrier()

            # --- Phase 2: g += gz @ W^T over both block-diagonal halves ---
            wl_base0 = w0t_ptr + (layer * n_focus + fid) * M0 * M0
            wl_base1 = w1t_ptr + (layer * n_focus + fid) * M1 * M1
            for pid_n in tl.static_range(NT0 + NT1):
                if pid_n < NT0:
                    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
                    n_mask = offs_n < M0
                    col = offs_n
                else:
                    offs_n = (pid_n - NT0) * BLOCK_N + tl.arange(0, BLOCK_N)
                    n_mask = offs_n < M1
                    col = M0 + offs_n
                acc = tl.zeros((BLOCK_E, BLOCK_N), dtype=tl.float32)
                if pid_n < NT0:
                    a_ptrs = gz_row[:, None] + offs_k[None, :]
                    w_ptrs = wl_base0 + offs_k[:, None] * M0 + offs_n[None, :]
                    for k0 in range(0, M0, BLOCK_K):
                        k_mask = (k0 + offs_k) < M0
                        a = tl.load(a_ptrs, mask=em & k_mask[None, :], other=0.0)
                        w = tl.load(
                            w_ptrs,
                            mask=k_mask[:, None] & n_mask[None, :],
                            other=0.0,
                        )
                        if a.dtype == tl.float32:
                            acc = tl.dot(a, w, acc, input_precision="ieee")
                        else:
                            acc = tl.dot(a, w, acc)
                        a_ptrs += BLOCK_K
                        w_ptrs += BLOCK_K * M0
                else:
                    a_ptrs = gz_row[:, None] + M0 + offs_k[None, :]
                    w_ptrs = wl_base1 + offs_k[:, None] * M1 + offs_n[None, :]
                    for k0 in range(0, M1, BLOCK_K):
                        k_mask = (k0 + offs_k) < M1
                        a = tl.load(a_ptrs, mask=em & k_mask[None, :], other=0.0)
                        w = tl.load(
                            w_ptrs,
                            mask=k_mask[:, None] & n_mask[None, :],
                            other=0.0,
                        )
                        if a.dtype == tl.float32:
                            acc = tl.dot(a, w, acc, input_precision="ieee")
                        else:
                            acc = tl.dot(a, w, acc)
                        a_ptrs += BLOCK_K
                        w_ptrs += BLOCK_K * M1
                g_prev = tl.load(
                    gu_row[:, None] + col[None, :],
                    mask=em & n_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                tl.store(
                    gu_row[:, None] + col[None, :],
                    g_prev + acc,
                    mask=em & n_mask[None, :],
                )
            # The next layer's pointwise phase reads the head just updated.
            tl.debug_barrier()

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
        and tensor.dtype in (torch.float32, torch.bfloat16, torch.float16)
    )


_PointwiseConfig = tuple[int, int, int]
_PointBackwardSchedule = tuple[bool, _PointwiseConfig, _PointwiseConfig]


# Per-focus surface size (bytes) up to which the whole-stack traversal runs
# as a single launch. The single kernel keeps its running head in the L2
# cache between phases; once one surface no longer fits alongside the layer
# inputs, its re-reads spill to HBM with a several-fold amplification and the
# per-layer kernels win despite their launch count.
_SINGLE_LAUNCH_MAX_SURFACE = 48 * 1024 * 1024


def _single_launch_traversal(z_all: torch.Tensor) -> bool:
    """Return whether the whole-stack kernel serves this shape well."""
    n_focus, n_edge, row = z_all.shape[1:]
    surface = n_focus * n_edge * row * z_all.element_size()
    return surface <= _SINGLE_LAUNCH_MAX_SURFACE


def _stack_weight_gradients(
    grad_out: Tensor,
    z_all: Tensor,
    u_final: Tensor,
    alpha: Tensor,
    state: _StackBackwardState,
    lmax: int,
    focus_dim: int,
    apply_alpha: bool,
    u0: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Contract the traversal surfaces into the stacked weight gradients.

    Every gated layer contributes ``u_l^T gz_l`` per block half and
    ``s_l^T gq_l`` for the gate projection; batching the layers into single
    cuBLAS calls keeps the launch count independent of the depth. The final
    identity layer contracts the stack input against the scaled output
    cotangent and occupies the last slot. When the exact stack input ``u0``
    is supplied, the bottom layer's block contractions are recomputed against
    it, replacing the recovered value's contribution.

    Parameters
    ----------
    grad_out : Tensor
        Edge-major output cotangent, with shape ``(E, F, ROW)``.
    z_all : Tensor
        Stacked pre-activations, with shape ``(NL, F, E, ROW)``.
    u_final : Tensor
        Input of the final identity layer, with shape ``(F, E, ROW)``.
    alpha : Tensor
        Focus competition weight, with shape ``(E, F)``.
    state : _StackBackwardState
        Surfaces retained by the traversal.
    lmax : int
        Maximum degree.
    focus_dim : int
        Per-focus channel width.
    apply_alpha : bool
        Whether the forward scaled its output by the competition weight.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Stacked gradients of ``(w0_all, w1_all, gw_all)``.
    """
    n_gated, n_focus, n_edge, row = z_all.shape
    m0 = (int(lmax) + 1) * int(focus_dim)
    m1 = row - m0
    dtype = grad_out.dtype
    scaled = grad_out * alpha.unsqueeze(-1).to(dtype) if apply_alpha else grad_out
    grad_final = scaled.permute(1, 0, 2).contiguous()

    u_flat = state.inputs.reshape(n_gated * n_focus, n_edge, row)
    gz_flat = state.grad_z.reshape(n_gated * n_focus, n_edge, row)
    gq_flat = state.grad_logit.reshape(n_gated * n_focus, n_edge, -1)
    z_flat = z_all.reshape(n_gated * n_focus, n_edge, row)

    gw0 = torch.empty(
        (n_gated + 1, n_focus, m0, m0), device=grad_out.device, dtype=dtype
    )
    gw1 = torch.empty(
        (n_gated + 1, n_focus, m1, m1), device=grad_out.device, dtype=dtype
    )
    torch.bmm(
        u_flat[:, :, :m0].transpose(1, 2),
        gz_flat[:, :, :m0],
        out=gw0[:n_gated].view(n_gated * n_focus, m0, m0),
    )
    torch.bmm(
        u_flat[:, :, m0:].transpose(1, 2),
        gz_flat[:, :, m0:],
        out=gw1[:n_gated].view(n_gated * n_focus, m1, m1),
    )
    torch.bmm(
        u_final[:, :, :m0].transpose(1, 2), grad_final[:, :, :m0], out=gw0[n_gated]
    )
    torch.bmm(
        u_final[:, :, m0:].transpose(1, 2), grad_final[:, :, m0:], out=gw1[n_gated]
    )
    if u0 is not None:
        torch.bmm(
            u0[:, :, :m0].transpose(1, 2),
            state.grad_z[0][:, :, :m0],
            out=gw0[0],
        )
        torch.bmm(
            u0[:, :, m0:].transpose(1, 2),
            state.grad_z[0][:, :, m0:],
            out=gw1[0],
        )
    ggw = torch.bmm(z_flat[:, :, : int(focus_dim)].transpose(1, 2), gq_flat).view(
        n_gated, n_focus, int(focus_dim), -1
    )
    return gw0, gw1, ggw


def _stack_train_traversal(
    grad_out: Tensor,
    z_all: Tensor,
    u_final: Tensor,
    alpha: Tensor,
    w0t_all: Tensor,
    w1t_all: Tensor,
    gw_all: Tensor,
    grad_z_upstream: Tensor | None,
    grad_u_upstream: Tensor | None,
    lmax: int,
    focus_dim: int,
    apply_alpha: bool,
    *,
    need_alpha: bool,
    keep: bool,
) -> tuple[Tensor, Tensor, _StackBackwardState]:
    """Run the whole-stack training backward as a single launch.

    Serves the narrow-channel regime (``Cf < GATE_BMM_MIN_FOCUS_DIM``), where
    the gate projection is a register dot. Returns the input and competition
    gradients together with the per-layer surfaces the weight gradients
    contract against; ``keep`` additionally retains the per-layer upstream
    gradients for the second order.
    """
    n_gated, n_focus, n_edge, row = z_all.shape
    device, dtype = grad_out.device, grad_out.dtype
    gate_width = lmax * focus_dim
    grad_u0 = torch.empty((n_focus, n_edge, row), device=device, dtype=dtype)
    grad_alpha = torch.empty((n_edge, n_focus), device=device, dtype=dtype)
    gz_all = torch.empty((n_gated, n_focus, n_edge, row), device=device, dtype=dtype)
    gq_all = torch.empty(
        (n_gated, n_focus, n_edge, gate_width), device=device, dtype=dtype
    )
    u_all = torch.empty((n_gated, n_focus, n_edge, row), device=device, dtype=dtype)
    up_all = (
        torch.empty((n_gated, n_focus, n_edge, row), device=device, dtype=dtype)
        if keep
        else gz_all
    )
    block_e, block_n, block_k, warps, stages = 32, 64, 64, 8, 2
    wrap_triton(_stack_train_traversal_kernel)[(triton.cdiv(n_edge, block_e), n_focus)](
        grad_out,
        z_all,
        u_final,
        alpha,
        w0t_all,
        w1t_all,
        gw_all,
        grad_z_upstream if grad_z_upstream is not None else grad_out,
        grad_u_upstream if grad_u_upstream is not None else grad_out,
        grad_u0,
        grad_alpha,
        gz_all,
        gq_all,
        u_all,
        up_all,
        n_edge,
        NL=n_gated,
        L=int(lmax),
        CF=int(focus_dim),
        APPLY_ALPHA=bool(apply_alpha),
        NEED_ALPHA=bool(apply_alpha and need_alpha),
        HAS_GZUP=grad_z_upstream is not None,
        HAS_GUUP=grad_u_upstream is not None,
        KEEP=bool(keep),
        BLOCK_E=block_e,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=warps,
        num_stages=stages,
    )
    return grad_u0, grad_alpha, _StackBackwardState(up_all, u_all, gz_all, gq_all)


def _point_backward_schedule(
    focus_dim: int, lmax: int, *, train: bool = False
) -> _PointBackwardSchedule:
    """Resolve the gate-projection and pointwise backward schedule.

    The training traversal recovers each layer's input and stores the
    gate-logit gradient inside the same kernel, a heavier register profile
    with its own swept table; ``train`` selects it. The fused-recompute win
    list applies to the inference profile only.
    """
    if train:
        # The training entries are swept with the gate sigmoids recomputed
        # inside the pointwise kernel (below the bmm regime), so the schedule
        # must launch the same variant.
        recompute_inside = focus_dim < GATE_BMM_MIN_FOCUS_DIM
        return (
            recompute_inside,
            point_train_config(focus_dim, lmax),
            recompute_config(focus_dim, lmax),
        )
    fused_config = point_recompute_config(focus_dim, lmax)
    return (
        fused_config is not None,
        fused_config or point_config(focus_dim, lmax),
        recompute_config(focus_dim, lmax),
    )


def _launch_stack_point_backward(
    grad: Tensor,
    z_all: Tensor,
    gw_all: Tensor,
    gwt_all: Tensor,
    sig: Tensor,
    grad_z: Tensor,
    grad_logit: Tensor,
    n_edge: int | torch.SymInt,
    layer: int,
    *,
    lmax: int,
    focus_dim: int,
    n_focus: int,
    use_bmm: bool,
    schedule: _PointBackwardSchedule,
    layer_output: Tensor | None = None,
    layer_input: Tensor | None = None,
    store_logit: bool = True,
) -> None:
    """Launch one gated layer's projection and pointwise backward.

    Passing ``layer_output`` and ``layer_input`` additionally recovers this
    layer's input from its output inside the same kernel, which is what the
    weight gradient contracts against.
    """
    recompute_sigmoid, point_cfg, recompute_cfg = schedule
    if not recompute_sigmoid:
        if use_bmm:
            torch.sigmoid(
                torch.bmm(z_all[layer, :, :, :focus_dim], gw_all[layer]).float(),
                out=sig,
            )
        else:
            block_m, warps, stages = recompute_cfg
            wrap_triton(_stack_recompute_kernel)[
                (triton.cdiv(n_edge, block_m), n_focus)
            ](
                z_all,
                gw_all,
                sig,
                n_edge,
                layer,
                L=lmax,
                CF=focus_dim,
                BLOCK_M=block_m,
                num_warps=warps,
                num_stages=stages,
            )
    block_m, warps, stages = point_cfg
    recover = layer_output is not None and layer_input is not None
    wrap_triton(_stack_point_bwd_kernel)[(triton.cdiv(n_edge, block_m), n_focus)](
        grad,
        z_all,
        sig,
        gwt_all,
        grad_z,
        grad_logit,
        layer_output if recover else grad,
        layer_input if recover else grad,
        n_edge,
        layer,
        L=lmax,
        CF=focus_dim,
        GLOGIT_OUT=use_bmm,
        GLOGIT_STORE=store_logit,
        RECOMPUTE_SIG=recompute_sigmoid,
        RECOVER_INPUT=recover,
        BLOCK_M=block_m,
        num_warps=warps,
        num_stages=stages,
    )


# ======================================================================
# Operator implementations (Triton on CUDA fp32, eager reference otherwise)
# ======================================================================
def _rotate_mix_impl(
    x: Tensor,
    src: Tensor,
    src_order: Tensor,
    src_rowptr: Tensor,
    wigner: Tensor,
    kc: Tensor,
    cb: Tensor,
    lmax: int,
    n_focus: int,
    rank: int,
) -> Tensor:
    # The source CSR view rides through the forward untouched so the autograd
    # context can hand it to the backward's segment reduction.
    del src_order, src_rowptr
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
    grad_wigner = wigner.new_zeros(wigner.shape)
    grad_kc = kc.new_empty(kc.shape)
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


def _gated_act_reference(
    z: Tensor,
    gw: Tensor,
    lmax: int,
    focus_dim: int,
) -> Tensor:
    """Eager ground truth for the standalone gated activation forward.

    Parameters
    ----------
    z : Tensor
        Pre-activation with shape ``(F, E, ROW)``.
    gw : Tensor
        Gate projection with shape ``(F, Cf, lmax * Cf)``.
    lmax : int
        Maximum degree.
    focus_dim : int
        Per-focus channel width.

    Returns
    -------
    Tensor
        Activated features with shape ``(F, E, ROW)``.
    """
    lmax = int(lmax)
    focus_dim = int(focus_dim)
    m0 = (lmax + 1) * focus_dim
    scalar = z[:, :, :focus_dim]
    sig = torch.sigmoid(torch.bmm(scalar.float(), gw.float())).to(z.dtype)
    return torch.cat(
        [
            scalar * torch.sigmoid(scalar.float()).to(z.dtype),
            z[:, :, focus_dim:m0] * sig,
            z[:, :, m0:] * sig.repeat(1, 1, 2),
        ],
        dim=-1,
    )


def _stack_point_bwd_reference(
    grad: Tensor,
    z: Tensor,
    gw: Tensor,
    layer_output: Tensor,
    lmax: int,
    focus_dim: int,
    fold_logit: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """Eager ground truth for one gated layer's pointwise backward.

    Differentiates ``act(z)`` of a single layer, where the scalar rows carry a
    SiLU and the remaining rows are scaled by a sigmoid gate driven by those
    same scalar rows. Being written in ATen, it is differentiable, which is what
    supplies the second order of the fused operator.

    Parameters
    ----------
    grad : Tensor
        Gradient of the layer output, with shape ``(F, E, ROW)``.
    z : Tensor
        Pre-activation of the layer, with shape ``(F, E, ROW)``.
    gw : Tensor
        Gate projection of the layer, with shape ``(F, Cf, lmax * Cf)``.
    lmax : int
        Maximum degree.
    focus_dim : int
        Per-focus channel width.
    layer_output : Tensor
        Output of the layer, with shape ``(F, E, ROW)``, from which the layer's
        input is recovered.
    fold_logit : bool
        Whether the gate-logit contraction back to the scalar rows is left to
        the caller. When False it is folded into the returned ``gz``.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        The pre-activation gradient ``(F, E, ROW)``, the gate-logit gradient
        ``(F, E, lmax * Cf)``, and the layer's input ``(F, E, ROW)``.
    """
    lmax = int(lmax)
    focus_dim = int(focus_dim)
    m0 = (lmax + 1) * focus_dim
    scalar = z[:, :, :focus_dim]
    sig = torch.sigmoid(torch.bmm(scalar, gw))
    sig2 = sig.repeat(1, 1, 2)
    silu_sig = torch.sigmoid(scalar)

    gz_scalar = grad[:, :, :focus_dim] * silu_sig * (1.0 + scalar * (1.0 - silu_sig))
    gz_gated0 = grad[:, :, focus_dim:m0] * sig
    gz_gated1 = grad[:, :, m0:] * sig2

    grad_sig = (grad[:, :, focus_dim:m0] * z[:, :, focus_dim:m0]) + (
        grad[:, :, m0:] * z[:, :, m0:]
    ).view(sig.shape[0], sig.shape[1], 2, -1).sum(2)
    grad_logit = grad_sig * sig * (1.0 - sig)
    if not fold_logit:
        gz_scalar = gz_scalar + torch.bmm(grad_logit, gw.transpose(1, 2))
    activation = torch.cat(
        [
            scalar * silu_sig,
            z[:, :, focus_dim:m0] * sig,
            z[:, :, m0:] * sig2,
        ],
        dim=-1,
    )
    return (
        torch.cat([gz_scalar, gz_gated0, gz_gated1], dim=-1),
        grad_logit,
        layer_output - activation,
    )


def _stack_point_bwd_impl(
    grad: Tensor,
    z: Tensor,
    gw: Tensor,
    gwt: Tensor,
    layer_output: Tensor,
    lmax: int,
    focus_dim: int,
    fold_logit: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """One gated layer's pointwise backward, fused into a single kernel.

    Also recovers the layer's input from ``layer_output`` through
    ``u_l = u_{l+1} - act(z_l)``; the activation is already in registers, so
    the input the weight gradient contracts against is free here.
    """
    if not _use_triton(grad):
        return _stack_point_bwd_reference(
            grad, z, gw, layer_output, int(lmax), int(focus_dim), bool(fold_logit)
        )
    lmax = int(lmax)
    focus_dim = int(focus_dim)
    n_focus, n_edge, row = grad.shape
    gate_width = lmax * focus_dim
    device, dtype = grad.device, grad.dtype
    grad_z = torch.empty((n_focus, n_edge, row), device=device, dtype=dtype)
    grad_logit = torch.empty(
        (n_focus, n_edge, gate_width), device=device, dtype=torch.float32
    )
    layer_input = torch.empty((n_focus, n_edge, row), device=device, dtype=dtype)
    if _has_no_edges(n_edge):
        return grad_z, grad_logit, layer_input
    sig = torch.empty((n_focus, n_edge, gate_width), device=device, dtype=torch.float32)
    _launch_stack_point_backward(
        grad,
        z.unsqueeze(0),
        gw.unsqueeze(0),
        gwt.unsqueeze(0),
        sig,
        grad_z,
        grad_logit,
        n_edge,
        0,
        lmax=lmax,
        focus_dim=focus_dim,
        n_focus=n_focus,
        use_bmm=bool(fold_logit),
        schedule=_point_backward_schedule(focus_dim, lmax, train=True),
        layer_output=layer_output,
        layer_input=layer_input,
    )
    return grad_z, grad_logit, layer_input


def _gated_act_use_bmm(focus_dim: int, lmax: int) -> bool:
    """Return whether the gate projection runs as a cuBLAS batched matmul.

    The register-dot form holds ``lmax`` tiles of width ``next_power_of_2(Cf)``
    per program; the measured crossover puts the 64-wide profile past its
    occupancy break-even from ``lmax = 4``, and at ``Cf >= 96`` the dot spills
    outright.  Past the boundary the projection and both logit contractions
    run as batched matmuls around pure pointwise kernel bodies.
    """
    return focus_dim >= GATE_BMM_MIN_FOCUS_DIM or (focus_dim >= 64 and lmax >= 4)


def _gated_act_impl(
    z: Tensor,
    gw: Tensor,
    gwt: Tensor,
    lmax: int,
    focus_dim: int,
) -> Tensor:
    """Standalone gated activation forward, one kernel per focus stream.

    The transposed gate projection is unused by the forward; it travels with
    the operator so the backward and the second order read it without a
    transposing copy of their own.
    """
    if not _use_triton(z):
        return _gated_act_reference(z, gw, int(lmax), int(focus_dim))
    lmax = int(lmax)
    focus_dim = int(focus_dim)
    n_focus, n_edge, row = z.shape
    out = torch.empty((n_focus, n_edge, row), device=z.device, dtype=z.dtype)
    if _has_no_edges(n_edge):
        return out
    use_bmm = _gated_act_use_bmm(focus_dim, lmax)
    if use_bmm:
        sig = torch.sigmoid(
            torch.bmm(z[:, :, :focus_dim], gw).float()
        ).contiguous()  # (F, E, L*Cf)
    else:
        sig = z.new_empty(0)
    block_m, warps, stages = gate_config(focus_dim, lmax)
    wrap_triton(_gated_act_fwd_kernel)[(triton.cdiv(n_edge, block_m), n_focus)](
        z,
        gw,
        sig,
        out,
        n_edge,
        L=lmax,
        CF=focus_dim,
        SIG_IN=use_bmm,
        BLOCK_M=block_m,
        num_warps=warps,
        num_stages=stages,
    )
    return out


def _gated_act_bwd_impl(
    grad: Tensor,
    z: Tensor,
    gw: Tensor,
    gwt: Tensor,
    lmax: int,
    focus_dim: int,
) -> tuple[Tensor, Tensor]:
    """Standalone gated activation backward, fused into a single kernel.

    Returns the pre-activation gradient and the gate-logit gradient the caller
    contracts against the scalar rows for the gate projection's weight
    gradient.  In the register-dot regime the gate-logit contraction back onto
    the scalar rows is folded into the kernel; in the wide-channel regime the
    kernel emits the gate-logit gradient and the contraction runs as a
    batched matmul here, mirroring the forward's projection split.
    """
    lmax = int(lmax)
    focus_dim = int(focus_dim)
    if not _use_triton(grad):
        grad_z, grad_logit, _ = _stack_point_bwd_reference(
            grad, z, gw, grad, lmax, focus_dim, fold_logit=False
        )
        return grad_z, grad_logit
    n_focus, n_edge, row = grad.shape
    gate_width = lmax * focus_dim
    device, dtype = grad.device, grad.dtype
    grad_z = torch.empty((n_focus, n_edge, row), device=device, dtype=dtype)
    grad_logit = torch.empty(
        (n_focus, n_edge, gate_width), device=device, dtype=torch.float32
    )
    if _has_no_edges(n_edge):
        return grad_z, grad_logit
    use_bmm = _gated_act_use_bmm(focus_dim, lmax)
    if use_bmm:
        # The batched-matmul regime keeps the kernel body pointwise: the
        # sigmoid surface comes from cuBLAS and the logit contraction runs
        # below.
        sig = torch.sigmoid(torch.bmm(z[:, :, :focus_dim], gw).float()).contiguous()
        _launch_stack_point_backward(
            grad,
            z.unsqueeze(0),
            gw.unsqueeze(0),
            gwt.unsqueeze(0),
            sig,
            grad_z,
            grad_logit,
            n_edge,
            0,
            lmax=lmax,
            focus_dim=focus_dim,
            n_focus=n_focus,
            use_bmm=True,
            schedule=(False, (64, 8, 2), recompute_config(focus_dim, lmax)),
        )
        grad_z[:, :, :focus_dim] += torch.bmm(grad_logit.to(dtype), gwt)
        return grad_z, grad_logit
    sig = torch.empty((n_focus, n_edge, gate_width), device=device, dtype=torch.float32)
    _launch_stack_point_backward(
        grad,
        z.unsqueeze(0),
        gw.unsqueeze(0),
        gwt.unsqueeze(0),
        sig,
        grad_z,
        grad_logit,
        n_edge,
        0,
        lmax=lmax,
        focus_dim=focus_dim,
        n_focus=n_focus,
        use_bmm=False,
        schedule=_point_backward_schedule(focus_dim, lmax, train=True),
    )
    return grad_z, grad_logit


def _mixing_stack_impl(
    u0: Tensor,
    alpha: Tensor,
    w0_all: Tensor,
    w1_all: Tensor,
    gw_all: Tensor,
    lmax: int,
    focus_dim: int,
    apply_alpha: bool,
) -> tuple[Tensor, Tensor, Tensor]:
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
        return x_local, z_all, u0

    m0_config, m1_config, _ = stack_fp32_configs(focus_dim, lmax)
    m0_bm, m0_bn, m0_bk, m0_warps, m0_stages = m0_config
    m1_bm, m1_bn, m1_bk, m1_warps, m1_stages = m1_config
    m0 = (lmax + 1) * focus_dim
    m1 = 2 * lmax * focus_dim
    gate_bm, gate_w, gate_s = gate_config(focus_dim, lmax)
    sig_by_bmm = focus_dim >= GATE_BMM_MIN_FOCUS_DIM
    m0_gate_config = stack_m0_gate_config(focus_dim, lmax)
    sig = torch.empty(
        (n_focus, n_edge, lmax * focus_dim), device=u0.device, dtype=torch.float32
    )

    u = u0
    for layer in range(n_gated):
        out = torch.empty_like(u)
        if m0_gate_config is not None:
            m0_gate_bm, m0_gate_bk, m0_gate_warps, m0_gate_stages = m0_gate_config
            wrap_triton(_stack_gemm_m0_gate_kernel)[
                (triton.cdiv(n_edge, m0_gate_bm), n_focus)
            ](
                u,
                w0_all,
                gw_all,
                out,
                z_all,
                sig,
                n_edge,
                layer,
                L=lmax,
                CF=focus_dim,
                BLOCK_M=m0_gate_bm,
                BLOCK_K=m0_gate_bk,
                num_warps=m0_gate_warps,
                num_stages=m0_gate_stages,
            )
        else:
            wrap_triton(_stack_gemm_m0_kernel)[
                (triton.cdiv(n_edge, m0_bm) * triton.cdiv(m0, m0_bn), n_focus)
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
                BLOCK_M=m0_bm,
                BLOCK_N=m0_bn,
                BLOCK_K=m0_bk,
                num_warps=m0_warps,
                num_stages=m0_stages,
            )
            if sig_by_bmm:
                # Wide-channel regime: sigmoid projection as a cuBLAS bmm on
                # the freshly written scalar rows of the pre-activation.
                torch.sigmoid(
                    torch.bmm(z_all[layer, :, :, :focus_dim], gw_all[layer]),
                    out=sig,
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
            (triton.cdiv(n_edge, m1_bm) * triton.cdiv(m1, m1_bn), n_focus)
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
            BLOCK_M=m1_bm,
            BLOCK_N=m1_bn,
            BLOCK_K=m1_bk,
            num_warps=m1_warps,
            num_stages=m1_stages,
        )
        u = out

    # ``u`` now holds the input of the final identity layer. Training needs it:
    # every gated layer's weight gradient contracts that layer's input against
    # its pre-activation gradient, and the backward recovers the inputs by
    # walking this one back through ``u_l = u_{l+1} - act(z_l)``, reusing the
    # saved pre-activations instead of storing one activation per layer.
    u_final = u

    # Final identity layer streams straight into the edge-major output layout.
    wrap_triton(_stack_gemm_m0_kernel)[
        (triton.cdiv(n_edge, m0_bm) * triton.cdiv(m0, m0_bn), n_focus)
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
        BLOCK_M=m0_bm,
        BLOCK_N=m0_bn,
        BLOCK_K=m0_bk,
        num_warps=m0_warps,
        num_stages=m0_stages,
    )
    wrap_triton(_stack_gemm_m1_kernel)[
        (triton.cdiv(n_edge, m1_bm) * triton.cdiv(m1, m1_bn), n_focus)
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
        BLOCK_M=m1_bm,
        BLOCK_N=m1_bn,
        BLOCK_K=m1_bk,
        num_warps=m1_warps,
        num_stages=m1_stages,
    )
    return x_local, z_all, u_final


class _StackBackwardState(NamedTuple):
    """Per-layer surfaces retained by a traversal for its own second order.

    All tensors stack the gated layers along the leading axis. ``upstream``
    holds the gradient entering each layer's pointwise backward (the
    linearization point of the second order), ``inputs`` the recovered layer
    inputs; ``grad_z`` and ``grad_logit`` the pre-activation and gate-logit
    gradients. The traversal's kernels write these slices directly, so
    retaining them costs no copy.
    """

    upstream: Tensor
    inputs: Tensor
    grad_z: Tensor
    grad_logit: Tensor


def _stack_backward_traversal(
    grad_out: Tensor,
    x_local: Tensor,
    z_all: Tensor,
    u_final: Tensor,
    alpha: Tensor,
    w0t_all: Tensor,
    w1t_all: Tensor,
    gw_all: Tensor,
    gwt_all: Tensor,
    grad_z_upstream: Tensor | None,
    grad_u_upstream: Tensor | None,
    lmax: int,
    focus_dim: int,
    apply_alpha: bool,
    *,
    with_weights: bool,
    keep: bool,
    need_alpha: bool = True,
    need_logit: bool = True,
    u0: Tensor | None = None,
) -> tuple[
    Tensor, Tensor, tuple[Tensor, Tensor, Tensor] | None, _StackBackwardState | None
]:
    """Walk the stack backwards once, in fused kernels end to end.

    The traversal always produces the input and competition-weight gradients.
    ``with_weights`` additionally contracts the per-layer weight gradients
    (recovering each layer's input inside the pointwise kernel), and ``keep``
    retains the per-layer surfaces the second order linearizes around.
    ``need_alpha`` and ``need_logit`` let a replay skip the competition
    gradient and the gate-logit store when no consumer exists. When the
    exact stack input ``u0`` is supplied, the bottom layer's block-weight
    gradients contract against it instead of the recovered value.

    The pre-activations and the final activation are outputs of the forward,
    so a differentiation of the whole graph may send gradients back through
    those channels as well; they join the traversal at the point where the
    forward produced the corresponding tensor.
    """
    n_gated, n_focus, n_edge, row = z_all.shape
    lmax = int(lmax)
    focus_dim = int(focus_dim)
    device, dtype = grad_out.device, grad_out.dtype

    _, _, bwd_config = stack_fp32_configs(focus_dim, lmax)
    block_m, block_n, block_k, warps, stages = bwd_config
    m0 = (lmax + 1) * focus_dim
    m1 = 2 * lmax * focus_dim
    n_tiles = triton.cdiv(m0, block_n) + triton.cdiv(m1, block_n)
    point_schedule = _point_backward_schedule(
        focus_dim, lmax, train=with_weights or keep
    )
    grad_alpha = torch.empty((n_edge, n_focus), device=device, dtype=dtype)

    # === Final layer: g = gz + gz @ W^T with gz = grad [* alpha] on the fly ===
    upstream_all = (
        torch.empty((n_gated, n_focus, n_edge, row), device=device, dtype=dtype)
        if keep
        else None
    )
    g_cur = (
        upstream_all[n_gated - 1]
        if keep and n_gated > 0
        else torch.empty((n_focus, n_edge, row), device=device, dtype=dtype)
    )
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
    if apply_alpha and need_alpha:
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
    if grad_u_upstream is not None:
        g_cur += grad_u_upstream

    weights: tuple[Tensor, Tensor, Tensor] | None = None
    if with_weights:
        # The weight gradients contract against the scaled cotangent in the
        # forward orientation ``z = u W``; the final identity layer occupies
        # the last slot.
        if apply_alpha:
            scaled = grad_out * alpha.unsqueeze(-1).to(dtype)
        else:
            scaled = grad_out
        grad_final = scaled.permute(1, 0, 2).contiguous()
        grad_w0_all = torch.empty(
            (n_gated + 1, n_focus, m0, m0), device=device, dtype=dtype
        )
        grad_w1_all = torch.empty(
            (n_gated + 1, n_focus, m1, m1), device=device, dtype=dtype
        )
        grad_gw_all = torch.empty(
            (n_gated, n_focus, focus_dim, lmax * focus_dim),
            device=device,
            dtype=dtype,
        )
        torch.bmm(
            u_final[:, :, :m0].transpose(1, 2),
            grad_final[:, :, :m0],
            out=grad_w0_all[n_gated],
        )
        torch.bmm(
            u_final[:, :, m0:].transpose(1, 2),
            grad_final[:, :, m0:],
            out=grad_w1_all[n_gated],
        )
        weights = (grad_w0_all, grad_w1_all, grad_gw_all)

    # === Gated layers in reverse ===
    # The per-layer pre-activation and gate-logit gradients are retained rather
    # than reused across layers: they are exactly the cotangents the weight
    # gradients contract against, and recomputing them later would cost a second
    # traversal of the stack.
    gate_width = lmax * focus_dim
    sig = torch.empty((n_focus, n_edge, gate_width), device=device, dtype=torch.float32)
    grad_z_all = torch.empty(
        (n_gated, n_focus, n_edge, row), device=device, dtype=dtype
    )
    use_bmm = focus_dim >= GATE_BMM_MIN_FOCUS_DIM
    store_logit = with_weights or use_bmm or (keep and need_logit)
    grad_logit_all = torch.empty(
        (n_gated if store_logit else 0, n_focus, n_edge, gate_width),
        device=device,
        dtype=dtype,
    )
    recover = with_weights or keep
    inputs_all = (
        torch.empty((n_gated, n_focus, n_edge, row), device=device, dtype=dtype)
        if keep
        else None
    )
    u_next = u_final
    for layer in range(n_gated - 1, -1, -1):
        gz = grad_z_all[layer]
        glogit = grad_logit_all[layer] if store_logit else sig
        if keep:
            u_layer = inputs_all[layer]
        elif recover:
            u_layer = torch.empty((n_focus, n_edge, row), device=device, dtype=dtype)
        else:
            u_layer = None
        _launch_stack_point_backward(
            g_cur,
            z_all,
            gw_all,
            gwt_all,
            sig,
            gz,
            glogit,
            n_edge,
            layer,
            lmax=lmax,
            focus_dim=focus_dim,
            n_focus=n_focus,
            use_bmm=use_bmm,
            schedule=point_schedule,
            layer_output=u_next if recover else None,
            layer_input=u_layer,
            store_logit=store_logit,
        )
        if use_bmm:
            # Gate-logit contraction back to the scalar rows via cuBLAS.
            gz[:, :, :focus_dim] += torch.bmm(glogit.to(gz.dtype), gwt_all[layer])

        if grad_z_upstream is not None:
            gz += grad_z_upstream[layer]
        if with_weights:
            u_in = u0 if (layer == 0 and u0 is not None) else u_layer
            torch.bmm(
                u_in[:, :, :m0].transpose(1, 2),
                gz[:, :, :m0],
                out=grad_w0_all[layer],
            )
            torch.bmm(
                u_in[:, :, m0:].transpose(1, 2),
                gz[:, :, m0:],
                out=grad_w1_all[layer],
            )
            torch.bmm(
                z_all[layer][:, :, :focus_dim].transpose(1, 2),
                glogit,
                out=grad_gw_all[layer],
            )
        g_next = (
            upstream_all[layer - 1]
            if keep and layer > 0
            else torch.empty((n_focus, n_edge, row), device=device, dtype=dtype)
        )
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
        if recover:
            u_next = u_layer
        g_cur = g_next
    state = None
    if keep:
        state = _StackBackwardState(
            upstream_all, inputs_all, grad_z_all, grad_logit_all
        )
    return g_cur, grad_alpha, weights, state


def _mixing_stack_bwd_impl(
    grad_out: Tensor,
    x_local: Tensor,
    z_all: Tensor,
    u_final: Tensor,
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
            u_final,
            alpha,
            w0t_all,
            w1t_all,
            gw_all,
            gwt_all,
            None,
            None,
            lmax,
            focus_dim,
            apply_alpha,
        )[:2]
    n_focus, n_edge, row = z_all.shape[1:]
    if _has_no_edges(n_edge):
        return (
            torch.empty(
                (n_focus, n_edge, row), device=grad_out.device, dtype=grad_out.dtype
            ),
            torch.empty(
                (n_edge, n_focus), device=grad_out.device, dtype=grad_out.dtype
            ),
        )
    grad_u0, grad_alpha, _, _ = _stack_backward_traversal(
        grad_out,
        x_local,
        z_all,
        u_final,
        alpha,
        w0t_all,
        w1t_all,
        gw_all,
        gwt_all,
        None,
        None,
        lmax,
        focus_dim,
        apply_alpha,
        with_weights=False,
        keep=False,
    )
    return grad_u0, grad_alpha


def _mixing_stack_train_bwd_impl(
    grad_out: Tensor,
    x_local: Tensor,
    z_all: Tensor,
    u_final: Tensor,
    alpha: Tensor,
    w0t_all: Tensor,
    w1t_all: Tensor,
    gw_all: Tensor,
    gwt_all: Tensor,
    u0: Tensor | None,
    grad_z_upstream: Tensor | None,
    grad_u_upstream: Tensor | None,
    lmax: int,
    focus_dim: int,
    apply_alpha: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Stack backward with parameter gradients, for the training path.

    Identical traversal to the inference backward, additionally contracting
    the per-layer block-weight and gate-projection gradients against the
    layer inputs, and accepting the gradients a differentiation of the whole
    graph sends back through the pre-activation and final-activation outputs.
    Kept separate from the inference operator so that a frozen model never
    pays for gradients it discards.

    Layer inputs above the bottom exist only as values recovered from the
    forward output, whose error is the accumulated forward rounding. The
    bottom layer's input is the operator operand ``u0`` itself, so its weight
    gradients contract against the exact value.

    The second order replays this traversal to recover its linearization
    points: retaining them across the force graph would hold four stacked
    edge-size buffers per convolution alive between the two differentiations,
    which costs more in memory traffic than the replay costs in compute.
    """
    if not _use_triton(grad_out):
        return _mixing_stack_backward_reference(
            grad_out,
            x_local,
            z_all,
            u_final,
            alpha,
            w0t_all,
            w1t_all,
            gw_all,
            gwt_all,
            grad_z_upstream,
            grad_u_upstream,
            lmax,
            focus_dim,
            apply_alpha,
        )[:5]
    n_gated, n_focus, n_edge, row = z_all.shape
    lmax = int(lmax)
    focus_dim = int(focus_dim)
    device, dtype = grad_out.device, grad_out.dtype
    if _has_no_edges(n_edge):
        m0 = (lmax + 1) * focus_dim
        m1 = 2 * lmax * focus_dim
        return (
            torch.empty((n_focus, n_edge, row), device=device, dtype=dtype),
            torch.empty((n_edge, n_focus), device=device, dtype=dtype),
            torch.zeros((n_gated + 1, n_focus, m0, m0), device=device, dtype=dtype),
            torch.zeros((n_gated + 1, n_focus, m1, m1), device=device, dtype=dtype),
            torch.zeros(
                (n_gated, n_focus, focus_dim, lmax * focus_dim),
                device=device,
                dtype=dtype,
            ),
        )
    if focus_dim < GATE_BMM_MIN_FOCUS_DIM and _single_launch_traversal(z_all):
        grad_u0, grad_alpha, state = _stack_train_traversal(
            grad_out,
            z_all,
            u_final,
            alpha,
            w0t_all,
            w1t_all,
            gw_all,
            grad_z_upstream,
            grad_u_upstream,
            lmax,
            focus_dim,
            apply_alpha,
            need_alpha=True,
            keep=False,
        )
        gw0, gw1, ggw = _stack_weight_gradients(
            grad_out, z_all, u_final, alpha, state, lmax, focus_dim, apply_alpha, u0
        )
        return grad_u0, grad_alpha, gw0, gw1, ggw
    grad_u0, grad_alpha, weights, _ = _stack_backward_traversal(
        grad_out,
        x_local,
        z_all,
        u_final,
        alpha,
        w0t_all,
        w1t_all,
        gw_all,
        gwt_all,
        grad_z_upstream,
        grad_u_upstream,
        lmax,
        focus_dim,
        apply_alpha,
        with_weights=True,
        keep=False,
        u0=u0,
    )
    return grad_u0, grad_alpha, *weights


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
# Atomic rather than inlined: letting the compiler inline this traversal has
# been observed to mis-size its inter-kernel buffers under dynamic shapes when
# several differently shaped convolutions share one graph.
_mixing_stack_train_bwd_op = torch.library.custom_op(
    "sezm_triton::so2_mixing_stack_train_bwd",
    _mixing_stack_train_bwd_impl,
    mutates_args=(),
)
_stack_point_bwd_op = torch.library.triton_op(
    "sezm_triton::so2_stack_point_bwd", mutates_args=()
)(_stack_point_bwd_impl)
_gated_act_op = torch.library.triton_op("sezm_triton::gated_act", mutates_args=())(
    _gated_act_impl
)
_gated_act_bwd_op = torch.library.triton_op(
    "sezm_triton::gated_act_bwd", mutates_args=()
)(_gated_act_bwd_impl)


@_rotate_mix_op.register_fake
def _(x, src, src_order, src_rowptr, wigner, kc, cb, lmax, n_focus, rank):
    focus_dim = x.shape[2] // n_focus
    return x.new_empty((n_focus, src.shape[0], (3 * lmax + 1) * focus_dim))


@_rotate_mix_bwd_op.register_fake
def _(grad_u, x, src, wigner, kc, cb, lmax, n_focus, rank):
    # Contiguous outputs regardless of the operand layouts, matching every
    # implementation branch.
    return (
        x.new_empty((src.shape[0], (lmax + 1) ** 2, x.shape[2])),
        wigner.new_empty(wigner.shape),
        kc.new_empty(kc.shape),
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
        u0.new_empty((n_focus, n_edge, row)),
    )


@_mixing_stack_bwd_op.register_fake
def _(
    grad_out,
    x_local,
    z_all,
    u_final,
    alpha,
    w0t_all,
    w1t_all,
    gw_all,
    gwt_all,
    lmax,
    focus_dim,
    apply_alpha,
):
    n_focus, n_edge, row = z_all.shape[1:]
    return (
        z_all.new_empty((n_focus, n_edge, row)),
        z_all.new_empty((n_edge, n_focus)),
    )


@_mixing_stack_train_bwd_op.register_fake
def _(
    grad_out,
    x_local,
    z_all,
    u_final,
    alpha,
    w0t_all,
    w1t_all,
    gw_all,
    gwt_all,
    u0,
    grad_z_upstream,
    grad_u_upstream,
    lmax,
    focus_dim,
    apply_alpha,
):
    n_gated, n_focus, n_edge, row = z_all.shape
    m0 = (lmax + 1) * focus_dim
    m1 = 2 * lmax * focus_dim
    return (
        z_all.new_empty((n_focus, n_edge, row)),
        z_all.new_empty((n_edge, n_focus)),
        z_all.new_empty((n_gated + 1, n_focus, m0, m0)),
        z_all.new_empty((n_gated + 1, n_focus, m1, m1)),
        z_all.new_empty((n_gated, n_focus, focus_dim, lmax * focus_dim)),
    )


def _segment_broadcast_impl(grad_out: Tensor, order: Tensor, row_ptr: Tensor) -> Tensor:
    """Scatter a per-segment value back onto the rows of that segment.

    The adjoint of :func:`_segment_sum_impl`: every row receives the value of
    the segment it belongs to. ``order`` is a permutation of the rows grouped by
    segment, so the segment of each row is recovered by expanding the CSR
    offsets and undoing the permutation.

    Registered as an operator so the compiler treats it as atomic: its interior
    indexing has no Inductor lowering, and appearing inline in a traced
    backward would fail the whole graph over to eager execution.

    Parameters
    ----------
    grad_out : Tensor
        Per-segment gradient with shape ``(n_seg, ...)``.
    order : Tensor
        Row indices grouped by segment, with shape ``(n_rows,)``.
    row_ptr : Tensor
        Segment offsets into ``order``, with shape ``(n_seg + 1,)``.

    Returns
    -------
    Tensor
        Per-row gradient with shape ``(n_rows, ...)``.
    """
    n_seg = row_ptr.shape[0] - 1
    counts = row_ptr[1:] - row_ptr[:-1]
    segment_of_sorted = torch.repeat_interleave(
        torch.arange(n_seg, device=grad_out.device, dtype=order.dtype), counts
    )
    rows = grad_out.index_select(0, segment_of_sorted)
    out = grad_out.new_zeros((order.shape[0], *grad_out.shape[1:]))
    return out.index_copy(0, order, rows)


_segment_broadcast_op = torch.library.custom_op(
    "sezm_triton::segment_broadcast", _segment_broadcast_impl, mutates_args=()
)


@_segment_broadcast_op.register_fake
def _(grad_out, order, row_ptr):
    return grad_out.new_empty((order.shape[0], *grad_out.shape[1:]))


def _segment_broadcast_setup_context(ctx, inputs, output):
    _, order, row_ptr = inputs
    ctx.save_for_backward(order, row_ptr)


def _segment_broadcast_backward(ctx, grad):
    order, row_ptr = ctx.saved_tensors
    return _segment_sum_op(grad.contiguous(), order, row_ptr), None, None


_segment_broadcast_op.register_autograd(
    _segment_broadcast_backward, setup_context=_segment_broadcast_setup_context
)


def rotate_mix_basis_grad(
    grad_u: Tensor,
    x: Tensor,
    src: Tensor,
    wigner: Tensor,
    kc: Tensor,
    cb: Tensor,
    lmax: int,
    n_focus: int,
    rank: int,
) -> Tensor:
    """Contract the degree kernel and the rotated feature against the cotangent.

    The operator is linear in the channel basis, so its gradient
    ``sum_{e,i,o} K[e,i,o,r] x_local[e,i,c] g[e,o,c]`` does not involve ``cb``.
    The rotated feature is not an output of the fused operator, so it is
    recomputed here through the same rotation kernel the forward uses, which is
    cheaper than widening the operator to carry an (E, reduced, C) activation
    across the autograd boundary.

    Parameters
    ----------
    grad_u : Tensor
        Upstream gradient in the focus-major layout ``(F, E, ROW)``.
    x : Tensor
        Node features with shape ``(N, D, C_wide)``.
    src : Tensor
        Source node index of each edge, with shape ``(E,)``.
    wigner : Tensor
        Wigner-D matrices with shape ``(E, D, D)``.
    kc : Tensor
        Projected degree kernel, flattened per edge.
    cb : Tensor
        Per-rank channel basis with shape ``(R, C_wide)``.
    lmax : int
        Maximum degree.
    n_focus : int
        Number of focus streams.
    rank : int
        Channel-basis rank; the mixer is basis-free when zero.

    Returns
    -------
    Tensor
        Gradient of the channel basis, shaped like ``cb``.
    """
    x_local = _block_to_local_op(x, src, wigner, int(lmax))  # (E, reduced, C_wide)
    n_edge, reduced, c_wide = x_local.shape
    focus_dim = c_wide // int(n_focus)
    n_deg = int(lmax) + 1
    grad_y = (
        grad_u.view(int(n_focus), n_edge, reduced, focus_dim)
        .permute(1, 2, 0, 3)
        .reshape(n_edge, reduced, c_wide)
    )
    kernel_flat = kc.view(n_edge, -1, int(rank))
    kernel_m0 = kernel_flat[:, : n_deg * n_deg].view(n_edge, n_deg, n_deg, int(rank))
    kernel_m1 = kernel_flat[:, n_deg * n_deg :].view(
        n_edge, int(lmax), int(lmax), int(rank)
    )
    blocks = (
        (kernel_m0, 0, n_deg),
        (kernel_m1, n_deg, int(lmax)),
        (kernel_m1, n_deg + int(lmax), int(lmax)),
    )
    grad_basis: Tensor | None = None
    for kernel, start, count in blocks:
        weighted = torch.einsum(
            "eior,eoc->reic", kernel, grad_y[:, start : start + count]
        )
        term = (weighted * x_local[:, start : start + count].unsqueeze(0)).sum(
            dim=(1, 2), dtype=torch.float32
        )
        grad_basis = term if grad_basis is None else grad_basis + term
    return grad_basis.to(cb.dtype).view_as(cb)


def _rotate_mix_setup_context(ctx, inputs, output):
    x, src, src_order, src_rowptr, wigner, kc, cb, lmax, n_focus, rank = inputs
    ctx.save_for_backward(x, src, src_order, src_rowptr, wigner, kc, cb)
    ctx.lmax = lmax
    ctx.n_focus = n_focus
    ctx.rank = rank


def _rotate_mix_backward(ctx, grad_u):
    x, src, src_order, src_rowptr, wigner, kc, cb = ctx.saved_tensors
    grad_u = grad_u.contiguous()
    grad_x_edge, grad_wigner, grad_kc = _rotate_mix_bwd_op(
        grad_u, x, src, wigner, kc, cb, ctx.lmax, ctx.n_focus, ctx.rank
    )
    # Contention-free segmented reduction of the per-edge node gradient through
    # the source CSR view the step builds once.
    grad_x = _segment_sum_op(grad_x_edge, src_order, src_rowptr)
    grad_cb = (
        rotate_mix_basis_grad(
            grad_u, x, src, wigner, kc, cb, ctx.lmax, ctx.n_focus, ctx.rank
        )
        if int(ctx.rank) > 0 and ctx.needs_input_grad[6]
        else None
    )
    return grad_x, None, None, None, grad_wigner, grad_kc, grad_cb, None, None, None


def _rotate_mix_bwd_setup_context(ctx, inputs, output):
    grad_u, x, src, wigner, kc, cb, lmax, n_focus, rank = inputs
    ctx.save_for_backward(grad_u, x, src, wigner, kc, cb)
    ctx.lmax = lmax
    ctx.n_focus = n_focus
    ctx.rank = rank


def _rotate_mix_bwd_backward(ctx, grad_grad_x, grad_grad_wigner, grad_grad_kc):
    """Second order of the fused rotate-and-mix.

    The operator is quadrilinear in ``(x, wigner, kc, cb)`` and emits the first
    three gradients, so the differentiated scalar is the sum of three adjoint
    terms, each of which substitutes exactly one cotangent into the forward.
    Every remaining derivative is one existing launch with one operand replaced;
    substituting more than one at a time would introduce cross terms.

    The ``x`` gradient is per-edge while ``x`` itself is per-node, so wherever
    that cotangent re-enters the operator the gather is made the identity and
    each edge stands in for its own node.
    """
    grad_u, x, src, wigner, kc, cb = ctx.saved_tensors
    lmax, n_focus, rank = ctx.lmax, ctx.n_focus, ctx.rank
    h_x, h_wigner, h_kc = grad_grad_x, grad_grad_wigner, grad_grad_kc
    if h_x is None and h_wigner is None and h_kc is None:
        return (None,) * 9

    edge_src = torch.arange(src.shape[0], device=src.device, dtype=src.dtype)

    def forward(x_arg: Tensor, src_arg: Tensor, w_arg: Tensor, k_arg: Tensor) -> Tensor:
        return _rotate_mix_op(
            x_arg, src_arg, edge_src, edge_src, w_arg, k_arg, cb, lmax, n_focus, rank
        )

    def backward(
        x_arg: Tensor, src_arg: Tensor, w_arg: Tensor, k_arg: Tensor
    ) -> tuple[Tensor, ...]:
        return _rotate_mix_bwd_op(
            grad_u, x_arg, src_arg, w_arg, k_arg, cb, lmax, n_focus, rank
        )

    def basis_grad(
        x_arg: Tensor, src_arg: Tensor, w_arg: Tensor, k_arg: Tensor
    ) -> Tensor:
        return rotate_mix_basis_grad(
            grad_u, x_arg, src_arg, w_arg, k_arg, cb, lmax, n_focus, rank
        )

    grad_grad_u: Tensor | None = None
    grad_x_edge: Tensor | None = None
    grad_wigner: Tensor | None = None
    grad_kc: Tensor | None = None
    grad_cb: Tensor | None = None
    wants_basis = int(rank) > 0 and ctx.needs_input_grad[5]
    needs_grad_u = ctx.needs_input_grad[0]

    if h_x is not None:
        if needs_grad_u:
            grad_grad_u = forward(h_x, edge_src, wigner, kc)
        _, term_wigner, term_kc = backward(h_x, edge_src, wigner, kc)
        grad_wigner = accumulate(grad_wigner, term_wigner)
        grad_kc = accumulate(grad_kc, term_kc)
        if wants_basis:
            grad_cb = accumulate(grad_cb, basis_grad(h_x, edge_src, wigner, kc))
    if h_wigner is not None:
        if needs_grad_u:
            grad_grad_u = accumulate(grad_grad_u, forward(x, src, h_wigner, kc))
        term_x, _, term_kc = backward(x, src, h_wigner, kc)
        grad_x_edge = accumulate(grad_x_edge, term_x)
        grad_kc = accumulate(grad_kc, term_kc)
        if wants_basis:
            grad_cb = accumulate(grad_cb, basis_grad(x, src, h_wigner, kc))
    if h_kc is not None:
        if needs_grad_u:
            grad_grad_u = accumulate(grad_grad_u, forward(x, src, wigner, h_kc))
        term_x, term_wigner, _ = backward(x, src, wigner, h_kc)
        grad_x_edge = accumulate(grad_x_edge, term_x)
        grad_wigner = accumulate(grad_wigner, term_wigner)
        if wants_basis:
            grad_cb = accumulate(grad_cb, basis_grad(x, src, wigner, h_kc))

    # The operator emits a per-edge ``x`` gradient that its caller reduces onto
    # nodes, so the second-order term inherits the same pending reduction: the
    # gradient of the per-node input is the gather adjoint of the accumulated
    # per-edge terms.
    grad_x = (
        None
        if grad_x_edge is None
        else x.new_zeros(x.shape).index_add(0, src, grad_x_edge)
    )

    # inputs: grad_u, x, src, wigner, kc, cb, lmax, n_focus, rank
    return (
        grad_grad_u,
        grad_x,
        None,
        grad_wigner,
        grad_kc,
        grad_cb,
        None,
        None,
        None,
    )


def _segment_sum_setup_context(ctx, inputs, output):
    rows, order, row_ptr = inputs
    ctx.save_for_backward(order, row_ptr)


def _segment_sum_backward(ctx, grad_out):
    order, row_ptr = ctx.saved_tensors
    return _segment_broadcast_op(grad_out.contiguous(), order, row_ptr), None, None


_rotate_mix_op.register_autograd(
    _rotate_mix_backward, setup_context=_rotate_mix_setup_context
)
_rotate_mix_bwd_op.register_autograd(
    _rotate_mix_bwd_backward, setup_context=_rotate_mix_bwd_setup_context
)
_segment_sum_op.register_autograd(
    _segment_sum_backward, setup_context=_segment_sum_setup_context
)

# Under AMP the activations arrive in bfloat16 while the packed weights, the
# Wigner-D buffer and the channel basis are still float32; these rules align
# every floating-point input to the training dtype exactly as the built-in
# matmuls do, and are inert outside an autocast region.
_rotate_mix_op.register_autocast("cuda", torch.bfloat16)
_segment_sum_op.register_autocast("cuda", torch.bfloat16)
_gated_act_op.register_autocast("cuda", torch.bfloat16)
_gated_act_bwd_op.register_autocast("cuda", torch.bfloat16)


def fused_gated_activation(
    z: Tensor,
    gw: Tensor,
    gwt: Tensor,
    lmax: int,
    focus_dim: int,
) -> Tensor:
    """Apply the gated SO(2) activation of one layer as a fused operator.

    The scalar (``l = 0``) rows pass through SiLU and drive one sigmoid gate
    per degree; each gate scales the three value rows that share it.  Forward,
    backward and second order each run as a single kernel per focus stream, so
    a force-loss training step traverses the activation without expanding it
    into per-operation elementwise kernels.

    Parameters
    ----------
    z : Tensor
        Pre-activation in the focus-major m-major layout, with shape
        ``(F, E, ROW)`` where ``ROW = (3 * lmax + 1) * Cf``.
    gw : Tensor
        Gate projection with shape ``(F, Cf, lmax * Cf)``, contiguous.
    gwt : Tensor
        Transposed gate projection with shape ``(F, lmax * Cf, Cf)``,
        contiguous.  Carried alongside ``gw`` so the backward reads it
        without a transposing copy.
    lmax : int
        Maximum degree.
    focus_dim : int
        Per-focus channel width ``Cf``.

    Returns
    -------
    Tensor
        Activated features with shape ``(F, E, ROW)``.
    """
    return _gated_act_op(z, gw, gwt, int(lmax), int(focus_dim))


def mixing_stack_layer_inputs(
    u_final: Tensor, z_all: Tensor, gw_all: Tensor, lmax: int, focus_dim: int
) -> Tensor:
    """Recover the input of every gated layer from the final activation.

    Each gated layer is a residual update ``u_{l+1} = u_l + act(z_l)`` whose
    activation depends only on the saved pre-activation, so the inputs are
    recovered by walking the residual backwards. This trades one pointwise pass
    per layer for not storing an extra ``(F, E, ROW)`` activation per layer in
    the forward.

    Parameters
    ----------
    u_final : Tensor
        Input of the final identity layer, with shape ``(F, E, ROW)``.
    z_all : Tensor
        Stacked gated-layer pre-activations, with shape ``(NL, F, E, ROW)``.
    gw_all : Tensor
        Stacked gate projections, with shape ``(NL, F, Cf, lmax * Cf)``.
    lmax : int
        Maximum degree.
    focus_dim : int
        Per-focus channel width.

    Returns
    -------
    Tensor
        Per-layer inputs stacked as ``(NL, F, E, ROW)``.
    """
    n_gated = z_all.shape[0]
    m0 = (int(lmax) + 1) * int(focus_dim)
    inputs = []
    u = u_final
    for layer in range(n_gated - 1, -1, -1):
        z = z_all[layer]
        scalar = z[:, :, : int(focus_dim)]
        sig = torch.sigmoid(torch.bmm(scalar, gw_all[layer]))
        act = torch.cat(
            [
                scalar * torch.sigmoid(scalar),
                z[:, :, int(focus_dim) : m0] * sig,
                z[:, :, m0:] * sig.repeat(1, 1, 2),
            ],
            dim=-1,
        )
        u = u - act
        inputs.append(u)
    return torch.stack(inputs[::-1])


def mixing_stack_weight_grads(
    layer_inputs: Tensor,
    grad_z_all: Tensor,
    grad_logit_all: Tensor,
    z_all: Tensor,
    u_final: Tensor,
    grad_final: Tensor,
    lmax: int,
    focus_dim: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Contract each layer's input against its pre-activation gradient.

    Every weight in the stack enters through a GEMM, so its gradient is the
    outer product of that GEMM's input with its output cotangent, reduced over
    the edge axis. cuBLAS handles that reduction well and an ATen expression is
    differentiable, which the second-order path needs.

    Parameters
    ----------
    layer_inputs : Tensor
        Per-gated-layer inputs, with shape ``(NL, F, E, ROW)``.
    grad_z_all : Tensor
        Per-gated-layer pre-activation gradients, with shape ``(NL, F, E, ROW)``.
    grad_logit_all : Tensor
        Per-gated-layer gate-logit gradients, with shape ``(NL, F, E, lmax*Cf)``.
    z_all : Tensor
        Stacked pre-activations, with shape ``(NL, F, E, ROW)``.
    u_final : Tensor
        Input of the final identity layer, with shape ``(F, E, ROW)``.
    grad_final : Tensor
        Output cotangent of the final identity layer, focus-major
        ``(F, E, ROW)`` and already scaled by the competition weight.
    lmax : int
        Maximum degree.
    focus_dim : int
        Per-focus channel width.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Gradients of ``(w0_all, w1_all, gw_all)``.
    """
    focus_dim = int(focus_dim)
    m0 = (int(lmax) + 1) * focus_dim
    grad_w0 = []
    grad_w1 = []
    grad_gw = []
    for layer in range(layer_inputs.shape[0]):
        u = layer_inputs[layer]
        gz = grad_z_all[layer]
        grad_w0.append(torch.bmm(u[:, :, :m0].transpose(1, 2), gz[:, :, :m0]))
        grad_w1.append(torch.bmm(u[:, :, m0:].transpose(1, 2), gz[:, :, m0:]))
        grad_gw.append(
            torch.bmm(
                z_all[layer][:, :, :focus_dim].transpose(1, 2),
                grad_logit_all[layer].to(gz.dtype),
            )
        )
    grad_w0.append(torch.bmm(u_final[:, :, :m0].transpose(1, 2), grad_final[:, :, :m0]))
    grad_w1.append(torch.bmm(u_final[:, :, m0:].transpose(1, 2), grad_final[:, :, m0:]))
    return torch.stack(grad_w0), torch.stack(grad_w1), torch.stack(grad_gw)


@_stack_point_bwd_op.register_fake
def _(grad, z, gw, gwt, layer_output, lmax, focus_dim, fold_logit):
    n_focus, n_edge, row = grad.shape
    return (
        torch.empty_like(grad),
        grad.new_empty((n_focus, n_edge, lmax * focus_dim), dtype=torch.float32),
        torch.empty_like(grad),
    )


@_gated_act_op.register_fake
def _(z, gw, gwt, lmax, focus_dim):
    return torch.empty_like(z)


@_gated_act_bwd_op.register_fake
def _(grad, z, gw, gwt, lmax, focus_dim):
    n_focus, n_edge, row = grad.shape
    return (
        torch.empty_like(grad),
        grad.new_empty((n_focus, n_edge, lmax * focus_dim), dtype=torch.float32),
    )


def _gated_act_setup_context(ctx, inputs, output):
    z, gw, gwt, lmax, focus_dim = inputs
    ctx.save_for_backward(z, gw, gwt)
    ctx.lmax = lmax
    ctx.focus_dim = focus_dim


def _gated_act_backward(ctx, grad):
    """First order of the standalone gated activation."""
    z, gw, gwt = ctx.saved_tensors
    lmax, focus_dim = int(ctx.lmax), int(ctx.focus_dim)
    grad_z, grad_logit = _gated_act_bwd_op(grad, z, gw, gwt, lmax, focus_dim)
    # The gate weight reduces the whole edge axis, which cuBLAS handles well;
    # expressed in ATen it stays differentiable for the second order.
    grad_gw = torch.bmm(z[:, :, :focus_dim].transpose(1, 2), grad_logit.to(z.dtype))
    return grad_z, grad_gw, None, None, None


_gated_act_op.register_autograd(
    _gated_act_backward, setup_context=_gated_act_setup_context
)


def _gated_act_bwd_setup_context(ctx, inputs, output):
    grad, z, gw, gwt, lmax, focus_dim = inputs
    ctx.save_for_backward(grad, z, gw, gwt)
    ctx.lmax = lmax
    ctx.focus_dim = focus_dim


def _gated_act_bwd_backward(ctx, grad_grad_z, grad_grad_logit):
    """Second order of the standalone gated activation.

    In the wide-channel regime the fused second-order kernel's register dots
    spill, so the elementwise body runs as a CUDA kernel with the projection
    and both scalar contractions expressed as batched matmuls -- the same
    regime split the forward and first order apply.  Without the CUDA library
    the ATen expression serves instead, lowering to compiler-fused pointwise
    kernels around the same contractions.
    """
    grad, z, gw, gwt = ctx.saved_tensors
    lmax, focus_dim = int(ctx.lmax), int(ctx.focus_dim)
    h_logit = grad_grad_logit.to(z.dtype) if grad_grad_logit is not None else None
    if _gated_act_use_bmm(focus_dim, lmax):
        grad_wrt_grad, grad_wrt_z, grad_wrt_gw = (
            gated_activation_second_order_reference(
                grad_grad_z,
                h_logit,
                grad,
                z,
                gw,
                lmax,
                focus_dim,
                fold_logit=False,
            )
        )
    else:
        grad_wrt_grad, grad_wrt_z, grad_wrt_gw = gated_activation_second_order(
            grad_grad_z,
            h_logit,
            grad,
            z,
            gw,
            gwt,
            lmax,
            focus_dim,
            fold_logit=False,
        )
    return grad_wrt_grad, grad_wrt_z, grad_wrt_gw, None, None, None


_gated_act_bwd_op.register_autograd(
    _gated_act_bwd_backward, setup_context=_gated_act_bwd_setup_context
)


def _stack_point_bwd_setup_context(ctx, inputs, output):
    grad, z, gw, gwt, layer_output, lmax, focus_dim, fold_logit = inputs
    ctx.save_for_backward(grad, z, gw)
    ctx.lmax = lmax
    ctx.focus_dim = focus_dim
    ctx.fold_logit = fold_logit


def _stack_point_bwd_backward(ctx, grad_grad_z, grad_grad_logit, grad_grad_input):
    """Second order of the gated layer's pointwise backward."""
    grad, z, gw = ctx.saved_tensors
    focus_dim, lmax = int(ctx.focus_dim), int(ctx.lmax)
    grad_wrt_grad, grad_wrt_z, grad_wrt_gw = gated_activation_second_order(
        grad_grad_z,
        grad_grad_logit.to(z.dtype) if grad_grad_logit is not None else None,
        grad,
        z,
        gw,
        gw.transpose(1, 2).contiguous(),
        lmax,
        focus_dim,
        ctx.fold_logit,
    )
    grad_wrt_output = None
    if grad_grad_input is not None:
        # The recovered input is ``u_{l+1} - act(z)``, so its cotangent passes
        # unchanged to the layer output and enters the pre-activation through
        # the activation's own vector-Jacobian product, which is exactly what
        # the first-order operator computes.
        grad_wrt_output = grad_grad_input
        gz_act, glogit_act, _ = _stack_point_bwd_op(
            grad_grad_input,
            z,
            gw,
            gw.transpose(1, 2).contiguous(),
            torch.zeros_like(grad_grad_input),
            lmax,
            focus_dim,
            False,
        )
        grad_wrt_z = grad_wrt_z - gz_act
        grad_wrt_gw = grad_wrt_gw - torch.bmm(
            z[:, :, :focus_dim].transpose(1, 2), glogit_act.to(z.dtype)
        )
    # inputs: grad, z, gw, gwt, layer_output, lmax, focus_dim, fold_logit
    return (
        grad_wrt_grad,
        grad_wrt_z,
        grad_wrt_gw,
        None,
        grad_wrt_output,
        None,
        None,
        None,
    )


_stack_point_bwd_op.register_autograd(
    _stack_point_bwd_backward, setup_context=_stack_point_bwd_setup_context
)


def _mixing_stack_train_second_order(
    h_u0: Tensor | None,
    h_alpha: Tensor | None,
    h_w0: Tensor | None,
    h_w1: Tensor | None,
    h_gw: Tensor | None,
    grad_out: Tensor,
    x_local: Tensor,
    z_all: Tensor,
    u_final: Tensor,
    alpha: Tensor,
    w0t_all: Tensor,
    w1t_all: Tensor,
    gw_all: Tensor,
    gwt_all: Tensor,
    u0: Tensor | None,
    grad_z_upstream: Tensor | None,
    grad_u_upstream: Tensor | None,
    lmax: int,
    focus_dim: int,
    apply_alpha: bool,
) -> tuple[
    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor
]:
    r"""Differentiate the training backward with respect to its own inputs.

    The first order walks the stack from the last gated layer to the first,

    .. math::

        \bar g_l = \bar g_{l+1} + P_l(\bar g_{l+1}, z_l, G_l)\,W_l^{\mathsf T},
        \qquad
        \bar W_l = u_l^{\mathsf T} \bar z_l, \quad
        \bar G_l = s_l^{\mathsf T} \bar q_l, \quad
        u_l = u_{l+1} - \mathrm{act}(z_l),

    and is linear in :math:`\bar g`. Its adjoint therefore walks the layers in
    the opposite direction. At layer :math:`l`, with :math:`h_l` the cotangent
    of :math:`\bar g_l` and :math:`h_{W_l}, h_{G_l}` the cotangents of the
    weight gradients, the cotangents of the pointwise outputs are

    .. math::

        h_{\bar z_l} = h_l W_l + u_l\,h_{W_l}, \qquad
        h_{\bar q_l} = s_l\,h_{G_l},

    which the fused second order of the gated activation converts into the
    adjoint increment of :math:`h_{l+1}` and the gradients with respect to
    :math:`z_l` and :math:`G_l`. Two further routes close the system: the
    residual contraction contributes :math:`\bar z_l^{\mathsf T} h_l` to the
    (transposed) weight, and the recovered layer inputs carry the cotangent

    .. math::

        h_{u_l} = h_{u_{l-1}} + \bar z_l\,h_{W_l}^{\mathsf T},

    whose route through the recovery ``u_l = u_{l+1} - act(z_l)`` subtracts the
    activation's own vector-Jacobian product from the pre-activation gradient.
    The head of the adjoint unwinds the final identity layer and the
    competition scale exactly as the first order applied them.

    Parameters
    ----------
    h_u0, h_alpha, h_w0, h_w1, h_gw : Tensor or None
        Cotangents of the operator's five outputs.
    grad_out : Tensor
        Edge-major output cotangent the first order was called with.
    x_local : Tensor
        Edge-major stack output (read only through the replayed traversal).
    z_all : Tensor
        Stacked pre-activations, with shape ``(NL, F, E, ROW)``.
    u_final : Tensor
        Input of the final identity layer, with shape ``(F, E, ROW)``.
    alpha : Tensor
        Focus competition weight, with shape ``(E, F)``.
    w0t_all, w1t_all : Tensor
        Stacked transposed block weights, as the operator receives them.
    gw_all, gwt_all : Tensor
        Stacked gate projections and their transposes.
    u0 : Tensor or None
        Stack input, with shape ``(F, E, ROW)``. When supplied, the bottom
        layer's weight gradients linearize around this exact value rather
        than the recovered one, and the corresponding input cotangent leaves
        through the ``u0`` slot instead of entering the recovery chain.
    lmax : int
        Maximum degree.
    focus_dim : int
        Per-focus channel width.
    apply_alpha : bool
        Whether the forward scaled its output by the competition weight.

    Returns
    -------
    tuple
        Gradients with respect to ``(grad_out, z_all, u_final, alpha,
        w0t_all, w1t_all, gw_all, u0, grad_z_upstream, grad_u_upstream)``.
        The ``alpha``, ``u0`` and the two upstream slots are zero-length
        sentinels when the scale was never applied or the corresponding
        operand was absent; operator schemas carry no optional returns, so
        the autograd wrapper restores the None semantics.
        The upstream gradients enter the first order additively, so their
        cotangents are exactly the pre-activation cotangent of the matching
        layer and the head of the adjoint before the identity layer unwinds.
    """
    n_gated, n_focus, n_edge, row = z_all.shape
    lmax = int(lmax)
    focus_dim = int(focus_dim)
    m0 = (lmax + 1) * focus_dim
    dtype = grad_out.dtype
    use_bmm = focus_dim >= GATE_BMM_MIN_FOCUS_DIM
    # Views in the forward orientation ``z = u W``.
    w0_all = w0t_all.transpose(2, 3)
    w1_all = w1t_all.transpose(2, 3)

    # === Replay the first-order traversal for its linearization points ===
    if use_bmm or not _single_launch_traversal(z_all):
        _, _, _, state = _stack_backward_traversal(
            grad_out,
            x_local,
            z_all,
            u_final,
            alpha,
            w0t_all,
            w1t_all,
            gw_all,
            gwt_all,
            grad_z_upstream,
            grad_u_upstream,
            lmax,
            focus_dim,
            apply_alpha,
            with_weights=False,
            keep=True,
            need_alpha=False,
            need_logit=h_gw is not None,
        )
    else:
        _, _, state = _stack_train_traversal(
            grad_out,
            z_all,
            u_final,
            alpha,
            w0t_all,
            w1t_all,
            gw_all,
            grad_z_upstream,
            grad_u_upstream,
            lmax,
            focus_dim,
            apply_alpha,
            need_alpha=False,
            keep=True,
        )
    if apply_alpha:
        scaled = grad_out * alpha.unsqueeze(-1).to(dtype)
    else:
        scaled = grad_out
    grad_final = scaled.permute(1, 0, 2).contiguous()

    h = (
        h_u0.contiguous()
        if h_u0 is not None
        else torch.zeros((n_focus, n_edge, row), device=grad_out.device, dtype=dtype)
    )
    hu: Tensor | None = None
    # Contiguous allocations: the operator contract promises contiguous
    # gradients whatever the (possibly folded, strided) operand layouts.
    grad_z_out = z_all.new_empty(z_all.shape)
    grad_gw_out = gw_all.new_zeros(gw_all.shape)
    grad_w0t_out = w0t_all.new_empty(w0t_all.shape)
    grad_w1t_out = w1t_all.new_empty(w1t_all.shape)
    grad_gz_up = (
        z_all.new_empty(z_all.shape)
        if grad_z_upstream is not None
        else grad_out.new_empty(0)
    )

    # === Adjoint traversal, first gated layer to last ===
    grad_u0_in: Tensor | None = None
    for layer in range(n_gated):
        gz = state.grad_z[layer]
        gq = state.grad_logit[layer]
        # The bottom layer linearizes around the exact stack input when it is
        # available; every other layer input exists only as a recovered value.
        exact_bottom = layer == 0 and u0 is not None
        u_layer = u0 if exact_bottom else state.inputs[layer]
        z_layer = z_all[layer]
        s_layer = z_layer[:, :, :focus_dim]
        gw = gw_all[layer]
        gwt = gwt_all[layer]

        # Cotangent of the pre-activation gradient: the residual contraction
        # plus the weight-gradient route through the recovered input.
        hgz0 = torch.bmm(h[:, :, :m0], w0_all[layer])
        hgz1 = torch.bmm(h[:, :, m0:], w1_all[layer])
        if h_w0 is not None:
            hgz0 = hgz0 + torch.bmm(u_layer[:, :, :m0], h_w0[layer])
        if h_w1 is not None:
            hgz1 = hgz1 + torch.bmm(u_layer[:, :, m0:], h_w1[layer])
        hgz = torch.cat([hgz0, hgz1], dim=-1)
        if grad_z_upstream is not None:
            # The upstream pre-activation gradient joined ``gz`` additively,
            # so its cotangent is the pre-activation cotangent itself.
            grad_gz_up[layer] = hgz

        # Cotangent of the gate-logit gradient.
        hgq = torch.bmm(s_layer, h_gw[layer]) if h_gw is not None else None
        if use_bmm:
            # The first order contracted the logit gradient onto the scalars
            # outside the pointwise kernel; its adjoint and its weight route
            # are therefore supplied here rather than inside the second-order
            # kernel.
            hgq = accumulate(hgq, torch.bmm(hgz[:, :, :focus_dim], gw))
            grad_gw_out[layer] += torch.bmm(
                hgz[:, :, :focus_dim].transpose(1, 2), gq.to(dtype)
            )

        # The residual contraction's weight route, against the pre-update h.
        torch.bmm(gz[:, :, :m0].transpose(1, 2), h[:, :, :m0], out=grad_w0t_out[layer])
        torch.bmm(gz[:, :, m0:].transpose(1, 2), h[:, :, m0:], out=grad_w1t_out[layer])

        h_next, _, dgw = gated_activation_second_order(
            hgz,
            hgq,
            state.upstream[layer],
            z_layer,
            gw,
            gwt,
            lmax,
            focus_dim,
            use_bmm,
            out_z=grad_z_out[layer],
            add_to=h,
        )
        grad_gw_out[layer] += dgw
        if h_gw is not None:
            # The gate-projection gradient also reads the scalar rows directly.
            grad_z_out[layer, :, :, :focus_dim] += torch.bmm(
                gq.to(dtype), h_gw[layer].transpose(1, 2)
            )

        # Cotangent of the layer input, and its route through the recovery
        # back onto the pre-activation and the gate projection. The exact
        # bottom input is an operand of the operator rather than a function
        # of the traversal, so its cotangent leaves directly.
        if h_w0 is not None or h_w1 is not None:
            src0 = (
                torch.bmm(gz[:, :, :m0], h_w0[layer].transpose(1, 2))
                if h_w0 is not None
                else torch.zeros_like(gz[:, :, :m0])
            )
            src1 = (
                torch.bmm(gz[:, :, m0:], h_w1[layer].transpose(1, 2))
                if h_w1 is not None
                else torch.zeros_like(gz[:, :, m0:])
            )
            src = torch.cat([src0, src1], dim=-1)
            if exact_bottom:
                grad_u0_in = src
            else:
                hu = accumulate(hu, src)
        if hu is not None:
            gz_hu, gq_hu, _ = _stack_point_bwd_impl(
                hu, z_layer, gw, gwt, hu, lmax, focus_dim, use_bmm
            )
            if use_bmm:
                gz_hu[:, :, :focus_dim] += torch.bmm(gq_hu.to(dtype), gwt)
            grad_z_out[layer] -= gz_hu
            grad_gw_out[layer] -= torch.bmm(s_layer.transpose(1, 2), gq_hu.to(dtype))

        h = h_next

    # The upstream final-activation gradient joined the head additively.
    grad_gu_up = h.clone() if grad_u_upstream is not None else grad_out.new_empty(0)

    # === Final identity layer and the competition scale ===
    h_gbar = h + torch.cat(
        [
            torch.bmm(h[:, :, :m0], w0_all[n_gated]),
            torch.bmm(h[:, :, m0:], w1_all[n_gated]),
        ],
        dim=-1,
    )
    torch.bmm(
        grad_final[:, :, :m0].transpose(1, 2),
        h[:, :, :m0],
        out=grad_w0t_out[n_gated],
    )
    torch.bmm(
        grad_final[:, :, m0:].transpose(1, 2),
        h[:, :, m0:],
        out=grad_w1t_out[n_gated],
    )
    if h_w0 is not None or h_w1 is not None:
        if h_w0 is not None:
            h_gbar[:, :, :m0] += torch.bmm(u_final[:, :, :m0], h_w0[n_gated])
        if h_w1 is not None:
            h_gbar[:, :, m0:] += torch.bmm(u_final[:, :, m0:], h_w1[n_gated])
        src0 = (
            torch.bmm(grad_final[:, :, :m0], h_w0[n_gated].transpose(1, 2))
            if h_w0 is not None
            else torch.zeros_like(grad_final[:, :, :m0])
        )
        src1 = (
            torch.bmm(grad_final[:, :, m0:], h_w1[n_gated].transpose(1, 2))
            if h_w1 is not None
            else torch.zeros_like(grad_final[:, :, m0:])
        )
        hu = accumulate(hu, torch.cat([src0, src1], dim=-1))

    grad_grad_out: Tensor | None = None
    if apply_alpha and h_alpha is not None:
        # ``grad_alpha`` contracted the raw cotangent against the unscaled
        # output; both factors receive its cotangent in turn.
        y_fm = u_final + torch.cat(
            [
                torch.bmm(u_final[:, :, :m0], w0_all[n_gated]),
                torch.bmm(u_final[:, :, m0:], w1_all[n_gated]),
            ],
            dim=-1,
        )
        grad_grad_out = h_alpha.unsqueeze(-1).to(dtype) * y_fm.permute(1, 0, 2)
        v = (h_alpha.unsqueeze(-1).to(dtype) * grad_out).permute(1, 0, 2).contiguous()
        hu = accumulate(
            hu,
            v
            + torch.cat(
                [
                    torch.bmm(v[:, :, :m0], w0t_all[n_gated]),
                    torch.bmm(v[:, :, m0:], w1t_all[n_gated]),
                ],
                dim=-1,
            ),
        )
        grad_w0t_out[n_gated] += torch.bmm(
            v[:, :, :m0].transpose(1, 2), u_final[:, :, :m0]
        )
        grad_w1t_out[n_gated] += torch.bmm(
            v[:, :, m0:].transpose(1, 2), u_final[:, :, m0:]
        )

    grad_u_final = (
        hu
        if hu is not None
        else torch.zeros((n_focus, n_edge, row), device=grad_out.device, dtype=dtype)
    )
    if apply_alpha:
        term = h_gbar.permute(1, 0, 2) * alpha.unsqueeze(-1).to(dtype)
        grad_grad_out = accumulate(grad_grad_out, term)
        grad_alpha_in = (h_gbar.permute(1, 0, 2) * grad_out).sum(dim=-1)
    else:
        grad_grad_out = accumulate(grad_grad_out, h_gbar.permute(1, 0, 2))
        grad_alpha_in = grad_out.new_empty(0)
    if u0 is not None and grad_u0_in is None:
        grad_u0_in = torch.zeros(
            (n_focus, n_edge, row), device=grad_out.device, dtype=dtype
        )
    return (
        grad_grad_out.contiguous(),
        grad_z_out,
        grad_u_final,
        grad_alpha_in,
        grad_w0t_out,
        grad_w1t_out,
        grad_gw_out,
        grad_u0_in if u0 is not None else grad_out.new_empty(0),
        grad_gz_up,
        grad_gu_up,
    )


def _mixing_stack_train_bwd_setup_context(ctx, inputs, output):
    (
        grad_out,
        x_local,
        z_all,
        u_final,
        alpha,
        w0t_all,
        w1t_all,
        gw_all,
        gwt_all,
        u0,
        grad_z_upstream,
        grad_u_upstream,
        lmax,
        focus_dim,
        apply_alpha,
    ) = inputs
    ctx.save_for_backward(
        grad_out,
        x_local,
        z_all,
        u_final,
        alpha,
        w0t_all,
        w1t_all,
        gw_all,
        gwt_all,
        u0,
        grad_z_upstream,
        grad_u_upstream,
    )
    # A force loss sends a cotangent only through the input gradient; the
    # parameter-gradient outputs feed the optimizer. Their cotangents must
    # stay ``None`` -- materialized zeros would both hide the force regime
    # from the second-order dispatch and drag the full input-recovery
    # adjoint through zero contributions.
    ctx.set_materialize_grads(False)
    ctx.lmax = lmax
    ctx.focus_dim = focus_dim
    ctx.apply_alpha = apply_alpha


_mixing_stack_train_bwd2_op = torch.library.custom_op(
    "sezm_triton::so2_mixing_stack_train_bwd2",
    _mixing_stack_train_second_order,
    mutates_args=(),
)


@_mixing_stack_train_bwd2_op.register_fake
def _(
    h_u0,
    h_alpha,
    h_w0,
    h_w1,
    h_gw,
    grad_out,
    x_local,
    z_all,
    u_final,
    alpha,
    w0t_all,
    w1t_all,
    gw_all,
    gwt_all,
    u0,
    grad_z_upstream,
    grad_u_upstream,
    lmax,
    focus_dim,
    apply_alpha,
):
    # Contiguous allocations throughout: the implementations return
    # contiguous gradients regardless of the (possibly folded, strided)
    # operand layouts, and the compiled graph asserts the fake strides.
    return (
        grad_out.new_empty(grad_out.shape),
        z_all.new_empty(z_all.shape),
        u_final.new_empty(u_final.shape),
        alpha.new_empty(alpha.shape) if apply_alpha else grad_out.new_empty(0),
        w0t_all.new_empty(w0t_all.shape),
        w1t_all.new_empty(w1t_all.shape),
        gw_all.new_empty(gw_all.shape),
        (u_final.new_empty(u_final.shape) if u0 is not None else grad_out.new_empty(0)),
        (
            z_all.new_empty(z_all.shape)
            if grad_z_upstream is not None
            else grad_out.new_empty(0)
        ),
        (
            u_final.new_empty(u_final.shape)
            if grad_u_upstream is not None
            else grad_out.new_empty(0)
        ),
    )


def _mixing_stack_train_bwd_backward(ctx, *grads: Tensor | None):
    """Second order of the training backward.

    Reached only by a force loss, which differentiates the backward once more.
    Being the highest order, nothing differentiates this body in turn; the
    adjoint traversal runs as one operator so that a tracer records an atomic
    node rather than inlining the traversal into the graph.
    """
    h_u0, h_alpha, h_w0, h_w1, h_gw = grads
    if all(g is None for g in grads):
        return (None,) * 15
    (
        grad_out,
        x_local,
        z_all,
        u_final,
        alpha,
        w0t_all,
        w1t_all,
        gw_all,
        gwt_all,
        u0,
        grad_z_upstream,
        grad_u_upstream,
    ) = ctx.saved_tensors
    (
        grad_grad_out,
        grad_z_out,
        grad_u_final,
        grad_alpha_in,
        grad_w0t_out,
        grad_w1t_out,
        grad_gw_out,
        grad_u0_in,
        grad_gz_up,
        grad_gu_up,
    ) = _mixing_stack_train_bwd2_op(
        h_u0,
        h_alpha,
        h_w0,
        h_w1,
        h_gw,
        grad_out,
        x_local,
        z_all,
        u_final,
        alpha,
        w0t_all,
        w1t_all,
        gw_all,
        gwt_all,
        u0,
        grad_z_upstream,
        grad_u_upstream,
        int(ctx.lmax),
        int(ctx.focus_dim),
        bool(ctx.apply_alpha),
    )
    # inputs: grad_out, x_local, z_all, u_final, alpha, w0t_all, w1t_all,
    # gw_all, gwt_all, u0, grad_z_upstream, grad_u_upstream, lmax, focus_dim,
    # apply_alpha. The transposed gate projection is a pure layout copy of
    # ``gw_all``, so its gradient is folded into the untransposed channel.
    return (
        grad_grad_out,
        None,
        grad_z_out,
        grad_u_final,
        grad_alpha_in if bool(ctx.apply_alpha) else None,
        grad_w0t_out,
        grad_w1t_out,
        grad_gw_out,
        None,
        grad_u0_in if u0 is not None else None,
        grad_gz_up if grad_z_upstream is not None else None,
        grad_gu_up if grad_u_upstream is not None else None,
        None,
        None,
        None,
    )


_mixing_stack_train_bwd_op.register_autograd(
    _mixing_stack_train_bwd_backward,
    setup_context=_mixing_stack_train_bwd_setup_context,
)


def _mixing_stack_setup_context(ctx, inputs, output):
    u0, alpha, w0_all, w1_all, gw_all, lmax, focus_dim, apply_alpha = inputs
    x_local, z_all, u_final = output
    ctx.save_for_backward(u0, alpha, x_local, z_all, u_final, w0_all, w1_all, gw_all)
    # The pre-activation and final-activation outputs usually have no
    # consumer; their cotangents must stay ``None`` rather than materialize
    # as zero surfaces the traversal would then add for nothing.
    ctx.set_materialize_grads(False)
    ctx.lmax = lmax
    ctx.focus_dim = focus_dim
    ctx.apply_alpha = apply_alpha


def _mixing_stack_backward(ctx, grad_out, grad_z, grad_u):
    u0, alpha, x_local, z_all, u_final, w0_all, w1_all, gw_all = ctx.saved_tensors
    # The second differentiation re-enters this backward through the saved
    # pre-activation and final-activation outputs alone; the result surface
    # has no cotangent on that path, and the traversal is linear in it.
    grad_out = (
        grad_out.contiguous() if grad_out is not None else torch.zeros_like(x_local)
    )
    # Training is distinguished by the parameters asking for gradients, with
    # two fallbacks: gradients arriving on the pre-activation or the
    # final-activation output, and the ambient grad mode of an eager
    # ``create_graph`` backward. A tracer compiles the backward with grad
    # disabled and reports its needs through ``needs_input_grad`` instead.
    wants_weights = any(ctx.needs_input_grad[2:5])
    if (
        wants_weights
        or grad_z is not None
        or grad_u is not None
        or torch.is_grad_enabled()
    ):
        # A force loss differentiates this backward again; the training
        # operator carries the hand-derived second order of the whole
        # traversal, so one fused operator serves both differentiations.
        grad_u0, grad_alpha, grad_w0, grad_w1, grad_gw = _mixing_stack_train_bwd_op(
            grad_out,
            x_local,
            z_all,
            u_final,
            alpha,
            w0_all.transpose(2, 3).contiguous(),
            w1_all.transpose(2, 3).contiguous(),
            gw_all,
            gw_all.transpose(2, 3).contiguous(),
            u0,
            grad_z,
            grad_u,
            ctx.lmax,
            ctx.focus_dim,
            ctx.apply_alpha,
        )
        return (
            grad_u0,
            grad_alpha if ctx.apply_alpha else None,
            grad_w0 if wants_weights else None,
            grad_w1 if wants_weights else None,
            grad_gw if wants_weights else None,
            None,
            None,
            None,
        )
    grad_u0, grad_alpha = _mixing_stack_bwd_op(
        grad_out,
        x_local,
        z_all,
        u_final,
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


_mixing_stack_op.register_autograd(
    _mixing_stack_backward, setup_context=_mixing_stack_setup_context
)
_mixing_stack_op.register_autocast("cuda", torch.bfloat16)


class _TritonRotateMix:
    """Per-convolution entry running rotate-to-local + degree mixing fused.

    Serves the level-1 training path, where the mixing stack itself stays with
    the compiler: the entry replaces the separate rotation kernel, the
    degree-expanded radial multiply and the focus-major relayout with the
    single ``so2_rotate_mix`` operator, whose backward reduces through the
    source CSR view and whose second order is hand-derived.  The call returns
    the mixing-stack input ``(F, E, ROW)`` together with the projected radial
    features whose ``l = 0`` slice feeds the attention aggregation.
    """

    def __init__(self, conv: SO2Convolution) -> None:
        self._conv = conv

    def __call__(
        self,
        x: Tensor,
        edge_cache: EdgeCache,
        radial_feat: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Rotate the gathered source features and apply the degree mixing.

        Parameters
        ----------
        x : Tensor
            Node features with shape (N, D, C_wide).
        edge_cache : EdgeCache
            Precomputed edge cache (provides ``src``, the Wigner ``D_full``
            and the source CSR view).
        radial_feat : Tensor
            Per-edge radial features with shape (E, lmax+1, C).

        Returns
        -------
        u0 : Tensor
            Mixing-stack input with shape (F, E, (3 * lmax + 1) * Cf).
        rad_feat : Tensor
            Projected radial features with shape (E, lmax+1, C_wide).
        """
        conv = self._conv
        src = edge_cache.src
        if conv.radial_hidden_proj is not None:
            rad_feat = conv.radial_hidden_proj(radial_feat)
        else:
            rad_feat = radial_feat
        mixer = conv.radial_degree_mixer
        if mixer is None:
            kc = rad_feat
            cb = rad_feat.new_zeros(1)
            rank = 0
        else:
            kc = torch.matmul(rad_feat.reshape(rad_feat.shape[0], -1), mixer.weight)
            cb = mixer.channel_basis.reshape(-1)
            rank = mixer.rank
        store = getattr(edge_cache, "csr_cache", None)
        csr = None if store is None else store.get("src")
        if csr is None:
            src_order = torch.argsort(src, dim=0, stable=True)
            counts = src.new_zeros(x.shape[0]).scatter_add(0, src, torch.ones_like(src))
            src_rowptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, 0)])
        else:
            src_order, src_rowptr = csr
        u0 = _rotate_mix_op(
            x.contiguous(),
            src,
            src_order,
            src_rowptr,
            edge_cache.D_full,
            kc.contiguous(),
            cb.contiguous(),
            conv.lmax,
            conv.n_focus,
            rank,
        )
        return u0, rad_feat


def _rotate_mix_supported(conv: SO2Convolution) -> bool:
    """Return whether ``conv`` matches the fused rotate-mix configuration.

    A subset of the full value-path support test: only the rotation and the
    radial degree mixing are replaced, so the mixing stack's own constraints
    (gated layers, identity final layer, no norms) do not apply.  The grid
    product is excluded because it consumes the destination-side rotation,
    which the operator does not produce.

    The hidden-width bound is a profitability boundary, not a correctness
    one.  The operator is quadrilinear, so a force loss differentiates it
    through several forward and backward re-entries per call; the fusion must
    save more materialized traffic than those re-entries cost.  The measured
    crossover sits at ``C_wide = 128``: the 128-wide Pro shape gains while the
    64-wide shapes lose, so narrower blocks keep the separate rotation and
    radial-mix kernels whose backwards are bilinear and trilinear.
    """
    return (
        SO2_VALUE_PATH_TRITON_AVAILABLE
        and conv.mmax == 1
        and 1 <= conv.lmax <= _MAX_LMAX
        and conv.so2_focus_dim in _SUPPORTED_FOCUS_DIMS
        and conv.hidden_channels >= 128
        and conv.node_wise_grid_product is None
        and conv.so2_linears[0].weight_m0.dtype is torch.float32
        and (
            conv.radial_degree_mixer is None
            or (
                conv.radial_degree_mixer.mode == "degree_channel"
                and 1 <= conv.radial_degree_mixer.rank <= _MAX_MIXER_RANK
            )
        )
    )


def make_triton_rotate_mix(conv: SO2Convolution) -> _TritonRotateMix | None:
    """Build the fused rotate-mix entry for a convolution block.

    Parameters
    ----------
    conv : SO2Convolution
        The convolution block to accelerate.

    Returns
    -------
    _TritonRotateMix or None
        The entry callable when Triton is available and ``conv`` matches the
        supported configuration; otherwise ``None`` and the caller keeps the
        separate rotation and radial-mix kernels.
    """
    if not _rotate_mix_supported(conv):
        return None
    return _TritonRotateMix(conv)


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

    def _pack_weights(self, *, differentiable: bool) -> tuple[Tensor, Tensor, Tensor]:
        """Stack the SO(2) block weights and gate projections per layer.

        Returns ``(w0_all, w1_all, gw_all)`` with shapes
        ``(n_layers, F, M0, M0)``, ``(n_layers, F, M1, M1)`` and
        ``(n_gated, F, Cf, lmax * Cf)``, all in the ``(in, out)`` convention.
        """
        conv = self._conv
        m0 = (conv.lmax + 1) * conv.so2_focus_dim
        w0_list, w1_list, gw_list = [], [], []
        for layer, linear in enumerate(conv.so2_linears):
            weight = linear._build_so2_weight()
            if not differentiable:
                weight = weight.detach()
            weight = weight.permute(1, 0, 2).contiguous()  # (F, D_m*Cf, D_m*Cf)
            w0_list.append(weight[:, :m0, :m0])
            w1_list.append(weight[:, m0:, m0:])
            non_linear = conv.non_linearities[layer]
            if type(non_linear).__name__ == "GatedActivation":
                gate = non_linear.gate_linear.weight
                if not differentiable:
                    gate = gate.detach()
                gw_list.append(
                    gate.view(
                        conv.so2_focus_dim,
                        conv.n_focus,
                        conv.lmax * conv.so2_focus_dim,
                    ).permute(1, 0, 2)
                )
        return (
            torch.stack(w0_list).contiguous(),
            torch.stack(w1_list).contiguous(),
            torch.stack(gw_list).contiguous(),
        )

    def __call__(
        self,
        x: Tensor,
        edge_cache: EdgeCache,
        radial_feat: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute the SO(2) local features and radial features via the fused ops.

        Parameters
        ----------
        x : Tensor
            Node features with shape (N, D, C_wide).
        edge_cache : EdgeCache
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
        w0_all, w1_all, gw_all = self._pack_weights(differentiable=self._conv.training)

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
        # The backward's segment reduction walks the source CSR view. The pt
        # descriptor builds it once per step and keeps it on the edge cache;
        # a cache-less caller (the reference backends) pays for its own.
        store = getattr(edge_cache, "csr_cache", None)
        csr = None if store is None else store.get("src")
        if csr is None:
            src_order = torch.argsort(src, dim=0, stable=True)
            counts = src.new_zeros(x.shape[0]).scatter_add(0, src, torch.ones_like(src))
            src_rowptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, 0)])
        else:
            src_order, src_rowptr = csr
        u0 = _rotate_mix_op(
            x.contiguous(),
            src,
            src_order,
            src_rowptr,
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
        x_local, _z_all, _u_final = self._stack_op(
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
