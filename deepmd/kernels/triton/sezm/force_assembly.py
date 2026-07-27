# SPDX-License-Identifier: LGPL-3.0-or-later
# pyright: reportMissingImports=false
# ruff: noqa: ANN001, ANN202
"""Segmented force / virial assembly from per-edge energy gradients.

Given the per-edge gradient ``g_e = dE / d(edge_vec_e)`` of an edge-based
energy, the extended force and per-atom virial are

    ``F_k = sum_{dst(e)=k} g_e - sum_{src(e)=k} g_e``
    ``W_k = 0.5 * sum_{e: k in {src(e), dst(e)}} ( -g_e (x) edge_vec_e )``.

The reference assembly issues four ``index_add`` scatters (force to both
endpoints, half virial to both endpoints) plus a materialized ``(E, 9)``
outer product.  Row-atomic scatters serialize on the colliding edges of each
atom, so this operator performs two CSR segment-reduction launches instead
(one over the destination order, one over the source order), each
recomputing the per-edge outer product on the fly.  One program owns one
extended atom; the 12 output scalars (3 force + 9 virial) accumulate in
float64 registers over the segment, which both removes the atomic
serialization and tightens the summation error over the reference fp32
atomics.

The operator is inference-only in practice: the caller keeps the reference
path whenever the force graph must remain differentiable (``create_graph``),
so no autograd formula is registered.
"""

from __future__ import (
    annotations,
)

import torch
from torch import (
    Tensor,
)
from torch.library import (
    wrap_triton,
)

__all__ = [
    "FORCE_ASSEMBLY_TRITON_AVAILABLE",
    "edge_force_assembly",
]

try:
    import triton
    import triton.language as tl

    FORCE_ASSEMBLY_TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without triton
    FORCE_ASSEMBLY_TRITON_AVAILABLE = False


# ======================================================================
# Eager reference / fallback implementation
# ======================================================================
def _force_assembly_reference(
    g: Tensor,
    edge_vec: Tensor,
    dst_order: Tensor,
    dst_row_ptr: Tensor,
    src_order: Tensor,
    src_row_ptr: Tensor,
) -> tuple[Tensor, Tensor]:
    """Eager ground truth built from the CSR topology via ``index_add``."""
    n_ext = dst_row_ptr.shape[0] - 1
    ar = torch.arange(n_ext, device=g.device, dtype=dst_order.dtype)
    dst = torch.repeat_interleave(ar, dst_row_ptr[1:] - dst_row_ptr[:-1])
    src = torch.repeat_interleave(ar, src_row_ptr[1:] - src_row_ptr[:-1])
    g_dst = g.index_select(0, dst_order)
    g_src = g.index_select(0, src_order)
    force = g.new_zeros((n_ext, 3))
    force.index_add_(0, dst, g_dst)
    force.index_add_(0, src, -g_src)
    half_w_dst = -0.5 * torch.einsum(
        "ek,ej->ekj", g_dst, edge_vec.index_select(0, dst_order)
    ).reshape(-1, 9)
    half_w_src = -0.5 * torch.einsum(
        "ek,ej->ekj", g_src, edge_vec.index_select(0, src_order)
    ).reshape(-1, 9)
    virial = g.new_zeros((n_ext, 9))
    virial.index_add_(0, dst, half_w_dst)
    virial.index_add_(0, src, half_w_src)
    return force, virial


# ======================================================================
# Triton kernels
# ======================================================================
if FORCE_ASSEMBLY_TRITON_AVAILABLE:

    @triton.jit
    def _force_segment_kernel(
        g_ptr,  # (E, 3) per-edge energy gradient
        ev_ptr,  # (E, 3) per-edge displacement
        order_ptr,  # (E,) edge ids sorted by the segment key
        row_ptr_ptr,  # (N_ext + 1,) CSR offsets into ``order``
        f_ptr,  # (N_ext, 3)
        w_ptr,  # (N_ext, 9)
        FORCE_SIGN: tl.constexpr,  # +1 for the dst pass, -1 for the src pass
        ACCUMULATE: tl.constexpr,  # add into the outputs instead of overwriting
    ):
        """One endpoint pass of the force / virial segment reduction.

        The virial lanes address the ``(3, 3)`` outer product through a
        padded 16-lane index ``(k, j) = (lane // 4, lane % 4)`` so both the
        force and virial rows stay vectorized; the outer product
        ``-0.5 * g_k * v_j`` is recomputed per edge in registers and never
        materialized.  Accumulation runs in float64.
        """
        node = tl.program_id(0).to(tl.int64)
        beg = tl.load(row_ptr_ptr + node).to(tl.int64)
        end = tl.load(row_ptr_ptr + node + 1).to(tl.int64)
        kf = tl.arange(0, 4)  # force lanes (3 used)
        kw = tl.arange(0, 16)  # virial lanes (9 used)
        f_mask = kf < 3
        w_mask = ((kw // 4) < 3) & ((kw % 4) < 3)
        acc_f = tl.zeros((4,), dtype=tl.float64)
        acc_w = tl.zeros((16,), dtype=tl.float64)
        for i in range(beg, end):
            e = tl.load(order_ptr + i).to(tl.int64)
            g_vec = tl.load(g_ptr + e * 3 + kf, mask=f_mask, other=0.0).to(tl.float64)
            v_j = tl.load(ev_ptr + e * 3 + kw % 4, mask=(kw % 4) < 3, other=0.0).to(
                tl.float64
            )
            g_k = tl.load(g_ptr + e * 3 + kw // 4, mask=(kw // 4) < 3, other=0.0).to(
                tl.float64
            )
            acc_f += g_vec
            acc_w -= 0.5 * g_k * v_j
        acc_f = acc_f * FORCE_SIGN
        w_col = (kw // 4) * 3 + (kw % 4)
        if ACCUMULATE:
            f_prev = tl.load(f_ptr + node * 3 + kf, mask=f_mask, other=0.0).to(
                tl.float64
            )
            acc_f += f_prev
            w_prev = tl.load(w_ptr + node * 9 + w_col, mask=w_mask, other=0.0).to(
                tl.float64
            )
            acc_w += w_prev
        tl.store(f_ptr + node * 3 + kf, acc_f.to(f_ptr.dtype.element_ty), mask=f_mask)
        tl.store(
            w_ptr + node * 9 + w_col, acc_w.to(w_ptr.dtype.element_ty), mask=w_mask
        )


# ======================================================================
# Dispatch, operator registration and public API
# ======================================================================
def _use_triton(tensor: Tensor) -> bool:
    return (
        FORCE_ASSEMBLY_TRITON_AVAILABLE
        and tensor.is_cuda
        and tensor.dtype in (torch.float32, torch.float64)
    )


def _force_assembly_impl(
    g: Tensor,
    edge_vec: Tensor,
    dst_order: Tensor,
    dst_row_ptr: Tensor,
    src_order: Tensor,
    src_row_ptr: Tensor,
) -> tuple[Tensor, Tensor]:
    if not _use_triton(g):
        return _force_assembly_reference(
            g, edge_vec, dst_order, dst_row_ptr, src_order, src_row_ptr
        )
    n_ext = dst_row_ptr.shape[0] - 1
    force = torch.empty((n_ext, 3), dtype=g.dtype, device=g.device)
    virial = torch.empty((n_ext, 9), dtype=g.dtype, device=g.device)
    wrap_triton(_force_segment_kernel)[(n_ext,)](
        g,
        edge_vec,
        dst_order,
        dst_row_ptr,
        force,
        virial,
        FORCE_SIGN=1,
        ACCUMULATE=False,
        num_warps=1,
        num_stages=2,
    )
    wrap_triton(_force_segment_kernel)[(n_ext,)](
        g,
        edge_vec,
        src_order,
        src_row_ptr,
        force,
        virial,
        FORCE_SIGN=-1,
        ACCUMULATE=True,
        num_warps=1,
        num_stages=2,
    )
    return force, virial


_force_assembly_op = torch.library.triton_op(
    "sezm_triton::edge_force_assembly", mutates_args=()
)(_force_assembly_impl)


@_force_assembly_op.register_fake
def _(g, edge_vec, dst_order, dst_row_ptr, src_order, src_row_ptr):
    n_ext = dst_row_ptr.shape[0] - 1
    return g.new_empty((n_ext, 3)), g.new_empty((n_ext, 9))


def edge_force_assembly(
    g: Tensor,
    edge_vec: Tensor,
    dst_order: Tensor,
    dst_row_ptr: Tensor,
    src_order: Tensor,
    src_row_ptr: Tensor,
) -> tuple[Tensor, Tensor]:
    """Assemble the extended force and per-atom virial from edge gradients.

    Parameters
    ----------
    g : Tensor
        Per-edge energy gradient ``dE / d(edge_vec)`` with shape (E, 3).
    edge_vec : Tensor
        Per-edge displacement vectors with shape (E, 3).
    dst_order, src_order : Tensor
        Edge ids sorted by destination / source extended index, each with
        shape (E,) (from ``torch.argsort``).
    dst_row_ptr, src_row_ptr : Tensor
        CSR offsets into the respective order with shape (N_ext + 1,)
        (from ``torch.searchsorted`` on the sorted keys); the length carries
        the extended-atom count, so the atom axis is never specialized.

    Returns
    -------
    force : Tensor
        Extended force with shape (N_ext, 3).
    virial : Tensor
        Extended per-atom virial with shape (N_ext, 9), split symmetrically
        between the two endpoints of each edge.
    """
    return _force_assembly_op(
        g, edge_vec, dst_order, dst_row_ptr, src_order, src_row_ptr
    )
