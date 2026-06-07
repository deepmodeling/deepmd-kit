# SPDX-License-Identifier: LGPL-3.0-or-later
# pyright: reportMissingImports=false
# ruff: noqa: ANN001
"""
CuTe-DSL fused SO(2) rotation kernels for SeZM / DPA4.

Status and benchmark conclusion
-------------------------------
This implementation is experimental and is **not** wired into the production
SO(2) convolution. The shipping accelerated rotation path uses the Triton
block-diagonal kernels in ``sezm_nn/triton/so2_rotation.py`` (enabled by
``DP_TRITON_INFER``); this module is retained for reference and further
experiments.

In head-to-head benchmarks against the compiled dense ``bmm`` and the Triton
kernels, the CuTe path had the best peak memory (roughly 2-4x lower than the
compiled dense path, lower than Triton) and won the forward pass, but its
``rotate_back`` backward -- and the forward+backward at large ``lmax`` (~10) --
were slower than cuBLAS. The Triton block-diagonal kernels were chosen for
production because their speed (2-8x over the dense baseline) and native
``torch.compile`` composability outweigh the CuTe memory advantage in the target
``lmax`` 2-5, ``mmax == 1`` regime.

Operator definitions (ground truth, fp32)
-----------------------------------------
Let ``x`` be packed node features, ``src`` the per-edge source-node indices,
``wigner`` the per-edge block-diagonal Wigner-D matrices, ``coeff_index`` the
``m``-major reduced-layout indices and ``dim_full = D`` the full packed SO(3)
dimension (``D <= Dw`` where ``wigner`` is ``(E, Dw, Dw)``).

``rotate_to_local`` lifts global node features into the per-edge local frame and
truncates to the reduced layout in one fused step::

    out[e, i, c] = sum_j  wigner[e, coeff_index[i], j] * x[src[e], j, c]
    # i in [0, Dm), j in [0, D), c in [0, C)

``rotate_back`` is the (column-selected) inverse rotation::

    out[e, i, c] = sum_j  wigner[e, i, coeff_index[j]] * x_local[e, j, c]
    # i in [0, D), j in [0, Dm), c in [0, C)

Both operators are batched (one tiny GEMM per edge) with two gathers fused in:
the Wigner row/column selection by ``coeff_index`` and the source-node gather by
``src``. Fusing the gathers means the large ``D_to_m`` ``(E, Dm, D)`` and
``x_src`` ``(E, D, C)`` intermediates produced by the eager ``index_select`` +
``bmm`` reference are never written to or read from global memory, which is the
main source of the speed/peak-memory advantage.

Backward (both feature *and* ``wigner`` gradients, required for forces)
----------------------------------------------------------------------
``rotate_to_local``::

    grad_edge[e, j, c] = sum_i  wigner[e, coeff_index[i], j] * grad_out[e, i, c]
    grad_x = scatter_add(grad_edge, dim=0, index=src)            # (N, D, C)
    grad_wigner[e, coeff_index[i], j] = sum_c grad_out[e, i, c] * x[src[e], j, c]

``rotate_back``::

    grad_x_local[e, j, c] = sum_i  wigner[e, i, coeff_index[j]] * grad_out[e, i, c]
    grad_wigner[e, i, coeff_index[j]] = sum_c grad_out[e, i, c] * x_local[e, j, c]

(all other entries of ``grad_wigner`` are zero).

Kernel design
-------------
Every kernel computes, per edge, a small matrix product ``out = A @ B`` (with
one operand gathered) using a **2D register-blocked GEMM**:

* one CUDA block per edge;
* the operand whose layout is ``(K, C)`` (the source-node / local / grad_out
  tile) is staged once into shared memory;
* each thread owns a ``TM x TN`` register tile of the output and sweeps the
  contraction dimension ``K``, loading ``TM`` values of ``A`` and ``TN`` values
  of ``B`` per step and issuing ``TM*TN`` FFMAs. This pushes the load:FFMA ratio
  to ``(TM+TN)/(TM*TN)`` so the kernel is compute-bound rather than
  load/store-unit bound;
* the per-output-row Wigner index gather (``coeff_index``) is hoisted out of the
  contraction loop into registers.

The two ``grad_wigner`` kernels are batched outer products (contraction over the
channel axis ``C``) and use the same register-blocked skeleton with a 2D tile
sweep over the ``(Dm, D)`` output. When both per-edge operands fit in shared
memory (small/medium ``lmax``) both are staged there; otherwise only
``grad_out`` is staged and the other operand streams from global memory through
L1. The ``rotate_to_local`` ``grad_x`` contribution is fused with its
source-node scatter via atomic adds, so neither a ``grad_edge`` intermediate nor
a separate ``index_add`` is materialized.

All accumulation is fp32 (no TF32), keeping the potential-energy surface smooth.

Composability
-------------
The kernels are wrapped with ``torch.library.custom_op`` (functional,
``mutates_args=()``) plus ``register_fake`` and ``register_autograd``. The
backward is itself a custom op, so ``torch.compile`` can include and
differentiate the whole thing as an opaque, side-effect-free operator. Kernels
are launched on torch's current CUDA stream so they order correctly with the
surrounding eager / compiled graph.
"""

from __future__ import (
    annotations,
)

import threading
from typing import (
    TYPE_CHECKING,
    Any,
)

import torch
from torch import (
    Tensor,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )

    from cuda.bindings import driver as _cuda_driver

try:
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    from cutlass.cute.runtime import (
        from_dlpack,
    )

    SEZM_CUTE_AVAILABLE = True
except Exception:  # pragma: no cover - import guard for non-CuTe environments
    SEZM_CUTE_AVAILABLE = False


# === Kernel tuning constants =================================================
# Register-tile dimensions (TM output rows x TN output cols per thread) and the
# block thread geometry. ``C`` (= 64) is the channel axis and is the N dimension
# for the matmul-like kernels; ``TN`` divides ``C``.
_TM = 4
_TN = 4
_BLOCK_ROWS = 16  # block.y for matmul-like kernels (block.x = C // TN)

# grad_wigner: budget (bytes) below which both operands are staged in shared
# memory (the fast path); above it (e.g. lmax=10) only grad_out is staged and
# the other operand streams from global memory through L1.
_GW_SMEM_BUDGET = 46000


def _gw_tile(D: int, Dm: int, C: int) -> tuple[int, int, int, int, bool]:
    """Pick (TM, TN, BX, BY, both_in_smem) for a grad_wigner output of (M, N).

    The register tile and block geometry are chosen so the block is well
    occupied for the given output size, and both operands are staged in shared
    memory when the per-edge tiles fit inside ``_GW_SMEM_BUDGET``.
    """
    both = (Dm + D) * C * 4 <= _GW_SMEM_BUDGET
    if D <= 20:  # small output (e.g. lmax=3): keep the tile/block small
        return 2, 2, 16, 16, both
    if D <= 50:  # medium output (e.g. lmax=5)
        return 8, 8, 8, 8, both
    if both:  # large output that still fits both operands (e.g. lmax=7)
        return 8, 4, 8, 16, both
    return 8, 8, 8, 8, both  # large output, only grad_out staged (e.g. lmax=10)


# === Eager reference (ground truth, also used as fallback) ===================
def _rotate_to_local_eager(
    x: Tensor, src: Tensor, wigner: Tensor, coeff_index: Tensor, dim_full: int
) -> Tensor:
    """Reference ``D_to_m @ x[src]`` used for fallback and validation."""
    d_to_m = wigner[:, :dim_full, :dim_full].index_select(1, coeff_index)
    return torch.bmm(d_to_m, x.index_select(0, src))


def _rotate_back_eager(
    x_local: Tensor, wigner: Tensor, coeff_index: Tensor, dim_full: int
) -> Tensor:
    """Reference ``Dt_from_m @ x_local`` used for fallback and validation."""
    dt_from_m = wigner[:, :dim_full, :dim_full].index_select(2, coeff_index)
    return torch.bmm(dt_from_m, x_local)


if SEZM_CUTE_AVAILABLE:
    _F32 = cutlass.Float32
    _I64 = cutlass.Int64

    # ------------------------------------------------------------------
    # Family 1: out(M, C) = A(M, K) @ S(K, C), with S staged in shared
    # memory and A read from the Wigner tensor with a per-element gather.
    # Specialized by how A[m, k] maps into the (Dw, Dw) Wigner block.
    # ------------------------------------------------------------------
    def _build_rotate_to_local_fwd(D: int, Dm: int, C: int) -> Callable:
        """``out[m=i, n=c] = sum_{k=j} wigner[e, idx[m], k] * x[src[e], k, n]``."""
        M, K = Dm, D
        TM, TN, BY = _TM, _TN, _BLOCK_ROWS
        BX = C // TN
        T = BX * BY

        @cute.kernel
        def kernel(m_x, m_src, m_w, m_idx, m_out) -> None:
            e, _, _ = cute.arch.block_idx()
            cx, ry, _ = cute.arch.thread_idx()
            smem = cute.arch.alloc_smem(_F32, K * C)
            s_s = cute.make_tensor(smem, cute.make_layout((K, C), stride=(C, 1)))
            src_e = m_src[e]
            x_node = m_x[src_e, None, None]
            tid = ry * BX + cx
            for kk in cutlass.range(tid, K * C, T):
                s_s[kk // C, kk % C] = x_node[kk // C, kk % C]
            cute.arch.sync_threads()

            w_e = m_w[e, None, None]
            out_e = m_out[e, None, None]
            for rt0 in cutlass.range(ry * TM, M, BY * TM):
                acc = cute.make_fragment((TM, TN), _F32)
                wi = cute.make_fragment((TM,), _I64)
                bf = cute.make_fragment((TN,), _F32)
                for t in range(TM):
                    wi[t] = m_idx[(rt0 + t) % M]  # gathered Wigner row
                    for n in range(TN):
                        acc[t, n] = _F32(0.0)
                for k in cutlass.range(K):
                    for n in range(TN):
                        bf[n] = s_s[k, cx * TN + n]
                    for t in range(TM):
                        a = w_e[wi[t], k]
                        for n in range(TN):
                            acc[t, n] = acc[t, n] + a * bf[n]
                for t in range(TM):
                    m = rt0 + t
                    if m < M:
                        for n in range(TN):
                            out_e[m, cx * TN + n] = acc[t, n]

        @cute.jit
        def host(m_x, m_src, m_w, m_idx, m_out, stream: _cuda_driver.CUstream) -> None:
            e = m_out.shape[0]
            kernel(m_x, m_src, m_w, m_idx, m_out).launch(
                grid=[e, 1, 1], block=[BX, BY, 1], stream=stream
            )

        return host

    def _build_rotate_back_fwd(D: int, Dm: int, C: int) -> Callable:
        """``out[m=i, n=c] = sum_{k=j} wigner[e, m, idx[k]] * x_local[e, k, n]``."""
        M, K = D, Dm
        TM, TN, BY = _TM, _TN, _BLOCK_ROWS
        BX = C // TN
        T = BX * BY

        @cute.kernel
        def kernel(m_xl, m_w, m_idx, m_out) -> None:
            e, _, _ = cute.arch.block_idx()
            cx, ry, _ = cute.arch.thread_idx()
            smem = cute.arch.alloc_smem(_F32, K * C)
            s_s = cute.make_tensor(smem, cute.make_layout((K, C), stride=(C, 1)))
            xl_e = m_xl[e, None, None]
            tid = ry * BX + cx
            for kk in cutlass.range(tid, K * C, T):
                s_s[kk // C, kk % C] = xl_e[kk // C, kk % C]
            cute.arch.sync_threads()

            w_e = m_w[e, None, None]
            out_e = m_out[e, None, None]
            for rt0 in cutlass.range(ry * TM, M, BY * TM):
                acc = cute.make_fragment((TM, TN), _F32)
                wr = cute.make_fragment((TM,), _I64)
                bf = cute.make_fragment((TN,), _F32)
                for t in range(TM):
                    wr[t] = (rt0 + t) % M  # direct Wigner row
                    for n in range(TN):
                        acc[t, n] = _F32(0.0)
                for k in cutlass.range(K):
                    kk = m_idx[k]  # gathered Wigner column
                    for n in range(TN):
                        bf[n] = s_s[k, cx * TN + n]
                    for t in range(TM):
                        a = w_e[wr[t], kk]
                        for n in range(TN):
                            acc[t, n] = acc[t, n] + a * bf[n]
                for t in range(TM):
                    m = rt0 + t
                    if m < M:
                        for n in range(TN):
                            out_e[m, cx * TN + n] = acc[t, n]

        @cute.jit
        def host(m_xl, m_w, m_idx, m_out, stream: _cuda_driver.CUstream) -> None:
            e = m_out.shape[0]
            kernel(m_xl, m_w, m_idx, m_out).launch(
                grid=[e, 1, 1], block=[BX, BY, 1], stream=stream
            )

        return host

    def _build_rotate_to_local_bwd_dx(D: int, Dm: int, C: int) -> Callable:
        """``grad_x[src[e], m=j, n=c] += sum_{k=i} wigner[e, idx[k], m] * grad_out[e, k, n]``.

        The per-edge gradient and the scatter-add into ``grad_x`` (indexed by
        ``src``) are fused: each block accumulates its tile and atomically adds it
        into the destination node. This avoids a materialized ``grad_edge`` tensor
        and a separate ``index_add`` pass.
        """
        M, K = D, Dm
        TM, TN, BY = _TM, _TN, _BLOCK_ROWS
        BX = C // TN
        T = BX * BY

        @cute.kernel
        def kernel(m_go, m_w, m_src, m_idx, m_gx) -> None:
            e, _, _ = cute.arch.block_idx()
            cx, ry, _ = cute.arch.thread_idx()
            smem = cute.arch.alloc_smem(_F32, K * C)
            s_s = cute.make_tensor(smem, cute.make_layout((K, C), stride=(C, 1)))
            go_e = m_go[e, None, None]
            tid = ry * BX + cx
            for kk in cutlass.range(tid, K * C, T):
                s_s[kk // C, kk % C] = go_e[kk // C, kk % C]
            cute.arch.sync_threads()

            w_e = m_w[e, None, None]
            gx_node = m_gx[m_src[e], None, None]  # (D, C) view into grad_x[src]
            gx_base = gx_node.iterator  # contiguous (C, 1): element (m, c) -> m*C + c
            for rt0 in cutlass.range(ry * TM, M, BY * TM):
                acc = cute.make_fragment((TM, TN), _F32)
                wc = cute.make_fragment((TM,), _I64)
                bf = cute.make_fragment((TN,), _F32)
                for t in range(TM):
                    wc[t] = (rt0 + t) % M  # direct Wigner column (= output row m)
                    for n in range(TN):
                        acc[t, n] = _F32(0.0)
                for k in cutlass.range(K):
                    kk = m_idx[k]  # gathered Wigner row
                    for n in range(TN):
                        bf[n] = s_s[k, cx * TN + n]
                    for t in range(TM):
                        a = w_e[kk, wc[t]]
                        for n in range(TN):
                            acc[t, n] = acc[t, n] + a * bf[n]
                for t in range(TM):
                    m = rt0 + t
                    if m < M:
                        for n in range(TN):
                            cute.arch.atomic_add(
                                gx_base + (m * C + cx * TN + n), acc[t, n]
                            )

        @cute.jit
        def host(m_go, m_w, m_src, m_idx, m_gx, stream: _cuda_driver.CUstream) -> None:
            e = m_go.shape[0]
            kernel(m_go, m_w, m_src, m_idx, m_gx).launch(
                grid=[e, 1, 1], block=[BX, BY, 1], stream=stream
            )

        return host

    def _build_rotate_back_bwd_dx(D: int, Dm: int, C: int) -> Callable:
        """``grad_x_local[m=j, n=c] = sum_{k=i} wigner[e, k, idx[m]] * grad_out[e, k, n]``."""
        M, K = Dm, D
        TM, TN, BY = _TM, _TN, _BLOCK_ROWS
        BX = C // TN
        T = BX * BY

        @cute.kernel
        def kernel(m_go, m_w, m_idx, m_gxl) -> None:
            e, _, _ = cute.arch.block_idx()
            cx, ry, _ = cute.arch.thread_idx()
            smem = cute.arch.alloc_smem(_F32, K * C)
            s_s = cute.make_tensor(smem, cute.make_layout((K, C), stride=(C, 1)))
            go_e = m_go[e, None, None]
            tid = ry * BX + cx
            for kk in cutlass.range(tid, K * C, T):
                s_s[kk // C, kk % C] = go_e[kk // C, kk % C]
            cute.arch.sync_threads()

            w_e = m_w[e, None, None]
            gxl_e = m_gxl[e, None, None]
            for rt0 in cutlass.range(ry * TM, M, BY * TM):
                acc = cute.make_fragment((TM, TN), _F32)
                wc = cute.make_fragment((TM,), _I64)
                bf = cute.make_fragment((TN,), _F32)
                for t in range(TM):
                    wc[t] = m_idx[(rt0 + t) % M]  # gathered Wigner column
                    for n in range(TN):
                        acc[t, n] = _F32(0.0)
                for k in cutlass.range(K):
                    for n in range(TN):
                        bf[n] = s_s[k, cx * TN + n]
                    for t in range(TM):
                        a = w_e[k, wc[t]]
                        for n in range(TN):
                            acc[t, n] = acc[t, n] + a * bf[n]
                for t in range(TM):
                    m = rt0 + t
                    if m < M:
                        for n in range(TN):
                            gxl_e[m, cx * TN + n] = acc[t, n]

        @cute.jit
        def host(m_go, m_w, m_idx, m_gxl, stream: _cuda_driver.CUstream) -> None:
            e = m_go.shape[0]
            kernel(m_go, m_w, m_idx, m_gxl).launch(
                grid=[e, 1, 1], block=[BX, BY, 1], stream=stream
            )

        return host

    # ------------------------------------------------------------------
    # Family 2: grad_wigner = grad_out @ other^T (contraction over the
    # channel axis C). 2D register-blocked sweep over the (M, N) output,
    # grad_out staged in shared memory, other read from global memory.
    # ------------------------------------------------------------------
    def _build_rotate_to_local_bwd_dw(D: int, Dm: int, C: int) -> Callable:
        """``grad_wigner[e, idx[m=i], n=j] = sum_{k=c} grad_out[e, m, k] * x[src[e], n, k]``."""
        M, N, K = Dm, D, C
        TM, TN, BX, BY, both = _gw_tile(D, Dm, C)
        T = BX * BY

        @cute.kernel
        def kernel(m_go, m_x, m_src, m_idx, m_gw) -> None:
            e, _, _ = cute.arch.block_idx()
            cx, ry, _ = cute.arch.thread_idx()
            sgo = cute.arch.alloc_smem(_F32, M * C)
            s_go = cute.make_tensor(sgo, cute.make_layout((M, C), stride=(C, 1)))
            go_e = m_go[e, None, None]
            src_e = m_src[e]
            x_node = m_x[src_e, None, None]  # (N=D, C)
            tid = ry * BX + cx
            for kk in cutlass.range(tid, M * C, T):
                s_go[kk // C, kk % C] = go_e[kk // C, kk % C]
            # Optionally stage the second operand in shared memory too.
            sx = cute.arch.alloc_smem(_F32, (N * C) if both else 1)
            s_x = cute.make_tensor(
                sx, cute.make_layout(((N, C) if both else (1, 1)), stride=(C, 1))
            )
            if cutlass.const_expr(both):
                for kk in cutlass.range(tid, N * C, T):
                    s_x[kk // C, kk % C] = x_node[kk // C, kk % C]
            cute.arch.sync_threads()

            gw_e = m_gw[e, None, None]
            for mt0 in cutlass.range(ry * TM, M, BY * TM):
                orow = cute.make_fragment((TM,), _I64)
                rt = cute.make_fragment((TM,), cutlass.Int32)
                for t in range(TM):
                    rt[t] = (mt0 + t) % M  # clamped smem row (hoisted out of K loop)
                    orow[t] = m_idx[rt[t]]  # gathered output row
                for nt0 in cutlass.range(cx * TN, N, BX * TN):
                    acc = cute.make_fragment((TM, TN), _F32)
                    af = cute.make_fragment((TM,), _F32)
                    bf = cute.make_fragment((TN,), _F32)
                    ct = cute.make_fragment((TN,), cutlass.Int32)
                    for n in range(TN):
                        ct[n] = (nt0 + n) % N  # clamped col (hoisted)
                    for t in range(TM):
                        for n in range(TN):
                            acc[t, n] = _F32(0.0)
                    for k in cutlass.range(K):
                        for t in range(TM):
                            af[t] = s_go[rt[t], k]
                        if cutlass.const_expr(both):
                            for n in range(TN):
                                bf[n] = s_x[ct[n], k]
                        else:
                            for n in range(TN):
                                bf[n] = x_node[ct[n], k]
                        for t in range(TM):
                            for n in range(TN):
                                acc[t, n] = acc[t, n] + af[t] * bf[n]
                    for t in range(TM):
                        if mt0 + t < M:
                            for n in range(TN):
                                if nt0 + n < N:
                                    gw_e[orow[t], nt0 + n] = acc[t, n]

        @cute.jit
        def host(m_go, m_x, m_src, m_idx, m_gw, stream: _cuda_driver.CUstream) -> None:
            e = m_go.shape[0]
            kernel(m_go, m_x, m_src, m_idx, m_gw).launch(
                grid=[e, 1, 1], block=[BX, BY, 1], stream=stream
            )

        return host

    def _build_rotate_back_bwd_dw(D: int, Dm: int, C: int) -> Callable:
        """``grad_wigner[e, m=i, idx[n=j]] = sum_{k=c} grad_out[e, m, k] * x_local[e, n, k]``."""
        M, N, K = D, Dm, C
        TM, TN, BX, BY, both = _gw_tile(D, Dm, C)
        T = BX * BY

        @cute.kernel
        def kernel(m_go, m_xl, m_idx, m_gw) -> None:
            e, _, _ = cute.arch.block_idx()
            cx, ry, _ = cute.arch.thread_idx()
            sgo = cute.arch.alloc_smem(_F32, M * C)
            s_go = cute.make_tensor(sgo, cute.make_layout((M, C), stride=(C, 1)))
            go_e = m_go[e, None, None]
            xl_e = m_xl[e, None, None]  # (N=Dm, C)
            tid = ry * BX + cx
            for kk in cutlass.range(tid, M * C, T):
                s_go[kk // C, kk % C] = go_e[kk // C, kk % C]
            sx = cute.arch.alloc_smem(_F32, (N * C) if both else 1)
            s_x = cute.make_tensor(
                sx, cute.make_layout(((N, C) if both else (1, 1)), stride=(C, 1))
            )
            if cutlass.const_expr(both):
                for kk in cutlass.range(tid, N * C, T):
                    s_x[kk // C, kk % C] = xl_e[kk // C, kk % C]
            cute.arch.sync_threads()

            gw_e = m_gw[e, None, None]
            for mt0 in cutlass.range(ry * TM, M, BY * TM):
                rt = cute.make_fragment((TM,), cutlass.Int32)
                for t in range(TM):
                    rt[t] = (mt0 + t) % M  # clamped smem row (hoisted out of K loop)
                for nt0 in cutlass.range(cx * TN, N, BX * TN):
                    acc = cute.make_fragment((TM, TN), _F32)
                    ocol = cute.make_fragment((TN,), _I64)
                    ct = cute.make_fragment((TN,), cutlass.Int32)
                    af = cute.make_fragment((TM,), _F32)
                    bf = cute.make_fragment((TN,), _F32)
                    for n in range(TN):
                        ct[n] = (nt0 + n) % N  # clamped col (hoisted)
                        ocol[n] = m_idx[ct[n]]  # gathered output column
                    for t in range(TM):
                        for n in range(TN):
                            acc[t, n] = _F32(0.0)
                    for k in cutlass.range(K):
                        for t in range(TM):
                            af[t] = s_go[rt[t], k]
                        if cutlass.const_expr(both):
                            for n in range(TN):
                                bf[n] = s_x[ct[n], k]
                        else:
                            for n in range(TN):
                                bf[n] = xl_e[ct[n], k]
                        for t in range(TM):
                            for n in range(TN):
                                acc[t, n] = acc[t, n] + af[t] * bf[n]
                    for t in range(TM):
                        i = mt0 + t
                        if i < M:
                            for n in range(TN):
                                if nt0 + n < N:
                                    gw_e[i, ocol[n]] = acc[t, n]

        @cute.jit
        def host(m_go, m_xl, m_idx, m_gw, stream: _cuda_driver.CUstream) -> None:
            e = m_go.shape[0]
            kernel(m_go, m_xl, m_idx, m_gw).launch(
                grid=[e, 1, 1], block=[BX, BY, 1], stream=stream
            )

        return host

    # === Compiled-kernel cache ==============================================
    _compiled_cache: dict[tuple, Any] = {}
    _cache_lock = threading.Lock()

    def _get_compiled(key: tuple, builder: Callable, example_args: tuple) -> Any:
        """Return a JIT-compiled host function, compiling and caching on miss."""
        comp = _compiled_cache.get(key)
        if comp is not None:
            return comp
        with _cache_lock:
            comp = _compiled_cache.get(key)
            if comp is None:
                host = builder(*key[1:])
                comp = cute.compile(host, *example_args)
                _compiled_cache[key] = comp
        return comp

    def _cute_f(t: Tensor) -> Any:
        """Wrap a contiguous (>=2D) fp32 tensor as a CuTe tensor (last dim leading)."""
        return from_dlpack(t).mark_layout_dynamic(leading_dim=t.dim() - 1)

    def _cute_i(t: Tensor) -> Any:
        """Wrap a contiguous 1D int64 tensor as a CuTe tensor."""
        return from_dlpack(t).mark_layout_dynamic()

    # === Low-level kernel dispatch (operate on plain, detached tensors) ======
    def _launch_rotate_to_local_fwd(
        x: Tensor, src: Tensor, wigner: Tensor, coeff_index: Tensor, dim_full: int
    ) -> Tensor:
        e = src.shape[0]
        d, dm, c = dim_full, coeff_index.shape[0], x.shape[2]
        out = torch.empty((e, dm, c), dtype=x.dtype, device=x.device)
        m_x, m_src, m_w = _cute_f(x), _cute_i(src), _cute_f(wigner)
        m_idx, m_out = _cute_i(coeff_index), _cute_f(out)
        stream = cutlass_torch.current_stream()
        comp = _get_compiled(
            ("rtl_fwd", d, dm, c),
            _build_rotate_to_local_fwd,
            (m_x, m_src, m_w, m_idx, m_out, stream),
        )
        comp(m_x, m_src, m_w, m_idx, m_out, stream)
        return out

    def _launch_rotate_back_fwd(
        x_local: Tensor, wigner: Tensor, coeff_index: Tensor, dim_full: int
    ) -> Tensor:
        e = x_local.shape[0]
        d, dm, c = dim_full, coeff_index.shape[0], x_local.shape[2]
        out = torch.empty((e, d, c), dtype=x_local.dtype, device=x_local.device)
        m_xl, m_w = _cute_f(x_local), _cute_f(wigner)
        m_idx, m_out = _cute_i(coeff_index), _cute_f(out)
        stream = cutlass_torch.current_stream()
        comp = _get_compiled(
            ("rb_fwd", d, dm, c),
            _build_rotate_back_fwd,
            (m_xl, m_w, m_idx, m_out, stream),
        )
        comp(m_xl, m_w, m_idx, m_out, stream)
        return out

    def _launch_rotate_to_local_bwd(
        grad_out: Tensor,
        x: Tensor,
        src: Tensor,
        wigner: Tensor,
        coeff_index: Tensor,
        dim_full: int,
    ) -> tuple[Tensor, Tensor]:
        n, e = x.shape[0], src.shape[0]
        d, dm, c = dim_full, coeff_index.shape[0], x.shape[2]
        stream = cutlass_torch.current_stream()

        # grad_x: per-edge gradient fused with the scatter-add into the source
        # node via atomic adds (no materialized grad_edge, no separate index_add).
        grad_x = torch.zeros((n, d, c), dtype=x.dtype, device=x.device)
        m_go, m_w = _cute_f(grad_out), _cute_f(wigner)
        m_src, m_idx, m_gx = _cute_i(src), _cute_i(coeff_index), _cute_f(grad_x)
        comp_dx = _get_compiled(
            ("rtl_bwd_dx", d, dm, c),
            _build_rotate_to_local_bwd_dx,
            (m_go, m_w, m_src, m_idx, m_gx, stream),
        )
        comp_dx(m_go, m_w, m_src, m_idx, m_gx, stream)

        # grad_wigner: per-edge outer product written into the gathered rows.
        grad_wigner = torch.zeros_like(wigner)
        m_x, m_gw = _cute_f(x), _cute_f(grad_wigner)
        comp_dw = _get_compiled(
            ("rtl_bwd_dw", d, dm, c),
            _build_rotate_to_local_bwd_dw,
            (m_go, m_x, m_src, m_idx, m_gw, stream),
        )
        comp_dw(m_go, m_x, m_src, m_idx, m_gw, stream)
        return grad_x, grad_wigner

    def _launch_rotate_back_bwd(
        grad_out: Tensor,
        x_local: Tensor,
        wigner: Tensor,
        coeff_index: Tensor,
        dim_full: int,
    ) -> tuple[Tensor, Tensor]:
        e = x_local.shape[0]
        d, dm, c = dim_full, coeff_index.shape[0], x_local.shape[2]
        stream = cutlass_torch.current_stream()

        grad_x_local = torch.empty(
            (e, dm, c), dtype=x_local.dtype, device=x_local.device
        )
        m_go, m_w = _cute_f(grad_out), _cute_f(wigner)
        m_idx, m_gxl = _cute_i(coeff_index), _cute_f(grad_x_local)
        comp_dx = _get_compiled(
            ("rb_bwd_dx", d, dm, c),
            _build_rotate_back_bwd_dx,
            (m_go, m_w, m_idx, m_gxl, stream),
        )
        comp_dx(m_go, m_w, m_idx, m_gxl, stream)

        grad_wigner = torch.zeros_like(wigner)
        m_xl, m_gw = _cute_f(x_local), _cute_f(grad_wigner)
        comp_dw = _get_compiled(
            ("rb_bwd_dw", d, dm, c),
            _build_rotate_back_bwd_dw,
            (m_go, m_xl, m_idx, m_gw, stream),
        )
        comp_dw(m_go, m_xl, m_idx, m_gw, stream)
        return grad_x_local, grad_wigner

    # === torch.library custom ops ===========================================
    # Forward + backward are registered as functional custom ops so the whole
    # operator is opaque to torch.compile yet correctly differentiable.

    @torch.library.custom_op(
        "sezm_cute::rotate_to_local", mutates_args=(), device_types="cuda"
    )
    def _op_rotate_to_local(
        x: Tensor,
        src: Tensor,
        wigner: Tensor,
        coeff_index: Tensor,
        dim_full: int,
    ) -> Tensor:
        return _launch_rotate_to_local_fwd(
            x.detach().contiguous(),
            src.detach().contiguous(),
            wigner.detach().contiguous(),
            coeff_index.detach().contiguous(),
            int(dim_full),
        )

    @_op_rotate_to_local.register_fake
    def _(
        x: Tensor,
        src: Tensor,
        wigner: Tensor,
        coeff_index: Tensor,
        dim_full: int,
    ) -> Tensor:
        return x.new_empty((src.shape[0], coeff_index.shape[0], x.shape[2]))

    @torch.library.custom_op(
        "sezm_cute::rotate_to_local_bwd", mutates_args=(), device_types="cuda"
    )
    def _op_rotate_to_local_bwd(
        grad_out: Tensor,
        x: Tensor,
        src: Tensor,
        wigner: Tensor,
        coeff_index: Tensor,
        dim_full: int,
    ) -> tuple[Tensor, Tensor]:
        return _launch_rotate_to_local_bwd(
            grad_out.detach().contiguous(),
            x.detach().contiguous(),
            src.detach().contiguous(),
            wigner.detach().contiguous(),
            coeff_index.detach().contiguous(),
            int(dim_full),
        )

    @_op_rotate_to_local_bwd.register_fake
    def _(
        grad_out: Tensor,
        x: Tensor,
        src: Tensor,
        wigner: Tensor,
        coeff_index: Tensor,
        dim_full: int,
    ) -> tuple[Tensor, Tensor]:
        return torch.empty_like(x), torch.empty_like(wigner)

    def _rtl_setup_context(ctx: Any, inputs: tuple, output: Tensor) -> None:
        x, src, wigner, coeff_index, dim_full = inputs
        ctx.save_for_backward(x, src, wigner, coeff_index)
        ctx.dim_full = int(dim_full)

    def _rtl_backward(ctx: Any, grad_out: Tensor) -> tuple:
        x, src, wigner, coeff_index = ctx.saved_tensors
        grad_x, grad_wigner = torch.ops.sezm_cute.rotate_to_local_bwd(
            grad_out, x, src, wigner, coeff_index, ctx.dim_full
        )
        return grad_x, None, grad_wigner, None, None

    _op_rotate_to_local.register_autograd(
        _rtl_backward, setup_context=_rtl_setup_context
    )

    @torch.library.custom_op(
        "sezm_cute::rotate_back", mutates_args=(), device_types="cuda"
    )
    def _op_rotate_back(
        x_local: Tensor,
        wigner: Tensor,
        coeff_index: Tensor,
        dim_full: int,
    ) -> Tensor:
        return _launch_rotate_back_fwd(
            x_local.detach().contiguous(),
            wigner.detach().contiguous(),
            coeff_index.detach().contiguous(),
            int(dim_full),
        )

    @_op_rotate_back.register_fake
    def _(
        x_local: Tensor,
        wigner: Tensor,
        coeff_index: Tensor,
        dim_full: int,
    ) -> Tensor:
        return x_local.new_empty((x_local.shape[0], dim_full, x_local.shape[2]))

    @torch.library.custom_op(
        "sezm_cute::rotate_back_bwd", mutates_args=(), device_types="cuda"
    )
    def _op_rotate_back_bwd(
        grad_out: Tensor,
        x_local: Tensor,
        wigner: Tensor,
        coeff_index: Tensor,
        dim_full: int,
    ) -> tuple[Tensor, Tensor]:
        return _launch_rotate_back_bwd(
            grad_out.detach().contiguous(),
            x_local.detach().contiguous(),
            wigner.detach().contiguous(),
            coeff_index.detach().contiguous(),
            int(dim_full),
        )

    @_op_rotate_back_bwd.register_fake
    def _(
        grad_out: Tensor,
        x_local: Tensor,
        wigner: Tensor,
        coeff_index: Tensor,
        dim_full: int,
    ) -> tuple[Tensor, Tensor]:
        return torch.empty_like(x_local), torch.empty_like(wigner)

    def _rb_setup_context(ctx: Any, inputs: tuple, output: Tensor) -> None:
        x_local, wigner, coeff_index, dim_full = inputs
        ctx.save_for_backward(x_local, wigner, coeff_index)
        ctx.dim_full = int(dim_full)

    def _rb_backward(ctx: Any, grad_out: Tensor) -> tuple:
        x_local, wigner, coeff_index = ctx.saved_tensors
        grad_x_local, grad_wigner = torch.ops.sezm_cute.rotate_back_bwd(
            grad_out, x_local, wigner, coeff_index, ctx.dim_full
        )
        return grad_x_local, grad_wigner, None, None

    _op_rotate_back.register_autograd(_rb_backward, setup_context=_rb_setup_context)


# === Public API ==============================================================
def _cute_usable(*tensors: Tensor) -> bool:
    """Return True when the CuTe fast path is available for these tensors."""
    if not SEZM_CUTE_AVAILABLE:
        return False
    return all(
        t.is_cuda and t.dtype == torch.float32 for t in tensors if t.is_floating_point()
    )


def rotate_to_local_cute(
    x: Tensor,
    src: Tensor,
    wigner: Tensor,
    coeff_index: Tensor,
    dim_full: int,
) -> Tensor:
    """
    Fused ``global -> local reduced`` rotation (CuTe fast path with eager fallback).

    Parameters
    ----------
    x
        Node features with shape ``(N, D, C)``.
    src
        Source-node indices with shape ``(E,)``.
    wigner
        Packed Wigner-D matrices with shape ``(E, Dw, Dw)`` (``Dw >= dim_full``).
    coeff_index
        Reduced-layout row indices with shape ``(Dm,)``.
    dim_full
        Full packed SO(3) dimension ``D``.

    Returns
    -------
    Tensor
        Rotated reduced-layout edge features with shape ``(E, Dm, C)``.

    Notes
    -----
    Experimental path that is not used in production. See the module docstring
    for the benchmark conclusion and why the Triton kernels were chosen instead.
    """
    if _cute_usable(x, wigner) and src.numel() > 0:
        return torch.ops.sezm_cute.rotate_to_local(
            x, src, wigner, coeff_index, int(dim_full)
        )
    return _rotate_to_local_eager(x, src, wigner, coeff_index, dim_full)


def rotate_back_cute(
    x_local: Tensor,
    wigner: Tensor,
    coeff_index: Tensor,
    dim_full: int,
) -> Tensor:
    """
    Fused ``local reduced -> global`` rotation (CuTe fast path with eager fallback).

    Parameters
    ----------
    x_local
        Reduced-layout edge features with shape ``(E, Dm, C)``.
    wigner
        Packed Wigner-D matrices with shape ``(E, Dw, Dw)`` (``Dw >= dim_full``).
    coeff_index
        Reduced-layout column indices with shape ``(Dm,)``.
    dim_full
        Full packed SO(3) dimension ``D``.

    Returns
    -------
    Tensor
        Lifted global-layout edge features with shape ``(E, D, C)``.

    Notes
    -----
    Experimental path that is not used in production. See the module docstring
    for the benchmark conclusion and why the Triton kernels were chosen instead.
    """
    if _cute_usable(x_local, wigner) and x_local.shape[0] > 0:
        return torch.ops.sezm_cute.rotate_back(
            x_local, wigner, coeff_index, int(dim_full)
        )
    return _rotate_back_eager(x_local, wigner, coeff_index, dim_full)
