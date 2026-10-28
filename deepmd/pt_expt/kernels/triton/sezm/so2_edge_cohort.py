# SPDX-License-Identifier: LGPL-3.0-or-later
"""Experimental edge-cohort backward for frozen, narrow SO(2) inference.

Keep native GEMM column parallelism and arithmetic, but traverse all layers
of one cohort before advancing to the next cohort. Scratch scales with the
cohort, not the entire graph. The saved activations and returned input gradient
still have full edge extent. This Python runtime launcher is an online-compile
prototype; standalone AOTInductor/C++ deployment is not validated.
"""

from __future__ import (
    annotations,
)

import torch
import triton
from torch import (
    Tensor,
)
from torch.library import (
    wrap_triton,
)

from . import so2_value_path as sv

COHORT_EDGES = 8192
SINGLE_COHORT_MAX_EDGES = 16384


def supports_edge_cohort(
    grad_out: Tensor, z_all: Tensor, lmax: int, focus_dim: int, apply_alpha: bool
) -> bool:
    """Restrict opt-in dispatch to the experimentally checked Neo layout."""
    return (
        grad_out.is_cuda
        and z_all.device == grad_out.device
        and grad_out.dtype == torch.float32
        and z_all.dtype == torch.float32
        and grad_out.is_contiguous()
        and z_all.is_contiguous()
        and lmax == 3
        and focus_dim == 32
        and apply_alpha
        and grad_out.ndim == 3
        and z_all.ndim == 4
        and grad_out.shape[1:] == (2, 320)
        and z_all.shape[:2] == (2, 2)
        and z_all.shape[2] == grad_out.shape[0]
        and z_all.shape[3] == 320
    )


def cohort_input_backward_impl(
    grad_out: Tensor,
    z_all: Tensor,
    alpha: Tensor,
    w0t: Tensor,
    w1t: Tensor,
    gw: Tensor,
    gwt: Tensor,
    cohort_edges: int,
) -> Tensor:
    """Compute input gradients without retaining unrelated forward outputs.

    Kernels infer scratch focus strides from the cohort edge count. Independent
    saved/output strides retain full-graph layout, including partial cohorts.
    Only an internally owned residual is overwritten: matrix input ``gz`` and
    GEMM output never alias, and caller-owned tensors remain read-only.
    """
    if not supports_edge_cohort(grad_out, z_all, 3, 32, True):
        raise ValueError("edge cohorts require contiguous FP32 Neo inference tensors")
    if cohort_edges <= 0:
        raise ValueError("cohort_edges must be positive")
    if not all(t.is_contiguous() for t in (alpha, w0t, w1t, gw, gwt)):
        raise ValueError("edge-cohort parameters must be contiguous")
    n_gated, n_focus, n_edge, row = z_all.shape
    result = grad_out.new_empty((n_focus, n_edge, row))
    if n_edge == 0:
        return result
    cap = min(n_edge, cohort_edges)
    g = grad_out.new_empty((n_focus, cap, row))
    gz = torch.empty_like(g)
    sig = grad_out.new_empty((n_focus, cap, 96))
    bm, bn, bk, warps, stages = sv.stack_fp32_configs(32, 3)[2]
    n_tiles = triton.cdiv(128, bn) + triton.cdiv(192, bn)
    inside, point_config, recompute_config = sv._point_backward_schedule(
        32, 3, train=False
    )
    for start in range(0, n_edge, cap):
        count = min(cap, n_edge - start)
        go = grad_out[start:]
        a = alpha[start:]
        z = z_all.view(-1)[start * row :]
        grid = (triton.cdiv(count, bm) * n_tiles, n_focus)
        sv._stack_gemm_bwd_kernel[grid](
            go,
            go,
            w0t,
            w1t,
            a,
            g,
            count,
            n_gated,
            L=3,
            CF=32,
            G_EDGE_MAJOR=True,
            FOLD_ALPHA=True,
            RES_IS_GZ=True,
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
            num_warps=warps,
            num_stages=stages,
        )
        for layer in range(n_gated - 1, -1, -1):
            if not inside:
                rbm, rw, rs = recompute_config
                sv._stack_recompute_kernel[(triton.cdiv(count, rbm), n_focus)](
                    z,
                    gw,
                    sig,
                    count,
                    layer,
                    L=3,
                    CF=32,
                    BLOCK_M=rbm,
                    saved_edges=n_edge,
                    num_warps=rw,
                    num_stages=rs,
                )
            pbm, pw, ps = point_config
            sv._stack_point_bwd_kernel[(triton.cdiv(count, pbm), n_focus)](
                g,
                z,
                sig,
                gwt,
                gz,
                sig,
                g,
                g,
                count,
                layer,
                L=3,
                CF=32,
                GLOGIT_OUT=False,
                GLOGIT_STORE=False,
                RECOMPUTE_SIG=inside,
                RECOVER_INPUT=False,
                BLOCK_M=pbm,
                saved_edges=n_edge,
                num_warps=pw,
                num_stages=ps,
            )
            # Residual elements are loaded before their matching output store.
            # Distinct (edge, focus, column) tiles own disjoint residual elements.
            target = result.view(-1)[start * row :] if layer == 0 else g
            sv._stack_gemm_bwd_kernel[grid](
                gz,
                g,
                w0t,
                w1t,
                gz,
                target,
                count,
                layer,
                L=3,
                CF=32,
                G_EDGE_MAJOR=False,
                FOLD_ALPHA=False,
                RES_IS_GZ=False,
                BLOCK_M=bm,
                BLOCK_N=bn,
                BLOCK_K=bk,
                output_edges=n_edge if layer == 0 else count,
                num_warps=warps,
                num_stages=stages,
            )
    return result


# Small edge sets must not pay the opaque multi-cohort launcher overhead.
# The residual and returned result share internally owned storage; gz never
# aliases GEMM output. See the inference-only dispatcher below.
@torch.library.triton_op("sezm_triton::so2_single_cohort_input_bwd", mutates_args=())
def _single_cohort_input_backward(
    grad_out: Tensor,
    z_all: Tensor,
    alpha: Tensor,
    w0t: Tensor,
    w1t: Tensor,
    gw: Tensor,
    gwt: Tensor,
    cohort_edges: int,
) -> Tensor:
    """Compute one small cohort with compiled launchers and owned residual reuse."""
    if not supports_edge_cohort(grad_out, z_all, 3, 32, True):
        raise ValueError("edge cohorts require contiguous FP32 Neo inference tensors")
    if cohort_edges <= 0:
        raise ValueError("cohort_edges must be positive")
    if not all(t.is_contiguous() for t in (alpha, w0t, w1t, gw, gwt)):
        raise ValueError("edge-cohort parameters must be contiguous")
    n_gated, n_focus, n_edge, row = z_all.shape
    result = grad_out.new_empty((n_focus, n_edge, row))
    if n_edge == 0:
        return result
    cap = n_edge
    g = result
    gz = torch.empty_like(g)
    sig = grad_out.new_empty((n_focus, cap, 96))
    bm, bn, bk, warps, stages = sv.stack_fp32_configs(32, 3)[2]
    n_tiles = triton.cdiv(128, bn) + triton.cdiv(192, bn)
    inside, point_config, recompute_config = sv._point_backward_schedule(
        32, 3, train=False
    )
    start = 0
    count = min(cap, n_edge - start)
    go = grad_out[start:]
    a = alpha[start:]
    z = z_all.view(-1)[start * row :]
    grid = (triton.cdiv(count, bm) * n_tiles, n_focus)
    wrap_triton(sv._stack_gemm_bwd_kernel)[grid](
        go,
        go,
        w0t,
        w1t,
        a,
        g,
        count,
        n_gated,
        L=3,
        CF=32,
        G_EDGE_MAJOR=True,
        FOLD_ALPHA=True,
        RES_IS_GZ=True,
        output_edges=n_edge,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        num_warps=warps,
        num_stages=stages,
    )
    for layer in range(n_gated - 1, -1, -1):
        if not inside:
            rbm, rw, rs = recompute_config
            wrap_triton(sv._stack_recompute_kernel)[triton.cdiv(count, rbm), n_focus](
                z,
                gw,
                sig,
                count,
                layer,
                L=3,
                CF=32,
                BLOCK_M=rbm,
                saved_edges=n_edge,
                num_warps=rw,
                num_stages=rs,
            )
        pbm, pw, ps = point_config
        wrap_triton(sv._stack_point_bwd_kernel)[triton.cdiv(count, pbm), n_focus](
            g,
            z,
            sig,
            gwt,
            gz,
            sig,
            g,
            g,
            count,
            layer,
            L=3,
            CF=32,
            GLOGIT_OUT=False,
            GLOGIT_STORE=False,
            RECOMPUTE_SIG=inside,
            RECOVER_INPUT=False,
            BLOCK_M=pbm,
            saved_edges=n_edge,
            num_warps=pw,
            num_stages=ps,
        )
        target = result.view(-1)[start * row :] if layer == 0 else g
        wrap_triton(sv._stack_gemm_bwd_kernel)[grid](
            gz,
            g,
            w0t,
            w1t,
            gz,
            target,
            count,
            layer,
            L=3,
            CF=32,
            G_EDGE_MAJOR=False,
            FOLD_ALPHA=False,
            RES_IS_GZ=False,
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
            output_edges=n_edge if layer == 0 else count,
            num_warps=warps,
            num_stages=stages,
        )
    return result


@torch.library.custom_op("sezm_triton::so2_cohort_input_bwd", mutates_args=())
def _cohort_input_backward(
    grad_out: Tensor,
    z_all: Tensor,
    alpha: Tensor,
    w0t: Tensor,
    w1t: Tensor,
    gw: Tensor,
    gwt: Tensor,
    cohort_edges: int,
) -> Tensor:
    """Keep the runtime edge loop opaque without retaining x_local/u_final."""
    return cohort_input_backward_impl(
        grad_out, z_all, alpha, w0t, w1t, gw, gwt, cohort_edges
    )


@_cohort_input_backward.register_fake
def _cohort_input_backward_fake(
    grad_out: Tensor,
    z_all: Tensor,
    alpha: Tensor,
    w0t: Tensor,
    w1t: Tensor,
    gw: Tensor,
    gwt: Tensor,
    cohort_edges: int,
) -> Tensor:
    """Preserve full input-gradient shape under FakeTensor tracing."""
    return grad_out.new_empty(z_all.shape[1:])


@torch.library.triton_op("sezm_triton::so2_cohort_alpha_bwd", mutates_args=())
def _cohort_alpha_backward(grad_out: Tensor, x_local: Tensor, alpha: Tensor) -> Tensor:
    """Expose alpha's short-lived output dependency to the graph scheduler."""
    n_edge, n_focus = alpha.shape
    result = torch.empty_like(alpha)
    if sv._has_no_edges(n_edge):
        return result
    bm, warps, stages = sv.gate_config(32, 3)
    wrap_triton(sv._stack_grad_alpha_kernel)[(triton.cdiv(n_edge, bm), n_focus)](
        grad_out,
        x_local,
        alpha,
        result,
        n_edge,
        L=3,
        CF=32,
        BLOCK_M=bm,
        num_warps=warps,
        num_stages=stages,
    )
    return result


def cohort_backward(
    grad_out: Tensor,
    x_local: Tensor,
    z_all: Tensor,
    u_final: Tensor,
    alpha: Tensor,
    w0t: Tensor,
    w1t: Tensor,
    gw: Tensor,
    gwt: Tensor,
    lmax: int,
    focus_dim: int,
    apply_alpha: bool,
) -> tuple[Tensor, Tensor]:
    """First-order dispatcher; training/higher derivatives never enter here."""
    del u_final
    if not supports_edge_cohort(grad_out, z_all, lmax, focus_dim, apply_alpha):
        raise ValueError("unsupported edge-cohort layout")
    grad_alpha = _cohort_alpha_backward(grad_out, x_local, alpha)
    # make_fx/AOT may trace with a small example and retain that route for
    # larger inputs. Keep edge partitioning inside the opaque runtime operator
    # so its loop always uses the actual edge count, not the tracing example.
    grad_u0 = _cohort_input_backward(
        grad_out, z_all, alpha, w0t, w1t, gw, gwt, COHORT_EDGES
    )
    return grad_u0, grad_alpha
