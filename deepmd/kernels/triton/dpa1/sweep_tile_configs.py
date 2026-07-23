# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201
r"""Sweep the launch configuration of a DPA1 fused environment convolution.

Both fused kernels have a two-parameter launch configuration resolved per
``(ng, H1)`` by :mod:`.tile_configs`: ``se_conv`` (node-parallel) is keyed by
the per-neighbor block width ``BLOCK_N`` and a warp count; ``edge_conv``
(edge-parallel) by the per-block edge count ``BLOCK_E`` and a warp count. This
module measures the candidate configurations for one channel width on synthetic
tensors at a production count, validating each against the eager reference
before timing, and reports the fastest forward+backward configuration that
stays spill-free.

The forward is memory-bound and tolerant of the configuration; the backward
holds a ``(BLOCK, ng)`` block live and collapses if it spills, so the sweep
scores forward+backward jointly.

Usage
-----
::

    python -m deepmd.kernels.triton.dpa1.sweep_tile_configs \\
        --kind {conv,edge} --ng NG --h1 H1 [--device cuda:0]

The printed ``(ng, h1): (BLOCK, num_warps)`` line is appended, under the device
name from ``torch.cuda.get_device_name``, to the relevant built-in table in
:mod:`.tile_configs` (``_CONV_BUILTIN`` / ``_EDGE_BUILTIN``).

Regeneration note
-----------------
Any change to a kernel body invalidates its existing table entries; rerun the
sweep for every covered ``(ng, H1)`` and refresh the table.
"""

from __future__ import (
    annotations,
)

import argparse
import itertools
from typing import (
    TYPE_CHECKING,
)

import torch
from torch import (
    Tensor,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )

from deepmd.kernels.triton.dpa1.edge_conv import (
    _edge_conv_bwd_impl,
    _edge_conv_fwd_impl,
    _edge_conv_reference,
)
from deepmd.kernels.triton.dpa1.se_conv import (
    _se_conv_bwd_impl,
    _se_conv_fwd_impl,
    _se_conv_reference,
)
from deepmd.kernels.triton.dpa1.tile_configs import (
    EDGE_DEFAULT_CONFIG,
)

# BLOCK_N candidates are powers of two; small values protect the backward
# register footprint, large values amortize the per-node loop overhead.
_BLOCK_N_CANDIDATES = (16, 32, 64, 128)
# BLOCK_E candidates (edges per program) are powers of two; small values keep
# the per-program register tile small, large values amortize launch overhead.
_BLOCK_E_CANDIDATES = (1, 2, 4, 8, 16, 32)
_WARP_CANDIDATES = (2, 4, 8)
_REL_TOL = 1e-5
# An edge_conv candidate is only recorded when it beats the universal default by
# more than ``1 - _EDGE_WIN_RATIO`` (5%); otherwise the default is kept, so a
# level-2 launch never regresses below level 1 across edge counts.
_EDGE_WIN_RATIO = 0.95


def _make_inputs(
    nodes: int, nnei: int, ng: int, h1: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    p = 4096
    z2 = torch.randn(nodes, nnei, ng, dtype=torch.float32, device=device)
    h1t = torch.randn(nodes, nnei, h1, dtype=torch.float32, device=device)
    idt = torch.ones(ng, dtype=torch.float32, device=device)
    tt = torch.randn(p, ng, dtype=torch.float32, device=device) * 0.3
    idx = torch.randint(0, p, (nodes * nnei,), dtype=torch.int64, device=device)
    sw = torch.rand(nodes, nnei, dtype=torch.float32, device=device)
    rr = torch.randn(nodes, nnei, 4, dtype=torch.float32, device=device)
    return z2, h1t, idt, tt, idx, sw, rr


def _bench(fn: Callable[[], object], iters: int = 40, warmup: int = 15) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def sweep(
    ng: int,
    h1: int,
    nnei: int = 181,
    nodes: int = 4096,
    device: torch.device | None = None,
) -> tuple[int, int]:
    """Measure and return the fastest spill-free ``(BLOCK_N, num_warps)``.

    Parameters
    ----------
    ng : int
        Embedding channel width.
    h1 : int
        Penultimate embedding width; ``ng == 2 * h1`` is required.
    nnei : int
        Neighbor count used to size the synthetic input.
    nodes : int
        Node count used to size the synthetic input.
    device : torch.device, optional
        CUDA device; defaults to the current device.

    Returns
    -------
    tuple[int, int]
        The winning ``(BLOCK_N, num_warps)``.
    """
    if ng not in (h1, 2 * h1):
        raise ValueError(
            "se_conv sweep requires a residual last layer (ng in {h1, 2*h1})"
        )
    resnet_mult = ng // h1
    device = device or torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    # The launch configuration is memory/register bound and keyed by (ng, H1)
    # only; the activation adds a few cheap elementwise ops and does not shift
    # the optimum, so the sweep times the ``tanh`` path (act = 0).
    act = 0
    # The launch configuration is keyed by (ng, H1) and is independent of the
    # tebd-input mode; the strip gate (gated = 1) is the register-heaviest case,
    # so its optimum is a safe upper bound for concat (gated = 0).
    gated = 1
    z2, h1t, idt, tt, idx, sw, rr = _make_inputs(nodes, nnei, ng, h1, device)
    ref = _se_conv_reference(z2, h1t, idt, tt, idx, sw, rr, resnet_mult, act, gated)
    gout = torch.randn_like(ref)

    def fwd_bwd(bn: int, nw: int) -> None:
        _se_conv_fwd_impl(
            z2, h1t, idt, tt, idx, sw, rr, resnet_mult, act, gated, bn, nw
        )
        _se_conv_bwd_impl(
            gout, z2, h1t, idt, tt, idx, sw, rr, resnet_mult, act, gated, bn, nw
        )

    results: list[tuple[float, int, int]] = []
    for bn, nw in itertools.product(_BLOCK_N_CANDIDATES, _WARP_CANDIDATES):
        try:
            out = _se_conv_fwd_impl(
                z2, h1t, idt, tt, idx, sw, rr, resnet_mult, act, gated, bn, nw
            )
            rel = (out - ref).abs().max().item() / ref.abs().max().item()
            if rel > _REL_TOL:
                continue
            t = _bench(lambda: fwd_bwd(bn, nw))
        except Exception:
            continue
        results.append((t, bn, nw))
        print(f"  BLOCK_N={bn:<4} num_warps={nw}: fwd+bwd {t * 1e3:8.1f} us")
    if not results:
        raise RuntimeError("no valid configuration found")
    best_t, best_bn, best_nw = min(results, key=lambda r: r[0])
    return best_bn, best_nw


def _make_edge_inputs(
    n_edge: int, n_node: int, ng: int, h1: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    p = 4096
    z2 = torch.randn(n_edge, ng, dtype=torch.float32, device=device)
    h1t = torch.randn(n_edge, h1, dtype=torch.float32, device=device)
    idt = torch.ones(ng, dtype=torch.float32, device=device)
    tt = torch.randn(p, ng, dtype=torch.float32, device=device) * 0.3
    idx = torch.randint(0, p, (n_edge,), dtype=torch.int64, device=device)
    sw = torch.rand(n_edge, dtype=torch.float32, device=device)
    rr = torch.randn(n_edge, 4, dtype=torch.float32, device=device)
    dst = torch.randint(0, n_node, (n_edge,), dtype=torch.int64, device=device)
    edge_mask = (torch.rand(n_edge, dtype=torch.float32, device=device) > 0.05).to(
        dtype=torch.float32
    )
    return z2, h1t, idt, tt, idx, sw, rr, dst, edge_mask


def sweep_edge(
    ng: int,
    h1: int,
    n_edge: int = 524288,
    n_node: int = 4096,
    device: torch.device | None = None,
) -> tuple[int, int]:
    """Measure and return the fastest spill-free ``edge_conv`` ``(BLOCK_E, num_warps)``.

    Unlike ``se_conv`` (whose optimum is edge-count-insensitive), the
    edge-parallel launch's optimum drifts mildly with the edge count through
    occupancy, so the sweep runs at a large, throughput-bound count where the
    ratios stabilize, and only returns a tuned configuration when it beats the
    universal default (:data:`.tile_configs.EDGE_DEFAULT_CONFIG`) by more than
    ``1 - _EDGE_WIN_RATIO``; otherwise it returns the default so a subsequent
    level-2 launch never regresses below level 1.

    Parameters
    ----------
    ng : int
        Embedding channel width.
    h1 : int
        Penultimate embedding width; ``ng in {h1, 2 * h1}`` for a residual layer.
    n_edge : int
        Edge count used to size the synthetic input.
    n_node : int
        Node count (segment count) used to size the scatter target.
    device : torch.device, optional
        CUDA device; defaults to the current device.

    Returns
    -------
    tuple[int, int]
        The winning ``(BLOCK_E, num_warps)``, or the universal default when no
        candidate is a clear win.
    """
    resnet_mult = ng // h1 if ng in (h1, 2 * h1) else 0
    device = device or torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    act = 0
    # The launch configuration is keyed by (ng, H1) and is independent of the
    # tebd-input mode; the strip gate (gated = 1) is the register-heaviest case,
    # so its optimum is a safe upper bound for concat (gated = 0).
    gated = 1
    # Shared leading arguments of the three edge_conv entry points, ordered to
    # match their signatures so each call splats this tuple and appends only its
    # launch configuration.
    edge_args = (
        *_make_edge_inputs(n_edge, n_node, ng, h1, device),
        n_node,
        resnet_mult,
        act,
        gated,
    )
    ref = _edge_conv_reference(*edge_args)
    gout = torch.randn_like(ref)

    def fwd_bwd(be: int, nw: int) -> None:
        _edge_conv_fwd_impl(*edge_args, be, nw)
        _edge_conv_bwd_impl(gout, *edge_args, be, nw)

    results: dict[tuple[int, int], float] = {}
    for be, nw in itertools.product(_BLOCK_E_CANDIDATES, _WARP_CANDIDATES):
        try:
            out = _edge_conv_fwd_impl(*edge_args, be, nw)
            rel = (out - ref).abs().max().item() / ref.abs().max().item()
            if rel > _REL_TOL:
                continue
            t = _bench(lambda: fwd_bwd(be, nw))
        except Exception:
            continue
        results[(be, nw)] = t
        print(f"  BLOCK_E={be:<4} num_warps={nw}: fwd+bwd {t * 1e3:8.1f} us")
    if not results:
        raise RuntimeError("no valid configuration found")
    best = min(results, key=results.get)
    # Only accept a tuned win over the default; the default is always a candidate.
    default_t = results.get(EDGE_DEFAULT_CONFIG)
    if default_t is not None and results[best] >= default_t * _EDGE_WIN_RATIO:
        return EDGE_DEFAULT_CONFIG
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("conv", "edge"), default="conv")
    parser.add_argument("--ng", type=int, required=True)
    parser.add_argument("--h1", type=int, required=True)
    parser.add_argument("--nnei", type=int, default=181)
    parser.add_argument("--nodes", type=int, default=4096)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if args.kind == "edge":
        block, nw = sweep_edge(args.ng, args.h1, device=device)
    else:
        block, nw = sweep(args.ng, args.h1, args.nnei, args.nodes, device)
    print(f'\n"{torch.cuda.get_device_name()}": {{')
    print(f"    ({args.ng}, {args.h1}): ({block}, {nw}),")
    print("}")


if __name__ == "__main__":
    main()
