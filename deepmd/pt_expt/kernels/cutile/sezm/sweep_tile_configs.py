# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201
"""Launch-configuration sweep for the cuTile SeZM kernels.

The sweep measures every candidate tile width and occupancy of one kernel family
at a saturated edge count and registers the winner for the current process
(:func:`~..launch_config.register_launch_config`). It is the tool that produces
the entries in :data:`~..launch_config.BUILTIN_LAUNCH_CONFIGS`, and it can be run
at freeze time so that a device without built-in coverage still bakes a tuned
launch into the frozen artifact.

Two properties of the SeZM graph must be reproduced or the result is misleading,
and both were got wrong once during development:

Extended atom count
    The source of an edge is any of the ``nall`` extended atoms, not one of the
    ``nloc`` local ones. At production cell sizes ``nall`` is larger by more than
    an order of magnitude, which makes the node feature table exceed the last
    level cache and shortens the mean source segment from over a hundred edges
    to about six. A sweep run against ``nloc`` source nodes selects a tile that
    is far too wide.

Saturation
    Tiles must be compared at an edge count that fills the device. A sweep on a
    small sample selects the configuration with the lowest launch overhead
    rather than the highest throughput.
"""

from __future__ import (
    annotations,
)

import itertools
from typing import (
    TYPE_CHECKING,
)

import torch

from . import (
    flash_atten,
    force_assembly,
    so2_mixing_stack,
    so2_rotate_mix,
)
from .flash_atten import (
    build_row_ptr,
)
from .indexing import (
    SO2TileLayout,
)
from .so2_mixing_stack import (
    pack_weights,
)
from .tile_configs import (
    LaunchConfig,
    register_tile_configs,
    tile_config,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )

__all__ = ["sweep_layout"]

#: Candidate tile widths per family. Every entry is a power of two, as the tile
#: model requires, and the ranges bracket the measured optimum on Blackwell with
#: room on both sides.
_CANDIDATES: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "rotate_mix_fwd": ((32, 64, 128, 256), (0, 2, 3, 4)),
    "rotate_mix_bwd": ((4, 8, 16, 32), (0, 2, 3, 4)),
    "mixing_stack_fwd": ((16, 32, 64), (0, 1, 2, 3)),
    "mixing_stack_bwd": ((16, 32, 64), (0, 1, 2, 3)),
    "flash_fwd": ((16, 32, 64), (0, 2, 3, 4)),
    "flash_bwd": ((16, 32, 64, 128), (0, 2, 3, 4)),
    "force_assembly": ((4, 8, 16, 32), (0,)),
}


def _bench(run: Callable[[], object], iters: int = 30, warmup: int = 8) -> float:
    """Return the mean wall time of ``run`` in milliseconds."""
    for _ in range(warmup):
        run()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        run()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters


def _block_diagonal(n_edge: int, lmax: int, device: torch.device) -> torch.Tensor:
    """Return a random Wigner-D stack supported on its degree blocks."""
    dim = (lmax + 1) ** 2
    wigner = torch.zeros(n_edge, dim, dim, device=device)
    for degree in range(lmax + 1):
        lo, hi = degree * degree, (degree + 1) ** 2
        wigner[:, lo:hi, lo:hi] = torch.randn(n_edge, hi - lo, hi - lo, device=device)
    return wigner


def _probes(
    layout: SO2TileLayout,
    n_focus: int,
    n_head: int,
    n_local: int,
    n_extended: int,
    n_edge: int,
    device: torch.device,
) -> dict[str, Callable[[], object]]:
    """Build one closure per family over synthetic operands of production size."""
    cf, dim, c_wide = layout.focus_dim, layout.dim, n_focus * layout.focus_dim
    x = torch.randn(n_extended, dim, c_wide, device=device)
    # A source may be any extended atom, a destination is always a local one.
    # The two endpoints therefore see segment-length distributions that differ
    # by more than an order of magnitude, and a probe that draws both uniformly
    # selects a tile that is far too narrow for the destination reduction.
    src = torch.randint(0, n_extended, (n_edge,), device=device)
    dst = torch.arange(n_local, device=device).repeat_interleave(n_edge // n_local)[
        :n_edge
    ]
    wigner = _block_diagonal(n_edge, layout.lmax, device)
    mixer = torch.randn(n_edge, layout.kernel_size, device=device)
    channel = torch.randn(c_wide, device=device)
    src_order = torch.argsort(src)
    src_row_ptr = build_row_ptr(src.index_select(0, src_order), n_extended)
    dst_order = torch.argsort(dst)
    dst_row_ptr = build_row_ptr(dst.index_select(0, dst_order), n_local)

    activation = torch.randn(n_focus, n_edge, layout.row, device=device)
    n_row = 3 * layout.lmax + 1
    x_local = torch.randn(n_edge, n_focus, n_row, cf, device=device)
    alpha = torch.rand(n_edge, n_focus, n_head, device=device)
    rescale = tuple((torch.rand(dim) + 0.5).tolist())
    grad_node = torch.randn(n_local, dim, c_wide, device=device)
    edge_grad = torch.randn(n_edge, 3, device=device)
    # The force assembly indexes both endpoints over the extended atoms, so the
    # destination topology is rebuilt on that axis while keeping its clustering.
    dst_ext_row_ptr = build_row_ptr(dst.index_select(0, dst_order), n_extended).long()
    grad_edge = torch.randn(n_edge, n_focus, layout.row, device=device)

    width0 = layout.n_m0 * cf
    width1 = layout.n_m1 * cf
    w0 = torch.randn(layout.n_layers, n_focus, width0, width0, device=device)
    w1 = torch.randn(layout.n_layers, n_focus, width1, width1, device=device)
    gw = torch.randn(layout.n_gated, n_focus, cf, layout.lmax * cf, device=device)
    packed = pack_weights(w0, w1, gw, layout)

    return {
        "rotate_mix_fwd": lambda: so2_rotate_mix._launch_forward(
            x, src, wigner, mixer, channel, layout, n_focus
        ),
        "rotate_mix_bwd": lambda: so2_rotate_mix._launch_backward(
            activation,
            x,
            src_order,
            src_row_ptr,
            wigner,
            mixer,
            channel,
            layout,
            n_focus,
        ),
        "mixing_stack_fwd": lambda: so2_mixing_stack._launch_forward(
            activation, packed, layout
        ),
        "mixing_stack_bwd": lambda: so2_mixing_stack._launch_backward(
            activation, grad_edge, packed, layout
        ),
        "flash_fwd": lambda: flash_atten._launch_forward(
            x_local,
            wigner,
            rescale,
            alpha,
            dst_order,
            dst_row_ptr,
            layout,
            n_focus,
            n_head,
        ),
        "flash_bwd": lambda: flash_atten._launch_backward(
            grad_node, x_local, wigner, rescale, alpha, dst, layout, n_focus, n_head
        ),
        # Both endpoints of the force assembly range over the extended atoms, so
        # its segments are as short as the rotate-and-mix backward's.
        "force_assembly": lambda: force_assembly._launch_forward(
            edge_grad,
            edge_grad,
            dst_order,
            dst_ext_row_ptr,
            src_order,
            src_row_ptr.long(),
        ),
    }


def sweep_layout(
    lmax: int,
    focus_dim: int,
    n_layers: int = 3,
    n_focus: int = 1,
    n_head: int = 1,
    n_local: int = 8000,
    n_extended: int = 216000,
    n_edge: int = 1264000,
    families: tuple[str, ...] = tuple(_CANDIDATES),
    verbose: bool = True,
) -> dict[str, LaunchConfig]:
    """Sweep one block layout and register the winning configuration of each family.

    Parameters
    ----------
    lmax, focus_dim, n_layers, n_focus, n_head : int
        Block layout to tune.
    n_local, n_extended, n_edge : int
        Graph size to tune at. The defaults describe an 8000-atom periodic cell
        at the production cutoff and saturate a Blackwell-class device.
    families : tuple[str, ...]
        Families to sweep; defaults to all tunable ones.
    verbose : bool
        Print each candidate as it is measured.

    Returns
    -------
    dict[str, LaunchConfig]
        The winning configuration of each swept family, already registered.

    Notes
    -----
    A candidate that fails to compile is skipped rather than raising: the
    register allocator rejects some tile and occupancy combinations, and a sweep
    that aborted on the first such pair would report no winner at all.
    """
    device = torch.device("cuda")
    layout = SO2TileLayout(lmax=lmax, focus_dim=focus_dim, n_layers=n_layers)
    key = layout.key
    probes = _probes(layout, n_focus, n_head, n_local, n_extended, n_edge, device)
    winners: dict[str, LaunchConfig] = {}
    for family in families:
        tiles, occupancies = _CANDIDATES[family]
        baseline = tile_config(family, key)
        best: tuple[float, LaunchConfig] | None = None
        for tile, occupancy in itertools.product(tiles, occupancies):
            candidate = LaunchConfig(tile=tile, occupancy=occupancy)
            register_tile_configs(family, key, candidate)
            try:
                elapsed = _bench(probes[family])
            except Exception as error:
                if verbose:
                    print(f"{family:18s} {candidate}  rejected: {error!r:.60}")
                continue
            if verbose:
                print(f"{family:18s} {candidate}  {elapsed:8.3f} ms")
            if best is None or elapsed < best[0]:
                best = (elapsed, candidate)
        register_tile_configs(family, key, best[1] if best else baseline)
        if best:
            winners[family] = best[1]
            if verbose:
                print(f"{family:18s} -> {best[1]} at {best[0]:.3f} ms")
    return winners
