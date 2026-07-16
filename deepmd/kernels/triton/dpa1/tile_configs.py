# SPDX-License-Identifier: LGPL-3.0-or-later
"""Launch-configuration resolution for the DPA1 fused environment convolutions.

Two memory-bound kernels are configured here, each reducing to a block width
and a warp count keyed by the channel width ``ng`` and the resnet width ``H1``:

- ``se_conv`` (strip / dense, node-parallel): one program owns a node and
  streams its neighbors, so the launch is ``(BLOCK_N, num_warps)`` -- neighbors
  per block. ``BLOCK_N`` bounds the live ``(BLOCK_N, channels)`` register
  footprint of the backward pass; oversized blocks spill and collapse
  throughput, so the universal default is kept small.
- ``edge_conv`` (concat / graph, edge-parallel): one program owns a block of
  edges and scatters them into their center nodes, so the launch is
  ``(BLOCK_E, num_warps)`` -- edges per block. ``BLOCK_E`` bounds the live
  ``(BLOCK_E, channels)`` register footprint of both passes.

The optimum depends on ``(ng, H1)`` (the register footprint) but is insensitive
to the neighbor / edge count, which only sets the loop trip count or the grid
size. Table keys are therefore ``(ng, H1)``.

Level policy (see :func:`deepmd.kernels.utils.triton_infer_level`):

- Level ``1`` always returns the family default, a single shape-independent
  configuration that never spills.
- Level ``>= 2`` consults the per-GPU table and falls back to the default on any
  miss, so an unswept device or channel width can only match or beat level 1,
  never regress below it.

New devices are added by running :mod:`.sweep_tile_configs` and appending a
block to the relevant built-in table under the device name reported by
``torch.cuda.get_device_name``.
"""

from __future__ import (
    annotations,
)

import torch

# (block, num_warps): ``block`` is ``BLOCK_N`` for se_conv and ``BLOCK_E`` for
# edge_conv. A small block keeps the backward register footprint within budget.
Config = tuple[int, int]

# se_conv (node-parallel) universal default.
DEFAULT_CONFIG: Config = (16, 4)
# edge_conv (edge-parallel) universal default.
EDGE_DEFAULT_CONFIG: Config = (8, 4)

# Per-GPU built-in tables keyed by (ng, H1). Values are the fastest spill-free
# forward+backward configuration produced by the fp32 sweep in
# :mod:`.sweep_tile_configs` for that channel width.
_CONV_BUILTIN: dict[str, dict[tuple[int, int], Config]] = {
    "NVIDIA H20": {
        (32, 16): (32, 2),
        (64, 32): (16, 2),
        (128, 64): (16, 2),
        (256, 128): (16, 4),
        (100, 50): (16, 2),
        (200, 100): (16, 4),
    },
}
_EDGE_BUILTIN: dict[str, dict[tuple[int, int], Config]] = {
    "NVIDIA H20": {
        (32, 16): (8, 4),
        (64, 32): (8, 2),
        (128, 64): (8, 2),
        (256, 128): (2, 2),
        (100, 50): (8, 2),
        (200, 100): (1, 2),
        (32, 32): (8, 4),
        (64, 64): (8, 2),
        (128, 128): (1, 2),
    },
}

# Per-GPU configs swept at freeze time by :mod:`deepmd.kernels.autotune` for
# shape keys the built-in tables do not cover. Process-local: the freeze traces
# on the target GPU, so these are baked into the exported ``.pt2``; they never
# persist across processes. Same schema as the built-in tables.
_CONV_RUNTIME: dict[str, dict[tuple[int, int], Config]] = {}
_EDGE_RUNTIME: dict[str, dict[tuple[int, int], Config]] = {}


def _register(
    runtime: dict[str, dict[tuple[int, int], Config]],
    device_name: str,
    ng: int,
    h1: int,
    config: Config,
) -> None:
    runtime.setdefault(device_name, {})[(int(ng), int(h1))] = config


def _covered(
    builtin: dict[str, dict[tuple[int, int], Config]],
    runtime: dict[str, dict[tuple[int, int], Config]],
    ng: int,
    h1: int,
) -> bool:
    if not torch.cuda.is_available():
        return False
    name = torch.cuda.get_device_name()
    key = (int(ng), int(h1))
    return key in runtime.get(name, {}) or key in builtin.get(name, {})


def _resolve(
    builtin: dict[str, dict[tuple[int, int], Config]],
    runtime: dict[str, dict[tuple[int, int], Config]],
    default: Config,
    ng: int,
    h1: int,
    level: int,
) -> Config:
    if level < 2 or not torch.cuda.is_available():
        return default
    name = torch.cuda.get_device_name()
    key = (int(ng), int(h1))
    runtime_dev = runtime.get(name, {})
    if key in runtime_dev:
        return runtime_dev[key]
    return builtin.get(name, {}).get(key, default)


# --- se_conv (node-parallel) ------------------------------------------------
def register_conv_config(device_name: str, ng: int, h1: int, config: Config) -> None:
    """Register a freshly swept ``se_conv`` launch for ``(ng, h1)``.

    Used by the freeze-time autotuner so a subsequent :func:`resolve_conv_config`
    (made while tracing) bakes the tuned launch into the exported artifact.
    """
    _register(_CONV_RUNTIME, device_name, ng, h1, config)


def has_conv_config(ng: int, h1: int) -> bool:
    """Whether a tuned ``se_conv`` entry (built-in or freeze-time) covers ``(ng, h1)``."""
    return _covered(_CONV_BUILTIN, _CONV_RUNTIME, ng, h1)


def resolve_conv_config(ng: int, h1: int, level: int) -> Config:
    """Resolve the ``(BLOCK_N, num_warps)`` for a fused ``se_conv`` launch.

    Level 1 forces the universal default; level ``>= 2`` consults the
    freeze-time and per-GPU tables with fallback. ``BLOCK_N`` is a power of two.
    """
    return _resolve(_CONV_BUILTIN, _CONV_RUNTIME, DEFAULT_CONFIG, ng, h1, level)


# --- edge_conv (edge-parallel) ----------------------------------------------
def register_edge_config(device_name: str, ng: int, h1: int, config: Config) -> None:
    """Register a freshly swept ``edge_conv`` launch for ``(ng, h1)``."""
    _register(_EDGE_RUNTIME, device_name, ng, h1, config)


def has_edge_config(ng: int, h1: int) -> bool:
    """Whether a tuned ``edge_conv`` entry (built-in or freeze-time) covers ``(ng, h1)``."""
    return _covered(_EDGE_BUILTIN, _EDGE_RUNTIME, ng, h1)


def resolve_edge_config(ng: int, h1: int, level: int) -> Config:
    """Resolve the ``(BLOCK_E, num_warps)`` for a fused ``edge_conv`` launch.

    Level 1 forces the universal default; level ``>= 2`` consults the
    freeze-time and per-GPU tables with fallback. ``BLOCK_E`` is a power of two.
    """
    return _resolve(_EDGE_BUILTIN, _EDGE_RUNTIME, EDGE_DEFAULT_CONFIG, ng, h1, level)
