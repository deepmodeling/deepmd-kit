# SPDX-License-Identifier: LGPL-3.0-or-later
"""Built-in launch-configuration data for the cuTile SeZM kernels.

This module is pure data: one nested mapping per GPU model, keyed by either the
exact device name reported by :func:`torch.cuda.get_device_name` or a stable
model-name prefix. The query layer in :mod:`.tile_configs` prefers an exact
match, then the longest prefix ending at a space boundary. Devices without an
entry here fall back to the conservative default of every kernel family (correct
on any CUDA device, merely not tuned).

Entry semantics
---------------
Every per-family table maps an exact shape key to either a launch configuration
or ``None``:

- a :class:`~.tile_configs.LaunchConfig` is the winning configuration measured by
  the sweep;
- ``None`` records that the family default won the sweep;
- an absent key means the shape was never swept on this GPU, which is what
  :mod:`.sweep_tile_configs` treats as work.

Coverage is per family and per key, and partial coverage is normal: a sweep is
dominated by tile-compiler time, so a layout is often tuned for the families that
matter most to it and left on defaults elsewhere. Resolution falls back family by
family, so an entry is never required to be complete.

Key conventions and value semantics are documented in :mod:`.tile_configs`;
regeneration is documented in :mod:`.sweep_tile_configs`. Every entry below was
measured at production graph size -- 8000 local atoms, 216000 extended atoms and
1.264e6 edges -- with TF32 disabled.
"""

from __future__ import (
    annotations,
)

from .tile_configs import (
    LaunchConfig,
)

__all__ = ["BUILTIN_TILE_CONFIGS"]

#: Shape key is ``(lmax, focus_dim)`` for the SO(2) kernels and the empty tuple
#: for the two shape-independent families.
BUILTIN_TILE_CONFIGS: dict[
    str, dict[str, dict[tuple[int, ...], LaunchConfig | None]]
] = {
    "NVIDIA RTX PRO 6000 Blackwell": {
        "rotate_mix_fwd": {
            (1, 32): LaunchConfig(tile=64, occupancy=3),
            (2, 32): LaunchConfig(tile=64, occupancy=2),
            (2, 64): LaunchConfig(tile=32, occupancy=0),
            (2, 128): LaunchConfig(tile=32, occupancy=2),
            (3, 32): LaunchConfig(tile=32, occupancy=0),
            (3, 64): LaunchConfig(tile=32, occupancy=2),
        },
        "rotate_mix_bwd": {
            (1, 32): LaunchConfig(tile=8, occupancy=4),
            (2, 32): LaunchConfig(tile=8, occupancy=3),
        },
        "mixing_stack_fwd": {
            (1, 32): LaunchConfig(tile=32, occupancy=3),
            (2, 32): LaunchConfig(tile=32, occupancy=0),
        },
        "mixing_stack_bwd": {
            (1, 32): LaunchConfig(tile=32, occupancy=3),
            (2, 32): LaunchConfig(tile=32, occupancy=2),
        },
        "flash_fwd": {
            (1, 32): LaunchConfig(tile=16, occupancy=3),
            (2, 32): LaunchConfig(tile=16, occupancy=4),
        },
        "flash_bwd": {
            (1, 32): LaunchConfig(tile=32, occupancy=2),
            (2, 32): LaunchConfig(tile=16, occupancy=2),
        },
        "force_assembly": {(): LaunchConfig(tile=16)},
    },
}
