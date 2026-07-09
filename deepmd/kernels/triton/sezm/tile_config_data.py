# SPDX-License-Identifier: LGPL-3.0-or-later
"""Built-in launch-configuration data for the shape-tuned SeZM Triton kernels.

This module is pure data: one nested mapping per GPU model, keyed by the
exact device name reported by :func:`torch.cuda.get_device_name`.  The
query layer in :mod:`.tile_configs` selects the sub-mapping of the running
GPU and resolves individual keys; devices without an entry here fall back
to the conservative defaults of every kernel family (correct on any CUDA
device, merely not tuned).

Entry semantics
---------------
Every per-family table maps an exact shape key to either a launch
configuration tuple or ``None``:

- a tuple is the winning configuration measured by the sweep;
- ``None`` records that the sweep ran and the tuned kernel did **not** beat
  its baseline for this key (win-list families) or that the default
  configuration itself won (default-keyed families) -- the fallback is the
  measured optimum, not a guess;
- an absent key means the shape was never swept on this GPU.  The freeze
  auto-tuner (:func:`.sweep_tile_configs.tune_missing_configs`) treats only
  absent keys as work.

Key conventions and value layouts are documented in :mod:`.tile_configs`;
regeneration is documented in :mod:`.sweep_tile_configs`.  All entries
below were swept at production edge counts (3e5 to 6.5e5 edges) with the
``(C_wide, lmax)``-keyed families measured at ``n_focus = 2``.
"""

from __future__ import (
    annotations,
)

__all__ = ["BUILTIN_TILE_CONFIGS"]

# fmt: off
BUILTIN_TILE_CONFIGS: dict[
    str, dict[str, dict[tuple[int, int], tuple | None]]
] = {
    "NVIDIA H20": {
        # (Cf, lmax) -> (BLOCK_M, num_warps, num_stages)
        "gate": {
            (32, 1): (32, 4, 2),
            (32, 2): (64, 4, 2),
            (32, 3): (64, 4, 2),
            (32, 4): (64, 4, 1),
            (32, 5): (64, 4, 1),
            (32, 6): (64, 4, 1),
            (64, 1): (32, 16, 2),
            (64, 2): (64, 8, 1),
            (64, 3): (64, 8, 2),
            (64, 4): (64, 8, 1),
            (64, 5): (16, 8, 2),
            (64, 6): (16, 8, 2),
            (96, 1): (8, 4, 2),
            (96, 2): (16, 8, 1),
            (96, 3): (8, 8, 2),
            (96, 4): (8, 8, 2),
            (96, 5): (8, 8, 1),
            (96, 6): (8, 8, 1),
            (128, 1): (16, 16, 1),
            (128, 2): (16, 16, 1),
            (128, 3): (32, 16, 1),
            (128, 4): (16, 16, 1),
            (128, 5): (16, 16, 1),
            (128, 6): (16, 16, 2),
        },
        # (Cf, lmax) -> (BLOCK_M, num_warps, num_stages); keys with
        # Cf >= GATE_BMM_MIN_FOCUS_DIM are structurally absent (the gate
        # projection runs as a cuBLAS bmm there and the recompute kernel is
        # never launched).
        "recompute": {
            (32, 1): (64, 4, 1),
            (32, 2): (32, 4, 1),
            (32, 3): (64, 4, 2),
            (32, 4): (32, 4, 1),
            (32, 5): (32, 4, 1),
            (32, 6): (32, 4, 2),
            (64, 1): (32, 4, 2),
            (64, 2): (64, 8, 2),
            (64, 3): (64, 8, 1),
            (64, 4): (64, 8, 2),
            (64, 5): (64, 8, 2),
            (64, 6): (16, 8, 1),
        },
        # (Cf, lmax) -> (BLOCK_M, num_warps, num_stages)
        "point": {
            (32, 1): (64, 8, 1),
            (32, 2): (16, 4, 1),
            (32, 3): (64, 8, 2),
            (32, 4): (16, 4, 1),
            (32, 5): (16, 4, 2),
            (32, 6): (16, 4, 2),
            (64, 1): (16, 4, 1),
            (64, 2): (16, 8, 1),
            (64, 3): (32, 8, 2),
            (64, 4): (32, 8, 2),
            (64, 5): (16, 8, 2),
            (64, 6): (16, 8, 1),
            (96, 1): (8, 4, 2),
            (96, 2): (8, 8, 2),
            (96, 3): (8, 8, 2),
            (96, 4): (8, 8, 2),
            (96, 5): (8, 8, 2),
            (96, 6): (8, 8, 1),
            (128, 1): (8, 8, 2),
            (128, 2): (8, 8, 2),
            (128, 3): (8, 8, 1),
            (128, 4): (8, 8, 2),
            (128, 5): (8, 8, 1),
            (128, 6): (8, 8, 1),
        },
        # (C_wide, lmax) -> (num_warps, num_stages); None records keys where
        # the upstream default (2, 2) itself won the sweep.
        "rotate_mix_fwd": {
            (64, 1): (1, 2),
            (64, 2): (1, 2),
            (64, 3): (1, 2),
            (64, 4): (1, 2),
            (64, 5): (1, 2),
            (64, 6): (1, 2),
            (128, 1): (1, 2),
            (128, 2): (1, 2),
            (128, 3): (1, 2),
            (128, 4): (2, 1),
            (128, 5): None,
            (128, 6): (1, 2),
            (192, 1): (1, 1),
            (192, 2): None,
            (192, 3): (2, 1),
            (192, 4): (1, 2),
            (192, 5): None,
            (192, 6): (1, 2),
            (256, 1): (1, 1),
            (256, 2): None,
            (256, 3): (1, 1),
            (256, 4): (1, 2),
            (256, 5): (4, 1),
            (256, 6): (1, 2),
        },
        # (C_wide, lmax) -> (BLOCK_E, num_warps, num_stages); win list
        # against the per-edge kernel, None keeps the per-edge kernel.
        "flash_bwd_block": {
            (64, 1): (4, 2, 1),
            (64, 2): (4, 2, 1),
            (64, 3): (4, 2, 2),
            (64, 4): (4, 2, 2),
            (64, 5): (4, 2, 1),
            (64, 6): (4, 2, 1),
            (128, 1): None,
            (128, 2): (2, 2, 1),
            (128, 3): None,
            (128, 4): None,
            (128, 5): (2, 2, 1),
            (128, 6): None,
            (192, 1): None,
            (192, 2): None,
            (192, 3): None,
            (192, 4): None,
            (192, 5): None,
            (192, 6): None,
            (256, 1): None,
            (256, 2): None,
            (256, 3): None,
            (256, 4): None,
            (256, 5): None,
            (256, 6): None,
        },
        # (C_wide, lmax) -> (BLOCK_E, num_warps, num_stages); win list
        # against the per-edge kernel, None keeps the per-edge kernel.
        "rotate_mix_bwd_block": {
            (64, 1): (8, 2, 1),
            (64, 2): (8, 4, 1),
            (64, 3): (4, 2, 2),
            (64, 4): (4, 2, 1),
            (64, 5): (4, 2, 2),
            (64, 6): (4, 2, 1),
            (128, 1): None,
            (128, 2): None,
            (128, 3): None,
            (128, 4): (4, 4, 1),
            (128, 5): (2, 2, 1),
            (128, 6): (2, 2, 1),
            (192, 1): None,
            (192, 2): None,
            (192, 3): None,
            (192, 4): None,
            (192, 5): None,
            (192, 6): None,
            (256, 1): None,
            (256, 2): None,
            (256, 3): None,
            (256, 4): None,
            (256, 5): None,
            (256, 6): None,
        },
        # (Cf, lmax) -> four (BLOCK_M, BLOCK_N, BLOCK_K, num_warps,
        # num_stages) GEMM configurations in the order (forward m0,
        # forward |m|=1, backward m0, backward |m|=1).  Every tuple entry
        # passed the fp64 exactness sweep; None would keep the fp32 stack.
        "stack_fp16x3": {
            (32, 1): ((128, 64, 64, 4, 1), (64, 64, 32, 4, 1), (128, 64, 32, 8, 1), (64, 64, 64, 4, 1)),
            (32, 2): ((64, 64, 32, 4, 3), (64, 64, 32, 4, 3), (64, 64, 32, 4, 3), (64, 64, 32, 4, 3)),
            (32, 3): ((64, 64, 32, 4, 3), (64, 64, 32, 4, 1), (64, 64, 32, 4, 3), (128, 64, 32, 8, 1)),
            (32, 4): ((64, 64, 32, 4, 1), (64, 64, 32, 4, 1), (64, 64, 32, 4, 1), (64, 64, 32, 4, 1)),
            (32, 5): ((128, 64, 32, 8, 1), (64, 64, 32, 4, 1), (64, 64, 64, 4, 1), (64, 64, 64, 4, 1)),
            (32, 6): ((64, 64, 32, 4, 1), (64, 64, 32, 4, 1), (64, 64, 32, 4, 1), (64, 128, 64, 4, 1)),
            (64, 1): ((64, 64, 32, 4, 3), (64, 64, 32, 4, 3), (64, 64, 32, 4, 3), (64, 64, 32, 4, 3)),
            (64, 2): ((128, 64, 32, 8, 1), (64, 64, 32, 4, 1), (128, 64, 32, 8, 1), (64, 64, 32, 4, 1)),
            (64, 3): ((64, 64, 32, 4, 1), (64, 64, 32, 4, 1), (64, 64, 32, 4, 1), (64, 128, 64, 4, 1)),
            (64, 4): ((64, 64, 32, 4, 1), (64, 64, 32, 4, 1), (64, 64, 32, 4, 1), (64, 64, 32, 4, 1)),
            (64, 5): ((64, 128, 64, 4, 1), (64, 64, 32, 4, 1), (64, 128, 64, 4, 1), (64, 128, 64, 4, 1)),
            (64, 6): ((64, 64, 32, 4, 1), (64, 64, 32, 4, 1), (64, 64, 32, 4, 1), (64, 128, 64, 4, 1)),
            (96, 1): ((128, 64, 32, 8, 1), (64, 64, 32, 4, 1), (128, 64, 32, 8, 1), (128, 64, 32, 8, 1)),
            (96, 2): ((64, 64, 32, 4, 1), (64, 64, 32, 4, 1), (64, 64, 32, 4, 1), (64, 128, 64, 4, 1)),
            (96, 3): ((64, 128, 64, 4, 1), (64, 64, 32, 4, 1), (64, 128, 64, 4, 1), (64, 64, 32, 4, 1)),
            (96, 4): ((64, 64, 32, 4, 1), (64, 64, 32, 4, 1), (64, 64, 32, 4, 1), (64, 128, 64, 4, 1)),
            (96, 5): ((64, 64, 32, 4, 1), (64, 64, 32, 4, 1), (64, 64, 32, 4, 1), (64, 64, 32, 4, 1)),
            (96, 6): ((64, 64, 32, 4, 1), (64, 64, 32, 4, 1), (64, 64, 32, 4, 1), (64, 64, 32, 4, 1)),
            (128, 1): ((64, 128, 64, 4, 1), (64, 64, 32, 4, 1), (64, 128, 64, 4, 1), (64, 128, 64, 4, 1)),
            (128, 2): ((64, 128, 64, 4, 1), (64, 64, 32, 4, 1), (64, 128, 64, 4, 1), (64, 64, 32, 4, 1)),
            (128, 3): ((64, 64, 32, 4, 1), (64, 64, 32, 4, 1), (64, 64, 32, 4, 1), (64, 128, 64, 4, 1)),
            (128, 4): ((64, 128, 64, 4, 1), (64, 64, 32, 4, 1), (64, 128, 64, 4, 1), (64, 64, 32, 4, 1)),
            (128, 5): ((64, 128, 64, 4, 1), (64, 64, 32, 4, 1), (64, 128, 64, 4, 1), (64, 128, 64, 4, 1)),
            (128, 6): ((64, 128, 64, 4, 1), (64, 64, 32, 4, 1), (64, 128, 64, 4, 1), (64, 128, 64, 4, 1)),
        },
    },
}
# fmt: on
