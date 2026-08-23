# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Launch-configuration lookup for the shape-tuned SeZM Triton kernels.

Configurations are resolved through two layers:

1. *Built-in tables* (:mod:`.tile_config_data`), keyed by an exact GPU name or
   a stable model-name prefix reported by :func:`torch.cuda.get_device_name`.
   Exact names take precedence, followed by the longest prefix ending at a
   space boundary, so edition suffixes can share one architecture table
   without confusing names such as H20 and H200.  A device without a built-in
   table resolves every key to the conservative default of its kernel family
   (correct on any CUDA device, merely not tuned).
2. *Runtime registrations* (:func:`register_tile_configs`), which take
   precedence over the built-in tables in the current process.  The freeze
   auto-tuner (:func:`.sweep_tile_configs.tune_missing_configs`) sweeps the
   shape keys of the checkpoint being frozen on the local GPU and registers
   the winners here, so the traced ``.pt2`` bakes tuned launches even on
   devices without built-in coverage.  Registrations are process-local by
   design: a ``.pt2`` is not portable across GPU models, so its tuning does
   not need to be either.

Two shape-key conventions are used:

- ``(focus_dim, lmax)`` for kernels whose register pressure is per focus
  stream (the value-path pointwise kernels and both stack GEMM paths);
  entries are valid for any focus count ``F``.
- ``(C_wide, lmax)`` with ``C_wide = n_focus * focus_dim`` for kernels that
  vectorize over the full hidden width (the rotate+mix kernels and the
  edge-block flash-attention backward).

Fallback behaviour on an unresolved key depends on the family:

- ``gate`` / ``recompute`` / ``point`` fall back to a spill-safe
  configuration of the same kernel and ``rotate_mix_fwd`` to the upstream
  default.  Their tile choices change only the launch schedule, and the
  conservative end degrades gracefully.
- ``flash_bwd_block`` and ``rotate_mix_bwd_block`` are win lists: a key
  resolves to a configuration only where the edge-block schedule beat the
  per-edge kernel by at least 3% in the sweep, and anything else keeps the
  per-edge kernel.  The edge-block schedule wins on narrow hidden widths
  (large per-edge cross-lane reduction overhead) and loses badly on wide
  ones (register-tile pressure), so the win list is the routing criterion,
  not merely a tuning hint.
- ``flash_bwd_edge`` pins the production per-edge launch instead of relying
  on Triton's first-call autotuner.  AOTInductor freezes the launch selected
  by its tiny trace sample, whose optimum can differ sharply from a saturated
  edge list; an unresolved key retains the upstream autotuner.
- ``point_recompute`` is a win list for folding gate-sigmoid recomputation
  into the backward pointwise kernel.  An unresolved key retains the two-
  kernel schedule because the fused kernel increases register pressure on
  some shapes.
- ``stack_m0_gate`` is a win list for folding the forward gate into the
  fp32 ``m = 0`` GEMM.  The fused program retains all degree-group outputs in
  registers, so it is used only where the eliminated memory round trip
  outweighs the higher register footprint.
- ``stack_fp32`` falls back to a conservative launch tuple shared by all
  three stack GEMM kernels.  Swept entries select independent tiles for the
  ``m = 0`` forward, ``|m| = 1`` forward and combined backward kernels; this
  separation matters on devices whose IEEE-fp32 throughput balance differs
  from H20.  Every swept tuple is checked through the whole stack against the
  fallback before registration because changing ``BLOCK_K`` may regroup fp32
  partial sums.
- ``stack_fp16x3`` is a validated win list: every entry passed the fp64
  exactness sweep for the exact kernel binary it launches, and an
  unresolved key keeps the fp32 mixing stack.  These entries are
  load-bearing for correctness -- some ``(num_warps, num_stages)``
  combinations of the three-``tl.dot`` k-loop are miscompiled by the Triton
  software pipeliner into silent NaN rows at production edge counts, and
  the affected set shifts with any change to the kernel body.  Never edit
  an fp16x3 entry by hand; always regenerate it through the sweep.

Register-pressure guidance
--------------------------
The winning ``BLOCK_M`` of the pointwise kernels shrinks monotonically as
the register-pressure product ``lmax * next_power_of_2(Cf)`` grows (wide
64-row tiles for ``Cf = 32``, narrow 8..16-row tiles at ``Cf >= 96``); a
candidate on the wrong side of the spill point can be an order of magnitude
slower, which is why the tables are exact-keyed rather than heuristic.  The
same product governs the edge-block backward kernels through their
``(BLOCK_E, C_wide)`` register tiles.

Wide-channel regime
-------------------
At ``Cf >= GATE_BMM_MIN_FOCUS_DIM`` the per-group ``CP x CP`` register dot of
the gate forward/backward spills regardless of the tile choice (a padded 96
behaves like 128).  In that regime the sigmoid projection and the gate-logit
contraction run as cuBLAS batched matmuls and the Triton kernels keep only
the pointwise work, so ``gate`` entries for those keys were swept with the
projection disabled and ``recompute`` entries do not exist.
"""

from __future__ import (
    annotations,
)

import functools

import torch

from .tile_config_data import (
    BUILTIN_TILE_CONFIGS,
)

__all__ = [
    "GATE_BMM_MIN_FOCUS_DIM",
    "TILE_CONFIG_FAMILIES",
    "flash_bwd_block_config",
    "flash_bwd_edge_config",
    "gate_config",
    "gated_second_order_config",
    "has_tile_config",
    "point_config",
    "point_recompute_config",
    "point_train_config",
    "recompute_config",
    "register_tile_configs",
    "rotate_mix_bwd_block_config",
    "rotate_mix_fwd_config",
    "stack_fp16x3_configs",
    "stack_fp32_configs",
    "stack_m0_gate_config",
]

# Per-focus channel width at or above which the gate sigmoid projection and
# the gate-logit contraction are delegated to cuBLAS batched matmuls.
GATE_BMM_MIN_FOCUS_DIM = 96

TILE_CONFIG_FAMILIES = (
    "gate",
    "recompute",
    "point",
    "point_train",
    "point_recompute",
    "gated_second_order",
    "rotate_mix_fwd",
    "flash_bwd_block",
    "flash_bwd_edge",
    "rotate_mix_bwd_block",
    "stack_fp32",
    "stack_m0_gate",
    "stack_fp16x3",
)

_POINTWISE_FALLBACK = (16, 8, 2)
_ROTATE_MIX_FWD_DEFAULT = (2, 2)
_STACK_GEMM_DEFAULT = (64, 64, 32, 4, 2)
_STACK_FP32_DEFAULT = (_STACK_GEMM_DEFAULT,) * 3

# Runtime registrations, highest lookup precedence.  Populated by the freeze
# auto-tuner and by manual sweep runs in the same process.
_RUNTIME: dict[str, dict[tuple[int, int], tuple | None]] = {
    family: {} for family in TILE_CONFIG_FAMILIES
}


def _match_builtin_tables(
    device_name: str,
) -> dict[str, dict[tuple[int, int], tuple | None]]:
    """Return the exact or longest whole-token-prefix table for a GPU name."""
    if device_name in BUILTIN_TILE_CONFIGS:
        return BUILTIN_TILE_CONFIGS[device_name]
    prefixes = [
        model_name
        for model_name in BUILTIN_TILE_CONFIGS
        if device_name.startswith(f"{model_name} ")
    ]
    if not prefixes:
        return {}
    return BUILTIN_TILE_CONFIGS[max(prefixes, key=len)]


@functools.cache
def _builtin_tables_for_device(
    device_index: int,
) -> dict[str, dict[tuple[int, int], tuple | None]]:
    """Resolve and cache the built-in tables for one CUDA device index."""
    return _match_builtin_tables(torch.cuda.get_device_name(device_index))


def _builtin_tables() -> dict[str, dict[tuple[int, int], tuple | None]]:
    """Return the built-in tables of the running GPU (empty when untuned)."""
    if not torch.cuda.is_available():
        return {}
    return _builtin_tables_for_device(torch.cuda.current_device())


def _lookup(family: str, key: tuple[int, int]) -> tuple | None:
    """Resolve ``key`` through the runtime and built-in layers.

    A ``None`` result folds together an explicit ``None`` entry (the sweep
    ran and the family default is the measured optimum) and an absent key
    (never swept on this GPU): the caller behaves identically in both cases.
    """
    runtime = _RUNTIME[family]
    if key in runtime:
        return runtime[key]
    return _builtin_tables().get(family, {}).get(key)


def _runtime_tile_configs(family: str) -> dict[tuple[int, int], tuple | None]:
    """Return the mutable runtime table of ``family``.

    Internal accessor for the sweep (which must restore pre-sweep entries
    when a run aborts) and for tests; regular callers register through
    :func:`register_tile_configs` only.
    """
    if family not in TILE_CONFIG_FAMILIES:
        raise ValueError(
            f"unknown tile-config family {family!r}; expected one of "
            f"{TILE_CONFIG_FAMILIES}"
        )
    return _RUNTIME[family]


def register_tile_configs(
    family: str, entries: dict[tuple[int, int], tuple | None]
) -> None:
    """Register swept launch configurations for the current process.

    Registered entries take precedence over the built-in tables and feed the
    same lookup functions, so a registration made before model construction
    is picked up by the construction-time operator bindings and baked into
    any subsequent trace.

    Parameters
    ----------
    family : str
        One of :data:`TILE_CONFIG_FAMILIES`.
    entries : dict[tuple[int, int], tuple or None]
        Shape keys mapped to the winning configuration, or to ``None`` to
        record that the sweep ran and the family default is the measured
        optimum for that key.

    Raises
    ------
    ValueError
        If ``family`` is not a known kernel family.
    """
    if family not in TILE_CONFIG_FAMILIES:
        raise ValueError(
            f"unknown tile-config family {family!r}; expected one of "
            f"{TILE_CONFIG_FAMILIES}"
        )
    _RUNTIME[family].update(entries)


def has_tile_config(family: str, key: tuple[int, int]) -> bool:
    """Return whether ``key`` has been swept on this GPU.

    An explicit ``None`` entry counts as swept because it records a measured
    default win.  Only keys absent from both the runtime and built-in layers
    report ``False``.  The freeze auto-tuner uses this to decide which keys
    still need work.
    """
    if family not in TILE_CONFIG_FAMILIES:
        raise ValueError(
            f"unknown tile-config family {family!r}; expected one of "
            f"{TILE_CONFIG_FAMILIES}"
        )
    return key in _RUNTIME[family] or key in _builtin_tables().get(family, {})


def gate_config(focus_dim: int, lmax: int) -> tuple[int, int, int]:
    """Return ``(BLOCK_M, num_warps, num_stages)`` for the gate forward kernel.

    Parameters
    ----------
    focus_dim : int
        Per-focus channel width ``Cf``.
    lmax : int
        Maximum spherical harmonic degree.

    Returns
    -------
    tuple[int, int, int]
        The swept launch configuration, or the spill-safe fallback for
        unresolved keys.
    """
    return _lookup("gate", (focus_dim, lmax)) or _POINTWISE_FALLBACK


def recompute_config(focus_dim: int, lmax: int) -> tuple[int, int, int]:
    """Return ``(BLOCK_M, num_warps, num_stages)`` for the gate recompute kernel.

    Parameters
    ----------
    focus_dim : int
        Per-focus channel width ``Cf``.
    lmax : int
        Maximum spherical harmonic degree.

    Returns
    -------
    tuple[int, int, int]
        The swept launch configuration, or the spill-safe fallback for
        unresolved keys.
    """
    return _lookup("recompute", (focus_dim, lmax)) or _POINTWISE_FALLBACK


def point_config(focus_dim: int, lmax: int) -> tuple[int, int, int]:
    """Return ``(BLOCK_M, num_warps, num_stages)`` for the backward pointwise kernel.

    Parameters
    ----------
    focus_dim : int
        Per-focus channel width ``Cf``.
    lmax : int
        Maximum spherical harmonic degree.

    Returns
    -------
    tuple[int, int, int]
        The swept launch configuration, or the spill-safe fallback for
        unresolved keys.
    """
    return _lookup("point", (focus_dim, lmax)) or _POINTWISE_FALLBACK


def point_train_config(focus_dim: int, lmax: int) -> tuple[int, int, int]:
    """Return the backward pointwise launch for the training variant.

    Training launches the same kernel with the layer-input recovery and the
    gate-logit store enabled, which raises register pressure and write
    traffic; its winning tile can differ from the inference entry by several
    times, so the variant carries its own table. Unresolved keys fall back to
    the inference entry, which is correct on any shape.

    Parameters
    ----------
    focus_dim : int
        Per-focus channel width ``Cf``.
    lmax : int
        Maximum spherical harmonic degree.

    Returns
    -------
    tuple[int, int, int]
        The swept ``(BLOCK_M, num_warps, num_stages)`` launch configuration.
    """
    return _lookup("point_train", (focus_dim, lmax)) or point_config(focus_dim, lmax)


def gated_second_order_config(focus_dim: int, lmax: int) -> tuple[int, int, int]:
    """Return ``(BLOCK_M, num_warps, num_stages)`` for the gated second order.

    The kernel differentiates one gated layer's backward; like the other
    pointwise kernels its winning tile shrinks as ``lmax * Cf`` grows, and a
    tile on the wrong side of the spill point costs close to an order of
    magnitude, so unresolved keys take the spill-safe pointwise fallback.

    Parameters
    ----------
    focus_dim : int
        Per-focus channel width ``Cf``.
    lmax : int
        Maximum spherical harmonic degree.

    Returns
    -------
    tuple[int, int, int]
        The swept launch configuration, or the spill-safe fallback for
        unresolved keys.
    """
    return _lookup("gated_second_order", (focus_dim, lmax)) or _POINTWISE_FALLBACK


def point_recompute_config(focus_dim: int, lmax: int) -> tuple[int, int, int] | None:
    """Return the fused recompute-point configuration, or ``None``.

    Parameters
    ----------
    focus_dim : int
        Per-focus channel width ``Cf``.
    lmax : int
        Maximum spherical harmonic degree.

    Returns
    -------
    tuple[int, int, int] or None
        ``(BLOCK_M, num_warps, num_stages)`` for a measured fused-schedule
        win.  ``None`` retains separate sigmoid recompute and pointwise
        kernels.
    """
    return _lookup("point_recompute", (focus_dim, lmax))


def rotate_mix_fwd_config(c_wide: int, lmax: int) -> tuple[int, int]:
    """Return ``(num_warps, num_stages)`` for the rotate+mix forward kernel.

    Parameters
    ----------
    c_wide : int
        Full hidden width ``n_focus * focus_dim``.
    lmax : int
        Maximum spherical harmonic degree.

    Returns
    -------
    tuple[int, int]
        The swept launch configuration, or the upstream default ``(2, 2)``
        for unresolved keys.
    """
    return _lookup("rotate_mix_fwd", (c_wide, lmax)) or _ROTATE_MIX_FWD_DEFAULT


def flash_bwd_block_config(c_wide: int, lmax: int) -> tuple[int, int, int] | None:
    """Return the edge-block flash-attention backward config, or ``None``.

    Parameters
    ----------
    c_wide : int
        Full hidden width ``n_focus * focus_dim``.
    lmax : int
        Maximum spherical harmonic degree.

    Returns
    -------
    tuple[int, int, int] or None
        ``(BLOCK_E, num_warps, num_stages)`` when the edge-block schedule won
        the sweep for this key; ``None`` keeps the per-edge kernel.
    """
    return _lookup("flash_bwd_block", (c_wide, lmax))


def flash_bwd_edge_config(c_wide: int, lmax: int) -> tuple[int, int] | None:
    """Return the production per-edge flash-backward launch, or ``None``.

    Parameters
    ----------
    c_wide : int
        Full hidden width ``n_focus * focus_dim``.
    lmax : int
        Maximum spherical harmonic degree.

    Returns
    -------
    tuple[int, int] or None
        ``(num_warps, num_stages)`` measured at a saturated edge count.
        ``None`` retains Triton's first-call autotuner on uncovered GPUs.
    """
    return _lookup("flash_bwd_edge", (c_wide, lmax))


def rotate_mix_bwd_block_config(c_wide: int, lmax: int) -> tuple[int, int, int] | None:
    """Return the edge-block rotate+mix backward config, or ``None``.

    Parameters
    ----------
    c_wide : int
        Full hidden width ``n_focus * focus_dim``.
    lmax : int
        Maximum spherical harmonic degree.

    Returns
    -------
    tuple[int, int, int] or None
        ``(BLOCK_E, num_warps, num_stages)`` when the edge-block schedule won
        the sweep for this key; ``None`` keeps the per-edge kernel.
    """
    return _lookup("rotate_mix_bwd_block", (c_wide, lmax))


def stack_fp32_configs(
    focus_dim: int, lmax: int
) -> tuple[
    tuple[int, int, int, int, int],
    tuple[int, int, int, int, int],
    tuple[int, int, int, int, int],
]:
    """Return the three IEEE-fp32 stack GEMM launch configurations.

    Parameters
    ----------
    focus_dim : int
        Per-focus channel width ``Cf``.
    lmax : int
        Maximum spherical harmonic degree.

    Returns
    -------
    tuple
        Three ``(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)``
        configurations in the order (forward ``m = 0``, forward
        ``|m| = 1``, combined backward).  Unresolved keys use the
        conservative configuration measured on H20.
    """
    return _lookup("stack_fp32", (focus_dim, lmax)) or _STACK_FP32_DEFAULT


def stack_fp16x3_configs(
    focus_dim: int, lmax: int
) -> (
    tuple[
        tuple[int, int, int, int, int],
        tuple[int, int, int, int, int],
        tuple[int, int, int, int, int],
        tuple[int, int, int, int, int],
    ]
    | None
):
    """Return the validated fp16x3 stack GEMM configs, or ``None``.

    Parameters
    ----------
    focus_dim : int
        Per-focus channel width ``Cf``.
    lmax : int
        Maximum spherical harmonic degree.

    Returns
    -------
    tuple or None
        The four ``(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)``
        configurations in the order (forward m0, forward |m|=1, backward m0,
        backward |m|=1) when the key passed the fp64 validation sweep;
        ``None`` keeps the fp32 mixing stack.  There is deliberately no
        fallback configuration: an unvalidated configuration may be
        miscompiled into silent NaN (see the module docstring).
    """
    return _lookup("stack_fp16x3", (focus_dim, lmax))


def stack_m0_gate_config(focus_dim: int, lmax: int) -> tuple[int, int, int, int] | None:
    """Return the fused fp32 m0-GEMM + gate launch, or ``None``.

    Parameters
    ----------
    focus_dim : int
        Per-focus channel width ``Cf``.
    lmax : int
        Maximum spherical harmonic degree.

    Returns
    -------
    tuple[int, int, int, int] or None
        ``(BLOCK_M, BLOCK_K, num_warps, num_stages)`` for a measured whole-
        gate win.  ``None`` retains the separate GEMM and gate kernels.
    """
    return _lookup("stack_m0_gate", (focus_dim, lmax))
