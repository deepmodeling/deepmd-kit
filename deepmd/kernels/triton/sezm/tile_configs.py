# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Launch-configuration lookup for the shape-tuned SeZM Triton kernels.

Configurations are resolved through two layers:

1. *Built-in tables* (:mod:`.tile_config_data`), keyed by the exact GPU name
   reported by :func:`torch.cuda.get_device_name`.  These ship with the
   package and hold the sweep results for the GPUs the maintainers have
   tuned; a device without a built-in table resolves every key to the
   conservative default of its kernel family (correct on any CUDA device,
   merely not tuned).
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
  stream (the value-path pointwise kernels and the fp16x3 stack GEMMs);
  entries are valid for any focus count ``F``.
- ``(C_wide, lmax)`` with ``C_wide = n_focus * focus_dim`` for kernels that
  vectorize over the full hidden width (the rotate+mix kernels and the
  edge-block flash-attention backward).

Fallback behaviour on an unresolved key depends on the family:

- ``gate`` / ``recompute`` / ``point`` fall back to a spill-safe
  configuration of the same kernel and ``rotate_mix_fwd`` to the upstream
  default: tile choices never affect numerical results (they change the
  schedule, not any reduction order), and the conservative end degrades
  gracefully.
- ``flash_bwd_block`` and ``rotate_mix_bwd_block`` are win lists: a key
  resolves to a configuration only where the edge-block schedule beat the
  per-edge kernel by at least 3% in the sweep, and anything else keeps the
  per-edge kernel.  The edge-block schedule wins on narrow hidden widths
  (large per-edge cross-lane reduction overhead) and loses badly on wide
  ones (register-tile pressure), so the win list is the routing criterion,
  not merely a tuning hint.
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
    "gate_config",
    "has_tile_config",
    "point_config",
    "recompute_config",
    "register_tile_configs",
    "rotate_mix_bwd_block_config",
    "rotate_mix_fwd_config",
    "stack_fp16x3_configs",
]

# Per-focus channel width at or above which the gate sigmoid projection and
# the gate-logit contraction are delegated to cuBLAS batched matmuls.
GATE_BMM_MIN_FOCUS_DIM = 96

TILE_CONFIG_FAMILIES = (
    "gate",
    "recompute",
    "point",
    "rotate_mix_fwd",
    "flash_bwd_block",
    "rotate_mix_bwd_block",
    "stack_fp16x3",
)

_POINTWISE_FALLBACK = (16, 8, 2)
_ROTATE_MIX_FWD_DEFAULT = (2, 2)

# Runtime registrations, highest lookup precedence.  Populated by the freeze
# auto-tuner and by manual sweep runs in the same process.
_RUNTIME: dict[str, dict[tuple[int, int], tuple | None]] = {
    family: {} for family in TILE_CONFIG_FAMILIES
}


@functools.cache
def _builtin_tables() -> dict[str, dict[tuple[int, int], tuple | None]]:
    """Return the built-in tables of the running GPU (empty when untuned)."""
    if not torch.cuda.is_available():
        return {}
    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    return BUILTIN_TILE_CONFIGS.get(device_name, {})


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

    An explicit ``None`` entry counts as swept (the default configuration is
    the measured optimum); only keys absent from both the runtime and the
    built-in layer report ``False``.  The freeze auto-tuner uses this to
    decide which keys still need work.
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
