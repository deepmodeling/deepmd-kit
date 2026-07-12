# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Environment-variable gates for the SeZM/DPA4 hardware-accelerated kernels.

This module centralizes the opt-in selectors that route inference through the
custom Triton and CuTe kernel packages. The gates are read once at model
construction time so that they become compile-time constants in the traced
(``make_fx``) graph.
"""

from __future__ import (
    annotations,
)

import os

_INFER_TRUE = ("1", "true", "yes", "on")

TRITON_INFER_LEVELS = (0, 1, 2, 3)


def triton_infer_level() -> int:
    """Return the opt-in Triton inference level from ``DP_TRITON_INFER``.

    The level is read at module construction time so that it becomes a
    compile-time constant in the traced (``make_fx``) graph. It only takes
    effect during inference; training always uses the dense reference path.

    Levels are cumulative:

    - ``0`` -- Triton disabled; every operation uses the dense reference path.
    - ``1`` -- universal kernels that need no launch-configuration table:
      block-diagonal rotation, radial degree mixing, the ``SO2Linear``
      block GEMM, Wigner monomials, flash-attention aggregation, and the
      segmented force assembly. These are either runtime-autotuned or run a
      single shape-independent configuration.
    - ``2`` -- adds kernels whose launch configuration is resolved from the
      swept ``(focus_dim, lmax)`` / ``(C_wide, lmax)`` tables in
      :mod:`.triton.tile_configs`: the fused SO(2) value path and the
      edge-block backward kernels. A key absent from a table falls back to
      the level-1 kernel (or a spill-safe configuration) for that operation,
      so unswept shapes never regress below level 1.
    - ``3`` -- adds the fp16x3 split-compensated mixing-stack GEMMs on
      tensor cores. Entries exist only for table keys whose configuration
      passed the fp64 validation sweep; unswept shapes keep the level-2 fp32
      stack. This level trades a bounded accuracy perturbation for speed
      (see :mod:`.triton.so2_stack_fp16x3`).

    Returns
    -------
    int
        The configured level in ``{0, 1, 2, 3}``.

    Raises
    ------
    ValueError
        If ``DP_TRITON_INFER`` is not an integer in ``{0, 1, 2, 3}``.
    """
    raw = os.environ.get("DP_TRITON_INFER", "0").strip()
    try:
        level = int(raw)
    except ValueError:
        raise ValueError(
            f"DP_TRITON_INFER must be an integer in {TRITON_INFER_LEVELS}, got {raw!r}"
        ) from None
    if level not in TRITON_INFER_LEVELS:
        raise ValueError(
            f"DP_TRITON_INFER must be one of {TRITON_INFER_LEVELS}, got {level}"
        )
    return level


def use_cute_infer() -> bool:
    """Return whether the opt-in CuTe inference operator is enabled.

    The flag is controlled by the ``DP_CUTE_INFER`` environment variable and is
    read at module construction time. It selects the fused CuTe SO(2) value-path
    operator (an independent path from ``DP_TRITON_INFER``) and only takes effect
    during inference; training always uses the dense reference path.

    Returns
    -------
    bool
        ``True`` when ``DP_CUTE_INFER`` is set to a truthy value.
    """
    return os.environ.get("DP_CUTE_INFER", "0").strip().lower() in _INFER_TRUE


def use_amp_infer() -> bool:
    """Return whether bf16 autocast is enabled for inference.

    The flag is controlled by the ``DP_AMP_INFER`` environment variable and is
    read at module construction time. It only affects inference when the
    descriptor's ``use_amp`` option is also enabled; training follows
    ``use_amp`` regardless of this environment variable.

    Returns
    -------
    bool
        ``True`` when ``DP_AMP_INFER`` is set to a truthy value.
    """
    return os.environ.get("DP_AMP_INFER", "0").strip().lower() in _INFER_TRUE
