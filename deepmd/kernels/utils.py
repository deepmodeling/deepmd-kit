# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Environment-variable gates for the SeZM/DPA4 hardware-accelerated kernels.

This module centralizes the opt-in selectors that route inference through the
custom Triton, CuTe and hand-written CUDA kernel packages. The gates are read
once at model construction time so that they become compile-time constants in
the traced (``make_fx``) graph.
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
    effect during inference.

    Levels are cumulative:

    - ``0`` -- Triton disabled; every operation uses the dense reference path.
    - ``1`` -- universal kernels that need no launch-configuration table; each
      either runtime-autotunes or runs a single shape-independent configuration.

        - All ``se``-family descriptors: the fused smooth environment matrix
          (:mod:`.triton.env_mat`), a drop-in for the ``prod_env_mat`` front end
          with a closed-form force backward.
        - SeZM/DPA4: block-diagonal rotation, radial degree mixing, the
          ``SO2Linear`` block GEMM, Wigner monomials, flash-attention
          aggregation, and the segmented force assembly.
        - DPA1 (``se_atten``): the fused environment convolution
          (:mod:`.triton.dpa1.se_conv`).

    - ``2`` -- adds kernels whose launch configuration is resolved from a swept
      table, falling back to the level-1 configuration so unswept shapes never
      regress below level 1.

        - SeZM/DPA4: the fused SO(2) value path and the edge-block backward
          kernels keyed by ``(focus_dim, lmax)`` / ``(C_wide, lmax)`` in
          :mod:`.triton.sezm.tile_configs`.
        - DPA1: the environment convolution keyed by ``(ng, H1)`` in
          :mod:`.triton.dpa1.tile_configs`.

    - ``3`` -- adds fp16x3 split-compensated GEMMs, which recover near-fp32
      accuracy on tensor cores and trade a bounded, validated accuracy
      perturbation for speed. Entries exist only for table keys whose
      configuration passed the fp64 validation sweep; unswept shapes keep the
      level-2 fp32 path.

        - SeZM/DPA4: the mixing-stack GEMMs
          (:mod:`.triton.sezm.so2_stack_fp16x3`).
        - DPA1 (``se_atten``): the compute-bound embedding last-layer GEMM
          (:mod:`.triton.dpa1.gemm_fp16x3`).

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


CUDA_INFER_LEVELS = (0, 1, 2)


def cuda_infer_level() -> int:
    """Return the opt-in CUDA mega-kernel inference level from ``DP_CUDA_INFER``.

    Read at model construction time so that it becomes a compile-time constant in
    the traced (``make_fx``) graph, independent of ``DP_TRITON_INFER``. It only
    takes effect during inference.

    Levels are cumulative:

    - ``0`` -- CUDA mega kernels disabled; every operation uses the Triton or
      dense reference path.
    - ``1`` -- the fused graph-lower operator suite (separate operators, force
      from ``autograd.grad``):

        - DPA1 (``se_atten``): the descriptor mega kernels
          (:mod:`.cuda.dpa1.graph_descriptor`) serve the concat,
          attention-free graph lower, and the energy fitting runs through the
          fused cuBLAS network (:mod:`.cuda.graph_fitting`).
        - All graph-lowered models: the force / virial assembly scatters
          through :mod:`.cuda.edge_force_virial`.

    - ``2`` -- adds the end-to-end energy-force operator: a graph-lowered energy
      model whose descriptor and fitting are both fused-eligible collapses its
      descriptor, fitting and analytic force / virial assembly into one operator
      that returns the force as a value (no autograd tape), numerically
      identical to level 1. A model outside that class falls back to the level-1
      operators, so level 2 never regresses below level 1.

        - DPA1 (``se_atten``): the attention-free graph lower with a
          fused-eligible energy fitting routes through
          :mod:`.cuda.dpa1.graph_energy_force`.

    Returns
    -------
    int
        The configured level in ``{0, 1, 2}``.

    Raises
    ------
    ValueError
        If ``DP_CUDA_INFER`` is not an integer in ``{0, 1, 2}``.
    """
    raw = os.environ.get("DP_CUDA_INFER", "0").strip()
    try:
        level = int(raw)
    except ValueError:
        raise ValueError(
            f"DP_CUDA_INFER must be an integer in {CUDA_INFER_LEVELS}, got {raw!r}"
        ) from None
    if level not in CUDA_INFER_LEVELS:
        raise ValueError(
            f"DP_CUDA_INFER must be one of {CUDA_INFER_LEVELS}, got {level}"
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
