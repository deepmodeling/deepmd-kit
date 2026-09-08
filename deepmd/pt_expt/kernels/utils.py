# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Environment-variable gates for the SeZM/DPA4 hardware-accelerated kernels.

This module centralizes the opt-in selectors that route inference and training
through the custom Triton, CuTe, CUDA and CPU kernel packages. The gates are read
once at model construction time so that they become compile-time constants in the
traced (``make_fx``) graph.

A kernel package that exists for more than one device is selected by
:func:`fused_operators_enabled` and :func:`fused_energy_force_enabled`, which
resolve against the device the graph will execute on. A package that exists
only on CUDA keeps :func:`cuda_infer_level`.

Training and inference are gated separately. An operator qualifies for inference
as soon as it reproduces the forward and the coordinate gradient with the
parameters held fixed, whereas training additionally requires gradients for
every parameter it consumes and a second derivative of its own backward, which
the force loss traverses. The two gates keep a deployed inference path frozen
while the training path opts in independently.
"""

from __future__ import (
    annotations,
)

import os

import torch

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


TRITON_TRAIN_LEVELS = (0, 1)


def triton_train_level() -> int:
    """Return the opt-in Triton training level from ``DP_TRITON_TRAIN``.

    The level is read at model construction time so that it becomes a
    compile-time constant in the traced graph. It only takes effect during
    training and is independent of ``DP_TRITON_INFER``: an operator reaches this
    gate only once it also produces the gradients of every parameter it consumes
    and its backward carries an autograd formula of its own, which the
    second-order force-loss traversal requires.

    - ``0`` -- Triton disabled; training uses the dense reference path.
    - ``1`` -- the second-order-complete universal kernels: the block-diagonal
      rotations, the radial degree mixer, the ``SO2Linear`` block GEMM and the
      flash-attention aggregation. Profitable on the wider shapes, where the
      device time they save exceeds the host cost of dispatching them from a
      backward that the compiler does not fuse.

    The fused CUDA value path is a separate, mutually exclusive training
    dispatch selected by ``DP_CUDA_TRAIN`` (see :func:`cuda_train_enabled`).

    Returns
    -------
    int
        The configured level in ``{0, 1}``.

    Raises
    ------
    ValueError
        If ``DP_TRITON_TRAIN`` is not an integer in ``{0, 1}``.
    """
    raw = os.environ.get("DP_TRITON_TRAIN", "0").strip()
    try:
        level = int(raw)
    except ValueError:
        raise ValueError(
            f"DP_TRITON_TRAIN must be an integer in {TRITON_TRAIN_LEVELS}, got {raw!r}"
        ) from None
    if level not in TRITON_TRAIN_LEVELS:
        raise ValueError(
            f"DP_TRITON_TRAIN must be one of {TRITON_TRAIN_LEVELS}, got {level}"
        )
    return level


def cuda_train_enabled() -> bool:
    """Return whether ``DP_CUDA_TRAIN`` selects the fused CUDA training path.

    Read at model construction time. When enabled, every supported
    ``SO2Convolution`` binds the fused CUDA value path (one kernel for the
    whole value stream, analytic first and second order in the CUDA library)
    and the value-stream dispatch prefers it over the Triton composition of
    that stream; the attention span downstream is independent and follows
    ``DP_TRITON_TRAIN``. The production operating point enables both.
    """
    return os.environ.get("DP_CUDA_TRAIN", "0").strip().lower() in _INFER_TRUE


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
        - DPA4 (``sezm``): the operators whose profit is memory traffic and
          therefore holds on every part measured -- the fused SO(3) grid pair
          product (:mod:`.cuda.dpa4.grid_pair`), which keeps the grid field in
          registers; the fused geometric initial embedding
          (:mod:`.cuda.dpa4.zonal_scatter`), which removes the per-edge message
          tensor; and the fused cutoff envelope and radial basis
          (:mod:`.cuda.dpa4.edge_radial`), which evaluates that chain once
          instead of once per consumer.
        - All graph-lowered models: the force / virial assembly scatters
          through :mod:`.cuda.edge_force_virial`.

    - ``2`` -- adds the end-to-end energy-force operator: a graph-lowered energy
      model whose descriptor and fitting are both fused-eligible collapses its
      descriptor, fitting and analytic force / virial assembly into one operator
      that returns the force as a value (no autograd tape), numerically
      identical to level 1. A model outside that class falls back to the level-1
      operators, so level 2 never regresses below level 1 there. It also adds
      the operators whose profit depends on the float32 throughput of the part:

        - DPA1 (``se_atten``): the attention-free graph lower with a
          fused-eligible energy fitting routes through
          :mod:`.cuda.dpa1.graph_energy_force`.
        - DPA4 (``sezm``): the fused SO(2) convolution
          (:mod:`.cuda.dpa4.so2_conv`), which spans the attention softmax, the
          rotations, the mixing stack and the destination reduction in float32
          SIMT arithmetic. It takes the mixing stack over completely, so the
          fp16x3 GEMMs of ``DP_TRITON_INFER >= 3`` no longer run and the two
          Triton levels coincide at this level.

          Whether the substitution pays is a property of both the part and the
          checkpoint, because what it trades is device traffic against
          arithmetic throughput. It wins where the composition it replaces is
          bandwidth bound and loses where that composition has enough
          arithmetic to reach the tensor cores: on an RTX PRO 6000 Blackwell
          (117 float32 TFLOP/s) 1.63x on ``nano`` and 1.77x on ``mini`` against
          1.05x on ``neo`` and 0.74x on ``air``, whose per-edge arithmetic is 4
          and 11 times ``mini``'s; and 0.79x throughout on an H20 (40
          TFLOP/s). Level 1 is the safe choice for a wide checkpoint or a
          tensor-core part.

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


def backend_device_type() -> str:
    """Return the device type the graph will execute on.

    Every export traces on CPU and moves the program to the backend device
    afterwards, so an operator selection cannot read the device of a traced
    tensor. It reads the backend device instead, which is the one the
    artifact is built for and the one an eager session runs on.

    Returns
    -------
    str
        ``"cuda"`` or ``"cpu"``.
    """
    from deepmd.pt_expt.utils.env import (
        DEVICE,
    )

    return DEVICE.type


def fused_operators_enabled() -> bool:
    """Return whether the fused graph operators may serve an inference call.

    The CUDA operators trade throughput against arithmetic in ways that
    depend on the part and the checkpoint, so they are opt-in through
    :func:`cuda_infer_level`. The CPU operators carry no such trade: they
    replace an Inductor lowering of the same arithmetic and are strictly
    faster wherever they apply, so they need no gate. Whether they apply at
    all is decided by :func:`operator_available` and by each operator's own
    eligibility predicate.

    Returns
    -------
    bool
        Whether the descriptor, fitting and force-assembly operators of the
        backend device are selectable.
    """
    return cuda_infer_level() >= 1 if backend_device_type() == "cuda" else True


def fused_energy_force_enabled() -> bool:
    """Return whether the end-to-end energy-force operator may serve a call.

    The operator collapses the descriptor, the fitting and the analytic force
    and virial assembly into one call that returns the force as a value
    instead of through an autograd tape.

    Returns
    -------
    bool
        Whether the composition is selectable on the backend device.
    """
    return cuda_infer_level() >= 2 if backend_device_type() == "cuda" else True


def operator_available(name: str) -> bool:
    """Return whether an operator carries a kernel for the backend device.

    The operator library is one shared object whose CUDA half is compiled
    only against a CUDA-enabled PyTorch, and whose CPU half is always
    present. Asking the dispatcher for the backend device's key therefore
    answers both "is the library loaded" and "was this half built", which a
    plain attribute lookup on ``torch.ops.deepmd`` cannot distinguish.

    Parameters
    ----------
    name
        Unqualified operator name inside the ``deepmd`` library.

    Returns
    -------
    bool
        Whether the operator is registered for the backend device.
    """
    if not isinstance(
        getattr(torch.ops.deepmd, name, None),
        torch._ops.OpOverloadPacket,
    ):
        return False
    key = "CUDA" if backend_device_type() == "cuda" else "CPU"
    return torch._C._dispatch_has_kernel_for_dispatch_key(f"deepmd::{name}", key)


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


def use_cutile_infer() -> bool:
    """Return whether the opt-in cuTile inference path is enabled.

    The flag is controlled by the ``DP_CUTILE_INFER`` environment variable and is
    read at module construction time. It selects a complete SeZM inference path
    written in the ``cuda.tile`` DSL and only takes effect during inference;
    training always uses the dense reference path.

    The path is mutually exclusive with ``DP_TRITON_INFER`` and
    ``DP_CUTE_INFER``: when it is enabled no Triton kernel executes, and a
    convolution whose layout it does not support falls back to the dense
    reference rather than to another accelerated backend. Enabling more than one
    of the three is rejected at construction.

    Returns
    -------
    bool
        ``True`` when ``DP_CUTILE_INFER`` is set to a truthy value.
    """
    return os.environ.get("DP_CUTILE_INFER", "0").strip().lower() in _INFER_TRUE


def use_amp_infer() -> bool:
    """Return whether bf16 autocast is enabled for inference.

    The flag is controlled by the ``DP_AMP_INFER`` environment variable and is
    read at module construction time. It controls inference independently of
    the descriptor's ``use_amp`` option; training follows ``use_amp`` regardless
    of this environment variable.

    Returns
    -------
    bool
        ``True`` when ``DP_AMP_INFER`` is set to a truthy value.
    """
    return os.environ.get("DP_AMP_INFER", "0").strip().lower() in _INFER_TRUE
