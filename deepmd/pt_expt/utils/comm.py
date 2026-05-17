# SPDX-License-Identifier: LGPL-3.0-or-later
"""Python-side fake / autograd registration for the C++-defined opaque
``deepmd_export::border_op`` and ``deepmd_export::border_op_backward``.

The op schemas and concrete CPU/CUDA implementations are defined in
``source/op/pt/comm.cc`` (registered under explicit dispatch keys so
``torch.export`` records them as opaque external calls instead of
decomposing into the C++ kernel — which would hit ``data_ptr()`` on
FakeTensors and fail).  Defining the schema in C++ also means a
``.pt2`` archive loaded by a pure-C++ process (LAMMPS via
``DeepPotPTExpt``) can dispatch through the registered op without
needing a Python interpreter.

This module adds the Python-only metadata that the ops still need:
    * ``register_fake`` so ``make_fx`` / ``torch.export`` can trace
      through them with FakeTensor inputs.
    * ``register_autograd`` so ``torch.autograd.grad`` (used inside
      ``forward_common_lower_exportable_with_comm``) flows gradients
      through the forward op back to its inputs.

Constraints discovered during de-risking (scratch/derisk_border_op.py):
    1. Both forward and backward outputs must NOT alias their inputs
       (the C++ kernels return the same tensor they modified) — the
       C++ wrapper layer in ``comm.cc`` clones them before exposing.
    2. The fake impls honour ``g1.dtype`` (no float64 hardcoding).
    3. ``register_autograd`` makes the forward op differentiable; the
       backward callback dispatches to the opaque
       ``deepmd_export::border_op_backward`` op so ``make_fx`` tracing
       through ``autograd.grad`` also sees a black box.
"""

from __future__ import (
    annotations,
)

import torch


def _check_underlying_ops_loaded() -> None:
    """Surface a clearer error when libdeepmd_op_pt.so isn't loaded.

    pt_expt depends on libdeepmd_op_pt.so for the ``deepmd_export::*``
    op schemas + impls.  Without it, the ops can't be registered for
    fake/autograd metadata and callers get a cryptic AttributeError
    on ``torch.ops.deepmd_export.border_op``.

    The .so is loaded as a side effect of ``import deepmd.pt`` (via
    ``deepmd/pt/cxx_op.py``).  We trigger that import here so callers
    don't have to remember to do it first — important for environments
    like DDP-spawned subprocesses that re-import modules from scratch
    and never see the test conftest's ``import deepmd.pt``.
    """
    if not (
        hasattr(torch.ops, "deepmd_export")
        and hasattr(torch.ops.deepmd_export, "border_op")
        and hasattr(torch.ops.deepmd_export, "border_op_backward")
    ):
        # Triggers cxx_op.py which torch.ops.load_library's the .so.
        try:
            import deepmd.pt  # noqa: F401
        except Exception:
            # If deepmd.pt itself fails to import, fall through to the
            # explicit RuntimeError below — clearer than re-raising a
            # potentially-unrelated import error.
            pass

    if not (
        hasattr(torch.ops, "deepmd_export")
        and hasattr(torch.ops.deepmd_export, "border_op")
        and hasattr(torch.ops.deepmd_export, "border_op_backward")
    ):
        raise RuntimeError(
            "torch.ops.deepmd_export.{border_op,border_op_backward} "
            "are not registered. Build libdeepmd_op_pt.so and ensure "
            "deepmd.pt is importable before this module."
        )


_check_underlying_ops_loaded()


# ---------------------------------------------------------------------------
# Fake (meta) impls — let make_fx / torch.export trace through.
# ---------------------------------------------------------------------------


@torch.library.register_fake("deepmd_export::border_op")
def _border_op_fake(
    sendlist: torch.Tensor,
    sendproc: torch.Tensor,
    recvproc: torch.Tensor,
    sendnum: torch.Tensor,
    recvnum: torch.Tensor,
    g1: torch.Tensor,
    communicator: torch.Tensor,
    nlocal: torch.Tensor,
    nghost: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(g1)


@torch.library.register_fake("deepmd_export::border_op_backward")
def _border_op_backward_fake(
    sendlist: torch.Tensor,
    sendproc: torch.Tensor,
    recvproc: torch.Tensor,
    sendnum: torch.Tensor,
    recvnum: torch.Tensor,
    grad_g1: torch.Tensor,
    communicator: torch.Tensor,
    nlocal: torch.Tensor,
    nghost: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(grad_g1)


# ---------------------------------------------------------------------------
# Autograd: route the forward op's backward through the backward op so
# ``make_fx`` tracing through ``torch.autograd.grad`` records both as
# opaque external calls.
# ---------------------------------------------------------------------------


def _border_op_setup_context(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: tuple,
    output: torch.Tensor,
) -> None:
    (
        sendlist,
        sendproc,
        recvproc,
        sendnum,
        recvnum,
        _g1,
        communicator,
        nlocal,
        nghost,
    ) = inputs
    ctx.save_for_backward(
        sendlist,
        sendproc,
        recvproc,
        sendnum,
        recvnum,
        communicator,
        nlocal,
        nghost,
    )


def _border_op_backward(
    ctx: torch.autograd.function.FunctionCtx,
    grad_output: torch.Tensor,
) -> tuple:
    (sendlist, sendproc, recvproc, sendnum, recvnum, communicator, nlocal, nghost) = (
        ctx.saved_tensors
    )
    grad_in = torch.ops.deepmd_export.border_op_backward(
        sendlist,
        sendproc,
        recvproc,
        sendnum,
        recvnum,
        grad_output,
        communicator,
        nlocal,
        nghost,
    )
    return (
        None,
        None,
        None,
        None,
        None,  # sendlist..recvnum
        grad_in,  # g1
        None,
        None,
        None,  # communicator, nlocal, nghost
    )


torch.library.register_autograd(
    "deepmd_export::border_op",
    _border_op_backward,
    setup_context=_border_op_setup_context,
)
