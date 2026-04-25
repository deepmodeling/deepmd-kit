# SPDX-License-Identifier: LGPL-3.0-or-later
"""Opaque torch.export wrapper around the deepmd MPI border_op.

The existing ``torch.ops.deepmd.border_op`` (registered by
``libdeepmd_op_pt.so``) is a ``CompositeImplicitAutograd`` op that wraps
``Border::apply`` for the torch.jit (pt) backend. ``torch.export`` /
AOTInductor try to *decompose* such ops into primitive aten ops, which
fails because the C++ kernel calls ``data_ptr()`` on inputs — illegal
during tracing on FakeTensors.

This module defines a NEW op ``deepmd_export::border_op`` via
``torch.library.custom_op``, marked opaque so ``torch.export`` records it
as a single black-box call. At runtime the loaded ``.pt2`` dispatches
back into ``torch.ops.deepmd.border_op`` (forward) or
``torch.ops.deepmd.border_op_backward`` (backward), preserving the MPI
exchange semantics.

Constraints discovered during de-risking (scratch/derisk_border_op.py):
    1. ``custom_op`` forbids returning a tensor that aliases an input —
       the underlying C++ op returns ``g1`` itself, so we ``.clone()``.
    2. The fake (meta) impl honours ``g1.dtype`` (no float64 hardcoding).
    3. ``register_autograd`` makes the op differentiable; the backward
       dispatches to ``deepmd::border_op_backward`` which performs the
       symmetric MPI exchange.
"""

from __future__ import (
    annotations,
)

import torch


@torch.library.custom_op("deepmd_export::border_op", mutates_args=())
def border_op_export(
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
    """Opaque wrapper around ``torch.ops.deepmd.border_op``.

    Performs MPI ghost-atom exchange of the embedding tensor ``g1`` so
    GNN message-passing layers can run under multi-rank LAMMPS. Inputs
    and outputs match the underlying op exactly except for the aliasing
    fix (see module docstring).
    """
    out = torch.ops.deepmd.border_op(
        sendlist,
        sendproc,
        recvproc,
        sendnum,
        recvnum,
        g1,
        communicator,
        nlocal,
        nghost,
    )
    if isinstance(out, (list, tuple)):
        out = out[0]
    # custom_op forbids output aliasing inputs; underlying op returns g1.
    return out.clone()


@border_op_export.register_fake
def _border_op_export_fake(
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
    grad_in = torch.ops.deepmd.border_op_backward(
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
    # Same aliasing concern as forward: the C++ backward returns the same
    # tensor object it modified; clone before handing back to autograd.
    return (
        None,
        None,
        None,
        None,
        None,  # sendlist..recvnum
        grad_in.clone(),  # g1
        None,
        None,
        None,  # communicator, nlocal, nghost
    )


border_op_export.register_autograd(
    _border_op_backward,
    setup_context=_border_op_setup_context,
)
