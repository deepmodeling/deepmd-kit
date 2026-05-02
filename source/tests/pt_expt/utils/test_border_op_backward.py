# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for the new C++ symbol ``deepmd::border_op_backward`` and
the pt_expt autograd path that dispatches to it.

Tests two distinct surfaces:

1. **Direct op call** — invokes ``torch.ops.deepmd.border_op_backward``
   with hand-built comm tensors (single-rank self-exchange via ctypes
   pointer trick). Verifies the symbol is registered, accepts the
   expected positional args, and produces a correctly-shaped output
   for both ``float32`` and ``float64`` (covers the ``backward_t``
   template's two specializations).

2. **Through the opaque wrapper** — exercises
   ``torch.ops.deepmd_export.border_op``'s ``register_autograd``
   pathway. Calls the wrapper inside an autograd context, asks for
   ``grad`` w.r.t. the ``g1`` input, and verifies the gradient flows
   through (matches the gradient produced by an equivalent
   ``index_select`` + ``index_add_`` Python implementation, which is
   the reference for the symmetric MPI exchange in single-rank).
"""

from __future__ import (
    annotations,
)

import ctypes

import numpy as np
import pytest
import torch

# comm self-bootstraps the underlying libdeepmd_op_pt.so when needed, so
# this single side-effect import is enough to register both the C++
# ops (deepmd::border_op_backward) and their fake/autograd metadata.
import deepmd.pt_expt.utils.comm  # noqa: F401  - registers deepmd_export::border_op


def _addr_of(np_arr: np.ndarray) -> int:
    return np_arr.ctypes.data_as(ctypes.c_void_p).value


def _build_self_swap(
    nloc: int,
    nghost: int,
    sendlist_indices: np.ndarray,
    keepalive: list,
    dtype: torch.dtype,
):
    """Build comm tensors for a single self-exchange swap."""
    sendlist_indices = np.ascontiguousarray(sendlist_indices, dtype=np.int32)
    keepalive.append(sendlist_indices)
    nswap = 1
    addr = _addr_of(sendlist_indices)
    sendlist = torch.tensor([addr], dtype=torch.int64)
    sendproc = torch.zeros(nswap, dtype=torch.int32)
    recvproc = torch.zeros(nswap, dtype=torch.int32)
    sendnum = torch.tensor([nghost], dtype=torch.int32)
    recvnum = torch.tensor([nghost], dtype=torch.int32)
    communicator = torch.zeros(1, dtype=torch.int64)
    nlocal_ts = torch.tensor(nloc, dtype=torch.int32)
    nghost_ts = torch.tensor(nghost, dtype=torch.int32)
    return (
        sendlist,
        sendproc,
        recvproc,
        sendnum,
        recvnum,
        communicator,
        nlocal_ts,
        nghost_ts,
    )


# ---------------------------------------------------------------------------
# 1. Direct op call: border_op_backward as a standalone op
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_border_op_backward_direct(dtype: torch.dtype) -> None:
    """``torch.ops.deepmd.border_op_backward`` is callable for both
    float32 and float64 inputs and returns a tensor of the expected
    shape on the input's device.
    """
    assert hasattr(torch.ops.deepmd, "border_op_backward"), (
        "Symbol not registered; rebuild libdeepmd_op_pt.so."
    )
    nloc, nghost = 5, 3
    nall = nloc + nghost
    n_dim = 4

    keepalive: list = []
    sendlist_indices = np.array([0, 1, 2], dtype=np.int32)
    comm = _build_self_swap(nloc, nghost, sendlist_indices, keepalive, dtype)

    grad_g1 = torch.ones(nall, n_dim, dtype=dtype)

    grad_in = torch.ops.deepmd.border_op_backward(
        comm[0],
        comm[1],
        comm[2],
        comm[3],
        comm[4],
        grad_g1,
        comm[5],
        comm[6],
        comm[7],
    )

    # backward must preserve dtype and shape, and run on the same device.
    assert grad_in.dtype == grad_g1.dtype
    assert tuple(grad_in.shape) == tuple(grad_g1.shape)
    assert grad_in.device == grad_g1.device


def test_border_op_backward_accumulation_semantics() -> None:
    """Single-rank self-exchange backward: each ghost slot's grad is
    accumulated into the local atom whose index sendlist points to.

    Reference: for forward ``g_ext[nloc + i] = g[sendlist[i]]``, the
    reverse is ``grad_g[sendlist[i]] += grad_g_ext[nloc + i]``.
    """
    nloc, nghost = 4, 4
    nall = nloc + nghost
    n_dim = 3

    # Each ghost slot maps back to a local atom: ghost 0->local 0, ghost
    # 1->local 1, etc. So backward should add grad_g_ext[nloc+i] into
    # grad_g[i] for i in [0, nghost).
    keepalive: list = []
    sendlist_indices = np.array([0, 1, 2, 3], dtype=np.int32)
    comm = _build_self_swap(
        nloc,
        nghost,
        sendlist_indices,
        keepalive,
        torch.float64,
    )

    # Distinct values per ghost slot so we can identify the routing.
    grad_g1 = torch.zeros(nall, n_dim, dtype=torch.float64)
    grad_g1[nloc + 0, 0] = 7.0
    grad_g1[nloc + 1, 1] = 11.0
    grad_g1[nloc + 2, 2] = 13.0
    grad_g1[nloc + 3, 0] = 17.0
    # Local part has its own grad too — must pass through unchanged.
    grad_g1[0, 1] = 1.0
    grad_g1[2, 2] = 2.0
    # Capture the input BEFORE the call: the C++ op writes
    # ``index_add_`` into the same tensor and returns it, so once
    # we've called the op the ``grad_g1`` reference points to the
    # modified buffer.  Snapshot first.
    grad_g1_orig = grad_g1.clone()
    grad_in = torch.ops.deepmd.border_op_backward(
        comm[0],
        comm[1],
        comm[2],
        comm[3],
        comm[4],
        grad_g1,
        comm[5],
        comm[6],
        comm[7],
    )

    # Expected: grad_g_local += grad_g_ext[nloc:] indexed by sendlist.
    # Ghost rows pass through unchanged (the C++ backward does not
    # zero them; the wrapper's autograd consumer is F.pad whose
    # backward drops them anyway).
    expected = grad_g1_orig.clone()
    for i, src_local_idx in enumerate(sendlist_indices.tolist()):
        expected[src_local_idx] += grad_g1_orig[nloc + i]
    np.testing.assert_allclose(
        grad_in.numpy(),
        expected.numpy(),
        atol=1e-12,
        rtol=0,
    )


# ---------------------------------------------------------------------------
# 2. Autograd path through the deepmd_export::border_op opaque wrapper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_border_op_export_autograd(dtype: torch.dtype) -> None:
    """End-to-end autograd through the opaque wrapper.

    Builds an inputs tensor with ``requires_grad=True``, calls the
    wrapper, sums the output, and asks for ``grad`` w.r.t. the input.
    The reported gradient must match a hand-computed reference based
    on the same self-exchange routing.
    """
    nloc, nghost = 3, 2
    nall = nloc + nghost
    n_dim = 4

    keepalive: list = []
    sendlist_indices = np.array([0, 1], dtype=np.int32)  # ghosts mirror locals 0,1
    comm = _build_self_swap(nloc, nghost, sendlist_indices, keepalive, dtype)

    # g1 is full nall-shape pre-padded; ghosts initialised to zero
    # (mirroring how repflows.forward feeds the wrapper).
    rng = np.random.default_rng(123)
    g1_np = rng.normal(size=(nall, n_dim)).astype(
        np.float32 if dtype == torch.float32 else np.float64,
    )
    g1_np[nloc:] = 0.0
    g1 = torch.tensor(g1_np, dtype=dtype, requires_grad=True)

    out = torch.ops.deepmd_export.border_op(
        comm[0],
        comm[1],
        comm[2],
        comm[3],
        comm[4],
        g1,
        comm[5],
        comm[6],
        comm[7],
    )
    # Sum so the upstream grad is all-ones at every position.
    loss = out.sum()
    (grad_in,) = torch.autograd.grad(loss, g1, create_graph=False)

    # Reference for LOCAL rows only: forward sets
    # ``out[nloc + i] = g1[sendlist[i]]`` for each ghost slot i and
    # passes local rows through.  With ``loss = out.sum()`` the
    # upstream grad is ones everywhere, so each local row k receives
    # 1 (from ``out[k] = g1[k]``) plus 1 for every ghost slot that
    # references k via ``sendlist``.
    expected_local = torch.ones(nloc, n_dim, dtype=dtype)
    for s in sendlist_indices:
        expected_local[int(s)] += 1.0
    rtol, atol = (0.0, 1e-5) if dtype == torch.float32 else (0.0, 1e-12)
    np.testing.assert_allclose(
        grad_in[:nloc].numpy(),
        expected_local.numpy(),
        atol=atol,
        rtol=rtol,
    )
    # Ghost rows of grad_in are not semantically meaningful: in
    # production the wrapper's input is ``F.pad(node_ebd, value=0)``
    # so the ghost-row gradient is consumed by ``F.pad``'s backward
    # (which drops it).  The C++ backward leaves them as the upstream
    # grad (here, ones), but we don't assert on it.
