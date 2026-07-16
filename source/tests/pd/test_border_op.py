# SPDX-License-Identifier: LGPL-3.0-or-later
"""Regression tests for Paddle border-exchange control and data tensors."""

import numpy as np
import paddle
import pytest

deepmd_op_pd = pytest.importorskip(
    "deepmd_op_pd", reason="the Paddle custom operator library is not built"
)


def _control_tensors(nswap: int) -> tuple[paddle.Tensor, ...]:
    """Create the common CPU control tensors for a border exchange."""
    return (
        paddle.zeros([nswap], dtype="int32"),  # sendproc
        paddle.zeros([nswap], dtype="int32"),  # recvproc
        paddle.zeros([nswap], dtype="int32"),  # sendnum
        paddle.zeros([nswap], dtype="int32"),  # recvnum
        paddle.zeros([1], dtype="int64"),  # unused communicator without MPI
    )


def test_border_op_accepts_no_swaps() -> None:
    """Scalar atom counts must remain readable when ``nswap == 0``."""
    sendproc, recvproc, sendnum, recvnum, communicator = _control_tensors(0)
    g1 = paddle.arange(6, dtype="float64").reshape([2, 3])

    result = deepmd_op_pd.border_op(
        paddle.zeros([0], dtype="int64"),
        sendproc,
        recvproc,
        sendnum,
        recvnum,
        g1,
        communicator,
        paddle.to_tensor([2], dtype="int32"),
        paddle.to_tensor([0], dtype="int32"),
    )

    np.testing.assert_array_equal(result.numpy(), g1.numpy())


def test_border_op_self_copy_uses_cpu_place() -> None:
    """A CUDA-enabled operator must not use a GPU copy for CPU tensors."""
    sendproc, recvproc, sendnum, recvnum, communicator = _control_tensors(1)
    sendnum = paddle.ones_like(sendnum)
    recvnum = paddle.ones_like(recvnum)

    # The C++ operator receives the LAMMPS send lists as pointer-valued int64
    # entries.  Keep this NumPy owner alive through the call so the pointed-to
    # int32 index remains valid.
    send_indices = np.array([1], dtype=np.int32)
    sendlist = paddle.to_tensor([send_indices.ctypes.data], dtype="int64")
    g1_leaf = paddle.to_tensor(
        [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]], stop_gradient=False
    )
    # Paddle rejects an in-place custom op on an autograd leaf.  This identity
    # keeps a leaf for checking gradients while letting border_op update g1.
    g1 = g1_leaf * 1.0

    result = deepmd_op_pd.border_op(
        sendlist,
        sendproc,
        recvproc,
        sendnum,
        recvnum,
        g1,
        communicator,
        paddle.to_tensor([2], dtype="int32"),
        paddle.to_tensor([1], dtype="int32"),
    )

    np.testing.assert_array_equal(
        result.numpy(), np.array([[1.0, 2.0], [3.0, 4.0], [3.0, 4.0]])
    )
    # Backpropagation runs the reverse self-swap, which needs the same
    # place-based CPU/GPU dispatch as the forward copy.
    result.sum().backward()
    np.testing.assert_array_equal(g1_leaf.grad.numpy(), np.ones([3, 2]))
