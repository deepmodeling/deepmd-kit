# SPDX-License-Identifier: LGPL-3.0-or-later
"""Regression tests for TensorFlow DeepEval parameter shorthand handling."""

import numpy as np

from deepmd.tf.infer.deep_eval import (
    DeepEval,
)


class _CapturingAutoBatchSize:
    """Capture inputs at the auto-batching boundary without loading a model."""

    def __init__(self) -> None:
        self.kwargs: dict | None = None

    def execute_all(
        self,
        inner_func,
        nframes: int,
        natoms: int,
        *args,
        **kwargs,
    ) -> np.ndarray:
        """Record canonical inputs that would be sliced into frame batches."""
        self.kwargs = kwargs
        return np.zeros((nframes, natoms, 1), dtype=np.float64)


def test_eval_descriptor_normalizes_shared_parameters_before_batching() -> None:
    """Shared parameters must gain a frame axis before auto batching."""
    auto_batch_size = _CapturingAutoBatchSize()
    backend = object.__new__(DeepEval)
    backend.dfparam = 2
    backend.daparam = 1
    backend.auto_batch_size = auto_batch_size

    fparam = np.array([0.25, -0.5])
    aparam = np.array([[0.1], [0.2]])
    descriptor = backend.eval_descriptor(
        np.zeros((2, 2, 3)),
        None,
        np.array([0, 1]),
        fparam=fparam,
        aparam=aparam,
    )

    assert descriptor.shape == (2, 2, 1)
    assert auto_batch_size.kwargs is not None
    np.testing.assert_array_equal(
        auto_batch_size.kwargs["fparam"], np.tile(fparam, (2, 1))
    )
    np.testing.assert_array_equal(
        auto_batch_size.kwargs["aparam"], np.tile(aparam[None, :, :], (2, 1, 1))
    )
