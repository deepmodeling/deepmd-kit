# SPDX-License-Identifier: LGPL-3.0-or-later
"""Exercise parameter shorthand through the pt_expt DeepEval adapter."""

from types import (
    SimpleNamespace,
)
from unittest.mock import (
    MagicMock,
)

import numpy as np

from deepmd.pt_expt.infer.deep_eval import (
    DeepEval,
)


def test_eval_standardizes_parameter_shorthand_before_dispatch() -> None:
    """The pt_expt adapter must normalize parameters before auto batching."""
    abstract_methods = getattr(DeepEval, "__abstractmethods__", frozenset())
    try:
        DeepEval.__abstractmethods__ = frozenset()
        backend = object.__new__(DeepEval)
    finally:
        DeepEval.__abstractmethods__ = abstract_methods

    nframes = 2
    natoms = 3
    fparam = np.array([0.25, -0.5])
    aparam = np.arange(natoms, dtype=np.float64)[:, None]
    backend._is_spin = False
    backend.get_dim_fparam = lambda: 2
    backend.get_dim_aparam = lambda: 1
    backend._get_request_defs = lambda atomic: [SimpleNamespace(name="energy")]
    backend._eval_func = lambda inner, numb_test, numb_atoms: inner
    backend._eval_model = MagicMock(return_value=(np.zeros((nframes, 1)),))

    result = backend.eval(
        np.zeros((nframes, natoms, 3)),
        None,
        np.zeros(natoms, dtype=np.int32),
        fparam=fparam,
        aparam=aparam,
    )

    np.testing.assert_array_equal(
        backend._eval_model.call_args.args[3], np.tile(fparam, (nframes, 1))
    )
    np.testing.assert_array_equal(
        backend._eval_model.call_args.args[4], np.tile(aparam, (nframes, 1, 1))
    )
    assert result.keys() == {"energy"}
