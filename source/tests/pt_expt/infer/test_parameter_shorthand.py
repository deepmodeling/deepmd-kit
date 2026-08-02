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


def _new_backend() -> DeepEval:
    """Build an uninitialized pt_expt DeepEval for adapter mocking."""
    abstract_methods = getattr(DeepEval, "__abstractmethods__", frozenset())
    try:
        DeepEval.__abstractmethods__ = frozenset()
        backend = object.__new__(DeepEval)
    finally:
        DeepEval.__abstractmethods__ = abstract_methods
    backend.get_dim_fparam = lambda: 2
    backend.get_dim_aparam = lambda: 1
    backend._require_dpmodel = lambda name: None
    backend._is_spin_model = lambda: False
    return backend


def test_eval_descriptor_standardizes_parameter_shorthand() -> None:
    """The descriptor route must normalize parameters before reshaping."""
    nframes = 2
    natoms = 3
    fparam = np.array([0.25, -0.5])
    aparam = np.arange(natoms, dtype=np.float64)[:, None]
    backend = _new_backend()
    dp_am = MagicMock()
    backend._dpmodel = MagicMock()
    backend._dpmodel.get_dp_atomic_model.return_value = dp_am
    descriptor = MagicMock()
    descriptor.detach.return_value = descriptor
    descriptor.cpu.return_value = descriptor
    descriptor.numpy.return_value = np.zeros((nframes, natoms, 4))
    dp_am.descriptor.return_value = (descriptor,)
    backend._prepare_nlist_inputs = MagicMock(
        return_value=(None, None, None, None, None, None, None, nframes, natoms)
    )

    backend.eval_descriptor(
        np.zeros((nframes, natoms, 3)),
        None,
        np.zeros(natoms, dtype=np.int32),
        fparam=fparam,
        aparam=aparam,
    )

    np.testing.assert_array_equal(
        backend._prepare_nlist_inputs.call_args.args[3],
        np.tile(fparam, (nframes, 1)),
    )
    np.testing.assert_array_equal(
        backend._prepare_nlist_inputs.call_args.args[4],
        np.tile(aparam, (nframes, 1, 1)),
    )


def test_eval_fitting_last_layer_standardizes_parameter_shorthand() -> None:
    """The fitting-last-layer route must normalize parameters before reshaping."""
    nframes = 2
    natoms = 3
    fparam = np.array([0.25, -0.5])
    aparam = np.arange(natoms, dtype=np.float64)[:, None]
    backend = _new_backend()
    dp_am = MagicMock()
    backend._dpmodel = MagicMock()
    backend._dpmodel.get_dp_atomic_model.return_value = dp_am
    dp_am.descriptor.return_value = (
        np.zeros((nframes, natoms, 4)),
        np.zeros((nframes, natoms, 3, 3)),
        np.zeros((nframes, natoms, 3, 3)),
        np.zeros((nframes, natoms, 3, 3)),
        np.zeros((nframes, natoms, 3)),
    )
    out = MagicMock()
    out.detach.return_value = out
    out.cpu.return_value = out
    out.numpy.return_value = np.zeros((nframes, natoms, 4))
    dp_am.fitting_net = MagicMock(return_value={"middle_output": out})
    ext_atype_t = np.zeros((nframes, natoms), dtype=np.int64)
    backend._prepare_nlist_inputs = MagicMock(
        return_value=(
            None,
            ext_atype_t,
            None,
            None,
            None,
            None,
            None,
            nframes,
            natoms,
        )
    )

    backend.eval_fitting_last_layer(
        np.zeros((nframes, natoms, 3)),
        None,
        np.zeros(natoms, dtype=np.int32),
        fparam=fparam,
        aparam=aparam,
    )

    np.testing.assert_array_equal(
        backend._prepare_nlist_inputs.call_args.args[3],
        np.tile(fparam, (nframes, 1)),
    )
    np.testing.assert_array_equal(
        backend._prepare_nlist_inputs.call_args.args[4],
        np.tile(aparam, (nframes, 1, 1)),
    )
