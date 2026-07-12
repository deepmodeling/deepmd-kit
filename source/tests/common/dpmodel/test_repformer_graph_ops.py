# SPDX-License-Identifier: LGPL-3.0-or-later
"""Per-op parity: repformer graph twins vs the dense reference ops on the
identical (shape-static, center-major) edge layout. Same math, fp64 =>
rtol/atol 1e-12."""

import itertools

import numpy as np
import pytest

from deepmd.dpmodel.descriptor.repformers import (
    _cal_hg,
    _cal_grrg,
    graph_cal_grrg,
    graph_cal_hg,
    graph_symmetrization_op,
    symmetrization_op,
)

NF, NLOC, NNEI, NG = 2, 5, 7, 6


def _mk(seed=0):
    rng = np.random.default_rng(seed)
    g = rng.normal(size=(NF, NLOC, NNEI, NG))
    h = rng.normal(size=(NF, NLOC, NNEI, 3))
    mask = rng.random((NF, NLOC, NNEI)) > 0.3
    sw = rng.random((NF, NLOC, NNEI)) * mask
    n_total = NF * NLOC
    dst = np.repeat(np.arange(n_total, dtype=np.int64), NNEI)
    return g, h, mask, sw, n_total, dst


@pytest.mark.parametrize(
    "smooth,use_sqrt_nnei", list(itertools.product([True, False], [True, False]))
)
def test_graph_cal_hg_parity(smooth, use_sqrt_nnei):
    g, h, mask, sw, n_total, dst = _mk()
    ref = _cal_hg(g, h, mask, sw, smooth=smooth, use_sqrt_nnei=use_sqrt_nnei)
    got = graph_cal_hg(
        g.reshape(-1, NG), h.reshape(-1, 3), mask.reshape(-1), sw.reshape(-1),
        dst, n_total, NNEI, smooth=smooth, use_sqrt_nnei=use_sqrt_nnei,
    )
    np.testing.assert_allclose(
        got, ref.reshape(n_total, 3, NG), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("axis_neuron", [2, 4])
def test_graph_symmetrization_op_parity(axis_neuron):
    g, h, mask, sw, n_total, dst = _mk(1)
    ref = symmetrization_op(g, h, mask, sw, axis_neuron)
    got = graph_symmetrization_op(
        g.reshape(-1, NG), h.reshape(-1, 3), mask.reshape(-1), sw.reshape(-1),
        dst, n_total, NNEI, axis_neuron,
    )
    np.testing.assert_allclose(
        got, ref.reshape(n_total, axis_neuron * NG), rtol=1e-12, atol=1e-12
    )


def test_graph_cal_hg_torch():
    import torch

    g, h, mask, sw, n_total, dst = _mk(2)
    ref = graph_cal_hg(
        g.reshape(-1, NG), h.reshape(-1, 3), mask.reshape(-1), sw.reshape(-1),
        dst, n_total, NNEI,
    )
    got = graph_cal_hg(
        torch.from_numpy(g.reshape(-1, NG)), torch.from_numpy(h.reshape(-1, 3)),
        torch.from_numpy(mask.reshape(-1)), torch.from_numpy(sw.reshape(-1)),
        torch.from_numpy(dst), n_total, NNEI,
    )
    np.testing.assert_allclose(got.numpy(), ref, rtol=1e-12)
