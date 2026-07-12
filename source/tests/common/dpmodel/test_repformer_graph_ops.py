# SPDX-License-Identifier: LGPL-3.0-or-later
"""Per-op parity: repformer graph twins vs the dense reference ops on the
identical (shape-static, center-major) edge layout. Same math, fp64 =>
rtol/atol 1e-12."""

import itertools

import numpy as np
import pytest

from deepmd.dpmodel.descriptor.repformers import (
    RepformerLayer,
    _cal_hg,
    _cal_grrg,
    _make_nei_g1,
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


def _mk_layer(g1_out_conv: bool, seed: int = 0) -> RepformerLayer:
    return RepformerLayer(
        rcut=4.0,
        rcut_smth=0.5,
        sel=NNEI,
        ntypes=2,
        g1_dim=8,
        g2_dim=NG,
        axis_neuron=2,
        update_chnnl_2=True,
        g1_out_conv=g1_out_conv,
        g1_out_mlp=True,
        precision="float64",
        seed=seed,
    )


def _mk_g1_nlist(seed=3):
    rng = np.random.default_rng(seed)
    n_total = NF * NLOC
    g1 = rng.normal(size=(n_total, 8))
    # ghost-free: neighbors index local atoms of the SAME frame
    nlist = rng.integers(0, NLOC, size=(NF, NLOC, NNEI)).astype(np.int64)
    mask = rng.random((NF, NLOC, NNEI)) > 0.3
    nlist = np.where(mask, nlist, -1)
    sw = rng.random((NF, NLOC, NNEI)) * mask
    # flat-edge view of the same topology
    src = (nlist + np.arange(NF)[:, None, None] * NLOC).reshape(-1)
    src = np.where(mask.reshape(-1), src, 0).astype(np.int64)
    dst = np.repeat(np.arange(n_total, dtype=np.int64), NNEI)
    return g1, nlist, mask, sw, src, dst, n_total


@pytest.mark.parametrize("g1_out_conv", [True, False])
def test_graph_update_g1_conv_parity(g1_out_conv):
    layer = _mk_layer(g1_out_conv)
    g1, nlist, mask, sw, src, dst, n_total = _mk_g1_nlist()
    rng = np.random.default_rng(4)
    g2 = rng.normal(size=(NF, NLOC, NNEI, NG))
    g1_ext = g1.reshape(NF, NLOC, 8)
    gg1 = _make_nei_g1(g1_ext, np.where(mask, nlist, 0))
    ref = layer._update_g1_conv(gg1, g2, mask, sw)
    got = layer._graph_update_g1_conv(
        np.take(g1, src, axis=0),
        g2.reshape(-1, NG),
        mask.reshape(-1),
        sw.reshape(-1),
        dst,
        n_total,
        NNEI,
    )
    np.testing.assert_allclose(
        got, ref.reshape(n_total, -1), rtol=1e-12, atol=1e-12
    )


def test_graph_update_g2_g1g1_parity():
    layer = _mk_layer(True)
    g1, nlist, mask, sw, src, dst, n_total = _mk_g1_nlist(5)
    g1_ext = g1.reshape(NF, NLOC, 8)
    gg1 = _make_nei_g1(g1_ext, np.where(mask, nlist, 0))
    ref = layer._update_g2_g1g1(g1_ext, gg1, mask, sw)
    got = layer._graph_update_g2_g1g1(
        g1, src, dst, mask.reshape(-1), sw.reshape(-1)
    )
    np.testing.assert_allclose(
        got, ref.reshape(n_total * NNEI, -1), rtol=1e-12, atol=1e-12
    )


def test_graph_update_g1_conv_torch():
    import torch

    layer = _mk_layer(True, seed=6)
    g1, nlist, mask, sw, src, dst, n_total = _mk_g1_nlist(7)
    rng = np.random.default_rng(8)
    g2 = rng.normal(size=(NF, NLOC, NNEI, NG))
    ref = layer._graph_update_g1_conv(
        np.take(g1, src, axis=0),
        g2.reshape(-1, NG),
        mask.reshape(-1),
        sw.reshape(-1),
        dst,
        n_total,
        NNEI,
    )
    got = layer._graph_update_g1_conv(
        torch.from_numpy(np.take(g1, src, axis=0)),
        torch.from_numpy(g2.reshape(-1, NG)),
        torch.from_numpy(mask.reshape(-1)),
        torch.from_numpy(sw.reshape(-1)),
        torch.from_numpy(dst),
        n_total,
        NNEI,
    )
    np.testing.assert_allclose(got.numpy(), ref, rtol=1e-12, atol=1e-12)
