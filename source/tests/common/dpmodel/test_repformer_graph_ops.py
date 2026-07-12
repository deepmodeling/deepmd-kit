# SPDX-License-Identifier: LGPL-3.0-or-later
"""Per-op parity: repformer graph twins vs the dense reference ops on the
identical (shape-static, center-major) edge layout. Same math, fp64 =>
rtol/atol 1e-12."""

import itertools

import numpy as np
import pytest

from deepmd.dpmodel.descriptor.repformers import (
    Atten2EquiVarApply,
    Atten2Map,
    Atten2MultiHeadApply,
    LocalAtten,
    RepformerLayer,
    _cal_hg,
    _cal_grrg,
    _make_nei_g1,
    _cal_grrg_graph,
    _cal_hg_graph,
    symmetrization_op_graph,
    symmetrization_op,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    center_edge_pairs,
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
def test_cal_hg_graph_parity(smooth, use_sqrt_nnei):
    g, h, mask, sw, n_total, dst = _mk()
    ref = _cal_hg(g, h, mask, sw, smooth=smooth, use_sqrt_nnei=use_sqrt_nnei)
    got = _cal_hg_graph(
        g.reshape(-1, NG), h.reshape(-1, 3), mask.reshape(-1), sw.reshape(-1),
        dst, n_total, NNEI, smooth=smooth, use_sqrt_nnei=use_sqrt_nnei,
    )
    np.testing.assert_allclose(
        got, ref.reshape(n_total, 3, NG), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("axis_neuron", [2, 4])
def test_symmetrization_op_graph_parity(axis_neuron):
    g, h, mask, sw, n_total, dst = _mk(1)
    ref = symmetrization_op(g, h, mask, sw, axis_neuron)
    got = symmetrization_op_graph(
        g.reshape(-1, NG), h.reshape(-1, 3), mask.reshape(-1), sw.reshape(-1),
        dst, n_total, NNEI, axis_neuron,
    )
    np.testing.assert_allclose(
        got, ref.reshape(n_total, axis_neuron * NG), rtol=1e-12, atol=1e-12
    )


def test_cal_hg_graph_torch():
    import torch

    g, h, mask, sw, n_total, dst = _mk(2)
    ref = _cal_hg_graph(
        g.reshape(-1, NG), h.reshape(-1, 3), mask.reshape(-1), sw.reshape(-1),
        dst, n_total, NNEI,
    )
    got = _cal_hg_graph(
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
def test_update_g1_conv_graph_parity(g1_out_conv):
    layer = _mk_layer(g1_out_conv)
    g1, nlist, mask, sw, src, dst, n_total = _mk_g1_nlist()
    rng = np.random.default_rng(4)
    g2 = rng.normal(size=(NF, NLOC, NNEI, NG))
    g1_ext = g1.reshape(NF, NLOC, 8)
    gg1 = _make_nei_g1(g1_ext, np.where(mask, nlist, 0))
    ref = layer._update_g1_conv(gg1, g2, mask, sw)
    got = layer._update_g1_conv_graph(
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


def test_update_g2_g1g1_graph_parity():
    layer = _mk_layer(True)
    g1, nlist, mask, sw, src, dst, n_total = _mk_g1_nlist(5)
    g1_ext = g1.reshape(NF, NLOC, 8)
    gg1 = _make_nei_g1(g1_ext, np.where(mask, nlist, 0))
    ref = layer._update_g2_g1g1(g1_ext, gg1, mask, sw)
    got = layer._update_g2_g1g1_graph(
        g1, src, dst, mask.reshape(-1), sw.reshape(-1)
    )
    np.testing.assert_allclose(
        got, ref.reshape(n_total * NNEI, -1), rtol=1e-12, atol=1e-12
    )


def test_update_g1_conv_graph_torch():
    import torch

    layer = _mk_layer(True, seed=6)
    g1, nlist, mask, sw, src, dst, n_total = _mk_g1_nlist(7)
    rng = np.random.default_rng(8)
    g2 = rng.normal(size=(NF, NLOC, NNEI, NG))
    ref = layer._update_g1_conv_graph(
        np.take(g1, src, axis=0),
        g2.reshape(-1, NG),
        mask.reshape(-1),
        sw.reshape(-1),
        dst,
        n_total,
        NNEI,
    )
    got = layer._update_g1_conv_graph(
        torch.from_numpy(np.take(g1, src, axis=0)),
        torch.from_numpy(g2.reshape(-1, NG)),
        torch.from_numpy(mask.reshape(-1)),
        torch.from_numpy(sw.reshape(-1)),
        torch.from_numpy(dst),
        n_total,
        NNEI,
    )
    np.testing.assert_allclose(got.numpy(), ref, rtol=1e-12, atol=1e-12)


def _pairs(mask, dst, n_total):
    q_e, k_e, pm = center_edge_pairs(
        dst,
        mask.reshape(-1),
        n_total,
        include_self=True,
        ordered=True,
        static_nnei=NNEI,
    )
    return q_e, k_e, pm


@pytest.mark.parametrize("has_gate,smooth", [(True, True), (False, True), (True, False)])
def test_atten2map_parity(has_gate, smooth):
    rng = np.random.default_rng(6)
    a2m = Atten2Map(NG, 4, 2, has_gate=has_gate, smooth=smooth, precision="float64", seed=7)
    g2 = rng.normal(size=(NF, NLOC, NNEI, NG))
    h2 = rng.normal(size=(NF, NLOC, NNEI, 3))
    mask = rng.random((NF, NLOC, NNEI)) > 0.3
    sw = rng.random((NF, NLOC, NNEI)) * mask
    n_total = NF * NLOC
    dst = np.repeat(np.arange(n_total, dtype=np.int64), NNEI)
    ref = a2m.call(g2, h2, mask, sw)  # (nf, nloc, nnei, nnei, nh)
    q_e, k_e, pm = _pairs(mask, dst, n_total)
    got = a2m.call_graph(
        g2.reshape(-1, NG),
        h2.reshape(-1, 3),
        sw.reshape(-1),
        q_e,
        k_e,
        pm,
        NF * NLOC * NNEI,
    )  # (P, nh)
    slot = np.arange(NF * NLOC * NNEI) % NNEI
    ctr = dst[np.asarray(q_e)]
    f, l = ctr // NLOC, ctr % NLOC
    ref_pairs = ref[f, l, slot[np.asarray(q_e)], slot[np.asarray(k_e)], :]
    # NOTE: even for pairs whose query edge is padding, both sides collapse to
    # an exact 0.0 (dense: post-softmax where(mask, 0, ...); graph: pair_mask
    # multiply, and in the smooth branch the fully-masked segment_softmax
    # group also yields exact 0.0), so no restriction to real-query pairs is
    # needed here -- verified empirically before relying on it.
    np.testing.assert_allclose(np.asarray(got), ref_pairs, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("has_gate,smooth", [(True, True), (False, True), (True, False)])
def test_atten2_mh_apply_parity(has_gate, smooth):
    rng = np.random.default_rng(9)
    nh = 3
    a2m = Atten2Map(NG, 4, nh, has_gate=has_gate, smooth=smooth, precision="float64", seed=10)
    mha = Atten2MultiHeadApply(NG, nh, precision="float64", seed=11)
    g2 = rng.normal(size=(NF, NLOC, NNEI, NG))
    h2 = rng.normal(size=(NF, NLOC, NNEI, 3))
    mask = rng.random((NF, NLOC, NNEI)) > 0.3
    sw = rng.random((NF, NLOC, NNEI)) * mask
    n_total = NF * NLOC
    dst = np.repeat(np.arange(n_total, dtype=np.int64), NNEI)
    e_tot = NF * NLOC * NNEI
    ref_AA = a2m.call(g2, h2, mask, sw)  # (nf, nloc, nnei, nnei, nh)
    ref = mha.call(ref_AA, g2)  # (nf, nloc, nnei, ng2)
    q_e, k_e, pm = _pairs(mask, dst, n_total)
    got_AA = a2m.call_graph(
        g2.reshape(-1, NG), h2.reshape(-1, 3), sw.reshape(-1), q_e, k_e, pm, e_tot
    )
    got = mha.call_graph(got_AA, g2.reshape(-1, NG), q_e, k_e, e_tot)
    np.testing.assert_allclose(
        np.asarray(got), ref.reshape(e_tot, NG), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("has_gate,smooth", [(True, True), (False, True), (True, False)])
def test_atten2_ev_apply_parity(has_gate, smooth):
    rng = np.random.default_rng(12)
    nh = 3
    a2m = Atten2Map(NG, 4, nh, has_gate=has_gate, smooth=smooth, precision="float64", seed=13)
    ev = Atten2EquiVarApply(NG, nh, precision="float64", seed=14)
    g2 = rng.normal(size=(NF, NLOC, NNEI, NG))
    h2 = rng.normal(size=(NF, NLOC, NNEI, 3))
    mask = rng.random((NF, NLOC, NNEI)) > 0.3
    sw = rng.random((NF, NLOC, NNEI)) * mask
    n_total = NF * NLOC
    dst = np.repeat(np.arange(n_total, dtype=np.int64), NNEI)
    e_tot = NF * NLOC * NNEI
    ref_AA = a2m.call(g2, h2, mask, sw)  # (nf, nloc, nnei, nnei, nh)
    ref = ev.call(ref_AA, h2)  # (nf, nloc, nnei, 3)
    q_e, k_e, pm = _pairs(mask, dst, n_total)
    got_AA = a2m.call_graph(
        g2.reshape(-1, NG), h2.reshape(-1, 3), sw.reshape(-1), q_e, k_e, pm, e_tot
    )
    got = ev.call_graph(got_AA, h2.reshape(-1, 3), q_e, k_e, e_tot)
    np.testing.assert_allclose(
        np.asarray(got), ref.reshape(e_tot, 3), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("smooth", [True, False])
def test_local_atten_parity(smooth):
    la = LocalAtten(8, 4, 2, smooth=smooth, precision="float64", seed=15)
    g1, nlist, mask, sw, src, dst, n_total = _mk_g1_nlist(20)
    g1_ext = g1.reshape(NF, NLOC, 8)
    gg1 = _make_nei_g1(g1_ext, np.where(mask, nlist, 0))
    ref = la.call(g1_ext, gg1, mask, sw)  # (nf, nloc, ng1)
    got = la.call_graph(
        g1,
        np.take(g1, src, axis=0),
        mask.reshape(-1),
        sw.reshape(-1),
        dst,
        n_total,
    )
    np.testing.assert_allclose(
        np.asarray(got), ref.reshape(n_total, 8), rtol=1e-12, atol=1e-12
    )


def test_atten2map_graph_torch():
    import torch

    rng = np.random.default_rng(16)
    a2m = Atten2Map(NG, 4, 2, has_gate=True, smooth=True, precision="float64", seed=17)
    g2 = rng.normal(size=(NF, NLOC, NNEI, NG))
    h2 = rng.normal(size=(NF, NLOC, NNEI, 3))
    mask = rng.random((NF, NLOC, NNEI)) > 0.3
    sw = rng.random((NF, NLOC, NNEI)) * mask
    n_total = NF * NLOC
    dst = np.repeat(np.arange(n_total, dtype=np.int64), NNEI)
    q_e, k_e, pm = _pairs(mask, dst, n_total)
    e_tot = NF * NLOC * NNEI
    ref = a2m.call_graph(
        g2.reshape(-1, NG), h2.reshape(-1, 3), sw.reshape(-1), q_e, k_e, pm, e_tot
    )
    got = a2m.call_graph(
        torch.from_numpy(g2.reshape(-1, NG)),
        torch.from_numpy(h2.reshape(-1, 3)),
        torch.from_numpy(sw.reshape(-1)),
        torch.from_numpy(q_e),
        torch.from_numpy(k_e),
        torch.from_numpy(pm),
        e_tot,
    )
    np.testing.assert_allclose(got.numpy(), ref, rtol=1e-12, atol=1e-12)
