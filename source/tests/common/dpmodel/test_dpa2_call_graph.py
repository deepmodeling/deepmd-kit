# SPDX-License-Identifier: LGPL-3.0-or-later
"""Block-level parity between the graph-native
``DescrptBlockRepformers.call_graph`` (the repformer block kernel: per-edge
env-mat, g2 embedding, layer loop with the ghost-exchange seam, final
``rot_mat``) and the legacy dense ``DescrptBlockRepformers.call`` on the SAME
neighbor list.

The graph path is ghost-free (``src`` is always a LOCAL owner), so the
per-layer ``_exchange_ghosts_graph`` seam is identity in dpmodel (the pt_expt
MPI override lives elsewhere); the dense path builds a genuine periodic
system with ghosts to also exercise its own (mapping-based) ``_exchange_ghosts``.
"""

import itertools

import numpy as np
import pytest

from deepmd.dpmodel.descriptor.repformers import (
    DescrptBlockRepformers,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    graph_from_dense_quartet,
)
from deepmd.dpmodel.utils.nlist import (
    extend_input_and_build_neighbor_list,
)


def _make_block(**kwargs) -> DescrptBlockRepformers:
    cfg = {
        "rcut": 4.0,
        "rcut_smth": 0.5,
        "sel": 24,
        "ntypes": 2,
        "nlayers": 2,
        "g1_dim": 6,
        "g2_dim": 4,
        "axis_neuron": 2,
        "attn1_hidden": 4,
        "attn1_nhead": 2,
        "attn2_hidden": 4,
        "attn2_nhead": 2,
        "precision": "float64",
        "seed": 42,
    }
    cfg.update(kwargs)
    return DescrptBlockRepformers(**cfg)


def _system(seed: int = 0, nf: int = 1, nloc: int = 6, box_size: float = 8.0):
    """Small 2-type periodic system."""
    rng = np.random.default_rng(seed)
    coord = rng.random((nf, nloc, 3)) * box_size
    atype = rng.integers(0, 2, size=(nf, nloc)).astype(np.int64)
    box = np.tile((np.eye(3) * box_size).reshape(1, 9), (nf, 1))
    return coord, atype, box


def _build_nlist(block: DescrptBlockRepformers, coord, atype, box):
    return extend_input_and_build_neighbor_list(
        coord,
        atype,
        block.get_rcut(),
        block.get_sel(),
        mixed_types=block.mixed_types(),
        box=box,
    )


def _dense_and_graph_call(block: DescrptBlockRepformers, seed: int = 7):
    """Run both the dense and graph block kernels on the SAME nlist.

    Returns dense outputs, graph outputs (reshaped to the dense (nf, nloc, ...)
    layout where applicable), and the "real edge" mask (nf, nloc, nnei) that
    accounts for BOTH nlist padding AND descriptor-level exclude_types --
    padding/excluded edges are not bit-comparable across the two fill
    conventions (dense fills padding slots with atom-0 geometry; the graph
    zero-fills), so g2/h2/sw parity is only asserted on real edges.
    """
    coord, atype, box = _system(seed)
    ext_coord, ext_atype, mapping, nlist = _build_nlist(block, coord, atype, box)
    nf, nloc = atype.shape
    nall = ext_atype.shape[1]
    nnei = nlist.shape[-1]

    rng = np.random.default_rng(seed + 1)
    g1_local = rng.normal(size=(nf, nloc, block.g1_dim))
    # only the first `nloc` slots of atype_embd_ext are read by the dense
    # `call` (see repformers.py:559, `xp_take_first_n(atype_embd_ext, 1, nloc)`);
    # the ghost portion is never touched (ghosts are re-derived per layer via
    # `_exchange_ghosts`+`mapping`), so it is safe to leave it zero.
    atype_embd_ext = np.zeros((nf, nall, block.g1_dim), dtype=np.float64)
    atype_embd_ext[:, :nloc, :] = g1_local

    dense_out = block.call(
        nlist,
        ext_coord,
        ext_atype,
        atype_embd_ext=atype_embd_ext,
        mapping=mapping,
    )

    graph, atype_local = graph_from_dense_quartet(ext_coord, ext_atype, nlist, mapping)
    g1_input = np.reshape(g1_local, (-1, block.g1_dim))
    graph_out = block.call_graph(graph, atype_local, g1_input, static_nnei=nnei)

    # the "real" edge mask, INCLUDING descriptor-level exclude_types, mirrors
    # the dense nlist-erasure at repformers.call:541-543.
    excl = block.emask.build_type_exclude_mask(nlist, ext_atype)
    real_mask = (nlist != -1) & np.asarray(excl, dtype=bool)

    return dense_out, graph_out, nf, nloc, nnei, real_mask


def _assert_block_parity(block: DescrptBlockRepformers, seed: int = 7) -> None:
    (
        (dense_g1, dense_g2, dense_h2, dense_rot_mat, dense_sw),
        (
            graph_g1,
            graph_g2,
            graph_h2,
            graph_rot_mat,
            graph_sw,
        ),
        nf,
        nloc,
        nnei,
        real_mask,
    ) = _dense_and_graph_call(block, seed)

    # g1: no neighbor axis -> exact reshape parity.
    np.testing.assert_allclose(
        np.reshape(graph_g1, dense_g1.shape), dense_g1, rtol=1e-12, atol=1e-12
    )
    # rot_mat: no neighbor axis -> exact reshape parity.
    np.testing.assert_allclose(
        np.reshape(graph_rot_mat, dense_rot_mat.shape),
        dense_rot_mat,
        rtol=1e-12,
        atol=1e-12,
    )
    assert not np.any(np.isnan(graph_g1))
    assert not np.any(np.isnan(graph_rot_mat))

    # g2 / h2 / sw carry a neighbor axis; padding (and excluded) slots use
    # DIFFERENT fill conventions between dense (atom-0 geometry via
    # `nlist = where(nlist==-1, 0, nlist)`) and the graph (zero-filled
    # `edge_vec` on masked edges) -- see edge_env_mat's docstring: "Padding
    # edges produce nonzero values but are masked ... downstream." Compare
    # only on the real (unmasked, unexcluded) neighbor slots.
    ng2 = dense_g2.shape[-1]
    graph_g2_dense_shape = np.reshape(graph_g2, (nf, nloc, nnei, ng2))
    graph_h2_dense_shape = np.reshape(graph_h2, (nf, nloc, nnei, 3))
    graph_sw_dense_shape = np.reshape(graph_sw, (nf, nloc, nnei))

    np.testing.assert_allclose(
        graph_g2_dense_shape[real_mask], dense_g2[real_mask], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        graph_h2_dense_shape[real_mask], dense_h2[real_mask], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        graph_sw_dense_shape[real_mask], dense_sw[real_mask], rtol=1e-12, atol=1e-12
    )
    # sw must be EXACTLY zero (not just unchecked) on padding slots for both
    # paths -- this is a hard contract (edge_env_mat / dense nlist_mask).
    np.testing.assert_allclose(graph_sw_dense_shape[~real_mask], 0.0, atol=0.0)
    np.testing.assert_allclose(dense_sw[~real_mask], 0.0, atol=0.0)


class TestRepformersBlockCallGraphParity:
    @pytest.mark.parametrize(
        "smooth,use_sqrt_nnei,direct_dist",
        list(itertools.product([True, False], [True, False], [True, False])),
    )
    def test_block_graph_equals_dense(self, smooth, use_sqrt_nnei, direct_dist):
        block = _make_block(
            smooth=smooth, use_sqrt_nnei=use_sqrt_nnei, direct_dist=direct_dist
        )
        _assert_block_parity(block)

    def test_block_graph_equals_dense_exclude_types(self):
        block = _make_block(exclude_types=[[0, 1]])
        _assert_block_parity(block)


class TestRepformersBlockCallGraphTorch:
    def test_call_graph_torch_smoke(self):
        import torch

        block = _make_block()
        coord, atype, box = _system(11)
        ext_coord, ext_atype, mapping, nlist = _build_nlist(block, coord, atype, box)
        nf, nloc = atype.shape
        nnei = nlist.shape[-1]
        rng = np.random.default_rng(12)
        g1_local = rng.normal(size=(nf, nloc, block.g1_dim))

        graph, atype_local = graph_from_dense_quartet(
            ext_coord, ext_atype, nlist, mapping
        )
        g1_input = np.reshape(g1_local, (-1, block.g1_dim))

        ref = block.call_graph(graph, atype_local, g1_input, static_nnei=nnei)

        t_graph = type(graph)(
            n_node=torch.from_numpy(np.asarray(graph.n_node)),
            edge_index=torch.from_numpy(np.asarray(graph.edge_index)),
            edge_vec=torch.from_numpy(np.asarray(graph.edge_vec)),
            edge_mask=torch.from_numpy(np.asarray(graph.edge_mask)),
        )
        got = block.call_graph(
            t_graph,
            torch.from_numpy(np.asarray(atype_local)),
            torch.from_numpy(g1_input),
            static_nnei=nnei,
        )
        for r, g in zip(ref, got, strict=True):
            assert isinstance(g, torch.Tensor)
            np.testing.assert_allclose(g.numpy(), np.asarray(r), rtol=1e-12, atol=1e-12)
            assert not torch.any(torch.isnan(g))


class TestRepformersBlockSlotIndependence:
    def test_mean_stddev_slot_independent(self):
        """PR-A-proven invariant: EnvMatStatSe stats are slot-independent
        (broadcast over the ``nnei`` axis), so reading ``self.mean[:, 0, :]`` /
        ``self.stddev[:, 0, :]`` in ``call_graph`` is valid for the FULL
        (ntypes, nnei, 4) stat tensor, not just the default zeros/ones.
        """
        block = _make_block(set_davg_zero=False)
        coord, atype, box = _system(3, nloc=8)
        sample = {"coord": coord, "atype": atype, "box": box}
        block.compute_input_stats([sample])

        mean = np.asarray(block.mean)
        stddev = np.asarray(block.stddev)
        assert mean.shape == (block.ntypes, block.nnei, 4)
        assert stddev.shape == (block.ntypes, block.nnei, 4)
        for i in range(block.nnei):
            np.testing.assert_array_equal(mean[:, i, :], mean[:, 0, :])
            np.testing.assert_array_equal(stddev[:, i, :], stddev[:, 0, :])


class TestExchangeGhostsGraphSeam:
    def test_identity_when_no_comm_dict(self):
        block = _make_block()
        g1 = np.random.default_rng(0).normal(size=(5, block.g1_dim))
        out = block._exchange_ghosts_graph(g1, None, n_total=5)
        assert out is g1

    def test_raises_notimplemented_with_comm_dict(self):
        block = _make_block()
        g1 = np.random.default_rng(0).normal(size=(5, block.g1_dim))
        with pytest.raises(NotImplementedError):
            block._exchange_ghosts_graph(g1, {"some": "dict"}, n_total=5)
