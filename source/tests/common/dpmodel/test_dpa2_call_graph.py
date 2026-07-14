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

from deepmd.dpmodel.descriptor.dpa2 import (
    DescrptDPA2,
    RepformerArgs,
    RepinitArgs,
)
from deepmd.dpmodel.descriptor.repformers import (
    DescrptBlockRepformers,
)
from deepmd.dpmodel.fitting import (
    InvarFitting,
)
from deepmd.dpmodel.model.edge_transform_output import (
    node_ownership_mask,
)
from deepmd.dpmodel.model.ener_model import (
    EnergyModel,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    from_dense_quartet,
    graph_from_dense_quartet,
)
from deepmd.dpmodel.utils.nlist import (
    build_multiple_neighbor_list,
    extend_input_and_build_neighbor_list,
    get_multiple_nlist_key,
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


def _block_system(seed: int = 0, nf: int = 1, nloc: int = 6, box_size: float = 8.0):
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
    coord, atype, box = _block_system(seed)
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
        coord, atype, box = _block_system(11)
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
        coord, atype, box = _block_system(3, nloc=8)
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


# ---------------------------------------------------------------------------
# Descriptor-level (Task 7): DescrptDPA2.uses_graph_lower / call_graph /
# _call_graph_adapter / the call() routing gate.
# ---------------------------------------------------------------------------


def _make_dpa2(
    repinit_rcut: float = 4.0,
    repinit_nsel: int = 200,
    repformer_rcut: float = 2.0,
    repformer_nsel: int = 150,
    use_three_body: bool = False,
    ntypes: int = 2,
    repformer_attn: bool = True,
    **kwargs,
) -> DescrptDPA2:
    repinit_kwargs = {
        "rcut": repinit_rcut,
        "rcut_smth": 0.5,
        "nsel": repinit_nsel,
        "neuron": [6, 12],
        "axis_neuron": 2,
        "tebd_dim": 4,
        "use_three_body": use_three_body,
    }
    if use_three_body:
        # keep the three-body block's (rcut, nsel) strictly BELOW the
        # repformer block's in the rcut-sorted nsel-ordering check
        # (DescrptDPA2.__init__ asserts nsel is non-decreasing with rcut
        # across repformer/three_body/repinit).
        repinit_kwargs.update(
            three_body_rcut=1.0,
            three_body_rcut_smth=0.3,
            three_body_sel=5,
            three_body_neuron=[2, 4],
        )
    repinit = RepinitArgs(**repinit_kwargs)
    repformer = RepformerArgs(
        rcut=repformer_rcut,
        rcut_smth=0.5,
        nsel=repformer_nsel,
        nlayers=2,
        g1_dim=6,
        g2_dim=4,
        axis_neuron=2,
        update_g1_has_attn=repformer_attn,
        update_g2_has_attn=repformer_attn,
        attn1_hidden=4,
        attn1_nhead=2,
        attn2_hidden=4,
        attn2_nhead=2,
    )
    cfg = {
        "ntypes": ntypes,
        "repinit": repinit,
        "repformer": repformer,
        "precision": "float64",
        "seed": 42,
    }
    cfg.update(kwargs)
    return DescrptDPA2(**cfg)


def _system(seed: int = 0, nf: int = 1, nloc: int = 8, box_size: float = 6.0):
    """Small 2-type periodic system."""
    rng = np.random.default_rng(seed)
    coord = rng.random((nf, nloc, 3)) * box_size
    atype = rng.integers(0, 2, size=(nf, nloc)).astype(np.int64)
    box = np.tile((np.eye(3) * box_size).reshape(1, 9), (nf, 1))
    return coord, atype, box


def _dense_quartet(descr: DescrptDPA2, coord, atype, box):
    return extend_input_and_build_neighbor_list(
        coord,
        atype,
        descr.get_rcut(),
        descr.get_sel(),
        mixed_types=descr.mixed_types(),
        box=box,
    )


class TestDPA2UsesGraphLowerGates:
    def test_default_true(self) -> None:
        descr = _make_dpa2()
        assert descr.uses_graph_lower() is True

    def test_use_three_body_false(self) -> None:
        descr = _make_dpa2(use_three_body=True)
        assert descr.uses_graph_lower() is False

    def test_disable_graph_lower(self) -> None:
        descr = _make_dpa2()
        assert descr.uses_graph_lower() is True
        descr.disable_graph_lower()
        assert descr.uses_graph_lower() is False
        # sticky: cannot be re-enabled
        assert descr.uses_graph_lower() is False

    def test_compress_gate(self) -> None:
        descr = _make_dpa2()
        assert descr.uses_graph_lower() is True
        descr.compress = True
        assert descr.uses_graph_lower() is False


class TestDPA2AdapterBitExact:
    """The money test: the dense->graph adapter must be BIT-EXACT vs the
    dense body for ANY sel, including a deliberately BINDING repformer sel
    (the slot mask in ``call_graph``'s ``_block_graph`` replicates
    ``build_multiple_neighbor_list``'s ``nlist[:, :, :ns]`` slicing).
    """

    @pytest.mark.parametrize(
        "repformer_nsel,box_size,nloc,seed",
        [
            (
                150,
                6.0,
                8,
                21,
            ),  # non-binding: repformer sel comfortably covers all neighbors
            (3, 3.0, 12, 22),  # binding: dense cluster, repformer rcut truncates hard
        ],
    )
    def test_adapter_bitexact_any_sel(
        self, repformer_nsel, box_size, nloc, seed
    ) -> None:
        descr = _make_dpa2(repinit_nsel=200, repformer_nsel=repformer_nsel)
        coord, atype, box = _system(seed=seed, nloc=nloc, box_size=box_size)
        ext_coord, ext_atype, mapping, nlist = _dense_quartet(descr, coord, atype, box)
        nf, n_loc, nnei = nlist.shape

        # confirm the "binding" case actually truncates (otherwise the test
        # would vacuously pass without exercising the slot mask): count real
        # neighbors within the repformer rcut among ALL nnei outer slots and
        # compare against the configured repformer nsel.
        rc = descr.repformers.get_rcut()
        ns = descr.repformers.get_nsel()
        full_sub = build_multiple_neighbor_list(ext_coord, nlist, [rc], [nnei])[
            get_multiple_nlist_key(rc, nnei)
        ]
        real_counts = (full_sub >= 0).sum(axis=-1)
        binds = bool(np.any(real_counts > ns))
        assert binds == (repformer_nsel == 3)

        dense = descr._call_dense(ext_coord, ext_atype, nlist, mapping=mapping)
        adapter = descr._call_graph_adapter(ext_coord, ext_atype, nlist, mapping)

        dense_g1, dense_rot_mat, dense_g2, dense_h2, dense_sw = dense
        adapter_g1, adapter_rot_mat, adapter_g2, adapter_h2, adapter_sw = adapter

        np.testing.assert_allclose(adapter_g1, dense_g1, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            adapter_rot_mat, dense_rot_mat, rtol=1e-12, atol=1e-12
        )
        assert adapter_g2 is None
        assert adapter_h2 is None
        assert adapter_sw.shape == dense_sw.shape == (nf, n_loc, ns)
        np.testing.assert_allclose(adapter_sw, dense_sw, rtol=1e-12, atol=1e-12)
        assert not np.any(np.isnan(adapter_g1))
        assert not np.any(np.isnan(adapter_rot_mat))

    def test_adapter_divergence_davg_nonzero_documented(self) -> None:
        """Pin the ONE known bit-exactness exception (see the ``Notes``
        block on :meth:`DescrptDPA2.call_graph`'s docstring): when
        ``set_davg_zero=False`` (nonzero ``repinit.mean``) AND
        ``exclude_types == []``, the dense
        ``DescrptBlockSeAtten.call`` (``attn_layer == 0``, always the case
        for DPA2's ``repinit``) leaks a padding-slot residual into its
        output -- ``PairExcludeMask.build_type_exclude_mask`` short-circuits
        to all-ones when nothing is excluded, so real nlist padding is never
        zeroed out of ``rr`` before it reaches ``gr``, and once
        ``mean != 0`` the (masked-to-zero-then-mean-subtracted) padding rows
        become nonzero garbage that depends on the padding COUNT (i.e. on
        ``sel``, not on real physics). The graph path masks padding/excluded
        edges out before every ``segment_sum`` and does not reproduce this
        leak -- its result is the physically correct one. This regime is
        NOT covered by ``test_adapter_bitexact_any_sel``'s bit-exactness
        guarantee, and is not fixed here (pre-existing dense-body bug, see
        ``.superpowers/sdd/task-7-report.md``'s "Known limitation").

        This test does not attempt to reproduce the "atom-at-extended-
        index-0" framing verbatim -- the leak is triggered by ANY real nlist
        padding once ``mean != 0``, independent of where index 0 physically
        sits (verified empirically: the same small, non-clustered system
        used elsewhere in this file already has padding, since
        ``repinit_nsel=200`` comfortably exceeds each atom's real neighbor
        count). What matters, and what this test pins, is: (a) with
        ``mean == 0`` (the ``set_davg_zero=True``-equivalent control) the
        adapter stays 1e-12 bit-exact vs dense, exactly like
        ``test_adapter_bitexact_any_sel``; (b) with ``mean != 0`` and
        ``exclude_types == []`` and real padding present, the adapter and
        dense DIVERGE by a non-trivial amount -- a deliberate, documented
        record of the dense-side bug, not a regression in the adapter.
        """
        repinit_nsel = 200
        descr = _make_dpa2(repinit_nsel=repinit_nsel, repformer_nsel=150)
        assert descr.repinit.exclude_types == []
        coord, atype, box = _system(seed=41, nloc=8, box_size=6.0)
        ext_coord, ext_atype, mapping, nlist = _dense_quartet(descr, coord, atype, box)

        # self-check: real nlist padding is actually exercised (otherwise
        # this test would be vacuous -- no padding, nothing to leak).
        real_counts = (nlist != -1).sum(axis=-1)
        assert bool(np.any(real_counts < repinit_nsel))

        # control: mean == 0 (fresh descriptor's default, i.e. the
        # set_davg_zero=True-equivalent state) -> bit-exact, as elsewhere.
        dense0 = descr._call_dense(ext_coord, ext_atype, nlist, mapping=mapping)
        adapter0 = descr._call_graph_adapter(ext_coord, ext_atype, nlist, mapping)
        np.testing.assert_allclose(adapter0[0], dense0[0], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(adapter0[1], dense0[1], rtol=1e-12, atol=1e-12)

        # set repinit's mean to a nonzero constant (mirrors how other tests
        # in this file assign davg/dstd directly; slot-independent per
        # TestRepformersBlockSlotIndependence's proven invariant).
        nnei_repinit = descr.repinit.get_nsel()
        descr.repinit.mean = np.full(
            (descr.ntypes, nnei_repinit, 4), 0.3, dtype=np.float64
        )

        dense1 = descr._call_dense(ext_coord, ext_atype, nlist, mapping=mapping)
        adapter1 = descr._call_graph_adapter(ext_coord, ext_atype, nlist, mapping)
        max_abs_diff = float(np.max(np.abs(adapter1[0] - dense1[0])))
        # divergence documented, not a regression: dense leaks a
        # padding-count-dependent residual that the graph correctly excludes.
        assert max_abs_diff > 1e-6
        assert not np.any(np.isnan(adapter1[0]))
        assert not np.any(np.isnan(adapter1[1]))


class TestDPA2BlockMaskReplicatesMultipleNlist:
    def test_block_mask_formula_replicates_multiple_nlist(self) -> None:
        """Standalone re-implementation of the ``_block_graph`` mask formula
        (dist filter + slot cap) -- kept as a cheap, fast sanity check of the
        FORMULA, but it never calls the shipped code, so it cannot catch a
        regression in ``_block_graph``'s actual slicing/masking. See
        ``test_block_graph_seam_matches_dense_sub_nlist`` below for the test
        that exercises the real shipped code path.
        """
        descr = _make_dpa2(repinit_nsel=200, repformer_nsel=5)
        coord, atype, box = _system(seed=31, nloc=12, box_size=3.0)
        ext_coord, ext_atype, mapping, nlist = _dense_quartet(descr, coord, atype, box)
        nf, nloc, nnei = nlist.shape

        graph, _ = graph_from_dense_quartet(ext_coord, ext_atype, nlist, mapping)
        dist = np.linalg.norm(np.asarray(graph.edge_vec), axis=-1)
        edge_mask = np.asarray(graph.edge_mask)

        rc = descr.repformers.get_rcut()
        ns = descr.repformers.get_nsel()
        e_ax = edge_mask.shape[0]
        slot = np.arange(e_ax) % nnei
        # graph analogue of DescrptDPA2.call_graph's ``_block_graph``: dist
        # mask always, slot mask when static_nnei (== nnei here) is set.
        block_mask = edge_mask & (dist <= rc) & (slot < ns)
        block_mask = block_mask.reshape(nf, nloc, nnei)

        # binding-sel sanity: without the slot cap, more than `ns` neighbors
        # survive the dist filter for at least one center (otherwise this
        # test would not exercise the slot mask at all).
        dist_only_mask = (edge_mask & (dist <= rc)).reshape(nf, nloc, nnei)
        assert np.any(dist_only_mask.sum(axis=-1) > ns)

        sub_nlist = build_multiple_neighbor_list(ext_coord, nlist, [rc], [ns])[
            get_multiple_nlist_key(rc, ns)
        ]
        expected = np.zeros((nf, nloc, nnei), dtype=bool)
        expected[:, :, :ns] = sub_nlist >= 0
        np.testing.assert_array_equal(block_mask, expected)

    def test_block_graph_seam_matches_dense_sub_nlist(self) -> None:
        """Exercise the REAL shipped ``_block_graph`` closure (private,
        reachable only through ``DescrptDPA2.call_graph``) via the public
        seam, on a fixture where the repformer block's ``(rc, ns)`` genuinely
        truncate the outer graph.

        Strategy: run the shipped ``descr.call_graph(...)`` end to end (this
        is what ``_call_graph_adapter`` calls in production), then
        independently reconstruct a REFERENCE for "what the repformers block
        saw" by:

        1. Replaying the repinit stage with the SAME public
           ``descr.repinit.call_graph`` call the shipped code makes
           (repinit's own ``nsel`` (200) is >= the outer ``static_nnei``
           here, so ``_block_graph`` takes its mask-only fast path -- no
           slicing to replicate, just the dist mask), to get the g1 that
           feeds into repformers.
        2. Building a graph PRE-TRUNCATED BY HAND to the dense
           ``build_multiple_neighbor_list`` sub-nlist (the same width-``ns``
           sub-nlist ``_call_dense`` itself uses), via
           ``graph_from_dense_quartet``.
        3. Calling the repformer BLOCK's own public ``call_graph`` on that
           pre-truncated graph.

        If ``_block_graph``'s internal slice+mask selected exactly the same
        edges as the dense sub-nlist, the shipped end-to-end output and this
        by-hand reference must be bit-identical.
        """
        import dataclasses

        descr = _make_dpa2(repinit_nsel=200, repformer_nsel=5)
        assert descr.add_tebd_to_repinit_out is False  # keeps the repinit replay simple
        coord, atype, box = _system(seed=31, nloc=12, box_size=3.0)
        ext_coord, ext_atype, mapping, nlist = _dense_quartet(descr, coord, atype, box)
        nf, nloc, nnei = nlist.shape

        rc = descr.repformers.get_rcut()
        ns = descr.repformers.get_nsel()
        full_sub = build_multiple_neighbor_list(ext_coord, nlist, [rc], [nnei])[
            get_multiple_nlist_key(rc, nnei)
        ]
        real_counts = (full_sub >= 0).sum(axis=-1)
        # non-vacuity: the slot cap must actually be exercised, otherwise
        # this test would pass even with a broken slice.
        assert np.any(real_counts > ns)

        graph, atype_local = graph_from_dense_quartet(
            ext_coord, ext_atype, nlist, mapping
        )

        # 1) the real, shipped, end-to-end code path.
        g1_full, rot_mat_full, sw_full = descr.call_graph(
            graph, atype_local, static_nnei=nnei, return_sw=True
        )

        # 2) replay repinit via the same public call the shipped code makes
        # (mask-only fast path: repinit's own nsel >= the outer static_nnei).
        tebd_table = descr.type_embedding.call()
        dist = np.linalg.norm(np.asarray(graph.edge_vec), axis=-1)
        repinit_mask = np.asarray(graph.edge_mask) & (dist <= descr.repinit.get_rcut())
        repinit_graph = dataclasses.replace(graph, edge_mask=repinit_mask)
        g1_repinit, _ = descr.repinit.call_graph(
            repinit_graph, atype_local, type_embedding=tebd_table, static_nnei=nnei
        )
        g1_repinit = descr.g1_shape_tranform(g1_repinit)

        # 3) hand-pre-truncate to the dense sub-nlist's kept entries, and run
        # the repformer BLOCK's own public call_graph on it directly.
        sub_nlist = build_multiple_neighbor_list(ext_coord, nlist, [rc], [ns])[
            get_multiple_nlist_key(rc, ns)
        ]
        graph_ref, atype_local_ref = graph_from_dense_quartet(
            ext_coord, ext_atype, sub_nlist, mapping
        )
        np.testing.assert_array_equal(
            np.asarray(atype_local_ref), np.asarray(atype_local)
        )
        g1_ref, _g2_ref, _h2_ref, rot_mat_ref, sw_ref = descr.repformers.call_graph(
            graph_ref, atype_local, g1_repinit, comm_dict=None, static_nnei=ns
        )
        if descr.concat_output_tebd:
            g1_inp = np.asarray(tebd_table)[np.asarray(atype_local)]
            g1_ref = np.concatenate([np.asarray(g1_ref), g1_inp], axis=-1)

        # the shipped internal slice+mask must have selected EXACTLY the
        # edges the dense sub-nlist keeps -- bit-identical, not just close.
        np.testing.assert_allclose(g1_full, g1_ref, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(rot_mat_full, rot_mat_ref, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(sw_full, sw_ref, rtol=0.0, atol=0.0)


class TestDPA2CallRouting:
    def test_call_routing_graph_eligible(self) -> None:
        # Even for a graph-ELIGIBLE config, the dense ``call`` runs the dense
        # body -- it is the cross-backend consistency reference and must match
        # the tf/pt/pd/jax dense descriptors bit-for-bit. DPA2's repinit is
        # always ``attn_layer == 0``, where ``_call_graph_adapter`` diverges
        # from ``_call_dense`` for non-trivial statistics (the accepted
        # padding-leak divergence, see DescrptDPA2.call / call_graph Notes).
        # The graph-native route is reached through ``call_graph``, never
        # ``call``.
        descr = _make_dpa2(repinit_nsel=200, repformer_nsel=150)
        assert descr.uses_graph_lower() is True
        coord, atype, box = _system(seed=41, nloc=8, box_size=6.0)
        ext_coord, ext_atype, mapping, nlist = _dense_quartet(descr, coord, atype, box)

        called = descr.call(ext_coord, ext_atype, nlist, mapping=mapping)
        dense = descr._call_dense(ext_coord, ext_atype, nlist, mapping=mapping)
        for c, d in zip(called, dense, strict=True):
            if c is None or d is None:
                assert c is None and d is None
            else:
                np.testing.assert_array_equal(c, d)

    def test_call_routing_three_body_dense(self) -> None:
        descr = _make_dpa2(repinit_nsel=200, repformer_nsel=150, use_three_body=True)
        assert descr.uses_graph_lower() is False
        coord, atype, box = _system(seed=42, nloc=8, box_size=6.0)
        ext_coord, ext_atype, mapping, nlist = _dense_quartet(descr, coord, atype, box)

        called = descr.call(ext_coord, ext_atype, nlist, mapping=mapping)
        dense = descr._call_dense(ext_coord, ext_atype, nlist, mapping=mapping)
        for c, d in zip(called, dense, strict=True):
            if c is None or d is None:
                assert c is None and d is None
            else:
                np.testing.assert_array_equal(c, d)


def _make_energy_model(
    repinit_nsel: int = 200, repformer_nsel: int = 150
) -> EnergyModel:
    # repformer_attn=False: mirrors test_dpa1_graph_model_energy.py's own
    # choice of attn_layer=0 for its carry-all parity fixture. The
    # CARRY-ALL graph builder has no padding slots, so smooth attention's
    # softmax there is genuinely sel-independent (real neighbors only) --
    # by design DIFFERENT from the dense body, which keeps sel-padding
    # slots in its softmax denominator (see DescrptDPA1.call_graph's
    # Notes). That divergence is intentional and orthogonal to what this
    # test checks (non-attention carry-all/dense parity at non-binding
    # sel); the shape-static adapter path (_call_graph_adapter, covered by
    # TestDPA2AdapterBitExact) is the one that reproduces the dense
    # attention exactly, including with attention enabled.
    ds = _make_dpa2(
        repinit_nsel=repinit_nsel, repformer_nsel=repformer_nsel, repformer_attn=False
    )
    ft = InvarFitting(
        "energy",
        ds.get_ntypes(),
        ds.get_dim_out(),
        1,
        mixed_types=ds.mixed_types(),
    )
    return EnergyModel(ds, ft, type_map=["foo", "bar"])


class TestDPA2ModelEnergyCarryAll:
    """Model-level: ``EnergyModel(dpa2).call_common(...,
    neighbor_graph_method="dense")`` vs the dense route at NON-binding sel
    (mirrors ``test_dpa1_graph_model_energy.py``).
    """

    @pytest.mark.parametrize("periodic", [True, False])
    def test_energy_parity_non_binding_sel(self, periodic) -> None:
        rng = np.random.default_rng(51)
        nloc = 8
        coord = rng.normal(size=(1, nloc, 3)) * 1.5
        atype = np.array([[0, 1, 0, 1, 0, 1, 0, 1]], dtype=np.int64)
        box = None
        if periodic:
            # large box so the cell is essentially non-periodic for rcut=4.0
            box = np.eye(3).reshape(1, 9) * 20.0
        model = _make_energy_model()

        dense = model.call_common(coord, atype, box, neighbor_graph_method="legacy")
        graph = model.call_common(coord, atype, box, neighbor_graph_method="dense")

        np.testing.assert_allclose(
            graph["energy_redu"], dense["energy_redu"], rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            graph["energy"], dense["energy"], rtol=1e-12, atol=1e-12
        )
        np.testing.assert_array_equal(graph["mask"], dense["mask"])


def _make_attn_energy_model() -> EnergyModel:
    """Attention-enabled twin of :func:`_make_energy_model` (both repformer
    attention channels ON) for the phantom-count-compensated smooth-attention
    tests below.
    """
    ds = _make_dpa2(repinit_nsel=200, repformer_nsel=150, repformer_attn=True)
    ft = InvarFitting(
        "energy",
        ds.get_ntypes(),
        ds.get_dim_out(),
        1,
        mixed_types=ds.mixed_types(),
    )
    return EnergyModel(ds, ft, type_map=["foo", "bar"])


class TestDPA2AttentionCarryAllSmoothParity:
    """Phantom-count-compensated smooth attention on the carry-all graph route.

    The dense smooth-attention softmax keeps exactly ``sel - n_real`` phantom
    terms (each ``exp(-attnw_shift)``) in every denominator -- a count that
    never changes with geometry. The graph route instead used one phantom per
    PRESENT edge (a geometry-dependent count): outputs differed from dense by
    ``O(sel * exp(-attnw_shift)) ~ 1e-7`` even at non-binding sel, and the
    energy jumped by ``O(exp(-attnw_shift))`` whenever an edge entered or left
    the graph at the model cutoff. With the compensation (masked pairs
    excluded from the softmax, ``max(sel - n_real, 0)`` phantoms added), the
    denominator is term-for-term the dense one at non-binding sel: parity at
    1e-12 AND exact continuity at the cutoff.
    """

    def test_attention_energy_parity_non_binding_sel(self) -> None:
        rng = np.random.default_rng(53)
        nloc = 8
        coord = rng.normal(size=(1, nloc, 3)) * 1.5
        atype = np.array([[0, 1, 0, 1, 0, 1, 0, 1]], dtype=np.int64)
        model = _make_attn_energy_model()

        dense = model.call_common(coord, atype, None, neighbor_graph_method="legacy")
        graph = model.call_common(coord, atype, None, neighbor_graph_method="dense")

        np.testing.assert_allclose(
            graph["energy_redu"], dense["energy_redu"], rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            graph["energy"], dense["energy"], rtol=1e-12, atol=1e-12
        )

    def test_attention_energy_smooth_at_model_cutoff(self) -> None:
        """Graph-route energy is continuous when an atom crosses the model
        cutoff (repinit rcut=4.0). A second atom sits inside the repformer
        cutoff so the attention softmaxes have >=2 members (the denominator-
        jump scenario). Without the compensation the jump is
        ``O(exp(-attnw_shift)) ~ 1e-10``; with it, only float-reassociation
        noise remains.
        """
        model = _make_attn_energy_model()
        atype = np.array([[0, 1, 1]], dtype=np.int64)
        rcut = 4.0  # _make_dpa2 default repinit (model) rcut

        def energy(r: float) -> float:
            coord = np.array(
                [[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, r, 0.0]]],
                dtype=np.float64,
            )
            ret = model.call_common(coord, atype, None, neighbor_graph_method="dense")
            return float(np.sum(ret["energy_redu"]))

        eps = 1e-8
        jump = abs(energy(rcut + eps) - energy(rcut - eps))
        assert jump < 1e-13, f"graph-route energy jump {jump:.3e} at rcut"


class TestNodeOwnershipMask:
    """Direct unit tests of :func:`node_ownership_mask` -- the per-frame
    block arithmetic (``cumulative_sum``/``take``) is where bugs hide, so
    this is exercised independently of any model.
    """

    def test_single_frame(self) -> None:
        n_node = np.array([5], dtype=np.int64)
        n_local = np.array([3], dtype=np.int64)
        mask = node_ownership_mask(n_node, n_local, n_total=5)
        np.testing.assert_array_equal(mask, np.array([True, True, True, False, False]))

    def test_multi_frame_different_n_local(self) -> None:
        # frame 0: 3 nodes, owns 2 -> [T, T, F]
        # frame 1: 2 nodes, owns 1 -> [T, F]
        n_node = np.array([3, 2], dtype=np.int64)
        n_local = np.array([2, 1], dtype=np.int64)
        mask = node_ownership_mask(n_node, n_local, n_total=5)
        np.testing.assert_array_equal(mask, np.array([True, True, False, True, False]))

    def test_all_owned_is_all_true(self) -> None:
        n_node = np.array([4, 3], dtype=np.int64)
        n_local = n_node.copy()
        mask = node_ownership_mask(n_node, n_local, n_total=7)
        np.testing.assert_array_equal(mask, np.ones(7, dtype=bool))

    def test_zero_owned_frame(self) -> None:
        # a frame that owns nothing (all halo) -- degenerate but must not crash.
        n_node = np.array([2, 3], dtype=np.int64)
        n_local = np.array([0, 2], dtype=np.int64)
        mask = node_ownership_mask(n_node, n_local, n_total=5)
        np.testing.assert_array_equal(mask, np.array([False, False, True, True, False]))


class TestOwnedNodeMaskEnergyReduction:
    """``n_local`` (owned-node mask) in the graph output reduction (Task 9):
    halo rows (index >= n_local[frame]) must be excluded from the
    DIFFERENTIATED per-frame energy, while ``atom_energy`` itself stays FULL.
    """

    def _make_graph_and_model(self, nloc: int = 5, seed: int = 3):
        rng = np.random.default_rng(seed)
        coord = rng.normal(size=(1, nloc, 3)) * 1.5
        atype = np.array([[ii % 2 for ii in range(nloc)]], dtype=np.int64)
        model = _make_energy_model()
        ext_coord, ext_atype, mapping, nlist = extend_input_and_build_neighbor_list(
            coord,
            atype,
            model.get_rcut(),
            model.get_sel(),
            mixed_types=model.mixed_types(),
            box=None,
        )
        ng = from_dense_quartet(ext_coord, nlist, mapping)
        atype_local = ext_atype.reshape(-1)[:nloc]
        return model, ng, atype_local

    def test_owned_mask_energy_reduction(self) -> None:
        """1 frame, 5 nodes, n_local=3: energy_redu == sum(atom_energy[:3]),
        and the difference from the unmasked (``n_local=None``) energy_redu
        equals sum(atom_energy[3:5]); ``atom_energy`` itself is unchanged.
        """
        model, ng, atype_local = self._make_graph_and_model(nloc=5)

        out_full = model.call_lower_graph(
            atype=atype_local,
            n_node=ng.n_node,
            edge_index=ng.edge_index,
            edge_vec=ng.edge_vec,
            edge_mask=ng.edge_mask,
        )
        n_local = np.array([3], dtype=np.int64)
        out_masked = model.call_lower_graph(
            atype=atype_local,
            n_node=ng.n_node,
            edge_index=ng.edge_index,
            edge_vec=ng.edge_vec,
            edge_mask=ng.edge_mask,
            n_local=n_local,
        )

        # atom_energy (per-node) is FULL and byte-identical regardless of n_local.
        np.testing.assert_allclose(
            out_masked["energy"], out_full["energy"], rtol=1e-12, atol=1e-12
        )

        atom_energy = out_full["energy"].reshape(-1)
        owned_sum = atom_energy[:3].sum()
        halo_sum = atom_energy[3:].sum()

        np.testing.assert_allclose(
            out_masked["energy_redu"].reshape(-1),
            [owned_sum],
            rtol=1e-12,
            atol=1e-12,
        )
        diff = (out_full["energy_redu"] - out_masked["energy_redu"]).reshape(-1)
        np.testing.assert_allclose(diff, [halo_sum], rtol=1e-12, atol=1e-12)

    def test_n_local_none_is_byte_identical(self) -> None:
        """Regression: omitting ``n_local`` (default ``None``) reproduces the
        pre-Task-9 unmasked reduction exactly.
        """
        model, ng, atype_local = self._make_graph_and_model(nloc=5, seed=11)

        out_default = model.call_lower_graph(
            atype=atype_local,
            n_node=ng.n_node,
            edge_index=ng.edge_index,
            edge_vec=ng.edge_vec,
            edge_mask=ng.edge_mask,
        )
        out_explicit_none = model.call_lower_graph(
            atype=atype_local,
            n_node=ng.n_node,
            edge_index=ng.edge_index,
            edge_vec=ng.edge_vec,
            edge_mask=ng.edge_mask,
            n_local=None,
        )
        np.testing.assert_array_equal(
            out_default["energy_redu"], out_explicit_none["energy_redu"]
        )
        np.testing.assert_array_equal(
            out_default["energy"], out_explicit_none["energy"]
        )


class TestGraphAttentionCutoffContinuityBindingSel:
    """Cutoff continuity when the carry-all degree EXCEEDS ``sel``.

    OutisLi repro: repformer ``rcut=2, nsel=1``, one neighbor fixed at
    ``r=1`` and a second crossing ``r=2`` -- just inside the cutoff every
    center has 2 neighbors > sel=1, so the CLAMPED phantom count was zero
    on both sides of the crossing and the departing pair/edge removed a
    finite ``exp(-attnw_shift)`` softmax-denominator term: the full-model
    energy stepped by ~-6.0e-10 (LocalAtten-only -3.6e-9, Atten2Map-only
    +4.9e-9 -- independent defects at repformers.py's two ``call_graph``
    sites).  The SIGNED count ``sel - n_real`` subtracts the excess beyond
    sel, so the boundary increment vanishes for arbitrary degree.  g1
    (LocalAtten) and g2 (Atten2Map) attention are exercised SEPARATELY.
    """

    def _descriptor_delta(self, *, g1_attn: bool, g2_attn: bool, eps: float):
        from deepmd.dpmodel.utils.neighbor_graph import (
            build_neighbor_graph,
        )

        repinit = RepinitArgs(
            rcut=4.0,
            rcut_smth=0.5,
            nsel=20,
            neuron=[6, 12],
            axis_neuron=2,
            tebd_dim=4,
        )
        repformer = RepformerArgs(
            rcut=2.0,
            rcut_smth=0.5,
            nsel=1,  # BINDING: degree 2 > sel 1 just inside the cutoff
            nlayers=2,
            g1_dim=6,
            g2_dim=4,
            axis_neuron=2,
            update_g1_has_attn=g1_attn,
            update_g2_has_attn=g2_attn,
            attn1_hidden=4,
            attn1_nhead=2,
            attn2_hidden=4,
            attn2_nhead=2,
        )
        descr = DescrptDPA2(
            ntypes=2,
            repinit=repinit,
            repformer=repformer,
            precision="float64",
            seed=42,
        )
        te = descr.type_embedding.call()
        atype = np.array([[0, 1, 1]], dtype=np.int64)
        outs = []
        for d in (2.0 - eps, 2.0 + eps):
            coord = np.zeros((1, 3, 3), dtype=np.float64)
            coord[0, 1, 0] = 1.0  # fixed neighbor at r = 1
            coord[0, 2, 0] = d  # crossing neighbor at r = 2 -+ eps
            g = build_neighbor_graph(coord, atype, None, descr.get_rcut())
            out, _ = descr.call_graph(g, atype.reshape(-1), type_embedding=te)
            outs.append(np.asarray(out))
        # anti-vacuity: just inside the cutoff, the center has 2 repformer
        # neighbors (r=1 and r=2-eps) > nsel=1 -- the binding regime where
        # the clamped scheme was discontinuous.
        return float(np.max(np.abs(outs[0] - outs[1])))

    @pytest.mark.parametrize(
        ("g1_attn", "g2_attn"),
        [(True, False), (False, True)],
        ids=["g1_local_atten", "g2_atten2map"],
    )
    def test_descriptor_continuous_at_repformer_cutoff(self, g1_attn, g2_attn):
        # The defect was a CONSTANT step (~5e-9) as eps -> 0; a smooth
        # descriptor changes only proportionally to eps.  At eps = 1e-12
        # anything above 1e-11 is a genuine discontinuity.
        delta_tiny = self._descriptor_delta(g1_attn=g1_attn, g2_attn=g2_attn, eps=1e-12)
        assert delta_tiny < 1e-11, (
            f"descriptor step {delta_tiny:.3e} across the repformer cutoff "
            "at binding sel (degree > sel): the attention softmax "
            "denominator is not smooth"
        )
        # scaling check: shrinking eps by 1e4 must shrink the (smooth)
        # difference, not plateau at a finite step.
        delta_small = self._descriptor_delta(g1_attn=g1_attn, g2_attn=g2_attn, eps=1e-8)
        assert delta_tiny < max(delta_small, 1e-13), (
            f"difference plateaus ({delta_small:.3e} -> {delta_tiny:.3e}): "
            "finite step at the cutoff"
        )
