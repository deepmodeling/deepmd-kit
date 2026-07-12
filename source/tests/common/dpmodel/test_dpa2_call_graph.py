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
from deepmd.dpmodel.model.ener_model import (
    EnergyModel,
)
from deepmd.dpmodel.utils.neighbor_graph import (
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
            (150, 6.0, 8, 21),  # non-binding: repformer sel comfortably covers all neighbors
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


class TestDPA2BlockMaskReplicatesMultipleNlist:
    def test_block_mask_replicates_multiple_nlist(self) -> None:
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


class TestDPA2CallRouting:
    def test_call_routing_graph_eligible(self) -> None:
        descr = _make_dpa2(repinit_nsel=200, repformer_nsel=150)
        assert descr.uses_graph_lower() is True
        coord, atype, box = _system(seed=41, nloc=8, box_size=6.0)
        ext_coord, ext_atype, mapping, nlist = _dense_quartet(descr, coord, atype, box)

        called = descr.call(ext_coord, ext_atype, nlist, mapping=mapping)
        adapter = descr._call_graph_adapter(ext_coord, ext_atype, nlist, mapping)
        for c, a in zip(called, adapter, strict=True):
            if c is None or a is None:
                assert c is None and a is None
            else:
                np.testing.assert_array_equal(c, a)

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


def _make_energy_model(repinit_nsel: int = 200, repformer_nsel: int = 150) -> EnergyModel:
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
