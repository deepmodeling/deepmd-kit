# SPDX-License-Identifier: LGPL-3.0-or-later
"""Graph-native se_atten attention (attn_layer > 0) vs the dense reference.

Regime-1 parity (NeighborGraph PR-D): the graph is built FROM the same dense
nlist (``from_dense_quartet``), so the neighbor sets are identical and the
graph attention must reproduce ``GatedAttentionLayer``/``NeighborGatedAttention``
bit-exactly (CPU rtol 1e-12) for ANY sel — binding or not.

The smooth branch needs the SHAPE-STATIC graph (``compact=False`` +
``static_nnei``): dense smooth keeps padding slots in the softmax DENOMINATOR
(weight ``exp(-attnw_shift)`` since ``sw = 0``), so bit-parity requires the
same padded pairs on the graph side. The compact (carry-all-like) form drops
padding pairs and is exercised on the non-smooth branch only.
"""

import numpy as np
import pytest

from deepmd.dpmodel.descriptor.dpa1 import (
    DescrptDPA1,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    from_dense_quartet,
)
from deepmd.dpmodel.utils.nlist import (
    extend_input_and_build_neighbor_list,
)

GLOBAL_SEED = 20260703


def _make(
    attn_layer,
    dotr=False,
    smooth=False,
    normalize=False,
    temperature=1.0,
    sel=(20,),
):
    # attention `smooth` is wired to smooth_type_embedding (NOT rcut_smth);
    # pass it explicitly — its default (True) would silently enable the
    # smooth branch in the non-smooth cases.
    return DescrptDPA1(
        rcut=4.0,
        rcut_smth=0.5,
        sel=list(sel),
        ntypes=2,
        neuron=[6, 12],
        axis_neuron=2,
        attn=8,
        attn_layer=attn_layer,
        attn_dotr=dotr,
        attn_mask=False,
        normalize=normalize,
        smooth_type_embedding=smooth,
        temperature=temperature,
        tebd_input_mode="concat",
        type_one_side=True,
        precision="float64",
        seed=GLOBAL_SEED,
    )


class TestGraphAttentionParity:
    def setup_method(self) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        self.nloc = 5
        self.coord = rng.normal(size=(1, self.nloc, 3)) * 1.5
        self.atype = np.array([[0, 1, 0, 1, 1]], dtype=np.int64)

    def _quartet(self, dd):
        return extend_input_and_build_neighbor_list(
            self.coord,
            self.atype,
            dd.get_rcut(),
            dd.get_sel(),
            mixed_types=dd.mixed_types(),
            box=None,
        )

    def _dense_vs_adapter(self, dd, rtol=1e-12):
        """Descriptor-level: legacy dense body vs the graph adapter (shape-static)."""
        ext_coord, ext_atype, mapping, nlist = self._quartet(dd)
        dense = dd._call_dense(ext_coord, ext_atype, nlist)
        graph = dd._call_graph_adapter(ext_coord, ext_atype, nlist, mapping)
        np.testing.assert_allclose(
            graph[0], dense[0], rtol=rtol, atol=rtol, err_msg="descriptor"
        )
        np.testing.assert_allclose(
            graph[1], dense[1], rtol=rtol, atol=rtol, err_msg="rot_mat"
        )

    # ── Task 5a/5b/5c/5e: full matrix on the shape-static adapter path ──────
    @pytest.mark.parametrize("sel", [(20,), (3,)])  # non-binding AND binding
    @pytest.mark.parametrize("attn_layer", [1, 2])  # single + stacked layers
    def test_core_layers_sel(self, attn_layer, sel) -> None:
        dd = _make(attn_layer, sel=sel)
        self._dense_vs_adapter(dd)

    @pytest.mark.parametrize("normalize", [False, True])  # q/k/v np_normalize
    @pytest.mark.parametrize("temperature", [None, 1.0])  # scaling source
    def test_normalize_temperature(self, normalize, temperature) -> None:
        dd = _make(1, normalize=normalize, temperature=temperature)
        self._dense_vs_adapter(dd)

    @pytest.mark.parametrize("dotr", [False, True])  # angular weighting
    @pytest.mark.parametrize("smooth", [False, True])  # switch-fn weighting
    def test_dotr_smooth(self, dotr, smooth) -> None:
        dd = _make(2, dotr=dotr, smooth=smooth, normalize=True, temperature=None)
        self._dense_vs_adapter(dd)

    # ── compact (carry-all-form) graph through the BLOCK kernel, non-smooth ──
    @pytest.mark.parametrize("attn_layer", [1, 2])  # single + stacked layers
    def test_block_compact_graph_no_smooth(self, attn_layer) -> None:
        dd = _make(attn_layer, dotr=True, normalize=True)
        blk = dd.se_atten
        ext_coord, ext_atype, mapping, nlist = self._quartet(dd)
        tebd = dd.type_embedding.call()
        nf, nall = ext_atype.shape
        atype_embd_ext = np.reshape(
            np.take(tebd, np.reshape(ext_atype, (-1,)), axis=0),
            (nf, nall, dd.tebd_dim),
        )
        dense_g, *_ = blk.call(
            nlist,
            ext_coord,
            ext_atype,
            atype_embd_ext=atype_embd_ext,
            mapping=None,
            type_embedding=tebd,
        )
        ng = from_dense_quartet(ext_coord, nlist, mapping)  # compact=True
        graph_g, _ = blk.call_graph(
            ng, np.reshape(ext_atype, (-1,)), type_embedding=tebd
        )
        np.testing.assert_allclose(
            graph_g.reshape(dense_g.shape), dense_g, rtol=1e-12, atol=1e-12
        )

    # ── smooth on the compact (carry-all) form: CLEAN DIVERGENCE by design ────
    def test_block_compact_graph_smooth_clean_divergence(self) -> None:
        """Carry-all smooth attention deliberately DIVERGES from dense.

        The dense smooth branch keeps sel-padding slots in the attention
        softmax DENOMINATOR at weight ``exp(-attnw_shift)``, which makes the
        dense output depend on ``sel`` itself (same physical neighbors,
        different sel => different output, up to ~1e-4). The carry-all graph
        drops those phantom terms — the sel-independent math (user decision
        2026-07-03, PR-D). Bit-parity (1e-12) is proven on the shape-static
        adapter (same padded pairs on both sides, ``test_dotr_smooth``); here
        we pin only that the compact form stays CLOSE to dense (the artifact
        is a bounded denominator perturbation) while NOT bit-equal.
        """
        dd = _make(1, smooth=True)
        blk = dd.se_atten
        ext_coord, ext_atype, mapping, nlist = self._quartet(dd)
        tebd = dd.type_embedding.call()
        nf, nall = ext_atype.shape
        atype_embd_ext = np.reshape(
            np.take(tebd, np.reshape(ext_atype, (-1,)), axis=0),
            (nf, nall, dd.tebd_dim),
        )
        dense_g, *_ = blk.call(
            nlist,
            ext_coord,
            ext_atype,
            atype_embd_ext=atype_embd_ext,
            mapping=None,
            type_embedding=tebd,
        )
        ng = from_dense_quartet(ext_coord, nlist, mapping)
        graph_g, _ = blk.call_graph(
            ng, np.reshape(ext_atype, (-1,)), type_embedding=tebd
        )
        graph_g = graph_g.reshape(dense_g.shape)
        # close (the artifact is a small denominator perturbation) ...
        np.testing.assert_allclose(graph_g, dense_g, rtol=1e-3, atol=1e-3)
        # ... but NOT bit-equal: the phantom-padding terms are really gone
        assert np.max(np.abs(graph_g - dense_g)) > 1e-9

    # ── torch namespace smoke (CLAUDE.md: catches numpy-weight leaks) ────────
    # NB: the smoke runs the BLOCK kernel with a torch type_embedding table;
    # the raw dpmodel adapter is numpy-weighted by design (pt_expt wraps it).
    def test_torch_block_matches_numpy(self) -> None:
        import torch

        dd = _make(2, dotr=True, smooth=True, normalize=True, temperature=None)
        blk = dd.se_atten
        ext_coord, ext_atype, mapping, nlist = self._quartet(dd)
        tebd = dd.type_embedding.call()
        ng = from_dense_quartet(ext_coord, nlist, mapping, compact=False)
        ref, _ = blk.call_graph(
            ng,
            np.reshape(ext_atype, (-1,)),
            type_embedding=tebd,
            static_nnei=nlist.shape[2],
        )
        ng_t = from_dense_quartet(
            torch.from_numpy(ext_coord),
            torch.from_numpy(nlist),
            torch.from_numpy(mapping),
            compact=False,
        )
        out, _ = blk.call_graph(
            ng_t,
            torch.from_numpy(np.reshape(ext_atype, (-1,))),
            type_embedding=torch.from_numpy(tebd),
            static_nnei=nlist.shape[2],
        )
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-12, atol=1e-12)


class TestGraphEligibility:
    def test_attention_concat_is_graph_eligible(self) -> None:
        assert _make(2).uses_graph_lower()

    def test_se_atten_v2_is_graph_eligible(self) -> None:
        """se_atten_v2 (tebd_input_mode='strip', smooth=True) is now graph-eligible.

        It is a DescrptDPA1 subclass with no exclude_types and no routing override,
        so admitting strip closes the 'se_atten_v2 is dense-only' gap. (Was: strip
        stayed dense.)
        """
        from deepmd.dpmodel.descriptor.se_atten_v2 import (
            DescrptSeAttenV2,
        )

        dd = DescrptSeAttenV2(rcut=4.0, rcut_smth=0.5, sel=[20], ntypes=2, attn_layer=2)
        assert dd.uses_graph_lower() is True

    def test_se_atten_v2_graph_equals_dense(self) -> None:
        """The graph-routed se_atten_v2 ``call`` is bit-exact with ``_call_dense``
        (the ``static_nnei`` adapter reproduces the dense phantom terms despite
        smooth=True) at a non-binding sel.
        """
        from deepmd.dpmodel.descriptor.se_atten_v2 import (
            DescrptSeAttenV2,
        )

        rng = np.random.default_rng(GLOBAL_SEED)
        nloc = 4
        coord = rng.normal(size=(1, nloc, 3)) * 1.5
        atype = np.array([[0, 1, 0, 1]], dtype=np.int64)
        dd = DescrptSeAttenV2(rcut=4.0, rcut_smth=0.5, sel=[20], ntypes=2, attn_layer=2)
        assert dd.uses_graph_lower() is True
        ext_coord, ext_atype, mapping, nlist = extend_input_and_build_neighbor_list(
            coord, atype, dd.get_rcut(), dd.get_sel(), mixed_types=True, box=None
        )
        routed = dd.call(ext_coord, ext_atype, nlist, mapping=mapping)
        dense = dd._call_dense(ext_coord, ext_atype, nlist)
        assert len(routed) == len(dense)
        for r, d in zip(routed, dense, strict=True):
            if r is None:
                assert d is None
                continue
            assert not np.any(np.isnan(r))
            np.testing.assert_allclose(r, d, rtol=1e-12, atol=1e-12)


class TestBindingSelDivergence:
    """At BINDING sel the carry-all graph attends over MORE neighbors than the
    sel-truncated dense path — outputs must differ (sanity, not parity;
    spec decision #17).
    """

    def test_carry_all_attention_differs_at_binding_sel(self) -> None:
        from deepmd.dpmodel.utils.neighbor_graph import (
            build_neighbor_graph,
        )

        rng = np.random.default_rng(GLOBAL_SEED)
        nloc = 6
        coord = rng.random((1, nloc, 3)) * 2.0  # dense blob => binding sel=2
        atype = np.array([[0, 1, 0, 1, 1, 0]], dtype=np.int64)
        dd = _make(2, dotr=True, sel=(2,))
        ext_coord, ext_atype, mapping, nlist = extend_input_and_build_neighbor_list(
            coord, atype, dd.get_rcut(), dd.get_sel(), mixed_types=True, box=None
        )
        assert (nlist >= 0).all(), "fixture must be sel-binding (all slots full)"
        tebd = dd.type_embedding.call()
        atype_embd_ext = np.reshape(
            np.take(tebd, np.reshape(ext_atype, (-1,)), axis=0),
            (1, ext_atype.shape[1], dd.tebd_dim),
        )
        dense_g, *_ = dd.se_atten.call(
            nlist,
            ext_coord,
            ext_atype,
            atype_embd_ext=atype_embd_ext,
            mapping=None,
            type_embedding=tebd,
        )
        graph = build_neighbor_graph(coord, atype, None, dd.get_rcut())
        graph_g, _ = dd.se_atten.call_graph(
            graph, atype.reshape(-1), type_embedding=tebd
        )
        assert np.max(np.abs(graph_g.reshape(dense_g.shape) - dense_g)) > 1e-6, (
            "carry-all attention must diverge from sel-truncated dense"
        )
