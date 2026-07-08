# SPDX-License-Identifier: LGPL-3.0-or-later
"""Bit-exact parity between the graph-native ``DescrptBlockSeAtten.call_graph``
(attn_layer=0) and the legacy dense ``DescrptBlockSeAtten.call`` on the SAME
neighbor list, for binding AND non-binding ``sel``.
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


class TestDpa1BlockCallGraph:
    def _make(self, sel, type_one_side=False):
        return DescrptDPA1(
            rcut=4.0,
            rcut_smth=0.5,
            sel=sel,
            ntypes=2,
            attn_layer=0,
            axis_neuron=2,
            neuron=[6, 12],
            type_one_side=type_one_side,
        )

    def setup_method(self) -> None:
        rng = np.random.default_rng(1)
        self.nloc = 4
        self.coord = rng.normal(size=(1, self.nloc, 3)) * 1.5
        self.atype = np.array([[0, 1, 0, 1]], dtype=np.int64)

    @pytest.mark.parametrize("type_one_side", [False, True])  # tebd concat branch
    @pytest.mark.parametrize("sel", [[20], [3]])  # non-binding AND binding
    def test_block_graph_equals_dense_any_sel(self, sel, type_one_side) -> None:
        """Graph block output is bit-exact with the dense block on the same nlist.

        ``type_one_side`` toggles the concat branch in the block: when True the
        per-edge feature concatenates only the NEIGHBOR tebd (no center tebd),
        so both the graph and dense paths must agree for either branch.
        """
        dd = self._make(sel, type_one_side=type_one_side)
        blk = dd.se_atten
        # build the dense nlist exactly as the descriptor would
        (
            ext_coord,
            ext_atype,
            mapping,
            nlist,
        ) = extend_input_and_build_neighbor_list(
            self.coord,
            self.atype,
            dd.get_rcut(),
            dd.get_sel(),
            mixed_types=dd.mixed_types(),
            box=None,
        )
        # type embedding as both paths use it
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
        graph_g, _rot_mat = blk.call_graph(
            ng,
            np.reshape(ext_atype, (-1,)),
            type_embedding=tebd,
        )
        np.testing.assert_allclose(
            graph_g.reshape(dense_g.shape),
            dense_g,
            rtol=1e-12,
            atol=1e-12,
        )

    # attn_layer > 0 is supported since NeighborGraph PR-D; parity is covered
    # by test_dpa1_graph_attention_parity.py (the fail-fast test was removed).

    def test_exclude_types_raises(self) -> None:
        """The graph block kernel fail-fasts for exclude_types (not yet applied)."""
        # the graph path does not yet apply type exclusion; it must fail-fast
        # rather than silently diverge from the dense path (which masks edges).
        dd = DescrptDPA1(
            rcut=4.0,
            rcut_smth=0.5,
            sel=[20],
            ntypes=2,
            attn_layer=0,
            exclude_types=[(0, 1)],
        )
        ng = from_dense_quartet(
            self.coord,
            -np.ones((1, self.nloc, 1), dtype=np.int64),  # any graph; guard fires first
            np.arange(self.nloc, dtype=np.int64)[None],
        )
        with pytest.raises(NotImplementedError):
            dd.se_atten.call_graph(
                ng, self.atype.reshape(-1), type_embedding=dd.type_embedding.call()
            )


class TestDpa1BlockCallGraphStrip:
    """Bit-exact parity between the graph-native ``call_graph`` and the dense
    ``call`` for ``tebd_input_mode='strip'``.

    Strip mode factorizes the per-neighbor feature as ``gg = gg_s*gg_t + gg_s``
    (radial embedding times type-pair strip embedding); it has no neighbor-axis
    coupling, so the graph translation is edge-for-edge and must be bit-exact
    with the dense block on the SAME neighbor list.
    """

    def setup_method(self) -> None:
        rng = np.random.default_rng(3)
        self.nloc = 4
        self.coord = rng.normal(size=(1, self.nloc, 3)) * 1.5
        self.atype = np.array([[0, 1, 0, 1]], dtype=np.int64)

    def _make(self, type_one_side: bool, smooth: bool, attn_layer: int) -> DescrptDPA1:
        return DescrptDPA1(
            rcut=4.0,
            rcut_smth=0.5,
            sel=[20],  # non-binding sel: carry-all graph == dense on real neighbors
            ntypes=2,
            attn_layer=attn_layer,
            axis_neuron=2,
            neuron=[6, 12],
            tebd_input_mode="strip",
            type_one_side=type_one_side,
            smooth_type_embedding=smooth,
        )

    def _assert_parity(self, dd: DescrptDPA1, compact: bool) -> None:
        (
            ext_coord,
            ext_atype,
            mapping,
            nlist,
        ) = extend_input_and_build_neighbor_list(
            self.coord,
            self.atype,
            dd.get_rcut(),
            dd.get_sel(),
            mixed_types=dd.mixed_types(),
            box=None,
        )
        tebd = dd.type_embedding.call()
        nf, nall = ext_atype.shape
        atype_embd_ext = np.reshape(
            np.take(tebd, np.reshape(ext_atype, (-1,)), axis=0),
            (nf, nall, dd.tebd_dim),
        )
        dense_g, *_ = dd.se_atten.call(
            nlist,
            ext_coord,
            ext_atype,
            atype_embd_ext=atype_embd_ext,
            mapping=None,
            type_embedding=tebd,
        )
        ng = from_dense_quartet(ext_coord, nlist, mapping, compact=compact)
        graph_g, _rot_mat = dd.se_atten.call_graph(
            ng,
            np.reshape(ext_atype, (-1,)),
            type_embedding=tebd,
        )
        assert not np.any(np.isnan(graph_g))
        np.testing.assert_allclose(
            graph_g.reshape(dense_g.shape),
            dense_g,
            rtol=1e-12,
            atol=1e-12,
        )

    @pytest.mark.parametrize(
        "type_one_side", [False, True]
    )  # two-side vs one-side strip table
    @pytest.mark.parametrize("smooth", [False, True])  # gg_t switch-smoothing branch
    def test_strip_attn0_equals_dense(self, type_one_side, smooth) -> None:
        """attn_layer=0: no attention, so strip parity is bit-exact for both smooth values."""
        dd = self._make(type_one_side, smooth, attn_layer=0)
        self._assert_parity(dd, compact=True)

    @pytest.mark.parametrize(
        "type_one_side", [False, True]
    )  # two-side vs one-side strip table
    def test_strip_attn2_equals_dense(self, type_one_side) -> None:
        """attn_layer=2, smooth=False: bit-exact (avoids by-design smooth softmax divergence)."""
        dd = self._make(type_one_side, smooth=False, attn_layer=2)
        self._assert_parity(dd, compact=False)
