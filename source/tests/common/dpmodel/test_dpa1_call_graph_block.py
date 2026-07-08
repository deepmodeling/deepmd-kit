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
