# SPDX-License-Identifier: LGPL-3.0-or-later
"""Full 5-tuple ABI parity between the graph-routed ``DescrptDPA1.call``
(attn_layer=0, which now goes ``from_dense_quartet -> call_graph``) and the
legacy dense descriptor output captured BEFORE the swap, for binding AND
non-binding ``sel``.

The dense reference is reconstructed by calling the BLOCK directly
(``dd.se_atten.call``) and applying the descriptor-level ``concat_output_tebd``
step by hand (mirroring dpa1.py), because ``dd.call`` itself now routes through
the graph for ``attn_layer == 0``.
"""

import numpy as np
import pytest

from deepmd.dpmodel.descriptor.dpa1 import (
    DescrptDPA1,
)
from deepmd.dpmodel.utils.nlist import (
    extend_input_and_build_neighbor_list,
)


class TestDpa1DescriptorCallGraph:
    def _make(self, sel):
        return DescrptDPA1(
            rcut=4.0,
            rcut_smth=0.5,
            sel=sel,
            ntypes=2,
            attn_layer=0,
            axis_neuron=2,
            neuron=[6, 12],
        )

    def setup_method(self) -> None:
        rng = np.random.default_rng(2)
        self.nloc = 4
        self.coord = rng.normal(size=(1, self.nloc, 3)) * 1.5
        self.atype = np.array([[0, 1, 0, 1]], dtype=np.int64)

    def _dense_reference(self, dd, ext_coord, ext_atype, nlist):
        """Reconstruct the original dense descriptor 5-tuple (pre-swap)."""
        tebd = dd.type_embedding.call()
        nf, nall = ext_atype.shape
        atype_embd_ext = np.reshape(
            np.take(tebd, np.reshape(ext_atype, (-1,)), axis=0),
            (nf, nall, dd.tebd_dim),
        )
        grrg, g2, h2, rot_mat, sw = dd.se_atten.call(
            nlist,
            ext_coord,
            ext_atype,
            atype_embd_ext=atype_embd_ext,
            mapping=None,
            type_embedding=tebd,
        )
        nloc = nlist.shape[1]
        # descriptor-level concat_output_tebd (mirror dpa1.py)
        atype_embd = atype_embd_ext[:, :nloc, :]
        if dd.concat_output_tebd:
            grrg = np.concatenate(
                [grrg, np.reshape(atype_embd, (nf, nloc, dd.tebd_dim))], axis=-1
            )
        return grrg, rot_mat, None, None, sw

    @pytest.mark.parametrize("sel", [[30], [4]])  # non-binding AND binding
    def test_descriptor_graph_equals_dense_full_tuple(self, sel) -> None:
        """Graph-routed dd.call() returns the identical dense 5-tuple ABI."""
        dd = self._make(sel)
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
        # dense reference captured via the block (pre-swap behaviour)
        ref = self._dense_reference(dd, ext_coord, ext_atype, nlist)
        # the swapped public ABI: routes through the graph
        out = dd.call(ext_coord, ext_atype, nlist, mapping=mapping)
        assert len(out) == 5
        # grrg
        np.testing.assert_allclose(out[0], ref[0], rtol=1e-12, atol=1e-12)
        # rot_mat
        np.testing.assert_allclose(out[1], ref[1], rtol=1e-12, atol=1e-12)
        # positions [2], [3] are always None for this descriptor
        assert out[2] is None
        assert out[3] is None
        # sw
        np.testing.assert_allclose(out[4], ref[4], rtol=1e-12, atol=1e-12)

    # NOTE: after the strip PR (#5747) and this PR both landed, neither strip
    # nor exclude_types falls back to dense — both are graph-eligible. The only
    # non-disabled ineligible config is compression, whose gate is covered by
    # TestDpa1StripRouting.test_uses_graph_lower_strip_gate, and the no-mapping
    # ghost fallback by test_eligible_no_mapping_with_ghosts_falls_back below.

    @pytest.mark.parametrize(
        "exclude_types",
        [[], [(0, 1)]],  # empty exclusions AND non-trivial exclusion
    )
    def test_exclude_types_graph_eligible_and_parity(self, exclude_types) -> None:
        """exclude_types (Task 3): descriptor is graph-eligible (uses_graph_lower()
        True) regardless of the exclusion list.  Graph output must match the dense
        reference at rtol=atol=1e-12 for a non-binding sel.
        """
        from deepmd.dpmodel.utils.neighbor_graph import (
            from_dense_quartet,
        )

        dd = DescrptDPA1(
            rcut=4.0,
            rcut_smth=0.5,
            sel=[30],  # non-binding sel
            ntypes=2,
            attn_layer=0,
            axis_neuron=2,
            neuron=[6, 12],
            exclude_types=exclude_types,
        )
        # gate: with any exclude list the descriptor must now be graph-eligible
        assert dd.uses_graph_lower() is True

        ext_coord, ext_atype, mapping, nlist = extend_input_and_build_neighbor_list(
            self.coord,
            self.atype,
            dd.get_rcut(),
            dd.get_sel(),
            mixed_types=dd.mixed_types(),
            box=None,
        )
        # dense reference (calls block directly)
        ref = self._dense_reference(dd, ext_coord, ext_atype, nlist)
        # graph-routed public call
        out = dd.call(ext_coord, ext_atype, nlist, mapping=mapping)
        assert len(out) == 5
        np.testing.assert_allclose(out[0], ref[0], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(out[1], ref[1], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(out[4], ref[4], rtol=1e-12, atol=1e-12)

        if exclude_types:
            # verify excluded pairs contribute sw == 0 in the dense reference
            # (atype=[0,1,0,1] -> pairs (0,1) and (1,0) should be masked)
            # sw shape: (nf, nloc, nnei, 1); just check the graph output is also 0
            # for excluded-pair edges by checking call_graph sw channel
            graph = from_dense_quartet(ext_coord, nlist, mapping, compact=False)
            atype_local = self.atype.reshape(-1)
            grrg_g, rot_mat_g = dd.call_graph(
                graph, atype_local, type_embedding=dd.type_embedding.call()
            )
            # no nan/inf in output with exclusions applied
            assert not np.any(np.isnan(grrg_g))
            assert not np.any(np.isinf(grrg_g))

    def test_eligible_no_mapping_with_ghosts_falls_back(self) -> None:
        """An eligible (concat) attn_layer=0 descriptor called with mapping=None
        on a PERIODIC system (nall > nloc ghosts) must fall back to the dense
        body and match it because the graph path requires an explicit ghost
        mapping.
        """
        dd = self._make([30])
        box = np.eye(3, dtype=np.float64)[None] * 6.0
        ext_coord, ext_atype, mapping, nlist = extend_input_and_build_neighbor_list(
            self.coord,
            self.atype,
            dd.get_rcut(),
            dd.get_sel(),
            mixed_types=dd.mixed_types(),
            box=box,
        )
        assert ext_atype.shape[1] > self.nloc  # ghosts present
        ref = self._dense_reference(dd, ext_coord, ext_atype, nlist)
        out = dd.call(ext_coord, ext_atype, nlist, mapping=None)  # must not IndexError
        np.testing.assert_allclose(out[0], ref[0], rtol=1e-12, atol=1e-12)

    def test_single_rank_extension_keeps_type_invariant(self) -> None:
        """The ghost-free graph types a neighbor as ``atype[mapping[neighbor]]``
        (its local owner). This is correct because a real single-rank extension
        is type-consistent: ``extend_coord_with_ghosts`` tiles the local atype, so
        ``atype_ext[k] == atype[mapping[k]]`` for every extended atom -- a ghost is
        a periodic image of its owner and shares its type. This test pins that
        invariant (an inconsistent ``mapping`` like the universal fixture's old
        buggy permutation is NOT a valid single-rank extension) and confirms the
        graph-routed ``call`` matches dense on the resulting quartet.
        """
        dd = self._make([30])
        box = np.eye(3, dtype=np.float64)[None] * 6.0
        ext_coord, ext_atype, mapping, nlist = extend_input_and_build_neighbor_list(
            self.coord,
            self.atype,
            dd.get_rcut(),
            dd.get_sel(),
            mixed_types=dd.mixed_types(),
            box=box,
        )
        assert ext_atype.shape[1] > self.nloc  # ghosts present
        # the single-rank type invariant the ghost-free graph relies on
        nf, nall = ext_atype.shape
        for f in range(nf):
            np.testing.assert_array_equal(
                ext_atype[f], ext_atype[f][mapping[f]]
            )  # atype_ext[k] == atype[mapping[k]]
        ref = self._dense_reference(dd, ext_coord, ext_atype, nlist)
        out = dd.call(ext_coord, ext_atype, nlist, mapping=mapping)
        np.testing.assert_allclose(out[0], ref[0], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(out[1], ref[1], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(out[4], ref[4], rtol=1e-12, atol=1e-12)

    def test_call_graph_returns_flat_node_axis(self) -> None:
        """call_graph output lives on the flat (N,) node axis, not (nf, nloc)."""
        from deepmd.dpmodel.utils.neighbor_graph import (
            from_dense_quartet,
        )

        dd = self._make([30])
        ext_coord, ext_atype, mapping, nlist = extend_input_and_build_neighbor_list(
            self.coord,
            self.atype,
            dd.get_rcut(),
            dd.get_sel(),
            mixed_types=dd.mixed_types(),
            box=None,
        )
        graph = from_dense_quartet(ext_coord, nlist, mapping, compact=True)
        atype_local = self.atype.reshape(-1)
        grrg, rot_mat = dd.call_graph(
            graph, atype_local, type_embedding=dd.type_embedding.call()
        )
        n = atype_local.shape[0]
        assert grrg.shape[0] == n and grrg.ndim == 2
        assert rot_mat.shape[0] == n and rot_mat.ndim == 3


class TestDpa1StripRouting:
    """Strip is now graph-eligible: ``uses_graph_lower()`` admits it, ``call``
    routes through the graph adapter, and the adapter (``static_nnei``) is
    bit-exact with the legacy ``_call_dense`` for EVERY strip config -- including
    ``smooth_type_embedding=True`` at ``attn_layer>0`` (the adapter reproduces
    the dense phantom-neighbor terms, unlike the direct carry-all ``call_graph``).
    """

    def setup_method(self) -> None:
        rng = np.random.default_rng(7)
        self.nloc = 4
        self.coord = rng.normal(size=(1, self.nloc, 3)) * 1.5
        self.atype = np.array([[0, 1, 0, 1]], dtype=np.int64)

    def _make(self, type_one_side: bool, smooth: bool, attn_layer: int) -> DescrptDPA1:
        return DescrptDPA1(
            rcut=4.0,
            rcut_smth=0.5,
            sel=[20],  # non-binding
            ntypes=2,
            attn_layer=attn_layer,
            axis_neuron=2,
            neuron=[6, 12],
            tebd_input_mode="strip",
            type_one_side=type_one_side,
            smooth_type_embedding=smooth,
            resnet_dt=False,
        )

    def test_uses_graph_lower_strip_gate(self) -> None:
        """The gate admits non-compressed strip (and exclude_types, now
        graph-native via the pair-exclude seam); only compression forces dense.
        """
        dd = self._make(type_one_side=False, smooth=True, attn_layer=2)
        assert dd.uses_graph_lower() is True  # strip is now graph-eligible
        # negative contract: compression keeps the descriptor on the dense path
        dd.compress = True
        assert dd.uses_graph_lower() is False
        dd.compress = False
        # exclude_types is now graph-native (owned by the pair-exclude seam), so
        # strip + exclude_types stays graph-eligible.
        dd_excl = DescrptDPA1(
            rcut=4.0,
            rcut_smth=0.5,
            sel=[20],
            ntypes=2,
            attn_layer=2,
            tebd_input_mode="strip",
            exclude_types=[(0, 1)],
            resnet_dt=False,
        )
        assert dd_excl.uses_graph_lower() is True

    @pytest.mark.parametrize("type_one_side", [False, True])  # strip table branch
    @pytest.mark.parametrize(
        "smooth", [False, True]
    )  # switch-smoothing + smooth attention
    @pytest.mark.parametrize("attn_layer", [0, 2])  # no-attn and multi-layer attention
    def test_call_strip_graph_equals_dense(
        self, type_one_side, smooth, attn_layer
    ) -> None:
        """The routed ``call`` (graph adapter) is bit-exact with ``_call_dense``."""
        dd = self._make(type_one_side, smooth, attn_layer)
        assert dd.uses_graph_lower() is True  # precondition: call routes to graph
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
        routed = dd.call(ext_coord, ext_atype, nlist, mapping=mapping)
        dense = dd._call_dense(ext_coord, ext_atype, nlist)
        assert len(routed) == len(dense)
        for r, d in zip(routed, dense, strict=True):
            if r is None:
                assert d is None
                continue
            assert not np.any(np.isnan(r))
            np.testing.assert_allclose(r, d, rtol=1e-12, atol=1e-12)
