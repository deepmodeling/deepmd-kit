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

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"tebd_input_mode": "strip"},  # strip tebd: graph unsupported -> dense
            {"exclude_types": [(0, 1)]},  # type exclusion: graph unsupported -> dense
        ],
    )
    def test_ineligible_config_falls_back_to_dense(self, kwargs) -> None:
        """attn_layer=0 configs the graph can't handle (strip tebd, exclude_types)
        must report uses_graph_lower()=False and run the dense body without
        raising (regression: Task-3 routing previously raised NotImplementedError).
        """
        dd = DescrptDPA1(
            rcut=4.0, rcut_smth=0.5, sel=[30], ntypes=2, attn_layer=0, **kwargs
        )
        assert dd.uses_graph_lower() is False
        ext_coord, ext_atype, mapping, nlist = extend_input_and_build_neighbor_list(
            self.coord,
            self.atype,
            dd.get_rcut(),
            dd.get_sel(),
            mixed_types=dd.mixed_types(),
            box=None,
        )
        out = dd.call(ext_coord, ext_atype, nlist, mapping=mapping)  # must not raise
        assert len(out) == 5

    def test_eligible_no_mapping_with_ghosts_falls_back(self) -> None:
        """An eligible (concat) attn_layer=0 descriptor called with mapping=None
        on a PERIODIC system (nall > nloc ghosts) must fall back to the dense
        body and match it (regression: the graph needs mapping for ghosts, the
        identity-mapping default previously indexed out of range).
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

    def test_inconsistent_mapping_stays_faithful_to_dense(self) -> None:
        """The dense->graph bridge must reproduce the dense 5-tuple even when the
        supplied ``mapping`` is INCONSISTENT (a ghost's extended type differs from
        its local owner's type). Real periodic systems never produce this -- a
        ghost is a periodic image of its owner -- but a synthetic external quartet
        can (e.g. the permuted ``mapping`` in the universal descriptor fixture).
        The dense path reads ``atype_ext[neighbor]`` directly, so the graph bridge
        must too (regression: it used ``atype[mapping[neighbor]]`` and diverged).
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
        # Corrupt EVERY ghost's extended type so ghost type != owner type, making
        # atype_ext[neighbor] != atype[mapping[neighbor]] for every ghost edge.
        ext_atype = np.array(ext_atype, copy=True)
        ext_atype[:, self.nloc :] = 1 - ext_atype[:, self.nloc :]
        # the corruption must actually be exercised: some ghost must be a neighbor
        ghost_in_nlist = np.any(nlist[nlist >= 0] >= self.nloc)
        assert ghost_in_nlist, "test is vacuous: no ghost appears in the nlist"
        # dense reference uses the corrupted atype_ext[neighbor] directly
        ref = self._dense_reference(dd, ext_coord, ext_atype, nlist)
        out = dd.call(ext_coord, ext_atype, nlist, mapping=mapping)
        np.testing.assert_allclose(out[0], ref[0], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(out[1], ref[1], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(out[4], ref[4], rtol=1e-12, atol=1e-12)
