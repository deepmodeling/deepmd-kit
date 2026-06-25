# SPDX-License-Identifier: LGPL-3.0-or-later
"""Bit-exact parity between the graph-native ``DescrptBlockSeAtten.call_graph``
(attn_layer=0) and the legacy dense ``DescrptBlockSeAtten.call`` on the SAME
neighbor list, for binding AND non-binding ``sel``.
"""

import unittest

import numpy as np

from deepmd.dpmodel.descriptor.dpa1 import (
    DescrptDPA1,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    from_dense_quartet,
)
from deepmd.dpmodel.utils.nlist import (
    extend_input_and_build_neighbor_list,
)


class TestDpa1BlockCallGraph(unittest.TestCase):
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

    def setUp(self) -> None:
        rng = np.random.default_rng(1)
        self.nloc = 4
        self.coord = rng.normal(size=(1, self.nloc, 3)) * 1.5
        self.atype = np.array([[0, 1, 0, 1]], dtype=np.int64)

    def test_block_graph_equals_dense_any_sel(self) -> None:
        for sel in ([20], [3]):  # non-binding AND binding
            with self.subTest(sel=sel):
                dd = self._make(sel)
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
                graph_g = blk.call_graph(
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

    def test_attn_layer_gt0_raises(self) -> None:
        dd = DescrptDPA1(rcut=4.0, rcut_smth=0.5, sel=[20], ntypes=2, attn_layer=2)
        with self.assertRaises(NotImplementedError):
            dd.se_atten.call_graph(None, np.array([0], dtype=np.int64))

    def test_exclude_types_raises(self) -> None:
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
        with self.assertRaises(NotImplementedError):
            dd.se_atten.call_graph(
                ng, self.atype.reshape(-1), type_embedding=dd.type_embedding.call()
            )


if __name__ == "__main__":
    unittest.main()
