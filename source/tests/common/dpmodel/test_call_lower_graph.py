# SPDX-License-Identifier: LGPL-3.0-or-later
"""Energy-level parity between the graph-native model lower
(``CM.call_lower_graph``) and the dense ``EnergyModel.call_lower`` on the SAME
neighbor list (regime-1: ``from_dense_quartet`` reproduces the nlist neighbors).

This suite checks ``energy`` (reduced per-frame) and ``atom_energy`` (per-atom).
"""

import unittest

import numpy as np

from deepmd.dpmodel.descriptor.dpa1 import (
    DescrptDPA1,
)
from deepmd.dpmodel.fitting import (
    InvarFitting,
)
from deepmd.dpmodel.model.ener_model import (
    EnergyModel,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    from_dense_quartet,
)
from deepmd.dpmodel.utils.nlist import (
    extend_input_and_build_neighbor_list,
)


class TestCallLowerGraph(unittest.TestCase):
    def _make_model(self):
        ds = DescrptDPA1(
            rcut=4.0,
            rcut_smth=0.5,
            sel=[30],
            ntypes=2,
            attn_layer=0,
            axis_neuron=2,
            neuron=[6, 12],
        )
        ft = InvarFitting(
            "energy",
            2,
            ds.get_dim_out(),
            1,
            mixed_types=ds.mixed_types(),
        )
        return EnergyModel(ds, ft, type_map=["foo", "bar"])

    def setUp(self) -> None:
        rng = np.random.default_rng(2)
        self.nloc = 4
        self.coord = rng.normal(size=(1, self.nloc, 3)) * 1.5
        self.atype = np.array([[0, 1, 0, 1]], dtype=np.int64)

    def test_graph_lower_matches_dense_lower(self) -> None:
        """Graph model lower energy/atom_energy match the dense lower on the same nlist."""
        model = self._make_model()
        (
            ext_coord,
            ext_atype,
            mapping,
            nlist,
        ) = extend_input_and_build_neighbor_list(
            self.coord,
            self.atype,
            model.get_rcut(),
            model.get_sel(),
            mixed_types=model.mixed_types(),
            box=None,
        )

        # dense ``call_common_lower`` returns the INTERNAL model_output_def keys
        # (``energy`` per-atom, ``energy_redu`` reduced), matching the
        # OUTPUT-AGNOSTIC graph lower.
        dense = model.call_common_lower(ext_coord, ext_atype, nlist, mapping)

        ng = from_dense_quartet(ext_coord, nlist, mapping)
        nloc = nlist.shape[1]
        out = model.call_lower_graph(
            atype=ext_atype.reshape(-1)[:nloc],
            n_node=ng.n_node,
            edge_index=ng.edge_index,
            edge_vec=ng.edge_vec,
            edge_mask=ng.edge_mask,
        )

        # reduced per-frame energy
        np.testing.assert_allclose(
            out["energy_redu"], dense["energy_redu"], rtol=1e-12, atol=1e-12
        )
        # per-atom energy
        np.testing.assert_allclose(
            out["energy"].reshape(dense["energy"].shape),
            dense["energy"],
            rtol=1e-12,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
