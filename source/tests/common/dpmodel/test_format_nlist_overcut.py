# SPDX-License-Identifier: LGPL-3.0-or-later
"""dpmodel counterpart of the pt ``test_format_nlist_overcut`` regression.

``_format_nlist`` exists in both backends (``deepmd/pt`` and ``deepmd/dpmodel``;
the latter is shared by ``pt_expt``). The pad branch previously did not drop
out-of-``rcut`` neighbors, so an over-``rcut`` neighbor list (what the C++/LAMMPS
path passes, unfiltered) leaked into the descriptor and made ``call_lower``
order-dependent. This test exercises the **dpmodel** (numpy) path directly --
the pt test only covers ``deepmd/pt``.

An over-``rcut`` nlist (``rcut + 2``, pad branch) is fed to ``call_lower`` both
as-is and reversed; both must match the canonical ``rcut``-bounded evaluation
(reduced + per-atom energy). Without the fix the reversed case diverges.
"""

import unittest

import numpy as np

from deepmd.dpmodel.model.model import (
    get_model,
)
from deepmd.dpmodel.utils.nlist import (
    extend_input_and_build_neighbor_list,
)

model_se_r = {
    "type_map": ["O", "H", "B"],
    "descriptor": {
        "type": "se_e2_r",
        "sel": [46, 92, 4],
        "rcut_smth": 0.50,
        "rcut": 4.00,
        "neuron": [25, 50, 100],
        "resnet_dt": False,
        "seed": 1,
    },
    "fitting_net": {"neuron": [24, 24, 24], "resnet_dt": True, "seed": 1},
    "data_stat_nbatch": 20,
}


class TestFormatNlistOvercutDP(unittest.TestCase):
    def setUp(self) -> None:
        self.model = get_model(model_se_r)
        rng = np.random.default_rng(20240131)
        self.natoms = 6
        self.cell = 6.0 * np.eye(3)
        self.coord = 5.5 * rng.random([self.natoms, 3])
        self.atype = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)

    def _lower(self, rcut_build, reverse):
        ec, ea, mp, nlist = extend_input_and_build_neighbor_list(
            self.coord[None],
            self.atype[None],
            rcut_build,
            sum(self.model.get_sel()),
            mixed_types=True,
            box=self.cell[None],
        )
        if reverse:
            nlist = nlist[..., ::-1]
        out = self.model.call_lower(ec, ea, nlist, mp, do_atomic_virial=False)
        return out["energy"], out["atom_energy"]

    def test_overcut_matches_canonical(self) -> None:
        rcut = self.model.get_rcut()
        er_ref, ea_ref = self._lower(rcut, reverse=False)
        for reverse in (False, True):  # over-cut nlist, kept order vs reversed
            with self.subTest(reverse=reverse):
                er, ea = self._lower(rcut + 2.0, reverse=reverse)
                np.testing.assert_allclose(er, er_ref, rtol=1e-10, atol=1e-10)
                np.testing.assert_allclose(ea, ea_ref, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
