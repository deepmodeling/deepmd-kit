# SPDX-License-Identifier: LGPL-3.0-or-later
"""Regression test for the ``_format_nlist`` pad-branch rcut filter.

The C++/LAMMPS neighbor list is built with ``rcut + skin`` and is passed into
``forward_lower`` WITHOUT any rcut filtering (see ``DeepPotPT.cc`` /
``copy_from_nlist``). Its width is the per-atom neighbor count, which can be
``<= nnei`` (``sum(sel)``) on sparse systems -- exactly the case in discussion
#5438 (width 39 < 100).

In that regime ``_format_nlist`` takes its pad branch (``n_nnei <= nnei`` and
``extra_nlist_sort`` is False). Previously that branch did NOT drop neighbors
beyond ``rcut``, so out-of-``rcut`` neighbors leaked into the descriptor, making
``forward_lower`` order-dependent (reversing the nlist changed the energy by
~1e-4 for se_r, ~4e-6 for se_a). The fix filters ``rr > rcut`` in the pad branch
too.

This test feeds an over-``rcut`` nlist (``rcut + 2``, pad branch) -- once as-is
and once reversed -- and asserts both match the canonical ``rcut``-bounded
evaluation (energy and force) to machine precision. Without the fix the
over-cut and reversed cases diverge from the canonical reference.
"""

import copy
import unittest

import torch

from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
)

from ...consistent.common import (
    parameterized,
)
from ...seed import (
    GLOBAL_SEED,
)
from .test_forward_lower import (
    reduce_tensor,
)
from .test_permutation import (
    model_se_e2_a,
)

dtype = torch.float64

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


@parameterized(
    (
        "se_a",
        "se_r",
    ),  # descriptor flavour (se_a damps over-rcut via direction; se_r does not)
    (False, True),  # reverse the over-cut nlist before forward_lower
)
class TestFormatNlistOvercut(unittest.TestCase):
    def setUp(self) -> None:
        flavour, self.reverse = self.param
        params = copy.deepcopy(model_se_e2_a if flavour == "se_a" else model_se_r)
        self.model = get_model(params).to(env.DEVICE)

    def _make_system(self):
        # sparse system: per-atom neighbor count stays below sum(sel), so the
        # over-cut nlist exercises the pad branch of _format_nlist.
        natoms = 6
        cell = 6.0 * torch.eye(3, dtype=dtype, device=env.DEVICE)
        generator = torch.Generator(device=env.DEVICE).manual_seed(GLOBAL_SEED)
        coord = 5.5 * torch.rand(
            [natoms, 3], dtype=dtype, device=env.DEVICE, generator=generator
        )
        atype = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64, device=env.DEVICE)
        return coord, atype, cell

    def _energy_force(self, coord, atype, cell, rcut_build, reverse):
        ec, ea, mp, nlist = extend_input_and_build_neighbor_list(
            coord.unsqueeze(0),
            atype.unsqueeze(0),
            rcut_build,
            sum(self.model.get_sel()),
            mixed_types=True,
            box=cell.unsqueeze(0),
        )
        if reverse:
            nlist = torch.flip(nlist, dims=[-1])
        out = self.model.forward_lower(ec, ea, nlist, mp, do_atomic_virial=False)
        force = reduce_tensor(out["extended_force"], mp, coord.shape[0])
        return out["energy"], force

    def test_overcut_matches_canonical(self) -> None:
        coord, atype, cell = self._make_system()
        rcut = self.model.get_rcut()

        # canonical: rcut-bounded nlist (no out-of-rcut neighbors), not reversed
        e_ref, f_ref = self._energy_force(coord, atype, cell, rcut, reverse=False)
        # over-cut nlist (rcut + 2 -> pad branch with out-of-rcut neighbors)
        e_out, f_out = self._energy_force(
            coord, atype, cell, rcut + 2.0, reverse=self.reverse
        )

        torch.testing.assert_close(e_out, e_ref, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(f_out, f_ref, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
