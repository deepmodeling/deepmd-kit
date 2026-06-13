# SPDX-License-Identifier: LGPL-3.0-or-later
"""Companion to ``test_compressed_se_a_forward_lower.py`` for discussion #5438.

The compressed ``se_e2_a`` descriptor produced wrong energy/forces when its
``forward_lower`` neighbor list was *not* pre-sorted and contained
out-of-``rcut`` (``sw == 0``) neighbors before the real ones -- the root cause
being the ``is_sorted`` early-termination in the ``tabulate_fusion_se_a`` op
(``source/lib/src/tabulate.cc``), which stops accumulating at the first
zero-direction neighbor.

``se_e2_r`` is expected to be *immune*: its ``tabulate_fusion_se_r`` kernel has
no such early-termination (it takes no ``is_sorted`` argument and always
iterates every neighbor), and the descriptor reduces over neighbors with an
order-independent ``mean``. Consequently ``need_sorted_nlist_for_lower()``
correctly stays ``False`` for se_r and needs no ``self.compress`` override.

This test locks in that immunity. It runs ``forward_lower`` on an over-``rcut``
FLAT nlist reversed so padding / out-of-``rcut`` neighbors precede the real ones
-- exactly the input that broke se_a -- and asserts that the *compressed* result
matches the *uncompressed* result on that **identical** nlist (energy and force
to machine precision). On that same input the buggy se_a op diverged grossly,
so this guard would catch an analogous se_r regression.

Why compare on the *same* nlist (not against a clean rcut-bounded one): this
over-cut nlist has width ``== nnei`` and contains out-of-``rcut`` neighbors, so
``format_nlist`` takes its *pad* branch (``n_nnei > nnei`` is false), which does
NOT re-sort or rcut-filter. The pad branch is therefore mildly order-dependent
(``nlist_distinguish_types`` truncates per-type sections in raw nlist order and
lets over-``rcut`` neighbors leak in), so reversing the nlist shifts the
*uncompressed* energy by ~1e-4. This is a property of ``format_nlist``, NOT of
the reduction, and it affects se_a and se_r identically (uncompressed se_a shifts
~4e-6 on the same input). Comparing compressed vs uncompressed on the *identical*
nlist cancels that shared pad-branch effect and isolates the compression op:
verified ``rel == 0`` (energy + force) -- whereas the buggy se_a op diverged
grossly on this same input.
"""

import copy
import unittest

import torch

from deepmd.pt.cxx_op import (
    ENABLE_CUSTOMIZED_OP,
)
from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
)

from ...seed import (
    GLOBAL_SEED,
)
from .test_forward_lower import (
    reduce_tensor,
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
    "fitting_net": {
        "neuron": [24, 24, 24],
        "resnet_dt": True,
        "seed": 1,
    },
    "data_stat_nbatch": 20,
}


@unittest.skipIf(not ENABLE_CUSTOMIZED_OP, "PyTorch customized OPs are not built")
class TestCompressedSeRForwardLower(unittest.TestCase):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_se_r)
        self.model = get_model(model_params).to(env.DEVICE)

    def _make_system(self):
        # a sparse system: atoms have fewer neighbors than ``sel`` within
        # ``rcut``, and some neighbors fall in ``(rcut, rcut + buffer]`` so the
        # over-cut neighbor list contains zero-``sw`` real neighbors.
        natoms = 6
        cell = 6.0 * torch.eye(3, dtype=dtype, device=env.DEVICE)
        generator = torch.Generator(device=env.DEVICE).manual_seed(GLOBAL_SEED)
        coord = 5.5 * torch.rand(
            [natoms, 3], dtype=dtype, device=env.DEVICE, generator=generator
        )
        atype = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64, device=env.DEVICE)
        return coord, atype, cell

    def _min_nbor_dist(self, coord, cell):
        # minimum image minimum pair distance, used as the compression lower bound
        box = torch.diagonal(cell)
        diff = coord[:, None, :] - coord[None, :, :]
        diff = diff - torch.round(diff / box) * box
        dist = torch.linalg.norm(diff, dim=-1)
        dist = dist + torch.eye(coord.shape[0], device=coord.device) * 1e10
        return float(dist.min())

    def test_unsorted_overcut_nlist(self) -> None:
        coord, atype, cell = self._make_system()
        rcut = self.model.get_rcut()
        sel = self.model.get_sel()

        # over-rcut FLAT neighbor list (mimics LAMMPS rcut+skin), reversed so the
        # out-of-rcut / padding neighbors precede the real ones -- the exact input
        # that broke compressed se_a.
        ec, ea, mp, nlist = extend_input_and_build_neighbor_list(
            coord.unsqueeze(0),
            atype.unsqueeze(0),
            rcut + 2.0,
            sum(sel),
            mixed_types=True,
            box=cell.unsqueeze(0),
        )
        nlist = torch.flip(nlist, dims=[-1])

        # reference: uncompressed forward_lower on this exact nlist
        ref = self.model.forward_lower(ec, ea, nlist, mp, do_atomic_virial=False)

        # enable compression (lower bound below the true min distance -> no
        # extrapolation) and rerun forward_lower on the IDENTICAL nlist
        self.model.min_nbor_dist = torch.tensor(
            0.9 * self._min_nbor_dist(coord, cell),
            dtype=env.GLOBAL_PT_FLOAT_PRECISION,
            device=env.DEVICE,
        )
        self.model.enable_compression()
        out = self.model.forward_lower(ec, ea, nlist, mp, do_atomic_virial=False)

        torch.testing.assert_close(out["energy"], ref["energy"], rtol=1e-10, atol=1e-10)
        natoms = coord.shape[0]
        f_ref = reduce_tensor(ref["extended_force"], mp, natoms)
        f_out = reduce_tensor(out["extended_force"], mp, natoms)
        torch.testing.assert_close(f_out, f_ref, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
