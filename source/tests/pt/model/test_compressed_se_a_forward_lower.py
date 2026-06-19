# SPDX-License-Identifier: LGPL-3.0-or-later
"""Regression test for discussion #5438.

A compressed ``se_e2_a`` model produced wrong energy/forces when evaluated
through ``forward_lower`` with a neighbor list that is *not* pre-sorted and
contains out-of-``rcut`` (``sw == 0``) neighbors before the real ones -- exactly
what the C++/LAMMPS inference path provides (its neighbor list uses
``rcut + skin`` and is not distance-sorted).

The compressed ``tabulate_fusion_se_a`` op uses an ``is_sorted``
early-termination that stops accumulating at the first neighbor whose env-mat
direction is zero (padding, or an out-of-``rcut`` neighbor with ``sw == 0``),
assuming such neighbors are trailing. When they appear before real neighbors the
op silently drops the real neighbors, giving wrong descriptors.

The fix makes ``DescrptBlockSeA.need_sorted_nlist_for_lower()`` return
``self.compress`` so the model forces an ``extra_nlist_sort`` (which filters
out-of-``rcut`` neighbors and moves all padding last) before the op runs.

Without the fix, ``test_unsorted_overcut_nlist`` fails (energy/force mismatch);
with the fix the compressed result matches the uncompressed reference.
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


@parameterized((True, False))  # type_one_side
@unittest.skipIf(not ENABLE_CUSTOMIZED_OP, "PyTorch customized OPs are not built")
class TestCompressedSeAForwardLower(unittest.TestCase):
    def setUp(self) -> None:
        (self.type_one_side,) = self.param
        model_params = copy.deepcopy(model_se_e2_a)
        model_params["descriptor"]["type_one_side"] = self.type_one_side
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

        # reference: uncompressed forward_lower with a clean rcut-bounded nlist
        ec, ea, mp, nlist = extend_input_and_build_neighbor_list(
            coord.unsqueeze(0),
            atype.unsqueeze(0),
            rcut,
            sel,
            mixed_types=self.model.mixed_types(),
            box=cell.unsqueeze(0),
        )
        ref = self.model.forward_lower(ec, ea, nlist, mp, do_atomic_virial=False)

        # enable compression (lower bound below the true min distance -> no extrapolation)
        self.model.min_nbor_dist = torch.tensor(
            0.9 * self._min_nbor_dist(coord, cell),
            dtype=env.GLOBAL_PT_FLOAT_PRECISION,
            device=env.DEVICE,
        )
        self.model.enable_compression()

        # over-rcut FLAT neighbor list (mimics LAMMPS rcut+skin), reversed so the
        # out-of-rcut / padding neighbors precede the real ones.
        ec2, ea2, mp2, nlist2 = extend_input_and_build_neighbor_list(
            coord.unsqueeze(0),
            atype.unsqueeze(0),
            rcut + 2.0,
            sum(sel),
            mixed_types=True,
            box=cell.unsqueeze(0),
        )
        nlist2 = torch.flip(nlist2, dims=[-1])
        out = self.model.forward_lower(ec2, ea2, nlist2, mp2, do_atomic_virial=False)

        torch.testing.assert_close(out["energy"], ref["energy"], rtol=1e-10, atol=1e-10)
        natoms = coord.shape[0]
        f_ref = reduce_tensor(ref["extended_force"], mp, natoms)
        f_out = reduce_tensor(out["extended_force"], mp2, natoms)
        torch.testing.assert_close(f_out, f_ref, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
