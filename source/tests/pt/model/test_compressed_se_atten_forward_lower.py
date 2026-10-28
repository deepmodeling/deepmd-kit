# SPDX-License-Identifier: LGPL-3.0-or-later
"""Regression coverage for compressed DPA1 on an unsorted lower nlist.

The geometric tabulation kernel folds trailing padding rows when its input is
sorted. LAMMPS supplies ``forward_lower`` with an rcut+skin neighbor list that
may put zero-switch, out-of-cutoff rows before real neighbors. A compressed
DPA1 model must therefore request the extra filtering/sorting pass before the
kernel applies that fold.
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
from .test_permutation import (
    model_dpa1,
)

dtype = torch.float64


@unittest.skipIf(not ENABLE_CUSTOMIZED_OP, "PyTorch customized OPs are not built")
class TestCompressedSeAttenForwardLower(unittest.TestCase):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_dpa1)
        # Geometric compression is available only for the strip representation
        # without attention layers.
        model_params["descriptor"]["tebd_input_mode"] = "strip"
        model_params["descriptor"]["attn_layer"] = 0
        self.model = get_model(model_params).to(env.DEVICE)

    def _make_system(self):
        """Create a sparse periodic system with neighbors in the skin region."""
        natoms = 6
        cell = 6.0 * torch.eye(3, dtype=dtype, device=env.DEVICE)
        generator = torch.Generator(device=env.DEVICE).manual_seed(GLOBAL_SEED)
        coord = 5.5 * torch.rand(
            [natoms, 3], dtype=dtype, device=env.DEVICE, generator=generator
        )
        atype = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64, device=env.DEVICE)
        return coord, atype, cell

    def _min_nbor_dist(self, coord, cell) -> float:
        """Return the periodic minimum pair distance for table construction."""
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

        # Use a clean cutoff-bounded list as the uncompressed reference.
        ec, ea, mapping, nlist = extend_input_and_build_neighbor_list(
            coord.unsqueeze(0),
            atype.unsqueeze(0),
            rcut,
            sel,
            mixed_types=self.model.mixed_types(),
            box=cell.unsqueeze(0),
        )
        ref = self.model.forward_lower(ec, ea, nlist, mapping, do_atomic_virial=False)

        self.model.min_nbor_dist = torch.tensor(
            0.9 * self._min_nbor_dist(coord, cell),
            dtype=env.GLOBAL_PT_FLOAT_PRECISION,
            device=env.DEVICE,
        )
        self.model.enable_compression()
        self.assertTrue(self.model.need_sorted_nlist_for_lower())

        # Mimic the unsorted rcut+skin list from LAMMPS and deliberately move
        # its out-of-cutoff/padding rows ahead of the in-cutoff neighbors.
        ec2, ea2, mapping2, nlist2 = extend_input_and_build_neighbor_list(
            coord.unsqueeze(0),
            atype.unsqueeze(0),
            rcut + 2.0,
            sum(sel),
            mixed_types=True,
            box=cell.unsqueeze(0),
        )
        safe_nlist = torch.clamp_min(nlist2, 0)
        gather_index = safe_nlist.reshape(1, -1, 1).expand(-1, -1, 3)
        neighbor_coord = torch.gather(ec2, 1, gather_index).view(
            1, coord.shape[0], -1, 3
        )
        center_coord = ec2[:, : coord.shape[0], :].unsqueeze(2)
        distance = torch.linalg.norm(neighbor_coord - center_coord, dim=-1)
        real_neighbor = nlist2 >= 0
        self.assertTrue(torch.any(real_neighbor & (distance <= rcut)).item())
        self.assertTrue(torch.any(real_neighbor & (distance > rcut)).item())

        nlist2 = torch.flip(nlist2, dims=[-1])
        out = self.model.forward_lower(
            ec2, ea2, nlist2, mapping2, do_atomic_virial=False
        )

        torch.testing.assert_close(out["energy"], ref["energy"], rtol=1e-10, atol=1e-10)
        natoms = coord.shape[0]
        ref_force = reduce_tensor(ref["extended_force"], mapping, natoms)
        out_force = reduce_tensor(out["extended_force"], mapping2, natoms)
        torch.testing.assert_close(out_force, ref_force, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
