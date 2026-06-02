# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the device-resident vesin.torch neighbor-list builder."""

import unittest

import numpy as np

from deepmd.dpmodel.utils.nlist import (
    extend_input_and_build_neighbor_list,
)
from deepmd.pt_expt.utils.nlist import (
    build_neighbor_list_vesin_torch,
    is_vesin_torch_available,
)


def _per_atom_neighbor_dists(ext_coord, nlist, coord):
    """Sorted, rounded valid-neighbor distances for each local atom."""
    ext_coord = np.asarray(ext_coord).reshape(-1, 3)
    coord = np.asarray(coord).reshape(-1, 3)
    out = []
    for i in range(coord.shape[0]):
        ds = [
            round(float(np.linalg.norm(ext_coord[j] - coord[i])), 6)
            for j in nlist[i]
            if j >= 0
        ]
        out.append(sorted(ds))
    return out


@unittest.skipIf(not is_vesin_torch_available(), "vesin.torch is not installed")
class TestNeighListVesinTorch(unittest.TestCase):
    """The vesin.torch builder must produce the same neighbor relationships as
    the native all-pairs builder, on whatever device the input lives.
    """

    def setUp(self) -> None:
        import torch

        rng = np.random.default_rng(20240602)
        self.nloc = 40
        self.rcut = 2.5
        self.sel = [30, 30]
        box_len = 6.0
        self.box_np = (np.eye(3) * box_len).reshape(1, 9)
        coord = (rng.random((self.nloc, 3)) * box_len).reshape(1, self.nloc, 3)
        atype = rng.integers(0, 2, self.nloc).reshape(1, self.nloc)
        self.coord_np = coord
        self.atype_np = atype
        self.coord_t = torch.tensor(coord, dtype=torch.float64)
        self.box_t = torch.tensor(
            (np.eye(3) * box_len).reshape(1, 3, 3), dtype=torch.float64
        )
        self.atype_t = torch.tensor(atype, dtype=torch.int64)

    def _native_dists(self, box_np, distinguish):
        ec, _, _, nl = extend_input_and_build_neighbor_list(
            self.coord_np,
            self.atype_np,
            self.rcut,
            self.sel,
            mixed_types=not distinguish,
            box=box_np,
        )
        return _per_atom_neighbor_dists(
            np.asarray(ec).reshape(-1, 3), np.asarray(nl)[0], self.coord_np[0]
        )

    def _vesin_dists(self, ect, nlt):
        return _per_atom_neighbor_dists(
            ect[0].cpu().numpy(), nlt[0].cpu().numpy(), self.coord_np[0]
        )

    def test_pbc_matches_native(self) -> None:
        ect, _, nlt, _ = build_neighbor_list_vesin_torch(
            self.coord_t, self.box_t, self.atype_t, self.rcut, self.sel, False
        )
        self.assertEqual(
            self._native_dists(self.box_np, False), self._vesin_dists(ect, nlt)
        )

    def test_nopbc_matches_native(self) -> None:
        ect, _, nlt, _ = build_neighbor_list_vesin_torch(
            self.coord_t, None, self.atype_t, self.rcut, self.sel, False
        )
        self.assertEqual(self._native_dists(None, False), self._vesin_dists(ect, nlt))

    def test_distinguish_types_matches_native(self) -> None:
        ect, _, nlt, _ = build_neighbor_list_vesin_torch(
            self.coord_t, self.box_t, self.atype_t, self.rcut, self.sel, True
        )
        self.assertEqual(
            self._native_dists(self.box_np, True), self._vesin_dists(ect, nlt)
        )

    def test_outputs_on_input_device(self) -> None:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ect, eat, nlt, mpt = build_neighbor_list_vesin_torch(
            self.coord_t.to(device),
            self.box_t.to(device),
            self.atype_t.to(device),
            self.rcut,
            self.sel,
            False,
        )
        for t in (ect, eat, nlt, mpt):
            self.assertEqual(t.device.type, device.type)


if __name__ == "__main__":
    unittest.main()
