# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import torch

from deepmd.pt.utils import env
from deepmd.pt_expt.model.edge_transform_output import edge_energy_deriv


class TestEdgeEnergyDeriv(unittest.TestCase):
    def test_force_matches_autograd_wrt_node_coords(self) -> None:
        """The graph force equals -dE/d(node coord): build edge_vec from node
        coords, so force from edge_energy_deriv == -autograd.grad(E, coords).
        """
        torch.manual_seed(0)
        N, nf = 5, 1
        n_node = torch.tensor([N], dtype=torch.int64, device=env.DEVICE)
        coord = torch.randn(
            N, 3, dtype=torch.float64, device=env.DEVICE, requires_grad=True
        )
        # a connected edge set (both directions), all real
        src = torch.tensor([0, 1, 1, 2, 3, 4], device=env.DEVICE)
        dst = torch.tensor([1, 0, 2, 1, 4, 3], device=env.DEVICE)
        edge_index = torch.stack([src, dst], 0)
        edge_mask = torch.ones(src.shape[0], dtype=torch.bool, device=env.DEVICE)
        edge_vec = coord[src] - coord[dst]  # differentiable wrt coord
        energy = (torch.sin(edge_vec).sum(-1) ** 2).sum()  # toy scalar energy
        force, av, gv = edge_energy_deriv(
            energy, edge_vec, edge_index, edge_mask, n_node, do_atomic_virial=True
        )
        # reference physical force = -dE/d(coord)
        f_ref = -torch.autograd.grad(energy, coord, retain_graph=True)[0]
        torch.testing.assert_close(force, f_ref, rtol=1e-10, atol=1e-10)
        # atom-virial sums (per frame) to the global virial
        torch.testing.assert_close(av.sum(0), gv[0], rtol=1e-10, atol=1e-10)
        self.assertEqual(gv.shape, (nf, 3, 3))

    def test_padding_edges_contribute_nothing(self) -> None:
        """A masked guard edge with a huge edge_vec must not change force/virial."""
        torch.manual_seed(1)
        N = 4
        n_node = torch.tensor([N], dtype=torch.int64, device=env.DEVICE)
        coord = torch.randn(
            N, 3, dtype=torch.float64, device=env.DEVICE, requires_grad=True
        )
        src = torch.tensor([0, 1, 2], device=env.DEVICE)
        dst = torch.tensor([1, 2, 3], device=env.DEVICE)
        ev = coord[src] - coord[dst]
        # append a masked guard edge with a huge vec
        guard = torch.tensor(
            [[99.0, 99.0, 99.0]], dtype=torch.float64, device=env.DEVICE
        )
        edge_vec = torch.cat([ev, guard], 0).detach().requires_grad_(True)
        edge_index = torch.tensor([[0, 1, 2, 0], [1, 2, 3, 0]], device=env.DEVICE)
        edge_mask = torch.tensor([True, True, True, False], device=env.DEVICE)
        energy = (edge_vec**2).sum()
        force, av, gv = edge_energy_deriv(
            energy, edge_vec, edge_index, edge_mask, n_node, do_atomic_virial=True
        )
        # run again with ONLY the real edges; results must match
        ev2 = edge_vec[:3].detach().requires_grad_(True)
        e2 = (ev2**2).sum()
        f2, av2, gv2 = edge_energy_deriv(
            e2, ev2, edge_index[:, :3], edge_mask[:3], n_node, do_atomic_virial=True
        )
        torch.testing.assert_close(force, f2, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(gv, gv2, rtol=1e-12, atol=1e-12)

    def test_atom_virial_optional(self) -> None:
        """do_atomic_virial=False returns None for atom_virial; force+virial still computed."""
        N = 3
        n_node = torch.tensor([N], dtype=torch.int64, device=env.DEVICE)
        coord = torch.randn(
            N, 3, dtype=torch.float64, device=env.DEVICE, requires_grad=True
        )
        edge_index = torch.tensor([[0, 1], [1, 0]], device=env.DEVICE)
        edge_mask = torch.ones(2, dtype=torch.bool, device=env.DEVICE)
        edge_vec = coord[edge_index[0]] - coord[edge_index[1]]
        energy = (edge_vec**2).sum()
        force, av, gv = edge_energy_deriv(
            energy, edge_vec, edge_index, edge_mask, n_node, do_atomic_virial=False
        )
        self.assertIsNone(av)
        self.assertEqual(force.shape, (N, 3))
        self.assertEqual(gv.shape, (1, 3, 3))


if __name__ == "__main__":
    unittest.main()
