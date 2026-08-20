# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import unittest
from unittest import (
    mock,
)

import torch

from deepmd.pt.utils import (
    env,
)
from deepmd.pt_expt.kernels.cutile import (
    CUTILE_AVAILABLE,
)
from deepmd.pt_expt.kernels.triton.sezm.force_assembly import (
    FORCE_ASSEMBLY_TRITON_AVAILABLE,
)
from deepmd.pt_expt.model.edge_transform_output import (
    edge_energy_deriv,
)


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

    def test_force_precision_downcasts_inference_only(self) -> None:
        """``force_precision`` assembles force / virial in that dtype for
        inference, matching the fp64 assembly in content (the gradient carries
        only the model precision); it is ignored when a graph is retained, so
        training / double-backward keeps the fp64 leaf dtype.
        """
        torch.manual_seed(2)
        N = 6
        n_node = torch.tensor([N], dtype=torch.int64, device=env.DEVICE)
        coord = torch.randn(N, 3, dtype=torch.float64, device=env.DEVICE)
        src = torch.tensor([0, 1, 2, 3, 4, 5], device=env.DEVICE)
        dst = torch.tensor([1, 0, 3, 2, 5, 4], device=env.DEVICE)
        edge_index = torch.stack([src, dst], 0)
        edge_mask = torch.ones(src.shape[0], dtype=torch.bool, device=env.DEVICE)

        def deriv(force_precision, create_graph):
            ev = (coord[src] - coord[dst]).detach().requires_grad_(True)
            energy = (torch.sin(ev).sum(-1) ** 2).sum()
            return edge_energy_deriv(
                energy,
                ev,
                edge_index,
                edge_mask,
                n_node,
                do_atomic_virial=True,
                create_graph=create_graph,
                force_precision=force_precision,
            )

        f64, av64, gv64 = deriv(None, False)
        f32, av32, gv32 = deriv(torch.float32, False)
        # inference: force / virial are emitted in the requested precision ...
        self.assertEqual(f32.dtype, torch.float32)
        self.assertEqual(gv32.dtype, torch.float32)
        self.assertEqual(av32.dtype, torch.float32)
        # ... and match the fp64 assembly to the fp32 floor.
        torch.testing.assert_close(f32.double(), f64, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(gv32.double(), gv64, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(av32.double(), av64, rtol=1e-5, atol=1e-5)
        # a retained graph keeps the fp64 leaf dtype (downcast suppressed).
        f_train, _, _ = deriv(torch.float32, True)
        self.assertEqual(f_train.dtype, torch.float64)

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

    @unittest.skipUnless(
        torch.cuda.is_available()
        and (FORCE_ASSEMBLY_TRITON_AVAILABLE or CUTILE_AVAILABLE),
        "an accelerated force-assembly backend and a CUDA device are required",
    )
    def test_accelerated_csr_paths_preserve_full_source_atom_virial(self) -> None:
        """The accelerated graph assembly matches the canonical CSR scatter."""
        device = torch.device("cuda")
        n_node = torch.tensor([3, 4], dtype=torch.int64, device=device)
        src = torch.tensor(
            [0, 1, 2, 1, 3, 4, 5, 6, 4, 3, 0],
            dtype=torch.int64,
            device=device,
        )
        dst = torch.tensor(
            [1, 2, 0, 0, 4, 5, 6, 3, 6, 5, 0],
            dtype=torch.int64,
            device=device,
        )
        edge_index = torch.stack([src, dst])
        edge_mask = torch.tensor([True] * 10 + [False], dtype=torch.bool, device=device)
        generator = torch.Generator(device=device).manual_seed(20260820)
        edge_value = torch.randn(
            src.shape[0], 3, dtype=torch.float32, device=device, generator=generator
        )
        edge_value[-1] = 100.0
        n_nodes = 9
        boundaries = torch.arange(n_nodes + 1, dtype=src.dtype, device=device)
        destination_order = torch.argsort(dst, stable=True)
        source_order = torch.argsort(src, stable=True)
        destination_row_ptr = torch.searchsorted(
            dst.index_select(0, destination_order), boundaries
        )
        source_row_ptr = torch.searchsorted(
            src.index_select(0, source_order), boundaries
        )

        def run(
            triton_level: str,
            cutile_level: str,
            *,
            do_atomic_virial: bool = True,
        ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
            edge_vec = edge_value.detach().clone().requires_grad_(True)
            energy = (edge_vec**3).sum()
            with mock.patch.dict(
                os.environ,
                {
                    "DP_TRITON_INFER": triton_level,
                    "DP_CUDA_INFER": "0",
                    "DP_CUTILE_INFER": cutile_level,
                },
            ):
                return edge_energy_deriv(
                    energy,
                    edge_vec,
                    edge_index,
                    edge_mask,
                    n_node,
                    destination_order,
                    destination_row_ptr,
                    source_order,
                    source_row_ptr,
                    node_capacity=n_nodes,
                    do_atomic_virial=do_atomic_virial,
                )

        force_ref, atom_virial_ref, virial_ref = run("0", "0")
        edge_grad = torch.where(
            edge_mask[:, None], 3.0 * edge_value**2, torch.zeros_like(edge_value)
        )
        edge_virial = -torch.einsum("ek,ej->ekj", edge_grad, edge_value)
        expected_atom_virial = torch.zeros(
            n_nodes, 3, 3, dtype=edge_value.dtype, device=device
        )
        expected_atom_virial.index_add_(0, src, edge_virial)
        backends = []
        if FORCE_ASSEMBLY_TRITON_AVAILABLE:
            backends.append(("triton", "1", "0"))
        if CUTILE_AVAILABLE:
            backends.append(("cutile", "0", "1"))
        for backend, triton_level, cutile_level in backends:
            with self.subTest(backend=backend):
                force, atom_virial, virial = run(triton_level, cutile_level)

                torch.testing.assert_close(force, force_ref, rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(
                    atom_virial, atom_virial_ref, rtol=1e-5, atol=1e-5
                )
                torch.testing.assert_close(virial, virial_ref, rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(
                    atom_virial, expected_atom_virial, rtol=1e-5, atol=1e-5
                )
                torch.testing.assert_close(force[7:], torch.zeros_like(force[7:]))
                torch.testing.assert_close(
                    atom_virial[7:], torch.zeros_like(atom_virial[7:])
                )

                force_without_atomic, atom_virial_none, virial_without_atomic = run(
                    triton_level,
                    cutile_level,
                    do_atomic_virial=False,
                )
                self.assertIsNone(atom_virial_none)
                torch.testing.assert_close(
                    force_without_atomic, force, rtol=1e-5, atol=1e-5
                )
                torch.testing.assert_close(
                    virial_without_atomic, virial, rtol=1e-5, atol=1e-5
                )


if __name__ == "__main__":
    unittest.main()
