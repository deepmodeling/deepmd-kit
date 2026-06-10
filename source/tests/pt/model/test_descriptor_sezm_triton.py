# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import torch

from deepmd.pt.model.descriptor.sezm_nn import (
    C3CutoffEnvelope,
    InnerClamp,
    RadialBasis,
    build_m_major_index,
    project_D_to_m,
    project_Dt_from_m,
)
from deepmd.pt.model.descriptor.sezm_nn.triton import (
    SEZM_TRITON_AVAILABLE,
    TritonRotationMode,
    edge_geometry_rbf_triton,
    resolve_triton_rotation_mode,
    rotate_back_triton,
    rotate_to_local_triton,
)

TRITON_CUDA_AVAILABLE = SEZM_TRITON_AVAILABLE and torch.cuda.is_available()


class TestSeZMTritonDispatch(unittest.TestCase):
    """Validate the SeZM Triton dispatch policy."""

    def test_resolve_rotation_mode_covers_small_generic_and_fallback(self) -> None:
        """Dispatch policy should cover small kernels, generic kernels, and fallback."""
        self.assertEqual(
            resolve_triton_rotation_mode(dim_full=1, reduced_dim=1),
            TritonRotationMode.SMALL_LE1,
        )
        self.assertEqual(
            resolve_triton_rotation_mode(dim_full=4, reduced_dim=4),
            TritonRotationMode.SMALL_LE1,
        )
        self.assertEqual(
            resolve_triton_rotation_mode(dim_full=9, reduced_dim=7),
            TritonRotationMode.SMALL_L2,
        )
        self.assertEqual(
            resolve_triton_rotation_mode(dim_full=16, reduced_dim=10),
            TritonRotationMode.SMALL_L3,
        )
        self.assertEqual(
            resolve_triton_rotation_mode(dim_full=25, reduced_dim=15),
            TritonRotationMode.EAGER_REFERENCE,
        )
        self.assertEqual(
            resolve_triton_rotation_mode(dim_full=25, reduced_dim=16),
            TritonRotationMode.GENERIC_TILED,
        )


@unittest.skipUnless(
    TRITON_CUDA_AVAILABLE,
    "SeZM Triton rotation tests require CUDA and Triton.",
)
class TestSeZMTritonEdgeGeometryRBF(unittest.TestCase):
    """Validate the Triton edge geometry/RBF chain against eager reference."""

    def _eager_reference(
        self,
        *,
        coord_flat: torch.Tensor,
        center_idx: torch.Tensor,
        neighbor_idx: torch.Tensor,
        edge_envelope: C3CutoffEnvelope,
        radial_basis: RadialBasis,
        inner_clamp: InnerClamp | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the eager reference geometry/RBF chain."""
        center_pos = coord_flat.index_select(0, center_idx)
        neighbor_pos = coord_flat.index_select(0, neighbor_idx)
        edge_vec = neighbor_pos - center_pos
        edge_len = torch.sqrt(
            torch.sum(edge_vec * edge_vec, dim=-1, keepdim=True) + 1.0e-14
        )
        if inner_clamp is not None:
            clamped = inner_clamp(edge_len)
            edge_vec = edge_vec * (clamped / edge_len)
            edge_len = clamped
        edge_env = edge_envelope(edge_len)
        edge_rbf = radial_basis(edge_len)
        return edge_vec, edge_len, edge_env, edge_rbf

    def test_edge_geometry_rbf_matches_reference_forward_backward(self) -> None:
        """Compare fused geometry/RBF chain with eager gather/clamp/envelope/rbf."""
        device = torch.device("cuda")
        dtype = torch.float32
        coord_ref = torch.randn(
            12,
            3,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        coord_triton = coord_ref.detach().clone().requires_grad_(True)
        center_idx = torch.randint(0, 12, (9,), device=device, dtype=torch.long)
        neighbor_idx = torch.randint(0, 12, (9,), device=device, dtype=torch.long)
        edge_envelope = C3CutoffEnvelope(rcut=6.0, exponent=5).to(device)
        radial_ref = RadialBasis(rcut=6.0, n_radial=6, dtype=dtype, exponent=7).to(
            device
        )
        radial_triton = RadialBasis(rcut=6.0, n_radial=6, dtype=dtype, exponent=7).to(
            device
        )
        radial_triton.load_state_dict(radial_ref.state_dict())

        out_ref = self._eager_reference(
            coord_flat=coord_ref,
            center_idx=center_idx,
            neighbor_idx=neighbor_idx,
            edge_envelope=edge_envelope,
            radial_basis=radial_ref,
            inner_clamp=None,
        )
        out_triton = edge_geometry_rbf_triton(
            coord_flat=coord_triton,
            center_coord_index=center_idx,
            neighbor_coord_index=neighbor_idx,
            edge_envelope=edge_envelope,
            radial_basis=radial_triton,
            eps=1.0e-7,
            inner_clamp=None,
        )
        for ref, tri in zip(out_ref, out_triton, strict=True):
            torch.testing.assert_close(tri, ref, atol=1.0e-5, rtol=1.0e-5)

        grad_out = tuple(torch.randn_like(ref) for ref in out_ref)
        grad_coord_ref, grad_freq_ref = torch.autograd.grad(
            out_ref,
            (coord_ref, radial_ref.adam_freqs),
            grad_outputs=grad_out,
        )
        grad_coord_triton, grad_freq_triton = torch.autograd.grad(
            out_triton,
            (coord_triton, radial_triton.adam_freqs),
            grad_outputs=grad_out,
        )
        torch.testing.assert_close(
            grad_coord_triton,
            grad_coord_ref,
            atol=2.0e-5,
            rtol=2.0e-5,
        )
        torch.testing.assert_close(
            grad_freq_triton,
            grad_freq_ref,
            atol=2.0e-5,
            rtol=2.0e-5,
        )

    def test_edge_geometry_rbf_matches_reference_with_inner_clamp(self) -> None:
        """Compare the clamped Triton path with eager reference."""
        device = torch.device("cuda")
        dtype = torch.float32
        coord_ref = torch.randn(
            10,
            3,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        coord_triton = coord_ref.detach().clone().requires_grad_(True)
        center_idx = torch.randint(0, 10, (7,), device=device, dtype=torch.long)
        neighbor_idx = torch.randint(0, 10, (7,), device=device, dtype=torch.long)
        edge_envelope = C3CutoffEnvelope(rcut=6.0, exponent=5).to(device)
        radial_ref = RadialBasis(rcut=6.0, n_radial=4, dtype=dtype, exponent=7).to(
            device
        )
        radial_triton = RadialBasis(rcut=6.0, n_radial=4, dtype=dtype, exponent=7).to(
            device
        )
        radial_triton.load_state_dict(radial_ref.state_dict())
        inner_clamp = InnerClamp(0.9, 1.3).to(device)

        out_ref = self._eager_reference(
            coord_flat=coord_ref,
            center_idx=center_idx,
            neighbor_idx=neighbor_idx,
            edge_envelope=edge_envelope,
            radial_basis=radial_ref,
            inner_clamp=inner_clamp,
        )
        out_triton = edge_geometry_rbf_triton(
            coord_flat=coord_triton,
            center_coord_index=center_idx,
            neighbor_coord_index=neighbor_idx,
            edge_envelope=edge_envelope,
            radial_basis=radial_triton,
            eps=1.0e-7,
            inner_clamp=inner_clamp,
        )
        for ref, tri in zip(out_ref, out_triton, strict=True):
            torch.testing.assert_close(tri, ref, atol=2.0e-5, rtol=2.0e-5)

        loss_ref = sum(x.square().sum() for x in out_ref)
        loss_triton = sum(x.square().sum() for x in out_triton)
        grad_coord_ref, grad_freq_ref = torch.autograd.grad(
            loss_ref,
            (coord_ref, radial_ref.adam_freqs),
        )
        grad_coord_triton, grad_freq_triton = torch.autograd.grad(
            loss_triton,
            (coord_triton, radial_triton.adam_freqs),
        )
        torch.testing.assert_close(
            grad_coord_triton,
            grad_coord_ref,
            atol=3.0e-5,
            rtol=3.0e-5,
        )
        torch.testing.assert_close(
            grad_freq_triton,
            grad_freq_ref,
            atol=3.0e-5,
            rtol=3.0e-5,
        )


@unittest.skipUnless(
    TRITON_CUDA_AVAILABLE,
    "SeZM Triton rotation tests require CUDA and Triton.",
)
class TestSeZMTritonSO2(unittest.TestCase):
    """Validate Triton SO(2) rotation kernels against the eager reference path."""

    def _require_cuda_bfloat16(self) -> None:
        """Skip the mixed-precision Triton tests when CUDA bf16 is unavailable."""
        if not torch.cuda.is_bf16_supported():
            self.skipTest("CUDA bfloat16 is required for mixed-precision Triton tests.")

    def test_rotate_to_local_matches_reference_forward_backward(self) -> None:
        """Compare fused Triton rotate-to-local with projected eager matmul."""
        device = torch.device("cuda")
        dtype = torch.float32
        n_node = 7
        n_edge = 11
        channels = 8
        for lmax, mmax in ((2, 1), (3, 1)):
            dim_full = (lmax + 1) ** 2
            coeff_index = build_m_major_index(lmax, mmax, device=device)
            src = torch.randint(0, n_node, (n_edge,), device=device, dtype=torch.long)
            x_ref = torch.randn(
                n_node,
                dim_full,
                channels,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            wigner_ref = torch.randn(
                n_edge,
                dim_full,
                dim_full,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            x_triton = x_ref.detach().clone().requires_grad_(True)
            wigner_triton = wigner_ref.detach().clone().requires_grad_(True)

            out_ref = torch.bmm(
                project_D_to_m(
                    D_full=wigner_ref,
                    coeff_index_m=coeff_index,
                    ebed_dim_full=dim_full,
                    cache=None,
                    key_lmax=lmax,
                    key_mmax=mmax,
                ),
                x_ref.index_select(0, src),
            )
            out_triton = rotate_to_local_triton(
                x=x_triton,
                src=src,
                wigner=wigner_triton,
                coeff_index=coeff_index,
                dim_full=dim_full,
            )
            torch.testing.assert_close(out_triton, out_ref, atol=1.0e-5, rtol=1.0e-5)

            grad_out = torch.randn_like(out_ref)
            grad_x_ref, grad_wigner_ref = torch.autograd.grad(
                out_ref,
                (x_ref, wigner_ref),
                grad_outputs=grad_out,
            )
            grad_x_triton, grad_wigner_triton = torch.autograd.grad(
                out_triton,
                (x_triton, wigner_triton),
                grad_outputs=grad_out,
            )
            torch.testing.assert_close(
                grad_x_triton,
                grad_x_ref,
                atol=1.0e-5,
                rtol=1.0e-5,
            )
            torch.testing.assert_close(
                grad_wigner_triton,
                grad_wigner_ref,
                atol=1.0e-5,
                rtol=1.0e-5,
            )

    def test_rotate_back_matches_reference_forward_backward(self) -> None:
        """Compare fused Triton rotate-back with projected eager matmul."""
        device = torch.device("cuda")
        dtype = torch.float32
        n_edge = 11
        channels = 8
        for lmax, mmax in ((2, 1), (3, 1)):
            dim_full = (lmax + 1) ** 2
            coeff_index = build_m_major_index(lmax, mmax, device=device)
            reduced_dim = int(coeff_index.numel())
            x_local_ref = torch.randn(
                n_edge,
                reduced_dim,
                channels,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            wigner_ref = torch.randn(
                n_edge,
                dim_full,
                dim_full,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            x_local_triton = x_local_ref.detach().clone().requires_grad_(True)
            wigner_triton = wigner_ref.detach().clone().requires_grad_(True)

            out_ref = torch.bmm(
                project_Dt_from_m(
                    Dt_full=wigner_ref,
                    coeff_index_m=coeff_index,
                    ebed_dim_full=dim_full,
                    cache=None,
                    key_lmax=lmax,
                    key_mmax=mmax,
                ),
                x_local_ref,
            )
            out_triton = rotate_back_triton(
                x_local=x_local_triton,
                wigner=wigner_triton,
                coeff_index=coeff_index,
                dim_full=dim_full,
            )
            torch.testing.assert_close(out_triton, out_ref, atol=1.0e-5, rtol=1.0e-5)

            grad_out = torch.randn_like(out_ref)
            grad_x_ref, grad_wigner_ref = torch.autograd.grad(
                out_ref,
                (x_local_ref, wigner_ref),
                grad_outputs=grad_out,
            )
            grad_x_triton, grad_wigner_triton = torch.autograd.grad(
                out_triton,
                (x_local_triton, wigner_triton),
                grad_outputs=grad_out,
            )
            torch.testing.assert_close(
                grad_x_triton,
                grad_x_ref,
                atol=1.0e-5,
                rtol=1.0e-5,
            )
            torch.testing.assert_close(
                grad_wigner_triton,
                grad_wigner_ref,
                atol=1.0e-5,
                rtol=1.0e-5,
            )

    def test_rotate_to_local_matches_mixed_precision_reference(self) -> None:
        """Compare Triton rotate-to-local with bf16 activations and fp32 Wigner."""
        self._require_cuda_bfloat16()
        device = torch.device("cuda")
        x_dtype = torch.bfloat16
        wigner_dtype = torch.float32
        n_node = 7
        n_edge = 11
        channels = 8
        for lmax, mmax in ((2, 1), (3, 1)):
            dim_full = (lmax + 1) ** 2
            coeff_index = build_m_major_index(lmax, mmax, device=device)
            src = torch.randint(0, n_node, (n_edge,), device=device, dtype=torch.long)
            x_ref = torch.randn(
                n_node,
                dim_full,
                channels,
                device=device,
                dtype=x_dtype,
                requires_grad=True,
            )
            wigner_ref = torch.randn(
                n_edge,
                dim_full,
                dim_full,
                device=device,
                dtype=wigner_dtype,
                requires_grad=True,
            )
            x_triton = x_ref.detach().clone().requires_grad_(True)
            wigner_triton = wigner_ref.detach().clone().requires_grad_(True)

            out_ref = torch.bmm(
                project_D_to_m(
                    D_full=wigner_ref,
                    coeff_index_m=coeff_index,
                    ebed_dim_full=dim_full,
                    cache=None,
                    key_lmax=lmax,
                    key_mmax=mmax,
                ).to(dtype=x_dtype),
                x_ref.index_select(0, src),
            )
            out_triton = rotate_to_local_triton(
                x=x_triton,
                src=src,
                wigner=wigner_triton,
                coeff_index=coeff_index,
                dim_full=dim_full,
            )
            torch.testing.assert_close(out_triton, out_ref, atol=3.0e-2, rtol=3.0e-2)

            grad_out = torch.randn_like(out_ref)
            grad_x_ref, grad_wigner_ref = torch.autograd.grad(
                out_ref,
                (x_ref, wigner_ref),
                grad_outputs=grad_out,
            )
            grad_x_triton, grad_wigner_triton = torch.autograd.grad(
                out_triton,
                (x_triton, wigner_triton),
                grad_outputs=grad_out,
            )
            torch.testing.assert_close(
                grad_x_triton,
                grad_x_ref,
                atol=3.0e-2,
                rtol=3.0e-2,
            )
            torch.testing.assert_close(
                grad_wigner_triton,
                grad_wigner_ref,
                atol=3.0e-2,
                rtol=3.0e-2,
            )

    def test_rotate_back_matches_mixed_precision_reference(self) -> None:
        """Compare Triton rotate-back with bf16 activations and fp32 Wigner."""
        self._require_cuda_bfloat16()
        device = torch.device("cuda")
        x_dtype = torch.bfloat16
        wigner_dtype = torch.float32
        n_edge = 11
        channels = 8
        for lmax, mmax in ((2, 1), (3, 1)):
            dim_full = (lmax + 1) ** 2
            coeff_index = build_m_major_index(lmax, mmax, device=device)
            reduced_dim = int(coeff_index.numel())
            x_local_ref = torch.randn(
                n_edge,
                reduced_dim,
                channels,
                device=device,
                dtype=x_dtype,
                requires_grad=True,
            )
            wigner_ref = torch.randn(
                n_edge,
                dim_full,
                dim_full,
                device=device,
                dtype=wigner_dtype,
                requires_grad=True,
            )
            x_local_triton = x_local_ref.detach().clone().requires_grad_(True)
            wigner_triton = wigner_ref.detach().clone().requires_grad_(True)

            out_ref = torch.bmm(
                project_Dt_from_m(
                    Dt_full=wigner_ref,
                    coeff_index_m=coeff_index,
                    ebed_dim_full=dim_full,
                    cache=None,
                    key_lmax=lmax,
                    key_mmax=mmax,
                ).to(dtype=x_dtype),
                x_local_ref,
            )
            out_triton = rotate_back_triton(
                x_local=x_local_triton,
                wigner=wigner_triton,
                coeff_index=coeff_index,
                dim_full=dim_full,
            )
            torch.testing.assert_close(out_triton, out_ref, atol=3.0e-2, rtol=3.0e-2)

            grad_out = torch.randn_like(out_ref)
            grad_x_ref, grad_wigner_ref = torch.autograd.grad(
                out_ref,
                (x_local_ref, wigner_ref),
                grad_outputs=grad_out,
            )
            grad_x_triton, grad_wigner_triton = torch.autograd.grad(
                out_triton,
                (x_local_triton, wigner_triton),
                grad_outputs=grad_out,
            )
            torch.testing.assert_close(
                grad_x_triton,
                grad_x_ref,
                atol=3.0e-2,
                rtol=3.0e-2,
            )
            torch.testing.assert_close(
                grad_wigner_triton,
                grad_wigner_ref,
                atol=3.0e-2,
                rtol=3.0e-2,
            )

    def test_rotate_to_local_matches_bfloat16_autocast_semantics(self) -> None:
        """Use the activation dtype selected by AMP for Triton rotate-to-local."""
        self._require_cuda_bfloat16()
        device = torch.device("cuda")
        act_dtype = torch.bfloat16
        wigner_dtype = torch.float32
        n_node = 7
        n_edge = 11
        dim_full = 16
        channels = 8
        coeff_index = build_m_major_index(3, 1, device=device)
        src = torch.randint(0, n_node, (n_edge,), device=device, dtype=torch.long)
        x_ref = torch.randn(
            n_node,
            dim_full,
            channels,
            device=device,
            dtype=act_dtype,
            requires_grad=True,
        )
        wigner_ref = torch.randn(
            n_edge,
            dim_full,
            dim_full,
            device=device,
            dtype=wigner_dtype,
            requires_grad=True,
        )
        x_triton = x_ref.detach().clone().requires_grad_(True)
        wigner_triton = wigner_ref.detach().clone().requires_grad_(True)

        D_m_prime = project_D_to_m(
            D_full=wigner_ref,
            coeff_index_m=coeff_index,
            ebed_dim_full=dim_full,
            cache=None,
            key_lmax=3,
            key_mmax=1,
        ).to(dtype=act_dtype)
        out_ref = torch.bmm(D_m_prime, x_ref.index_select(0, src))
        out_triton = rotate_to_local_triton(
            x=x_triton,
            src=src,
            wigner=wigner_triton,
            coeff_index=coeff_index,
            dim_full=dim_full,
        )
        torch.testing.assert_close(out_triton, out_ref, atol=5.0e-2, rtol=5.0e-2)

        grad_out = torch.randn_like(out_ref)
        grad_x_ref, grad_wigner_ref = torch.autograd.grad(
            out_ref,
            (x_ref, wigner_ref),
            grad_outputs=grad_out,
        )
        grad_x_triton, grad_wigner_triton = torch.autograd.grad(
            out_triton,
            (x_triton, wigner_triton),
            grad_outputs=grad_out,
        )
        torch.testing.assert_close(
            grad_x_triton,
            grad_x_ref,
            atol=5.0e-2,
            rtol=5.0e-2,
        )
        torch.testing.assert_close(
            grad_wigner_triton,
            grad_wigner_ref,
            atol=5.0e-2,
            rtol=5.0e-2,
        )

    def test_rotate_back_matches_bfloat16_autocast_semantics(self) -> None:
        """Use the activation dtype selected by AMP for Triton rotate-back."""
        self._require_cuda_bfloat16()
        device = torch.device("cuda")
        act_dtype = torch.bfloat16
        wigner_dtype = torch.float32
        n_edge = 11
        dim_full = 16
        channels = 8
        coeff_index = build_m_major_index(3, 1, device=device)
        reduced_dim = int(coeff_index.numel())
        x_local_ref = torch.randn(
            n_edge,
            reduced_dim,
            channels,
            device=device,
            dtype=act_dtype,
            requires_grad=True,
        )
        wigner_ref = torch.randn(
            n_edge,
            dim_full,
            dim_full,
            device=device,
            dtype=wigner_dtype,
            requires_grad=True,
        )
        x_local_triton = x_local_ref.detach().clone().requires_grad_(True)
        wigner_triton = wigner_ref.detach().clone().requires_grad_(True)

        Dt_from_m = project_Dt_from_m(
            Dt_full=wigner_ref,
            coeff_index_m=coeff_index,
            ebed_dim_full=dim_full,
            cache=None,
            key_lmax=3,
            key_mmax=1,
        ).to(dtype=act_dtype)
        out_ref = torch.bmm(Dt_from_m, x_local_ref)
        out_triton = rotate_back_triton(
            x_local=x_local_triton,
            wigner=wigner_triton,
            coeff_index=coeff_index,
            dim_full=dim_full,
        )
        torch.testing.assert_close(out_triton, out_ref, atol=5.0e-2, rtol=5.0e-2)

        grad_out = torch.randn_like(out_ref)
        grad_x_ref, grad_wigner_ref = torch.autograd.grad(
            out_ref,
            (x_local_ref, wigner_ref),
            grad_outputs=grad_out,
        )
        grad_x_triton, grad_wigner_triton = torch.autograd.grad(
            out_triton,
            (x_local_triton, wigner_triton),
            grad_outputs=grad_out,
        )
        torch.testing.assert_close(
            grad_x_triton,
            grad_x_ref,
            atol=5.0e-2,
            rtol=5.0e-2,
        )
        torch.testing.assert_close(
            grad_wigner_triton,
            grad_wigner_ref,
            atol=5.0e-2,
            rtol=5.0e-2,
        )

    def test_generic_small_k_falls_back_to_reference_forward_backward(self) -> None:
        """Fallback to eager bmm when generic Triton tiles would have K < 16."""
        device = torch.device("cuda")
        dtype = torch.float32
        lmax, mmax = 4, 0
        dim_full = (lmax + 1) ** 2
        n_node = 7
        n_edge = 11
        channels = 8
        coeff_index = build_m_major_index(lmax, mmax, device=device)
        self.assertLess(int(coeff_index.numel()), 16)

        src = torch.randint(0, n_node, (n_edge,), device=device, dtype=torch.long)
        x_ref = torch.randn(
            n_node,
            dim_full,
            channels,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        wigner_ref = torch.randn(
            n_edge,
            dim_full,
            dim_full,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        x_triton = x_ref.detach().clone().requires_grad_(True)
        wigner_triton = wigner_ref.detach().clone().requires_grad_(True)

        out_ref = torch.bmm(
            project_D_to_m(
                D_full=wigner_ref,
                coeff_index_m=coeff_index,
                ebed_dim_full=dim_full,
                cache=None,
                key_lmax=lmax,
                key_mmax=mmax,
            ),
            x_ref.index_select(0, src),
        )
        out_triton = rotate_to_local_triton(
            x=x_triton,
            src=src,
            wigner=wigner_triton,
            coeff_index=coeff_index,
            dim_full=dim_full,
        )
        torch.testing.assert_close(out_triton, out_ref, atol=1.0e-5, rtol=1.0e-5)

        grad_out = torch.randn_like(out_ref)
        grad_x_ref, grad_wigner_ref = torch.autograd.grad(
            out_ref,
            (x_ref, wigner_ref),
            grad_outputs=grad_out,
        )
        grad_x_triton, grad_wigner_triton = torch.autograd.grad(
            out_triton,
            (x_triton, wigner_triton),
            grad_outputs=grad_out,
        )
        torch.testing.assert_close(
            grad_x_triton,
            grad_x_ref,
            atol=1.0e-5,
            rtol=1.0e-5,
        )
        torch.testing.assert_close(
            grad_wigner_triton,
            grad_wigner_ref,
            atol=1.0e-5,
            rtol=1.0e-5,
        )

        x_local_ref = torch.randn(
            n_edge,
            int(coeff_index.numel()),
            channels,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        wigner_back_ref = torch.randn(
            n_edge,
            dim_full,
            dim_full,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        x_local_triton = x_local_ref.detach().clone().requires_grad_(True)
        wigner_back_triton = wigner_back_ref.detach().clone().requires_grad_(True)

        out_back_ref = torch.bmm(
            project_Dt_from_m(
                Dt_full=wigner_back_ref,
                coeff_index_m=coeff_index,
                ebed_dim_full=dim_full,
                cache=None,
                key_lmax=lmax,
                key_mmax=mmax,
            ),
            x_local_ref,
        )
        out_back_triton = rotate_back_triton(
            x_local=x_local_triton,
            wigner=wigner_back_triton,
            coeff_index=coeff_index,
            dim_full=dim_full,
        )
        torch.testing.assert_close(
            out_back_triton,
            out_back_ref,
            atol=1.0e-5,
            rtol=1.0e-5,
        )

        grad_back = torch.randn_like(out_back_ref)
        grad_x_local_ref, grad_wigner_back_ref = torch.autograd.grad(
            out_back_ref,
            (x_local_ref, wigner_back_ref),
            grad_outputs=grad_back,
        )
        grad_x_local_triton, grad_wigner_back_triton = torch.autograd.grad(
            out_back_triton,
            (x_local_triton, wigner_back_triton),
            grad_outputs=grad_back,
        )
        torch.testing.assert_close(
            grad_x_local_triton,
            grad_x_local_ref,
            atol=1.0e-5,
            rtol=1.0e-5,
        )
        torch.testing.assert_close(
            grad_wigner_back_triton,
            grad_wigner_back_ref,
            atol=1.0e-5,
            rtol=1.0e-5,
        )

    def test_generic_large_k_matches_reference_forward_backward(self) -> None:
        """Exercise the true generic Triton path when reduced_dim >= 16."""
        device = torch.device("cuda")
        dtype = torch.float32
        n_node = 7
        n_edge = 11
        channels = 8
        for lmax, mmax in ((4, 2), (4, 4), (5, 2)):
            dim_full = (lmax + 1) ** 2
            coeff_index = build_m_major_index(lmax, mmax, device=device)
            self.assertGreaterEqual(int(coeff_index.numel()), 16)

            src = torch.randint(0, n_node, (n_edge,), device=device, dtype=torch.long)
            x_ref = torch.randn(
                n_node,
                dim_full,
                channels,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            wigner_ref = torch.randn(
                n_edge,
                dim_full,
                dim_full,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            x_triton = x_ref.detach().clone().requires_grad_(True)
            wigner_triton = wigner_ref.detach().clone().requires_grad_(True)

            out_ref = torch.bmm(
                project_D_to_m(
                    D_full=wigner_ref,
                    coeff_index_m=coeff_index,
                    ebed_dim_full=dim_full,
                    cache=None,
                    key_lmax=lmax,
                    key_mmax=mmax,
                ),
                x_ref.index_select(0, src),
            )
            out_triton = rotate_to_local_triton(
                x=x_triton,
                src=src,
                wigner=wigner_triton,
                coeff_index=coeff_index,
                dim_full=dim_full,
            )
            torch.testing.assert_close(out_triton, out_ref, atol=1.0e-5, rtol=1.0e-5)

            grad_out = torch.randn_like(out_ref)
            grad_x_ref, grad_wigner_ref = torch.autograd.grad(
                out_ref,
                (x_ref, wigner_ref),
                grad_outputs=grad_out,
            )
            grad_x_triton, grad_wigner_triton = torch.autograd.grad(
                out_triton,
                (x_triton, wigner_triton),
                grad_outputs=grad_out,
            )
            torch.testing.assert_close(
                grad_x_triton,
                grad_x_ref,
                atol=1.0e-5,
                rtol=1.0e-5,
            )
            torch.testing.assert_close(
                grad_wigner_triton,
                grad_wigner_ref,
                atol=1.0e-5,
                rtol=1.0e-5,
            )

            x_local_ref = torch.randn(
                n_edge,
                int(coeff_index.numel()),
                channels,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            wigner_back_ref = torch.randn(
                n_edge,
                dim_full,
                dim_full,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            x_local_triton = x_local_ref.detach().clone().requires_grad_(True)
            wigner_back_triton = wigner_back_ref.detach().clone().requires_grad_(True)

            out_back_ref = torch.bmm(
                project_Dt_from_m(
                    Dt_full=wigner_back_ref,
                    coeff_index_m=coeff_index,
                    ebed_dim_full=dim_full,
                    cache=None,
                    key_lmax=lmax,
                    key_mmax=mmax,
                ),
                x_local_ref,
            )
            out_back_triton = rotate_back_triton(
                x_local=x_local_triton,
                wigner=wigner_back_triton,
                coeff_index=coeff_index,
                dim_full=dim_full,
            )
            torch.testing.assert_close(
                out_back_triton,
                out_back_ref,
                atol=1.0e-5,
                rtol=1.0e-5,
            )

            grad_back = torch.randn_like(out_back_ref)
            grad_x_local_ref, grad_wigner_back_ref = torch.autograd.grad(
                out_back_ref,
                (x_local_ref, wigner_back_ref),
                grad_outputs=grad_back,
            )
            grad_x_local_triton, grad_wigner_back_triton = torch.autograd.grad(
                out_back_triton,
                (x_local_triton, wigner_back_triton),
                grad_outputs=grad_back,
            )
            torch.testing.assert_close(
                grad_x_local_triton,
                grad_x_local_ref,
                atol=1.0e-5,
                rtol=1.0e-5,
            )
            torch.testing.assert_close(
                grad_wigner_back_triton,
                grad_wigner_back_ref,
                atol=1.0e-5,
                rtol=1.0e-5,
            )
