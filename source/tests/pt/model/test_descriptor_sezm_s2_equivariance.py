# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import torch

from deepmd.pt.model.descriptor.sezm_nn import (
    S2GridProjector,
    SwiGLUS2Activation,
    WignerDCalculator,
    build_m_major_index,
    quaternion_z_rotation,
    resolve_s2_grid_resolution,
)
from deepmd.pt.utils import (
    env,
)


def _random_quaternion(
    n_batch: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample normalized quaternions in ``(w, x, y, z)`` order."""
    q = torch.randn(n_batch, 4, device=device, dtype=dtype)
    return q / torch.sqrt(torch.sum(q * q, dim=-1, keepdim=True))


def _rotate_ndfc(x: torch.Tensor, d_matrix: torch.Tensor) -> torch.Tensor:
    """Rotate coefficient-layout tensors with shape ``(N, D, F, C)``."""
    return torch.einsum("nij,njfc->nifc", d_matrix, x)


def _rotate_nfdc(x: torch.Tensor, d_matrix: torch.Tensor) -> torch.Tensor:
    """Rotate coefficient-layout tensors with shape ``(N, F, D, C)``."""
    return torch.einsum("nij,nfjc->nfic", d_matrix, x)


def _max_abs_equivariance_error(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    """Compute the maximum absolute equivariance error."""
    return float(torch.max(torch.abs(lhs - rhs)).item())


class TestS2GridProjector(unittest.TestCase):
    """Test S2 projection invariants."""

    def setUp(self) -> None:
        self.device = env.DEVICE
        torch.manual_seed(0)

    def test_lebedev_roundtrip_preserves_bandlimited_coefficients(self) -> None:
        """Lebedev quadrature should reconstruct coefficients up to lmax."""
        projector = S2GridProjector(
            lmax=3,
            dtype=torch.float64,
            grid_resolution_list=None,
            coefficient_layout="packed",
            grid_method="lebedev",
        )
        x = torch.randn(
            5, projector.coeff_dim, 3, device=self.device, dtype=torch.float64
        )
        y = projector.from_grid(projector.to_grid(x))
        torch.testing.assert_close(y, x, atol=1e-12, rtol=1e-12)


class TestSwiGLUS2Equivariance(unittest.TestCase):
    """Test default-grid equivariance of full-m and truncated SwiGLU-S2 activations."""

    def setUp(self) -> None:
        self.device = env.DEVICE
        torch.manual_seed(0)

    def test_default_full_m_grid_counts_keep_s2_activation_equivariant(self) -> None:
        """Default full-m S2 activation grids should keep SO(3) equivariance."""
        # Each case is (lmax, full_m_grid, fp64_tol, fp32_tol).
        # e3nn full_m_grid is [R_phi, R_theta] after the square-grid lift.
        # Lebedev full_m_grid is [precision, n_points].
        cases_by_method = {
            "e3nn": [
                (2, [8, 8], 3.63e-7, 4.77e-7),
                (3, [12, 12], 7.04e-7, 6.86e-7),
                (4, [14, 14], 7.97e-7, 1.55e-6),
                (5, [18, 18], 1.48e-6, 1.50e-6),
                (6, [20, 20], 4.14e-6, 2.28e-6),
                (7, [24, 24], 3.20e-6, 2.03e-6),
            ],
            "lebedev": [
                (2, [7, 26], 2.31e-14, 2.39e-7),
                (3, [9, 38], 3.58e-14, 3.58e-7),
                (4, [13, 74], 5.82e-14, 6.56e-7),
                (5, [15, 86], 3.22e-14, 6.56e-7),
                (6, [19, 146], 7.99e-14, 8.35e-7),
                (7, [21, 170], 6.87e-14, 8.80e-7),
            ],
        }
        dtype_cases = [
            (torch.float64, 0),
            (torch.float32, 1),
        ]
        n_batch = 3
        n_focus = 1
        channels = 2

        for dtype, tolerance_index in dtype_cases:
            for method, cases in cases_by_method.items():
                for lmax, expected_full_m_grid, *tolerances in cases:
                    with self.subTest(
                        method=method,
                        dtype=dtype,
                        lmax=lmax,
                        grid=expected_full_m_grid,
                    ):
                        self._assert_default_full_m_s2_activation_equivariance(
                            grid_method=method,
                            lmax=lmax,
                            expected_full_m_grid=expected_full_m_grid,
                            n_batch=n_batch,
                            n_focus=n_focus,
                            channels=channels,
                            dtype=dtype,
                            tolerance=tolerances[tolerance_index],
                        )

    def _assert_default_full_m_s2_activation_equivariance(
        self,
        *,
        grid_method: str,
        lmax: int,
        expected_full_m_grid: list[int],
        n_batch: int,
        n_focus: int,
        channels: int,
        dtype: torch.dtype,
        tolerance: float,
    ) -> None:
        """Assert full-m S2 activation equivariance for one method/dtype/lmax case."""
        torch.manual_seed(1234 + lmax)
        default_grid = resolve_s2_grid_resolution(
            lmax,
            lmax,
            method=grid_method,
        )
        full_m_grid = (
            [max(default_grid), max(default_grid)]
            if grid_method == "e3nn"
            else default_grid
        )
        self.assertEqual(full_m_grid, expected_full_m_grid)

        activation = SwiGLUS2Activation(
            lmax=lmax,
            channels=channels,
            dtype=dtype,
            n_focus=n_focus,
            layout="ndfc",
            grid_resolution_list=full_m_grid,
            coefficient_layout="packed",
            grid_method=grid_method,
            mlp_bias=False,
            trainable=False,
            seed=17 + lmax,
        )
        self.assertEqual(activation.grid_resolution_list, expected_full_m_grid)

        x = torch.randn(
            n_batch,
            (lmax + 1) ** 2,
            n_focus,
            2 * channels,
            device=self.device,
            dtype=dtype,
        )
        quat = _random_quaternion(n_batch, device=self.device, dtype=dtype)
        d_matrix, _ = WignerDCalculator(lmax=lmax, dtype=dtype)(quat)

        y_rotated_input = activation(_rotate_ndfc(x, d_matrix))
        y_then_rotated = _rotate_ndfc(activation(x), d_matrix)
        max_error = _max_abs_equivariance_error(
            y_rotated_input,
            y_then_rotated,
        )

        self.assertLessEqual(max_error, tolerance)

    def test_default_mmax_truncated_grid_counts_keep_s2_activation_z_equivariant(
        self,
    ) -> None:
        """Default mmax-truncated S2 activation grids should keep z-equivariance."""
        # Each case is (lmax, mmax, truncated_grid, fp64_tol, fp32_tol).
        # e3nn truncated_grid is [R_phi, R_theta] used by the m-major path.
        # Lebedev truncated_grid is [precision, n_points].
        cases_by_method = {
            "e3nn": {
                1: [
                    (2, [6, 8], 2.36e-7, 3.58e-7),
                    (3, [6, 12], 1.22e-7, 5.97e-7),
                    (4, [6, 14], 1.12e-6, 9.54e-7),
                    (5, [6, 18], 1.11e-7, 1.44e-6),
                    (6, [6, 20], 7.64e-7, 1.91e-6),
                    (7, [6, 24], 2.17e-7, 1.91e-6),
                ],
                2: [
                    (2, [8, 8], 4.02e-7, 8.35e-7),
                    (3, [8, 12], 6.00e-7, 8.35e-7),
                    (4, [8, 14], 6.02e-7, 1.67e-6),
                    (5, [8, 18], 1.19e-6, 1.55e-6),
                    (6, [8, 20], 1.33e-6, 2.15e-6),
                    (7, [8, 24], 1.41e-6, 2.63e-6),
                ],
            },
            "lebedev": {
                1: [
                    (2, [7, 26], 2.31e-14, 2.39e-7),
                    (3, [9, 38], 3.56e-14, 2.99e-7),
                    (4, [13, 74], 1.04e-13, 9.54e-7),
                    (5, [15, 86], 9.35e-14, 7.16e-7),
                    (6, [19, 146], 8.56e-14, 2.15e-6),
                    (7, [21, 170], 2.09e-13, 3.34e-6),
                ],
                2: [
                    (2, [7, 26], 1.50e-14, 2.39e-7),
                    (3, [9, 38], 5.71e-14, 3.58e-7),
                    (4, [13, 74], 9.15e-14, 5.97e-7),
                    (5, [15, 86], 7.83e-14, 4.77e-7),
                    (6, [19, 146], 1.29e-13, 9.54e-7),
                    (7, [21, 170], 1.57e-13, 1.44e-6),
                ],
            },
        }
        dtype_cases = [
            (torch.float64, 0),
            (torch.float32, 1),
        ]
        n_batch = 3
        n_focus = 2
        channels = 2

        for dtype, tolerance_index in dtype_cases:
            for method, cases_by_mmax in cases_by_method.items():
                for mmax, cases in cases_by_mmax.items():
                    for lmax, expected_truncated_grid, *tolerances in cases:
                        with self.subTest(
                            method=method,
                            dtype=dtype,
                            lmax=lmax,
                            mmax=mmax,
                            grid=expected_truncated_grid,
                        ):
                            self._assert_default_mmax_truncated_grid_z_equivariance(
                                grid_method=method,
                                lmax=lmax,
                                mmax=mmax,
                                expected_truncated_grid=expected_truncated_grid,
                                n_batch=n_batch,
                                n_focus=n_focus,
                                channels=channels,
                                dtype=dtype,
                                tolerance=tolerances[tolerance_index],
                            )

    def _assert_default_mmax_truncated_grid_z_equivariance(
        self,
        *,
        grid_method: str,
        lmax: int,
        mmax: int,
        expected_truncated_grid: list[int],
        n_batch: int,
        n_focus: int,
        channels: int,
        dtype: torch.dtype,
        tolerance: float,
    ) -> None:
        """Assert mmax-truncated S2 activation z-equivariance for one case."""
        torch.manual_seed(2234 + lmax + 100 * mmax)
        truncated_grid = resolve_s2_grid_resolution(
            lmax,
            mmax,
            method=grid_method,
        )
        self.assertEqual(truncated_grid, expected_truncated_grid)

        activation = SwiGLUS2Activation(
            lmax=lmax,
            mmax=mmax,
            channels=channels,
            dtype=dtype,
            n_focus=n_focus,
            layout="nfdc",
            grid_resolution_list=truncated_grid,
            coefficient_layout="m_major",
            grid_method=grid_method,
            mlp_bias=False,
            trainable=False,
            seed=27 + lmax + 100 * mmax,
        )
        self.assertEqual(activation.grid_resolution_list, expected_truncated_grid)

        coeff_index = build_m_major_index(lmax, mmax, device=self.device)
        x = torch.randn(
            n_batch,
            n_focus,
            int(coeff_index.numel()),
            2 * channels,
            device=self.device,
            dtype=dtype,
        )
        gamma = torch.randn(n_batch, device=self.device, dtype=dtype)
        quaternion = quaternion_z_rotation(gamma)
        d_matrix, _ = WignerDCalculator(lmax=lmax, dtype=dtype)(quaternion)
        d_matrix_reduced = d_matrix.index_select(1, coeff_index).index_select(
            2,
            coeff_index,
        )

        y_rotated_input = activation(_rotate_nfdc(x, d_matrix_reduced))
        y_then_rotated = _rotate_nfdc(activation(x), d_matrix_reduced)
        max_error = _max_abs_equivariance_error(
            y_rotated_input,
            y_then_rotated,
        )

        self.assertLessEqual(max_error, tolerance)
