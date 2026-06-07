# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import torch

from deepmd.pt.model.descriptor.sezm_nn import (
    S2GridNet,
    S2GridProjector,
    SO3GridNet,
    SO3GridProjector,
    WignerDCalculator,
    build_edge_quaternion,
    build_m_major_index,
    load_lebedev_rule,
    quaternion_multiply,
    quaternion_z_rotation,
    resolve_s2_grid_resolution,
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


def _legal_so3_frame_mask(projector: SO3GridProjector) -> torch.Tensor:
    mask = torch.ones(
        projector.coeff_dim,
        dtype=torch.bool,
        device=projector.to_grid_mat.device,
    )
    n_frames = projector.n_frames
    for degree in range(projector.lmax + 1):
        for m_order in range(-degree, degree + 1):
            packed_idx = degree * degree + degree + m_order
            for frame_pos, frame_order in enumerate(projector.frame_set):
                flat_idx = packed_idx * n_frames + frame_pos
                if flat_idx >= projector.coeff_dim:
                    continue
                if abs(frame_order) > degree:
                    mask[flat_idx] = False
    return mask


class TestS2GridProjector(unittest.TestCase):
    """Test S2 projection invariants."""

    def setUp(self) -> None:
        self.device = torch.device("cpu")
        torch.manual_seed(0)

    def test_lebedev_roundtrip_preserves_bandlimited_coefficients(self) -> None:
        """Lebedev quadrature should reconstruct coefficients up to lmax."""
        projector = S2GridProjector(
            lmax=3,
            dtype=torch.float64,
            grid_resolution_list=None,
            coefficient_layout="packed",
            grid_method="lebedev",
        ).to(self.device)
        x = torch.randn(
            5, projector.coeff_dim, 3, device=self.device, dtype=torch.float64
        )
        y = projector.from_grid(projector.to_grid(x))
        torch.testing.assert_close(y, x, atol=1e-12, rtol=1e-12)


class TestSwiGLUS2Equivariance(unittest.TestCase):
    """Test default-grid equivariance of full-m and truncated SwiGLU-S2 activations."""

    def setUp(self) -> None:
        self.device = torch.device("cpu")
        torch.manual_seed(0)

    def test_default_full_m_grid_counts_keep_s2_activation_equivariant(self) -> None:
        """Default full-m S2 activation grids should keep SO(3) equivariance."""
        # Each case is (lmax, full_m_grid, fp64_tol, fp32_tol).
        # e3nn full_m_grid is [R_phi, R_theta] after the square-grid lift.
        # Lebedev full_m_grid is [precision, n_points].
        cases_by_method = {
            "e3nn": [
                (2, [8, 8], 4.20e-7, 5.00e-6),  # local: fp64=3.62e-7, fp32=4.77e-7
                (3, [12, 12], 8.10e-7, 5.00e-6),  # local: fp64=7.04e-7, fp32=6.86e-7
                (4, [14, 14], 9.20e-7, 5.00e-6),  # local: fp64=7.97e-7, fp32=1.55e-6
                (5, [18, 18], 1.70e-6, 5.00e-6),  # local: fp64=1.48e-6, fp32=1.49e-6
                (6, [20, 20], 4.80e-6, 5.00e-6),  # local: fp64=4.14e-6, fp32=2.27e-6
                (7, [24, 24], 3.70e-6, 6.00e-6),  # local: fp64=3.19e-6, fp32=2.03e-6
            ],
            "lebedev": [
                (2, [7, 26], 1.00e-12, 5.00e-6),  # local: fp64=2.31e-14, fp32=2.38e-7
                (3, [9, 38], 1.00e-12, 5.00e-6),  # local: fp64=3.58e-14, fp32=3.58e-7
                (4, [13, 74], 1.00e-12, 5.00e-6),  # local: fp64=5.82e-14, fp32=6.56e-7
                (5, [15, 86], 1.00e-12, 5.00e-6),  # local: fp64=3.22e-14, fp32=6.56e-7
                (6, [19, 146], 1.00e-12, 5.00e-6),  # local: fp64=7.99e-14, fp32=8.35e-7
                (7, [21, 170], 1.00e-12, 5.00e-6),  # local: fp64=6.86e-14, fp32=8.79e-7
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
        op_type: str = "glu",
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

        activation = S2GridNet(
            lmax=lmax,
            channels=channels,
            dtype=dtype,
            n_focus=n_focus,
            mode="self",
            op_type=op_type,
            layout="ndfc",
            grid_resolution_list=full_m_grid,
            coefficient_layout="packed",
            grid_method=grid_method,
            mlp_bias=False,
            trainable=False,
            seed=17 + lmax,
        ).to(self.device)
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
        d_matrix, _ = WignerDCalculator(lmax=lmax, dtype=dtype).to(self.device)(quat)

        y_rotated_input = activation(_rotate_ndfc(x, d_matrix))
        y_then_rotated = _rotate_ndfc(activation(x), d_matrix)
        max_error = _max_abs_equivariance_error(
            y_rotated_input,
            y_then_rotated,
        )

        self.assertLessEqual(max_error, tolerance)

    def test_polynomial_grid_mlp_full_m_s2_equivariance(self) -> None:
        """S2 grid MLP should keep full-m SO(3) equivariance."""
        # Each case is (lmax, full_m_grid, fp64_tol, fp32_tol).
        cases_by_method = {
            "e3nn": [
                (2, [8, 8], 2.00e-7, 5.00e-6),  # local: fp64=7.28e-8, fp32=1.34e-7
                (3, [12, 12], 8.00e-7, 5.00e-6),  # local: fp64=1.87e-7, fp32=1.34e-7
                (4, [14, 14], 8.00e-7, 5.00e-6),  # local: fp64=2.17e-7, fp32=4.47e-7
                (5, [18, 18], 1.20e-6, 5.00e-6),  # local: fp64=1.96e-7, fp32=5.96e-7
                (6, [20, 20], 1.20e-6, 5.00e-6),  # local: fp64=9.70e-7, fp32=9.88e-7
                (7, [24, 24], 1.50e-6, 5.00e-6),  # local: fp64=8.25e-7, fp32=9.24e-7
            ],
            "lebedev": [
                (2, [7, 26], 1.00e-12, 5.00e-6),  # local: fp64=4.05e-15, fp32=4.47e-8
                (3, [9, 38], 1.00e-12, 5.00e-6),  # local: fp64=7.19e-15, fp32=7.45e-8
                (4, [13, 74], 1.00e-12, 5.00e-6),  # local: fp64=1.57e-14, fp32=2.53e-7
                (5, [15, 86], 1.00e-12, 5.00e-6),  # local: fp64=7.59e-15, fp32=2.12e-7
                (6, [19, 146], 1.00e-12, 5.00e-6),  # local: fp64=2.21e-14, fp32=2.98e-7
                (7, [21, 170], 1.00e-12, 5.00e-6),  # local: fp64=2.73e-14, fp32=6.85e-7
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
                            op_type="mlp",
                        )

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
                    (2, [6, 8], 2.80e-7, 5.00e-6),  # local: fp64=2.36e-7, fp32=3.58e-7
                    (3, [6, 12], 1.50e-7, 5.00e-6),  # local: fp64=1.22e-7, fp32=5.96e-7
                    (4, [6, 14], 1.33e-6, 5.00e-6),  # local: fp64=1.12e-6, fp32=9.54e-7
                    (5, [6, 18], 1.30e-7, 5.00e-6),  # local: fp64=1.10e-7, fp32=1.43e-6
                    (6, [6, 20], 9.00e-7, 5.00e-6),  # local: fp64=7.64e-7, fp32=1.91e-6
                    (7, [6, 24], 2.60e-7, 5.00e-6),  # local: fp64=2.17e-7, fp32=1.91e-6
                ],
                2: [
                    (2, [8, 8], 4.70e-7, 5.00e-6),  # local: fp64=4.01e-7, fp32=8.34e-7
                    (3, [8, 12], 7.00e-7, 5.00e-6),  # local: fp64=5.99e-7, fp32=8.34e-7
                    (4, [8, 14], 7.00e-7, 5.00e-6),  # local: fp64=6.02e-7, fp32=1.67e-6
                    (5, [8, 18], 1.40e-6, 5.00e-6),  # local: fp64=1.19e-6, fp32=1.55e-6
                    (6, [8, 20], 1.55e-6, 5.00e-6),  # local: fp64=1.33e-6, fp32=2.15e-6
                    (7, [8, 24], 1.65e-6, 5.00e-6),  # local: fp64=1.41e-6, fp32=2.62e-6
                ],
            },
            "lebedev": {
                1: [
                    (
                        2,
                        [7, 26],
                        1.00e-12,
                        5.00e-6,
                    ),  # local: fp64=2.31e-14, fp32=2.38e-7
                    (
                        3,
                        [9, 38],
                        1.00e-12,
                        5.00e-6,
                    ),  # local: fp64=3.55e-14, fp32=2.98e-7
                    (
                        4,
                        [13, 74],
                        1.00e-12,
                        5.00e-6,
                    ),  # local: fp64=1.04e-13, fp32=9.54e-7
                    (
                        5,
                        [15, 86],
                        1.00e-12,
                        5.00e-6,
                    ),  # local: fp64=9.34e-14, fp32=7.15e-7
                    (
                        6,
                        [19, 146],
                        1.00e-12,
                        5.00e-6,
                    ),  # local: fp64=8.56e-14, fp32=2.15e-6
                    (
                        7,
                        [21, 170],
                        1.00e-12,
                        5.00e-6,
                    ),  # local: fp64=2.08e-13, fp32=3.34e-6
                ],
                2: [
                    (
                        2,
                        [7, 26],
                        1.00e-12,
                        5.00e-6,
                    ),  # local: fp64=1.50e-14, fp32=2.38e-7
                    (
                        3,
                        [9, 38],
                        1.00e-12,
                        5.00e-6,
                    ),  # local: fp64=5.71e-14, fp32=3.58e-7
                    (
                        4,
                        [13, 74],
                        1.00e-12,
                        5.00e-6,
                    ),  # local: fp64=9.15e-14, fp32=5.96e-7
                    (
                        5,
                        [15, 86],
                        1.00e-12,
                        5.00e-6,
                    ),  # local: fp64=7.83e-14, fp32=4.77e-7
                    (
                        6,
                        [19, 146],
                        1.00e-12,
                        5.00e-6,
                    ),  # local: fp64=1.29e-13, fp32=9.54e-7
                    (
                        7,
                        [21, 170],
                        1.00e-12,
                        5.00e-6,
                    ),  # local: fp64=1.56e-13, fp32=1.43e-6
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
        op_type: str = "glu",
    ) -> None:
        """Assert mmax-truncated S2 activation z-equivariance for one case."""
        torch.manual_seed(2234 + lmax + 100 * mmax)
        truncated_grid = resolve_s2_grid_resolution(
            lmax,
            mmax,
            method=grid_method,
        )
        self.assertEqual(truncated_grid, expected_truncated_grid)

        activation = S2GridNet(
            lmax=lmax,
            mmax=mmax,
            channels=channels,
            dtype=dtype,
            n_focus=n_focus,
            mode="self",
            op_type=op_type,
            layout="nfdc",
            grid_resolution_list=truncated_grid,
            coefficient_layout="m_major",
            grid_method=grid_method,
            mlp_bias=False,
            trainable=False,
            seed=27 + lmax + 100 * mmax,
        ).to(self.device)
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
        d_matrix, _ = WignerDCalculator(lmax=lmax, dtype=dtype).to(self.device)(
            quaternion
        )
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

    def test_polynomial_grid_mlp_mmax_truncated_s2_z_equivariance(self) -> None:
        """S2 grid MLP should keep mmax-truncated z-equivariance."""
        # Each case is (lmax, mmax, truncated_grid, fp64_tol, fp32_tol).
        cases_by_method = {
            "e3nn": {
                1: [
                    (2, [6, 8], 2.00e-7, 5.00e-6),  # local: fp64=5.74e-8, fp32=1.19e-7
                    (3, [6, 12], 2.00e-7, 5.00e-6),  # local: fp64=2.16e-8, fp32=1.49e-7
                    (4, [6, 14], 8.00e-7, 5.00e-6),  # local: fp64=4.90e-7, fp32=2.38e-7
                    (5, [6, 18], 2.00e-7, 5.00e-6),  # local: fp64=3.29e-8, fp32=3.58e-7
                    (6, [6, 20], 4.00e-7, 5.00e-6),  # local: fp64=7.78e-8, fp32=4.17e-7
                    (7, [6, 24], 4.00e-7, 5.00e-6),  # local: fp64=1.14e-7, fp32=6.56e-7
                ],
                2: [
                    (2, [8, 8], 2.00e-7, 5.00e-6),  # local: fp64=7.34e-8, fp32=1.19e-7
                    (3, [8, 12], 4.00e-7, 5.00e-6),  # local: fp64=6.49e-8, fp32=6.56e-7
                    (4, [8, 14], 8.00e-7, 5.00e-6),  # local: fp64=1.33e-7, fp32=2.09e-7
                    (5, [8, 18], 4.00e-7, 5.00e-6),  # local: fp64=2.63e-7, fp32=2.40e-7
                    (6, [8, 20], 8.00e-7, 5.00e-6),  # local: fp64=3.10e-7, fp32=7.45e-7
                    (7, [8, 24], 8.00e-7, 5.00e-6),  # local: fp64=2.95e-7, fp32=2.38e-7
                ],
            },
            "lebedev": {
                1: [
                    (
                        2,
                        [7, 26],
                        1.00e-12,
                        5.00e-6,
                    ),  # local: fp64=2.28e-15, fp32=7.45e-8
                    (
                        3,
                        [9, 38],
                        1.00e-12,
                        5.00e-6,
                    ),  # local: fp64=4.04e-15, fp32=1.79e-7
                    (
                        4,
                        [13, 74],
                        1.00e-12,
                        5.00e-6,
                    ),  # local: fp64=5.44e-14, fp32=2.38e-7
                    (
                        5,
                        [15, 86],
                        1.00e-12,
                        5.00e-6,
                    ),  # local: fp64=1.99e-14, fp32=2.38e-7
                    (
                        6,
                        [19, 146],
                        1.00e-12,
                        5.00e-6,
                    ),  # local: fp64=1.81e-14, fp32=9.54e-7
                    (
                        7,
                        [21, 170],
                        1.00e-12,
                        5.00e-6,
                    ),  # local: fp64=4.86e-14, fp32=3.87e-7
                ],
                2: [
                    (
                        2,
                        [7, 26],
                        1.00e-12,
                        5.00e-6,
                    ),  # local: fp64=2.84e-15, fp32=7.45e-8
                    (
                        3,
                        [9, 38],
                        1.00e-12,
                        5.00e-6,
                    ),  # local: fp64=5.33e-15, fp32=1.19e-7
                    (
                        4,
                        [13, 74],
                        1.00e-12,
                        5.00e-6,
                    ),  # local: fp64=7.45e-15, fp32=1.19e-7
                    (
                        5,
                        [15, 86],
                        1.00e-12,
                        5.00e-6,
                    ),  # local: fp64=1.68e-14, fp32=1.19e-7
                    (
                        6,
                        [19, 146],
                        1.00e-12,
                        5.00e-6,
                    ),  # local: fp64=2.62e-14, fp32=4.77e-7
                    (
                        7,
                        [21, 170],
                        1.00e-12,
                        5.00e-6,
                    ),  # local: fp64=1.98e-14, fp32=2.38e-7
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
                                op_type="mlp",
                            )


class TestSO3GridProjector(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2026)
        self.device = torch.device("cpu")

    def test_roundtrip_preserves_legal_frame_coefficients(self) -> None:
        # WignerDCalculator-based grids keep local macOS fp64 round-trip errors
        # below 4.1e-13 for lmax=1..6 without a dual-basis correction.
        for lmax in range(1, 7):
            with self.subTest(lmax=lmax):
                torch.manual_seed(8100 + lmax)
                projector = SO3GridProjector(
                    lmax=lmax,
                    kmax=1,
                    dtype=torch.float64,
                ).to(self.device)
                x = torch.randn(
                    2,
                    projector.coeff_dim,
                    2,
                    dtype=torch.float64,
                    device=self.device,
                )
                mask = _legal_so3_frame_mask(projector)
                x[:, ~mask, :] = 0.0
                y = projector.from_grid(projector.to_grid(x))
                torch.testing.assert_close(
                    y[:, mask, :],
                    x[:, mask, :],
                    atol=1e-12,
                    rtol=1e-12,
                )
                self.assertLess(float(y[:, ~mask, :].abs().max()), 1e-14)

    def test_projection_matrices_match_direct_wigner_construction(self) -> None:
        projector = SO3GridProjector(
            lmax=2,
            kmax=1,
            dtype=torch.float64,
            lebedev_precision=17,
        )
        points, weights = load_lebedev_rule(
            projector.lebedev_precision,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        gamma = torch.arange(
            projector.n_gamma,
            dtype=torch.float64,
            device=points.device,
        ) * (2.0 * torch.pi / projector.n_gamma)
        edge_quaternion = build_edge_quaternion(points, eps=1e-14)
        edge_quaternion = edge_quaternion.repeat_interleave(projector.n_gamma, dim=0)
        gamma_quaternion = quaternion_z_rotation(gamma).repeat(points.shape[0], 1)
        grid_quaternion = quaternion_multiply(gamma_quaternion, edge_quaternion)
        wigner_grid, _ = WignerDCalculator(
            lmax=projector.lmax,
            dtype=torch.float64,
        ).to(grid_quaternion.device)(grid_quaternion)
        wigner_grid = wigner_grid.transpose(-1, -2).contiguous()
        haar_weight = weights.repeat_interleave(projector.n_gamma) / projector.n_gamma
        to_grid_ref = torch.zeros_like(projector.to_grid_mat)
        from_grid_ref = torch.zeros_like(projector.from_grid_mat)
        for degree in range(projector.lmax + 1):
            for m_order in range(-degree, degree + 1):
                packed_idx = degree * degree + degree + m_order
                for frame_pos, frame_order in enumerate(projector.frame_set):
                    flat_idx = packed_idx * projector.n_frames + frame_pos
                    if abs(frame_order) > degree:
                        continue
                    row = degree * degree + degree + m_order
                    col = degree * degree + degree + frame_order
                    values = wigner_grid[:, row, col]
                    to_grid_ref[:, flat_idx] = values
                    from_grid_ref[flat_idx] = (2 * degree + 1) * haar_weight * values
        torch.testing.assert_close(projector.to_grid_mat, to_grid_ref)
        torch.testing.assert_close(projector.from_grid_mat, from_grid_ref)

    def test_k_zero_slice_matches_wigner_zonal_convention(self) -> None:
        lmax = 6
        projector = SO3GridProjector(lmax=lmax, kmax=0, dtype=torch.float64)
        points, _ = load_lebedev_rule(
            projector.lebedev_precision,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        edge_quaternion = build_edge_quaternion(points, eps=1e-14)
        zonal = (
            WignerDCalculator(lmax=lmax, dtype=torch.float64)
            .to(edge_quaternion.device)
            .forward_zonal(edge_quaternion, lmin=1)
        )
        torch.testing.assert_close(
            projector.to_grid_mat[:, 0], torch.ones_like(points[:, 0])
        )
        torch.testing.assert_close(
            projector.to_grid_mat[:, 1:], zonal, atol=1e-14, rtol=1e-14
        )

    def test_quadratic_gamma_rule_resolves_kmax_two_products(self) -> None:
        projector = SO3GridProjector(lmax=2, kmax=2, dtype=torch.float64)
        self.assertEqual(projector.n_gamma, 7)


class TestSO3GridNet(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2027)
        self.device = torch.device("cpu")

    def test_swiglu_so3_grid_net_equivariance(self) -> None:
        # Each case is (lmax, fp64_tol, fp32_tol). Observed reference errors:
        # fp64: [7.58e-14, 1.50e-13, 1.95e-13, 4.70e-13, 2.84e-13, 7.04e-13]
        # fp32: [2.99e-7, 2.15e-6, 1.55e-6, 2.31e-6, 4.30e-6, 5.13e-6]
        cases = [
            (1, 2e-13, 6e-7),
            (2, 3e-13, 3e-6),
            (3, 4e-13, 3e-6),
            (4, 8e-13, 4e-6),
            (5, 6e-13, 6e-6),
            (6, 1e-12, 7e-6),
        ]
        channels = 2
        dtype_cases = [
            (torch.float64, 1),
            (torch.float32, 2),
        ]
        for dtype, tolerance_index in dtype_cases:
            for case in cases:
                lmax = case[0]
                tolerance = case[tolerance_index]
                with self.subTest(dtype=dtype, lmax=lmax):
                    torch.manual_seed(
                        7100 + lmax + (0 if dtype is torch.float64 else 100)
                    )
                    net = SO3GridNet(
                        lmax=lmax,
                        kmax=1,
                        channels=channels,
                        n_focus=1,
                        mode="self",
                        op_type="glu",
                        dtype=dtype,
                        layout="ndfc",
                        trainable=False,
                    ).to(self.device)
                    x = torch.randn(
                        2,
                        (lmax + 1) ** 2,
                        1,
                        net.query_channels,
                        dtype=dtype,
                        device=self.device,
                    )
                    quat = _random_quaternion(2, dtype=dtype, device=self.device)
                    d_matrix, _ = WignerDCalculator(lmax=lmax, dtype=dtype).to(
                        self.device
                    )(quat)
                    y_rotated_input = net(_rotate_ndfc(x, d_matrix))
                    y_then_rotated = _rotate_ndfc(net(x), d_matrix)
                    torch.testing.assert_close(
                        y_rotated_input,
                        y_then_rotated,
                        atol=tolerance,
                        rtol=tolerance,
                    )

    def test_polynomial_grid_mlp_so3_grid_net_equivariance(self) -> None:
        for lmax in range(1, 5):
            with self.subTest(lmax=lmax):
                torch.manual_seed(8200 + lmax)
                net = SO3GridNet(
                    lmax=lmax,
                    kmax=1,
                    channels=2,
                    n_focus=1,
                    mode="self",
                    op_type="mlp",
                    dtype=torch.float64,
                    layout="ndfc",
                    trainable=False,
                ).to(self.device)
                x = torch.randn(
                    2,
                    (lmax + 1) ** 2,
                    1,
                    net.query_channels,
                    dtype=torch.float64,
                    device=self.device,
                )
                quat = _random_quaternion(2, dtype=torch.float64, device=self.device)
                d_matrix, _ = WignerDCalculator(lmax=lmax, dtype=torch.float64).to(
                    self.device
                )(quat)
                y_rotated_input = net(_rotate_ndfc(x, d_matrix))
                y_then_rotated = _rotate_ndfc(net(x), d_matrix)
                torch.testing.assert_close(
                    y_rotated_input,
                    y_then_rotated,
                    atol=1e-12,
                    rtol=1e-12,
                )

    def test_scalar_router_grid_branch_so3_grid_net_equivariance(self) -> None:
        for lmax in range(1, 5):
            with self.subTest(lmax=lmax):
                torch.manual_seed(8300 + lmax)
                net = SO3GridNet(
                    lmax=lmax,
                    kmax=1,
                    channels=2,
                    n_focus=1,
                    mode="self",
                    op_type="branch",
                    dtype=torch.float64,
                    layout="ndfc",
                    grid_branches=2,
                    trainable=False,
                ).to(self.device)
                x = torch.randn(
                    2,
                    (lmax + 1) ** 2,
                    1,
                    net.query_channels,
                    dtype=torch.float64,
                    device=self.device,
                )
                quat = _random_quaternion(2, dtype=torch.float64, device=self.device)
                d_matrix, _ = WignerDCalculator(lmax=lmax, dtype=torch.float64).to(
                    self.device
                )(quat)
                y_rotated_input = net(_rotate_ndfc(x, d_matrix))
                y_then_rotated = _rotate_ndfc(net(x), d_matrix)
                torch.testing.assert_close(
                    y_rotated_input,
                    y_then_rotated,
                    atol=1e-12,
                    rtol=1e-12,
                )

    def test_kmax_two_quadratic_grid_ops_are_equivariant(self) -> None:
        for op_type in ["glu", "mlp", "branch"]:
            with self.subTest(op_type=op_type):
                torch.manual_seed(8400)
                net = SO3GridNet(
                    lmax=2,
                    kmax=2,
                    channels=2,
                    n_focus=1,
                    mode="self",
                    op_type=op_type,
                    dtype=torch.float64,
                    layout="ndfc",
                    grid_branches=2,
                    trainable=False,
                ).to(self.device)
                x = torch.randn(
                    2,
                    9,
                    1,
                    net.query_channels,
                    dtype=torch.float64,
                    device=self.device,
                )
                quat = _random_quaternion(2, dtype=torch.float64, device=self.device)
                d_matrix, _ = WignerDCalculator(lmax=2, dtype=torch.float64).to(
                    self.device
                )(quat)
                y_rotated_input = net(_rotate_ndfc(x, d_matrix))
                y_then_rotated = _rotate_ndfc(net(x), d_matrix)
                torch.testing.assert_close(
                    y_rotated_input,
                    y_then_rotated,
                    atol=1e-12,
                    rtol=1e-12,
                )


class TestSO3CounterExample(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2028)
        self.device = torch.device("cpu")
        self.l1_to_cartesian = torch.tensor(
            [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
            device=self.device,
            dtype=torch.float64,
        )

    def test_so3_features_span_odd_targets_but_s2_features_do_not(self) -> None:
        n_sample = 2048
        channels = 48
        x = torch.randn(n_sample, 3, 3, dtype=torch.float64, device=self.device)
        target, target_det = self._odd_targets_from_l1_coefficients(x)

        s2_projector = S2GridProjector(
            lmax=2,
            dtype=torch.float64,
            grid_method="lebedev",
            grid_resolution_list=[17, 110],
        )
        so3_projector = SO3GridProjector(
            lmax=2,
            kmax=1,
            dtype=torch.float64,
            lebedev_precision=17,
        )
        s2_features_by_m = self._s2_quadratic_features(x[:, :2], s2_projector, channels)
        so3_features_by_m = self._so3_quadratic_features(
            x[:, :2],
            so3_projector,
            channels,
        )
        s2_residual = self._best_linear_residual(
            s2_features_by_m.reshape(n_sample * 3, channels),
            target,
        )
        so3_residual = self._best_linear_residual(
            so3_features_by_m.reshape(n_sample * 3, so3_projector.n_frames * channels),
            target,
        )
        s2_det_features = self._couple_l1_features_to_scalar(s2_features_by_m, x[:, 2])
        so3_det_features = self._couple_l1_features_to_scalar(
            so3_features_by_m,
            x[:, 2],
        )
        s2_det_residual = self._best_linear_residual(s2_det_features, target_det)
        so3_det_residual = self._best_linear_residual(so3_det_features, target_det)
        self.assertGreater(s2_residual, 0.9)
        self.assertLess(so3_residual, 0.35)
        self.assertGreater(s2_det_residual, 0.9)
        self.assertLess(so3_det_residual, 0.35)

    def _s2_quadratic_features(
        self,
        x: torch.Tensor,
        projector: S2GridProjector,
        channels: int,
    ) -> torch.Tensor:
        weight = torch.randn(2, 2 * channels, dtype=x.dtype, device=x.device)
        coeff = x.new_zeros(x.shape[0], projector.coeff_dim, 2 * channels)
        coeff[:, 1:4, :] = torch.einsum("bmi,ic->bmc", x.transpose(1, 2), weight)
        grid = projector.to_grid(coeff)
        grid_a, grid_b = grid.chunk(2, dim=-1)
        out = projector.from_grid(grid_a * grid_b)
        return out[:, 1:4, :]

    def _so3_quadratic_features(
        self,
        x: torch.Tensor,
        projector: SO3GridProjector,
        channels: int,
    ) -> torch.Tensor:
        n_frames = projector.n_frames
        weight = torch.randn(
            2,
            2 * n_frames * channels,
            dtype=x.dtype,
            device=x.device,
        )
        coeff = x.new_zeros(x.shape[0], projector.coeff_dim, 2 * channels)
        mixed = torch.einsum("bmi,ic->bmc", x.transpose(1, 2), weight)
        for local_m in range(3):
            packed_idx = 1 + local_m
            start = packed_idx * n_frames
            stop = start + n_frames
            coeff[:, start:stop, :] = mixed[:, local_m, :].reshape(
                x.shape[0],
                n_frames,
                2 * channels,
            )
        grid = projector.to_grid(coeff)
        grid_a, grid_b = grid.chunk(2, dim=-1)
        out = projector.from_grid(grid_a * grid_b)
        rows = []
        for local_m in range(3):
            packed_idx = 1 + local_m
            rows.extend(range(packed_idx * n_frames, (packed_idx + 1) * n_frames))
        return out[:, rows, :].reshape(x.shape[0], 3, n_frames * channels)

    def _couple_l1_features_to_scalar(
        self,
        features: torch.Tensor,
        vector: torch.Tensor,
    ) -> torch.Tensor:
        features_cartesian = torch.einsum(
            "bmp,mi->bip",
            features,
            self.l1_to_cartesian.to(dtype=features.dtype, device=features.device),
        )
        vector_cartesian = self._l1_coefficients_to_cartesian(vector)
        return torch.einsum("bip,bi->bp", features_cartesian, vector_cartesian)

    def _odd_targets_from_l1_coefficients(
        self,
        coeff: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        first = self._l1_coefficients_to_cartesian(coeff[:, 0])
        second = self._l1_coefficients_to_cartesian(coeff[:, 1])
        third = self._l1_coefficients_to_cartesian(coeff[:, 2])
        cross_cartesian = torch.linalg.cross(first, second, dim=-1)
        cross_coeff = self._cartesian_to_l1_coefficients(cross_cartesian)
        determinant = torch.sum(cross_cartesian * third, dim=-1)
        return cross_coeff, determinant

    def _l1_coefficients_to_cartesian(self, coeff: torch.Tensor) -> torch.Tensor:
        return coeff @ self.l1_to_cartesian.to(dtype=coeff.dtype, device=coeff.device)

    def _cartesian_to_l1_coefficients(self, vector: torch.Tensor) -> torch.Tensor:
        return (
            vector @ self.l1_to_cartesian.to(dtype=vector.dtype, device=vector.device).T
        )

    def _best_linear_residual(
        self,
        features: torch.Tensor,
        target: torch.Tensor,
    ) -> float:
        y = target.reshape(-1, 1)
        solution = torch.linalg.lstsq(features, y).solution
        residual = features @ solution - y
        return float(residual.norm() / y.norm().clamp_min(1e-30))


if __name__ == "__main__":
    unittest.main()
