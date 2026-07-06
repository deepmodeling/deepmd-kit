# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Grid projection helpers for DPA4/SeZM function-space nonlinearities.

The projectors in this module only handle basis transforms.  They do not apply
channel mixing or nonlinearities.  A projector maps coefficient tensors to a
fixed quadrature grid, and maps grid fields back to coefficients with the
matching quadrature rule.

This module is the dpmodel (array-API) port of
``deepmd.pt.model.descriptor.sezm_nn.projection``.
"""

from __future__ import (
    annotations,
)

import math
from typing import (
    Any,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
    NativeOP,
)
from deepmd.dpmodel.array_api import (
    xp_asarray_nodetach,
)
from deepmd.dpmodel.utils.lebedev import (
    LEBEDEV_PRECISION_TO_NPOINTS,
    load_lebedev_rule,
)
from deepmd.dpmodel.utils.spherical_harmonics import (
    real_spherical_harmonics,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .indexing import (
    build_l_major_index,
    build_m_major_index,
    so3_packed_index,
)
from .wignerd import (
    WignerDCalculator,
    build_edge_quaternion,
    quaternion_multiply,
    quaternion_z_rotation,
)


class BaseGridProjector(NativeOP):
    """
    Base class for fixed coefficient-to-grid projection matrices.

    Subclasses build ``to_grid_mat`` with shape ``(G, J)`` and
    ``from_grid_mat`` with shape ``(J, G)``, where ``G`` is the number of grid
    samples and ``J`` is the flattened coefficient axis consumed by the grid
    net.  For ordinary S2 projections, ``J`` is the SO(3) feature coefficient
    axis: ``D = (lmax + 1)^2`` in packed layout, or the retained ``D_m`` axis in
    m-major layout.  For SO(3) frame projections, ``J = D * n_frames`` with
    frame index packed inside each coefficient row.
    """

    def __init__(
        self,
        *,
        lmax: int,
        mmax: int | None,
        precision: str = DEFAULT_PRECISION,
        n_frames: int,
        coefficient_layout: str,
    ) -> None:
        self.lmax = int(lmax)
        self.mmax = int(self.lmax if mmax is None else mmax)
        if self.mmax < 0:
            raise ValueError("`mmax` must be non-negative")
        if self.mmax > self.lmax:
            raise ValueError("`mmax` must be <= `lmax`")
        self.coefficient_layout = str(coefficient_layout).lower()
        if self.coefficient_layout not in {"packed", "m_major"}:
            raise ValueError(
                "`coefficient_layout` must be either 'packed' or 'm_major'"
            )
        self.precision = precision
        self.n_frames = int(n_frames)
        self.packed_dim = int((self.lmax + 1) ** 2)

        coeff_index = self._build_coefficient_index()
        to_grid_mat, from_grid_mat = self._build_projection_mats(coeff_index)
        self.coeff_dim = int(to_grid_mat.shape[1])
        self.grid_size = int(to_grid_mat.shape[0])
        if self.coeff_dim != int(from_grid_mat.shape[0]):
            raise ValueError("Projection matrix coefficient axes `J` do not match")
        if self.grid_size != int(from_grid_mat.shape[1]):
            raise ValueError("Projection matrix grid axes `G` do not match")
        prec = PRECISION_DICT[self.precision.lower()]
        self.to_grid_mat = np.ascontiguousarray(to_grid_mat, dtype=prec)
        self.from_grid_mat = np.ascontiguousarray(from_grid_mat, dtype=prec)

    def call(self, *args: Any, **kwargs: Any) -> Any:
        """Projectors expose ``to_grid``/``from_grid``; there is no forward."""
        raise NotImplementedError(
            "BaseGridProjector has no forward; use `to_grid` or `from_grid`"
        )

    def to_grid(self, embedding: Any) -> Any:
        """Project flattened coefficients ``(N, J, C)`` to grid fields ``(N, G, C)``."""
        xp = array_api_compat.array_namespace(embedding)
        to_grid_mat = xp_asarray_nodetach(
            xp, self.to_grid_mat[...], device=array_api_compat.device(embedding)
        )
        to_grid_mat = xp.astype(to_grid_mat, embedding.dtype)
        # einsum "gj,njc->ngc" as a broadcast batched matmul
        return xp.matmul(to_grid_mat[None, ...], embedding)

    def from_grid(self, grid: Any) -> Any:
        """Project grid fields ``(N, G, C)`` back to flattened coefficients ``(N, J, C)``."""
        xp = array_api_compat.array_namespace(grid)
        from_grid_mat = xp_asarray_nodetach(
            xp, self.from_grid_mat[...], device=array_api_compat.device(grid)
        )
        from_grid_mat = xp.astype(from_grid_mat, grid.dtype)
        # einsum "jg,ngc->njc" as a broadcast batched matmul
        return xp.matmul(from_grid_mat[None, ...], grid)

    def _build_coefficient_index(self) -> np.ndarray:
        """Build the coefficient subset consumed by the projector matrices."""
        if self.coefficient_layout == "m_major":
            return build_m_major_index(self.lmax, self.mmax)
        if self.mmax == self.lmax:
            return np.arange((self.lmax + 1) ** 2, dtype=np.int64)
        return build_l_major_index(self.lmax, self.mmax)

    def _build_projection_mats(
        self,
        coeff_index: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build ``to_grid_mat (G, J)`` and ``from_grid_mat (J, G)``."""
        raise NotImplementedError


class S2GridProjector(BaseGridProjector):
    """
    Project SO(3) coefficients to/from a flattened S2 grid.

    Parameters
    ----------
    lmax
        Maximum spherical harmonic degree.
    mmax
        Maximum order kept in the coefficient layout. If None, use ``lmax``.
    precision
        Buffer precision used by the projection matrices.
    grid_resolution_list
        Two-element resolution list. For ``grid_method='e3nn'`` it is
        ``[R_phi, R_theta]`` and is converted to the ``e3nn``
        ``(lat, long) = (R_theta, R_phi)`` ordering. For
        ``grid_method='lebedev'`` it is ``[precision, n_points]``.
    coefficient_layout
        Coefficient ordering expected by the caller:
        - ``"packed"``: packed ``(l, m)`` order, optionally truncated by ``mmax``.
        - ``"m_major"``: reduced m-major order used inside ``SO2Convolution``.
    grid_method
        S2 quadrature backend. Must be ``"e3nn"`` or ``"lebedev"``.
    """

    def __init__(
        self,
        *,
        lmax: int,
        mmax: int | None = None,
        precision: str = DEFAULT_PRECISION,
        grid_resolution_list: list[int] | None = None,
        coefficient_layout: str = "packed",
        grid_method: str = "e3nn",
    ) -> None:
        lmax_i = int(lmax)
        mmax_i = int(lmax_i if mmax is None else mmax)
        self.grid_method = str(grid_method).lower()
        if self.grid_method not in {"e3nn", "lebedev"}:
            raise ValueError("`grid_method` must be either 'e3nn' or 'lebedev'")

        self.grid_resolution_list = _normalize_s2_grid_resolution(
            lmax_i,
            mmax_i,
            grid_resolution_list,
            method=self.grid_method,
        )
        if self.grid_method == "e3nn":
            self.phi_resolution, self.theta_resolution = self.grid_resolution_list
            self.lebedev_precision = 0
            self.lebedev_npoints = 0
        else:
            self.phi_resolution = 0
            self.theta_resolution = 0
            self.lebedev_precision, self.lebedev_npoints = self.grid_resolution_list

        super().__init__(
            lmax=lmax_i,
            mmax=mmax_i,
            precision=precision,
            n_frames=1,
            coefficient_layout=coefficient_layout,
        )

    def _rescale_truncated_orders(self, mat: np.ndarray) -> None:
        if self.lmax == self.mmax:
            return
        for degree in range(self.lmax + 1):
            if degree <= self.mmax:
                continue
            start_idx = degree * degree
            length = 2 * degree + 1
            rescale = math.sqrt(length / float(2 * self.mmax + 1))
            mat[:, :, start_idx : start_idx + length] *= rescale

    def _rescale_truncated_matrix(self, mat: np.ndarray) -> None:
        if self.lmax == self.mmax:
            return
        for degree in range(self.lmax + 1):
            if degree <= self.mmax:
                continue
            start_idx = degree * degree
            length = 2 * degree + 1
            rescale = math.sqrt(length / float(2 * self.mmax + 1))
            mat[:, start_idx : start_idx + length] *= rescale

    def _build_projection_mats(
        self,
        coeff_index: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.grid_method == "lebedev":
            return self._build_lebedev_projection_mats(coeff_index)
        return self._build_e3nn_projection_mats(coeff_index)

    def _build_e3nn_projection_mats(
        self,
        coeff_index: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Under the component normalization, the e3nn ``ToS2Grid``/``FromS2Grid``
        # product-grid buffers evaluate the real spherical harmonics on the
        # ``(beta, alpha)`` tensor-product grid: sampling
        # ``real_spherical_harmonics`` on those grid points reproduces
        # ``einsum("mbi,am->bai", ToS2Grid.shb, ToS2Grid.sha)``, and synthesis of
        # the from-grid matrix multiplies in the e3nn beta quadrature weights.
        # This keeps the e3nn and Lebedev S2 backends drop-in replacements for
        # the same grid net.
        res_beta = int(self.theta_resolution)
        res_alpha = int(self.phi_resolution)
        betas = (np.arange(res_beta, dtype=np.float64) + 0.5) / res_beta * math.pi
        alphas = np.arange(res_alpha, dtype=np.float64) / res_alpha * (2.0 * math.pi)
        beta_grid, alpha_grid = np.meshgrid(betas, alphas, indexing="ij")
        grid_points = np.stack(
            [
                np.sin(beta_grid) * np.sin(alpha_grid),
                np.cos(beta_grid),
                np.sin(beta_grid) * np.cos(alpha_grid),
            ],
            axis=-1,
        )
        harmonics = real_spherical_harmonics(grid_points, self.lmax)
        scale = math.sqrt(float(self.lmax + 1))
        degree_factors = np.asarray(
            [
                float(2 * degree + 1)
                for degree in range(self.lmax + 1)
                for _ in range(2 * degree + 1)
            ],
            dtype=np.float64,
        )
        # e3nn beta quadrature weights (``FromS2Grid._quadrature_weights``),
        # one weight per beta row, scaled by ``res_beta**2 / res_alpha``.
        half = res_beta // 2
        order = np.arange(half, dtype=np.float64)
        beta_index = np.arange(2 * half, dtype=np.float64)
        quad_inner = np.sum(
            np.sin(
                (2.0 * beta_index[:, None] + 1.0)
                * (2.0 * order[None, :] + 1.0)
                * math.pi
                / (4.0 * half)
            )
            / (2.0 * order[None, :] + 1.0),
            axis=1,
        )
        quad_weight = (
            (2.0 / half)
            * np.sin(math.pi * (2.0 * beta_index + 1.0) / (4.0 * half))
            * quad_inner
        )
        quad_weight /= 2.0 * (2 * half) ** 2
        quad_weight = quad_weight * (res_beta**2 / res_alpha)
        to_grid_mat = harmonics / scale
        from_grid_mat = harmonics * (
            quad_weight[:, None, None] * scale * degree_factors[None, None, :]
        )
        self._rescale_truncated_orders(to_grid_mat)
        self._rescale_truncated_orders(from_grid_mat)

        to_grid_mat = np.reshape(to_grid_mat, (res_beta * res_alpha, -1))[
            :, coeff_index
        ]
        from_grid_mat = np.reshape(from_grid_mat, (res_beta * res_alpha, -1)).T[
            coeff_index, :
        ]
        return to_grid_mat, from_grid_mat

    def _build_lebedev_projection_mats(
        self,
        coeff_index: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        points, weights = load_lebedev_rule(self.lebedev_precision)
        points = np.asarray(points, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        # exact numpy replacement for e3nn spherical_harmonics(
        #     list(range(lmax + 1)), points, normalize=True, normalization="norm")
        harmonics = real_spherical_harmonics(points, self.lmax)
        # Match the component-normalized product-grid convention used by
        # e3nn's ToS2Grid/FromS2Grid pair so both S2 backends are drop-in
        # replacements for the same grid net.
        scale = math.sqrt(float(self.lmax + 1))
        degree_factors = np.asarray(
            [
                float(2 * degree + 1)
                for degree in range(self.lmax + 1)
                for _ in range(2 * degree + 1)
            ],
            dtype=np.float64,
        )
        to_grid_mat = harmonics / scale
        from_grid_mat = harmonics * (weights[:, None] * scale * degree_factors[None, :])
        self._rescale_truncated_matrix(to_grid_mat)
        self._rescale_truncated_matrix(from_grid_mat)

        to_grid_mat = to_grid_mat[:, coeff_index]
        from_grid_mat = from_grid_mat[:, coeff_index].T
        return to_grid_mat, from_grid_mat

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "S2GridProjector",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "mmax": self.mmax,
                "precision": np.dtype(PRECISION_DICT[self.precision]).name,
                "grid_resolution_list": self.grid_resolution_list,
                "coefficient_layout": self.coefficient_layout,
                "grid_method": self.grid_method,
            },
            "@variables": {},
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> S2GridProjector:
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "S2GridProjector":
            raise ValueError(f"Invalid class for S2GridProjector: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        config = data.pop("config")
        data.pop("@variables", None)
        return cls(**config)


class SO3GridProjector(BaseGridProjector):
    """
    Project SO(3) coefficients to/from a Wigner-D grid with frame indices.

    The coefficient axis is packed as ``(l, m, k)`` with ordinary SeZM
    ``(l, m)`` order outside and the configured frame set inside each row.  A
    frame index outside ``[-l, l]`` is kept as a zero column/row.  This keeps the
    tensor layout regular while preserving the exact per-degree frame support.
    """

    def __init__(
        self,
        *,
        lmax: int,
        mmax: int | None = None,
        kmax: int = 1,
        precision: str = DEFAULT_PRECISION,
        lebedev_precision: int | None = None,
        coefficient_layout: str = "packed",
    ) -> None:
        lmax_i = int(lmax)
        mmax_i = int(lmax_i if mmax is None else mmax)
        self.kmax = int(kmax)
        if self.kmax < 0:
            raise ValueError("`kmax` must be non-negative")
        self.frame_set = _build_so3_frame_set(self.kmax)
        self.frame_zero_index = self.frame_set.index(0)
        self.lebedev_precision, self.lebedev_npoints, self.n_gamma = resolve_so3_grid(
            lmax_i,
            kmax=self.kmax,
            lebedev_precision=lebedev_precision,
        )
        super().__init__(
            lmax=lmax_i,
            mmax=mmax_i,
            precision=precision,
            n_frames=len(self.frame_set),
            coefficient_layout=coefficient_layout,
        )
        self.frame_values = np.asarray(self.frame_set, dtype=np.int64)

    def _build_projection_mats(
        self,
        coeff_index: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        points, weights = load_lebedev_rule(self.lebedev_precision)
        points = np.asarray(points, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        gamma = np.arange(self.n_gamma, dtype=np.float64) * (
            2.0 * math.pi / float(self.n_gamma)
        )
        edge_quaternion = build_edge_quaternion(points, eps=1e-14)
        edge_quaternion = np.repeat(edge_quaternion, self.n_gamma, axis=0)
        gamma_quaternion = np.tile(quaternion_z_rotation(gamma), (points.shape[0], 1))
        grid_quaternion = quaternion_multiply(gamma_quaternion, edge_quaternion)
        wigner_grid, _ = WignerDCalculator(self.lmax, precision="float64")(
            grid_quaternion
        )
        # ``build_edge_quaternion`` follows SeZM's global-to-local convention.
        # The transpose below stores the local m=0 column in the same layout
        # as ``WignerDCalculator.forward_zonal`` and extends it to k != 0.
        wigner_grid = np.ascontiguousarray(np.swapaxes(wigner_grid, -1, -2))
        haar_weight = np.repeat(weights, self.n_gamma) / float(self.n_gamma)

        grid_size = int(grid_quaternion.shape[0])
        coeff_dim = int(coeff_index.shape[0] * len(self.frame_set))
        to_grid_mat = np.zeros((grid_size, coeff_dim), dtype=np.float64)
        from_grid_mat = np.zeros((coeff_dim, grid_size), dtype=np.float64)

        for degree in range(self.lmax + 1):
            degree_factor = float(2 * degree + 1)
            for m_order in range(-degree, degree + 1):
                packed_idx = so3_packed_index(degree, m_order)
                coeff_positions = np.argwhere(coeff_index == packed_idx)
                if coeff_positions.size == 0:
                    continue
                coeff_pos = int(coeff_positions[0, 0])
                for frame_pos, frame_order in enumerate(self.frame_set):
                    flat_idx = coeff_pos * len(self.frame_set) + frame_pos
                    if abs(frame_order) > degree:
                        continue
                    row = so3_packed_index(degree, m_order)
                    col = so3_packed_index(degree, frame_order)
                    values = wigner_grid[:, row, col]
                    to_grid_mat[:, flat_idx] = values
                    from_grid_mat[flat_idx, :] = degree_factor * haar_weight * values
        return to_grid_mat, from_grid_mat

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "SO3GridProjector",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "mmax": self.mmax,
                "kmax": self.kmax,
                "precision": np.dtype(PRECISION_DICT[self.precision]).name,
                "lebedev_precision": self.lebedev_precision,
                "coefficient_layout": self.coefficient_layout,
            },
            "@variables": {},
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> SO3GridProjector:
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "SO3GridProjector":
            raise ValueError(f"Invalid class for SO3GridProjector: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        config = data.pop("config")
        data.pop("@variables", None)
        return cls(**config)


def resolve_s2_grid_resolution(
    lmax: int,
    mmax: int,
    *,
    method: str = "e3nn",
) -> list[int]:
    """
    Resolve the default S2 grid resolution.

    For ``method='e3nn'``, the automatic default uses even azimuthal sampling
    ``R_phi = 2 * mmax + 4`` and even polar sampling
    ``R_theta = ceil_even(3 * lmax + 2)``.

    For ``method='lebedev'``, the automatic default picks the smallest packaged
    Lebedev rule whose algebraic precision is at least ``3 * lmax`` and returns
    ``[precision, n_points]``.
    """
    method = str(method).lower()
    if method not in {"e3nn", "lebedev"}:
        raise ValueError("`method` must be either 'e3nn' or 'lebedev'")
    if method == "lebedev":
        required_precision = 3 * int(lmax)
        for precision, n_points in LEBEDEV_PRECISION_TO_NPOINTS.items():
            if precision >= required_precision:
                return [precision, n_points]
        raise ValueError(
            f"No packaged Lebedev rule has precision >= {required_precision}"
        )

    phi_resolution = 2 * int(mmax) + 4
    theta_resolution = 3 * int(lmax) + 2
    theta_resolution += theta_resolution % 2
    return [phi_resolution, theta_resolution]


def resolve_so3_grid(
    lmax: int,
    *,
    kmax: int = 1,
    lebedev_precision: int | None = None,
) -> tuple[int, int, int]:
    """
    Resolve the default SO(3) quadrature as Lebedev sphere times gamma samples.

    The Lebedev precision follows the same conservative ``3*lmax`` rule used by
    the S2 grid path.  The gamma grid is chosen for the quadratic grid products
    used by the SO(3) grid nets, whose third-angle frequency can reach
    ``k1 + k2 - kout``.
    """
    lmax_i = int(lmax)
    kmax_i = int(kmax)
    if kmax_i < 0:
        raise ValueError("`kmax` must be non-negative")
    if lebedev_precision is None:
        required_precision = 3 * lmax_i
        for precision, n_points in LEBEDEV_PRECISION_TO_NPOINTS.items():
            if precision >= required_precision:
                lebedev_precision = precision
                lebedev_npoints = n_points
                break
        else:
            raise ValueError(
                f"No packaged Lebedev rule has precision >= {required_precision}"
            )
    else:
        lebedev_precision = int(lebedev_precision)
        lebedev_npoints = LEBEDEV_PRECISION_TO_NPOINTS.get(lebedev_precision)
        if lebedev_npoints is None:
            raise ValueError(
                f"Lebedev rule with precision {lebedev_precision} is not packaged"
            )

    # A quadratic product followed by analysis can contain gamma frequencies up
    # to ``3*kmax``.  A uniform grid with more samples than that frequency
    # resolves the integer Fourier modes exactly.
    n_gamma = 1 if kmax_i == 0 else 3 * kmax_i + 1
    return int(lebedev_precision), int(lebedev_npoints), int(n_gamma)


def _normalize_s2_grid_resolution(
    lmax: int,
    mmax: int,
    grid_resolution_list: list[int] | None,
    *,
    method: str,
) -> list[int]:
    """Resolve default grids or validate already-resolved low-level grids."""
    method = str(method).lower()
    if grid_resolution_list is None:
        return resolve_s2_grid_resolution(lmax, mmax, method=method)
    if method == "lebedev":
        if len(grid_resolution_list) != 2:
            raise ValueError(
                "Lebedev `grid_resolution_list` must be [precision, n_points]"
            )
        precision = int(grid_resolution_list[0])
        n_points = int(grid_resolution_list[1])
        expected_n_points = LEBEDEV_PRECISION_TO_NPOINTS.get(precision)
        if expected_n_points != n_points:
            raise ValueError(
                "Lebedev `grid_resolution_list` must match a packaged "
                f"[precision, n_points] pair; got [{precision}, {n_points}]"
            )
        return [precision, n_points]

    if len(grid_resolution_list) != 2:
        raise ValueError("`grid_resolution_list` must contain two integers")
    resolution = [int(grid_resolution_list[0]), int(grid_resolution_list[1])]
    if resolution[0] < 1 or resolution[1] < 1:
        raise ValueError("grid resolutions must be positive")
    return resolution


def _build_so3_frame_set(kmax: int) -> list[int]:
    """Build the symmetric frame-index set with zero first."""
    kmax_i = int(kmax)
    if kmax_i < 0:
        raise ValueError("`kmax` must be non-negative")
    return [0, *[frame for kk in range(1, kmax_i + 1) for frame in (-kk, kk)]]
