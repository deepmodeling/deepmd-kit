# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Grid projection helpers for SeZM function-space nonlinearities.

The projectors in this module only handle basis transforms.  They do not apply
channel mixing or nonlinearities.  A projector maps coefficient tensors to a
fixed quadrature grid, and maps grid fields back to coefficients with the
matching quadrature rule.
"""

from __future__ import (
    annotations,
)

import math
from typing import (
    Any,
)

import torch
import torch.nn as nn
from e3nn.o3 import (
    FromS2Grid,
    ToS2Grid,
    spherical_harmonics,
)

from deepmd.pt.utils.env import (
    PRECISION_DICT,
    RESERVED_PRECISION_DICT,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .indexing import (
    build_l_major_index,
    build_m_major_index,
    so3_packed_index,
)
from .lebedev import (
    LEBEDEV_PRECISION_TO_NPOINTS,
    load_lebedev_rule,
)
from .wignerd import (
    WignerDCalculator,
    build_edge_quaternion,
    quaternion_multiply,
    quaternion_z_rotation,
)


class BaseGridProjector(nn.Module):
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
        dtype: torch.dtype,
        n_frames: int,
        coefficient_layout: str,
    ) -> None:
        super().__init__()
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
        self.dtype = dtype
        self.device = torch.device("cpu")
        self.precision = RESERVED_PRECISION_DICT[dtype]
        self.n_frames = int(n_frames)
        self.packed_dim = int((self.lmax + 1) ** 2)

        coeff_index = self._build_coefficient_index(device=torch.device("cpu"))
        to_grid_mat, from_grid_mat = self._build_projection_mats(coeff_index)
        self.coeff_dim = int(to_grid_mat.shape[1])
        self.grid_size = int(to_grid_mat.shape[0])
        if self.coeff_dim != int(from_grid_mat.shape[0]):
            raise ValueError("Projection matrix coefficient axes `J` do not match")
        if self.grid_size != int(from_grid_mat.shape[1]):
            raise ValueError("Projection matrix grid axes `G` do not match")
        self.register_buffer(
            "to_grid_mat",
            to_grid_mat.to(device=self.device, dtype=self.dtype),
            persistent=False,
        )
        self.register_buffer(
            "from_grid_mat",
            from_grid_mat.to(device=self.device, dtype=self.dtype),
            persistent=False,
        )

    def to_grid(self, embedding: torch.Tensor) -> torch.Tensor:
        """Project flattened coefficients ``(N, J, C)`` to grid fields ``(N, G, C)``."""
        to_grid_mat = self.to_grid_mat.to(
            device=embedding.device,
            dtype=embedding.dtype,
        )
        return torch.einsum("gj,njc->ngc", to_grid_mat, embedding)

    def from_grid(self, grid: torch.Tensor) -> torch.Tensor:
        """Project grid fields ``(N, G, C)`` back to flattened coefficients ``(N, J, C)``."""
        from_grid_mat = self.from_grid_mat.to(
            device=grid.device,
            dtype=grid.dtype,
        )
        return torch.einsum("jg,ngc->njc", from_grid_mat, grid)

    def _build_coefficient_index(self, device: torch.device) -> torch.Tensor:
        """Build the coefficient subset consumed by the projector matrices."""
        if self.coefficient_layout == "m_major":
            return build_m_major_index(self.lmax, self.mmax, device=device)
        if self.mmax == self.lmax:
            return torch.arange((self.lmax + 1) ** 2, device=device, dtype=torch.long)
        return build_l_major_index(self.lmax, self.mmax, device=device)

    def _build_projection_mats(
        self,
        coeff_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
    dtype
        Buffer dtype used by the projection matrices.
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
        dtype: torch.dtype,
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
            dtype=dtype,
            n_frames=1,
            coefficient_layout=coefficient_layout,
        )

    def _rescale_truncated_orders(self, mat: torch.Tensor) -> None:
        if self.lmax == self.mmax:
            return
        for degree in range(self.lmax + 1):
            if degree <= self.mmax:
                continue
            start_idx = degree * degree
            length = 2 * degree + 1
            rescale = math.sqrt(length / float(2 * self.mmax + 1))
            mat[:, :, start_idx : start_idx + length].mul_(rescale)

    def _rescale_truncated_matrix(self, mat: torch.Tensor) -> None:
        if self.lmax == self.mmax:
            return
        for degree in range(self.lmax + 1):
            if degree <= self.mmax:
                continue
            start_idx = degree * degree
            length = 2 * degree + 1
            rescale = math.sqrt(length / float(2 * self.mmax + 1))
            mat[:, start_idx : start_idx + length].mul_(rescale)

    def _build_projection_mats(
        self,
        coeff_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.grid_method == "lebedev":
            return self._build_lebedev_projection_mats(coeff_index)
        return self._build_e3nn_projection_mats(coeff_index)

    def _build_e3nn_projection_mats(
        self,
        coeff_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.device("cpu"):
            to_grid = ToS2Grid(
                self.lmax,
                (self.theta_resolution, self.phi_resolution),
                normalization="component",
                device="cpu",
            )
            to_grid_mat = torch.einsum("mbi,am->bai", to_grid.shb, to_grid.sha).detach()
            self._rescale_truncated_orders(to_grid_mat)

            from_grid = FromS2Grid(
                (self.theta_resolution, self.phi_resolution),
                self.lmax,
                normalization="component",
                device="cpu",
            )
            from_grid_mat = torch.einsum(
                "am,mbi->bai", from_grid.sha, from_grid.shb
            ).detach()
            self._rescale_truncated_orders(from_grid_mat)

        to_grid_mat = to_grid_mat.flatten(0, 1).index_select(1, coeff_index)
        from_grid_mat = (
            from_grid_mat.flatten(0, 1).permute(1, 0).index_select(0, coeff_index)
        )
        return to_grid_mat, from_grid_mat

    def _build_lebedev_projection_mats(
        self,
        coeff_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.device("cpu"):
            points, weights = load_lebedev_rule(
                self.lebedev_precision,
                dtype=torch.float64,
                device=torch.device("cpu"),
            )
            harmonics = spherical_harmonics(
                list(range(self.lmax + 1)),
                points,
                normalize=True,
                normalization="norm",
            )
            # Match the component-normalized product-grid convention used by
            # e3nn's ToS2Grid/FromS2Grid pair so both S2 backends are drop-in
            # replacements for the same grid net.
            scale = math.sqrt(float(self.lmax + 1))
            degree_factors = harmonics.new_tensor(
                [
                    float(2 * degree + 1)
                    for degree in range(self.lmax + 1)
                    for _ in range(2 * degree + 1)
                ]
            )
            to_grid_mat = harmonics / scale
            from_grid_mat = harmonics * (
                weights[:, None] * scale * degree_factors[None, :]
            )
            self._rescale_truncated_matrix(to_grid_mat)
            self._rescale_truncated_matrix(from_grid_mat)

        to_grid_mat = to_grid_mat.index_select(1, coeff_index)
        from_grid_mat = from_grid_mat.index_select(1, coeff_index).transpose(0, 1)
        return to_grid_mat, from_grid_mat

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "S2GridProjector",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "mmax": self.mmax,
                "precision": RESERVED_PRECISION_DICT[self.dtype],
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
        precision = config.pop("precision")
        config["dtype"] = PRECISION_DICT[precision]
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
        dtype: torch.dtype,
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
            dtype=dtype,
            n_frames=len(self.frame_set),
            coefficient_layout=coefficient_layout,
        )
        self.register_buffer(
            "frame_values",
            torch.tensor(self.frame_set, dtype=torch.long, device=self.device),
            persistent=False,
        )

    def _build_projection_mats(
        self,
        coeff_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.device("cpu"):
            points, weights = load_lebedev_rule(
                self.lebedev_precision,
                dtype=torch.float64,
                device=torch.device("cpu"),
            )
            gamma = torch.arange(
                self.n_gamma, dtype=torch.float64, device=points.device
            ) * (2.0 * math.pi / float(self.n_gamma))
            edge_quaternion = build_edge_quaternion(points, eps=1e-14)
            edge_quaternion = edge_quaternion.repeat_interleave(self.n_gamma, dim=0)
            gamma_quaternion = quaternion_z_rotation(gamma).repeat(points.shape[0], 1)
            grid_quaternion = quaternion_multiply(gamma_quaternion, edge_quaternion)
            wigner_grid, _ = WignerDCalculator(self.lmax, dtype=torch.float64).to(
                torch.device("cpu")
            )(grid_quaternion)
            # ``build_edge_quaternion`` follows SeZM's global-to-local convention.
            # The transpose below stores the local m=0 column in the same layout
            # as ``WignerDCalculator.forward_zonal`` and extends it to k != 0.
            wigner_grid = wigner_grid.transpose(-1, -2).contiguous()
            haar_weight = weights.repeat_interleave(self.n_gamma) / float(self.n_gamma)

            grid_size = int(grid_quaternion.shape[0])
            coeff_dim = int(coeff_index.numel() * len(self.frame_set))
            to_grid_mat = torch.zeros(
                grid_size,
                coeff_dim,
                dtype=torch.float64,
                device=points.device,
            )
            from_grid_mat = torch.zeros(
                coeff_dim,
                grid_size,
                dtype=torch.float64,
                device=points.device,
            )

            for degree in range(self.lmax + 1):
                degree_factor = float(2 * degree + 1)
                for m_order in range(-degree, degree + 1):
                    packed_idx = so3_packed_index(degree, m_order)
                    coeff_positions = (coeff_index == packed_idx).nonzero(
                        as_tuple=False
                    )
                    if coeff_positions.numel() == 0:
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
                        from_grid_mat[flat_idx, :] = (
                            degree_factor * haar_weight * values
                        )
        return to_grid_mat, from_grid_mat

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "SO3GridProjector",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "mmax": self.mmax,
                "kmax": self.kmax,
                "precision": RESERVED_PRECISION_DICT[self.dtype],
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
        precision = config.pop("precision")
        config["dtype"] = PRECISION_DICT[precision]
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
