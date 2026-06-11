# SPDX-License-Identifier: LGPL-3.0-or-later
"""
S2 grid projection helpers for DPA4/SeZM function-space nonlinearities.

This module is the dpmodel port of
``deepmd.pt.model.descriptor.sezm_nn.projection``, restricted to the Lebedev
S2 quadrature path used by the core DPA4 configuration
(``lebedev_quadrature=True``). The projectors only handle basis transforms:
a projector maps coefficient tensors to a fixed quadrature grid, and maps
grid fields back to coefficients with the matching quadrature rule.

Ported names: ``BaseGridProjector``, ``S2GridProjector`` (Lebedev branch),
``resolve_s2_grid_resolution`` (as-is, both methods — pure arithmetic), and
``_normalize_s2_grid_resolution``.

Skipped names (SO(3) Wigner-D grid machinery; consumed only by
``SO3GridNet`` in pt ``grid_net.py``, which backs the ``node_wise_so3``,
``message_node_so3``, and ``ffn_so3_grid`` paths — all disabled in the core
DPA4 config): ``SO3GridProjector``, ``resolve_so3_grid``,
``_build_so3_frame_set``.

Not ported (guarded): the e3nn product-grid branch of ``S2GridProjector``
(``grid_method="e3nn"``, i.e. ``lebedev_quadrature=False``) raises
``NotImplementedError`` at construction. Only the Lebedev path reproduces
to-grid/from-grid roundtrip identities at machine precision.

The Lebedev projection matrices are assembled at init time with pure numpy:
``load_lebedev_rule`` replaces the pt Lebedev loader (same packaged data) and
``real_spherical_harmonics`` exactly replaces the e3nn call
``spherical_harmonics(list(range(lmax+1)), points, normalize=True,
normalization="norm")``, so the buffers match the pt float64 buffers to
machine precision.
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
)


class BaseGridProjector(NativeOP):
    """
    Base class for fixed coefficient-to-grid projection matrices.

    Subclasses build ``to_grid_mat`` with shape ``(G, J)`` and
    ``from_grid_mat`` with shape ``(J, G)``, where ``G`` is the number of grid
    samples and ``J`` is the flattened coefficient axis consumed by the grid
    net. For ordinary S2 projections, ``J`` is the SO(3) feature coefficient
    axis: ``D = (lmax + 1)^2`` in packed layout, or the retained ``D_m`` axis
    in m-major layout.
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
        self.to_grid_mat = np.ascontiguousarray(to_grid_mat).astype(prec)
        self.from_grid_mat = np.ascontiguousarray(from_grid_mat).astype(prec)

    def call(self, *args: Any, **kwargs: Any) -> Any:
        """Projectors expose ``to_grid``/``from_grid``; there is no forward."""
        raise NotImplementedError(
            "BaseGridProjector has no forward; use `to_grid` or `from_grid`"
        )

    def to_grid(self, embedding: Any) -> Any:
        """Project flattened coefficients ``(N, J, C)`` to grid fields ``(N, G, C)``."""
        xp = array_api_compat.array_namespace(embedding)
        to_grid_mat = xp.asarray(
            self.to_grid_mat[...], device=array_api_compat.device(embedding)
        )
        if to_grid_mat.dtype != embedding.dtype:
            to_grid_mat = xp.astype(to_grid_mat, embedding.dtype)
        # einsum "gj,njc->ngc" as a broadcast batched matmul
        return xp.matmul(to_grid_mat[None, ...], embedding)

    def from_grid(self, grid: Any) -> Any:
        """Project grid fields ``(N, G, C)`` back to flattened coefficients ``(N, J, C)``."""
        xp = array_api_compat.array_namespace(grid)
        from_grid_mat = xp.asarray(
            self.from_grid_mat[...], device=array_api_compat.device(grid)
        )
        if from_grid_mat.dtype != grid.dtype:
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
    Project SO(3) coefficients to/from a flattened S2 grid (Lebedev only).

    Parameters
    ----------
    lmax
        Maximum spherical harmonic degree.
    mmax
        Maximum order kept in the coefficient layout. If None, use ``lmax``.
    precision
        Buffer precision used by the projection matrices.
    grid_resolution_list
        Two-element resolution list ``[precision, n_points]`` for
        ``grid_method='lebedev'``. If None, resolved automatically.
    coefficient_layout
        Coefficient ordering expected by the caller:
        - ``"packed"``: packed ``(l, m)`` order, optionally truncated by ``mmax``.
        - ``"m_major"``: reduced m-major order used inside ``SO2Convolution``.
    grid_method
        S2 quadrature backend. Must be ``"e3nn"`` or ``"lebedev"``; only
        ``"lebedev"`` (``lebedev_quadrature=True``) is ported to dpmodel.
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
        if self.grid_method == "e3nn":
            raise NotImplementedError(
                "grid_method='e3nn' (lebedev_quadrature=False) is not ported "
                "to dpmodel; use lebedev_quadrature=True"
            )

        self.grid_resolution_list = _normalize_s2_grid_resolution(
            lmax_i,
            mmax_i,
            grid_resolution_list,
            method=self.grid_method,
        )
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
        """Serialize the S2GridProjector to a dict (pt-compatible format)."""
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
        """Deserialize an S2GridProjector from a dict."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "S2GridProjector":
            raise ValueError(f"Invalid class for S2GridProjector: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        config = data.pop("config")
        data.pop("@variables", None)
        return cls(
            lmax=int(config["lmax"]),
            mmax=int(config["mmax"]),
            precision=str(config["precision"]),
            grid_resolution_list=config["grid_resolution_list"],
            coefficient_layout=str(config["coefficient_layout"]),
            grid_method=str(config["grid_method"]),
        )


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
