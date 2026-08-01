# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Quaternion-based Wigner-D and edge-frame utilities for DPA4/SeZM.

This module defines the quaternion helpers and Wigner-D evaluator used to
construct edge-aligned SO(3) rotation blocks in SeZM.

This module is the dpmodel (array-API) port of
``deepmd.pt.model.descriptor.sezm_nn.wignerd``.
"""

from __future__ import (
    annotations,
)

import math
from itertools import (
    permutations,
)
from typing import (
    Any,
    ClassVar,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel import (
    DEFAULT_PRECISION,
    NativeOP,
)
from deepmd.dpmodel.array_api import (
    xp_asarray_nodetach,
    xp_take_along_axis,
)
from deepmd.dpmodel.common import (
    get_xp_precision,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


class CaseCoefficients:
    """
    Polynomial tables for one magnitude-ordered branch of the quaternion Wigner path.

    The generic Wigner-D evaluation factors each matrix element into:
    - a phase term carried by the arguments of ``Ra`` and ``Rb``;
    - a real magnitude term evaluated by Horner recursion.

    The magnitude formula has two numerically stable branches, depending on whether
    ``|Ra| >= |Rb|`` or the opposite. Each branch stores the branch-specific Horner
    coefficients and the powers of ``|Ra|`` / ``|Rb|`` that sit outside the Horner
    polynomial.
    """

    def __init__(
        self,
        *,
        coeff: np.ndarray,
        horner: np.ndarray,
        poly_len: np.ndarray,
        ra_exp: np.ndarray,
        rb_exp: np.ndarray,
        sign: np.ndarray,
    ) -> None:
        self.coeff = coeff
        self.horner = horner
        self.poly_len = poly_len
        self.ra_exp = ra_exp
        self.rb_exp = rb_exp
        self.sign = sign


class WignerPolynomialCoefficients:
    """
    Precomputed coefficient tables for the generic quaternion Wigner evaluator.

    Only one half of each real block is stored explicitly. The remaining entries are
    reconstructed from the exact symmetry

    ``D^l_{-m',-m} = (-1)^(m' - m) * conj(D^l_{m',m})``.

    This keeps the runtime path branch-free with respect to ``(l, m', m)`` while
    preserving the exact packed ``(l, m)`` layout used everywhere else in SeZM.
    """

    def __init__(
        self,
        *,
        lmin: int,
        lmax: int,
        size: int,
        max_poly_len: int,
        n_primary: int,
        n_derived: int,
        primary_row: np.ndarray,
        primary_col: np.ndarray,
        case1: CaseCoefficients,
        case2: CaseCoefficients,
        mp_plus_m: np.ndarray,
        m_minus_mp: np.ndarray,
        diagonal_mask: np.ndarray,
        anti_diagonal_mask: np.ndarray,
        special_2m: np.ndarray,
        anti_diag_sign: np.ndarray,
        derived_row: np.ndarray,
        derived_col: np.ndarray,
        derived_primary_idx: np.ndarray,
        derived_sign: np.ndarray,
    ) -> None:
        self.lmin = int(lmin)
        self.lmax = int(lmax)
        self.size = int(size)
        self.max_poly_len = int(max_poly_len)
        self.n_primary = int(n_primary)
        self.n_derived = int(n_derived)

        self.primary_row = primary_row
        self.primary_col = primary_col
        self.case1 = case1
        self.case2 = case2
        self.mp_plus_m = mp_plus_m
        self.m_minus_mp = m_minus_mp
        self.diagonal_mask = diagonal_mask
        self.anti_diagonal_mask = anti_diagonal_mask
        self.special_2m = special_2m
        self.anti_diag_sign = anti_diag_sign
        self.derived_row = derived_row
        self.derived_col = derived_col
        self.derived_primary_idx = derived_primary_idx
        self.derived_sign = derived_sign


class WignerSmallOrderCoefficients:
    """
    Precomputed low-order quaternion polynomial kernels in the SeZM packed basis.

    Only kernels required by the owning ``WignerDCalculator`` are registered:

    - ``C_l2`` stores the degree-4 tensor-contraction coefficients.
    - ``C_l3`` .. ``C_l10`` store flattened monomial coefficient matrices.
    - ``C_combined_l3l4`` lifts the ``l=3`` basis to degree 8 and stacks it with
      ``l=4`` so both blocks can be produced by one matrix multiply.
    - ``C_combined_l5l6`` applies the same degree-12 stacking for ``l=5,6``.
    - ``C_combined_l7l8`` applies the same degree-16 stacking for ``l=7,8``.
    - ``C_combined_l9l10`` applies the same degree-20 stacking for ``l=9,10``.
    - ``exp_l3`` .. ``exp_l10`` store the monomial exponent tables used by the
      runtime gather/prod path.
    """

    _EXTRA_KERNELS_BY_LMAX: ClassVar[tuple[tuple[int, tuple[str, ...]], ...]] = (
        (3, ("C_l3", "exp_l3")),
        (4, ("C_l4", "C_combined_l3l4", "exp_l4")),
        (5, ("C_l5", "exp_l5")),
        (6, ("C_l6", "C_combined_l5l6", "exp_l6")),
        (7, ("C_l7", "exp_l7")),
        (8, ("C_l8", "C_combined_l7l8", "exp_l8")),
        (9, ("C_l9", "exp_l9")),
        (10, ("C_l10", "C_combined_l9l10", "exp_l10")),
    )

    def __init__(
        self,
        *,
        lmax: int,
        kernels: dict[str, np.ndarray],
    ) -> None:
        for name in self.required_kernel_names(lmax):
            setattr(self, name, kernels[name])

    @classmethod
    def required_kernel_names(cls, lmax: int) -> tuple[str, ...]:
        """Return low-order kernel names required for ``lmax``."""
        names = ["C_l2"]
        for threshold, extra_names in cls._EXTRA_KERNELS_BY_LMAX:
            if lmax >= threshold:
                names.extend(extra_names)
        return tuple(names)


def _safe_norm_nd(x: Any, eps: float = 1e-7) -> Any:
    """Compute an ``L2`` norm with smooth epsilon regularization."""
    xp = array_api_compat.array_namespace(x)
    in_dtype = x.dtype
    # ``str(dtype)`` matches both "float16" and "bfloat16" across namespaces.
    promote = "float16" in str(in_dtype)
    if promote:
        x = xp.astype(x, xp.float32)
    norm = xp.sqrt(xp.sum(x * x, axis=-1, keepdims=True) + eps * eps)
    if promote:
        norm = xp.astype(norm, in_dtype)
    return norm


def quaternion_normalize(q: Any, eps: float = 1e-7) -> Any:
    """Normalize quaternions with a differentiable epsilon floor."""
    return q / _safe_norm_nd(q, eps)


def quaternion_multiply(q1: Any, q2: Any) -> Any:
    """Hamilton product for batched quaternions in ``(w, x, y, z)`` order."""
    xp = array_api_compat.array_namespace(q1, q2)
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return xp.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quaternion_to_rotation_matrix(q: Any) -> Any:
    """
    Convert unit quaternions to 3x3 rotation matrices.

    The returned matrix is the active rotation represented by ``q``. In SeZM this is
    the global->local edge rotation, so multiplying the edge direction by this matrix
    sends it to local ``+Z``.
    """
    xp = array_api_compat.array_namespace(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    x2 = x * x
    y2 = y * y
    z2 = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return xp.stack(
        [
            xp.stack(
                [1.0 - 2.0 * (y2 + z2), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                axis=-1,
            ),
            xp.stack(
                [2.0 * (xy + wz), 1.0 - 2.0 * (x2 + z2), 2.0 * (yz - wx)],
                axis=-1,
            ),
            xp.stack(
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (x2 + y2)],
                axis=-1,
            ),
        ],
        axis=-2,
    )


def quaternion_z_rotation(gamma: Any) -> Any:
    """
    Create quaternions for a rotation about the local ``+Z`` axis.

    Parameters
    ----------
    gamma
        Roll angles in radians with shape ``(E,)``.

    Returns
    -------
    Array
        Quaternions with shape ``(E, 4)`` in ``(w, x, y, z)`` order.
    """
    xp = array_api_compat.array_namespace(gamma)
    half_gamma = 0.5 * gamma
    w = xp.cos(half_gamma)
    x = xp.zeros_like(gamma)
    y = xp.zeros_like(gamma)
    z = xp.sin(half_gamma)
    return xp.stack([w, x, y, z], axis=-1)


def _smooth_step_cinf(x: Any) -> Any:
    """
    Smooth ``C^inf`` step on ``[0, 1]``.

    This function equals exactly 0 and 1 at the endpoints, and transitions with all
    derivatives vanishing there. It is used only to blend the two valid quaternion
    charts; the geometric constraint itself is still enforced by the charts.
    """
    xp = array_api_compat.array_namespace(x)
    x_clamped = xp.clip(x, min=0.0, max=1.0)
    eps = float(xp.finfo(x_clamped.dtype).eps)
    left = xp.exp(-1.0 / xp.clip(x_clamped, min=eps))
    right = xp.exp(-1.0 / xp.clip(1.0 - x_clamped, min=eps))
    interior = left / (left + right)
    return xp.where(
        x_clamped <= 0.0,
        xp.zeros_like(x_clamped),
        xp.where(x_clamped >= 1.0, xp.ones_like(x_clamped), interior),
    )


def quaternion_nlerp(
    q0: Any,
    q1: Any,
    weight: Any,
    *,
    eps: float = 1e-7,
) -> Any:
    """
    Normalized linear interpolation on the shortest quaternion arc.

    ``q`` and ``-q`` represent the same spatial rotation. Aligning signs before the
    interpolation guarantees that the blended chart stays on the shorter great-circle
    segment in ``S^3``.
    """
    xp = array_api_compat.array_namespace(q0, q1, weight)
    dot = xp.sum(q0 * q1, axis=-1, keepdims=True)
    q1_aligned = xp.where(dot < 0.0, -q1, q1)
    blended = (1.0 - weight[..., None]) * q0 + weight[..., None] * q1_aligned
    return quaternion_normalize(blended, eps)


def _build_edge_quaternion_chart_pos_z(
    edge_unit: Any,
    eps: float,
) -> Any:
    """Quaternion chart that is exact away from the ``-Z`` pole."""
    xp = array_api_compat.array_namespace(edge_unit)
    x = edge_unit[..., 0]
    y = edge_unit[..., 1]
    z = edge_unit[..., 2]
    q = xp.stack([1.0 + z, y, -x, xp.zeros_like(x)], axis=-1)
    return quaternion_normalize(q, eps)


def _build_edge_quaternion_chart_neg_z(
    edge_unit: Any,
    eps: float,
) -> Any:
    """Quaternion chart that is exact away from the ``+Z`` pole."""
    xp = array_api_compat.array_namespace(edge_unit)
    x = edge_unit[..., 0]
    y = edge_unit[..., 1]
    z = edge_unit[..., 2]
    q = xp.stack([-x, xp.zeros_like(x), 1.0 - z, y], axis=-1)
    return quaternion_normalize(q, eps)


def build_edge_quaternion(
    edge_vec: Any,
    *,
    edge_len: Any = None,
    eps: float = 1e-7,
) -> Any:
    """
    Build stable edge quaternions for the SeZM local ``+Z`` convention.

    The returned quaternion represents the global->local edge rotation, so applying its
    rotation matrix to the unit edge direction yields exactly ``(0, 0, 1)``. Two exact
    quaternion charts are used:

    - a ``+Z`` chart that is regular everywhere except the antipodal ``-Z`` pole;
    - a ``-Z`` chart that is regular everywhere except the antipodal ``+Z`` pole.

    Both charts encode the same edge-aligned local frame. A smooth ``C^inf`` blend in
    the overlap region removes the hard pole switch while keeping the represented
    rotation on the correct quaternion branch.

    Parameters
    ----------
    edge_vec
        Edge vectors with shape ``(E, 3)``.
    edge_len
        Optional edge lengths with shape ``(E, 1)``. When omitted, lengths are
        recomputed from ``edge_vec``.
    eps
        Numerical floor used in vector and quaternion normalization.

    Returns
    -------
    Array
        Unit quaternions with shape ``(E, 4)`` in ``(w, x, y, z)`` order.
    """
    xp = array_api_compat.array_namespace(edge_vec)
    if edge_len is None:
        edge_len = _safe_norm_nd(edge_vec, eps)
    else:
        edge_len = xp.sqrt(edge_len * edge_len + eps * eps)
    edge_unit = edge_vec / edge_len
    q_pos = _build_edge_quaternion_chart_pos_z(edge_unit, eps)
    q_neg = _build_edge_quaternion_chart_neg_z(edge_unit, eps)
    blend = _smooth_step_cinf(0.5 * (edge_unit[..., 2] + 1.0))
    return quaternion_nlerp(q_neg, q_pos, blend, eps=eps)


class WignerDCalculator(NativeOP):
    """
    Quaternion-driven Wigner-D blocks for the SeZM packed real spherical basis.

    Input quaternions represent the global->local edge rotation that sends the edge
    direction to local ``+Z``. The returned block-diagonal matrix keeps the packed
    SeZM real spherical-harmonics layout, so downstream code continues to consume
    ``D_full`` and ``Dt_full`` directly.

    Runtime structure:
    - ``l=0``: scalar identity block;
    - ``l=1``: direct quaternion -> Cartesian rotation -> real l=1 block;
    - ``l=2``: dedicated degree-4 quaternion tensor contraction;
    - ``l=3,4``: dedicated quaternion monomial kernels;
    - ``l=5,6``: dedicated quaternion monomial kernels;
    - ``l=7,8``: dedicated quaternion monomial kernels;
    - ``l=9,10``: dedicated quaternion monomial kernels;
    - ``l>=11``: generic quaternion polynomial path with precomputed coefficient tables.
    """

    _SMALL_ORDER_CACHE_CPU_FP64: ClassVar[dict[str, np.ndarray] | None] = None

    def __init__(
        self,
        lmax: int,
        *,
        eps: float = 1e-7,
        precision: str = DEFAULT_PRECISION,
    ) -> None:
        self.lmax = int(lmax)
        if self.lmax < 0:
            raise ValueError("`lmax` must be non-negative")
        self.precision = precision
        self.eps = float(eps)
        self.dim_full = (self.lmax + 1) ** 2
        self.poly_lmin = 11
        self.poly_offset = self.poly_lmin * self.poly_lmin

        self.l1_perm = np.array([1, 2, 0], dtype=np.int64)
        l1_sign = np.array([-1.0, -1.0, 1.0], dtype=np.float64)
        self.l1_sign_outer = np.outer(l1_sign, l1_sign)

        if self.lmax >= 2:
            self.small_order_kernels = self._build_small_order_kernels(lmax=self.lmax)

        if self.lmax >= self.poly_lmin:
            self.poly_coeffs = self._precompute_wigner_coefficients(
                self.lmax,
                lmin=self.poly_lmin,
            )
            blocks = self._precompute_real_basis_blocks(
                lmin=self.poly_lmin,
                lmax=self.lmax,
            )
            U_re, U_im, U_re_t, U_im_t = self._assemble_block_diagonal_real_basis(
                blocks
            )
            self.poly_u_re = U_re
            self.poly_u_im = U_im
            self.poly_u_re_t = U_re_t
            self.poly_u_im_t = U_im_t

        # Functional block-diagonal assembly: precompute a gather index mapping each
        # flat ``(row, col)`` of ``D_full`` to its slot in the concatenated per-degree
        # block values (the trailing slot holds the off-block zero). Degree ``l``
        # occupies rows/cols ``[l**2, (l+1)**2)`` regardless of which kernel produced
        # it, so the index is fully determined by ``lmax``. This replaces pt's
        # in-place block assignment with a path safe for torch.export.
        dim = self.dim_full
        small_lmax = min(self.lmax, 10)
        segments = [(ell * ell, 2 * ell + 1) for ell in range(small_lmax + 1)]
        if self.lmax >= self.poly_lmin:
            segments.append((self.poly_offset, dim - self.poly_offset))
        n_values = sum(block_dim * block_dim for _, block_dim in segments)
        full_idx = np.full(dim * dim, n_values, dtype=np.int64)
        base = 0
        for offset, block_dim in segments:
            local = np.arange(block_dim, dtype=np.int64)
            rows = offset + local[:, None]
            cols = offset + local[None, :]
            src = base + (local[:, None] * block_dim + local[None, :])
            full_idx[(rows * dim + cols).reshape(-1)] = src.reshape(-1)
            base += block_dim * block_dim
        self.full_gather_idx = full_idx

    def call(self, edge_quaternion: Any) -> tuple[Any, Any]:
        """
        Build packed block-diagonal Wigner-D matrices from edge quaternions.

        Parameters
        ----------
        edge_quaternion
            Unit quaternions with shape ``(E, 4)`` representing the global->local
            edge rotation.

        Returns
        -------
        tuple[Array, Array]
            ``(D_full, Dt_full)`` with shape ``(E, (lmax+1)^2, (lmax+1)^2)``.
        """
        xp = array_api_compat.array_namespace(edge_quaternion)
        dtype = get_xp_precision(xp, self.precision)
        device = array_api_compat.device(edge_quaternion)
        edge_quaternion = quaternion_normalize(
            xp.astype(edge_quaternion, dtype),
            eps=self.eps,
        )
        n_edge = edge_quaternion.shape[0]

        blocks = [xp.ones((n_edge, 1, 1), dtype=dtype, device=device)]
        if self.lmax >= 1:
            blocks.append(self._compute_l1_block(edge_quaternion))

        if self.lmax >= 2:
            blocks.append(self._compute_l2_block(edge_quaternion))

        if self.lmax >= 3:
            if self.lmax >= 4:
                D_l3, D_l4 = self._compute_l3l4_blocks(edge_quaternion)
                blocks.append(D_l3)
                blocks.append(D_l4)
            else:
                blocks.append(self._compute_l3_block(edge_quaternion))

        if self.lmax >= 5:
            if self.lmax >= 6:
                D_l5, D_l6 = self._compute_l5l6_blocks(edge_quaternion)
                blocks.append(D_l5)
                blocks.append(D_l6)
            else:
                blocks.append(self._compute_l5_block(edge_quaternion))

        if self.lmax >= 7:
            if self.lmax >= 8:
                D_l7, D_l8 = self._compute_l7l8_blocks(edge_quaternion)
                blocks.append(D_l7)
                blocks.append(D_l8)
            else:
                blocks.append(self._compute_l7_block(edge_quaternion))

        if self.lmax >= 9:
            if self.lmax >= 10:
                D_l9, D_l10 = self._compute_l9l10_blocks(edge_quaternion)
                blocks.append(D_l9)
                blocks.append(D_l10)
            else:
                blocks.append(self._compute_l9_block(edge_quaternion))

        if self.lmax >= self.poly_lmin:
            ra_re, ra_im, rb_re, rb_im = self._quaternion_to_ra_rb_real(edge_quaternion)
            D_re, D_im = self._wigner_d_matrix_realpair(
                ra_re,
                ra_im,
                rb_re,
                rb_im,
                self.poly_coeffs,
                dtype=dtype,
            )
            D_poly = self._wigner_d_pair_to_real(
                D_re,
                D_im,
                (
                    self.poly_u_re,
                    self.poly_u_im,
                    self.poly_u_re_t,
                    self.poly_u_im_t,
                ),
                lmax=self.lmax,
                lmin=self.poly_lmin,
            )
            blocks.append(D_poly)

        # Gather the per-degree blocks into the dense block-diagonal layout.
        values = xp.concat(
            [xp.reshape(b, (n_edge, b.shape[-1] * b.shape[-1])) for b in blocks]
            + [xp.zeros((n_edge, 1), dtype=dtype, device=device)],
            axis=1,
        )
        idx = xp_asarray_nodetach(xp, self.full_gather_idx, device=device)
        D_full = xp.reshape(
            xp.take(values, idx, axis=1),
            (n_edge, self.dim_full, self.dim_full),
        )
        Dt_full = xp.matrix_transpose(D_full)
        return D_full, Dt_full

    def forward_zonal(
        self,
        edge_quaternion: Any,
        lmin: int = 1,
    ) -> Any:
        """
        Build local ``m=0`` to global coupling for GIE.

        The returned layout matches the packed node rows for degrees
        ``lmin..lmax``: each degree contributes ``2l+1`` values in packed
        ``m=-l..l`` order. These values are equivalent to gathering
        ``Dt_full[:, row(l, m), col(l, 0)]`` from :meth:`call` over the
        same degree range.

        Parameters
        ----------
        edge_quaternion
            Unit quaternions with shape ``(E, 4)`` representing the global->local
            edge rotation.
        lmin
            First degree to return.

        Returns
        -------
        Array
            Zonal coupling with shape
            ``(E, (lmax + 1) ** 2 - lmin ** 2)``.
        """
        lmin = int(lmin)
        if lmin < 1:
            raise ValueError("`lmin` must be >= 1")
        xp = array_api_compat.array_namespace(edge_quaternion)
        dtype = get_xp_precision(xp, self.precision)
        device = array_api_compat.device(edge_quaternion)
        n_edge = edge_quaternion.shape[0]
        if self.lmax < lmin:
            return xp.zeros((n_edge, 0), dtype=dtype, device=device)
        edge_quaternion = quaternion_normalize(
            xp.astype(edge_quaternion, dtype),
            eps=self.eps,
        )

        zonal_blocks: list[Any] = []
        if lmin <= 1 <= self.lmax:
            zonal_blocks.append(self._compute_l1_block(edge_quaternion)[:, 1, :])

        if lmin <= 2 <= self.lmax:
            zonal_blocks.append(self._compute_l2_block(edge_quaternion)[:, 2, :])

        if self.lmax >= 3 and lmin <= 4:
            if self.lmax >= 4:
                D_l3, D_l4 = self._compute_l3l4_blocks(edge_quaternion)
                if lmin <= 3:
                    zonal_blocks.append(D_l3[:, 3, :])
                zonal_blocks.append(D_l4[:, 4, :])
            else:
                zonal_blocks.append(self._compute_l3_block(edge_quaternion)[:, 3, :])

        if self.lmax >= 5 and lmin <= 6:
            if self.lmax >= 6:
                D_l5, D_l6 = self._compute_l5l6_blocks(edge_quaternion)
                if lmin <= 5:
                    zonal_blocks.append(D_l5[:, 5, :])
                zonal_blocks.append(D_l6[:, 6, :])
            else:
                zonal_blocks.append(self._compute_l5_block(edge_quaternion)[:, 5, :])

        if self.lmax >= 7 and lmin <= 8:
            if self.lmax >= 8:
                D_l7, D_l8 = self._compute_l7l8_blocks(edge_quaternion)
                if lmin <= 7:
                    zonal_blocks.append(D_l7[:, 7, :])
                zonal_blocks.append(D_l8[:, 8, :])
            else:
                zonal_blocks.append(self._compute_l7_block(edge_quaternion)[:, 7, :])

        if self.lmax >= 9 and lmin <= 10:
            if self.lmax >= 10:
                D_l9, D_l10 = self._compute_l9l10_blocks(edge_quaternion)
                if lmin <= 9:
                    zonal_blocks.append(D_l9[:, 9, :])
                zonal_blocks.append(D_l10[:, 10, :])
            else:
                zonal_blocks.append(self._compute_l9_block(edge_quaternion)[:, 9, :])

        if self.lmax >= self.poly_lmin and lmin <= self.lmax:
            ra_re, ra_im, rb_re, rb_im = self._quaternion_to_ra_rb_real(edge_quaternion)
            D_re, D_im = self._wigner_d_matrix_realpair(
                ra_re,
                ra_im,
                rb_re,
                rb_im,
                self.poly_coeffs,
                dtype=dtype,
            )
            D_poly = self._wigner_d_pair_to_real(
                D_re,
                D_im,
                (
                    self.poly_u_re,
                    self.poly_u_im,
                    self.poly_u_re_t,
                    self.poly_u_im_t,
                ),
                lmax=self.lmax,
                lmin=self.poly_lmin,
            )
            poly_lmin = max(lmin, self.poly_lmin)
            offset = 0
            for degree in range(self.poly_lmin, self.lmax + 1):
                block_size = 2 * degree + 1
                block_end = offset + block_size
                if degree >= poly_lmin:
                    zonal_blocks.append(D_poly[:, offset + degree, offset:block_end])
                offset = block_end

        return xp.concat(zonal_blocks, axis=1)

    @classmethod
    def _get_small_order_cache_cpu_fp64(cls, lmax: int) -> dict[str, np.ndarray]:
        """Generate the required low-order kernel coefficients on CPU fp64."""
        target_lmax = min(max(int(lmax), 2), 10)
        if cls._SMALL_ORDER_CACHE_CPU_FP64 is None:
            cls._SMALL_ORDER_CACHE_CPU_FP64 = {}
        cache = cls._SMALL_ORDER_CACHE_CPU_FP64
        required_names = WignerSmallOrderCoefficients.required_kernel_names(target_lmax)
        if any(name not in cache for name in required_names):
            cache.update(cls._generate_small_order_cache_cpu_fp64(target_lmax))
        return cache

    @classmethod
    def _build_small_order_kernels(
        cls,
        *,
        lmax: int,
    ) -> WignerSmallOrderCoefficients:
        """Instantiate the specialized ``l=2..10`` kernels on the requested device/dtype."""
        cache = cls._get_small_order_cache_cpu_fp64(lmax)
        kernels = {}
        for name in WignerSmallOrderCoefficients.required_kernel_names(lmax):
            kernels[name] = cache[name]
        return WignerSmallOrderCoefficients(
            lmax=lmax,
            kernels=kernels,
        )

    @classmethod
    def _generate_small_order_cache_cpu_fp64(cls, lmax: int) -> dict[str, np.ndarray]:
        """
        Generate the low-order kernel coefficients from the generic SeZM reference path.

        The coefficients are exact module constants. They are solved once in fp64 on CPU,
        validated against the generic quaternion polynomial evaluator, and then reused by
        every `WignerDCalculator` instance.
        """
        target_lmax = min(max(int(lmax), 2), 10)
        rng = np.random.default_rng(20260404)

        max_monomials = math.comb(2 * target_lmax + 3, 3)
        n_fit = min(2048, max(128, 2 * max_monomials))
        q_fit = rng.standard_normal((n_fit, 4))
        q_fit = quaternion_normalize(q_fit, eps=float(np.finfo(np.float64).eps))
        ref_blocks = cls._compute_generic_reference_blocks(q_fit, lmax=target_lmax)

        monomials: dict[int, list[tuple[int, int, int, int]]] = {}
        exponents: dict[int, np.ndarray] = {}
        coefficients: dict[int, np.ndarray] = {}
        cache: dict[str, np.ndarray] = {}

        for ell in range(2, target_lmax + 1):
            monomials[ell] = cls._generate_monomials(4, 2 * ell)
            exponents[ell] = cls._monomials_to_exponent_tensor(monomials[ell])
            coeff = cls._solve_monomial_coefficients(
                q_fit,
                ref_blocks[ell],
                exponents[ell],
            )
            if ell == 2:
                cache["C_l2"] = cls._build_l2_contraction_tensor(coeff, monomials[2])
            else:
                coefficients[ell] = coeff
                cache[f"C_l{ell}"] = coeff
                cache[f"exp_l{ell}"] = exponents[ell]

        combined_builders = {
            4: ("C_combined_l3l4", cls._build_combined_l3l4),
            6: ("C_combined_l5l6", cls._build_combined_l5l6),
            8: ("C_combined_l7l8", cls._build_combined_l7l8),
            10: ("C_combined_l9l10", cls._build_combined_l9l10),
        }
        for even_ell, (name, builder) in combined_builders.items():
            if target_lmax >= even_ell:
                odd_ell = even_ell - 1
                cache[name] = builder(
                    coefficients[odd_ell],
                    coefficients[even_ell],
                    monomials[odd_ell],
                    monomials[even_ell],
                )

        return cache

    @classmethod
    def _compute_generic_reference_blocks(
        cls,
        edge_quaternion: Any,
        *,
        lmax: int,
    ) -> dict[int, np.ndarray]:
        """Evaluate the generic SeZM polynomial path and extract per-degree blocks."""
        coeffs = cls._precompute_wigner_coefficients(
            lmax,
            lmin=2,
        )
        blocks = cls._precompute_real_basis_blocks(
            lmin=2,
            lmax=lmax,
        )
        ra_re, ra_im, rb_re, rb_im = cls._quaternion_to_ra_rb_real(edge_quaternion)
        D_re, D_im = cls._wigner_d_matrix_realpair(
            ra_re,
            ra_im,
            rb_re,
            rb_im,
            coeffs,
        )
        D_ref = cls._wigner_d_pair_to_real(
            D_re,
            D_im,
            blocks,
            lmax=lmax,
            lmin=2,
        )
        ref_blocks: dict[int, np.ndarray] = {}
        offset = 0
        for ell in range(2, lmax + 1):
            block_size = 2 * ell + 1
            block_end = offset + block_size
            ref_blocks[ell] = D_ref[:, offset:block_end, offset:block_end]
            offset = block_end
        return ref_blocks

    @classmethod
    def _solve_monomial_coefficients(
        cls,
        edge_quaternion: np.ndarray,
        D_block: np.ndarray,
        monomial_exponents: np.ndarray,
    ) -> np.ndarray:
        """Solve the flattened monomial coefficient matrix for one low-order block."""
        max_power = int(monomial_exponents.sum(axis=1).max())
        powers = cls._precompute_powers(edge_quaternion, max_power)
        M = cls._build_monomial_matrix(powers, monomial_exponents)
        Y = np.reshape(D_block, (edge_quaternion.shape[0], -1))
        return np.ascontiguousarray(np.linalg.lstsq(M, Y, rcond=None)[0].T)

    @staticmethod
    def _build_l2_contraction_tensor(
        C_l2_flat: np.ndarray,
        monomials: list[tuple[int, int, int, int]],
    ) -> np.ndarray:
        """Expand degree-4 monomial coefficients into the symmetric einsum tensor form."""
        C_l2 = np.zeros((5, 5, 4, 4, 4, 4), dtype=C_l2_flat.dtype)
        for flat_idx, coeff_row in enumerate(C_l2_flat):
            i = flat_idx // 5
            j = flat_idx % 5
            for coeff, (a, b, c, d) in zip(coeff_row, monomials, strict=True):
                if abs(float(coeff)) < 1e-15:
                    continue
                pool = [0] * a + [1] * b + [2] * c + [3] * d
                unique_permutations = set(permutations(pool, 4))
                share = coeff / len(unique_permutations)
                for p0, p1, p2, p3 in unique_permutations:
                    C_l2[i, j, p0, p1, p2, p3] = share
        return C_l2

    @staticmethod
    def _generate_monomials(
        n_vars: int,
        total_degree: int,
    ) -> list[tuple[int, ...]]:
        """Generate all monomials of fixed total degree in lexicographic order."""
        monomials: list[tuple[int, ...]] = []

        def _recurse(
            remaining_vars: int,
            remaining_degree: int,
            current: list[int],
        ) -> None:
            if remaining_vars == 1:
                monomials.append((*current, remaining_degree))
                return
            for i in range(remaining_degree + 1):
                _recurse(remaining_vars - 1, remaining_degree - i, [*current, i])

        _recurse(n_vars, total_degree, [])
        return monomials

    @staticmethod
    def _monomials_to_exponent_tensor(
        monomials: list[tuple[int, ...]],
    ) -> np.ndarray:
        """Convert monomial tuples to an ``int64`` exponent table."""
        return np.array(monomials, dtype=np.int64)

    @staticmethod
    def _build_combined_l3l4(
        C_l3: np.ndarray,
        C_l4: np.ndarray,
        monomials_l3: list[tuple[int, int, int, int]],
        monomials_l4: list[tuple[int, int, int, int]],
    ) -> np.ndarray:
        """Lift the ``l=3`` basis to degree 8 and stack it with the ``l=4`` basis."""
        mono8_to_idx = {mono: idx for idx, mono in enumerate(monomials_l4)}
        C_l3_lifted = np.zeros(
            (C_l3.shape[0], len(monomials_l4)),
            dtype=C_l3.dtype,
        )
        for j, (a, b, c, d) in enumerate(monomials_l3):
            for mono8 in (
                (a + 2, b, c, d),
                (a, b + 2, c, d),
                (a, b, c + 2, d),
                (a, b, c, d + 2),
            ):
                C_l3_lifted[:, mono8_to_idx[mono8]] += C_l3[:, j]
        return np.concatenate([C_l3_lifted, C_l4], axis=0)

    @staticmethod
    def _build_combined_l5l6(
        C_l5: np.ndarray,
        C_l6: np.ndarray,
        monomials_l5: list[tuple[int, int, int, int]],
        monomials_l6: list[tuple[int, int, int, int]],
    ) -> np.ndarray:
        """Lift the ``l=5`` basis to degree 12 and stack it with the ``l=6`` basis."""
        mono12_to_idx = {mono: idx for idx, mono in enumerate(monomials_l6)}
        C_l5_lifted = np.zeros(
            (C_l5.shape[0], len(monomials_l6)),
            dtype=C_l5.dtype,
        )
        for j, (a, b, c, d) in enumerate(monomials_l5):
            for mono12 in (
                (a + 2, b, c, d),
                (a, b + 2, c, d),
                (a, b, c + 2, d),
                (a, b, c, d + 2),
            ):
                C_l5_lifted[:, mono12_to_idx[mono12]] += C_l5[:, j]
        return np.concatenate([C_l5_lifted, C_l6], axis=0)

    @staticmethod
    def _build_combined_l7l8(
        C_l7: np.ndarray,
        C_l8: np.ndarray,
        monomials_l7: list[tuple[int, int, int, int]],
        monomials_l8: list[tuple[int, int, int, int]],
    ) -> np.ndarray:
        """Lift the ``l=7`` basis to degree 16 and stack it with the ``l=8`` basis."""
        mono16_to_idx = {mono: idx for idx, mono in enumerate(monomials_l8)}
        C_l7_lifted = np.zeros(
            (C_l7.shape[0], len(monomials_l8)),
            dtype=C_l7.dtype,
        )
        for j, (a, b, c, d) in enumerate(monomials_l7):
            for mono16 in (
                (a + 2, b, c, d),
                (a, b + 2, c, d),
                (a, b, c + 2, d),
                (a, b, c, d + 2),
            ):
                C_l7_lifted[:, mono16_to_idx[mono16]] += C_l7[:, j]
        return np.concatenate([C_l7_lifted, C_l8], axis=0)

    @staticmethod
    def _build_combined_l9l10(
        C_l9: np.ndarray,
        C_l10: np.ndarray,
        monomials_l9: list[tuple[int, int, int, int]],
        monomials_l10: list[tuple[int, int, int, int]],
    ) -> np.ndarray:
        """Lift the ``l=9`` basis to degree 20 and stack it with the ``l=10`` basis."""
        mono20_to_idx = {mono: idx for idx, mono in enumerate(monomials_l10)}
        C_l9_lifted = np.zeros(
            (C_l9.shape[0], len(monomials_l10)),
            dtype=C_l9.dtype,
        )
        for j, (a, b, c, d) in enumerate(monomials_l9):
            for mono20 in (
                (a + 2, b, c, d),
                (a, b + 2, c, d),
                (a, b, c + 2, d),
                (a, b, c, d + 2),
            ):
                C_l9_lifted[:, mono20_to_idx[mono20]] += C_l9[:, j]
        return np.concatenate([C_l9_lifted, C_l10], axis=0)

    @staticmethod
    def _precompute_powers(
        q: Any,
        max_power: int,
    ) -> Any:
        """Precompute powers ``q_i^k`` as a dense table with shape ``(4, max_power+1, E)``.

        The table is built by an explicit multiply chain: a ``cumprod`` over
        the short power axis lowers to a scan whose forward and leave-one-out
        backward cost several milliseconds per model call at typical edge
        counts, whereas the unrolled chain stays a fusable pointwise sequence.
        """
        xp = array_api_compat.array_namespace(q)
        components = xp.permute_dims(q, (1, 0))
        ones = xp.ones_like(components)
        if max_power == 0:
            return ones[:, None, :]
        powers = [ones, components]
        for _ in range(max_power - 1):
            powers.append(powers[-1] * components)
        return xp.stack(powers, axis=1)

    @staticmethod
    def _build_monomial_matrix(
        powers: Any,
        monomial_exponents: Any,
    ) -> Any:
        """Assemble the monomial design matrix for one fixed degree.

        The four gathered factor rows are combined by explicit multiplies:
        ``prod(dim=0)`` lowers to a ``cumprod`` scan pair (forward plus
        leave-one-out backward) on the large ``(4, M, E)`` intermediate,
        while two multiply levels keep the chain pointwise and fusable.
        """
        xp = array_api_compat.array_namespace(powers)
        n_mono = monomial_exponents.shape[0]
        n_edge = powers.shape[-1]
        gather_idx = xp.broadcast_to(
            xp.permute_dims(monomial_exponents, (1, 0))[:, :, None],
            (4, n_mono, n_edge),
        )
        selected = xp_take_along_axis(powers, gather_idx, axis=1)
        product = (selected[0] * selected[1]) * (selected[2] * selected[3])
        return xp.permute_dims(product, (1, 0))

    def _monomial_matrix(
        self,
        edge_quaternion: Any,
        exp_name: str,
        max_power: int,
    ) -> Any:
        """Evaluate one degree kernel's monomial basis via the dense power-table chain."""
        xp = array_api_compat.array_namespace(edge_quaternion)
        device = array_api_compat.device(edge_quaternion)
        powers = self._precompute_powers(edge_quaternion, max_power)
        return self._build_monomial_matrix(
            powers,
            xp_asarray_nodetach(
                xp, getattr(self.small_order_kernels, exp_name), device=device
            ),
        )

    def _compute_l1_block(self, edge_quaternion: Any) -> Any:
        """Compute the vector block directly from the Cartesian rotation matrix."""
        xp = array_api_compat.array_namespace(edge_quaternion)
        device = array_api_compat.device(edge_quaternion)
        rot_mat = quaternion_to_rotation_matrix(edge_quaternion)
        perm = xp_asarray_nodetach(xp, self.l1_perm, device=device)
        rot_perm = xp.take(xp.take(rot_mat, perm, axis=-2), perm, axis=-1)
        sign = xp_asarray_nodetach(
            xp, self.l1_sign_outer, dtype=edge_quaternion.dtype, device=device
        )
        return rot_perm * sign

    def _compute_l2_block(self, edge_quaternion: Any) -> Any:
        """Compute the ``l=2`` block from the degree-4 quaternion contraction."""
        xp = array_api_compat.array_namespace(edge_quaternion)
        device = array_api_compat.device(edge_quaternion)
        n_edge = edge_quaternion.shape[0]
        q2 = edge_quaternion[..., :, None] * edge_quaternion[..., None, :]
        q4 = q2[..., :, :, None, None] * q2[..., None, None, :, :]
        c_l2 = xp_asarray_nodetach(
            xp,
            self.small_order_kernels.C_l2,
            dtype=edge_quaternion.dtype,
            device=device,
        )
        # einsum "nabcd,ijabcd->nij" as a flattened matmul over the (a, b, c, d) axes.
        q4_flat = xp.reshape(q4, (n_edge, 256))
        c_flat = xp.reshape(c_l2, (25, 256))
        out = xp.matmul(q4_flat, xp.permute_dims(c_flat, (1, 0)))
        return xp.reshape(out, (n_edge, 5, 5))

    def _compute_l3_block(self, edge_quaternion: Any) -> Any:
        """Compute the ``l=3`` block from the dedicated degree-6 monomial kernel."""
        xp = array_api_compat.array_namespace(edge_quaternion)
        device = array_api_compat.device(edge_quaternion)
        n_edge = edge_quaternion.shape[0]
        monomials = self._monomial_matrix(edge_quaternion, "exp_l3", 6)
        c = xp_asarray_nodetach(
            xp,
            self.small_order_kernels.C_l3,
            dtype=edge_quaternion.dtype,
            device=device,
        )
        D_flat = xp.matmul(monomials, xp.permute_dims(c, (1, 0)))
        return xp.reshape(D_flat, (n_edge, 7, 7))

    def _compute_l3l4_blocks(
        self,
        edge_quaternion: Any,
    ) -> tuple[Any, Any]:
        """Compute the ``l=3`` and ``l=4`` blocks from one shared degree-8 kernel."""
        xp = array_api_compat.array_namespace(edge_quaternion)
        device = array_api_compat.device(edge_quaternion)
        n_edge = edge_quaternion.shape[0]
        monomials = self._monomial_matrix(edge_quaternion, "exp_l4", 8)
        c = xp_asarray_nodetach(
            xp,
            self.small_order_kernels.C_combined_l3l4,
            dtype=edge_quaternion.dtype,
            device=device,
        )
        D_flat = xp.matmul(monomials, xp.permute_dims(c, (1, 0)))
        D_l3 = xp.reshape(D_flat[:, :49], (n_edge, 7, 7))
        D_l4 = xp.reshape(D_flat[:, 49:], (n_edge, 9, 9))
        return D_l3, D_l4

    def _compute_l5_block(self, edge_quaternion: Any) -> Any:
        """Compute the ``l=5`` block from the dedicated degree-10 monomial kernel."""
        xp = array_api_compat.array_namespace(edge_quaternion)
        device = array_api_compat.device(edge_quaternion)
        n_edge = edge_quaternion.shape[0]
        monomials = self._monomial_matrix(edge_quaternion, "exp_l5", 10)
        c = xp_asarray_nodetach(
            xp,
            self.small_order_kernels.C_l5,
            dtype=edge_quaternion.dtype,
            device=device,
        )
        D_flat = xp.matmul(monomials, xp.permute_dims(c, (1, 0)))
        return xp.reshape(D_flat, (n_edge, 11, 11))

    def _compute_l5l6_blocks(
        self,
        edge_quaternion: Any,
    ) -> tuple[Any, Any]:
        """Compute the ``l=5`` and ``l=6`` blocks from one shared degree-12 kernel."""
        xp = array_api_compat.array_namespace(edge_quaternion)
        device = array_api_compat.device(edge_quaternion)
        n_edge = edge_quaternion.shape[0]
        monomials = self._monomial_matrix(edge_quaternion, "exp_l6", 12)
        c = xp_asarray_nodetach(
            xp,
            self.small_order_kernels.C_combined_l5l6,
            dtype=edge_quaternion.dtype,
            device=device,
        )
        D_flat = xp.matmul(monomials, xp.permute_dims(c, (1, 0)))
        D_l5 = xp.reshape(D_flat[:, :121], (n_edge, 11, 11))
        D_l6 = xp.reshape(D_flat[:, 121:], (n_edge, 13, 13))
        return D_l5, D_l6

    def _compute_l7_block(self, edge_quaternion: Any) -> Any:
        """Compute the ``l=7`` block from the dedicated degree-14 monomial kernel."""
        xp = array_api_compat.array_namespace(edge_quaternion)
        device = array_api_compat.device(edge_quaternion)
        n_edge = edge_quaternion.shape[0]
        powers = self._precompute_powers(edge_quaternion, 14)
        monomials = self._build_monomial_matrix(
            powers,
            xp_asarray_nodetach(xp, self.small_order_kernels.exp_l7, device=device),
        )
        c = xp_asarray_nodetach(
            xp,
            self.small_order_kernels.C_l7,
            dtype=edge_quaternion.dtype,
            device=device,
        )
        D_flat = xp.matmul(monomials, xp.permute_dims(c, (1, 0)))
        return xp.reshape(D_flat, (n_edge, 15, 15))

    def _compute_l7l8_blocks(
        self,
        edge_quaternion: Any,
    ) -> tuple[Any, Any]:
        """Compute the ``l=7`` and ``l=8`` blocks from one shared degree-16 kernel."""
        xp = array_api_compat.array_namespace(edge_quaternion)
        device = array_api_compat.device(edge_quaternion)
        n_edge = edge_quaternion.shape[0]
        powers = self._precompute_powers(edge_quaternion, 16)
        monomials = self._build_monomial_matrix(
            powers,
            xp_asarray_nodetach(xp, self.small_order_kernels.exp_l8, device=device),
        )
        c = xp_asarray_nodetach(
            xp,
            self.small_order_kernels.C_combined_l7l8,
            dtype=edge_quaternion.dtype,
            device=device,
        )
        D_flat = xp.matmul(monomials, xp.permute_dims(c, (1, 0)))
        D_l7 = xp.reshape(D_flat[:, :225], (n_edge, 15, 15))
        D_l8 = xp.reshape(D_flat[:, 225:], (n_edge, 17, 17))
        return D_l7, D_l8

    def _compute_l9_block(self, edge_quaternion: Any) -> Any:
        """Compute the ``l=9`` block from the dedicated degree-18 monomial kernel."""
        xp = array_api_compat.array_namespace(edge_quaternion)
        device = array_api_compat.device(edge_quaternion)
        n_edge = edge_quaternion.shape[0]
        powers = self._precompute_powers(edge_quaternion, 18)
        monomials = self._build_monomial_matrix(
            powers,
            xp_asarray_nodetach(xp, self.small_order_kernels.exp_l9, device=device),
        )
        c = xp_asarray_nodetach(
            xp,
            self.small_order_kernels.C_l9,
            dtype=edge_quaternion.dtype,
            device=device,
        )
        D_flat = xp.matmul(monomials, xp.permute_dims(c, (1, 0)))
        return xp.reshape(D_flat, (n_edge, 19, 19))

    def _compute_l9l10_blocks(
        self,
        edge_quaternion: Any,
    ) -> tuple[Any, Any]:
        """Compute the ``l=9`` and ``l=10`` blocks from one shared degree-20 kernel."""
        xp = array_api_compat.array_namespace(edge_quaternion)
        device = array_api_compat.device(edge_quaternion)
        n_edge = edge_quaternion.shape[0]
        powers = self._precompute_powers(edge_quaternion, 20)
        monomials = self._build_monomial_matrix(
            powers,
            xp_asarray_nodetach(xp, self.small_order_kernels.exp_l10, device=device),
        )
        c = xp_asarray_nodetach(
            xp,
            self.small_order_kernels.C_combined_l9l10,
            dtype=edge_quaternion.dtype,
            device=device,
        )
        D_flat = xp.matmul(monomials, xp.permute_dims(c, (1, 0)))
        D_l9 = xp.reshape(D_flat[:, :361], (n_edge, 19, 19))
        D_l10 = xp.reshape(D_flat[:, 361:], (n_edge, 21, 21))
        return D_l9, D_l10

    @staticmethod
    def _factorial_table(n: int) -> np.ndarray:
        """Return ``[0!, 1!, ..., n!]`` in the requested dtype/device."""
        table = np.zeros(n + 1, dtype=np.float64)
        table[0] = 1.0
        for i in range(1, n + 1):
            table[i] = table[i - 1] * i
        return table

    @staticmethod
    def _binomial(n: int, k: int, factorial: np.ndarray) -> float:
        """Evaluate ``C(n, k)`` from a precomputed factorial table."""
        if k < 0 or k > n:
            return 0.0
        return float(factorial[n] / (factorial[k] * factorial[n - k]))

    @staticmethod
    def _allocate_case_coeffs(
        n_primary: int,
        max_poly_len: int,
    ) -> CaseCoefficients:
        """Allocate one branch of Horner tables for the quaternion Wigner evaluator."""
        return CaseCoefficients(
            coeff=np.zeros(n_primary, dtype=np.float64),
            horner=np.zeros((n_primary, max_poly_len), dtype=np.float64),
            poly_len=np.zeros(n_primary, dtype=np.int64),
            ra_exp=np.zeros(n_primary, dtype=np.float64),
            rb_exp=np.zeros(n_primary, dtype=np.float64),
            sign=np.zeros(n_primary, dtype=np.float64),
        )

    @staticmethod
    def _compute_case_coefficients(
        case: CaseCoefficients,
        idx: int,
        ell: int,
        mp: int,
        m: int,
        sqrt_factor: float,
        factorial: np.ndarray,
        *,
        is_case1: bool,
    ) -> None:
        """
        Fill one Horner branch for a fixed ``(ell, mp, m)`` entry.

        The closed-form quaternion Wigner formula is reorganized so that only the ratio
        ``-(|Rb|/|Ra|)^2`` or ``-(|Ra|/|Rb|)^2`` enters the Horner chain. This avoids a
        large family of per-entry runtime branches and keeps the generic path stable for
        every ``ell``.
        """
        if is_case1:
            rho_min = max(0, mp - m)
            rho_max = min(ell + mp, ell - m)
        else:
            rho_min = max(0, -(mp + m))
            rho_max = min(ell - m, ell - mp)

        if rho_min > rho_max:
            return

        if is_case1:
            binom1 = WignerDCalculator._binomial(ell + mp, rho_min, factorial)
            binom2 = WignerDCalculator._binomial(ell - mp, ell - m - rho_min, factorial)
        else:
            binom1 = WignerDCalculator._binomial(ell + mp, ell - m - rho_min, factorial)
            binom2 = WignerDCalculator._binomial(ell - mp, rho_min, factorial)
        case.coeff[idx] = sqrt_factor * binom1 * binom2

        poly_len = rho_max - rho_min + 1
        case.poly_len[idx] = poly_len
        for i, rho in enumerate(range(rho_max, rho_min, -1)):
            if is_case1:
                n1 = ell + mp - rho + 1
                n2 = ell - m - rho + 1
                d1 = rho
                d2 = m - mp + rho
            else:
                n1 = ell - m - rho + 1
                n2 = ell - mp - rho + 1
                d1 = rho
                d2 = mp + m + rho
            if d1 != 0 and d2 != 0:
                case.horner[idx, i] = (n1 * n2) / (d1 * d2)

        if is_case1:
            case.ra_exp[idx] = 2 * ell + mp - m - 2 * rho_min
            case.rb_exp[idx] = m - mp + 2 * rho_min
            case.sign[idx] = (-1) ** rho_min
        else:
            case.ra_exp[idx] = mp + m + 2 * rho_min
            case.rb_exp[idx] = 2 * ell - mp - m - 2 * rho_min
            case.sign[idx] = ((-1) ** (ell - m)) * ((-1) ** rho_min)

    @staticmethod
    def _finalize_case_coefficients(
        case: CaseCoefficients,
        max_poly_len: int,
    ) -> None:
        """Attach runtime-ready masks and fused coefficients for one Horner branch."""
        step_count = np.clip(case.poly_len - 1, 0, None)
        if max_poly_len > 1:
            horner_step_mask = (
                np.arange(max_poly_len - 1, dtype=case.poly_len.dtype)[None, :]
                < step_count[:, None]
            )
        else:
            horner_step_mask = np.zeros((case.poly_len.shape[0], 0), dtype=np.bool_)
        case.valid_mask = case.poly_len > 0
        case.horner_step_mask = horner_step_mask
        case.signed_coeff = case.sign * case.coeff

    @staticmethod
    def _vectorized_horner(
        ratio: Any,
        horner_coeffs: Any,
        horner_step_mask: Any,
    ) -> Any:
        """Evaluate many varying-length Horner chains in one batched loop."""
        xp = array_api_compat.array_namespace(ratio)
        device = array_api_compat.device(ratio)
        n_batch = ratio.shape[0]
        n_elements = horner_coeffs.shape[0]
        result = xp.ones((n_batch, n_elements), dtype=ratio.dtype, device=device)
        if horner_step_mask.shape[1] == 0:
            return result
        ratio = ratio[:, None]
        for i in range(horner_step_mask.shape[1]):
            new_result = 1.0 + result * (ratio * horner_coeffs[None, :, i])
            result = xp.where(horner_step_mask[None, :, i], new_result, result)
        return result

    @staticmethod
    def _compute_case_magnitude(
        log_ra: Any,
        log_rb: Any,
        ratio: Any,
        case: CaseCoefficients,
    ) -> Any:
        """Compute the real magnitude factor for one stable Horner branch."""
        xp = array_api_compat.array_namespace(log_ra)
        device = array_api_compat.device(log_ra)
        horner_sum = WignerDCalculator._vectorized_horner(
            ratio,
            xp_asarray_nodetach(xp, case.horner, device=device),
            xp_asarray_nodetach(xp, case.horner_step_mask, device=device),
        )
        ra_powers = xp.exp(
            log_ra[:, None]
            * xp_asarray_nodetach(xp, case.ra_exp, device=device)[None, :]
        )
        rb_powers = xp.exp(
            log_rb[:, None]
            * xp_asarray_nodetach(xp, case.rb_exp, device=device)[None, :]
        )
        signed_coeff = xp_asarray_nodetach(xp, case.signed_coeff, device=device)
        magnitude = signed_coeff[None, :] * ra_powers * rb_powers
        return magnitude * horner_sum

    @staticmethod
    def _build_complex_to_real_sh_block(ell: int) -> np.ndarray:
        """
        Build the complex-to-real basis transform for one ``ell`` block.

        The packed real basis follows the SeZM convention
        ``m = -ell, ..., +ell`` inside each block. This unitary transform defines the
        real tesseral basis used by the packed ``D_full`` layout.
        """
        size = 2 * ell + 1
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        U = np.zeros((size, size), dtype=np.complex128)
        for m in range(-ell, ell + 1):
            row = m + ell
            if m == 0:
                U[row, ell] = 1.0
            elif m > 0:
                U[row, m + ell] = inv_sqrt2
                U[row, -m + ell] = ((-1) ** m) * inv_sqrt2
            else:
                U[row, -m + ell] = -1j * inv_sqrt2
                U[row, m + ell] = ((-1) ** m) * 1j * inv_sqrt2
        return U

    @staticmethod
    def _precompute_real_basis_blocks(
        *,
        lmin: int,
        lmax: int,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Precompute complex-to-real basis transforms for ``ell in [lmin, lmax]``."""
        if lmin > lmax:
            return []
        blocks: list[tuple[np.ndarray, np.ndarray]] = []
        for ell in range(lmin, lmax + 1):
            U = WignerDCalculator._build_complex_to_real_sh_block(ell)
            blocks.append((U.real.astype(np.float64), U.imag.astype(np.float64)))
        return blocks

    @staticmethod
    def _assemble_block_diagonal_real_basis(
        U_blocks: list[tuple[np.ndarray, np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Assemble per-``ell`` real-basis blocks into one block-diagonal transform."""
        if not U_blocks:
            empty = np.zeros((0, 0), dtype=np.float64)
            return empty, empty, empty, empty

        size = sum(U_re.shape[0] for U_re, _ in U_blocks)
        U_re_full = np.zeros((size, size), dtype=np.float64)
        U_im_full = np.zeros((size, size), dtype=np.float64)
        offset = 0
        for U_re, U_im in U_blocks:
            block_size = U_re.shape[0]
            block_end = offset + block_size
            U_re_full[offset:block_end, offset:block_end] = U_re
            U_im_full[offset:block_end, offset:block_end] = U_im
            offset = block_end
        return (
            U_re_full,
            U_im_full,
            np.ascontiguousarray(U_re_full.T),
            np.ascontiguousarray(U_im_full.T),
        )

    @staticmethod
    def _quaternion_to_ra_rb_real(
        q: Any,
    ) -> tuple[Any, Any, Any, Any]:
        """
        Decompose quaternion components into the Cayley-Klein pair used by the generic path.

        For ``q = (w, x, y, z)`` the SeZM real-basis convention is aligned by

        ``Ra = w - i z`` and ``Rb = y - i x``.

        This pairing matches the packed SeZM real spherical-harmonics ordering used by
        the block-diagonal ``D_full`` layout.
        """
        w = q[..., 0]
        x = q[..., 1]
        y = q[..., 2]
        z = q[..., 3]
        return w, -z, y, -x

    @staticmethod
    def _precompute_wigner_coefficients(
        lmax: int,
        *,
        lmin: int = 0,
    ) -> WignerPolynomialCoefficients:
        """
        Precompute the generic quaternion Wigner coefficient tables.

        The runtime path only performs batched Horner evaluation and symmetry scatter.
        All factorial ratios, branch exponents, and packed matrix indices are resolved once
        here, which keeps the forward path independent of ``ell`` and stable for arbitrary
        ``lmax``.
        """
        if lmin < 0:
            raise ValueError("`lmin` must be non-negative")
        if lmax < lmin:
            raise ValueError("`lmax` must be >= `lmin`")

        factorial = WignerDCalculator._factorial_table(2 * lmax + 1)
        n_total = sum((2 * ell + 1) ** 2 for ell in range(lmin, lmax + 1))
        n_primary = sum(
            1
            for ell in range(lmin, lmax + 1)
            for mp in range(-ell, ell + 1)
            for m in range(-ell, ell + 1)
            if mp + m > 0 or (mp + m == 0 and mp >= 0)
        )
        n_derived = n_total - n_primary
        max_poly_len = lmax + 1
        size = (lmax + 1) ** 2 - lmin * lmin

        primary_row = np.zeros(n_primary, dtype=np.int64)
        primary_col = np.zeros(n_primary, dtype=np.int64)
        mp_plus_m = np.zeros(n_primary, dtype=np.float64)
        m_minus_mp = np.zeros(n_primary, dtype=np.float64)
        diagonal_mask = np.zeros(n_primary, dtype=np.bool_)
        anti_diagonal_mask = np.zeros(n_primary, dtype=np.bool_)
        special_2m = np.zeros(n_primary, dtype=np.float64)
        anti_diag_sign = np.zeros(n_primary, dtype=np.float64)
        case1 = WignerDCalculator._allocate_case_coeffs(
            n_primary,
            max_poly_len,
        )
        case2 = WignerDCalculator._allocate_case_coeffs(
            n_primary,
            max_poly_len,
        )
        derived_row = np.zeros(n_derived, dtype=np.int64)
        derived_col = np.zeros(n_derived, dtype=np.int64)
        derived_primary_idx = np.zeros(n_derived, dtype=np.int64)
        derived_sign = np.zeros(n_derived, dtype=np.float64)

        primary_map: dict[tuple[int, int], int] = {}
        primary_idx = 0
        block_start = 0
        for ell in range(lmin, lmax + 1):
            block_size = 2 * ell + 1
            for mp_local in range(block_size):
                mp = mp_local - ell
                for m_local in range(block_size):
                    m = m_local - ell
                    row = block_start + mp_local
                    col = block_start + m_local
                    is_primary = (mp + m > 0) or (mp + m == 0 and mp >= 0)
                    if not is_primary:
                        continue

                    primary_map[(row, col)] = primary_idx
                    primary_row[primary_idx] = row
                    primary_col[primary_idx] = col
                    mp_plus_m[primary_idx] = mp + m
                    m_minus_mp[primary_idx] = m - mp
                    diagonal_mask[primary_idx] = mp == m
                    anti_diagonal_mask[primary_idx] = mp == -m
                    special_2m[primary_idx] = 2 * m
                    anti_diag_sign[primary_idx] = (-1) ** (ell - m)

                    sqrt_factor = math.sqrt(
                        float(factorial[ell + m] * factorial[ell - m])
                        / float(factorial[ell + mp] * factorial[ell - mp])
                    )
                    WignerDCalculator._compute_case_coefficients(
                        case1,
                        primary_idx,
                        ell,
                        mp,
                        m,
                        sqrt_factor,
                        factorial,
                        is_case1=True,
                    )
                    WignerDCalculator._compute_case_coefficients(
                        case2,
                        primary_idx,
                        ell,
                        mp,
                        m,
                        sqrt_factor,
                        factorial,
                        is_case1=False,
                    )
                    primary_idx += 1
            block_start += block_size

        derived_idx = 0
        block_start = 0
        for ell in range(lmin, lmax + 1):
            block_size = 2 * ell + 1
            for mp_local in range(block_size):
                mp = mp_local - ell
                for m_local in range(block_size):
                    m = m_local - ell
                    row = block_start + mp_local
                    col = block_start + m_local
                    is_primary = (mp + m > 0) or (mp + m == 0 and mp >= 0)
                    if is_primary:
                        continue

                    derived_row[derived_idx] = row
                    derived_col[derived_idx] = col
                    derived_primary_idx[derived_idx] = primary_map[
                        (block_start + (-mp + ell), block_start + (-m + ell))
                    ]
                    derived_sign[derived_idx] = (-1) ** (mp - m)
                    derived_idx += 1
            block_start += block_size

        WignerDCalculator._finalize_case_coefficients(case1, max_poly_len)
        WignerDCalculator._finalize_case_coefficients(case2, max_poly_len)

        coeffs = WignerPolynomialCoefficients(
            lmin=lmin,
            lmax=lmax,
            size=size,
            max_poly_len=max_poly_len,
            n_primary=n_primary,
            n_derived=n_derived,
            primary_row=primary_row,
            primary_col=primary_col,
            case1=case1,
            case2=case2,
            mp_plus_m=mp_plus_m,
            m_minus_mp=m_minus_mp,
            diagonal_mask=diagonal_mask,
            anti_diagonal_mask=anti_diagonal_mask,
            special_2m=special_2m,
            anti_diag_sign=anti_diag_sign,
            derived_row=derived_row,
            derived_col=derived_col,
            derived_primary_idx=derived_primary_idx,
            derived_sign=derived_sign,
        )

        # Functional scatter index: maps each flat ``(row, col)`` of the packed
        # ``(size, size)`` matrix to its source slot in
        # ``concat([primary, derived, zero])``. Off-block positions point at the
        # trailing zero slot. This replaces pt's in-place ``D[:, row, col] = value``
        # scatter with an export-safe gather.
        flat_to_src = np.full(size * size, n_primary + n_derived, dtype=np.int64)
        flat_to_src[primary_row * size + primary_col] = np.arange(
            n_primary, dtype=np.int64
        )
        flat_to_src[derived_row * size + derived_col] = n_primary + np.arange(
            n_derived, dtype=np.int64
        )
        coeffs.flat_gather_idx = flat_to_src

        return coeffs

    @staticmethod
    def _wigner_d_matrix_realpair(
        ra_re: Any,
        ra_im: Any,
        rb_re: Any,
        rb_im: Any,
        coeffs: WignerPolynomialCoefficients,
        *,
        dtype: Any = None,
    ) -> tuple[Any, Any]:
        """
        Evaluate the complex Wigner blocks in real/imaginary form.

        The runtime path uses only real arithmetic. The complex phase is represented by
        two real tensors, while the polynomial and magnitude algebra is evaluated in
        ``fp64`` before the result is cast back to the requested output dtype.
        """
        xp = array_api_compat.array_namespace(ra_re)
        device = array_api_compat.device(ra_re)
        n_batch = ra_re.shape[0]
        output_dtype = ra_re.dtype if dtype is None else dtype
        if coeffs.size == 0:
            zeros = xp.zeros((n_batch, 0, 0), dtype=output_dtype, device=device)
            return zeros, zeros

        f64 = xp.float64
        ra_re = xp.astype(ra_re, f64)
        ra_im = xp.astype(ra_im, f64)
        rb_re = xp.astype(rb_re, f64)
        rb_im = xp.astype(rb_im, f64)

        def cv(arr: np.ndarray) -> Any:
            return xp_asarray_nodetach(xp, arr, device=device)

        eps = float(np.finfo(np.float64).eps)
        eps_sq = eps * eps
        ra_sq = ra_re * ra_re + ra_im * ra_im
        rb_sq = rb_re * rb_re + rb_im * rb_im
        ra_small = ra_sq <= eps_sq
        rb_small = rb_sq <= eps_sq
        ra = xp.sqrt(xp.clip(ra_sq, min=eps_sq))
        rb = xp.sqrt(xp.clip(rb_sq, min=eps_sq))
        general_mask = ~ra_small & ~rb_small
        use_case1 = (ra >= rb) & general_mask
        use_case2 = (ra < rb) & general_mask

        safe_ra_re = xp.where(ra_small, xp.ones_like(ra_re), ra_re)
        safe_ra_im = xp.where(ra_small, xp.zeros_like(ra_im), ra_im)
        safe_rb_re = xp.where(rb_small, xp.ones_like(rb_re), rb_re)
        safe_rb_im = xp.where(rb_small, xp.zeros_like(rb_im), rb_im)
        phia = xp.atan2(safe_ra_im, safe_ra_re)
        phib = xp.atan2(safe_rb_im, safe_rb_re)

        phase = (
            phia[:, None] * cv(coeffs.mp_plus_m)[None, :]
            + phib[:, None] * cv(coeffs.m_minus_mp)[None, :]
        )
        exp_phase_re = xp.cos(phase)
        exp_phase_im = xp.sin(phase)

        safe_ra = xp.clip(ra, min=eps)
        safe_rb = xp.clip(rb, min=eps)
        log_ra = xp.log(safe_ra)
        log_rb = xp.log(safe_rb)

        result_re = xp.zeros((n_batch, coeffs.n_primary), dtype=f64, device=device)
        result_im = xp.zeros((n_batch, coeffs.n_primary), dtype=f64, device=device)

        special_2m = cv(coeffs.special_2m)
        anti_rows = ra_small
        anti_log_rb = xp.where(anti_rows, log_rb, xp.zeros_like(log_rb))
        anti_phib = xp.where(anti_rows, phib, xp.zeros_like(phib))
        rb_power_mag = xp.exp(anti_log_rb[:, None] * special_2m[None, :])
        rb_power_phase = anti_phib[:, None] * special_2m[None, :]
        anti_diag_sign = cv(coeffs.anti_diag_sign)
        anti_re = anti_diag_sign[None, :] * rb_power_mag * xp.cos(rb_power_phase)
        anti_im = anti_diag_sign[None, :] * rb_power_mag * xp.sin(rb_power_phase)
        anti_mask = ra_small[:, None] & cv(coeffs.anti_diagonal_mask)[None, :]
        result_re = xp.where(anti_mask, anti_re, result_re)
        result_im = xp.where(anti_mask, anti_im, result_im)

        diag_rows = rb_small & ~ra_small
        diag_log_ra = xp.where(diag_rows, log_ra, xp.zeros_like(log_ra))
        diag_phia = xp.where(diag_rows, phia, xp.zeros_like(phia))
        ra_power_mag = xp.exp(diag_log_ra[:, None] * special_2m[None, :])
        ra_power_phase = diag_phia[:, None] * special_2m[None, :]
        diag_re = ra_power_mag * xp.cos(ra_power_phase)
        diag_im = ra_power_mag * xp.sin(ra_power_phase)
        diag_mask = diag_rows[:, None] & cv(coeffs.diagonal_mask)[None, :]
        result_re = xp.where(diag_mask, diag_re, result_re)
        result_im = xp.where(diag_mask, diag_im, result_im)

        ratio1 = -(rb * rb) / (safe_ra * safe_ra)
        case1_rows = use_case1
        magnitude1 = WignerDCalculator._compute_case_magnitude(
            xp.where(case1_rows, log_ra, xp.zeros_like(log_ra)),
            xp.where(case1_rows, log_rb, xp.zeros_like(log_rb)),
            xp.where(case1_rows, ratio1, xp.zeros_like(ratio1)),
            coeffs.case1,
        )
        val1_re = magnitude1 * exp_phase_re
        val1_im = magnitude1 * exp_phase_im
        mask1 = case1_rows[:, None] & cv(coeffs.case1.valid_mask)[None, :]
        result_re = xp.where(mask1, val1_re, result_re)
        result_im = xp.where(mask1, val1_im, result_im)

        ratio2 = -(ra * ra) / (safe_rb * safe_rb)
        case2_rows = use_case2
        magnitude2 = WignerDCalculator._compute_case_magnitude(
            xp.where(case2_rows, log_ra, xp.zeros_like(log_ra)),
            xp.where(case2_rows, log_rb, xp.zeros_like(log_rb)),
            xp.where(case2_rows, ratio2, xp.zeros_like(ratio2)),
            coeffs.case2,
        )
        val2_re = magnitude2 * exp_phase_re
        val2_im = magnitude2 * exp_phase_im
        mask2 = case2_rows[:, None] & cv(coeffs.case2.valid_mask)[None, :]
        result_re = xp.where(mask2, val2_re, result_re)
        result_im = xp.where(mask2, val2_im, result_im)

        # Symmetry completion + scatter as one functional gather (see
        # ``_precompute_wigner_coefficients`` for ``flat_gather_idx``).
        derived_primary_idx = cv(coeffs.derived_primary_idx)
        derived_sign = cv(coeffs.derived_sign)
        primary_re = xp.take(result_re, derived_primary_idx, axis=1)
        primary_im = xp.take(result_im, derived_primary_idx, axis=1)
        derived_re = derived_sign[None, :] * primary_re
        derived_im = -derived_sign[None, :] * primary_im
        zero_col = xp.zeros((n_batch, 1), dtype=f64, device=device)
        flat_idx = cv(coeffs.flat_gather_idx)
        D_re = xp.reshape(
            xp.take(
                xp.concat([result_re, derived_re, zero_col], axis=1),
                flat_idx,
                axis=1,
            ),
            (n_batch, coeffs.size, coeffs.size),
        )
        D_im = xp.reshape(
            xp.take(
                xp.concat([result_im, derived_im, zero_col], axis=1),
                flat_idx,
                axis=1,
            ),
            (n_batch, coeffs.size, coeffs.size),
        )
        return xp.astype(D_re, output_dtype), xp.astype(D_im, output_dtype)

    @staticmethod
    def _wigner_d_pair_to_real(
        D_re: Any,
        D_im: Any,
        U_blocks: list[tuple[np.ndarray, np.ndarray]] | tuple[Any, Any, Any, Any],
        *,
        lmax: int,
        lmin: int,
    ) -> Any:
        """
        Convert complex Wigner blocks to the current real packed basis.

        Each block applies the SeZM complex-to-real basis transform for its degree.
        This preserves the packed ``(l, m)`` contract of ``D_full`` and ``Dt_full``.
        """
        xp = array_api_compat.array_namespace(D_re)
        device = array_api_compat.device(D_re)
        n_batch = D_re.shape[0]
        if lmin > lmax:
            return xp.zeros((n_batch, 0, 0), dtype=D_re.dtype, device=device)

        if isinstance(U_blocks, list):
            U_re, U_im, U_re_t, U_im_t = (
                WignerDCalculator._assemble_block_diagonal_real_basis(U_blocks)
            )
        else:
            U_re, U_im, U_re_t, U_im_t = U_blocks

        U_re = xp_asarray_nodetach(xp, U_re, dtype=D_re.dtype, device=device)
        U_im = xp_asarray_nodetach(xp, U_im, dtype=D_re.dtype, device=device)
        U_re_t = xp_asarray_nodetach(xp, U_re_t, dtype=D_re.dtype, device=device)
        U_im_t = xp_asarray_nodetach(xp, U_im_t, dtype=D_re.dtype, device=device)

        temp_re = xp.matmul(D_re, U_re_t) + xp.matmul(D_im, U_im_t)
        temp_im = xp.matmul(D_im, U_re_t) - xp.matmul(D_re, U_im_t)
        return xp.matmul(U_re, temp_re) - xp.matmul(U_im, temp_im)

    def serialize(self) -> dict[str, Any]:
        """Serialize WignerDCalculator (lmax and precision are stored by parent)."""
        return {
            "@class": "WignerDCalculator",
            "@version": 1,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> WignerDCalculator:
        """Deserialize WignerDCalculator - parent handles lmax/precision reconstruction."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "WignerDCalculator":
            raise ValueError(f"Invalid class for WignerDCalculator: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        raise NotImplementedError(
            "WignerDCalculator.deserialize should be called by parent with lmax/precision"
        )
