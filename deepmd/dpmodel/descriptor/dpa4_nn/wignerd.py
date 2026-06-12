# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Quaternion-based Wigner-D and edge-frame utilities for the DPA4/SeZM descriptor.

This module is the dpmodel port of ``deepmd.pt.model.descriptor.sezm_nn.wignerd``.
It defines the quaternion helpers and the Wigner-D evaluator used to construct
edge-aligned SO(3) rotation blocks.

Port notes
----------
- The pt reference evaluates the ``l=2..10`` blocks with monomial kernels whose
  coefficients are solved at init time by ``torch.linalg.lstsq`` against the
  generic closed-form quaternion polynomial path (seeded ``torch.randn`` fit
  points). That fit is a performance optimization and is not bit-reproducible
  without torch. The dpmodel port instead evaluates the generic closed-form
  path (the very reference the pt kernels are fitted to) for every ``l >= 2``.
  Outputs agree with pt within the fp64 round-off of the pt fit (validated by
  the parity tests at ``rtol=1e-12, atol=1e-14``).
- All coefficient tables are plain numpy arrays computed at ``__init__`` time;
  ``call`` is pure array-API. The block-diagonal matrices are assembled
  functionally with a precomputed ``xp.take`` gather index (no ``__setitem__``
  on traced values), so the path is safe for later torch.export
  functionalization.
- Random-gamma gauge randomization is NOT part of this module in pt either:
  it lives in ``edge_cache`` and only consumes the deterministic helpers
  ``quaternion_z_rotation`` / ``quaternion_multiply`` ported here.

Serialization contract: pt ``WignerDCalculator.serialize()`` emits only
``{"@class", "@version"}`` (all buffers are derived constants rebuilt from
``lmax``/``dtype`` by the parent). The dpmodel port mirrors that contract.
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
    NativeOP,
)
from deepmd.dpmodel.array_api import (
    xp_asarray_nodetach,
)
from deepmd.dpmodel.common import (
    get_xp_precision,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .utils import (
    safe_norm,
)


def quaternion_normalize(q: Any, eps: float = 1e-7) -> Any:
    """Normalize quaternions with a differentiable epsilon floor."""
    # safe_norm is the array-API port of pt's _safe_norm_nd (same formula)
    return q / safe_norm(q, eps)


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

    The returned matrix is the active rotation represented by ``q``. In SeZM
    this is the global->local edge rotation, so multiplying the edge direction
    by this matrix sends it to local ``+Z``.
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

    This function equals exactly 0 and 1 at the endpoints, and transitions with
    all derivatives vanishing there. It is used only to blend the two valid
    quaternion charts; the geometric constraint itself is still enforced by the
    charts. The interior denominator ``left + right`` is bounded below by
    ``exp(-2)`` on the clamped domain, so the dead branches of the ``where``
    never divide by zero (gradient-safe).
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

    ``q`` and ``-q`` represent the same spatial rotation. Aligning signs before
    the interpolation guarantees that the blended chart stays on the shorter
    great-circle segment in ``S^3``.
    """
    xp = array_api_compat.array_namespace(q0, q1, weight)
    dot = xp.sum(q0 * q1, axis=-1, keepdims=True)
    q1_aligned = xp.where(dot < 0.0, -q1, q1)
    blended = (1.0 - weight[..., None]) * q0 + weight[..., None] * q1_aligned
    return quaternion_normalize(blended, eps)


def _build_edge_quaternion_chart_pos_z(edge_unit: Any, eps: float) -> Any:
    """Quaternion chart that is exact away from the ``-Z`` pole."""
    xp = array_api_compat.array_namespace(edge_unit)
    x = edge_unit[..., 0]
    y = edge_unit[..., 1]
    z = edge_unit[..., 2]
    q = xp.stack([1.0 + z, y, -x, xp.zeros_like(x)], axis=-1)
    return quaternion_normalize(q, eps)


def _build_edge_quaternion_chart_neg_z(edge_unit: Any, eps: float) -> Any:
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

    The returned quaternion represents the global->local edge rotation, so
    applying its rotation matrix to the unit edge direction yields exactly
    ``(0, 0, 1)``. Two exact quaternion charts are used:

    - a ``+Z`` chart that is regular everywhere except the antipodal ``-Z`` pole;
    - a ``-Z`` chart that is regular everywhere except the antipodal ``+Z`` pole.

    Both charts encode the same edge-aligned local frame. A smooth ``C^inf``
    blend in the overlap region removes the hard pole switch while keeping the
    represented rotation on the correct quaternion branch.

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
        edge_len = safe_norm(edge_vec, eps)
    else:
        edge_len = xp.sqrt(edge_len * edge_len + eps * eps)
    edge_unit = edge_vec / edge_len
    q_pos = _build_edge_quaternion_chart_pos_z(edge_unit, eps)
    q_neg = _build_edge_quaternion_chart_neg_z(edge_unit, eps)
    blend = _smooth_step_cinf(0.5 * (edge_unit[..., 2] + 1.0))
    return quaternion_nlerp(q_neg, q_pos, blend, eps=eps)


def _factorial_table(n: int) -> np.ndarray:
    """Return ``[0!, 1!, ..., n!]`` in fp64 (iterative, matching pt bit-exactly)."""
    table = np.zeros(n + 1, dtype=np.float64)
    table[0] = 1.0
    for i in range(1, n + 1):
        table[i] = table[i - 1] * i
    return table


def _binomial(n: int, k: int, factorial: np.ndarray) -> float:
    """Evaluate ``C(n, k)`` from a precomputed factorial table."""
    if k < 0 or k > n:
        return 0.0
    return float(factorial[n] / (factorial[k] * factorial[n - k]))


class _CaseTables:
    """
    Plain numpy tables for one magnitude-ordered branch of the quaternion Wigner path.

    Mirrors pt ``CaseCoefficients`` (init-time constants only).
    """

    def __init__(self, n_primary: int, max_poly_len: int) -> None:
        self.coeff = np.zeros(n_primary, dtype=np.float64)
        self.horner = np.zeros((n_primary, max_poly_len), dtype=np.float64)
        self.poly_len = np.zeros(n_primary, dtype=np.int64)
        self.ra_exp = np.zeros(n_primary, dtype=np.float64)
        self.rb_exp = np.zeros(n_primary, dtype=np.float64)
        self.sign = np.zeros(n_primary, dtype=np.float64)
        # filled by _finalize_case_tables
        self.valid_mask: np.ndarray | None = None
        self.horner_step_mask: np.ndarray | None = None
        self.signed_coeff: np.ndarray | None = None


def _compute_case_coefficients(
    case: _CaseTables,
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

    The closed-form quaternion Wigner formula is reorganized so that only the
    ratio ``-(|Rb|/|Ra|)^2`` or ``-(|Ra|/|Rb|)^2`` enters the Horner chain.
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
        binom1 = _binomial(ell + mp, rho_min, factorial)
        binom2 = _binomial(ell - mp, ell - m - rho_min, factorial)
    else:
        binom1 = _binomial(ell + mp, ell - m - rho_min, factorial)
        binom2 = _binomial(ell - mp, rho_min, factorial)
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


def _finalize_case_tables(case: _CaseTables, max_poly_len: int) -> None:
    """Attach runtime-ready masks and fused coefficients for one Horner branch."""
    step_count = np.clip(case.poly_len - 1, 0, None)
    if max_poly_len > 1:
        horner_step_mask = (
            np.arange(max_poly_len - 1, dtype=np.int64)[None, :] < step_count[:, None]
        )
    else:
        horner_step_mask = np.zeros((case.poly_len.shape[0], 0), dtype=np.bool_)
    case.valid_mask = case.poly_len > 0
    case.horner_step_mask = horner_step_mask
    case.signed_coeff = case.sign * case.coeff


class _PolyTables:
    """
    Precomputed coefficient tables for the generic quaternion Wigner evaluator.

    Mirrors pt ``WignerPolynomialCoefficients`` (init-time numpy constants only).
    Only one half of each real block is stored explicitly. The remaining
    entries are reconstructed from the exact symmetry
    ``D^l_{-m',-m} = (-1)^(m' - m) * conj(D^l_{m',m})``.
    """

    def __init__(self, lmin: int, lmax: int) -> None:
        if lmin < 0:
            raise ValueError("`lmin` must be non-negative")
        if lmax < lmin:
            raise ValueError("`lmax` must be >= `lmin`")

        factorial = _factorial_table(2 * lmax + 1)
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

        self.lmin = lmin
        self.lmax = lmax
        self.size = size
        self.max_poly_len = max_poly_len
        self.n_primary = n_primary
        self.n_derived = n_derived

        self.primary_row = np.zeros(n_primary, dtype=np.int64)
        self.primary_col = np.zeros(n_primary, dtype=np.int64)
        self.mp_plus_m = np.zeros(n_primary, dtype=np.float64)
        self.m_minus_mp = np.zeros(n_primary, dtype=np.float64)
        self.diagonal_mask = np.zeros(n_primary, dtype=np.bool_)
        self.anti_diagonal_mask = np.zeros(n_primary, dtype=np.bool_)
        self.special_2m = np.zeros(n_primary, dtype=np.float64)
        self.anti_diag_sign = np.zeros(n_primary, dtype=np.float64)
        self.case1 = _CaseTables(n_primary, max_poly_len)
        self.case2 = _CaseTables(n_primary, max_poly_len)
        self.derived_row = np.zeros(n_derived, dtype=np.int64)
        self.derived_col = np.zeros(n_derived, dtype=np.int64)
        self.derived_primary_idx = np.zeros(n_derived, dtype=np.int64)
        self.derived_sign = np.zeros(n_derived, dtype=np.float64)

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
                    self.primary_row[primary_idx] = row
                    self.primary_col[primary_idx] = col
                    self.mp_plus_m[primary_idx] = mp + m
                    self.m_minus_mp[primary_idx] = m - mp
                    self.diagonal_mask[primary_idx] = mp == m
                    self.anti_diagonal_mask[primary_idx] = mp == -m
                    self.special_2m[primary_idx] = 2 * m
                    self.anti_diag_sign[primary_idx] = (-1) ** (ell - m)

                    sqrt_factor = math.sqrt(
                        float(factorial[ell + m] * factorial[ell - m])
                        / float(factorial[ell + mp] * factorial[ell - mp])
                    )
                    _compute_case_coefficients(
                        self.case1,
                        primary_idx,
                        ell,
                        mp,
                        m,
                        sqrt_factor,
                        factorial,
                        is_case1=True,
                    )
                    _compute_case_coefficients(
                        self.case2,
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

                    self.derived_row[derived_idx] = row
                    self.derived_col[derived_idx] = col
                    self.derived_primary_idx[derived_idx] = primary_map[
                        (block_start + (-mp + ell), block_start + (-m + ell))
                    ]
                    self.derived_sign[derived_idx] = (-1) ** (mp - m)
                    derived_idx += 1
            block_start += block_size

        _finalize_case_tables(self.case1, max_poly_len)
        _finalize_case_tables(self.case2, max_poly_len)

        # Functional scatter replacement: gather index mapping each flat
        # (row, col) of the packed (size, size) matrix to its source slot in
        # ``concat([primary, derived, zero_slot])``. Entries outside the
        # diagonal blocks point at the trailing zero slot.
        flat_to_src = np.full(size * size, n_primary + n_derived, dtype=np.int64)
        flat_to_src[self.primary_row * size + self.primary_col] = np.arange(
            n_primary, dtype=np.int64
        )
        flat_to_src[self.derived_row * size + self.derived_col] = n_primary + np.arange(
            n_derived, dtype=np.int64
        )
        self.flat_gather_idx = flat_to_src


def _build_complex_to_real_sh_block(ell: int) -> np.ndarray:
    """
    Build the complex-to-real basis transform for one ``ell`` block.

    The packed real basis follows the SeZM convention ``m = -ell, ..., +ell``
    inside each block. This unitary transform defines the real tesseral basis
    used by the packed ``D_full`` layout.
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


def _assemble_block_diagonal_real_basis(
    lmin: int, lmax: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble per-``ell`` real-basis blocks into one block-diagonal transform."""
    size = sum(2 * ell + 1 for ell in range(lmin, lmax + 1))
    U_re_full = np.zeros((size, size), dtype=np.float64)
    U_im_full = np.zeros((size, size), dtype=np.float64)
    offset = 0
    for ell in range(lmin, lmax + 1):
        U = _build_complex_to_real_sh_block(ell)
        block_size = 2 * ell + 1
        block_end = offset + block_size
        U_re_full[offset:block_end, offset:block_end] = U.real
        U_im_full[offset:block_end, offset:block_end] = U.imag
        offset = block_end
    return (
        U_re_full,
        U_im_full,
        np.ascontiguousarray(U_re_full.T),
        np.ascontiguousarray(U_im_full.T),
    )


def _vectorized_horner(
    xp: Any,
    ratio: Any,
    horner_coeffs: Any,
    horner_step_mask: Any,
) -> Any:
    """Evaluate many varying-length Horner chains in one batched loop."""
    n_batch = ratio.shape[0]
    n_elements = horner_coeffs.shape[0]
    result = xp.ones(
        (n_batch, n_elements),
        dtype=ratio.dtype,
        device=array_api_compat.device(ratio),
    )
    if horner_step_mask.shape[1] == 0:
        return result
    ratio = ratio[:, None]
    for i in range(horner_step_mask.shape[1]):
        new_result = 1.0 + result * (ratio * horner_coeffs[None, :, i])
        result = xp.where(horner_step_mask[None, :, i], new_result, result)
    return result


class WignerDCalculator(NativeOP):
    """
    Quaternion-driven Wigner-D blocks for the SeZM packed real spherical basis.

    Input quaternions represent the global->local edge rotation that sends the
    edge direction to local ``+Z``. The returned block-diagonal matrix keeps
    the packed SeZM real spherical-harmonics layout, so downstream code
    consumes ``D_full`` and ``Dt_full`` directly.

    Runtime structure:

    - ``l=0``: scalar identity block;
    - ``l=1``: direct quaternion -> Cartesian rotation -> real ``l=1`` block;
    - ``l>=2``: generic quaternion polynomial path with precomputed coefficient
      tables (the pt reference path that the pt monomial kernels are fitted to;
      see the module docstring).

    Parameters
    ----------
    lmax : int
        Maximum spherical-harmonics degree.
    eps : float
        Numerical floor used in quaternion normalization.
    precision : str
        Working floating-point precision of the returned blocks. The internal
        polynomial algebra is evaluated in fp64 (as in pt) before the result is
        cast back.
    """

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

        # l=1 block constants: permutation [1, 2, 0] is applied structurally in
        # _compute_l1_block; the sign pattern is a plain numpy constant.
        l1_sign = np.array([-1.0, -1.0, 1.0], dtype=np.float64)
        self.l1_sign_outer = np.outer(l1_sign, l1_sign)

        if self.lmax >= 2:
            self.poly_tables = _PolyTables(lmin=2, lmax=self.lmax)
            (
                self.poly_u_re,
                self.poly_u_im,
                self.poly_u_re_t,
                self.poly_u_im_t,
            ) = _assemble_block_diagonal_real_basis(2, self.lmax)

        # Functional block-diagonal assembly: gather index mapping each flat
        # (row, col) of D_full to its source slot in the concatenated value
        # vector [l0 ones (1), l1 block (9), packed l>=2 block (size^2),
        # trailing zero slot]. No __setitem__ on traced values is needed.
        n_l1 = 9 if self.lmax >= 1 else 0
        n_packed = (self.dim_full - 4) ** 2 if self.lmax >= 2 else 0
        zero_slot = 1 + n_l1 + n_packed
        full_idx = np.full(self.dim_full * self.dim_full, zero_slot, dtype=np.int64)
        full_idx[0] = 0  # D_full[:, 0, 0] = 1
        if self.lmax >= 1:
            for i in range(3):
                for j in range(3):
                    full_idx[(1 + i) * self.dim_full + (1 + j)] = 1 + 3 * i + j
        if self.lmax >= 2:
            packed_size = self.dim_full - 4
            for i in range(packed_size):
                for j in range(packed_size):
                    full_idx[(4 + i) * self.dim_full + (4 + j)] = (
                        1 + n_l1 + packed_size * i + j
                    )
        self.full_gather_idx = full_idx

    def call(self, edge_quaternion: Any) -> tuple[Any, Any]:
        """
        Build packed block-diagonal Wigner-D matrices from edge quaternions.

        Parameters
        ----------
        edge_quaternion : Array
            Unit quaternions with shape ``(E, 4)`` representing the
            global->local edge rotation.

        Returns
        -------
        tuple[Array, Array]
            ``(D_full, Dt_full)`` with shape ``(E, (lmax+1)^2, (lmax+1)^2)``.
        """
        xp = array_api_compat.array_namespace(edge_quaternion)
        dtype = get_xp_precision(xp, self.precision)
        device = array_api_compat.device(edge_quaternion)
        q = quaternion_normalize(
            xp.astype(edge_quaternion, dtype),
            eps=self.eps,
        )
        n_edge = q.shape[0]

        segments = [xp.ones((n_edge, 1), dtype=dtype, device=device)]
        if self.lmax >= 1:
            segments.append(
                xp.reshape(self._compute_l1_block(q, xp, dtype, device), (n_edge, 9))
            )
        if self.lmax >= 2:
            packed = self._compute_packed_blocks(q, xp, dtype, device)
            packed_size = self.dim_full - 4
            segments.append(xp.reshape(packed, (n_edge, packed_size * packed_size)))
        segments.append(xp.zeros((n_edge, 1), dtype=dtype, device=device))
        values = xp.concat(segments, axis=1)
        idx = xp_asarray_nodetach(xp, self.full_gather_idx, device=device)
        D_full = xp.reshape(
            xp.take(values, idx, axis=1),
            (n_edge, self.dim_full, self.dim_full),
        )
        Dt_full = xp.matrix_transpose(D_full)
        return D_full, Dt_full

    def forward_zonal(self, edge_quaternion: Any, lmin: int = 1) -> Any:
        """
        Build local ``m=0`` to global coupling for GIE.

        The returned layout matches the packed node rows for degrees
        ``lmin..lmax``: each degree contributes ``2l+1`` values in packed
        ``m=-l..l`` order. These values are equivalent to gathering
        ``Dt_full[:, row(l, m), col(l, 0)]`` from :meth:`call` over the same
        degree range.

        Parameters
        ----------
        edge_quaternion : Array
            Unit quaternions with shape ``(E, 4)`` representing the
            global->local edge rotation.
        lmin : int
            First degree to return.

        Returns
        -------
        Array
            Zonal coupling with shape ``(E, (lmax + 1) ** 2 - lmin ** 2)``.
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
        q = quaternion_normalize(
            xp.astype(edge_quaternion, dtype),
            eps=self.eps,
        )

        zonal_blocks = []
        if lmin <= 1 <= self.lmax:
            zonal_blocks.append(self._compute_l1_block(q, xp, dtype, device)[:, 1, :])
        if self.lmax >= 2:
            packed = self._compute_packed_blocks(q, xp, dtype, device)
            offset = 0
            for degree in range(2, self.lmax + 1):
                block_size = 2 * degree + 1
                block_end = offset + block_size
                if degree >= lmin:
                    zonal_blocks.append(packed[:, offset + degree, offset:block_end])
                offset = block_end
        return xp.concat(zonal_blocks, axis=1)

    def _compute_l1_block(self, q: Any, xp: Any, dtype: Any, device: Any) -> Any:
        """Compute the vector block directly from the Cartesian rotation matrix."""
        rot = quaternion_to_rotation_matrix(q)
        # row/column permutation [1, 2, 0], applied structurally (no gather)
        rot = xp.stack([rot[..., 1, :], rot[..., 2, :], rot[..., 0, :]], axis=-2)
        rot = xp.stack([rot[..., 1], rot[..., 2], rot[..., 0]], axis=-1)
        sign = xp_asarray_nodetach(xp, self.l1_sign_outer, dtype=dtype, device=device)
        return rot * sign

    def _compute_packed_blocks(self, q: Any, xp: Any, dtype: Any, device: Any) -> Any:
        """Evaluate the packed real Wigner blocks for ``l = 2..lmax``."""
        # Cayley-Klein pair: Ra = w - i z, Rb = y - i x (SeZM convention)
        ra_re = q[..., 0]
        ra_im = -q[..., 3]
        rb_re = q[..., 2]
        rb_im = -q[..., 1]
        D_re, D_im = self._wigner_d_matrix_realpair(
            ra_re, ra_im, rb_re, rb_im, xp, dtype, device
        )
        u_re = xp_asarray_nodetach(xp, self.poly_u_re, dtype=dtype, device=device)
        u_im = xp_asarray_nodetach(xp, self.poly_u_im, dtype=dtype, device=device)
        u_re_t = xp_asarray_nodetach(xp, self.poly_u_re_t, dtype=dtype, device=device)
        u_im_t = xp_asarray_nodetach(xp, self.poly_u_im_t, dtype=dtype, device=device)
        temp_re = xp.matmul(D_re, u_re_t) + xp.matmul(D_im, u_im_t)
        temp_im = xp.matmul(D_im, u_re_t) - xp.matmul(D_re, u_im_t)
        return xp.matmul(u_re, temp_re) - xp.matmul(u_im, temp_im)

    def _wigner_d_matrix_realpair(
        self,
        ra_re: Any,
        ra_im: Any,
        rb_re: Any,
        rb_im: Any,
        xp: Any,
        out_dtype: Any,
        device: Any,
    ) -> tuple[Any, Any]:
        """
        Evaluate the complex Wigner blocks in real/imaginary form.

        The runtime path uses only real arithmetic. The complex phase is
        represented by two real tensors, while the polynomial and magnitude
        algebra is evaluated in fp64 before the result is cast back to the
        requested output dtype. All denominators are eps-floored before any
        division (gradient-safe masked-denominator idiom, as in pt).
        """
        coeffs = self.poly_tables
        n_batch = ra_re.shape[0]
        f64 = xp.float64
        ra_re = xp.astype(ra_re, f64)
        ra_im = xp.astype(ra_im, f64)
        rb_re = xp.astype(rb_re, f64)
        rb_im = xp.astype(rb_im, f64)

        def cv(arr: np.ndarray) -> Any:  # constant table -> xp on input device
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

        for case, case_rows, ratio in (
            (
                coeffs.case1,
                use_case1,
                -(rb * rb) / (safe_ra * safe_ra),
            ),
            (
                coeffs.case2,
                use_case2,
                -(ra * ra) / (safe_rb * safe_rb),
            ),
        ):
            magnitude = self._compute_case_magnitude(
                xp,
                xp.where(case_rows, log_ra, xp.zeros_like(log_ra)),
                xp.where(case_rows, log_rb, xp.zeros_like(log_rb)),
                xp.where(case_rows, ratio, xp.zeros_like(ratio)),
                case,
                device,
            )
            val_re = magnitude * exp_phase_re
            val_im = magnitude * exp_phase_im
            mask = case_rows[:, None] & cv(case.valid_mask)[None, :]
            result_re = xp.where(mask, val_re, result_re)
            result_im = xp.where(mask, val_im, result_im)

        # Functional scatter into the dense packed matrix: derive the
        # symmetry-completed entries by gather, then place primary + derived
        # values with one precomputed take index (zero slot for off-block).
        derived_idx = cv(coeffs.derived_primary_idx)
        derived_sign = cv(coeffs.derived_sign)
        primary_re = xp.take(result_re, derived_idx, axis=1)
        primary_im = xp.take(result_im, derived_idx, axis=1)
        derived_re = derived_sign[None, :] * primary_re
        derived_im = -derived_sign[None, :] * primary_im
        zero_col = xp.zeros((n_batch, 1), dtype=f64, device=device)
        flat_idx = cv(coeffs.flat_gather_idx)
        D_re = xp.reshape(
            xp.take(
                xp.concat([result_re, derived_re, zero_col], axis=1), flat_idx, axis=1
            ),
            (n_batch, coeffs.size, coeffs.size),
        )
        D_im = xp.reshape(
            xp.take(
                xp.concat([result_im, derived_im, zero_col], axis=1), flat_idx, axis=1
            ),
            (n_batch, coeffs.size, coeffs.size),
        )
        return xp.astype(D_re, out_dtype), xp.astype(D_im, out_dtype)

    @staticmethod
    def _compute_case_magnitude(
        xp: Any,
        log_ra: Any,
        log_rb: Any,
        ratio: Any,
        case: _CaseTables,
        device: Any,
    ) -> Any:
        """Compute the real magnitude factor for one stable Horner branch."""
        horner_sum = _vectorized_horner(
            xp,
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
