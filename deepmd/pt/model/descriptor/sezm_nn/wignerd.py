# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Quaternion-based Wigner-D and edge-frame utilities for SeZM.

This module defines the quaternion helpers and Wigner-D evaluator used to
construct edge-aligned SO(3) rotation blocks in SeZM.
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

import torch
import torch.nn as nn

from deepmd.pt.utils import (
    env,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .utils import (
    nvtx_range,
)


class CaseCoefficients(nn.Module):
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
        coeff: torch.Tensor,
        horner: torch.Tensor,
        poly_len: torch.Tensor,
        ra_exp: torch.Tensor,
        rb_exp: torch.Tensor,
        sign: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("coeff", coeff, persistent=True)
        self.register_buffer("horner", horner, persistent=True)
        self.register_buffer("poly_len", poly_len, persistent=True)
        self.register_buffer("ra_exp", ra_exp, persistent=True)
        self.register_buffer("rb_exp", rb_exp, persistent=True)
        self.register_buffer("sign", sign, persistent=True)


class WignerPolynomialCoefficients(nn.Module):
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
        primary_row: torch.Tensor,
        primary_col: torch.Tensor,
        case1: CaseCoefficients,
        case2: CaseCoefficients,
        mp_plus_m: torch.Tensor,
        m_minus_mp: torch.Tensor,
        diagonal_mask: torch.Tensor,
        anti_diagonal_mask: torch.Tensor,
        special_2m: torch.Tensor,
        anti_diag_sign: torch.Tensor,
        derived_row: torch.Tensor,
        derived_col: torch.Tensor,
        derived_primary_idx: torch.Tensor,
        derived_sign: torch.Tensor,
    ) -> None:
        super().__init__()
        self.lmin = int(lmin)
        self.lmax = int(lmax)
        self.size = int(size)
        self.max_poly_len = int(max_poly_len)
        self.n_primary = int(n_primary)
        self.n_derived = int(n_derived)

        self.register_buffer("primary_row", primary_row, persistent=True)
        self.register_buffer("primary_col", primary_col, persistent=True)
        self.case1 = case1
        self.case2 = case2
        self.register_buffer("mp_plus_m", mp_plus_m, persistent=True)
        self.register_buffer("m_minus_mp", m_minus_mp, persistent=True)
        self.register_buffer("diagonal_mask", diagonal_mask, persistent=True)
        self.register_buffer("anti_diagonal_mask", anti_diagonal_mask, persistent=True)
        self.register_buffer("special_2m", special_2m, persistent=True)
        self.register_buffer("anti_diag_sign", anti_diag_sign, persistent=True)
        self.register_buffer("derived_row", derived_row, persistent=True)
        self.register_buffer("derived_col", derived_col, persistent=True)
        self.register_buffer(
            "derived_primary_idx", derived_primary_idx, persistent=True
        )
        self.register_buffer("derived_sign", derived_sign, persistent=True)


class WignerSmallOrderCoefficients(nn.Module):
    """
    Precomputed low-order quaternion polynomial kernels in the SeZM packed basis.

    The tensors in this container provide the specialized ``l=2`` and ``l=3,4``
    kernels used by the hybrid Wigner runtime:
    - ``C_l2`` stores the degree-4 tensor-contraction coefficients;
    - ``C_l3`` / ``C_l4`` store flattened monomial coefficient matrices;
    - ``C_combined_l3l4`` lifts the ``l=3`` basis to degree 8 and stacks it with
      ``l=4`` so both blocks can be produced by one matrix multiply;
    - ``exp_l3`` / ``exp_l4`` store the monomial exponent tables used by the runtime
      gather/prod path.
    """

    def __init__(
        self,
        *,
        C_l2: torch.Tensor,
        C_l3: torch.Tensor,
        C_l4: torch.Tensor,
        C_combined_l3l4: torch.Tensor,
        exp_l3: torch.Tensor,
        exp_l4: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("C_l2", C_l2, persistent=True)
        self.register_buffer("C_l3", C_l3, persistent=True)
        self.register_buffer("C_l4", C_l4, persistent=True)
        self.register_buffer("C_combined_l3l4", C_combined_l3l4, persistent=True)
        self.register_buffer("exp_l3", exp_l3, persistent=True)
        self.register_buffer("exp_l4", exp_l4, persistent=True)


def _safe_norm_nd(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute an ``L2`` norm with smooth epsilon regularization."""
    in_dtype = x.dtype
    if in_dtype in (torch.float16, torch.bfloat16):
        x = x.float()
    norm = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + eps * eps)
    return norm.to(dtype=in_dtype)


def quaternion_normalize(q: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Normalize quaternions with a differentiable epsilon floor."""
    return q / _safe_norm_nd(q, eps)


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product for batched quaternions in ``(w, x, y, z)`` order."""
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert unit quaternions to 3x3 rotation matrices.

    The returned matrix is the active rotation represented by ``q``. In SeZM this is
    the global->local edge rotation, so multiplying the edge direction by this matrix
    sends it to local ``+Z``.
    """
    w, x, y, z = q.unbind(dim=-1)
    x2 = x * x
    y2 = y * y
    z2 = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return torch.stack(
        [
            torch.stack(
                [1.0 - 2.0 * (y2 + z2), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                dim=-1,
            ),
            torch.stack(
                [2.0 * (xy + wz), 1.0 - 2.0 * (x2 + z2), 2.0 * (yz - wx)],
                dim=-1,
            ),
            torch.stack(
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (x2 + y2)],
                dim=-1,
            ),
        ],
        dim=-2,
    )


def quaternion_z_rotation(gamma: torch.Tensor) -> torch.Tensor:
    """
    Create quaternions for a rotation about the local ``+Z`` axis.

    Parameters
    ----------
    gamma
        Roll angles in radians with shape ``(E,)``.

    Returns
    -------
    torch.Tensor
        Quaternions with shape ``(E, 4)`` in ``(w, x, y, z)`` order.
    """
    half_gamma = 0.5 * gamma
    w = torch.cos(half_gamma)
    x = torch.zeros_like(gamma)
    y = torch.zeros_like(gamma)
    z = torch.sin(half_gamma)
    return torch.stack([w, x, y, z], dim=-1)


def _smooth_step_cinf(x: torch.Tensor) -> torch.Tensor:
    """
    Smooth ``C^inf`` step on ``[0, 1]``.

    This function equals exactly 0 and 1 at the endpoints, and transitions with all
    derivatives vanishing there. It is used only to blend the two valid quaternion
    charts; the geometric constraint itself is still enforced by the charts.
    """
    x_clamped = x.clamp(0.0, 1.0)
    eps = torch.finfo(x_clamped.dtype).eps
    left = torch.exp(-1.0 / torch.clamp(x_clamped, min=eps))
    right = torch.exp(-1.0 / torch.clamp(1.0 - x_clamped, min=eps))
    interior = left / (left + right)
    return torch.where(
        x_clamped <= 0.0,
        torch.zeros_like(x_clamped),
        torch.where(x_clamped >= 1.0, torch.ones_like(x_clamped), interior),
    )


def quaternion_nlerp(
    q0: torch.Tensor,
    q1: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Normalized linear interpolation on the shortest quaternion arc.

    ``q`` and ``-q`` represent the same spatial rotation. Aligning signs before the
    interpolation guarantees that the blended chart stays on the shorter great-circle
    segment in ``S^3``.
    """
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    q1_aligned = torch.where(dot < 0.0, -q1, q1)
    blended = (1.0 - weight.unsqueeze(-1)) * q0 + weight.unsqueeze(-1) * q1_aligned
    return quaternion_normalize(blended, eps)


def _build_edge_quaternion_chart_pos_z(
    edge_unit: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Quaternion chart that is exact away from the ``-Z`` pole."""
    x = edge_unit[..., 0]
    y = edge_unit[..., 1]
    z = edge_unit[..., 2]
    q = torch.stack([1.0 + z, y, -x, torch.zeros_like(x)], dim=-1)
    return quaternion_normalize(q, eps)


def _build_edge_quaternion_chart_neg_z(
    edge_unit: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Quaternion chart that is exact away from the ``+Z`` pole."""
    x = edge_unit[..., 0]
    y = edge_unit[..., 1]
    z = edge_unit[..., 2]
    q = torch.stack([-x, torch.zeros_like(x), 1.0 - z, y], dim=-1)
    return quaternion_normalize(q, eps)


def build_edge_quaternion(
    edge_vec: torch.Tensor,
    *,
    edge_len: torch.Tensor | None = None,
    eps: float = 1e-7,
) -> torch.Tensor:
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
    torch.Tensor
        Unit quaternions with shape ``(E, 4)`` in ``(w, x, y, z)`` order.
    """
    if edge_len is None:
        edge_len = _safe_norm_nd(edge_vec, eps)
    else:
        edge_len = torch.sqrt(edge_len * edge_len + eps * eps)
    edge_unit = edge_vec / edge_len
    q_pos = _build_edge_quaternion_chart_pos_z(edge_unit, eps)
    q_neg = _build_edge_quaternion_chart_neg_z(edge_unit, eps)
    blend = _smooth_step_cinf(0.5 * (edge_unit[..., 2] + 1.0))
    return quaternion_nlerp(q_neg, q_pos, blend, eps=eps)


class WignerDCalculator(nn.Module):
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
    - ``l>=5``: generic quaternion polynomial path with precomputed coefficient tables.
    """

    _SMALL_ORDER_CACHE_CPU_FP64: ClassVar[dict[str, torch.Tensor] | None] = None

    def __init__(
        self,
        lmax: int,
        *,
        eps: float = 1e-7,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        if self.lmax < 0:
            raise ValueError("`lmax` must be non-negative")
        self.dtype = dtype
        self.device = env.DEVICE
        self.eps = float(eps)
        self.dim_full = (self.lmax + 1) ** 2
        self.poly_lmin = 5
        self.poly_offset = self.poly_lmin * self.poly_lmin

        self.register_buffer(
            "l1_perm",
            torch.tensor([1, 2, 0], dtype=torch.int64, device=self.device),
            persistent=True,
        )
        l1_sign = torch.tensor([-1.0, -1.0, 1.0], dtype=self.dtype, device=self.device)
        self.register_buffer(
            "l1_sign_outer",
            torch.outer(l1_sign, l1_sign),
            persistent=True,
        )

        if self.lmax >= 2:
            self.small_order_kernels = self._build_small_order_kernels(
                dtype=self.dtype,
                device=self.device,
            )

        if self.lmax >= self.poly_lmin:
            coeffs = self._precompute_wigner_coefficients(
                self.lmax,
                dtype=torch.float64,
                device=torch.device("cpu"),
                lmin=self.poly_lmin,
            )
            self.poly_coeffs = coeffs.to(device=self.device)
            blocks = self._precompute_real_basis_blocks(
                lmin=self.poly_lmin,
                lmax=self.lmax,
                dtype=torch.float64,
                device=torch.device("cpu"),
            )
            U_re, U_im, U_re_t, U_im_t = self._assemble_block_diagonal_real_basis(
                blocks
            )
            self.register_buffer(
                "poly_u_re",
                U_re.to(device=self.device, dtype=self.dtype),
                persistent=True,
            )
            self.register_buffer(
                "poly_u_im",
                U_im.to(device=self.device, dtype=self.dtype),
                persistent=True,
            )
            self.register_buffer(
                "poly_u_re_t",
                U_re_t.to(device=self.device, dtype=self.dtype),
                persistent=True,
            )
            self.register_buffer(
                "poly_u_im_t",
                U_im_t.to(device=self.device, dtype=self.dtype),
                persistent=True,
            )

    def forward(
        self, edge_quaternion: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build packed block-diagonal Wigner-D matrices from edge quaternions.

        Parameters
        ----------
        edge_quaternion
            Unit quaternions with shape ``(E, 4)`` representing the global->local
            edge rotation.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(D_full, Dt_full)`` with shape ``(E, (lmax+1)^2, (lmax+1)^2)``.
        """
        edge_quaternion = quaternion_normalize(
            edge_quaternion.to(dtype=self.dtype),
            eps=self.eps,
        )
        n_edge = edge_quaternion.shape[0]
        D_full = torch.zeros(
            n_edge,
            self.dim_full,
            self.dim_full,
            dtype=edge_quaternion.dtype,
            device=edge_quaternion.device,
        )
        D_full[:, 0, 0] = 1.0

        if self.lmax >= 1:
            with nvtx_range("WignerD/l1"):
                D_full[:, 1:4, 1:4] = self._compute_l1_block(edge_quaternion)

        if self.lmax >= 2:
            with nvtx_range("WignerD/l2"):
                D_full[:, 4:9, 4:9] = self._compute_l2_block(edge_quaternion)

        if self.lmax >= 3:
            if self.lmax >= 4:
                with nvtx_range("WignerD/l3l4"):
                    D_l3, D_l4 = self._compute_l3l4_blocks(edge_quaternion)
                    D_full[:, 9:16, 9:16] = D_l3
                    D_full[:, 16:25, 16:25] = D_l4
            else:
                with nvtx_range("WignerD/l3"):
                    D_full[:, 9:16, 9:16] = self._compute_l3_block(edge_quaternion)

        if self.lmax >= self.poly_lmin:
            with nvtx_range("WignerD/polynomial"):
                ra_re, ra_im, rb_re, rb_im = self._quaternion_to_ra_rb_real(
                    edge_quaternion
                )
                D_re, D_im = self._wigner_d_matrix_realpair(
                    ra_re,
                    ra_im,
                    rb_re,
                    rb_im,
                    self.poly_coeffs,
                    dtype=self.dtype,
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
                D_full[:, self.poly_offset :, self.poly_offset :] = D_poly

        Dt_full = D_full.transpose(-1, -2).contiguous()
        return D_full, Dt_full

    @classmethod
    def _get_small_order_cache_cpu_fp64(cls) -> dict[str, torch.Tensor]:
        """Generate the low-order kernel coefficients once per process on CPU fp64."""
        if cls._SMALL_ORDER_CACHE_CPU_FP64 is None:
            cls._SMALL_ORDER_CACHE_CPU_FP64 = cls._generate_small_order_cache_cpu_fp64()
        return cls._SMALL_ORDER_CACHE_CPU_FP64

    @classmethod
    def _build_small_order_kernels(
        cls,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> WignerSmallOrderCoefficients:
        """Instantiate the specialized ``l=2,3,4`` kernels on the requested device/dtype."""
        cache = cls._get_small_order_cache_cpu_fp64()
        return WignerSmallOrderCoefficients(
            C_l2=cache["C_l2"].to(device=device, dtype=dtype),
            C_l3=cache["C_l3"].to(device=device, dtype=dtype),
            C_l4=cache["C_l4"].to(device=device, dtype=dtype),
            C_combined_l3l4=cache["C_combined_l3l4"].to(device=device, dtype=dtype),
            exp_l3=cache["exp_l3"].to(device=device),
            exp_l4=cache["exp_l4"].to(device=device),
        )

    @classmethod
    def _generate_small_order_cache_cpu_fp64(cls) -> dict[str, torch.Tensor]:
        """
        Generate the low-order kernel coefficients from the generic SeZM reference path.

        The coefficients are exact module constants. They are solved once in fp64 on CPU,
        validated against the generic quaternion polynomial evaluator, and then reused by
        every `WignerDCalculator` instance.
        """
        dtype = torch.float64
        device = torch.device("cpu")
        generator = torch.Generator()
        generator.manual_seed(20260404)

        q_fit = torch.randn(2048, 4, dtype=dtype, device=device, generator=generator)
        q_fit = quaternion_normalize(q_fit, eps=torch.finfo(dtype).eps)
        ref_blocks = cls._compute_generic_reference_blocks(
            q_fit, lmax=4, dtype=dtype, device=device
        )

        monomials_l2 = cls._generate_monomials(4, 4)
        monomials_l3 = cls._generate_monomials(4, 6)
        monomials_l4 = cls._generate_monomials(4, 8)
        exp_l2 = cls._monomials_to_exponent_tensor(monomials_l2, device=device)
        exp_l3 = cls._monomials_to_exponent_tensor(monomials_l3, device=device)
        exp_l4 = cls._monomials_to_exponent_tensor(monomials_l4, device=device)

        C_l2_flat = cls._solve_monomial_coefficients(
            q_fit,
            ref_blocks[2],
            exp_l2,
        )
        C_l3 = cls._solve_monomial_coefficients(q_fit, ref_blocks[3], exp_l3)
        C_l4 = cls._solve_monomial_coefficients(q_fit, ref_blocks[4], exp_l4)
        C_l2 = cls._build_l2_contraction_tensor(C_l2_flat, monomials_l2)
        C_combined_l3l4 = cls._build_combined_l3l4(
            C_l3, C_l4, monomials_l3, monomials_l4
        )

        q_val = torch.randn(256, 4, dtype=dtype, device=device, generator=generator)
        q_val = quaternion_normalize(q_val, eps=torch.finfo(dtype).eps)
        ref_val = cls._compute_generic_reference_blocks(
            q_val, lmax=4, dtype=dtype, device=device
        )
        test_val = cls._evaluate_small_order_blocks(
            q_val,
            C_l2=C_l2,
            C_l3=C_l3,
            C_l4=C_l4,
            exp_l3=exp_l3,
            exp_l4=exp_l4,
        )
        thresholds = {2: 1e-10, 3: 1e-10, 4: 1e-10}
        for ell in (2, 3, 4):
            err = (test_val[ell] - ref_val[ell]).abs().max().item()
            if err > thresholds[ell]:
                raise RuntimeError(
                    f"Failed to generate stable SeZM Wigner coefficients for l={ell}: max_err={err}"
                )

        return {
            "C_l2": C_l2,
            "C_l3": C_l3,
            "C_l4": C_l4,
            "C_combined_l3l4": C_combined_l3l4,
            "exp_l3": exp_l3,
            "exp_l4": exp_l4,
        }

    @classmethod
    def _compute_generic_reference_blocks(
        cls,
        edge_quaternion: torch.Tensor,
        *,
        lmax: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> dict[int, torch.Tensor]:
        """Evaluate the generic SeZM polynomial path and extract the ``l=2,3,4`` blocks."""
        coeffs = cls._precompute_wigner_coefficients(
            lmax,
            dtype=dtype,
            device=device,
            lmin=2,
        )
        blocks = cls._precompute_real_basis_blocks(
            lmin=2,
            lmax=lmax,
            dtype=dtype,
            device=device,
        )
        ra_re, ra_im, rb_re, rb_im = cls._quaternion_to_ra_rb_real(edge_quaternion)
        D_re, D_im = cls._wigner_d_matrix_realpair(
            ra_re,
            ra_im,
            rb_re,
            rb_im,
            coeffs,
            dtype=dtype,
        )
        D_ref = cls._wigner_d_pair_to_real(
            D_re,
            D_im,
            blocks,
            lmax=lmax,
            lmin=2,
        )
        return {
            2: D_ref[:, 0:5, 0:5],
            3: D_ref[:, 5:12, 5:12],
            4: D_ref[:, 12:21, 12:21],
        }

    @classmethod
    def _solve_monomial_coefficients(
        cls,
        edge_quaternion: torch.Tensor,
        D_block: torch.Tensor,
        monomial_exponents: torch.Tensor,
    ) -> torch.Tensor:
        """Solve the flattened monomial coefficient matrix for one low-order block."""
        max_power = int(monomial_exponents.sum(dim=1).max().item())
        powers = cls._precompute_powers(edge_quaternion, max_power)
        M = cls._build_monomial_matrix(powers, monomial_exponents)
        Y = D_block.reshape(edge_quaternion.shape[0], -1)
        return torch.linalg.lstsq(M, Y).solution.transpose(0, 1).contiguous()

    @staticmethod
    def _build_l2_contraction_tensor(
        C_l2_flat: torch.Tensor,
        monomials: list[tuple[int, int, int, int]],
    ) -> torch.Tensor:
        """Expand degree-4 monomial coefficients into the symmetric einsum tensor form."""
        C_l2 = torch.zeros(
            5, 5, 4, 4, 4, 4, dtype=C_l2_flat.dtype, device=C_l2_flat.device
        )
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

    @classmethod
    def _evaluate_small_order_blocks(
        cls,
        edge_quaternion: torch.Tensor,
        *,
        C_l2: torch.Tensor,
        C_l3: torch.Tensor,
        C_l4: torch.Tensor,
        exp_l3: torch.Tensor,
        exp_l4: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        """Evaluate the specialized ``l=2,3,4`` kernels for validation and caching."""
        q2 = edge_quaternion.unsqueeze(-1) * edge_quaternion.unsqueeze(-2)
        q4 = q2.unsqueeze(-1).unsqueeze(-1) * q2.unsqueeze(-3).unsqueeze(-3)
        D_l2 = torch.einsum("nabcd,ijabcd->nij", q4, C_l2)

        powers6 = cls._precompute_powers(edge_quaternion, 6)
        M3 = cls._build_monomial_matrix(powers6, exp_l3)
        D_l3 = torch.matmul(M3, C_l3.transpose(0, 1)).view(
            edge_quaternion.shape[0], 7, 7
        )

        powers8 = cls._precompute_powers(edge_quaternion, 8)
        M4 = cls._build_monomial_matrix(powers8, exp_l4)
        D_l4 = torch.matmul(M4, C_l4.transpose(0, 1)).view(
            edge_quaternion.shape[0], 9, 9
        )
        return {
            2: D_l2,
            3: D_l3,
            4: D_l4,
        }

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
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Convert monomial tuples to an ``int64`` exponent table."""
        return torch.tensor(monomials, dtype=torch.int64, device=device)

    @staticmethod
    def _build_combined_l3l4(
        C_l3: torch.Tensor,
        C_l4: torch.Tensor,
        monomials_l3: list[tuple[int, int, int, int]],
        monomials_l4: list[tuple[int, int, int, int]],
    ) -> torch.Tensor:
        """Lift the ``l=3`` basis to degree 8 and stack it with the ``l=4`` basis."""
        mono8_to_idx = {mono: idx for idx, mono in enumerate(monomials_l4)}
        C_l3_lifted = torch.zeros(
            C_l3.shape[0],
            len(monomials_l4),
            dtype=C_l3.dtype,
            device=C_l3.device,
        )
        for j, (a, b, c, d) in enumerate(monomials_l3):
            for mono8 in (
                (a + 2, b, c, d),
                (a, b + 2, c, d),
                (a, b, c + 2, d),
                (a, b, c, d + 2),
            ):
                C_l3_lifted[:, mono8_to_idx[mono8]] += C_l3[:, j]
        return torch.cat([C_l3_lifted, C_l4], dim=0)

    @staticmethod
    def _precompute_powers(
        q: torch.Tensor,
        max_power: int,
    ) -> torch.Tensor:
        """Precompute powers ``q_i^k`` as a dense table with shape ``(4, max_power+1, E)``."""
        components = q.transpose(0, 1)
        if max_power == 0:
            return torch.ones(4, 1, q.shape[0], dtype=q.dtype, device=q.device)
        repeated = components.unsqueeze(1).expand(4, max_power, q.shape[0])
        positive_powers = torch.cumprod(repeated, dim=1)
        return torch.cat(
            [
                torch.ones(4, 1, q.shape[0], dtype=q.dtype, device=q.device),
                positive_powers,
            ],
            dim=1,
        )

    @staticmethod
    def _build_monomial_matrix(
        powers: torch.Tensor,
        monomial_exponents: torch.Tensor,
    ) -> torch.Tensor:
        """Assemble the monomial design matrix for one fixed degree by gather/prod."""
        gather_idx = (
            monomial_exponents.transpose(0, 1)
            .unsqueeze(-1)
            .expand(
                4,
                monomial_exponents.shape[0],
                powers.shape[-1],
            )
        )
        selected = torch.gather(powers, 1, gather_idx)
        return selected.prod(dim=0).transpose(0, 1).contiguous()

    def _compute_l1_block(self, edge_quaternion: torch.Tensor) -> torch.Tensor:
        """Compute the vector block directly from the Cartesian rotation matrix."""
        rot_mat = quaternion_to_rotation_matrix(edge_quaternion)
        rot_perm = rot_mat.index_select(-2, self.l1_perm).index_select(-1, self.l1_perm)
        return rot_perm * self.l1_sign_outer

    def _compute_l2_block(self, edge_quaternion: torch.Tensor) -> torch.Tensor:
        """Compute the ``l=2`` block from the degree-4 quaternion contraction."""
        q2 = edge_quaternion.unsqueeze(-1) * edge_quaternion.unsqueeze(-2)
        q4 = q2.unsqueeze(-1).unsqueeze(-1) * q2.unsqueeze(-3).unsqueeze(-3)
        return torch.einsum(
            "nabcd,ijabcd->nij",
            q4,
            self.small_order_kernels.C_l2,
        )

    def _compute_l3_block(self, edge_quaternion: torch.Tensor) -> torch.Tensor:
        """Compute the ``l=3`` block from the dedicated degree-6 monomial kernel."""
        powers = self._precompute_powers(edge_quaternion, 6)
        monomials = self._build_monomial_matrix(
            powers,
            self.small_order_kernels.exp_l3,
        )
        D_flat = torch.matmul(
            monomials,
            self.small_order_kernels.C_l3.transpose(0, 1),
        )
        return D_flat.view(edge_quaternion.shape[0], 7, 7)

    def _compute_l3l4_blocks(
        self,
        edge_quaternion: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the ``l=3`` and ``l=4`` blocks from one shared degree-8 kernel."""
        powers = self._precompute_powers(edge_quaternion, 8)
        monomials = self._build_monomial_matrix(
            powers,
            self.small_order_kernels.exp_l4,
        )
        D_flat = torch.matmul(
            monomials,
            self.small_order_kernels.C_combined_l3l4.transpose(0, 1),
        )
        D_l3 = D_flat[:, :49].view(edge_quaternion.shape[0], 7, 7)
        D_l4 = D_flat[:, 49:].view(edge_quaternion.shape[0], 9, 9)
        return D_l3, D_l4

    @staticmethod
    def _factorial_table(
        n: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """Return ``[0!, 1!, ..., n!]`` in the requested dtype/device."""
        table = torch.zeros(n + 1, dtype=dtype, device=device)
        table[0] = 1.0
        for i in range(1, n + 1):
            table[i] = table[i - 1] * i
        return table

    @staticmethod
    def _binomial(n: int, k: int, factorial: torch.Tensor) -> float:
        """Evaluate ``C(n, k)`` from a precomputed factorial table."""
        if k < 0 or k > n:
            return 0.0
        return float(factorial[n] / (factorial[k] * factorial[n - k]))

    @staticmethod
    def _allocate_case_coeffs(
        n_primary: int,
        max_poly_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> CaseCoefficients:
        """Allocate one branch of Horner tables for the quaternion Wigner evaluator."""
        return CaseCoefficients(
            coeff=torch.zeros(n_primary, dtype=dtype, device=device),
            horner=torch.zeros(n_primary, max_poly_len, dtype=dtype, device=device),
            poly_len=torch.zeros(n_primary, dtype=torch.int64, device=device),
            ra_exp=torch.zeros(n_primary, dtype=dtype, device=device),
            rb_exp=torch.zeros(n_primary, dtype=dtype, device=device),
            sign=torch.zeros(n_primary, dtype=dtype, device=device),
        )

    @staticmethod
    def _compute_case_coefficients(
        case: CaseCoefficients,
        idx: int,
        ell: int,
        mp: int,
        m: int,
        sqrt_factor: float,
        factorial: torch.Tensor,
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
        step_count = torch.clamp(case.poly_len - 1, min=0)
        if max_poly_len > 1:
            horner_step_mask = torch.arange(
                max_poly_len - 1,
                dtype=case.poly_len.dtype,
                device=case.poly_len.device,
            ).unsqueeze(0) < step_count.unsqueeze(1)
        else:
            horner_step_mask = torch.zeros(
                case.poly_len.shape[0],
                0,
                dtype=torch.bool,
                device=case.poly_len.device,
            )
        case.register_buffer("valid_mask", case.poly_len > 0, persistent=True)
        case.register_buffer("horner_step_mask", horner_step_mask, persistent=True)
        case.register_buffer("signed_coeff", case.sign * case.coeff, persistent=True)

    @staticmethod
    def _vectorized_horner(
        ratio: torch.Tensor,
        horner_coeffs: torch.Tensor,
        horner_step_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate many varying-length Horner chains in one batched loop."""
        n_batch = ratio.shape[0]
        n_elements = horner_coeffs.shape[0]
        result = torch.ones(n_batch, n_elements, dtype=ratio.dtype, device=ratio.device)
        if horner_step_mask.shape[1] == 0:
            return result
        ratio = ratio.unsqueeze(1).expand(n_batch, n_elements)
        for i in range(horner_step_mask.shape[1]):
            new_result = 1.0 + result * (ratio * horner_coeffs[:, i].unsqueeze(0))
            result = torch.where(
                horner_step_mask[:, i].unsqueeze(0), new_result, result
            )
        return result

    @staticmethod
    def _compute_case_magnitude(
        log_ra: torch.Tensor,
        log_rb: torch.Tensor,
        ratio: torch.Tensor,
        case: CaseCoefficients,
    ) -> torch.Tensor:
        """Compute the real magnitude factor for one stable Horner branch."""
        horner_sum = WignerDCalculator._vectorized_horner(
            ratio,
            case.horner,
            case.horner_step_mask,
        )
        ra_powers = torch.exp(torch.outer(log_ra, case.ra_exp))
        rb_powers = torch.exp(torch.outer(log_rb, case.rb_exp))
        magnitude = case.signed_coeff.unsqueeze(0) * ra_powers * rb_powers
        return magnitude * horner_sum

    @staticmethod
    def _scatter_primary_to_matrix(
        result: torch.Tensor,
        D: torch.Tensor,
        coeffs: WignerPolynomialCoefficients,
    ) -> None:
        """Scatter the explicitly stored primary entries into the dense block matrix."""
        D[:, coeffs.primary_row, coeffs.primary_col] = result

    @staticmethod
    def _build_complex_to_real_sh_block(
        ell: int,
        *,
        dtype: torch.dtype = torch.complex128,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build the complex-to-real basis transform for one ``ell`` block.

        The packed real basis follows the SeZM convention
        ``m = -ell, ..., +ell`` inside each block. This unitary transform defines the
        real tesseral basis used by the packed ``D_full`` layout.
        """
        size = 2 * ell + 1
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        U = torch.zeros(size, size, dtype=dtype, device=device)
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
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Precompute complex-to-real basis transforms for ``ell in [lmin, lmax]``."""
        if lmin > lmax:
            return []
        complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        blocks: list[tuple[torch.Tensor, torch.Tensor]] = []
        for ell in range(lmin, lmax + 1):
            U = WignerDCalculator._build_complex_to_real_sh_block(
                ell,
                dtype=complex_dtype,
                device=device,
            )
            blocks.append((U.real.to(dtype=dtype), U.imag.to(dtype=dtype)))
        return blocks

    @staticmethod
    def _assemble_block_diagonal_real_basis(
        U_blocks: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assemble per-``ell`` real-basis blocks into one block-diagonal transform."""
        if not U_blocks:
            empty = torch.zeros(
                0,
                0,
                dtype=env.GLOBAL_PT_FLOAT_PRECISION,
                device=env.DEVICE,
            )
            return empty, empty, empty, empty

        size = sum(U_re.shape[0] for U_re, _ in U_blocks)
        dtype = U_blocks[0][0].dtype
        device = U_blocks[0][0].device
        U_re_full = torch.zeros(size, size, dtype=dtype, device=device)
        U_im_full = torch.zeros(size, size, dtype=dtype, device=device)
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
            U_re_full.transpose(-1, -2).contiguous(),
            U_im_full.transpose(-1, -2).contiguous(),
        )

    @staticmethod
    def _quaternion_to_ra_rb_real(
        q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        dtype: torch.dtype,
        device: torch.device,
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

        factorial = WignerDCalculator._factorial_table(2 * lmax + 1, dtype, device)
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

        primary_row = torch.zeros(n_primary, dtype=torch.int64, device=device)
        primary_col = torch.zeros(n_primary, dtype=torch.int64, device=device)
        mp_plus_m = torch.zeros(n_primary, dtype=dtype, device=device)
        m_minus_mp = torch.zeros(n_primary, dtype=dtype, device=device)
        diagonal_mask = torch.zeros(n_primary, dtype=torch.bool, device=device)
        anti_diagonal_mask = torch.zeros(n_primary, dtype=torch.bool, device=device)
        special_2m = torch.zeros(n_primary, dtype=dtype, device=device)
        anti_diag_sign = torch.zeros(n_primary, dtype=dtype, device=device)
        case1 = WignerDCalculator._allocate_case_coeffs(
            n_primary,
            max_poly_len,
            dtype,
            device,
        )
        case2 = WignerDCalculator._allocate_case_coeffs(
            n_primary,
            max_poly_len,
            dtype,
            device,
        )
        derived_row = torch.zeros(n_derived, dtype=torch.int64, device=device)
        derived_col = torch.zeros(n_derived, dtype=torch.int64, device=device)
        derived_primary_idx = torch.zeros(n_derived, dtype=torch.int64, device=device)
        derived_sign = torch.zeros(n_derived, dtype=dtype, device=device)

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

        return WignerPolynomialCoefficients(
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

    @staticmethod
    def _wigner_d_matrix_realpair(
        ra_re: torch.Tensor,
        ra_im: torch.Tensor,
        rb_re: torch.Tensor,
        rb_im: torch.Tensor,
        coeffs: WignerPolynomialCoefficients,
        *,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the complex Wigner blocks in real/imaginary form.

        The runtime path uses only real arithmetic. The complex phase is represented by
        two real tensors, while the polynomial and magnitude algebra is evaluated in
        ``fp64`` before the result is cast back to the requested output dtype.
        """
        n_batch = ra_re.shape[0]
        output_dtype = ra_re.dtype if dtype is None else dtype
        if coeffs.size == 0:
            zeros = torch.zeros(n_batch, 0, 0, dtype=output_dtype, device=ra_re.device)
            return zeros, zeros

        ra_re = ra_re.to(torch.float64)
        ra_im = ra_im.to(torch.float64)
        rb_re = rb_re.to(torch.float64)
        rb_im = rb_im.to(torch.float64)
        if (
            coeffs.case1.coeff.dtype != torch.float64
            or coeffs.primary_row.device != ra_re.device
        ):
            coeffs = coeffs.to(device=ra_re.device, dtype=torch.float64)

        dtype = torch.float64
        device = ra_re.device

        eps = torch.finfo(dtype).eps
        eps_sq = eps * eps
        ra_sq = ra_re * ra_re + ra_im * ra_im
        rb_sq = rb_re * rb_re + rb_im * rb_im
        ra_small = ra_sq <= eps_sq
        rb_small = rb_sq <= eps_sq
        ra = torch.sqrt(torch.clamp(ra_sq, min=eps_sq))
        rb = torch.sqrt(torch.clamp(rb_sq, min=eps_sq))
        general_mask = ~ra_small & ~rb_small
        use_case1 = (ra >= rb) & general_mask
        use_case2 = (ra < rb) & general_mask

        safe_ra_re = torch.where(ra_small, torch.ones_like(ra_re), ra_re)
        safe_ra_im = torch.where(ra_small, torch.zeros_like(ra_im), ra_im)
        safe_rb_re = torch.where(rb_small, torch.ones_like(rb_re), rb_re)
        safe_rb_im = torch.where(rb_small, torch.zeros_like(rb_im), rb_im)
        phia = torch.atan2(safe_ra_im, safe_ra_re)
        phib = torch.atan2(safe_rb_im, safe_rb_re)

        phase = torch.outer(phia, coeffs.mp_plus_m) + torch.outer(
            phib, coeffs.m_minus_mp
        )
        exp_phase_re = torch.cos(phase)
        exp_phase_im = torch.sin(phase)

        safe_ra = torch.clamp(ra, min=eps)
        safe_rb = torch.clamp(rb, min=eps)
        log_ra = torch.log(safe_ra)
        log_rb = torch.log(safe_rb)

        result_re = torch.zeros(n_batch, coeffs.n_primary, dtype=dtype, device=device)
        result_im = torch.zeros_like(result_re)

        anti_rows = ra_small
        anti_log_rb = torch.where(anti_rows, log_rb, torch.zeros_like(log_rb))
        anti_phib = torch.where(anti_rows, phib, torch.zeros_like(phib))
        rb_power_mag = torch.exp(torch.outer(anti_log_rb, coeffs.special_2m))
        rb_power_phase = torch.outer(anti_phib, coeffs.special_2m)
        anti_re = (
            coeffs.anti_diag_sign.unsqueeze(0)
            * rb_power_mag
            * torch.cos(rb_power_phase)
        )
        anti_im = (
            coeffs.anti_diag_sign.unsqueeze(0)
            * rb_power_mag
            * torch.sin(rb_power_phase)
        )
        anti_mask = ra_small.unsqueeze(1) & coeffs.anti_diagonal_mask.unsqueeze(0)
        result_re = torch.where(anti_mask, anti_re, result_re)
        result_im = torch.where(anti_mask, anti_im, result_im)

        diag_rows = rb_small & ~ra_small
        diag_log_ra = torch.where(diag_rows, log_ra, torch.zeros_like(log_ra))
        diag_phia = torch.where(diag_rows, phia, torch.zeros_like(phia))
        ra_power_mag = torch.exp(torch.outer(diag_log_ra, coeffs.special_2m))
        ra_power_phase = torch.outer(diag_phia, coeffs.special_2m)
        diag_re = ra_power_mag * torch.cos(ra_power_phase)
        diag_im = ra_power_mag * torch.sin(ra_power_phase)
        diag_mask = diag_rows.unsqueeze(1) & coeffs.diagonal_mask.unsqueeze(0)
        result_re = torch.where(diag_mask, diag_re, result_re)
        result_im = torch.where(diag_mask, diag_im, result_im)

        ratio1 = -(rb * rb) / (safe_ra * safe_ra)
        case1_rows = use_case1
        magnitude1 = WignerDCalculator._compute_case_magnitude(
            torch.where(case1_rows, log_ra, torch.zeros_like(log_ra)),
            torch.where(case1_rows, log_rb, torch.zeros_like(log_rb)),
            torch.where(case1_rows, ratio1, torch.zeros_like(ratio1)),
            coeffs.case1,
        )
        val1_re = magnitude1 * exp_phase_re
        val1_im = magnitude1 * exp_phase_im
        mask1 = case1_rows.unsqueeze(1) & coeffs.case1.valid_mask.unsqueeze(0)
        result_re = torch.where(mask1, val1_re, result_re)
        result_im = torch.where(mask1, val1_im, result_im)

        ratio2 = -(ra * ra) / (safe_rb * safe_rb)
        case2_rows = use_case2
        magnitude2 = WignerDCalculator._compute_case_magnitude(
            torch.where(case2_rows, log_ra, torch.zeros_like(log_ra)),
            torch.where(case2_rows, log_rb, torch.zeros_like(log_rb)),
            torch.where(case2_rows, ratio2, torch.zeros_like(ratio2)),
            coeffs.case2,
        )
        val2_re = magnitude2 * exp_phase_re
        val2_im = magnitude2 * exp_phase_im
        mask2 = case2_rows.unsqueeze(1) & coeffs.case2.valid_mask.unsqueeze(0)
        result_re = torch.where(mask2, val2_re, result_re)
        result_im = torch.where(mask2, val2_im, result_im)

        D_re = torch.zeros(
            n_batch, coeffs.size, coeffs.size, dtype=dtype, device=device
        )
        D_im = torch.zeros_like(D_re)
        WignerDCalculator._scatter_primary_to_matrix(result_re, D_re, coeffs)
        WignerDCalculator._scatter_primary_to_matrix(result_im, D_im, coeffs)

        if coeffs.n_derived > 0:
            primary_re = result_re[:, coeffs.derived_primary_idx]
            primary_im = result_im[:, coeffs.derived_primary_idx]
            derived_sign = coeffs.derived_sign.unsqueeze(0)
            derived_re = derived_sign * primary_re
            derived_im = -derived_sign * primary_im
            D_re[:, coeffs.derived_row, coeffs.derived_col] = derived_re
            D_im[:, coeffs.derived_row, coeffs.derived_col] = derived_im

        return D_re.to(output_dtype), D_im.to(output_dtype)

    @staticmethod
    def _wigner_d_pair_to_real(
        D_re: torch.Tensor,
        D_im: torch.Tensor,
        U_blocks: list[tuple[torch.Tensor, torch.Tensor]]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        lmax: int,
        lmin: int,
    ) -> torch.Tensor:
        """
        Convert complex Wigner blocks to the current real packed basis.

        Each block applies the SeZM complex-to-real basis transform for its degree.
        This preserves the packed ``(l, m)`` contract of ``D_full`` and ``Dt_full``.
        """
        n_batch = D_re.shape[0]
        if lmin > lmax:
            return torch.zeros(n_batch, 0, 0, dtype=D_re.dtype, device=D_re.device)

        if isinstance(U_blocks, list):
            U_re, U_im, U_re_t, U_im_t = (
                WignerDCalculator._assemble_block_diagonal_real_basis(U_blocks)
            )
        else:
            U_re, U_im, U_re_t, U_im_t = U_blocks

        if U_re.dtype != D_re.dtype or U_re.device != D_re.device:
            U_re = U_re.to(dtype=D_re.dtype, device=D_re.device)
            U_im = U_im.to(dtype=D_re.dtype, device=D_re.device)
            U_re_t = U_re_t.to(dtype=D_re.dtype, device=D_re.device)
            U_im_t = U_im_t.to(dtype=D_re.dtype, device=D_re.device)

        temp_re = torch.matmul(D_re, U_re_t) + torch.matmul(D_im, U_im_t)
        temp_im = torch.matmul(D_im, U_re_t) - torch.matmul(D_re, U_im_t)
        return torch.matmul(U_re, temp_re) - torch.matmul(U_im, temp_im)

    def serialize(self) -> dict[str, Any]:
        """Serialize WignerDCalculator (lmax and dtype are stored by parent)."""
        return {
            "@class": "WignerDCalculator",
            "@version": 1,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> WignerDCalculator:
        """Deserialize WignerDCalculator - parent handles lmax/dtype reconstruction."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "WignerDCalculator":
            raise ValueError(f"Invalid class for WignerDCalculator: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        raise NotImplementedError(
            "WignerDCalculator.deserialize should be called by parent with lmax/dtype"
        )
