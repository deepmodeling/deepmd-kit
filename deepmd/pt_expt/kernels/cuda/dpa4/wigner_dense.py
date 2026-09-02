# SPDX-License-Identifier: LGPL-3.0-or-later
"""Bindings for the fused DPA4 / SeZM dense Wigner-D build.

The CUDA operator ``deepmd::dpa4_wigner_dense`` (see
``source/op/pt/dpa4/wigner_dense.cu``) turns edge quaternions into the packed
block-diagonal pair ``(D_full, Dt_full)`` in one kernel. Every element of the
packed matrix is a homogeneous polynomial of degree ``2 l`` in the unit
quaternion; the tables built here fit those polynomials once per degree
against the reference calculator and store them as one sparse element-major
list, which the kernel evaluates in registers against a per-edge power table.

The module-composition path pays five full-size passes over the ``(E, D, D)``
pair (monomial basis, GEMM, zero fill, block scatter, transposed copy); the
fused operator pays the quaternion read and the two output writes.

The polynomial is differentiated as written. Its radial gradient component
(the homogeneity direction) is projected out upstream by the quaternion
normalization, so the extension ambiguity off the unit sphere does not reach
the geometry, matching the run-table operator of the fused convolution.
"""

from __future__ import (
    annotations,
)

from functools import (
    cache,
)
from typing import (
    Any,
)

import torch

from .so2_conv import (
    _monomial_exponents,
    _monomials,
)

__all__ = [
    "WignerDenseCuda",
    "ensure_registered",
    "make_cuda_wigner_dense",
    "op_available",
    "wigner_dense_tables",
]

_DENSE_TABLE_CACHE: dict[int, tuple[torch.Tensor, ...]] = {}

# Degrees above ten leave the dedicated monomial path of the reference
# calculator as well, so the fit target would change; the gate declines them.
_MAX_LMAX = 10

# Fitted coefficients are exact rationals recovered to about 1e-12 in fp64;
# entries below this magnitude are structural zeros of the polynomial.
_PRUNE_TOL = 1e-9


def op_available() -> bool:
    """Whether the C++ ``deepmd::dpa4_wigner_dense`` op is loaded."""
    op = getattr(torch.ops.deepmd, "dpa4_wigner_dense", None)
    return isinstance(op, torch._ops.OpOverloadPacket)


def wigner_dense_tables(lmax: int) -> tuple[torch.Tensor, ...]:
    """
    Sparse element-major polynomial tables of the packed Wigner pair.

    Every block-diagonal element ``(l, r, c)`` is fitted on its own degree
    ``2 l`` monomial basis (residual below 1e-11 across the supported
    degrees) and pruned to its structural non-zeros -- between 2.5 and 44
    entries per element on average over the supported degrees, so the whole
    table stays L2-resident.

    Parameters
    ----------
    lmax : int
        Maximum spherical-harmonic degree of the packed matrix.

    Returns
    -------
    tuple of torch.Tensor
        ``(elem_ptr, elem_pos, entry_coeff, entry_mono)`` on the CPU with
        shapes (NB + 1,) int32, (NB,) int32, (K,) fp32 and (K,) int32, where
        ``NB = sum_l (2 l + 1)^2`` counts the block-diagonal elements,
        ``elem_pos`` packs the dense position ``r * D + c`` and
        ``entry_mono`` packs the four exponents into one byte each.
    """
    cached = _DENSE_TABLE_CACHE.get(lmax)
    if cached is not None:
        return cached
    from deepmd.pt.model.descriptor.sezm_nn.wignerd import (
        WignerDCalculator,
        quaternion_normalize,
    )

    calc = WignerDCalculator(lmax=lmax, dtype=torch.float64)
    device = next(calc.buffers()).device if any(True for _ in calc.buffers()) else "cpu"
    generator = torch.Generator().manual_seed(2026)
    n_fit = max(4 * _monomial_exponents(2 * lmax).shape[0], 4096)
    q = quaternion_normalize(
        torch.randn(n_fit, 4, dtype=torch.float64, generator=generator, device="cpu")
    ).to(device)
    d_full, _ = calc(q)

    dim = (lmax + 1) ** 2
    elem_ptr = [0]
    elem_pos: list[int] = []
    coeffs: list[float] = []
    monos: list[int] = []
    for l in range(lmax + 1):
        start = l * l
        size = 2 * l + 1
        block = d_full[:, start : start + size, start : start + size]
        exps = _monomial_exponents(2 * l).to(device)
        sol = torch.linalg.lstsq(
            _monomials(q, exps), block.reshape(n_fit, -1)
        ).solution.cpu()  # (M_2l, size * size)
        exps = exps.cpu()
        for r in range(size):
            for c in range(size):
                column = sol[:, r * size + c]
                for m in torch.nonzero(column.abs() > _PRUNE_TOL).flatten().tolist():
                    a, b, cc, d = (int(v) for v in exps[m])
                    coeffs.append(float(column[m]))
                    monos.append(a | (b << 8) | (cc << 16) | (d << 24))
                elem_pos.append((start + r) * dim + (start + c))
                elem_ptr.append(len(coeffs))

    tables = (
        torch.tensor(elem_ptr, dtype=torch.int32, device="cpu"),
        torch.tensor(elem_pos, dtype=torch.int32, device="cpu"),
        torch.tensor(coeffs, dtype=torch.float32, device="cpu"),
        torch.tensor(monos, dtype=torch.int32, device="cpu"),
    )
    _DENSE_TABLE_CACHE[lmax] = tables
    return tables


def _forward_fake(
    quat: torch.Tensor,
    elem_ptr: torch.Tensor,
    elem_pos: torch.Tensor,
    entry_coeff: torch.Tensor,
    entry_mono: torch.Tensor,
    lmax: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del elem_ptr, elem_pos, entry_coeff, entry_mono
    dim = (lmax + 1) ** 2
    n_edge = quat.shape[0]
    return (
        quat.new_empty((n_edge, dim, dim)),
        quat.new_empty((n_edge, dim, dim)),
    )


def _backward_fake(
    g_d: torch.Tensor,
    g_dt: torch.Tensor,
    quat: torch.Tensor,
    elem_ptr: torch.Tensor,
    elem_pos: torch.Tensor,
    entry_coeff: torch.Tensor,
    entry_mono: torch.Tensor,
    lmax: int,
) -> torch.Tensor:
    del g_d, g_dt, elem_ptr, elem_pos, entry_coeff, entry_mono, lmax
    return quat.new_empty(quat.shape)


def _setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    del output
    quat, elem_ptr, elem_pos, entry_coeff, entry_mono = inputs[:5]
    ctx.save_for_backward(quat, elem_ptr, elem_pos, entry_coeff, entry_mono)
    ctx.lmax = inputs[5]


def _backward(ctx: Any, g_d: torch.Tensor, g_dt: torch.Tensor) -> tuple:
    quat, elem_ptr, elem_pos, entry_coeff, entry_mono = ctx.saved_tensors
    g_quat = torch.ops.deepmd.dpa4_wigner_dense_backward(
        g_d.contiguous(),
        g_dt.contiguous(),
        quat,
        elem_ptr,
        elem_pos,
        entry_coeff,
        entry_mono,
        ctx.lmax,
    )
    return g_quat, None, None, None, None, None


@cache
def _register_ops() -> None:
    """Register fake and autograd implementations once."""
    torch.library.register_fake("deepmd::dpa4_wigner_dense")(_forward_fake)
    torch.library.register_fake("deepmd::dpa4_wigner_dense_backward")(_backward_fake)
    torch.library.register_autograd(
        "deepmd::dpa4_wigner_dense", _backward, setup_context=_setup_context
    )


def ensure_registered() -> None:
    """Register fake and autograd implementations when the op is available."""
    if op_available():
        _register_ops()


class WignerDenseCuda:
    """Model entry matching the ``WignerCalculatorFn`` contract.

    The polynomial fit runs at construction, which is always an eager
    context; the call path only migrates the finished tables, so tracing the
    model never re-enters the fit.
    """

    def __init__(self, lmax: int) -> None:
        self.lmax = int(lmax)
        self._cpu_tables = wigner_dense_tables(self.lmax)
        self._tables: tuple[torch.Tensor, ...] | None = None

    def tables(self, device: torch.device) -> tuple[torch.Tensor, ...]:
        """The four table tensors on the compute device."""
        if self._tables is None or self._tables[0].device != device:
            self._tables = tuple(t.to(device) for t in self._cpu_tables)
        return self._tables

    def __call__(self, quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build the packed block-diagonal Wigner pair from unit quaternions.

        Parameters
        ----------
        quat : torch.Tensor
            Unit quaternions with shape (E, 4) in ``(w, x, y, z)`` order.

        Returns
        -------
        tuple of torch.Tensor
            ``(D_full, Dt_full)`` with shape (E, (lmax + 1)^2, (lmax + 1)^2),
            matching the reference calculator.
        """
        ensure_registered()
        elem_ptr, elem_pos, entry_coeff, entry_mono = self.tables(quat.device)
        return torch.ops.deepmd.dpa4_wigner_dense(
            quat, elem_ptr, elem_pos, entry_coeff, entry_mono, self.lmax
        )


def make_cuda_wigner_dense(lmax: int, dtype: torch.dtype) -> WignerDenseCuda | None:
    """Bind the fused dense Wigner build for one degree, or decline.

    Returns
    -------
    WignerDenseCuda or None
        ``None`` when the operator is absent, the compute dtype is not
        float32, or the degree is outside the fitted monomial range.
    """
    if not op_available():
        return None
    if dtype != torch.float32:
        return None
    if not 1 <= int(lmax) <= _MAX_LMAX:
        return None
    return WignerDenseCuda(lmax)
