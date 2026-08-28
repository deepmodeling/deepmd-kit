# SPDX-License-Identifier: LGPL-3.0-or-later
"""Bindings for the fused DPA4 / SeZM cutoff envelope and radial basis.

The CUDA operator ``deepmd::dpa4_edge_radial`` (see
``source/op/pt/dpa4/edge_radial.cu``) evaluates both quantities the edge cache
derives from the pair distance::

    env[e] = keep[e] * E_p1(r)
    rbf[e, n] = keep[e] * phi_n(r) * E_p2(r)

Written as tensor operations this chain is cheap enough that the compiler
inlines it into every consumer of ``env`` and ``rbf`` and re-evaluates it there,
so a 96 MB pass is paid several times over. Behind an operator boundary it runs
once.

The basis frequencies are inference-time constants here and take no gradient;
the path is only selected outside training.
"""

from __future__ import (
    annotations,
)

import math
from functools import (
    cache,
)
from typing import (
    Any,
)

import torch

__all__ = [
    "BESSEL",
    "GAUSSIAN",
    "EdgeRadialCuda",
    "edge_radial",
    "ensure_registered",
    "make_cuda_edge_radial",
    "op_available",
    "series_coefficients",
]

BESSEL = 0
GAUSSIAN = 1

# Longest Horner series the operator stages, mirroring ``kMaxSeries`` in
# ``source/op/pt/dpa4/edge_radial.cu``.
_MAX_SERIES = 16


def op_available() -> bool:
    """Whether the C++ ``deepmd::dpa4_edge_radial`` op is loaded."""
    op = getattr(torch.ops.deepmd, "dpa4_edge_radial", None)
    return isinstance(op, torch._ops.OpOverloadPacket)


def series_coefficients(exponent: int) -> tuple[float, ...]:
    """Positive binomial coefficients of the C3 envelope series."""
    return tuple(float(math.comb(k + 3, 3)) for k in range(exponent))


def supported(exponent_env: int, exponent_rbf: int) -> bool:
    """Whether both envelope orders fit the staged series limit."""
    return 2 <= exponent_env <= _MAX_SERIES and 2 <= exponent_rbf <= _MAX_SERIES


def _forward_fake(
    edge_len: torch.Tensor,
    keep: torch.Tensor,
    freqs: torch.Tensor,
    env_series: torch.Tensor,
    rbf_series: torch.Tensor,
    rcut: float,
    gaussian_coeff: float,
    basis: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del keep, env_series, rbf_series, rcut, gaussian_coeff, basis
    n_edge = edge_len.numel()
    return (
        edge_len.new_empty((n_edge, 1)),
        edge_len.new_empty((n_edge, freqs.numel())),
    )


def _backward_fake(
    grad_env: torch.Tensor,
    grad_rbf: torch.Tensor,
    edge_len: torch.Tensor,
    keep: torch.Tensor,
    freqs: torch.Tensor,
    env_series: torch.Tensor,
    rbf_series: torch.Tensor,
    rcut: float,
    gaussian_coeff: float,
    basis: int,
) -> torch.Tensor:
    del grad_env, grad_rbf, keep, freqs, env_series, rbf_series
    del rcut, gaussian_coeff, basis
    return edge_len.new_empty((edge_len.numel(), 1))


def _setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    del output
    edge_len, keep, freqs, env_series, rbf_series = inputs[:5]
    ctx.save_for_backward(edge_len, keep, freqs, env_series, rbf_series)
    ctx.rcut, ctx.gaussian_coeff, ctx.basis = inputs[5:8]


def _backward(ctx: Any, grad_env: torch.Tensor, grad_rbf: torch.Tensor) -> tuple:
    edge_len, keep, freqs, env_series, rbf_series = ctx.saved_tensors
    grad_len = torch.ops.deepmd.dpa4_edge_radial_backward(
        grad_env.contiguous(),
        grad_rbf.contiguous(),
        edge_len,
        keep,
        freqs,
        env_series,
        rbf_series,
        ctx.rcut,
        ctx.gaussian_coeff,
        ctx.basis,
    )
    return grad_len.reshape(edge_len.shape), None, None, None, None, None, None, None


@cache
def _register_ops() -> None:
    """Register fake and autograd implementations once."""
    torch.library.register_fake("deepmd::dpa4_edge_radial")(_forward_fake)
    torch.library.register_fake("deepmd::dpa4_edge_radial_backward")(_backward_fake)
    torch.library.register_autograd(
        "deepmd::dpa4_edge_radial", _backward, setup_context=_setup_context
    )


def ensure_registered() -> None:
    """Register fake and autograd implementations when the op is available."""
    if op_available():
        _register_ops()


def edge_radial(
    edge_len: torch.Tensor,
    keep: torch.Tensor,
    freqs: torch.Tensor,
    env_series: torch.Tensor,
    rbf_series: torch.Tensor,
    rcut: float,
    gaussian_coeff: float,
    basis: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate the cutoff envelope and the radial basis of every edge.

    Parameters
    ----------
    edge_len : torch.Tensor
        Pair distances with shape (E, 1) in Å.
    keep : torch.Tensor
        Per-edge keep weight with shape (E, 1), zero on excluded pairs.
    freqs : torch.Tensor
        Basis frequencies with shape (1, n_radial): wave numbers for the Bessel
        family, centers in Å for the Gaussian one.
    env_series : torch.Tensor
        Horner coefficients of the edge envelope with shape (p1,).
    rbf_series : torch.Tensor
        Horner coefficients of the basis envelope with shape (p2,).
    rcut : float
        Cutoff radius in Å.
    gaussian_coeff : float
        Exponent scale of the Gaussian family, unused for Bessel.
    basis : int
        ``BESSEL`` or ``GAUSSIAN``.

    Returns
    -------
    tuple of torch.Tensor
        The envelope with shape (E, 1) and the basis with shape (E, n_radial).
    """
    ensure_registered()
    return torch.ops.deepmd.dpa4_edge_radial(
        edge_len, keep, freqs, env_series, rbf_series, rcut, gaussian_coeff, basis
    )


class EdgeRadialCuda:
    """Model entry binding one envelope and one basis to the fused operator.

    The Horner series are compile-time constants of the two modules and are
    materialized once per device; the basis frequencies are read from the live
    parameter on every call, so a checkpoint loaded after construction is
    picked up.
    """

    def __init__(self, envelope: Any, basis: Any) -> None:
        self._envelope = envelope
        self._basis = basis
        self._rcut = float(envelope.rcut)
        self._basis_type = BESSEL if basis.basis_type == "bessel" else GAUSSIAN
        self._env = series_coefficients(envelope.p)
        self._rbf = series_coefficients(basis.envelope.p)
        self._series: tuple[torch.Tensor, torch.Tensor] | None = None

    def series(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """The two Horner series on the compute device."""
        if self._series is None or self._series[0].device != device:
            self._series = (
                torch.tensor(self._env, dtype=torch.float32, device=device),
                torch.tensor(self._rbf, dtype=torch.float32, device=device),
            )
        return self._series

    def __call__(
        self, edge_len: torch.Tensor, keep: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the keep-weighted envelope (E, 1) and basis (E, n_radial).

        Parameters
        ----------
        edge_len : torch.Tensor
            Pair distances with shape (E, 1) in Å.
        keep : torch.Tensor
            Per-edge keep weight with shape (E, 1).

        Returns
        -------
        tuple of torch.Tensor
            The envelope and the radial basis, matching the composition of
            ``C3CutoffEnvelope`` and ``RadialBasis`` scaled by ``keep``.
        """
        env_series, rbf_series = self.series(edge_len.device)
        return edge_radial(
            edge_len,
            keep,
            self._basis.adam_freqs,
            env_series,
            rbf_series,
            self._rcut,
            float(self._basis.gaussian_coeff),
            self._basis_type,
        )


def make_cuda_edge_radial(envelope: Any, basis: Any) -> EdgeRadialCuda | None:
    """Bind the fused operator to a matching envelope and basis, or decline.

    Returns
    -------
    EdgeRadialCuda or None
        ``None`` when the operator is absent, the two modules disagree on the
        cutoff, the compute precision is unsupported, or an envelope order is
        outside the staged series limit.
    """
    if not op_available():
        return None
    if float(envelope.rcut) != float(basis.rcut):
        return None
    if basis.adam_freqs.dtype is not torch.float32:
        return None
    if not supported(int(envelope.p), int(basis.envelope.p)):
        return None
    if basis.basis_type not in ("bessel", "gaussian"):
        return None
    return EdgeRadialCuda(envelope, basis)
