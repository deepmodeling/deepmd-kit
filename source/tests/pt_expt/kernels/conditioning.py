# SPDX-License-Identifier: LGPL-3.0-or-later
"""Arbitration of a fused training operator against its eager reference.

A fused operator and the eager expression it replaces reduce in different
orders, so they never agree bitwise and a golden value would only record one
machine's rounding. What is verifiable is *conditioning*: both sides compute
the same mathematical function, so evaluating both in the working precision
and comparing each against the same expression evaluated in float64 separates
a logic error (orders of magnitude) from a reduction-order difference (a small
multiple of what the eager side already carries).

Every check here therefore runs three evaluations of one quantity -- the
float64 ground truth, the eager reference in the working precision, and the
fused operator in the working precision -- and bounds the fused error by a
multiple of the eager error. That multiple is the only tolerance knob, and it
is a statement about the operator's numerics, not about a platform. In
particular there is no operator-specific absolute tolerance: a bound that does
not reference the eager error would record one machine's rounding and would
mask a real precision regression the moment it is loosened to make a run pass.

The one exception is degenerate: where the eager reference happens to be exact
in the working precision, a multiple of zero would reject any reduction-order
difference at all. A single rounding of the working precision is admitted for
that case, which is a property of the format rather than of the operator.

Both errors are extremes over a tensor, so for a quantity with few elements --
a per-focus bias with two entries, say -- a single draw of the operands is a
poor estimate: the ratio of two small-sample extremes swings by an order of
magnitude between draws even when both sides carry the same rounding. The
verdict is therefore taken on the median over independent draws, which is what
:func:`median_deviations` is for.

For operators with an analytic second order the same treatment applies to the
differentiated quantities: a force loss differentiates the convolution twice,
so the second-order projection is arbitrated by the eager autograd of the
same expression, which is exact for a fixed multilinear composition.
"""

from __future__ import (
    annotations,
)

from dataclasses import (
    dataclass,
)
from statistics import (
    median,
)
from typing import (
    TYPE_CHECKING,
)

import torch

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Sequence,
    )


@dataclass(frozen=True)
class Deviation:
    """The distance of one quantity from the float64 ground truth.

    Attributes
    ----------
    name : str
        Quantity label, used in assertion messages.
    eager : float
        Error of the eager reference relative to the ground-truth scale.
    fused : float
        Error of the fused operator relative to the same scale.
    bound : float
        Largest fused error the comparison accepts.
    """

    name: str
    eager: float
    fused: float
    bound: float

    @property
    def ok(self) -> bool:
        """Whether the fused error stayed within the accepted bound."""
        return self.fused <= self.bound

    def __str__(self) -> str:
        verdict = "ok" if self.ok else "FAIL"
        return (
            f"{self.name}: fused-vs-fp64 {self.fused:.3e} "
            f"eager-vs-fp64 {self.eager:.3e} bound {self.bound:.3e} [{verdict}]"
        )


def deviations(
    names: Sequence[str],
    gold: Sequence[torch.Tensor],
    eager: Sequence[torch.Tensor],
    fused: Sequence[torch.Tensor],
    *,
    factor: float,
    working_dtype: torch.dtype,
    project: Callable[[str, torch.Tensor], torch.Tensor] | None = None,
) -> list[Deviation]:
    """
    Measure how far the eager and fused evaluations sit from the ground truth.

    Parameters
    ----------
    names : Sequence[str]
        Quantity labels, one per tensor in the three sequences.
    gold : Sequence[torch.Tensor]
        The quantities evaluated in float64.
    eager : Sequence[torch.Tensor]
        The eager reference evaluated in the working precision.
    fused : Sequence[torch.Tensor]
        The fused operator evaluated in the working precision.
    factor : float
        Multiple of the eager error still attributed to reduction order.
    working_dtype : torch.dtype
        Precision both evaluations under test ran in. One rounding of this
        format is admitted where the eager reference came out exact, which is
        the only case a multiple of the eager error cannot express.
    project : Callable, optional
        Restriction applied to both sides before comparing, for quantities
        defined only on a subdomain (a gradient the kernels populate on the
        structural support of a block-diagonal operand, say). Receives the
        quantity name and tensor.

    Returns
    -------
    list of Deviation
        One entry per quantity, in the order given.
    """
    if project is None:

        def project(name: str, tensor: torch.Tensor) -> torch.Tensor:
            return tensor

    unit_roundoff = float(torch.finfo(working_dtype).eps)
    measured = []
    for name, truth, ref, got in zip(names, gold, eager, fused, strict=True):
        truth, ref, got = (project(name, t) for t in (truth, ref, got))
        # The scale floor keeps a quantity that is identically zero (an
        # inactive gradient slot) from turning rounding into a large relative
        # error.
        scale = truth.abs().max().clamp_min(1.0).item()
        eager_error = (ref - truth).abs().max().item() / scale
        measured.append(
            Deviation(
                name=name,
                eager=eager_error,
                fused=(got - truth).abs().max().item() / scale,
                bound=max(factor * eager_error, unit_roundoff),
            )
        )
    return measured


def median_deviations(runs: Sequence[Sequence[Deviation]]) -> list[Deviation]:
    """
    Reduce independent draws of the same comparison to their median.

    Each draw reports the extreme error over a tensor, which for a quantity
    with few elements is itself a noisy statistic. Taking the median of both
    sides across draws removes that sampling noise without touching the
    criterion: the bound is still a multiple of the eager reference's own
    error, now estimated from several draws instead of one.

    Parameters
    ----------
    runs : Sequence[Sequence[Deviation]]
        One deviation table per draw, all listing the same quantities in the
        same order.

    Returns
    -------
    list of Deviation
        One entry per quantity, holding the median eager error, the median
        fused error, and the median bound.

    Raises
    ------
    ValueError
        If no draws were given, or the tables disagree on the quantities.
    """
    if not runs:
        raise ValueError("median_deviations needs at least one draw")
    names = [entry.name for entry in runs[0]]
    for table in runs[1:]:
        if [entry.name for entry in table] != names:
            raise ValueError("the draws report different quantities")
    return [
        Deviation(
            name=name,
            eager=median(table[index].eager for table in runs),
            fused=median(table[index].fused for table in runs),
            bound=median(table[index].bound for table in runs),
        )
        for index, name in enumerate(names)
    ]


def assert_conditioned(measured: Sequence[Deviation]) -> None:
    """
    Fail with the full deviation table when any quantity exceeds its bound.

    Parameters
    ----------
    measured : Sequence[Deviation]
        Deviations produced by :func:`deviations`.

    Raises
    ------
    AssertionError
        If any deviation exceeds its bound.
    """
    failures = [entry for entry in measured if not entry.ok]
    if failures:
        table = "\n".join(f"  {entry}" for entry in measured)
        raise AssertionError(
            f"{len(failures)} quantities exceed the eager conditioning:\n{table}"
        )


def grad_chain(
    outputs: torch.Tensor,
    leaves: Sequence[torch.Tensor],
    cotangent: torch.Tensor,
    second_cotangents: Sequence[tuple[int, torch.Tensor]] = (),
) -> tuple[torch.Tensor, ...]:
    """
    Evaluate a quantity together with its first and second-order projections.

    The projections reproduce what a force loss asks of an operator: the first
    order supplies the parameter gradients, and the second order differentiates
    a selected subset of them again, since their producers sit on the
    coordinate graph.

    Parameters
    ----------
    outputs : torch.Tensor
        The operator output.
    leaves : Sequence[torch.Tensor]
        The differentiation targets, in reporting order.
    cotangent : torch.Tensor
        Cotangent contracted with ``outputs`` to form the first-order scalar.
    second_cotangents : Sequence[tuple[int, torch.Tensor]]
        Pairs of (leaf index, cotangent) contracted with the first-order
        gradients to form the second-order scalar. Empty skips the second
        order.

    Returns
    -------
    tuple of torch.Tensor
        The output, the first-order gradients, and (when a second order was
        requested) the second-order gradients, all cast to float64. Gradients
        the second order leaves untouched are reported as explicit zeros.
    """
    create_graph = bool(second_cotangents)
    first = torch.autograd.grad(
        (outputs.double() * cotangent).sum(), leaves, create_graph=create_graph
    )
    if not create_graph:
        return (outputs.double(), *(g.double() for g in first))
    scalar = sum(
        (first[index].double() * cot.reshape_as(first[index])).sum()
        for index, cot in second_cotangents
    )
    second = torch.autograd.grad(scalar, leaves, allow_unused=True)
    return (
        outputs.double(),
        *(g.double() for g in first),
        *(
            (torch.zeros_like(leaf) if g is None else g).double()
            for g, leaf in zip(second, leaves, strict=True)
        ),
    )
