# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Second-order autograd for the fused multilinear SeZM operators.

Training a SeZM model against a force label differentiates twice: the force is
``-d(energy)/d(coord)``, and the force loss is then differentiated again with
respect to the parameters. Every gradient path therefore has to be traversed
one order deeper than inference needs, which is why each fused operator must
supply an autograd formula not only for its forward but also for its backward.

Multilinear structure
---------------------
Every fused operator in this package is *multilinear*: its forward

.. math::

    y = F(a_1, \\ldots, a_n)

is linear in each argument :math:`a_i` separately (the rotations are bilinear in
the feature and the Wigner-D matrix, the block GEMM in the activation and the
weight, the radial mixer and the flash aggregation trilinear). Multilinearity is
what makes the second order expressible with the operators that already exist:
no new kernel is required.

Write the first-order backward as

.. math::

    B(\\bar y, a_1, \\ldots, a_n) = (g_1, \\ldots, g_n), \\qquad
    g_j = B_j\\bigl(\\bar y, \\{a_k\\}_{k \\neq j}\\bigr),

where each :math:`g_j` is independent of :math:`a_j` because :math:`F` is linear
in it. Given the incoming second-order cotangents :math:`h_j` (the gradient with
respect to :math:`g_j`), the scalar being differentiated is

.. math::

    S = \\sum_i \\langle h_i, g_i \\rangle
      = \\sum_i \\langle \\bar y, F(a_1, \\ldots, h_i, \\ldots, a_n) \\rangle,

using the adjoint identity :math:`\\langle h_i, B_i(\\bar y, \\cdot) \\rangle =
\\langle \\bar y, F(\\ldots, h_i, \\ldots) \\rangle`. Differentiating :math:`S`
gives the whole second-order formula in terms of :math:`F` and :math:`B`:

.. math::

    \\nabla_{\\bar y} S &= \\sum_i F(a_1, \\ldots, h_i, \\ldots, a_n), \\\\
    \\nabla_{a_j} S &= \\sum_{i \\neq j}
        B_j\\bigl(\\bar y, \\{a_k\\}_{k \\neq j, k \\neq i}, h_i\\bigr).

The bilinear case
-----------------
For ``n = 2`` the second identity collapses to a *single* backward call. Since
:math:`B_1` depends only on :math:`a_2` and :math:`B_2` only on :math:`a_1`,
substituting both cotangents at once yields both components correctly:

.. math::

    (\\nabla_{a_1} S, \\nabla_{a_2} S) = B(\\bar y, h_1, h_2).

For ``n \\geq 3`` this shortcut is invalid -- substituting every argument would
introduce cross terms -- so the trilinear operators substitute one cotangent per
call and keep the components that actually depend on it.

Cost
----
The second-order pass costs ``n`` forward launches and one (bilinear) or ``n``
(general) backward launches, all of them the same kernels the first order uses.
Components whose cotangent is absent are skipped, so a graph that needs only the
coordinate gradient does not pay for the parameter path.
"""

from __future__ import (
    annotations,
)

from typing import (
    TYPE_CHECKING,
)

import torch
from torch import (
    Tensor,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )


def bilinear_second_order(
    forward: Callable[[Tensor, Tensor], Tensor],
    backward: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]],
    lhs: Tensor,
    rhs: Tensor,
    grad_lhs: Tensor | None,
    grad_rhs: Tensor | None,
    needs_grad_out: bool = True,
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    """
    Differentiate the backward of a bilinear operator.

    Implements the bilinear specialization derived in the module docstring::

        grad_grad_out = F(h_lhs, rhs) + F(lhs, h_rhs)
        (grad_lhs, grad_rhs) = B(h_lhs, h_rhs)

    The single backward call is exact because the ``lhs`` component of ``B``
    depends only on ``rhs`` and vice versa, so substituting both cotangents at
    once cannot mix them. A missing cotangent contributes nothing and is passed
    to ``B`` as an explicit zero, which keeps the call count at one.

    Parameters
    ----------
    forward : Callable[[Tensor, Tensor], Tensor]
        The operator's forward, closed over its non-differentiable arguments.
    backward : Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]
        The operator's first-order backward, closed over the output cotangent
        and the non-differentiable arguments, returning the gradients with
        respect to ``(lhs, rhs)`` in that order.
    lhs : Tensor
        First linear argument of the forward.
    rhs : Tensor
        Second linear argument of the forward.
    grad_lhs : Tensor or None
        Incoming second-order cotangent of the ``lhs`` gradient.
    grad_rhs : Tensor or None
        Incoming second-order cotangent of the ``rhs`` gradient.
    needs_grad_out : bool, default=True
        Whether the caller consumes the gradient with respect to the output
        cotangent. It costs two forward launches, so a graph that does not
        propagate past this point skips them.

    Returns
    -------
    tuple[Tensor or None, Tensor or None, Tensor or None]
        Gradients with respect to the output cotangent, ``lhs`` and ``rhs``.
    """
    if grad_lhs is None and grad_rhs is None:
        return None, None, None

    grad_grad_out: Tensor | None = None
    if needs_grad_out:
        if grad_lhs is not None:
            grad_grad_out = forward(grad_lhs, rhs)
        if grad_rhs is not None:
            grad_grad_out = accumulate(grad_grad_out, forward(lhs, grad_rhs))

    out_lhs, out_rhs = backward(
        grad_lhs if grad_lhs is not None else torch.zeros_like(lhs),
        grad_rhs if grad_rhs is not None else torch.zeros_like(rhs),
    )
    return grad_grad_out, out_lhs, out_rhs


def accumulate(total: Tensor | None, term: Tensor | None) -> Tensor | None:
    """
    Add a second-order contribution to a running total, tolerating absences.

    A cotangent is absent whenever the consumer of that output does not
    propagate a gradient into it, which makes both the running total and the
    incoming term optional.

    Parameters
    ----------
    total : Tensor or None
        Running sum, or None when no term has been added yet.
    term : Tensor or None
        Contribution to add, or None when it does not exist.

    Returns
    -------
    Tensor or None
        The updated sum, or None when both inputs are absent.
    """
    if term is None:
        return total
    return term if total is None else total + term


def zeros_like_if_needed(reference: Tensor, needed: bool) -> Tensor | None:
    """
    Return a zero tensor shaped like ``reference`` when a gradient is required.

    A backward that is reached with every cotangent absent still has to return a
    correctly shaped zero for the inputs autograd asked about.

    Parameters
    ----------
    reference : Tensor
        Tensor whose shape, dtype and device the result matches.
    needed : bool
        Whether the caller requires a gradient for this input.

    Returns
    -------
    Tensor or None
        A zero tensor when ``needed``, otherwise None.
    """
    return torch.zeros_like(reference) if needed else None
