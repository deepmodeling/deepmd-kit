# SPDX-License-Identifier: LGPL-3.0-or-later
"""Correctness of the destination-segmented attention softmax operator.

The operator normalizes the attention logits over each destination segment
against a per-channel null mass, with the cutoff envelope entering as
``env**2`` so a muted edge drops out of the normalization entirely. Forward,
backward and second order each run as one CSR-segmented kernel, so a force
loss traverses the normalization without expanding the scatter/gather chain
into materialized surfaces.

The operator runs in float32 on both sides of the comparison (the caller casts
the logits before the call, since the normalization is where a reduced-precision
maximum would shift the whole segment). What is verified is therefore the
segmented reduction itself: the eager reference builds the same quantity out of
``scatter_reduce`` and ``index_select``, and the two are held to the
conditioning argument of :mod:`.conditioning`, plus the exact invariants the
normalization must satisfy.
"""

from __future__ import (
    annotations,
)

import pytest
import torch

from deepmd.pt_expt.kernels.triton.sezm.segment_softmax import (
    SEGMENT_SOFTMAX_TRITON_AVAILABLE,
    _segment_softmax_reference,
    segment_softmax,
)

from .conditioning import (
    assert_conditioned,
    deviations,
    grad_chain,
    median_deviations,
)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
    pytest.mark.skipif(
        not SEGMENT_SOFTMAX_TRITON_AVAILABLE, reason="Triton is unavailable"
    ),
]

# ``(n_node, n_edge, n_channel)``: the deployed attention widths are one or two
# focus streams times one to eight heads. The edge counts cover a dense
# neighbourhood, a sparse one, and the single-node degenerate segment.
SOFTMAX_SHAPES = [
    (128, 2048, 2),
    (512, 4096, 8),
    (64, 512, 16),
    (1, 96, 4),
]

# Independent operand draws the verdict is taken over; see
# :func:`.conditioning.median_deviations`.
DRAW_SEEDS = (11, 2027, 40529)


class _SegmentSoftmaxCase:
    """One segment layout with operands shared by every evaluation of it.

    ``seed`` selects the draw, so a comparison can be repeated over
    independent operands.
    """

    def __init__(
        self,
        n_node: int,
        n_edge: int,
        n_channel: int,
        *,
        seed: int,
        muted_fraction: float = 0.15,
    ) -> None:
        device = torch.device("cuda")
        torch.manual_seed(seed)
        self.n_node, self.n_edge = n_node, n_edge

        self.dst = torch.randint(0, n_node, (n_edge,), device=device, dtype=torch.long)
        self.logits = torch.randn(n_edge, n_channel, device=device, dtype=torch.float64)
        envelope = torch.rand(n_edge, device=device, dtype=torch.float64)
        # A muted edge (non-positive envelope) must leave the normalization
        # entirely, not merely be scaled to zero afterwards; the frozen-zone
        # invariance of the model depends on it.
        muted = torch.rand(n_edge, device=device) < muted_fraction
        self.envelope = envelope.masked_fill(muted, 0.0)
        self.null_logit = torch.randn(n_channel, device=device, dtype=torch.float64)
        self.cotangent = torch.randn_like(self.logits)
        self.second_cotangents = ((0, torch.randn_like(self.logits)),)

        order = torch.argsort(self.dst, dim=0, stable=True)
        counts = self.dst.new_zeros(n_node).scatter_add(
            0, self.dst, torch.ones_like(self.dst)
        )
        self.csr = (order, torch.cat([counts.new_zeros(1), torch.cumsum(counts, 0)]))

    @staticmethod
    def quantity_names() -> list[str]:
        """Labels of every quantity the evaluation reports."""
        return ["fwd", "d/d logits", "d2/d logits"]

    def evaluate(self, *, fused: bool, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
        """
        Run one evaluation of the normalization and its differentiated forms.

        Parameters
        ----------
        fused : bool
            Whether to call the fused operator or the eager reference.
        dtype : torch.dtype
            Working precision of the leaves.

        Returns
        -------
        tuple of torch.Tensor
            The weights and their first and second order gradients.
        """
        logits = self.logits.to(dtype).clone().requires_grad_(True)
        envelope = self.envelope.to(dtype)
        null_logit = self.null_logit.to(dtype)
        if fused:
            alpha = segment_softmax(logits, envelope, null_logit, *self.csr, self.dst)
        else:
            alpha = _segment_softmax_reference(
                logits, envelope, null_logit, self.dst, self.n_node
            )
        return grad_chain(alpha, [logits], self.cotangent, self.second_cotangents)


@pytest.mark.parametrize(("n_node", "n_edge", "n_channel"), SOFTMAX_SHAPES)
def test_float32_matches_eager_conditioning(
    n_node: int, n_edge: int, n_channel: int
) -> None:
    """Hold the segmented operator to the eager reference's float32 error."""
    runs = []
    for seed in DRAW_SEEDS:
        case = _SegmentSoftmaxCase(n_node, n_edge, n_channel, seed=seed)
        runs.append(
            deviations(
                case.quantity_names(),
                case.evaluate(fused=False, dtype=torch.float64),
                case.evaluate(fused=False, dtype=torch.float32),
                case.evaluate(fused=True, dtype=torch.float32),
                # The kernel reduces each segment in CSR order while the
                # reference scatters across the whole edge axis, so the two
                # differ by the order of one summation over the segment.
                factor=4.0,
                working_dtype=torch.float32,
            )
        )
    assert_conditioned(median_deviations(runs))


@pytest.mark.parametrize(("n_node", "n_edge", "n_channel"), SOFTMAX_SHAPES)
def test_normalization_invariants(n_node: int, n_edge: int, n_channel: int) -> None:
    """Assert the two properties the normalization must satisfy exactly.

    A muted edge carries no weight, and the weights of every segment sum to
    strictly less than one, the deficit being the null mass. Both are
    structural: they hold in any precision and do not depend on the reference.
    """
    case = _SegmentSoftmaxCase(n_node, n_edge, n_channel, seed=DRAW_SEEDS[0])
    logits = case.logits.to(torch.float32)
    alpha = segment_softmax(
        logits,
        case.envelope.to(torch.float32),
        case.null_logit.to(torch.float32),
        *case.csr,
        case.dst,
    )

    muted = case.envelope <= 0.0
    assert torch.all(alpha[muted] == 0.0), "a muted edge carries weight"

    segment_mass = torch.zeros(
        n_node, n_channel, device=alpha.device, dtype=torch.float64
    )
    segment_mass.index_add_(0, case.dst, alpha.double())
    assert torch.all(segment_mass <= 1.0 + 1e-6), "a segment exceeds unit mass"
    # Every segment keeps a strictly positive null mass, so no segment
    # saturates: this is what lets a node with no surviving neighbour stay
    # well defined.
    assert torch.all(segment_mass < 1.0), "a segment consumed the null mass"
