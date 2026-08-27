# SPDX-License-Identifier: LGPL-3.0-or-later
"""Correctness of the fused coefficient-grid pair operator used in training.

The operator evaluates ``from_grid(to_grid(left) * to_grid(right))`` without
materializing the grid field, which is 39 times larger than its coefficient
operand at the production SO(3) shape. It is a fixed multilinear composition,
so the eager autograd of the same expression is exact and arbitrates the
operator's forward, backward and second order.

The comparison follows the conditioning argument of :mod:`.conditioning`: both
sides run in the working precision and are judged against the float64
evaluation of the reference, with the bound expressed as a multiple of the
eager reference's own distance from that truth.
"""

from __future__ import (
    annotations,
)

import pytest
import torch

from deepmd.pt_expt.kernels.triton.sezm.grid_pair import (
    GRID_PAIR_TRITON_AVAILABLE,
    _built_in_launch_config,
    grid_pair_train,
)

from .conditioning import (
    assert_conditioned,
    deviations,
    grad_chain,
    median_deviations,
)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
    pytest.mark.skipif(not GRID_PAIR_TRITON_AVAILABLE, reason="Triton is unavailable"),
]

# ``(lmax, n_frames, n_focus, channels, n_grid)`` spanning the deployed grid
# shapes. The slot count ``(lmax + 1)^2 * n_frames`` drives the operator's
# two-stage tiling of the contraction axis, so the set covers a power-of-two
# slot count and counts that force the split.
GRID_SHAPES = [
    (3, 3, 2, 32, 152),
    (5, 3, 2, 64, 344),
    (5, 3, 1, 64, 344),
    (6, 3, 2, 96, 460),
]

# Independent operand draws the verdict is taken over; see
# :func:`.conditioning.median_deviations`.
DRAW_SEEDS = (11, 2027, 40529)


def test_blackwell_launch_table_uses_exact_grid_shape() -> None:
    """Keep production launch pins scoped to the swept device and grid."""
    shape_key = (
        "_grid_pair_bwd2_kernel",
        147,
        96,
        2,
        3,
        False,
        584,
        torch.bfloat16,
    )
    assert _built_in_launch_config(
        "NVIDIA RTX PRO 6000 Blackwell Server Edition", shape_key
    ) == (16, 32, 2)
    assert _built_in_launch_config("NVIDIA H20", shape_key) is None
    alternate_grid = (*shape_key[:-2], 460, shape_key[-1])
    assert (
        _built_in_launch_config(
            "NVIDIA RTX PRO 6000 Blackwell Server Edition", alternate_grid
        )
        is None
    )


def _eager_pair(
    left: torch.Tensor,
    right: torch.Tensor,
    to_grid: torch.Tensor,
    from_grid: torch.Tensor,
    n_frames: int,
) -> torch.Tensor:
    """Reference composition on the frame-packed ``(N, D, F, K * C)`` layout."""
    n_batch, coeff_dim, n_focus, packed = left.shape
    n_grid = to_grid.shape[0]
    to_slots = to_grid.reshape(n_grid, coeff_dim, n_frames)
    from_slots = from_grid.reshape(n_grid, coeff_dim, n_frames)
    left_view = left.reshape(n_batch, coeff_dim, n_focus, n_frames, -1)
    right_view = right.reshape(n_batch, coeff_dim, n_focus, n_frames, -1)
    left_grid = torch.einsum("gdk,ndfkc->ngfc", to_slots, left_view)
    right_grid = torch.einsum("gdk,ndfkc->ngfc", to_slots, right_view)
    out = torch.einsum("gdk,ngfc->ndfkc", from_slots, left_grid * right_grid)
    return out.reshape(n_batch, coeff_dim, n_focus, packed)


class _GridPairCase:
    """One grid shape with operands shared by every evaluation of it.

    ``seed`` selects the draw, so a comparison can be repeated over
    independent operands.
    """

    def __init__(
        self,
        lmax: int,
        n_frames: int,
        n_focus: int,
        channels: int,
        n_grid: int,
        *,
        seed: int,
        n_node: int = 300,
    ) -> None:
        device = torch.device("cuda")
        torch.manual_seed(seed)
        self.n_frames = n_frames
        coeff_dim = (lmax + 1) ** 2
        double = {"device": device, "dtype": torch.float64}

        self.left = torch.randn(
            n_node, coeff_dim, n_focus, n_frames * channels, **double
        )
        self.right = torch.randn_like(self.left)
        # The projectors are scaled by the grid count so the round trip keeps
        # the operands' magnitude and the comparison is not dominated by one
        # side's dynamic range.
        self.to_grid = torch.randn(n_grid, coeff_dim * n_frames, **double) / (
            n_grid**0.5
        )
        self.from_grid = torch.randn_like(self.to_grid) / n_grid**0.5
        self.cotangent = torch.randn_like(self.left)
        self.second_cotangents = (
            (0, torch.randn_like(self.left)),
            (1, torch.randn_like(self.right)),
        )

    @staticmethod
    def quantity_names() -> list[str]:
        """Labels of every quantity the evaluation reports."""
        return ["fwd", "d/d left", "d/d right", "d2/d left", "d2/d right"]

    def evaluate(
        self,
        *,
        fused: bool,
        dtype: torch.dtype,
        amp: bool,
        strided: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """
        Run one evaluation of the pair product and its differentiated forms.

        Parameters
        ----------
        fused : bool
            Whether to call the fused operator or the eager composition.
        dtype : torch.dtype
            Working precision of the leaves.
        amp : bool
            Whether to run inside bfloat16 autocast. Both sides lower to the
            same reduced-precision regime there, so the comparison stays
            inside one ambient mode.
        strided : bool, default=False
            Whether coefficient operands use a non-contiguous trailing stride,
            as the channel slices entering the production grid nets do.

        Returns
        -------
        tuple of torch.Tensor
            The output and its first and second order gradients.
        """

        def make_leaf(value: torch.Tensor) -> torch.Tensor:
            value = value.to(dtype)
            if not strided:
                return value.clone().requires_grad_(True)
            storage = torch.empty(
                (*value.shape[:-1], value.shape[-1] * 2),
                device=value.device,
                dtype=value.dtype,
            )
            view = storage[..., ::2]
            view.copy_(value)
            return view.requires_grad_(True)

        left = make_leaf(self.left)
        right = make_leaf(self.right)
        to_grid, from_grid = self.to_grid.to(dtype), self.from_grid.to(dtype)
        context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if amp
            else torch.autocast("cuda", enabled=False)
        )
        with context:
            evaluate = grid_pair_train if fused else _eager_pair
            out = evaluate(left, right, to_grid, from_grid, self.n_frames)
        return grad_chain(out, [left, right], self.cotangent, self.second_cotangents)


def _compare(
    shape: tuple[int, int, int, int, int],
    *,
    amp: bool,
    strided: bool = False,
) -> None:
    """Arbitrate the fused pair product against the eager composition."""
    working = torch.bfloat16 if amp else torch.float32
    runs = []
    for seed in DRAW_SEEDS:
        case = _GridPairCase(*shape, seed=seed)
        runs.append(
            deviations(
                case.quantity_names(),
                case.evaluate(
                    fused=False, dtype=torch.float64, amp=False, strided=strided
                ),
                case.evaluate(
                    fused=False, dtype=torch.float32, amp=amp, strided=strided
                ),
                case.evaluate(
                    fused=True, dtype=torch.float32, amp=amp, strided=strided
                ),
                # The operator walks the grid axis in its natural order while
                # the eager chain reduces through cuBLAS, so the two agree to
                # the conditioning of the same contraction. Under bfloat16 the
                # fused walk keeps float32 partials where the eager chain
                # rounds the grid field itself, so it normally sits closer to
                # the truth than the reference does.
                factor=5.0,
                working_dtype=working,
            )
        )
    assert_conditioned(median_deviations(runs))


@pytest.mark.parametrize(
    ("lmax", "n_frames", "n_focus", "channels", "n_grid"), GRID_SHAPES
)
def test_float32_matches_eager_conditioning(
    lmax: int, n_frames: int, n_focus: int, channels: int, n_grid: int
) -> None:
    """Hold the fused pair product to the eager composition's float32 error."""
    _compare((lmax, n_frames, n_focus, channels, n_grid), amp=False)


@pytest.mark.parametrize(
    ("lmax", "n_frames", "n_focus", "channels", "n_grid"), GRID_SHAPES
)
def test_autocast_bfloat16_matches_eager_conditioning(
    lmax: int, n_frames: int, n_focus: int, channels: int, n_grid: int
) -> None:
    """Hold the same bound under the bfloat16 autocast of production training."""
    _compare((lmax, n_frames, n_focus, channels, n_grid), amp=True)


def test_noncontiguous_operands_match_eager_conditioning() -> None:
    """Cover the channel-slice strides supplied by production grid nets."""
    _compare(GRID_SHAPES[1], amp=True, strided=True)
