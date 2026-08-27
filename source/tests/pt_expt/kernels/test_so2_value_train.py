# SPDX-License-Identifier: LGPL-3.0-or-later
"""Correctness of the fused CUDA SO(2) value path used in training.

One CUDA operator spans the training value stream up to the attention
aggregation: the block-diagonal Wigner rotation of the gathered source
features, the radial degree mixing, the cross-focus competition weight, and
the gated SO(2) mixing stack. Its backward and second order are hand-derived
rather than traced, so they are arbitrated here against the eager autograd of
the same composition, which is exact for this fixed multilinear expression.

The comparison follows the conditioning argument of
:mod:`.conditioning`: both sides are evaluated in the working precision and
judged against the float64 evaluation of the reference, so a logic error is
separated from the reduction-order difference the fusion necessarily
introduces.
"""

from __future__ import (
    annotations,
)

import pytest
import torch

try:
    # Loads ``libdeepmd_op_pt.so``, which registers the hand-written operators.
    import deepmd.pt.cxx_op  # noqa: F401
except ImportError:
    pass

from deepmd.pt_expt.kernels.cuda.dpa4.so2_conv_train import (
    op_available,
)
from deepmd.pt_expt.kernels.triton.sezm.so2_value_path import (
    SO2_VALUE_PATH_TRITON_AVAILABLE,
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
        not SO2_VALUE_PATH_TRITON_AVAILABLE,
        reason="the eager reference lives in the Triton value-path module",
    ),
]

# ``(lmax, n_focus, focus_dim, mixing_layers, mixer_rank, focus_compete)``
# spanning the deployed DPA4 block shapes: the narrow two-focus block, the
# wider rank-2 mixer, the single-focus block without a competition head (which
# exercises the ``rank == 0`` degree-wise multiply), and the degree-six
# 384-channel Ultra layouts with either four 96-wide or three 128-wide focuses.
BLOCK_SHAPES = [
    (3, 2, 32, 3, 1, True),
    (5, 2, 64, 4, 2, True),
    (3, 1, 64, 3, 0, False),
    (6, 2, 96, 4, 1, True),
    (6, 4, 96, 4, 4, True),
    (6, 3, 128, 4, 4, True),
]

# Competition-head constants of the deployed configuration.
SOFTMAX_TAU = 1.0
LABEL_SMOOTHING = 0.02

LEAF_NAMES = (
    "x",
    "wigner",
    "kernel",
    "basis",
    "compete_w",
    "compete_b",
    "w0",
    "w1",
    "gw",
)


def _block_diagonal_mask(lmax: int, device: torch.device) -> torch.Tensor:
    """Structural support of the Wigner-D matrix, one block per degree."""
    dim = (lmax + 1) ** 2
    mask = torch.zeros(dim, dim, device=device, dtype=torch.float64)
    for degree in range(lmax + 1):
        base, width = degree * degree, 2 * degree + 1
        mask[base : base + width, base : base + width] = 1.0
    return mask


def _pack_wigner_rows(wigner: torch.Tensor, lmax: int) -> torch.Tensor:
    """Pack the m=0 and m=+-1 rows consumed by the reduced rotation."""
    m0, mm, mp = [], [], []
    for degree in range(lmax + 1):
        start, end = degree * degree, (degree + 1) ** 2
        row0 = start + degree
        m0.append(wigner[:, row0, start:end])
        if degree >= 1:
            mm.append(wigner[:, row0 - 1, start:end])
            mp.append(wigner[:, row0 + 1, start:end])
    return torch.cat(m0 + mm + mp, dim=1)


class _ValuePathCase:
    """One block shape with operands shared by every evaluation of it.

    The operands are drawn once in float64 and cast per evaluation, so the
    eager reference, the fused operator and the float64 ground truth all see
    the same numbers and the only difference between them is the arithmetic
    that consumes them. ``seed`` selects the draw, so a comparison can be
    repeated over independent operands.
    """

    def __init__(
        self,
        lmax: int,
        n_focus: int,
        focus_dim: int,
        layers: int,
        rank: int,
        compete: bool,
        *,
        seed: int,
        n_node: int = 512,
        n_edge: int = 2048,
    ) -> None:
        device = torch.device("cuda")
        torch.manual_seed(seed)
        self.lmax, self.n_focus, self.focus_dim = lmax, n_focus, focus_dim
        self.rank, self.compete = rank, compete
        self.n_edge, self.device = n_edge, device

        dim = (lmax + 1) ** 2
        c_wide = n_focus * focus_dim
        m0, m1 = (lmax + 1) * focus_dim, 2 * lmax * focus_dim
        n_gated = layers - 1
        double = {"device": device, "dtype": torch.float64}

        self.mask = _block_diagonal_mask(lmax, device)
        self.src = torch.randint(0, n_node, (n_edge,), device=device, dtype=torch.long)
        # The operator reads only the structural non-zeros of the rotation, so
        # the operand is masked to that support and the dense reference then
        # agrees with it by construction.
        wigner = torch.randn(n_edge, dim, dim, **double) * self.mask
        if rank == 0:
            kernel = torch.randn(n_edge, lmax + 1, c_wide, **double)
            basis = torch.zeros(1, **double)
        else:
            kernel_slots = dim + lmax * lmax
            kernel = 0.3 * torch.randn(n_edge, kernel_slots * rank, **double)
            basis = torch.randn(rank, c_wide, **double)
        self.operands = (
            torch.randn(n_node, dim, c_wide, **double),
            wigner,
            kernel,
            basis,
            0.05 * torch.randn(focus_dim, n_focus, **double),
            0.05 * torch.randn(n_focus, **double),
            0.2 * torch.randn(n_gated + 1, n_focus, m0, m0, **double),
            0.2 * torch.randn(n_gated + 1, n_focus, m1, m1, **double),
            0.3 * torch.randn(n_gated, n_focus, focus_dim, lmax * focus_dim, **double),
        )
        self.requires_grad = (
            True,
            True,
            True,
            rank > 0,
            compete,
            compete,
            True,
            True,
            True,
        )
        self.cotangent = torch.randn(
            n_edge, n_focus, (3 * lmax + 1) * focus_dim, **double
        )
        # A force loss differentiates the node features, the rotation and the
        # radial kernel again: their producers sit on the coordinate graph.
        # The Wigner cotangent lives on the same structural support.
        self.second_cotangents = (
            (0, torch.randn_like(self.operands[0])),
            (1, torch.randn_like(wigner) * self.mask),
            (2, torch.randn_like(kernel)),
        )

        order = torch.argsort(self.src, dim=0, stable=True)
        counts = self.src.new_zeros(n_node).scatter_add(
            0, self.src, torch.ones_like(self.src)
        )
        self.csr = (order, torch.cat([counts.new_zeros(1), torch.cumsum(counts, 0)]))

    @property
    def active_names(self) -> list[str]:
        """Names of the leaves this shape differentiates."""
        return [
            name
            for name, active in zip(LEAF_NAMES, self.requires_grad, strict=True)
            if active
        ]

    def quantity_names(self, second: bool) -> list[str]:
        """Labels of every quantity the evaluation reports."""
        names = ["fwd"] + [f"d/d {name}" for name in self.active_names]
        if second:
            names += [f"d2/d {name}" for name in self.active_names]
        return names

    def restrict(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Restrict a Wigner gradient to the structural block diagonal."""
        if name.endswith("wigner"):
            return tensor * self.mask
        return tensor

    def evaluate(
        self,
        *,
        fused: bool,
        dtype: torch.dtype,
        amp: bool,
        second: bool,
    ) -> tuple[torch.Tensor, ...]:
        """
        Run one evaluation of the value path and its differentiated forms.

        Parameters
        ----------
        fused : bool
            Whether to call the fused CUDA operator or the eager reference.
        dtype : torch.dtype
            Working precision of the leaves.
        amp : bool
            Whether to run inside bfloat16 autocast. The operator's autocast
            rule casts every floating-point input, so the reference is fed the
            same casts explicitly and both sides run one numerical regime.
        second : bool
            Whether to evaluate the second-order projection.

        Returns
        -------
        tuple of torch.Tensor
            The output and its first (and optionally second) order gradients.
        """
        from deepmd.pt_expt.kernels.cuda.dpa4.so2_conv_train import (
            _value_train_op,
        )
        from deepmd.pt_expt.kernels.triton.sezm.so2_value_path import (
            _mixing_stack_reference,
            _rotate_mix_reference,
        )

        leaves = tuple(
            operand.to(dtype).clone().requires_grad_(active)
            for operand, active in zip(self.operands, self.requires_grad, strict=True)
        )
        targets = [leaf for leaf in leaves if leaf.requires_grad]
        inputs = tuple(leaf.to(torch.bfloat16) for leaf in leaves) if amp else leaves
        x, wigner, kernel, basis, compete_w, compete_b, w0, w1, gw = inputs
        kernel_flat = kernel.reshape(self.n_edge, -1) if self.rank > 0 else kernel
        basis_flat = basis.reshape(-1) if self.rank > 0 else basis

        context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if amp
            else torch.autocast("cuda", enabled=False)
        )
        with context:
            if fused:
                out, *_ = _value_train_op(
                    x,
                    self.src,
                    self.csr[0],
                    self.csr[1],
                    _pack_wigner_rows(wigner, self.lmax),
                    kernel_flat,
                    basis_flat,
                    compete_w if self.compete else None,
                    compete_b if self.compete else None,
                    w0,
                    w1,
                    gw,
                    self.lmax,
                    self.n_focus,
                    self.rank,
                    self.compete,
                    SOFTMAX_TAU,
                    LABEL_SMOOTHING,
                )
            else:
                u0 = _rotate_mix_reference(
                    x,
                    self.src,
                    wigner,
                    kernel_flat,
                    basis_flat,
                    self.lmax,
                    self.n_focus,
                    self.rank,
                )
                out, *_ = _mixing_stack_reference(
                    u0,
                    self._competition(u0, compete_w, compete_b),
                    w0,
                    w1,
                    gw,
                    self.lmax,
                    self.focus_dim,
                    self.compete,
                )
        return grad_chain(
            out,
            targets,
            self.cotangent,
            self._second_targets(targets, leaves) if second else (),
        )

    def _competition(
        self,
        u0: torch.Tensor,
        compete_w: torch.Tensor,
        compete_b: torch.Tensor,
    ) -> torch.Tensor:
        """Label-smoothed cross-focus softmax over the scalar rows."""
        if not self.compete:
            return torch.ones(
                self.n_edge, self.n_focus, device=self.device, dtype=u0.dtype
            )
        gate = u0[:, :, : self.focus_dim].permute(1, 0, 2)
        logits = (
            torch.einsum("efi,if->ef", gate.float(), compete_w.float())
            + compete_b.float()
        )
        weights = torch.softmax(logits / SOFTMAX_TAU, dim=1)
        smoothed = weights * (1.0 - LABEL_SMOOTHING) + LABEL_SMOOTHING / self.n_focus
        return smoothed.to(u0.dtype)

    def _second_targets(
        self,
        targets: list[torch.Tensor],
        leaves: tuple[torch.Tensor, ...],
    ) -> list[tuple[int, torch.Tensor]]:
        """Map the second-order cotangents onto positions in ``targets``.

        The lookup is by identity: ``list.index`` would compare tensors
        elementwise.
        """
        positions = {id(leaf): index for index, leaf in enumerate(targets)}
        return [
            (positions[id(leaves[leaf_index])], cotangent)
            for leaf_index, cotangent in self.second_cotangents
        ]


# Independent operand draws the verdict is taken over. A per-focus gradient has
# as few as two entries, so one draw's extreme error is a noisy statistic; the
# median across draws is what the bound is applied to.
DRAW_SEEDS = (11, 2027, 40529)


def _compare(shape: tuple[int, int, int, int, int, bool], *, amp: bool) -> None:
    """Arbitrate the fused value path against the eager reference on ``shape``."""
    if not op_available():
        pytest.skip("the DPA4 CUDA training operators are unavailable")
    working = torch.bfloat16 if amp else torch.float32
    runs = []
    for seed in DRAW_SEEDS:
        case = _ValuePathCase(*shape, seed=seed)
        common = {"dtype": torch.float32, "amp": amp, "second": True}
        runs.append(
            deviations(
                case.quantity_names(second=True),
                case.evaluate(fused=False, dtype=torch.float64, amp=False, second=True),
                case.evaluate(fused=False, **common),
                case.evaluate(fused=True, **common),
                # The fusion holds every inter-layer activation in shared
                # memory and recovers each layer's input from the forward
                # output rather than storing it, so its rounding is
                # distributed differently from the eager graph's while
                # remaining the same magnitude. A logic error sits orders of
                # magnitude above that.
                factor=4.0,
                working_dtype=working,
                project=case.restrict,
            )
        )
    assert_conditioned(median_deviations(runs))


@pytest.mark.parametrize(
    ("lmax", "focus", "cf", "layers", "rank", "compete"), BLOCK_SHAPES
)
def test_float32_matches_eager_conditioning(
    lmax: int, focus: int, cf: int, layers: int, rank: int, compete: bool
) -> None:
    """Hold the fused value path to the eager reference's own float32 error."""
    _compare((lmax, focus, cf, layers, rank, compete), amp=False)


@pytest.mark.parametrize(
    ("lmax", "focus", "cf", "layers", "rank", "compete"), BLOCK_SHAPES
)
def test_autocast_bfloat16_matches_eager_conditioning(
    lmax: int, focus: int, cf: int, layers: int, rank: int, compete: bool
) -> None:
    """Hold the same bound under the bfloat16 autocast of production training."""
    _compare((lmax, focus, cf, layers, rank, compete), amp=True)


def test_float64_agrees_with_eager_to_reduction_order() -> None:
    """Separate logic from precision: in float64 both sides must coincide.

    The kernels keep float accumulators internally, so a float64 evaluation of
    the fused path and of the eager reference differ only by reduction order.
    Any structural disagreement -- a mis-indexed block, a dropped gradient
    term -- survives the precision increase and shows up here.
    """
    if not op_available():
        pytest.skip("the DPA4 CUDA training operators are unavailable")
    case = _ValuePathCase(*BLOCK_SHAPES[0], seed=DRAW_SEEDS[0])
    common = {"dtype": torch.float64, "amp": False, "second": True}
    reference = case.evaluate(fused=False, **common)
    fused = case.evaluate(fused=True, **common)
    for name, truth, got in zip(
        case.quantity_names(second=True), reference, fused, strict=True
    ):
        truth, got = case.restrict(name, truth), case.restrict(name, got)
        scale = truth.abs().max().clamp_min(1.0).item()
        error = (got - truth).abs().max().item() / scale
        assert error <= 5e-6, f"{name}: float64 disagreement {error:.3e}"
