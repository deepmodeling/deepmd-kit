# SPDX-License-Identifier: LGPL-3.0-or-later
"""Correctness of the direct-sum operator of the latent-charge long range.

The CUDA kernel evaluates every pair in float32 from positions relative to
the first node of each frame, carried as float32 head and tail, and sums in
compensated float32 before a float64 reduction; the total-charge constraint
and the per-frame reductions sum in float64 in a fixed order. It is held to
the float64 evaluation over the explicit pairs, at a tolerance relative to
the scale of each output, and to the exact invariants of a pair sum: padding
nodes and frames do not interact, the forces of a frame cancel, and the
energy, force, charge gradient and virial follow from one another.
"""

from __future__ import (
    annotations,
)

from typing import (
    Any,
)

import numpy as np
import pytest
import torch

from deepmd.dpmodel.long_range import (
    COULOMB_CONSTANT,
)
from deepmd.pt_expt.kernels.long_range import (
    _reference,
    kernel_operator_arguments,
    latent_charge_direct_sum,
    op_available,
)
from deepmd.pt_expt.long_range.kernel import (
    LESKernel,
    SOGKernel,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not op_available(),
    reason="the direct-sum operator is CUDA only",
)

KERNELS = {
    "les": lambda: LESKernel(alpha=0.8),
    "sog": lambda: SOGKernel(
        amplitude=[0.7, -0.2, 0.1], width=[0.8, 1.7, 3.2], precision="float32"
    ),
}


def _system(
    n_node: list[int], channels: int, padding: int, seed: int, offset: float
) -> tuple[torch.Tensor, ...]:
    """Frames of random nodes far from the origin, with trailing padding nodes.

    Every seventh node carries no charge, and every frame a total charge.
    """
    rng = np.random.default_rng(seed)
    total = sum(n_node) + padding
    coord = offset + rng.uniform(0.0, 4.0 * max(n_node) ** (1 / 3), (total, 3))
    charge = rng.normal(0.0, 0.5, (total, channels))
    device = torch.device("cuda")
    node_mask = torch.ones(total, dtype=torch.bool, device=device)
    node_mask[::7] = False
    return (
        torch.tensor(coord, dtype=torch.float64, device=device),
        torch.tensor(charge, dtype=torch.float32, device=device),
        node_mask,
        torch.tensor(n_node, dtype=torch.int64, device=device),
        torch.tensor(
            rng.integers(-2, 3, len(n_node)), dtype=torch.float64, device=device
        ),
    )


def _evaluate(
    system: tuple[torch.Tensor, ...], kernel: Any, channel: int
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    coord, charge, node_mask, n_node, total_charge = system
    actual = latent_charge_direct_sum(
        coord,
        charge,
        node_mask,
        n_node,
        total_charge,
        kernel,
        channel,
        COULOMB_CONSTANT,
    )
    code, coefficient = kernel_operator_arguments(kernel)
    expected = _reference(
        coord,
        charge.double(),
        node_mask,
        n_node,
        total_charge,
        coefficient.double(),
        torch.zeros(1, dtype=torch.int64),
        code,
        channel,
        COULOMB_CONSTANT,
    )
    return actual, expected


NAMES = ("atom_energy", "charge_gradient", "force", "atom_virial", "virial")


@pytest.mark.parametrize("kernel_name", ["les", "sog"])
@pytest.mark.parametrize(
    ("n_node", "channels", "padding", "channel"),
    [
        ([37, 1, 0, 300], 1, 5, 0),
        ([37, 1, 0, 300], 1, 5, -1),
        ([2048], 3, 0, -1),
        ([5, 129], 8, 3, 2),
    ],
)
def test_matches_float64_pair_sum(
    kernel_name: str, n_node: list[int], channels: int, padding: int, channel: int
) -> None:
    """Every output agrees with the float64 pair sum."""
    system = _system(n_node, channels, padding, 7, 1.0e3)
    actual, expected = _evaluate(system, KERNELS[kernel_name](), channel)
    _assert_close(actual, expected)
    # Padding nodes join no frame.
    if padding:
        for value in actual[:4]:
            assert torch.all(value[-padding:] == 0)


@pytest.mark.parametrize("alpha", [1e-3, 0.03, 0.3])
@pytest.mark.parametrize("channel", [0, -1])
def test_les_matches_float64_pair_sum_at_small_alpha_r(
    alpha: float, channel: int
) -> None:
    """The erf kernel keeps its precision where alpha r is small.

    The two terms of the slope of erf(x) / x agree to second order in x, so
    that the slope of an approximate erf loses precision in proportion to
    1 / x^2 at atomic distances and a small alpha.
    """
    system = _system([96], 2, 0, 5, 0.0)
    _assert_close(*_evaluate(system, LESKernel(alpha=alpha), channel))


def _assert_close(
    actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...]
) -> None:
    """Every output agrees with the float64 pair sum, relative to its largest entry."""
    for name, value, reference in zip(NAMES, actual, expected, strict=True):
        assert value.dtype == (
            torch.float64 if name == "atom_energy" else torch.float32
        )
        scale = reference.abs().max().item()
        error = (value.double() - reference.double()).abs().max().item()
        assert error <= 2e-5 * scale, (name, error, scale)


@pytest.mark.parametrize("kernel_name", ["les", "sog"])
def test_invariants(kernel_name: str) -> None:
    """Frames are independent, forces cancel, and the virial is the pair virial."""
    n_node = [64, 200]
    system = _system(n_node, 2, 0, 11, 0.0)
    coord, charge, node_mask, n_node_t, total_charge = system
    together = latent_charge_direct_sum(
        coord,
        charge,
        node_mask,
        n_node_t,
        total_charge,
        KERNELS[kernel_name](),
        1,
        COULOMB_CONSTANT,
    )
    _, _, force, atom_virial, virial = together
    # Each frame alone gives the same result.
    first = n_node[0]
    alone = latent_charge_direct_sum(
        coord[:first],
        charge[:first],
        node_mask[:first],
        n_node_t[:1],
        total_charge[:1],
        KERNELS[kernel_name](),
        1,
        COULOMB_CONSTANT,
    )
    for both, single in zip(together, alone, strict=True):
        torch.testing.assert_close(both[: single.shape[0]], single, rtol=0, atol=0)
    # Internal forces sum to zero per frame, and the virial of a frame is
    # sum_i r_i (x) f_i.
    for frame, (begin, end) in enumerate(((0, first), (first, sum(n_node)))):
        frame_force = force[begin:end].double()
        assert frame_force.sum(0).abs().max() <= 1e-5 * frame_force.abs().max()
        position = coord[begin:end] - coord[begin]
        expected = torch.einsum("ni,nj->ij", position, frame_force)
        scale = expected.abs().max().item()
        for value in (
            atom_virial[begin:end].double().sum(0).reshape(3, 3),
            virial[frame].double().reshape(3, 3),
        ):
            torch.testing.assert_close(value, expected, rtol=0, atol=2e-5 * scale)


@pytest.mark.parametrize("kernel_name", ["les", "sog"])
@pytest.mark.parametrize("channel", [-1, 1])
def test_reference_outputs_are_energy_derivatives(
    kernel_name: str, channel: int
) -> None:
    """The float64 force and charge gradient differentiate the energy."""
    coord, charge, node_mask, n_node, total_charge = _system([9, 6], 2, 2, 3, 0.0)
    coord = coord.clone().requires_grad_(True)
    charge = charge.double().requires_grad_(True)
    kernel = KERNELS[kernel_name]()
    code, coefficient = kernel_operator_arguments(kernel)
    atom_energy, charge_gradient, force, _, _ = _reference(
        coord,
        charge,
        node_mask,
        n_node,
        total_charge,
        coefficient.double(),
        torch.zeros(1, dtype=torch.int64),
        code,
        channel,
        COULOMB_CONSTANT,
    )
    grad_coord, grad_charge = torch.autograd.grad(atom_energy.sum(), (coord, charge))
    torch.testing.assert_close(force, -grad_coord, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(
        charge_gradient.detach(), grad_charge, rtol=1e-9, atol=1e-9
    )


def test_bitwise_reproducible() -> None:
    """Two evaluations of the same input agree bit for bit."""
    system = _system([3000], 1, 0, 5, 0.0)
    first = latent_charge_direct_sum(*system, KERNELS["les"](), 0, COULOMB_CONSTANT)
    second = latent_charge_direct_sum(*system, KERNELS["les"](), 0, COULOMB_CONSTANT)
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a, b)
