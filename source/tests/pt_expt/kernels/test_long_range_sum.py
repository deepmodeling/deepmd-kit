# SPDX-License-Identifier: LGPL-3.0-or-later
"""Correctness of the operator of the latent-charge long range.

For non-periodic frames the CUDA kernel evaluates every pair in float32 from
positions relative to the first node of each frame, carried as float32 head
and tail, and sums in compensated float32 before a float64 reduction; the
total-charge constraint and the per-frame reductions sum in float64 in a fixed
order. It is held to the float64 evaluation over the explicit pairs, at a
tolerance relative to the scale of each output, and to the exact invariants of
a pair sum: padding nodes and frames do not interact, the forces of a frame
cancel, and the energy, force, charge gradient and virial follow from one
another.

For periodic frames the CUDA kernels evaluate the mesh sum in float32 on
B-spline weights formed in float64, and are held to the float64 mesh
evaluation of the training path, differentiated by autograd.
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
from deepmd.dpmodel.utils.neighbor_graph import (
    mesh_dims_of_cells,
)
from deepmd.pt_expt.kernels.long_range import (
    _reference,
    kernel_operator_arguments,
    latent_charge_sum,
    op_available,
)
from deepmd.pt_expt.long_range.kernel import (
    LESKernel,
    SOGKernel,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not op_available(),
    reason="the long-range operator is CUDA only",
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


def _reference_of(
    system: tuple[torch.Tensor, ...],
    kernel: Any,
    channel: int,
    box: torch.Tensor | None = None,
    order: int = 8,
    spacing: float = 1.0,
) -> tuple[torch.Tensor, ...]:
    """The float64 reference of the operator."""
    coord, charge, node_mask, n_node, total_charge = system
    code, coefficient = kernel_operator_arguments(kernel)
    if box is None:
        box = torch.zeros(n_node.shape[0], 9, dtype=torch.float64)
    return _reference(
        coord,
        charge.double(),
        node_mask,
        n_node,
        total_charge,
        box,
        coefficient,
        torch.zeros(1, dtype=torch.int64),
        code,
        channel,
        COULOMB_CONSTANT,
        order,
        spacing,
    )


def _evaluate(
    system: tuple[torch.Tensor, ...],
    kernel: Any,
    channel: int,
    box: torch.Tensor | None = None,
    order: int = 8,
    spacing: float = 1.0,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    actual = latent_charge_sum(
        *system,
        kernel,
        channel,
        COULOMB_CONSTANT,
        box=box,
        order=order,
        spacing=spacing,
    )
    return actual, _reference_of(system, kernel, channel, box, order, spacing)


NAMES = ("atom_energy", "charge_gradient", "force", "atom_virial", "virial")


@pytest.mark.parametrize("kernel_name", ["les", "sog"])
@pytest.mark.parametrize(
    ("n_node", "channels", "padding", "channel"),
    [
        ([37, 1, 0, 300], 1, 5, 0),
        ([37, 1, 0, 300], 1, 5, -1),
        ([2048], 3, 0, -1),
        ([5, 129], 8, 3, 2),
        # A frame of 47 owner blocks of 24 chunks, which takes the wide tiles
        # of the pair kernel on devices of up to 3384 resident block slots.
        ([6000], 2, 4, 1),
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
    actual: tuple[torch.Tensor, ...],
    expected: tuple[torch.Tensor, ...],
    tolerance: float = 2e-5,
) -> None:
    """Every output agrees with the float64 reference, relative to its largest entry."""
    for name, value, reference in zip(NAMES, actual, expected, strict=True):
        assert value.dtype == (
            torch.float64 if name == "atom_energy" else torch.float32
        )
        scale = reference.abs().max().item()
        error = (value.double() - reference.double()).abs().max().item()
        assert error <= tolerance * scale, (name, error, scale)


@pytest.mark.parametrize(
    ("kernel_name", "n_node"),
    [
        ("les", [64, 200]),
        ("sog", [64, 200]),
        # A first frame that ends inside a block of the pair kernel, whose
        # chunks then reach past the partners of that frame. The erf kernel
        # decides its far field over the lanes of a warp, which such a block
        # shares between the frames, so that it is checked on frames of whole
        # warps.
        ("sog", [150, 200]),
    ],
)
def test_invariants(kernel_name: str, n_node: list[int]) -> None:
    """Frames are independent, forces cancel, and the virial is the pair virial."""
    system = _system(n_node, 2, 0, 11, 0.0)
    coord, charge, node_mask, n_node_t, total_charge = system
    together = latent_charge_sum(
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
    # The first frame alone gives the same result.
    first = n_node[0]
    alone = latent_charge_sum(
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
    atom_energy, charge_gradient, force, _, _ = _reference_of(
        (coord, charge, node_mask, n_node, total_charge),
        KERNELS[kernel_name](),
        channel,
    )
    grad_coord, grad_charge = torch.autograd.grad(atom_energy.sum(), (coord, charge))
    torch.testing.assert_close(force, -grad_coord, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(
        charge_gradient.detach(), grad_charge, rtol=1e-9, atol=1e-9
    )


def test_bitwise_reproducible() -> None:
    """Two evaluations of the same input agree bit for bit, with or without cells."""
    system = _system([3000], 1, 0, 5, 0.0)
    for box in (None, _cells(1, 40.0)):
        first = latent_charge_sum(
            *system, KERNELS["les"](), 0, COULOMB_CONSTANT, box=box
        )
        second = latent_charge_sum(
            *system, KERNELS["les"](), 0, COULOMB_CONSTANT, box=box
        )
        for a, b in zip(first, second, strict=True):
            assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# Periodic frames
# ---------------------------------------------------------------------------


def _cells(frame_count: int, length: float, seed: int = 3) -> torch.Tensor:
    """Triclinic cells of about ``length`` per side, row-major, one row per frame."""
    rng = np.random.default_rng(seed)
    cells = []
    for _ in range(frame_count):
        cell = np.diag(length * rng.uniform(0.8, 1.2, 3))
        cell[1, 0], cell[2, 0], cell[2, 1] = length * rng.uniform(-0.2, 0.2, 3)
        cells.append(cell.reshape(9))
    return torch.tensor(np.asarray(cells), dtype=torch.float64)


@pytest.mark.parametrize("kernel_name", ["les", "sog"])
@pytest.mark.parametrize(
    ("n_node", "channels", "padding", "channel", "order"),
    [
        ([37, 1, 300], 1, 5, 0, 8),
        ([37, 1, 300], 1, 5, -1, 4),
        ([500], 3, 0, -1, 6),
        ([5, 129], 8, 3, 2, 8),
    ],
)
def test_mesh_matches_float64_reference(
    kernel_name: str,
    n_node: list[int],
    channels: int,
    padding: int,
    channel: int,
    order: int,
) -> None:
    """Every output of a periodic frame agrees with the mesh evaluation of training.

    The positions extend beyond the cell, so that the fractional coordinates
    wrap.
    """
    system = _system(n_node, channels, padding, 7, -5.0)
    box = _cells(len(n_node), 12.0)
    actual, expected = _evaluate(
        system, KERNELS[kernel_name](), channel, box, order, 0.6
    )
    _assert_close(actual, expected, 1e-6)
    if padding:
        for value in actual[:4]:
            assert torch.all(value[-padding:] == 0)


@pytest.mark.parametrize("kernel_name", ["les", "sog"])
def test_mesh_virial_is_the_strain_derivative(kernel_name: str) -> None:
    """The virial of a periodic frame is the strain derivative of its energy.

    Straining the positions and the cell together by ``I + eps`` changes the
    energy by ``-W : eps``; the atomic virial of the charged nodes sums to W.
    The central difference takes the float64 mesh energy, whose rounding stays
    far below that of the float32 outputs of the operator.
    """
    system = _system([60], 2, 0, 13, 0.0)
    coord, charge, node_mask, n_node, total_charge = system
    box = _cells(1, 10.0)
    kernel = KERNELS[kernel_name]()
    _, _, _, atom_virial, virial = latent_charge_sum(
        *system, kernel, 0, COULOMB_CONSTANT, box=box, spacing=0.5
    )
    h = 1e-5
    finite = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            energies = []
            for sign in (1.0, -1.0):
                strain = torch.eye(3, dtype=torch.float64)
                strain[a, b] += sign * h
                strained = (
                    coord @ strain.T.to(coord.device),
                    charge,
                    node_mask,
                    n_node,
                    total_charge,
                )
                cell = box.reshape(3, 3) @ strain.T
                # Both strained cells keep the mesh of the cell.
                assert mesh_dims_of_cells(cell.reshape(1, 9).numpy(), 0.5) == (
                    mesh_dims_of_cells(box.numpy(), 0.5)
                )
                energies.append(
                    _reference_of(strained, kernel, 0, cell.reshape(1, 9), 8, 0.5)[
                        0
                    ].sum()
                )
            finite[a, b] = -(energies[0] - energies[1]).item() / (2 * h)
    scale = np.abs(finite).max()
    np.testing.assert_allclose(
        virial[0].double().cpu().numpy().reshape(3, 3), finite, atol=2e-6 * scale
    )
    np.testing.assert_allclose(
        atom_virial.double().sum(0).cpu().numpy().reshape(3, 3),
        finite,
        atol=2e-6 * scale,
    )


@pytest.mark.parametrize("order", [2, 3, 5, 7, 9, 10, 11, 12])
def test_mesh_orders(order: int) -> None:
    """Every B-spline order agrees with the mesh evaluation of training.

    The atomic energies of these small cells are a fraction of the self
    energy, to a few 1e-7 of which the float32 mesh resolves them.
    """
    system = _system([40, 23], 2, 3, 5, -5.0)
    box = _cells(2, 9.0)
    actual, expected = _evaluate(system, KERNELS["sog"](), 1, box, order, 0.7)
    _assert_close(actual, expected, 1e-5)


def test_mesh_stencil_wider_than_the_axis() -> None:
    """A stencil wider than an axis of the mesh wraps around it more than once."""
    system = _system([12], 1, 0, 9, 0.0)
    box = _cells(1, 3.0)
    # Four points per axis, against eight of the stencil.
    assert mesh_dims_of_cells(box.numpy(), 1.0) == (4, 4, 4)
    actual, expected = _evaluate(system, KERNELS["les"](), 0, box, 8, 1.0)
    _assert_close(actual, expected, 1e-6)


@pytest.mark.parametrize("periodic", [False, True])
def test_frame_without_charged_nodes(periodic: bool) -> None:
    """A frame whose nodes carry no charge contributes zeros, the others are unaffected."""
    n_node = [40, 6, 50]
    coord, charge, node_mask, n_node_t, total_charge = _system(n_node, 2, 0, 5, 0.0)
    node_mask[40:46] = False
    system = (coord, charge, node_mask, n_node_t, total_charge)
    box = _cells(3, 10.0) if periodic else None
    actual, expected = _evaluate(system, KERNELS["les"](), 0, box, 8, 0.6)
    _assert_close(actual, expected, 1e-6)
    for value in actual[:4]:
        assert torch.all(value[40:46] == 0)
    assert torch.all(actual[4][1] == 0)


def test_mesh_frames_are_independent() -> None:
    """Every periodic frame is evaluated on its own mesh, bit for bit."""
    n_node = [64, 200]
    system = _system(n_node, 2, 0, 11, 0.0)
    coord, charge, node_mask, n_node_t, total_charge = system
    box = _cells(2, 9.0)
    kernel = KERNELS["sog"]()
    together = latent_charge_sum(
        *system, kernel, 1, COULOMB_CONSTANT, box=box, spacing=0.5
    )
    first = n_node[0]
    alone = latent_charge_sum(
        coord[:first],
        charge[:first],
        node_mask[:first],
        n_node_t[:1],
        total_charge[:1],
        kernel,
        1,
        COULOMB_CONSTANT,
        box=box[:1],
        spacing=0.5,
    )
    for both, single in zip(together, alone, strict=True):
        torch.testing.assert_close(both[: single.shape[0]], single, rtol=0, atol=0)


def _mesh_sizes(box: torch.Tensor, spacing: float) -> list[tuple[int, int, int]]:
    """The mesh size of every frame of ``box``."""
    return [mesh_dims_of_cells(cell.reshape(1, 9).numpy(), spacing) for cell in box]


def test_captured_meshes_read_the_cells_at_replay() -> None:
    """A CUDA graph of periodic frames evaluates the cells of its box at every replay.

    The graph reads the cells where they lie in pinned memory, so that a
    replay after a strain that keeps the sizes of the meshes agrees bit for
    bit with an eager evaluation of the strained cells; cells in pageable
    memory cannot be captured.
    """
    system = _system([64, 200], 2, 0, 11, 0.0)
    kernel = KERNELS["sog"]().to("cuda")

    def evaluate(box: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return latent_charge_sum(
            *system, kernel, 1, COULOMB_CONSTANT, box=box, spacing=0.5
        )

    cells = _cells(2, 9.0)
    strained = cells * 1.005
    assert _mesh_sizes(strained, 0.5) == _mesh_sizes(cells, 0.5)
    expected = {"first": evaluate(cells), "strained": evaluate(strained)}
    with pytest.raises(RuntimeError, match="pinned"):
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            evaluate(cells)
    box = cells.pin_memory()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = evaluate(box)
    for name, cell in (("strained", strained), ("first", cells)):
        box.copy_(cell)
        graph.replay()
        torch.cuda.synchronize()
        for value, reference in zip(captured, expected[name], strict=True):
            torch.testing.assert_close(
                value, reference, rtol=0, atol=0, msg=lambda m, n=name: f"{n}: {m}"
            )


def test_captured_meshes_outlive_other_shapes() -> None:
    """A CUDA graph of periodic frames replays after 17 other mesh shapes, bit for bit.

    The operator keeps the transforms of 16 mesh shapes and releases the least
    recently used past them, except those of a live graph, which runs them at
    every replay.
    """
    system = _system([64, 200], 2, 0, 11, 0.0)
    kernel = KERNELS["sog"]().to("cuda")

    def evaluate(box: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return latent_charge_sum(
            *system, kernel, 1, COULOMB_CONSTANT, box=box, spacing=0.5
        )

    box = _cells(2, 9.0).pin_memory()
    expected = evaluate(box)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = evaluate(box)
    others: set[tuple[int, int, int]] = set()
    length = 9.0
    while len(others) < 17:
        length += 0.5
        cells = _cells(2, length)
        evaluate(cells)
        others |= set(_mesh_sizes(cells, 0.5)) - set(_mesh_sizes(box, 0.5))
    graph.replay()
    for value, reference in zip(captured, expected, strict=True):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)


def test_a_capture_needs_an_eager_call_for_its_meshes() -> None:
    """A CUDA graph captures only meshes that an eager call has evaluated."""
    system = _system([12], 1, 0, 9, 0.0)
    kernel = KERNELS["les"]().to("cuda")
    # A cell whose mesh of 16 x 16 x 240 points no other test evaluates.
    box = torch.tensor(
        [[4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 60.0]], dtype=torch.float64
    ).pin_memory()
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="none has evaluated the mesh of 16 x 16"):
        with torch.cuda.graph(graph):
            latent_charge_sum(
                *system, kernel, 0, COULOMB_CONSTANT, box=box, spacing=0.25
            )


def test_mixed_frames_are_rejected() -> None:
    """A call holds periodic frames or non-periodic ones, not both."""
    system = _system([10, 10], 1, 0, 1, 0.0)
    box = _cells(2, 9.0)
    box[1] = 0.0
    with pytest.raises(RuntimeError, match="periodic"):
        latent_charge_sum(*system, KERNELS["les"](), 0, COULOMB_CONSTANT, box=box)
