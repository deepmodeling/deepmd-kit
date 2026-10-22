# SPDX-License-Identifier: LGPL-3.0-or-later
"""Primitives of the long-range modules: pair topology, kernels and constraint."""

import itertools

import numpy as np
import pytest
from scipy.special import (
    erf,
)

from deepmd.dpmodel.fitting.latent_charge import (
    LatentChargeFitting,
)
from deepmd.dpmodel.long_range import (
    BaseInteractionKernel,
    LatentChargeLongRange,
    LESKernel,
    SOGKernel,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    build_k_grid,
    build_pair_graph,
    build_topologies,
    enumerate_k_grid_bounds,
    pair_force_virial,
    unordered_pair_index,
)


def _brute_force_pairs(counts: list[int], mask: np.ndarray | None) -> np.ndarray:
    pairs = []
    start = 0
    for count in counts:
        nodes = [
            node for node in range(start, start + count) if mask is None or mask[node]
        ]
        pairs += itertools.combinations(nodes, 2)
        start += count
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2).T


@pytest.mark.parametrize(
    "counts", [[0], [1], [2], [3, 0, 1, 5], [7, 200, 1, 0, 33], [1000]]
)
@pytest.mark.parametrize("masked", [False, True])
def test_pair_index_matches_brute_force(counts: list[int], masked: bool) -> None:
    """Every unordered pair of valid nodes of a frame appears once, in row-major order."""
    mask = np.random.default_rng(0).random(sum(counts)) > 0.3 if masked else None
    index = unordered_pair_index(np.asarray(counts), mask)
    np.testing.assert_array_equal(index, _brute_force_pairs(counts, mask))


def test_pair_graph_guards_and_periodic_rejection() -> None:
    """The pair axis ends in masked guards; a cell is rejected."""
    coord = np.random.default_rng(1).normal(size=(4, 3))
    atype = np.array([0, 1, 1, 0])
    pairs = build_pair_graph(coord, np.array([4]), atype)
    assert pairs.index.shape == (2, 8)
    assert pairs.mask.tolist() == [True] * 6 + [False] * 2
    np.testing.assert_allclose(
        pairs.vec[:6], coord[pairs.index[0, :6]] - coord[pairs.index[1, :6]]
    )
    assert build_topologies((), coord, np.array([4]), atype=atype) == {}
    with pytest.raises(NotImplementedError, match="non-periodic"):
        build_topologies(
            ("all_pairs",), coord, np.array([4]), np.eye(3)[None], atype=atype
        )


def test_pair_force_virial_of_a_central_potential() -> None:
    """For E = sum 1/r the assembled force is Newtonian and the virial sums."""
    coord = np.array([[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [0.0, 1.5, 0.3]])
    pairs = build_pair_graph(coord, np.array([3]), np.array([0, 1, 0]))
    # d(1/r)/d(vec) on the real pairs; the zero-length guards carry no gradient.
    r = np.where(pairs.mask, np.linalg.norm(pairs.vec, axis=-1), 1.0)
    grad = np.where(pairs.mask[:, None], -pairs.vec / r[:, None] ** 3, 0.0)
    force, atom_virial, virial = pair_force_virial(
        grad, pairs.vec, pairs.index, pairs.mask, np.array([3])
    )
    expected = np.zeros((3, 3))
    for i, j in itertools.permutations(range(3), 2):
        d = coord[i] - coord[j]
        expected[i] += d / np.linalg.norm(d) ** 3
    np.testing.assert_allclose(force, expected, rtol=1e-12)
    np.testing.assert_allclose(atom_virial.sum(0), virial[0], rtol=1e-12)


def test_les_kernel_closed_form() -> None:
    """LES is erf(alpha r)/r, finite at contact."""
    kernel = LESKernel(alpha=0.7)
    r = np.array([0.0, 0.5, 2.0, 9.0])
    values = kernel(r**2)
    np.testing.assert_allclose(values[1:], erf(0.7 * r[1:]) / r[1:], rtol=1e-10)
    np.testing.assert_allclose(values[0], 2 * 0.7 / np.sqrt(np.pi), rtol=1e-10)


def test_sog_kernel_closed_form_and_defaults() -> None:
    """SOG is sum_k a_k exp(-r^2 / 2 sigma_k^2); the defaults discretize 1/r."""
    kernel = SOGKernel(amplitude=[0.7, -0.2], width=[0.8, 1.7])
    r = np.array([0.0, 1.1, 3.0])
    expected = 0.7 * np.exp(-(r**2) / (2 * 0.8**2)) - 0.2 * np.exp(
        -(r**2) / (2 * 1.7**2)
    )
    np.testing.assert_allclose(kernel(r**2), expected, rtol=1e-12)
    default = SOGKernel(n_gaussian=3, base_width=2.0, width_ratio=1.5)
    np.testing.assert_allclose(np.exp(default.log_width), [2.0, 3.0, 4.5])
    np.testing.assert_allclose(
        default.amplitude, np.sqrt(2 / np.pi) * np.log(1.5) / np.array([2.0, 3.0, 4.5])
    )
    # The default kernel exerts the Coulomb force, r^2 K'(r) = -1, between a
    # few times its first width and a fraction of its widest one.
    default = SOGKernel()
    r = np.geomspace(6.5, 150.0, 200)
    step = 1e-4 * r
    slope = (default((r + step) ** 2) - default((r - step) ** 2)) / (2 * step)
    np.testing.assert_allclose(r**2 * slope, -1.0, rtol=0, atol=5e-3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"type": "les", "alpha": 0.0},
        {"type": "sog", "width": [1.0, -2.0]},
        {"type": "sog", "width": [1.0, 2.0], "amplitude": [1.0, 2.0, 3.0]},
        {"type": "sog", "width": [1.0, 2.0]},
        {"type": "sog", "width_ratio": 1.0},
        {"type": "sog", "amplitude": 1.0, "width_ratio": 0.5},
    ],
)
def test_kernel_parameter_domain(kwargs: dict) -> None:
    """Widths must be positive and increase by default; amplitudes match the Gaussians."""
    kernel_cls = BaseInteractionKernel.get_class_by_type(kwargs.pop("type"))
    with pytest.raises(ValueError):
        kernel_cls(**kwargs)


@pytest.mark.parametrize("kernel", [LESKernel(alpha=1.3), SOGKernel(n_gaussian=4)])
def test_kernel_round_trip(kernel: BaseInteractionKernel) -> None:
    """A kernel record restores the same parameters."""
    restored = BaseInteractionKernel.deserialize(kernel.serialize())
    r_sq = np.linspace(0.0, 30.0, 7)
    np.testing.assert_allclose(restored(r_sq), kernel(r_sq), rtol=0, atol=0)


def _long_range(constraint: dict | None, channels: int = 2) -> LatentChargeLongRange:
    charge_net = LatentChargeFitting(
        "latent_charge", 2, 4, channels, neuron=[4], numb_fparam=1, seed=1
    )
    return LatentChargeLongRange(charge_net, LESKernel(), constraint)


def test_constraint_shifts_one_channel_over_the_charged_atoms() -> None:
    """The constrained channel sums to the target over the charged atoms only."""
    long_range = _long_range({"source": "fparam", "channel": 1, "fparam_index": 0})
    rng = np.random.default_rng(2)
    charge = rng.normal(size=(7, 2))
    node_mask = np.array([True, True, False, True, True, True, False])
    charge[~node_mask] = 0.0
    n_node = np.array([4, 3])
    frame_id = np.array([0, 0, 0, 0, 1, 1, 1])
    fparam = np.array([[1.5], [-2.0]])
    constrained = long_range._constrain(
        charge, node_mask, frame_id, n_node, fparam, None
    )
    np.testing.assert_array_equal(constrained[:, 0], charge[:, 0])
    np.testing.assert_array_equal(constrained[~node_mask], 0.0)
    totals = np.bincount(frame_id, weights=constrained[:, 1], minlength=2)
    np.testing.assert_allclose(totals, [1.5, -2.0], rtol=1e-12)
    # A frame without charged atoms is left as it is.
    empty = long_range._constrain(
        charge * 0.0,
        np.zeros(7, dtype=bool),
        frame_id,
        n_node,
        fparam,
        None,
    )
    np.testing.assert_array_equal(empty, 0.0)


@pytest.mark.parametrize(
    "constraint",
    [
        {"source": "total"},
        {"source": "neutral", "channel": 2},
        {"source": "fparam", "fparam_index": 1},
    ],
)
def test_constraint_configuration_is_validated(constraint: dict) -> None:
    """Unknown sources and out-of-range channels or columns are rejected."""
    with pytest.raises(ValueError):
        _long_range(constraint)


def test_long_range_round_trip() -> None:
    """The module record holds the charge network, kernel and constraint."""
    long_range = _long_range({"source": "neutral"})
    record = long_range.serialize()
    restored = LatentChargeLongRange.deserialize(record)
    assert restored.serialize()["charge_constraint"] == record["charge_constraint"]
    assert restored.required_topologies() == ("all_pairs",)
    assert not restored.uses_charge_spin()


# ---------------------------------------------------------------------------
# Periodic (k-space) primitives
# ---------------------------------------------------------------------------


def _spherical_ft(
    kernel: str, k: float, n_r: int = 20000, r_max: float = 60.0
) -> float:
    """Isotropic Fourier transform of the kernel by direct radial quadrature."""
    r = np.linspace(1e-9, r_max, n_r)
    if kernel == "les":
        k_r = erf(0.8 * r) / r
    else:
        amplitude = np.array([0.7, -0.2, 0.1])
        width = np.array([0.8, 1.7, 3.2])
        k_r = np.sum(
            amplitude[None, :] * np.exp(-(r[:, None] ** 2) / (2 * width**2)), -1
        )
    return float(np.trapezoid(4.0 * np.pi * r**2 * k_r * np.sinc(k * r / np.pi), r))


@pytest.mark.parametrize("kernel", ["les", "sog"])
def test_kernel_kspace_matches_radial_ft(kernel: str) -> None:
    """``kernel.kspace`` is the analytic Fourier transform of the real-space kernel."""
    k_sq = np.array([0.5, 1.0, 2.0, 5.0]) ** 2
    if kernel == "les":
        mod = LESKernel(alpha=0.8)
    else:
        mod = SOGKernel(amplitude=[0.7, -0.2, 0.1], width=[0.8, 1.7, 3.2])
    got = mod.kspace(k_sq)
    for kk, value in zip(np.sqrt(k_sq), got, strict=True):
        if kernel == "les":
            # erf(alpha r)/r is not integrable; quadrature of the integrable
            # complement erfc(alpha r)/r = 1/r - erf(alpha r)/r converges, and
            # K_hat(k) = 4 pi / k^2 - FT(erfc(alpha r)/r).
            r = np.linspace(1e-9, 60.0, 20000)
            k_r = (1.0 - erf(0.8 * r)) / r  # erfc(0.8 r)/r
            expected = 4.0 * np.pi / kk**2 - float(
                np.trapezoid(4.0 * np.pi * r**2 * k_r * np.sinc(kk * r / np.pi), r)
            )
        else:
            expected = _spherical_ft(kernel, kk)
        np.testing.assert_allclose(value, expected, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("kernel", ["les", "sog"])
def test_kernel_contact_value_matches_real_space_limit(kernel: str) -> None:
    """``kernel.contact_value`` is K(r) at contact."""
    if kernel == "les":
        mod = LESKernel(alpha=0.8)
    else:
        mod = SOGKernel(amplitude=[0.7, -0.2, 0.1], width=[0.8, 1.7, 3.2])
    r_sq = np.array([1e-12])
    np.testing.assert_allclose(
        mod.contact_value(r_sq), mod(r_sq)[0], rtol=1e-6, atol=1e-9
    )


def test_build_k_grid_orthogonality_and_guards() -> None:
    """The k points satisfy k . a_d = 2 pi n_d, are ordered and carry guards."""
    box = np.array([[8.0, 0.6, 0.0], [0.0, 7.5, 0.4], [0.0, 0.0, 9.0]])[None]
    grid = build_k_grid(box, 2.0, np.array([4]), coord=np.zeros((4, 3)))
    # guard points are two masked (1e-6, 0, 0) entries
    assert grid.mask[-2:].tolist() == [False, False]
    assert grid.mask[:-2].all()
    n_integer = grid.vec[:-2] @ box[0].T / (2 * np.pi)
    np.testing.assert_allclose(n_integer, np.round(n_integer), atol=1e-10)
    k_sq = np.sum(grid.vec[:-2] ** 2, axis=-1)
    assert (k_sq > 0).all() and (k_sq <= 4.0 + 1e-12).all()
    np.testing.assert_allclose(grid.volume, np.abs(np.linalg.det(box[0])))
    # every integer point inside the sphere is present exactly once
    ns = np.round(n_integer).astype(int)
    assert len({tuple(row) for row in ns}) == ns.shape[0]
    rec = 2 * np.pi * np.linalg.inv(box[0])
    n_d = np.ceil(2.0 * np.linalg.norm(box[0], axis=1) / (2 * np.pi)).astype(int)
    count = 0
    for n0 in range(-n_d[0], n_d[0] + 1):
        for n1 in range(-n_d[1], n_d[1] + 1):
            for n2 in range(-n_d[2], n_d[2] + 1):
                if (n0, n1, n2) != (0, 0, 0):
                    k = rec @ np.array([n0, n1, n2], dtype=float)
                    count += int(np.sum(k * k) <= 4.0)
    assert count == ns.shape[0]


def test_build_topologies_periodic_branching() -> None:
    """A cell switches the pair topology to guards and the k grid on, and back."""
    from deepmd.dpmodel.utils.neighbor_graph import (
        ALL_PAIRS,
        K_GRID,
    )

    coord = np.random.default_rng(1).normal(size=(4, 3))
    atype = np.array([0, 1, 1, 0])
    n_node = np.array([4])
    box = np.eye(3)[None] * 10.0
    names = (ALL_PAIRS, K_GRID)
    params = {K_GRID: {"k_max": 2.0}}
    with_box = build_topologies(
        names, coord, n_node, box, atype=atype, topology_params=params
    )
    assert with_box[ALL_PAIRS].index.shape[1] == 2  # guards only
    assert not with_box[ALL_PAIRS].mask.any()
    assert with_box[K_GRID].mask.sum() > 0
    without_box = build_topologies(
        names, coord, n_node, None, atype=atype, topology_params=params
    )
    assert without_box[ALL_PAIRS].mask.any()
    assert not without_box[K_GRID].mask.any()  # guards only
    # a pair-only declaration still rejects a cell loudly
    with pytest.raises(NotImplementedError, match="non-periodic"):
        build_topologies((ALL_PAIRS,), coord, n_node, box, atype=atype)
    # the k grid needs its build parameter
    with pytest.raises(ValueError, match="k_max"):
        build_topologies(names, coord, n_node, box, atype=atype)


# ---------------------------------------------------------------------------
# Fixed k-grid mode
# ---------------------------------------------------------------------------


def _periodic_long_range(**kwargs: object) -> LatentChargeLongRange:
    charge_net = LatentChargeFitting("latent_charge", 2, 4, 1, neuron=[4], seed=1)
    return LatentChargeLongRange(
        charge_net, LESKernel(alpha=0.3), None, periodic=True, **kwargs
    )


def _cells() -> np.ndarray:
    """Two cells of different sizes, the second the larger one."""
    return np.array(
        [
            [[8.0, 0.6, 0.0], [0.0, 7.5, 0.4], [0.0, 0.0, 9.0]],
            [[12.0, 0.5, 0.0], [0.2, 11.0, 0.3], [0.1, 0.4, 10.5]],
        ]
    )


def test_enumerate_k_grid_bounds() -> None:
    """The grid is the lexicographic meshgrid of the bounds, without the zero vector."""
    n_grid = enumerate_k_grid_bounds([2, 1, 0])
    assert n_grid.shape == (5 * 3 * 1 - 1, 3)
    assert not np.any(np.all(n_grid == 0, axis=-1))
    expected = [
        (n0, n1, 0) for n0 in range(-2, 3) for n1 in range(-1, 2) if (n0, n1) != (0, 0)
    ]
    np.testing.assert_array_equal(n_grid, np.asarray(expected, dtype=np.int64))
    with pytest.raises(ValueError, match="bounds"):
        enumerate_k_grid_bounds([1, -1, 0])


def test_fixed_k_grid_matches_ragged_enumeration() -> None:
    """The masked points of the fixed grid are exactly the ragged k points."""
    from deepmd.dpmodel.utils.neighbor_graph import (
        build_fixed_k_grid,
        enumerate_k_grid_bounds,
        k_grid_bounds_of_cells,
    )

    box = _cells()
    k_max = 2.0
    n_grid = enumerate_k_grid_bounds(np.max(k_grid_bounds_of_cells(box, k_max), axis=0))
    nk = n_grid.shape[0]
    fixed = build_fixed_k_grid(box, n_grid, k_max, np.array([4, 5]))
    # a static, guard-free layout: nf * nk points, frame-major
    assert fixed.vec.shape == (2 * nk, 3)
    np.testing.assert_array_equal(fixed.frame, np.repeat([0, 1], nk))
    ragged = build_k_grid(box, k_max, np.array([4, 5]))
    for frame in range(2):
        expected = {
            tuple(row)
            for row in np.round(ragged.vec[(ragged.frame == frame) & ragged.mask], 8)
        }
        got = {
            tuple(row)
            for row in np.round(fixed.vec[(fixed.frame == frame) & fixed.mask], 8)
        }
        assert got == expected
    # the smaller cell masks more points of the shared grid
    assert fixed.mask[:nk].sum() < fixed.mask[nk:].sum()
    np.testing.assert_allclose(fixed.volume, np.abs(np.linalg.det(box)), rtol=1e-12)


def test_fixed_k_grid_truncates_oversized_cells() -> None:
    """A cell beyond the grid is served by the grid alone (truncated k-sum), with a warning."""
    from deepmd.dpmodel.utils.neighbor_graph import (
        build_fixed_k_grid,
        enumerate_k_grid_bounds,
        k_grid_bounds_of_cells,
    )

    box = _cells()
    k_max = 2.0
    n_grid = enumerate_k_grid_bounds(np.max(k_grid_bounds_of_cells(box, k_max), axis=0))
    nk = n_grid.shape[0]
    with pytest.warns(UserWarning, match="truncated"):
        bigger = build_fixed_k_grid(box * 2.0, n_grid, k_max, np.array([4, 5]))
    # the layout is untouched: static, guard-free, every grid point present
    assert bigger.vec.shape == (2 * nk, 3)
    np.testing.assert_array_equal(bigger.frame, np.repeat([0, 1], nk))
    # every grid point is still a valid wave number of the larger cell; the
    # mask keeps the whole k_max sphere the grid reaches
    k2 = np.sum(bigger.vec**2, axis=-1)
    np.testing.assert_array_equal(bigger.mask, (k2 > 0) & (k2 <= k_max * k_max))
    # the larger cell activates more of the shared grid than the original did
    original = build_fixed_k_grid(box, n_grid, k_max, np.array([4, 5]))
    assert bigger.mask.sum() > original.mask.sum()
    # a larger wave-number cutoff (a drifting kernel) truncates the same way
    with pytest.warns(UserWarning, match="truncated"):
        build_fixed_k_grid(box, n_grid, k_max * 2.0, np.array([4, 5]))


def test_fixed_k_grid_mode_configuration() -> None:
    """The fixed mode requires periodicity, and the bounds the fixed mode."""
    with pytest.raises(ValueError, match="k_grid_mode"):
        _periodic_long_range(k_grid_mode="dense")
    with pytest.raises(ValueError, match="periodic"):
        LatentChargeLongRange(
            LatentChargeFitting("latent_charge", 2, 4, 1, neuron=[4], seed=1),
            LESKernel(),
            None,
            k_grid_mode="fixed",
        )
    with pytest.raises(ValueError, match="k_grid_bounds"):
        _periodic_long_range(k_grid_bounds=[3, 3, 3])
    # explicit bounds build the grid at construction, no statistics needed
    long_range = _periodic_long_range(k_grid_mode="fixed", k_grid_bounds=[2, 2, 2])
    assert long_range.k_grid_n.shape == (5**3 - 1, 3)


def test_fixed_k_grid_stats_size_the_grid() -> None:
    """The data statistics size the grid by the largest sampled cell."""
    long_range = _periodic_long_range(k_grid_mode="fixed")
    # the grid is not sized before the statistics run
    coord = np.zeros((9, 3))
    with pytest.raises(ValueError, match="not sized yet"):
        long_range.build_topologies(
            coord, np.array([4, 5]), _cells(), atype=np.zeros(9, dtype=np.int64)
        )
    long_range.compute_input_stats(
        [{"box": _cells()[:1].reshape(1, 9)}, {"box": _cells()}]
    )
    k_max = long_range.kernel.kspace_cutoff(long_range.kspace_tolerance)
    from deepmd.dpmodel.utils.neighbor_graph import (
        k_grid_bounds_of_cells,
    )

    expected = np.max(k_grid_bounds_of_cells(_cells(), k_max), axis=0)
    np.testing.assert_array_equal(np.max(np.abs(long_range.k_grid_n), axis=0), expected)
    topologies = long_range.build_topologies(
        coord, np.array([4, 5]), _cells(), atype=np.zeros(9, dtype=np.int64)
    )
    assert topologies["k_grid"].vec.shape == (2 * long_range.k_grid_n.shape[0], 3)
    # statistics without a cell cannot size the grid
    unsized = _periodic_long_range(k_grid_mode="fixed")
    with pytest.raises(ValueError, match="no simulation cell"):
        unsized.compute_input_stats([{"coord": np.zeros((1, 3))}])


def test_fixed_k_grid_round_trip() -> None:
    """The record keeps the grid mode and the sized grid."""
    long_range = _periodic_long_range(k_grid_mode="fixed")
    long_range.compute_input_stats([{"box": _cells()}])
    record = long_range.serialize()
    assert record["k_grid_mode"] == "fixed"
    restored = LatentChargeLongRange.deserialize(record)
    assert restored.k_grid_mode == "fixed"
    np.testing.assert_array_equal(restored.k_grid_n, long_range.k_grid_n)
    # the default mode serializes no grid fields
    plain = LatentChargeLongRange.deserialize(_periodic_long_range().serialize())
    assert plain.k_grid_mode == "ragged"
    assert plain.k_grid_n is None


# ---------------------------------------------------------------------------
# Decoupled k grid
# ---------------------------------------------------------------------------


def test_decoupled_k_grid_requires_periodic() -> None:
    """Decoupling is only meaningful for a periodic module."""
    with pytest.raises(ValueError, match="decouple_k_grid"):
        LatentChargeLongRange(
            LatentChargeFitting("latent_charge", 2, 4, 1, neuron=[4], seed=1),
            LESKernel(),
            None,
            decouple_k_grid=True,
        )


def test_decoupled_k_grid_ignores_kernel_drift() -> None:
    """A decoupled grid keeps the k points of the initial kernel however the
    kernel parameters move; a coupled one re-derives them every batch.
    """
    coord = np.random.default_rng(3).normal(size=(5, 3))
    atype = np.zeros(5, dtype=np.int64)
    n_node = np.array([5])
    box = np.eye(3)[None] * 8.0

    decoupled = _periodic_long_range(decouple_k_grid=True)
    coupled = _periodic_long_range()
    # the frozen cutoff is the cutoff of the initial kernel
    assert decoupled.frozen_k_max == pytest.approx(
        LESKernel(alpha=0.3).kspace_cutoff(1e-8)
    )
    initial = decoupled.build_topologies(coord, n_node, box, atype=atype)["k_grid"]
    reference = coupled.build_topologies(coord, n_node, box, atype=atype)["k_grid"]
    np.testing.assert_array_equal(initial.vec, reference.vec)
    # a larger alpha moves the cutoff out, growing the coupled grid
    for module in (decoupled, coupled):
        module.kernel.log_alpha = np.log(1.5)
    drifted = decoupled.build_topologies(coord, n_node, box, atype=atype)["k_grid"]
    moved = coupled.build_topologies(coord, n_node, box, atype=atype)["k_grid"]
    np.testing.assert_array_equal(drifted.vec, initial.vec)
    np.testing.assert_array_equal(drifted.mask, initial.mask)
    assert moved.vec.shape[0] > initial.vec.shape[0]


def test_decoupled_fixed_grid_keeps_its_mask() -> None:
    """The mask of a decoupled fixed grid ignores the kernel drift."""
    coord = np.zeros((5, 3))
    atype = np.zeros(5, dtype=np.int64)
    n_node = np.array([5])
    box = np.eye(3)[None] * 8.0

    long_range = _periodic_long_range(
        k_grid_mode="fixed", k_grid_bounds=[5, 5, 5], decouple_k_grid=True
    )
    before = long_range.build_topologies(coord, n_node, box, atype=atype)["k_grid"]
    # a smaller alpha moves the cutoff in; a coupled mask would tighten
    long_range.kernel.log_alpha = np.log(0.05)
    after = long_range.build_topologies(coord, n_node, box, atype=atype)["k_grid"]
    np.testing.assert_array_equal(after.mask, before.mask)
    np.testing.assert_array_equal(after.vec, before.vec)


def test_decoupled_k_grid_round_trip() -> None:
    """The frozen cutoff survives serialization, past the kernel drift."""
    long_range = _periodic_long_range(decouple_k_grid=True)
    frozen = long_range.frozen_k_max
    long_range.kernel.log_alpha = np.log(1.5)
    record = long_range.serialize()
    assert record["decouple_k_grid"] is True
    assert record["frozen_k_max"] == pytest.approx(frozen)
    restored = LatentChargeLongRange.deserialize(record)
    assert restored.decouple_k_grid
    assert restored.frozen_k_max == pytest.approx(frozen)
    # ...even though the restored (drifted) kernel would derive a different one
    assert restored.kernel.kspace_cutoff(1e-8) != pytest.approx(frozen)
    # the default stays coupled and serializes nothing
    plain = LatentChargeLongRange.deserialize(_periodic_long_range().serialize())
    assert not plain.decouple_k_grid
    assert plain.frozen_k_max is None
