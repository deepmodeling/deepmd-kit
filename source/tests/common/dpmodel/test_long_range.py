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
    COULOMB_CONSTANT,
    BaseInteractionKernel,
    LatentChargeLongRange,
    LESKernel,
    SOGKernel,
)
from deepmd.dpmodel.model.model import (
    get_model,
)
from deepmd.dpmodel.utils.exclude_mask import (
    PairExcludeMask,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    K_BLOCK,
    build_k_grid,
    build_pair_graph,
    build_topologies,
    enumerate_k_grid_bounds,
    pair_force_virial,
    unordered_pair_index,
)
from deepmd.utils.argcheck import (
    long_range_args,
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


def test_kernel_defaults_to_les() -> None:
    """A configuration without a kernel, or without its type, selects LES."""
    args = long_range_args()
    config = args.normalize_value({"type": "latent_charge"}, trim_pattern="_*")
    assert config["kernel"] == {"type": "les", "alpha": 1.0, "trainable": True}
    config = args.normalize_value(
        {"type": "latent_charge", "kernel": {"alpha": 2.0}}, trim_pattern="_*"
    )
    assert config["kernel"]["type"] == "les"
    model = get_model(
        {
            "type_map": ["A", "B"],
            "descriptor": {"type": "dpa4c", "rcut": 3.0, "channels": 8, "seed": 1},
            "fitting_net": {"type": "ener", "neuron": [8], "seed": 2},
            "long_range": {"type": "latent_charge", "charge_net": {"neuron": [4]}},
        }
    )
    kernel = model.atomic_model.long_range.kernel
    assert isinstance(kernel, LESKernel)
    np.testing.assert_allclose(np.exp(kernel.log_alpha), 1.0)


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
    """The k points satisfy k . a_d = 2 pi n_d, every frame fills a row of its own."""
    box = np.array([[8.0, 0.6, 0.0], [0.0, 7.5, 0.4], [0.0, 0.0, 9.0]])[None]
    grid = build_k_grid(box, 2.0, np.zeros((4, 3)), np.array([4]))
    # the points fill whole blocks, the last one completed by guard points
    assert grid.mask.shape == grid.vec.shape[:3]
    assert grid.vec.shape[2] == K_BLOCK
    flat_mask = grid.mask.reshape(-1)
    count = int(flat_mask.sum())
    assert flat_mask[:count].all() and not flat_mask[count:].any()
    np.testing.assert_array_equal(grid.slot, [[0, 1, 2, 3]])
    vec = grid.vec.reshape(-1, 3)[:count]
    n_integer = vec @ box[0].T / (2 * np.pi)
    np.testing.assert_allclose(n_integer, np.round(n_integer), atol=1e-10)
    k_sq = np.sum(vec**2, axis=-1)
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
    assert with_box[ALL_PAIRS].index.shape[1] == 4  # guards only
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
    # the reciprocal-space sum excludes no pair; a non-periodic batch still does
    pair_excl = PairExcludeMask(2, [(0, 1)])
    with pytest.raises(NotImplementedError, match="pair_exclude_types"):
        build_topologies(
            names,
            coord,
            n_node,
            box,
            atype=atype,
            pair_excl=pair_excl,
            topology_params=params,
        )
    excluded = build_topologies(
        names, coord, n_node, None, atype=atype, pair_excl=pair_excl
    )
    assert 0 < excluded[ALL_PAIRS].mask.sum() < without_box[ALL_PAIRS].mask.sum()
    # a cell without volume is no periodic system
    with pytest.raises(ValueError, match="no volume"):
        build_topologies(
            names, coord, n_node, np.zeros((1, 9)), atype=atype, topology_params=params
        )


def test_build_k_grid_pads_the_rows_of_the_frames() -> None:
    """Every frame keeps its own points and nodes; shorter rows end in masked guards."""
    box = np.stack([np.diag([8.0, 7.5, 9.0]), np.diag([5.0, 5.5, 6.0])])
    n_node = np.array([3, 5])
    grid = build_k_grid(box, 2.0, np.zeros((8, 3)), n_node)
    alone = [
        build_k_grid(box[f : f + 1], 2.0, np.zeros((n, 3)), np.array([n]))
        for f, n in enumerate(n_node)
    ]
    counts = [int(single.mask.sum()) for single in alone]
    blocks = max(-(-max(counts) // K_BLOCK), 2)
    assert grid.vec.shape == (2, blocks, K_BLOCK, 3)
    for f, single in enumerate(alone):
        mask = grid.mask[f].reshape(-1)
        vec = grid.vec[f].reshape(-1, 3)
        np.testing.assert_array_equal(mask, np.arange(blocks * K_BLOCK) < counts[f])
        np.testing.assert_allclose(
            vec[: counts[f]], single.vec[0].reshape(-1, 3)[: counts[f]]
        )
        assert np.all(vec[counts[f] :] == [1e-6, 0.0, 0.0])
    np.testing.assert_array_equal(grid.slot, [[0, 1, 2, -1, -1], [3, 4, 5, 6, 7]])


@pytest.mark.parametrize("kernel", [LESKernel(alpha=0.6), SOGKernel(n_gaussian=3)])
def test_kspace_energy_of_padded_frames_matches_frames_alone(
    kernel: BaseInteractionKernel,
) -> None:
    """Frames of unequal node counts, an empty one among them, sum as they do alone."""
    from deepmd.dpmodel.long_range.reciprocal import (
        kspace_energy,
    )

    box = np.stack(
        [np.diag([8.0, 7.5, 9.0]), np.diag([6.0, 6.5, 7.0]), np.diag([5.0, 5.5, 6.0])]
    )
    n_node = np.array([4, 0, 6])
    rng = np.random.default_rng(3)
    coord = rng.uniform(0.0, 5.0, (10, 3))
    charge = rng.normal(0.0, 0.5, (10, 2))
    energy = kspace_energy(charge, build_k_grid(box, 2.5, coord, n_node), kernel)
    start = 0
    for f, n in enumerate(n_node):
        if n == 0:
            continue
        nodes = slice(start, start + n)
        alone = build_k_grid(box[f : f + 1], 2.5, coord[nodes], np.array([n]))
        np.testing.assert_allclose(
            energy[nodes], kspace_energy(charge[nodes], alone, kernel), rtol=1e-12
        )
        start += n


# ---------------------------------------------------------------------------
# Fixed k-grid mode
# ---------------------------------------------------------------------------


def _module_topologies(
    module: LatentChargeLongRange,
    coord: np.ndarray,
    n_node: np.ndarray,
    box: np.ndarray | None,
    atype: np.ndarray,
) -> dict:
    """The topologies of a module, built with its parameters."""
    return build_topologies(
        module.required_topologies(),
        coord,
        n_node,
        box,
        atype=atype,
        topology_params=module.topology_params(),
    )


def _periodic_long_range(**kwargs: object) -> LatentChargeLongRange:
    """A periodic module, over the explicit wave numbers unless ``k_grid_mode`` says otherwise."""
    charge_net = LatentChargeFitting("latent_charge", 2, 4, 1, neuron=[4], seed=1)
    kwargs.setdefault("k_grid_mode", "ragged")
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
    bounds = np.max(k_grid_bounds_of_cells(box, k_max), axis=0)
    nk = enumerate_k_grid_bounds(bounds).shape[0]
    n_node = np.array([4, 5])
    fixed = build_fixed_k_grid(box, bounds, k_max, np.zeros((9, 3)), n_node)
    # a static layout: the blocks of the grid in every row
    assert fixed.vec.shape == (2, -(-nk // K_BLOCK), K_BLOCK, 3)
    ragged = build_k_grid(box, k_max, np.zeros((9, 3)), n_node)
    for frame in range(2):
        expected = {
            tuple(row) for row in np.round(ragged.vec[frame][ragged.mask[frame]], 8)
        }
        got = {tuple(row) for row in np.round(fixed.vec[frame][fixed.mask[frame]], 8)}
        assert got == expected
    # the smaller cell masks more points of the shared grid
    assert fixed.mask[0].sum() < fixed.mask[1].sum()
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
    bounds = np.max(k_grid_bounds_of_cells(box, k_max), axis=0)
    nk = enumerate_k_grid_bounds(bounds).shape[0]
    n_node = np.array([4, 5])
    with pytest.warns(UserWarning, match="truncated"):
        bigger = build_fixed_k_grid(box * 2.0, bounds, k_max, np.zeros((9, 3)), n_node)
    # the layout is untouched: static, every grid point present
    assert bigger.vec.shape == (2, -(-nk // K_BLOCK), K_BLOCK, 3)
    # every grid point is still a valid wave number of the larger cell; the
    # mask keeps the whole k_max sphere the grid reaches
    k2 = np.sum(bigger.vec**2, axis=-1)
    real = (np.arange(bigger.mask[0].size) < nk).reshape(bigger.mask.shape[1:])
    np.testing.assert_array_equal(bigger.mask, real & (k2 <= k_max * k_max))
    # the larger cell activates more of the shared grid than the original did
    original = build_fixed_k_grid(box, bounds, k_max, np.zeros((9, 3)), n_node)
    assert bigger.mask.sum() > original.mask.sum()
    # a larger wave-number cutoff (a drifting kernel) truncates the same way
    with pytest.warns(UserWarning, match="truncated"):
        build_fixed_k_grid(box, bounds, k_max * 2.0, np.zeros((9, 3)), n_node)


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
    np.testing.assert_array_equal(long_range.k_grid_bounds, [2, 2, 2])
    with pytest.raises(ValueError, match="non-negative"):
        _periodic_long_range(k_grid_mode="fixed", k_grid_bounds=[0, 0, 0])


def test_fixed_k_grid_stats_size_the_grid() -> None:
    """The data statistics size the grid by the largest sampled cell."""
    long_range = _periodic_long_range(k_grid_mode="fixed")
    # the grid is not sized before the statistics run
    coord = np.zeros((9, 3))
    atype = np.zeros(9, dtype=np.int64)
    with pytest.raises(ValueError, match="not sized yet"):
        _module_topologies(long_range, coord, np.array([4, 5]), _cells(), atype)
    long_range.compute_input_stats(
        [{"box": _cells()[:1].reshape(1, 9)}, {"box": _cells()}]
    )
    k_max = long_range.kernel.kspace_cutoff(long_range.kspace_tolerance)
    from deepmd.dpmodel.utils.neighbor_graph import (
        k_grid_bounds_of_cells,
    )

    expected = np.max(k_grid_bounds_of_cells(_cells(), k_max), axis=0)
    np.testing.assert_array_equal(long_range.k_grid_bounds, expected)
    topologies = _module_topologies(
        long_range, coord, np.array([4, 5]), _cells(), atype
    )
    nk = np.prod(2 * expected + 1) - 1
    assert topologies["k_grid"].vec.shape == (2, -(-nk // K_BLOCK), K_BLOCK, 3)
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
    np.testing.assert_array_equal(restored.k_grid_bounds, long_range.k_grid_bounds)
    # the ragged mode serializes no grid bounds
    plain = LatentChargeLongRange.deserialize(_periodic_long_range().serialize())
    assert plain.k_grid_mode == "ragged"
    assert plain.k_grid_bounds is None
    # a periodic module evaluates on the mesh unless told otherwise, and a
    # periodic record without a mode is one of the explicit wave numbers
    charge_net = LatentChargeFitting("latent_charge", 2, 4, 1, neuron=[4], seed=1)
    default = LatentChargeLongRange(charge_net, LESKernel(), None, periodic=True)
    assert default.k_grid_mode == "pme"
    assert default.required_topologies() == ("all_pairs", "pme_mesh")
    record = default.serialize()
    assert record["k_grid_mode"] == "pme"
    record.pop("k_grid_mode")
    assert LatentChargeLongRange.deserialize(record).k_grid_mode == "ragged"


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
    initial = _module_topologies(decoupled, coord, n_node, box, atype)["k_grid"]
    reference = _module_topologies(coupled, coord, n_node, box, atype)["k_grid"]
    np.testing.assert_array_equal(initial.vec, reference.vec)
    # a larger alpha moves the cutoff out, growing the coupled grid
    for module in (decoupled, coupled):
        module.kernel.log_alpha = np.log(1.5)
    drifted = _module_topologies(decoupled, coord, n_node, box, atype)["k_grid"]
    moved = _module_topologies(coupled, coord, n_node, box, atype)["k_grid"]
    np.testing.assert_array_equal(drifted.vec, initial.vec)
    np.testing.assert_array_equal(drifted.mask, initial.mask)
    assert np.sum(moved.mask) > np.sum(initial.mask)


def test_decoupled_fixed_grid_keeps_its_mask() -> None:
    """The mask of a decoupled fixed grid ignores the kernel drift."""
    coord = np.zeros((5, 3))
    atype = np.zeros(5, dtype=np.int64)
    n_node = np.array([5])
    box = np.eye(3)[None] * 8.0

    long_range = _periodic_long_range(
        k_grid_mode="fixed", k_grid_bounds=[5, 5, 5], decouple_k_grid=True
    )
    before = _module_topologies(long_range, coord, n_node, box, atype)["k_grid"]
    # a smaller alpha moves the cutoff in; a coupled mask would tighten
    long_range.kernel.log_alpha = np.log(0.05)
    after = _module_topologies(long_range, coord, n_node, box, atype)["k_grid"]
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


# ---------------------------------------------------------------------------
# PME mesh mode
# ---------------------------------------------------------------------------


def _pme_long_range(**kwargs: object) -> LatentChargeLongRange:
    kwargs.setdefault("k_grid_mode", "pme")
    return _periodic_long_range(**kwargs)


def test_pme_mesh_grid_build_and_flatten() -> None:
    """The mesh grid packs the cells and the mesh, flags the frames and flattens."""
    from deepmd.dpmodel.utils.neighbor_graph import (
        PME_MESH,
        build_topologies,
        flatten_topologies,
        mesh_dims_of_cells,
        unflatten_topologies,
    )

    coord = np.random.default_rng(1).normal(size=(4, 3))
    atype = np.array([0, 1, 1, 0])
    box = _cells()
    params = {PME_MESH: {"spacing": 1.0}}
    tops = build_topologies(
        ("all_pairs", PME_MESH),
        coord,
        np.array([2, 2]),
        box,
        atype=atype,
        topology_params=params,
    )
    mesh = tops[PME_MESH]
    np.testing.assert_array_equal(mesh.box, box)
    assert mesh.mask.shape == (2,) and mesh.mask.all()
    dims = tuple(m.shape[0] for m in (mesh.mode_1, mesh.mode_2, mesh.mode_3))
    assert dims == mesh_dims_of_cells(box, 1.0)
    # the modes of an axis are its integer frequencies in FFT order
    n1 = dims[0]
    expected = np.arange(n1, dtype=float)
    np.testing.assert_array_equal(
        mesh.mode_1, np.where(expected > n1 // 2, expected - n1, expected)
    )
    assert tops["all_pairs"].index.shape[1] == 4  # guards only
    flat = flatten_topologies(tops)
    assert len(flat) == 3 + 8
    back = unflatten_topologies(("all_pairs", PME_MESH), flat)
    np.testing.assert_array_equal(back[PME_MESH].box, box)
    np.testing.assert_array_equal(back[PME_MESH].mask, mesh.mask)
    np.testing.assert_array_equal(back[PME_MESH].mode_3, mesh.mode_3)
    # a non-periodic batch: real pairs, masked-out mesh
    tops_np = build_topologies(
        ("all_pairs", PME_MESH),
        coord,
        np.array([2, 2]),
        None,
        atype=atype,
        topology_params=params,
    )
    assert tops_np["all_pairs"].mask.any()
    assert not tops_np[PME_MESH].mask.any()
    # missing build parameters are an error
    with pytest.raises(ValueError, match="spacing"):
        build_topologies((PME_MESH,), coord, np.array([4]), box, atype=atype)


def test_pme_mesh_dims_resolve_every_cell() -> None:
    """The mesh is the smallest even 7-smooth one resolving every cell."""
    from deepmd.dpmodel.utils.neighbor_graph import (
        mesh_dims_of_cells,
    )

    for spacing in (0.37, 0.5, 1.1):
        dims = mesh_dims_of_cells(_cells(), spacing)
        longest = np.max(np.linalg.norm(_cells(), axis=-1), axis=0)
        for n, a in zip(dims, longest, strict=True):
            assert n % 2 == 0 and n >= a / spacing
            m = n
            for p in (2, 3, 5, 7):
                while m % p == 0:
                    m //= p
            assert m == 1
            # no smaller even 7-smooth size reaches the spacing
            for smaller in range(n - 2, 1, -2):
                m = smaller
                for p in (2, 3, 5, 7):
                    while m % p == 0:
                        m //= p
                if m == 1:
                    assert smaller < a / spacing
                    break
    # the larger cell alone sizes the mesh of the batch
    assert mesh_dims_of_cells(_cells(), 0.5) == mesh_dims_of_cells(_cells()[1:], 0.5)


def _pme_energy(module, charge, coord, box) -> float:
    from deepmd.dpmodel.long_range.reciprocal import (
        mesh_energy,
    )
    from deepmd.dpmodel.utils.neighbor_graph import (
        build_mesh_grid,
    )

    mesh = build_mesh_grid(box[None], module.mesh_spacing(), coord)
    frame_id = np.zeros(coord.shape[0], dtype=np.int64)
    energy = mesh_energy(charge, mesh, frame_id, module.kernel, module.pme_order)
    return float(np.sum(energy))


def test_pme_energy_converges_to_direct_reference() -> None:
    """The PME energy approaches the direct k-space sum as the mesh refines."""
    coord = 3.0 + np.random.default_rng(7).uniform(0.0, 1.4, (6, 3))
    box = np.array([[9.0, 0.4, 0.0], [0.0, 8.5, 0.3], [0.0, 0.0, 9.5]])
    charge = np.array([[1.0], [-0.5], [1.0], [-0.5], [-0.5], [1.0]])
    charge = charge - charge.mean(axis=0, keepdims=True)

    kernel = LESKernel(alpha=0.3)
    # converged direct k-space reference (tol 1e-10)
    k_max = kernel.kspace_cutoff(1e-10)
    rec = 2.0 * np.pi * np.linalg.inv(box)
    n_d = np.ceil(k_max * np.linalg.norm(box, axis=1) / (2 * np.pi)).astype(int)
    grids = np.meshgrid(*[np.arange(-n, n + 1) for n in n_d], indexing="ij")
    n_all = np.stack(grids, -1).reshape(-1, 3)
    k = n_all @ rec.T
    k2 = (k * k).sum(-1)
    keep = (k2 > 0) & (k2 <= k_max * k_max)
    k, k2 = k[keep], k2[keep]
    phase = k @ coord.T
    rho = (np.cos(phase) + 1j * np.sin(phase)) @ charge
    s_sq = (np.abs(rho) ** 2).sum(-1)
    k_hat = kernel.kspace(k2)
    volume = abs(np.linalg.det(box))
    k0 = 2.0 * 0.3 / np.sqrt(np.pi)
    e_ref = COULOMB_CONSTANT * (
        (k_hat * s_sq).sum() / (2 * volume) - 0.5 * k0 * (charge * charge).sum()
    )
    errors = []
    for spacing in (2.0, None):
        module = LatentChargeLongRange(
            LatentChargeFitting("latent_charge", 2, 4, 1, neuron=[4], seed=1),
            kernel,
            None,
            periodic=True,
            k_grid_mode="pme",
            pme_spacing=spacing,
        )
        errors.append(abs(_pme_energy(module, charge, coord, box) - e_ref))
    assert errors[1] < errors[0]  # refinement improves the energy
    # the default mesh reaches the accuracy it is chosen for
    assert errors[1] < 1e-6 * abs(e_ref)


def test_pme_mesh_follows_the_cells_of_the_batch() -> None:
    """The mesh of a batch resolves its cells at the spacing, without statistics."""
    from deepmd.dpmodel.utils.neighbor_graph import (
        mesh_dims_of_cells,
    )

    long_range = _pme_long_range()
    coord = np.zeros((9, 3))
    atype = np.zeros(9, dtype=np.int64)
    small = _module_topologies(long_range, coord, np.array([4, 5]), _cells()[:1], atype)
    large = _module_topologies(
        long_range, coord, np.array([4, 5]), 2.0 * _cells(), atype
    )
    spacing = long_range.mesh_spacing()
    assert spacing == pytest.approx(
        0.75 * np.pi / long_range.kernel.kspace_cutoff(long_range.kspace_tolerance)
    )
    for topologies, box in ((small, _cells()[:1]), (large, 2.0 * _cells())):
        mesh = topologies["pme_mesh"]
        dims = tuple(m.shape[0] for m in (mesh.mode_1, mesh.mode_2, mesh.mode_3))
        assert dims == mesh_dims_of_cells(box, spacing)
        assert mesh.mask.all()


def test_pme_mode_configuration() -> None:
    """The pme mode requires periodicity, and so do the mesh parameters."""
    with pytest.raises(ValueError, match="periodic"):
        LatentChargeLongRange(
            LatentChargeFitting("latent_charge", 2, 4, 1, neuron=[4], seed=1),
            LESKernel(),
            None,
            k_grid_mode="pme",
        )
    for kwargs in ({"pme_spacing": 1.0}, {"pme_order": 6}):
        with pytest.raises(ValueError, match="periodic"):
            LatentChargeLongRange(
                LatentChargeFitting("latent_charge", 2, 4, 1, neuron=[4], seed=1),
                LESKernel(),
                None,
                **kwargs,
            )
    # every periodic mode takes them: a frozen model evaluates on the mesh
    assert _periodic_long_range(pme_spacing=1.0).mesh_spacing() == 1.0
    assert _periodic_long_range().pme_order == 8
    with pytest.raises(ValueError, match="pme_order"):
        _pme_long_range(pme_order=1)
    with pytest.raises(ValueError, match="pme_spacing"):
        _pme_long_range(pme_spacing=-1.0)
    with pytest.raises(ValueError, match="k_grid_mode"):
        _periodic_long_range(k_grid_mode="dense")


def test_pme_round_trip() -> None:
    """The record keeps the pme mode, the order and the spacing."""
    long_range = _pme_long_range(pme_spacing=1.0, pme_order=5)
    record = long_range.serialize()
    assert record["k_grid_mode"] == "pme"
    assert record["pme_order"] == 5
    assert record["pme_spacing"] == 1.0
    restored = LatentChargeLongRange.deserialize(record)
    assert restored.k_grid_mode == "pme"
    assert restored.pme_order == 5
    assert restored.pme_spacing == 1.0
    # the default order is recorded too, so that it survives a change of default
    assert _periodic_long_range().serialize()["pme_order"] == 8
