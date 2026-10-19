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
    build_pair_graph,
    build_topologies,
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
