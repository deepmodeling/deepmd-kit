# SPDX-License-Identifier: LGPL-3.0-or-later
"""Independent integer oracle for bounded neighbor statistics.

Coordinates, cells and cutoffs have a common power-of-two denominator. Image
bounds and squared distances below are obtained with Python integer arithmetic,
without using the production geometry helpers or neighbor-statistics kernel.
"""

import itertools
import math
from functools import (
    cache,
)
from types import (
    SimpleNamespace,
)

import numpy as np
import pytest

from deepmd.dpmodel.utils.neighbor_stat import (
    NeighborStat,
)

SCALE = 256
NTYPES = 3


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def exact_image_bounds(cells, cutoff):
    """ceil(rcut / face_distance), using integer inequalities and no sqrt."""
    bounds = [0, 0, 0]
    for matrix in cells:
        rows = [[int(v) for v in row] for row in matrix]
        cross = _cross(rows[1], rows[2])
        determinant = sum(a * b for a, b in zip(rows[0], cross, strict=True))
        assert determinant > 0
        denominator = determinant**2
        for axis in range(3):
            face = _cross(rows[(axis + 1) % 3], rows[(axis + 2) % 3])
            numerator = cutoff**2 * sum(v * v for v in face)
            bound = math.isqrt(numerator // denominator)
            if bound * bound * denominator < numerator:
                bound += 1
            bounds[axis] = max(bounds[axis], bound)
    return bounds


def integer_reference(records, cutoff):
    """Explicitly enumerate real centers, real neighbors and periodic images."""
    minimum = 100 * SCALE**2  # Existing get_stat initialization.
    maximum = [0] * NTYPES
    mixed_maximum = 0
    for periodic, arrays in records:
        cells = arrays["cell"]
        bounds = exact_image_bounds(cells, cutoff) if periodic else [0, 0, 0]
        images = list(itertools.product(*(range(-n, n + 1) for n in bounds)))
        for frame, (coords, types) in enumerate(
            zip(arrays["coord"], arrays["type"], strict=True)
        ):
            coords = [[int(v) for v in row] for row in coords]
            cell = [[int(v) for v in row] for row in cells[frame]]
            shifts = [
                tuple(sum(image[k] * cell[k][d] for k in range(3)) for d in range(3))
                for image in images
            ]
            for center, center_type in enumerate(types):
                if center_type < 0:
                    continue
                counts = [0] * NTYPES
                for neighbor, neighbor_type in enumerate(types):
                    if neighbor_type < 0:
                        continue
                    for image, shift in zip(images, shifts, strict=True):
                        if center == neighbor and image == (0, 0, 0):
                            continue
                        distance = sum(
                            (coords[neighbor][d] + shift[d] - coords[center][d]) ** 2
                            for d in range(3)
                        )
                        minimum = min(minimum, distance)
                        if distance < cutoff**2:
                            counts[int(neighbor_type)] += 1
                maximum = [max(a, b) for a, b in zip(maximum, counts, strict=True)]
                mixed_maximum = max(mixed_maximum, sum(counts))
    assert minimum > 0, "Generated real atoms overlap"
    distance = math.sqrt(minimum) / SCALE
    return (distance, np.asarray(maximum)), (distance, np.asarray([mixed_maximum]))


def as_data(records):
    """Use several systems and sets, including changes in periodicity."""
    systems = []
    for periodic, arrays in records:
        nf = arrays["coord"].shape[0]
        loaded = {
            "coord": arrays["coord"].reshape(nf, -1).astype(np.float64) / SCALE,
            "type": arrays["type"].copy(),
            "box": arrays["cell"].reshape(nf, 9).astype(np.float64) / SCALE,
        }
        systems.append(
            SimpleNamespace(
                pbc=periodic, dirs=["set.000"], _load_set=lambda _, value=loaded: value
            )
        )
    return SimpleNamespace(data_systems=systems)


@cache
def integer_case(seed):
    rng = np.random.default_rng(91763 + seed)
    nf = (1, 7, 8, 9, 17, 23)[seed % 6]
    na = (2, 3, 5)[(seed // 6) % 3]
    periodic = seed % 4 != 0
    cutoff = (64, 128, 256, 384)[seed % 4]
    cells, coords = [], []
    for frame in range(nf):
        matrix = np.diag(rng.integers(32, 49, size=3))
        matrix[1, 0] = rng.integers(-24, 25)
        matrix[2, :2] = rng.integers(-16, 17, size=2)
        if frame == nf - 1 and periodic:
            matrix[0, 0] = 13
        points = rng.integers(0, 16, size=(na, 3))
        while np.unique(points, axis=0).shape[0] != na:
            points = rng.integers(0, 16, size=(na, 3))
        cells.append(matrix * 16)
        coords.append(points @ matrix)
    arrays = {
        "cell": np.asarray(cells),
        "coord": np.asarray(coords),
        "type": rng.integers(0, NTYPES, size=(nf, na), dtype=np.int32),
    }
    if seed % 5 == 1:
        arrays["coord"][:, 1] += 2 * arrays["cell"][:, 0]
    if na >= 3:
        arrays["type"][:, -1] = -1
        arrays["coord"][:, -1] = arrays["coord"][:, 0]
    parts = [slice(None)] if nf <= 8 or seed % 2 else [slice(0, 5), slice(5, None)]
    records = [
        (periodic, {name: values[part].copy() for name, values in arrays.items()})
        for part in parts
    ]
    if seed % 6 == 4:
        # A nonperiodic system follows a periodic one (or vice versa).
        records.append(
            (not periodic, {name: values[:2].copy() for name, values in arrays.items()})
        )
    return records, cutoff, integer_reference(records, cutoff)


def check_exact_case(stat_type, monkeypatch, seed, mixed, cap):
    records, cutoff, reference = integer_case(seed)
    data = as_data(records)
    if cap is None:
        monkeypatch.delenv("DP_NEIGHBOR_STAT_BATCH_SIZE", raising=False)
    else:
        monkeypatch.setenv("DP_NEIGHBOR_STAT_BATCH_SIZE", str(cap))
    monkeypatch.setenv("DP_INFER_BATCH_SIZE", "1000000")
    before = [
        (
            s._load_set("set.000")["coord"].copy(),
            s._load_set("set.000")["type"].copy(),
            s._load_set("set.000")["box"].copy(),
        )
        for s in data.data_systems
    ]
    actual = stat_type(NTYPES, cutoff / SCALE, mixed).get_stat(data)
    expected = reference[int(mixed)]
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    for system, original in zip(data.data_systems, before, strict=True):
        for name, array in zip(("coord", "type", "box"), original, strict=True):
            np.testing.assert_array_equal(system._load_set("set.000")[name], array)


@pytest.mark.parametrize("seed", range(24))
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("cap", [1, 3, 8, 100, None])
def test_matches_independent_integer_oracle(monkeypatch, seed, mixed, cap):
    check_exact_case(NeighborStat, monkeypatch, seed, mixed, cap)


def check_permutations(stat_type, monkeypatch, seed, mixed):
    records, cutoff, reference = integer_case(seed)
    rng = np.random.default_rng(seed)
    transformed = []
    for periodic, arrays in records:
        frames = rng.permutation(len(arrays["coord"]))
        atoms = rng.permutation(arrays["coord"].shape[1])
        changed = {
            "coord": arrays["coord"][frames][:, atoms]
            + [SCALE, -2 * SCALE, SCALE // 2],
            "type": arrays["type"][frames][:, atoms],
            "cell": arrays["cell"][frames],
        }
        transformed.append((periodic, changed))
    monkeypatch.delenv("DP_NEIGHBOR_STAT_BATCH_SIZE", raising=False)
    monkeypatch.delenv("DP_INFER_BATCH_SIZE", raising=False)
    actual = stat_type(NTYPES, cutoff / SCALE, mixed).get_stat(as_data(transformed))
    expected = reference[int(mixed)]
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])


@pytest.mark.parametrize("seed", [1, 5, 10, 17, 22])
@pytest.mark.parametrize("mixed", [False, True])
def test_frame_atom_order_and_translation(monkeypatch, seed, mixed):
    check_permutations(NeighborStat, monkeypatch, seed, mixed)


def check_cutoff_boundary(stat_type, monkeypatch, periodic, mixed, separation, cap):
    coords = np.tile([[0, 0, 0], [separation, 0, 0], [0, 0, 0]], (9, 1, 1))
    records = [
        (
            periodic,
            {
                "coord": coords,
                "cell": np.tile(np.eye(3, dtype=int) * 4 * SCALE, (9, 1, 1)),
                "type": np.tile([0, 1, -1], (9, 1)),
            },
        )
    ]
    monkeypatch.setenv("DP_NEIGHBOR_STAT_BATCH_SIZE", str(cap))
    actual = stat_type(NTYPES, 1.0, mixed).get_stat(as_data(records))
    assert actual[0] == separation / SCALE
    if mixed:
        expected = [int(separation < SCALE)]
    else:
        expected = [int(separation < SCALE), int(separation < SCALE), 0]
    np.testing.assert_array_equal(actual[1], expected)


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("separation", [SCALE - 1, SCALE, SCALE + 1])
@pytest.mark.parametrize("cap", [1, 8, 100])
def test_cutoff_boundary_and_virtual_overlap(
    monkeypatch, periodic, mixed, separation, cap
):
    check_cutoff_boundary(NeighborStat, monkeypatch, periodic, mixed, separation, cap)


def check_overlap_in_tail(stat_type, monkeypatch, periodic, cap):
    coord = np.tile([[0, 0, 0], [SCALE, 0, 0]], (9, 1, 1))
    coord[-1, 1] = coord[-1, 0]
    records = [
        (
            periodic,
            {
                "coord": coord,
                "cell": np.tile(np.eye(3, dtype=int) * 4 * SCALE, (9, 1, 1)),
                "type": np.zeros((9, 2), dtype=np.int32),
            },
        )
    ]
    monkeypatch.setenv("DP_NEIGHBOR_STAT_BATCH_SIZE", str(cap))
    with pytest.raises(RuntimeError, match="overlapping"):
        stat_type(NTYPES, 1.5).get_stat(as_data(records))


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("cap", [1, 8, 100])
def test_real_overlap_in_final_partial_batch(monkeypatch, periodic, cap):
    check_overlap_in_tail(NeighborStat, monkeypatch, periodic, cap)
