# SPDX-License-Identifier: LGPL-3.0-or-later
"""Bounded neighbor statistics must retain all frames and both extrema."""

from pathlib import (
    Path,
)
from types import (
    SimpleNamespace,
)
from typing import (
    Any,
)

import numpy as np
import pytest

from deepmd.dpmodel.utils.neighbor_stat import (
    NeighborStat,
    NeighborStatOP,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.utils.neighbor_stat import (
    iter_neighbor_stat_batches,
)


def make_data(path: Path, pbc: bool = True) -> DeepmdDataSystem:
    """Two sets, mixed atom types, varying tilted cells, and a short tail."""
    path.mkdir()
    (path / "type.raw").write_text("0 1 0 1\n")
    (path / "type_map.raw").write_text("A B\n")
    if not pbc:
        (path / "nopbc").touch()
    rng = np.random.default_rng(20261117)
    for index, nframes in enumerate((11, 6)):
        directory = path / f"set.{index:03d}"
        directory.mkdir()
        cells = np.array(
            [
                [[2.5 + f * 0.1, 0, 0], [0.3, 3.0, 0], [0.1, 0.2, 3.5]]
                for f in range(nframes)
            ]
        )
        coords = rng.random((nframes, 4, 3)) @ cells
        # Put the global shortest pair in the final partial batch.
        if index == 1:
            coords[-1, 1] = coords[-1, 0] + [0.03125, 0, 0]
        types = np.tile([0, 1, 0, 1], (nframes, 1))
        types[1::2] = [1, 0, 1, 0]
        np.save(directory / "coord.npy", coords.reshape(nframes, -1))
        np.save(directory / "box.npy", cells.reshape(nframes, 9))
        np.save(directory / "real_atom_types.npy", types)
    return DeepmdDataSystem(
        systems=[str(path)],
        batch_size=1,
        test_size=1,
        rcut=3.0,
        type_map=["A", "B"],
        trn_all_set=True,
    )


def reference_stat(data: DeepmdDataSystem, rcut: float, mixed: bool) -> tuple:
    """The original full-set operator and reductions, without batching."""
    op = NeighborStatOP(2, rcut, mixed)
    minimum = np.inf
    maximum = np.zeros(1 if mixed else 2, dtype=int)
    for system in data.data_systems:
        for path in system.dirs:
            arrays = system._load_set(path)
            distance, counts = op(
                arrays["coord"],
                arrays["type"],
                arrays["box"] if system.pbc else None,
            )
            minimum = min(minimum, np.min(distance))
            maximum = np.maximum(maximum, np.max(counts, axis=0))
    return np.sqrt(minimum), maximum


@pytest.mark.parametrize("pbc", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("rcut", [1e-6, 3.0])
@pytest.mark.parametrize("batch_size", [1, 3, 8, 100])
def test_numpy_batched_matches_full_sets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pbc: bool,
    mixed: bool,
    rcut: float,
    batch_size: int,
) -> None:
    data = make_data(tmp_path / "system", pbc)
    expected = reference_stat(data, rcut, mixed)
    monkeypatch.setenv("DP_NEIGHBOR_STAT_BATCH_SIZE", str(batch_size))
    stat = NeighborStat(2, rcut, mixed)
    original = stat.op
    calls = []

    def record(coord: np.ndarray, atype: np.ndarray, cell: Any, nbuff: Any) -> tuple:
        calls.append(coord.shape[0])
        return original(coord, atype, cell, nbuff)

    stat.op = record
    actual = stat.get_stat(data)
    assert sum(calls) == 17
    assert max(calls) <= batch_size
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(actual[1], expected[1])


def test_default_cap_and_aligned_frame_slices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DP_NEIGHBOR_STAT_BATCH_SIZE", raising=False)
    data = make_data(tmp_path / "system")
    chunks = list(iter_neighbor_stat_batches(data, 3.0))
    assert [c.shape[0] for c, _, _, _, _ in chunks] == [8, 3, 6]
    for system in data.data_systems:
        for path in system.dirs:
            original = system._load_set(path)
            selected = [chunk for chunk in chunks if chunk[3] == str(path)]
            for index, key in enumerate(("coord", "type", "box")):
                np.testing.assert_array_equal(
                    np.concatenate([chunk[index] for chunk in selected]), original[key]
                )


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "bad", ""])
def test_invalid_cap_fails_before_loading(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("DP_NEIGHBOR_STAT_BATCH_SIZE", value)
    # Validation must happen before even accessing a data system.
    with pytest.raises(ValueError, match="DP_NEIGHBOR_STAT_BATCH_SIZE"):
        list(iter_neighbor_stat_batches(SimpleNamespace(), 3.0))


# These cells have bounds [1, 1, 1] and [3, 1, 1], respectively, at rcut=1/64.
SKEW_CELL = np.array([[1, 0, 0], [2, 0.125, 0], [0, 0, 8]], dtype=np.float64)
WIDE_IMAGE_CELL = np.array([[1, 0, 0], [129, 1, 0], [0, 0, 8]], dtype=np.float64)


def make_periodic_counterexample(
    path: Path, case: str, reverse: bool = False
) -> tuple[DeepmdDataSystem, float]:
    """Nine frames whose full-set image range differs from their first eight.

    In the skew case, image (-2, 1, 0) of SKEW_CELL has length 1/8,
    outside rcut. In the unwrapped case, image (-2, 0, 0) brings the
    second type-0 atom to distance 1/8, inside rcut. The last frame
    supplies the image range needed to find these pairs in the first eight.
    """
    if case == "skew":
        rcut = 1 / 64
        coord = np.zeros((9, 1, 3))
        cell = np.stack([SKEW_CELL] * 8 + [WIDE_IMAGE_CELL])
        atype = np.zeros((9, 1), dtype=np.int32)
    else:
        assert case == "unwrapped"
        rcut = 0.25
        coord = np.tile(np.array([[0.0, 0.0, 0.0], [8.125, 0.0, 0.0]]), (9, 1, 1))
        coord[-1] = [[0, 0, 0], [0, 2, 0]]
        cell = np.tile(np.eye(3) * 4, (9, 1, 1))
        cell[-1] = np.diag([0.125, 4, 4])
        atype = np.zeros((9, 2), dtype=np.int32)
        atype[-1] = 1
    if reverse:
        coord, cell, atype = coord[::-1], cell[::-1], atype[::-1]
    path.mkdir()
    (path / "type.raw").write_text(" ".join(["0"] * atype.shape[1]) + "\n")
    (path / "type_map.raw").write_text("A B\n")
    subset = path / "set.000"
    subset.mkdir()
    np.save(subset / "coord.npy", coord.reshape(9, -1))
    np.save(subset / "box.npy", cell.reshape(9, 9))
    np.save(subset / "real_atom_types.npy", atype)
    return DeepmdDataSystem(
        systems=[str(path)],
        batch_size=1,
        test_size=1,
        rcut=rcut,
        type_map=["A", "B"],
        trn_all_set=True,
    ), rcut


def check_periodic_batches(
    stat_type: type,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    reverse: bool,
    mixed: bool,
    cap: int | None,
) -> None:
    data, rcut = make_periodic_counterexample(tmp_path / "system", case, reverse)
    if cap is None:
        monkeypatch.delenv("DP_NEIGHBOR_STAT_BATCH_SIZE", raising=False)
    else:
        monkeypatch.setenv("DP_NEIGHBOR_STAT_BATCH_SIZE", str(cap))
    monkeypatch.setenv("DP_INFER_BATCH_SIZE", "1000000")
    expected = reference_stat(data, rcut, mixed)
    # Independently known values also protect against a changed reference kernel.
    assert expected[0] == 0.125
    counts = (
        ([0] if mixed else [0, 0]) if case == "skew" else ([2] if mixed else [1, 2])
    )
    np.testing.assert_array_equal(expected[1], counts)
    actual = stat_type(2, rcut, mixed).get_stat(data)
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])


@pytest.mark.parametrize("case", ["skew", "unwrapped"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("cap", [1, 8, 100, None])
def test_periodic_batches_keep_full_set_statistics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    reverse: bool,
    mixed: bool,
    cap: int | None,
) -> None:
    check_periodic_batches(
        NeighborStat, tmp_path, monkeypatch, case, reverse, mixed, cap
    )


def test_periodic_bounds_are_shared_only_within_each_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DP_NEIGHBOR_STAT_BATCH_SIZE", "8")
    arrays = {
        "wide": {
            "coord": np.zeros((9, 3)),
            "type": np.zeros((9, 1), dtype=np.int32),
            "box": np.stack([SKEW_CELL] * 8 + [WIDE_IMAGE_CELL]).reshape(9, 9),
        },
        "narrow": {
            "coord": np.zeros((9, 3)),
            "type": np.zeros((9, 1), dtype=np.int32),
            "box": np.tile(SKEW_CELL.reshape(1, 9), (9, 1)),
        },
    }
    data = SimpleNamespace(
        data_systems=[
            SimpleNamespace(pbc=True, dirs=list(arrays), _load_set=arrays.__getitem__)
        ]
    )
    chunks = list(iter_neighbor_stat_batches(data, 1 / 64))
    np.testing.assert_array_equal(
        [chunk[4] for chunk in chunks], [[3, 1, 1], [3, 1, 1], [1, 1, 1], [1, 1, 1]]
    )
    assert chunks[0][4] is chunks[1][4]
    assert chunks[2][4] is chunks[3][4]
    # A later set must not inherit the previous set's larger image range.
    stats = list(NeighborStat(2, 1 / 64).iterator(data))
    assert [minimum for _, minimum, _ in stats] == [1 / 64, 1, 1, 1]
