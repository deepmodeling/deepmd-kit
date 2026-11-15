# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the batched neighbor statistics of the pt_expt backend."""

from pathlib import (
    Path,
)
from typing import (
    Any,
)

import numpy as np

from deepmd.pt_expt.utils.neighbor_stat import (
    NeighborStat,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)

NATOMS = 27
NFRAMES = 8
SET_SIZE = 4

# A 3x3x3 grid spaced 1.0 apart in a 3.0 box: the shortest distance is
# exactly 1.0 and, just past it, every atom sees its six axis-aligned
# images. Used for the analytical check.
UNIFORM_SPACING = 1.0
UNIFORM_RCUT = 1.001

# For the batching comparison the cutoff also admits the second shell
# (sqrt(2) * spacing), which is what makes the neighbor count depend on
# the spacing of each individual frame.
MIXED_RCUT = 1.5
SPARSE_SPACING = 1.2
DENSE_SPACING = 1.0
CLOSE_PAIR_DISTANCE = 0.5


def _grid(spacing: float) -> np.ndarray:
    """Return a 3x3x3 grid of ``NATOMS`` positions with the given spacing."""
    axis = np.arange(3) * spacing
    return np.vstack([g.ravel() for g in np.meshgrid(*([axis] * 3), indexing="ij")]).T


def _write_system(path: Path, coords: np.ndarray, boxes: np.ndarray) -> None:
    """Write frames as a multi-set system in the DeePMD npy layout."""
    nframes = coords.shape[0]
    path.mkdir(parents=True, exist_ok=True)
    (path / "type.raw").write_text("\n".join(["0"] * NATOMS) + "\n")
    (path / "type_map.raw").write_text("TYPE\n")
    for index, start in enumerate(range(0, nframes, SET_SIZE)):
        subset = path / f"set.{index:03d}"
        subset.mkdir()
        frames = slice(start, start + SET_SIZE)
        nsub = coords[frames].shape[0]
        np.save(subset / "coord.npy", coords[frames].reshape(nsub, NATOMS * 3))
        np.save(subset / "box.npy", boxes[frames])
        np.save(subset / "energy.npy", np.zeros(nsub))
        np.save(subset / "force.npy", np.zeros((nsub, NATOMS * 3)))


def _box(spacing: float) -> np.ndarray:
    return (np.eye(3) * spacing * 3).ravel()


def _uniform_frames() -> tuple[np.ndarray, np.ndarray]:
    """Identical frames, so the statistics are known analytically."""
    coords = np.repeat(_grid(UNIFORM_SPACING)[np.newaxis], NFRAMES, axis=0)
    boxes = np.tile(_box(UNIFORM_SPACING), (NFRAMES, 1))
    return coords, boxes


def _mixed_frames() -> tuple[np.ndarray, np.ndarray]:
    """Frames whose two extrema are contributed by different sets.

    The first set holds a frame with one pair pulled close, which owns the
    global minimum distance but barely changes any neighbor count. The last
    set holds a denser frame, whose second shell falls inside the cutoff and
    owns the global maximum neighbor count. A reduction that keeps only one
    batch therefore cannot report both.
    """
    sparse = _grid(SPARSE_SPACING)
    close_pair = sparse.copy()
    # index 9 is (spacing, 0, 0); move atom 0 to sit CLOSE_PAIR_DISTANCE from it
    close_pair[0] = sparse[9] - np.array([CLOSE_PAIR_DISTANCE, 0.0, 0.0])
    dense = _grid(DENSE_SPACING)

    coords = np.repeat(sparse[np.newaxis], NFRAMES, axis=0)
    boxes = np.tile(_box(SPARSE_SPACING), (NFRAMES, 1))
    coords[0] = close_pair
    coords[-1] = dense
    boxes[-1] = _box(DENSE_SPACING)
    return coords, boxes


def _stat_with_batch_size(
    monkeypatch: Any, system: str, rcut: float, batch_atoms: int
) -> tuple[tuple[float, np.ndarray], list[int]]:
    """Run the statistics at a pinned batch size.

    ``DP_INFER_BATCH_SIZE`` is read when :class:`AutoBatchSize` is built, so
    it has to be set before the ``NeighborStat`` is constructed.

    Returns
    -------
    tuple
        The ``(min_nbor_dist, max_nbor_size)`` that ``get_stat`` produced.
    list
        The number of frames handed to each ``_execute`` call, one entry per
        call, so the caller can tell how the work was split.
    """
    frames_per_call: list[int] = []
    # A context keeps the patched ``_execute`` from leaking into a later run,
    # which would otherwise chain the spies and mix their records.
    with monkeypatch.context() as patch:
        patch.setenv("DP_INFER_BATCH_SIZE", str(batch_atoms))
        original = NeighborStat._execute

        def spy(
            self: NeighborStat,
            coord: np.ndarray,
            atype: np.ndarray,
            cell: np.ndarray | None,
        ) -> tuple[np.ndarray, np.ndarray]:
            frames_per_call.append(coord.shape[0])
            return original(self, coord, atype, cell)

        patch.setattr(NeighborStat, "_execute", spy)

        data = DeepmdDataSystem(
            systems=[system],
            batch_size=1,
            test_size=1,
            rcut=rcut,
            type_map=["TYPE"],
        )
        data.get_batch()
        stat = NeighborStat(data.get_ntypes(), rcut, mixed_type=False).get_stat(data)
    return stat, frames_per_call


def _single_frame_stat(
    monkeypatch: Any, path: Path, coord: np.ndarray, box: np.ndarray
) -> tuple[float, np.ndarray]:
    """Statistics of one frame on its own, as a reference extremum."""
    _write_system(path, coord[np.newaxis], box[np.newaxis])
    stat, _ = _stat_with_batch_size(monkeypatch, str(path), MIXED_RCUT, NATOMS)
    return stat


def test_batching_splits_the_work_and_keeps_the_result(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """Batching must change how the work is split, not what it produces.

    Sending a whole set at once allocates an ``[nframes, nloc, nall, 3]``
    tensor, which is why the statistics are batched. Both the split and the
    reduction across batches are asserted: the frames are built so that the
    two extrema come from different sets, so keeping only one batch would
    get one of them wrong.
    """
    coords, boxes = _mixed_frames()
    system = tmp_path / "system"
    _write_system(system, coords, boxes)

    whole, whole_calls = _stat_with_batch_size(
        monkeypatch, str(system), MIXED_RCUT, NATOMS * NFRAMES
    )
    batched, batched_calls = _stat_with_batch_size(
        monkeypatch, str(system), MIXED_RCUT, NATOMS
    )

    # one call per set when everything fits, one call per frame when it does not
    assert whole_calls == [SET_SIZE] * (NFRAMES // SET_SIZE)
    assert batched_calls == [1] * NFRAMES
    assert len(batched_calls) > len(whole_calls)

    # the frame in the first set owns the shortest distance, the one in the
    # last set the largest neighbor count
    first = _single_frame_stat(monkeypatch, tmp_path / "first", coords[0], boxes[0])
    last = _single_frame_stat(monkeypatch, tmp_path / "last", coords[-1], boxes[-1])
    assert first[0] < last[0]
    assert first[1].max() < last[1].max()

    for stat in (whole, batched):
        assert stat[0] == first[0]
        np.testing.assert_array_equal(stat[1], last[1])


def test_batched_statistics_are_correct(monkeypatch: Any, tmp_path: Path) -> None:
    """The reduction over batches reproduces the known geometry.

    On a unit grid the shortest distance is 1.0, and just past it each atom
    sees its six axis-aligned images.
    """
    coords, boxes = _uniform_frames()
    system = tmp_path / "system"
    _write_system(system, coords, boxes)

    (min_nbor_dist, max_nbor_size), _ = _stat_with_batch_size(
        monkeypatch, str(system), UNIFORM_RCUT, NATOMS
    )

    assert min_nbor_dist == UNIFORM_SPACING
    np.testing.assert_array_equal(max_nbor_size, [6])
