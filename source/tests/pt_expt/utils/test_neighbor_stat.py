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

# A 3x3x3 grid of one atom type, spaced 1.0 apart in a 3.0 box, so the
# shortest neighbor distance is exactly 1.0 and every atom has the same
# neighborhood regardless of which frames land in a batch.
NATOMS = 27
BOX_LENGTH = 3.0
NFRAMES = 8
SET_SIZE = 4
RCUT = 1.001


def _write_system(path: Path) -> None:
    """Write a multi-frame, multi-set system in the DeePMD npy layout."""
    grid = np.mgrid[0:2:3j, 0:2:3j, 0:2:3j]
    positions = np.vstack([axis.ravel() for axis in grid]).T
    coord = np.repeat(positions[np.newaxis, :, :], NFRAMES, axis=0).reshape(
        NFRAMES, NATOMS * 3
    )
    box = np.tile((np.eye(3) * BOX_LENGTH).ravel(), (NFRAMES, 1))

    path.mkdir(parents=True, exist_ok=True)
    (path / "type.raw").write_text("\n".join(["0"] * NATOMS) + "\n")
    (path / "type_map.raw").write_text("TYPE\n")
    for index, start in enumerate(range(0, NFRAMES, SET_SIZE)):
        subset = path / f"set.{index:03d}"
        subset.mkdir()
        frames = slice(start, start + SET_SIZE)
        np.save(subset / "coord.npy", coord[frames])
        np.save(subset / "box.npy", box[frames])
        np.save(subset / "energy.npy", np.zeros(SET_SIZE))
        np.save(subset / "force.npy", np.zeros((SET_SIZE, NATOMS * 3)))


def _stat_with_batch_size(
    monkeypatch: Any, system: str, batch_atoms: int
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
            rcut=RCUT,
            type_map=["TYPE"],
        )
        data.get_batch()
        stat = NeighborStat(data.get_ntypes(), RCUT, mixed_type=False).get_stat(data)
    return stat, frames_per_call


def test_batching_splits_the_work_and_keeps_the_result(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """Batching must change how the work is split, not what it produces.

    Sending a whole set at once allocates an ``[nframes, nloc, nall, 3]``
    tensor, which is why the statistics are batched. Comparing the results
    alone would pass without any batching, so the split itself is asserted.
    """
    system = tmp_path / "system"
    _write_system(system)

    whole, whole_calls = _stat_with_batch_size(
        monkeypatch, str(system), NATOMS * NFRAMES
    )
    batched, batched_calls = _stat_with_batch_size(monkeypatch, str(system), NATOMS)

    # one call per set when everything fits, one call per frame when it does not
    assert whole_calls == [SET_SIZE] * (NFRAMES // SET_SIZE)
    assert batched_calls == [1] * NFRAMES
    assert len(batched_calls) > len(whole_calls)

    assert batched[0] == whole[0]
    np.testing.assert_array_equal(batched[1], whole[1])


def test_batched_statistics_are_correct(monkeypatch: Any, tmp_path: Path) -> None:
    """The reduction over batches reproduces the known geometry.

    On a unit grid the shortest distance is 1.0, and within ``rcut`` each
    atom sees its six axis-aligned images.
    """
    system = tmp_path / "system"
    _write_system(system)

    (min_nbor_dist, max_nbor_size), _ = _stat_with_batch_size(
        monkeypatch, str(system), NATOMS
    )

    assert min_nbor_dist == 1.0
    np.testing.assert_array_equal(max_nbor_size, [6])
