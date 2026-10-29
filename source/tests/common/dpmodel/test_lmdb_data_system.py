# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared LMDB data-system statistics and bounded validation views."""

from pathlib import (
    Path,
)

import numpy as np

from deepmd.dpmodel.utils.lmdb_data import (
    DistributedLmdbBatchSampler,
)
from deepmd.dpmodel.utils.lmdb_data_system import (
    LmdbDataSystem,
)
from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.model_stat import (
    make_stat_input,
)

from .test_lmdb_data import (
    _create_lmdb,
    _create_mixed_nloc_lmdb,
)


def test_ragged_batches_statistics_and_chunked_validation(tmp_path: Path) -> None:
    """Training, statistics and validation retain every frame of mixed sizes."""
    path = _create_mixed_nloc_lmdb(str(tmp_path / "mixed.lmdb"))
    data = LmdbDataSystem(path, ["O", "H"], batch_size="mix:30", seed=1, num_workers=0)
    data.add_data_requirements(
        [
            DataRequirementItem("energy", 1, atomic=False, must=False, high_prec=True),
            DataRequirementItem("force", 3, atomic=True, must=False),
        ]
    )
    data.use_ragged_batches(True)
    try:
        batch = data.get_batch()
        assert batch["atype"].ndim == 1
        assert batch["coord"].shape[0] == int(batch["n_node"].sum())
        assert (batch["atype"] >= 0).all()
        pass_length = data.get_batch_pass_length()
        assert pass_length == data.nbatches[0]
        frame_ids = list(batch["fid"])
        for _ in range(pass_length - 1):
            frame_ids.extend(data.get_batch()["fid"])
        np.testing.assert_array_equal(np.sort(frame_ids), np.arange(10))

        sampled = make_stat_input(data, 100)
        assert {sample["atype"].shape[1] for sample in sampled} == {6, 9, 12}
        assert sum(sample["coord"].shape[0] for sample in sampled) == 10

        frame_count = 0
        for view in data.data_systems:
            for chunk in view.iter_test(chunk_atoms=24):
                assert chunk["type"].ndim == 2
                frame_count += chunk["type"].shape[0]
        assert frame_count == 10
    finally:
        data.close()


def test_grouped_batches_match_rank_order_without_crossing_epochs(
    tmp_path: Path,
) -> None:
    """A controller consumes rank-strided batches with the same epoch padding."""
    path = _create_lmdb(str(tmp_path / "grouped.lmdb"), nframes=10, natoms=6)
    data = LmdbDataSystem(
        path,
        ["O", "H"],
        batch_size=2,
        seed=17,
        num_workers=0,
        batch_multiple=2,
    )
    ranks = [
        DistributedLmdbBatchSampler(data._reader, rank, 2, seed=17) for rank in range(2)
    ]
    try:
        assert data.get_batch_pass_length() == 6
        for epoch in range(2):
            for sampler in ranks:
                sampler.set_epoch(epoch)
            expected = [
                batch
                for pair in zip(*[list(sampler) for sampler in ranks], strict=True)
                for batch in pair
            ]
            actual = [data.get_batch()["fid"] for _ in range(6)]
            assert actual == expected
            assert data.get_batch_pass_length() == 6
    finally:
        data.close()
