# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compact LMDB restart state through the real pt_expt data adapter."""

import copy
import json
from contextlib import (
    ExitStack,
    closing,
)
from unittest.mock import (
    patch,
)

import lmdb
import msgpack
import pytest

from deepmd.dpmodel.utils import lmdb_data as lmdb_module
from deepmd.utils.data import (
    DataRequirementItem,
)

from .test_lmdb_data import (
    _create_lmdb,
    _create_lmdb_with_system_ids,
    _create_mixed_nloc_lmdb,
)


def _system(path, **kwargs):
    from deepmd.pt_expt.utils.lmdb_dataset import (
        LmdbDataSystem,
    )

    options = {"batch_size": 2, "seed": 7, "num_workers": 0, **kwargs}
    return LmdbDataSystem(path, ["O", "H"], **options)


def _draw(data, count):
    """Record frame identity and the consumed pass, not the prefetched pass."""
    return [
        (data.get_batch()["fid"], data.get_batch_pass_length()) for _ in range(count)
    ]


@pytest.fixture
def lmdb_path(tmp_path):
    return _create_lmdb(str(tmp_path / "progress.lmdb"), nframes=10)


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("seed", [None, 7])
@pytest.mark.parametrize("consumed", [0, 1, 4, 5, 6])
def test_logical_cursor_ignores_prefetch(lmdb_path, workers, seed, consumed):
    """Restart before, inside and across a pass, with real process prefetch."""
    options = {"num_workers": workers, "seed": seed}
    with closing(_system(lmdb_path, **options)) as original:
        _draw(original, consumed)
        saved = original.state_dict()
        assert saved["epoch"] == consumed // 5
        assert saved["batch_index"] == consumed % 5
        assert isinstance(saved["seed"], int)
        # Plain bounded metadata, not arrays, decoded data or permutations.
        assert len(json.dumps(saved)) < 2000
        expected = _draw(original, 12)

        with closing(_system(lmdb_path, **options)) as resumed:
            resumed.load_state_dict(saved)
            assert resumed.state_dict() == saved
            assert _draw(resumed, 12) == expected
            if workers and consumed:
                assert original._batch_iterator.started
                assert resumed._batch_iterator.started


def test_restore_is_lazy_until_trainer_configuration(tmp_path):
    """Reader/iterator construction predates label requirements and layout."""
    path = _create_mixed_nloc_lmdb(str(tmp_path / "mixed.lmdb"))
    with lmdb.open(path) as env, env.begin(write=True) as txn:
        for index in (1, 4, 7):
            key = format(index, "012d").encode()
            frame = msgpack.unpackb(txn.get(key), raw=False)
            del frame["energies"]
            txn.put(key, msgpack.packb(frame, use_bin_type=True))
    requirements = [DataRequirementItem("energy", 1)]
    options = {"batch_size": "mix:24", "seed": 1}
    build_batches = lmdb_module._build_all_batches
    with closing(_system(path, **options)) as original:
        original.add_data_requirements(requirements)
        original.use_ragged_batches(True)
        _draw(original, 3)
        saved = original.state_dict()
        expected = _draw(original, 15)

        with (
            patch.object(
                lmdb_module, "_build_all_batches", side_effect=AssertionError("sampled")
            ),
            closing(_system(path, **options)) as resumed,
        ):
            with patch.object(
                resumed._reader, "decode_batch", side_effect=AssertionError("decoded")
            ):
                resumed.load_state_dict(saved)
                assert resumed.state_dict() == saved
            # Registration must still be legal, and not plan a sampler pass.
            resumed.add_data_requirements(requirements)
            resumed.use_ragged_batches(True)
            assert resumed._sampler._batches is None
            # End the build guard before actually asking for the next batch.
            with patch.object(lmdb_module, "_build_all_batches", new=build_batches):
                assert _draw(resumed, 15) == expected


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("consumed", [4, 5, 6, 9])
def test_ragged_mixed_pass_lengths(tmp_path, workers, consumed):
    """A five-batch pass followed by a four-batch pass retains its boundary."""
    path = _create_mixed_nloc_lmdb(str(tmp_path / "mixed.lmdb"))
    options = {"batch_size": "mix:24", "seed": 1, "num_workers": workers}
    with closing(_system(path, **options)) as original:
        original.use_ragged_batches(True)
        history = _draw(original, consumed)
        assert history[0][1] == 5
        if consumed > 5:
            assert history[5][1] == 4
        saved = original.state_dict()
        expected = _draw(original, 15)
        with closing(_system(path, **options)) as resumed:
            resumed.load_state_dict(saved)
            resumed.use_ragged_batches(True)
            assert resumed.get_batch_pass_length() == expected[0][1]
            assert _draw(resumed, 15) == expected


@pytest.mark.parametrize("seed", [None, 7])
@pytest.mark.parametrize("consumed", [1, 3, 4])
def test_equal_rank_cursors_reconstruct_each_shard(lmdb_path, seed, consumed):
    """Equal progress reconstructs each rank's partition of the same pass."""
    with ExitStack() as stack:
        originals = [
            stack.enter_context(
                closing(_system(lmdb_path, seed=seed, rank=rank, world_size=2))
            )
            for rank in range(2)
        ]
        for data in originals:
            _draw(data, consumed)
        saved = originals[0].state_dict()
        assert originals[1].state_dict() == saved
        for rank, original in enumerate(originals):
            expected = _draw(original, 10)
            resumed = stack.enter_context(
                closing(_system(lmdb_path, seed=seed, rank=rank, world_size=2))
            )
            resumed.load_state_dict(saved)
            assert _draw(resumed, 10) == expected


def test_restore_drops_existing_prefetch(lmdb_path):
    with closing(_system(lmdb_path, num_workers=2)) as data:
        _draw(data, 3)
        saved = data.state_dict()
        expected = _draw(data, 9)
        pool = data._batch_iterator._pool
        with (
            patch.object(lmdb_module, "_build_all_batches") as build,
            patch.object(data._reader, "decode_batch") as decode,
            patch.object(lmdb_module, "_release_lmdb_executor") as release,
        ):
            data.load_state_dict(saved)
            build.assert_not_called()
            decode.assert_not_called()
            release.assert_not_called()
        assert data._batch_iterator._pool is pool
        assert _draw(data, 9) == expected


@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize("diagnostic_reads", [0, 1])
def test_same_pass_restore_reuses_sampling_plan(
    lmdb_path, world_size, diagnostic_reads
):
    """Display diagnostics must not rebuild a dataset-sized plan each time."""
    with closing(_system(lmdb_path, batch_size=1, world_size=world_size)) as data:
        _draw(data, 1)
        saved = data.state_dict()
        with patch.object(
            lmdb_module,
            "_build_all_batches",
            side_effect=AssertionError("rebuilt unchanged sampling plan"),
        ):
            _draw(data, diagnostic_reads)
            data.load_state_dict(saved)
            expected = _draw(data, 1)
            data.load_state_dict(saved)
            assert _draw(data, 1) == expected


@pytest.mark.parametrize("world_size", [1, 2])
def test_cached_plan_before_layout_selection_is_invalidated(tmp_path, world_size):
    path = _create_mixed_nloc_lmdb(str(tmp_path / "mixed.lmdb"))
    options = {"batch_size": "mix:24", "seed": 1, "world_size": world_size}
    with closing(_system(path, **options)) as original:
        original.use_ragged_batches(True)
        _draw(original, 1)
        saved = original.state_dict()
        expected = _draw(original, 10)
        with closing(_system(path, **options)) as resumed:
            # An earlier metadata query is allowed to have cached the default
            # rectangular plan before the trainer selects the final layout.
            assert resumed.nbatches[0] > 0
            resumed.load_state_dict(saved)
            resumed.use_ragged_batches(True)
            assert _draw(resumed, 10) == expected


def test_unequal_rank_cursors_restore_independently(lmdb_path):
    """Local filtering may have consumed a different batch count per rank."""
    with ExitStack() as stack:
        saved = []
        for rank in range(2):
            original = stack.enter_context(
                closing(_system(lmdb_path, rank=rank, world_size=2))
            )
            _draw(original, rank + 1)
            saved.append(original.state_dict())
            expected = _draw(original, 10)
            resumed = stack.enter_context(
                closing(_system(lmdb_path, rank=rank, world_size=2))
            )
            resumed.load_state_dict(saved[-1])
            assert _draw(resumed, 10) == expected
        assert saved[0]["batch_index"] != saved[1]["batch_index"]


def test_statistics_do_not_advance_training(lmdb_path):
    with closing(_system(lmdb_path)) as data:
        saved = data.state_dict()
        data.get_stat_batch(0)
        assert data.state_dict() == saved


def test_auto_prob_targets_are_reproduced_and_validated(tmp_path):
    path = _create_lmdb_with_system_ids(str(tmp_path / "weighted.lmdb"), [4, 8])
    style = "prob_sys_size;0:1:0.7;1:2:0.3"
    with closing(_system(path, auto_prob_style=style)) as original:
        _draw(original, 4)
        saved = original.state_dict()
        expected = _draw(original, 12)
        with closing(_system(path, auto_prob_style=style)) as resumed:
            resumed.load_state_dict(saved)
            assert _draw(resumed, 12) == expected
        with closing(_system(path, auto_prob_style="prob_sys_size")) as changed:
            with pytest.raises(ValueError, match="block_targets"):
                changed.load_state_dict(saved)


@pytest.mark.parametrize(
    ("changed", "field"),
    [
        ({"batch_size": 3}, "batch_rule"),
        ({"seed": 12}, "configured_seed"),
        ({"world_size": 2}, "world_size"),
    ],
)
def test_sampling_configuration_mismatch(lmdb_path, changed, field):
    with closing(_system(lmdb_path)) as original:
        _draw(original, 2)
        saved = original.state_dict()
        with closing(_system(lmdb_path, **changed)) as resumed:
            with pytest.raises(ValueError, match=field):
                resumed.load_state_dict(saved)


@pytest.mark.parametrize("field", ["ragged", "availability_keys"])
def test_late_configuration_mismatch_is_checked_before_decode(tmp_path, field):
    path = _create_mixed_nloc_lmdb(str(tmp_path / "mixed.lmdb"))
    with closing(_system(path, batch_size="mix:24")) as original:
        if field == "ragged":
            original.use_ragged_batches(True)
        else:
            original.add_data_requirements([DataRequirementItem("energy", 1)])
        _draw(original, 2)
        saved = original.state_dict()
        with closing(_system(path, batch_size="mix:24")) as resumed:
            resumed.load_state_dict(saved)
            with (
                patch.object(resumed._reader, "decode_batch") as decode,
                pytest.raises(ValueError, match=field),
            ):
                resumed.get_batch()
            decode.assert_not_called()


@pytest.mark.parametrize(
    ("key", "value"),
    [("version", 99), ("epoch", -1), ("batch_index", 1.5), ("seed", None)],
)
def test_invalid_state_is_rejected(lmdb_path, key, value):
    with closing(_system(lmdb_path)) as data:
        saved = data.state_dict()
        invalid = copy.deepcopy(saved)
        invalid[key] = value
        with pytest.raises(ValueError):
            data.load_state_dict(invalid)
        assert data.state_dict() == saved


def test_out_of_range_cursor_is_checked_lazily(lmdb_path):
    with closing(_system(lmdb_path)) as data:
        saved = data.state_dict()
        saved["batch_index"] = 5
        data.load_state_dict(saved)
        with pytest.raises(ValueError, match="out of range"):
            data.get_batch()


def test_dataset_transaction_change_is_rejected(lmdb_path):
    with closing(_system(lmdb_path)) as original:
        saved = original.state_dict()
    with lmdb.open(lmdb_path) as env, env.begin(write=True) as txn:
        key = b"000000000000"
        frame = msgpack.unpackb(txn.get(key), raw=False)
        del frame["energies"]
        txn.put(key, msgpack.packb(frame, use_bin_type=True))
    with closing(_system(lmdb_path)) as resumed:
        with pytest.raises(ValueError, match="transaction"):
            resumed.load_state_dict(saved)


def test_different_dataset_is_rejected(lmdb_path, tmp_path):
    other = _create_lmdb(str(tmp_path / "other.lmdb"), nframes=10)
    with closing(_system(lmdb_path)) as original, closing(_system(other)) as resumed:
        with pytest.raises(ValueError, match="path"):
            resumed.load_state_dict(original.state_dict())
