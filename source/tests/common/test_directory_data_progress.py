# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compact, lazy batch-progress restoration for directory and HDF5 sources."""

import copy
import json
from unittest.mock import (
    patch,
)

import h5py
import numpy as np
import pytest

from deepmd.utils import random as dp_random
from deepmd.utils.data import (
    DeepmdData,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)


def _write_system(root, *, hdf5=False, nsets=1, copies=False, offset=0, nframes=7):
    """Write uniquely identifiable frames with optional zero/repeated copies."""
    arrays = {"type.raw": np.array([0, 0], dtype=np.int32)}
    for index in range(nsets):
        size = nframes + index
        frame_ids = offset + index * 100 + np.arange(size)
        prefix = f"set.{index:03d}/"
        arrays[prefix + "coord.npy"] = np.repeat(frame_ids[:, None], 6, axis=1)
        arrays[prefix + "box.npy"] = np.tile(np.eye(3).reshape(1, 9), (size, 1))
        arrays[prefix + "energy.npy"] = frame_ids[:, None] * 10
        if copies:
            arrays[prefix + "numb_copy.npy"] = (np.arange(size) % 4)[:, None]
    if hdf5:
        filename = root.with_suffix(".h5")
        with h5py.File(filename, "w") as handle:
            for name, array in arrays.items():
                handle.create_dataset("system/" + name, data=array)
        return str(filename) + "#/system"
    root.mkdir()
    for name, array in arrays.items():
        target = root / name
        target.parent.mkdir(exist_ok=True)
        if name.endswith(".raw"):
            np.savetxt(target, array, fmt="%d")
        else:
            np.save(target, array)
    return str(root)


def _assert_batch_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)


@pytest.mark.parametrize("hdf5", [False, True], ids=["directory", "hdf5"])
@pytest.mark.parametrize("nsets", [1, 3])
@pytest.mark.parametrize("copies", [False, True])
@pytest.mark.parametrize("consumed", [0, 1, 3, 11])
def test_reader_resume_multiple_rounds(tmp_path, hdf5, nsets, copies, consumed):
    """Resume before/within/after sets, including previously reshuffled caches."""
    path = _write_system(tmp_path / "data", hdf5=hdf5, nsets=nsets, copies=copies)
    original = DeepmdData(path)
    original.add("energy", 1)
    dp_random.seed(482)
    for _ in range(consumed):
        original.get_batch(2)
    progress = original.state_dict()
    rng_state = dp_random.get_state()
    expected = [(original.get_batch(2), dp_random.random()) for _ in range(40)]

    restored = DeepmdData(path)
    restored.load_state_dict(progress)
    # Requirements can be installed after logical state restoration.
    restored.add("energy", 1)
    dp_random.set_state(rng_state)
    for batch, random_value in expected:
        _assert_batch_equal(restored.get_batch(2), batch)
        assert dp_random.random() == random_value


@pytest.mark.parametrize("hdf5", [False, True], ids=["directory", "hdf5"])
def test_restore_is_lazy_and_invalidates_changed_requirements(tmp_path, hdf5):
    path = _write_system(tmp_path / "data", hdf5=hdf5)
    data = DeepmdData(path)
    data.get_batch(2)
    progress = data.state_dict()
    data.get_test()
    assert hasattr(data, "batch_set")
    assert hasattr(data, "test_set")
    data.add("energy", 1)
    with (
        patch.object(data, "_load_set", side_effect=AssertionError("decoded set")),
        patch.object(data, "_get_nframes", side_effect=AssertionError("scanned set")),
        patch.object(
            type(data.dirs[0]), "load_numpy", side_effect=AssertionError("read data")
        ),
    ):
        data.load_state_dict(progress)
        assert not hasattr(data, "batch_set")
        assert not hasattr(data, "test_set")
    with patch.object(data, "_load_set", wraps=data._load_set) as load:
        batch = data.get_batch(2)
        load.assert_called_once()
    np.testing.assert_array_equal(batch["energy"][:, 0], batch["coord"][:, 0] * 10)


@pytest.mark.parametrize("extra_batches", [1, 5])
def test_in_process_restore_reuses_canonical_cache(tmp_path, extra_batches):
    """Display/statistics snapshots must not force another set decode."""
    path = _write_system(tmp_path / "data")
    data = DeepmdData(path)
    data.get_batch(2)
    state = data.state_dict()
    expected = data.get_batch(2)
    for _ in range(extra_batches):
        data.get_batch(2)
    cached_set = data.batch_set
    with patch.object(data, "_load_set", side_effect=AssertionError("decoded set")):
        data.load_state_dict(state)
        _assert_batch_equal(data.get_batch(2), expected)
    assert data.batch_set is cached_set


def test_in_process_restore_discards_other_set_cache(tmp_path):
    path = _write_system(tmp_path / "data", nsets=2)
    data = DeepmdData(path)
    data.get_batch(2)
    state = data.state_dict()
    expected = data.get_batch(2)
    for _ in range(2):
        data.get_batch(2)
    assert data._batch_set_name == data.dirs[1]
    data.load_state_dict(state)
    assert not hasattr(data, "batch_set")
    _assert_batch_equal(data.get_batch(2), expected)


def test_exhausted_set_is_not_decoded_on_resume(tmp_path):
    path = _write_system(tmp_path / "data", nsets=2)
    original = DeepmdData(path)
    for _ in range(3):
        original.get_batch(2)
    state = original.state_dict()
    restored = DeepmdData(path)
    restored.load_state_dict(state)
    with patch.object(restored, "_load_set", wraps=restored._load_set) as load:
        restored.get_batch(2)
    assert [call.args[0] for call in load.call_args_list] == [restored.dirs[1]]


def test_numb_copy_does_not_compound_between_rounds(tmp_path):
    path = _write_system(tmp_path / "data", copies=True)
    data = DeepmdData(path)
    frame_ids = np.arange(7)
    expected_ids = np.repeat(frame_ids, frame_ids % 4)
    for _ in range(5):
        batch = data.get_batch(len(expected_ids))
        np.testing.assert_array_equal(np.sort(batch["coord"][:, 0]), expected_ids)
        assert data.batch_set["coord"].shape[0] == 7


@pytest.mark.parametrize("copies", [False, True])
def test_batch_order_matches_canonical_shuffle(tmp_path, copies):
    """Index-only batching retains the established shuffle/RNG semantics."""
    path = _write_system(tmp_path / "data", nsets=3, copies=copies)
    data = DeepmdData(path)
    dp_random.seed(716)
    for index in range(6):
        rng_state = dp_random.get_state()
        expected, _ = data._shuffle_data(data._load_set(data.dirs[index % 3]))
        expected_rng = dp_random.get_state()
        dp_random.set_state(rng_state)
        # An oversized batch returns the entire set, preserving legacy
        # short-batch behavior rather than crossing into the following set.
        _assert_batch_equal(data.get_batch(1000), expected)
        assert dp_random.get_state() == expected_rng


@pytest.mark.parametrize("hdf5", [False, True], ids=["directory", "hdf5"])
@pytest.mark.parametrize("batch_size", [2, "mixed:3"])
def test_system_resume_random_choices(tmp_path, hdf5, batch_size):
    paths = [
        _write_system(
            tmp_path / f"data{index}",
            hdf5=hdf5,
            nsets=index + 1,
            copies=True,
            offset=index * 1000,
        )
        for index in range(2)
    ]
    original = DeepmdDataSystem(paths, batch_size, 1, sys_probs=[0.3, 0.7])
    dp_random.seed(918)
    for _ in range(13):
        original.get_batch()
    progress = original.state_dict()
    rng_state = dp_random.get_state()
    expected = [original.get_batch() for _ in range(60)]
    restored = DeepmdDataSystem(paths, batch_size, 1, sys_probs=[0.3, 0.7])
    with patch.object(DeepmdData, "_load_set", side_effect=AssertionError("decoded")):
        restored.load_state_dict(progress)
    dp_random.set_state(rng_state)
    for batch in expected:
        _assert_batch_equal(restored.get_batch(), batch)


@pytest.mark.parametrize("batch_size", [2, "mixed:3"])
def test_system_resume_restores_exact_probabilities_after_roundoff(
    tmp_path, batch_size
):
    paths = [
        _write_system(tmp_path / f"data{index}", offset=index * 1000)
        for index in range(2)
    ]
    # Put the first random draw exactly on a selection threshold. Even a
    # one-ULP probability difference then changes the selected system.
    dp_random.seed(1)
    threshold = float(dp_random.random())
    saved_probs = np.array([threshold, 1 - threshold])
    rounded_threshold = np.nextafter(threshold, np.inf)
    recomputed_probs = np.array([rounded_threshold, 1 - rounded_threshold])
    dp_random.seed(1)
    assert dp_random.choice(2, p=saved_probs) == 1
    dp_random.seed(1)
    assert dp_random.choice(2, p=recomputed_probs) == 0

    original = DeepmdDataSystem(paths, batch_size, 1, sys_probs=saved_probs.tolist())
    state = original.state_dict()
    dp_random.seed(1)
    rng_state = dp_random.get_state()
    expected = [original.get_batch() for _ in range(20)]
    expected_rng_state = dp_random.get_state()

    restored = DeepmdDataSystem(
        paths, batch_size, 1, sys_probs=recomputed_probs.tolist()
    )
    with patch.object(DeepmdData, "_load_set", side_effect=AssertionError("decoded")):
        restored.load_state_dict(state)
    np.testing.assert_array_equal(restored.sys_probs, saved_probs)
    # The restored probabilities must not alias the caller's checkpoint.
    state["sys_probs"][0] = 0
    np.testing.assert_array_equal(restored.sys_probs, saved_probs)
    dp_random.set_state(rng_state)
    for batch in expected:
        _assert_batch_equal(restored.get_batch(), batch)
    assert dp_random.get_state() == expected_rng_state


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_system_resume_preserves_exactly_matching_probability_dtype(tmp_path, dtype):
    paths = [
        _write_system(tmp_path / f"data{index}", offset=index * 1000)
        for index in range(3)
    ]
    # Float32 thirds are valid for NumPy choice but their float64 sum is
    # outside its tighter float64 normalization tolerance.
    probabilities = list(np.full(3, 1 / 3, dtype=dtype))
    original = DeepmdDataSystem(paths, 2, 1, sys_probs=probabilities)
    state = original.state_dict()
    dp_random.seed(193)
    rng_state = dp_random.get_state()
    expected = [original.get_batch() for _ in range(20)]

    restored = DeepmdDataSystem(paths, 2, 1, sys_probs=probabilities)
    restored.load_state_dict(state)
    assert restored.sys_probs.dtype == dtype
    np.testing.assert_array_equal(restored.sys_probs, probabilities)
    dp_random.set_state(rng_state)
    for batch in expected:
        _assert_batch_equal(restored.get_batch(), batch)


@pytest.mark.parametrize(
    "saved_probs",
    [
        None,
        0.5,
        [0.5],
        [[0.3, 0.7]],
        [0.3, 0.7, 0.0],
        [float("nan"), 0.7],
        [float("inf"), 0.7],
        [-0.3, 1.3],
        ["invalid", 0.7],
        [0.4, 0.6],
        [0.3 + 1e-8, 0.7 - 1e-8],
    ],
)
def test_invalid_system_probabilities_are_rejected_without_mutation(
    tmp_path, saved_probs
):
    paths = [_write_system(tmp_path / f"data{index}") for index in range(2)]
    data = DeepmdDataSystem(paths, 2, 1, sys_probs=[0.3, 0.7])
    data.get_batch()
    initial = data.state_dict()
    invalid = copy.deepcopy(initial)
    invalid["sys_probs"] = saved_probs
    with pytest.raises(ValueError, match="probabilities"):
        data.load_state_dict(invalid)
    assert data.state_dict() == initial


@pytest.mark.parametrize("saved_zero", [False, True])
def test_system_probability_roundoff_does_not_change_excluded_systems(
    tmp_path, saved_zero
):
    paths = [_write_system(tmp_path / f"data{index}") for index in range(2)]
    positive = np.nextafter(0.0, 1.0)
    current_probs = [positive if saved_zero else 0.0, 1.0]
    data = DeepmdDataSystem(paths, 2, 1, sys_probs=current_probs)
    initial = data.state_dict()
    invalid = copy.deepcopy(initial)
    invalid["sys_probs"] = [0.0 if saved_zero else positive, 1.0]
    with pytest.raises(ValueError, match="probabilities"):
        data.load_state_dict(invalid)
    assert data.state_dict() == initial


def test_system_probabilities_are_not_restored_before_source_validation(tmp_path):
    paths = [_write_system(tmp_path / f"data{index}") for index in range(2)]
    data = DeepmdDataSystem(paths, 2, 1, sys_probs=[0.3, 0.7])
    initial = data.state_dict()
    invalid = copy.deepcopy(initial)
    invalid["sys_probs"][0] = np.nextafter(0.3, 1.0)
    invalid["systems"][-1]["version"] = 99
    with pytest.raises(ValueError):
        data.load_state_dict(invalid)
    assert data.state_dict() == initial


def test_checkpoint_is_compact_and_detached(tmp_path):
    path = _write_system(tmp_path / "data", nframes=10000)
    data = DeepmdData(path)
    data.get_batch(2)
    state = data.state_dict()
    # The fixed-size RNG state is JSON-serializable; no frame permutation or
    # label arrays should scale this snapshot with the number of frames.
    assert len(json.dumps(state)) < 12000
    assert state["set_size"] == 10000
    state["identity"]["prefix_sum"][0] = -1
    state["shuffle_state"][1][0] = 0
    assert data.state_dict()["identity"]["prefix_sum"][0] == 10000
    assert data.state_dict()["shuffle_state"][1][0] != 0


@pytest.mark.parametrize("change", ["source", "batch_size", "sys_probs", "order"])
def test_incompatible_system_progress_is_rejected(tmp_path, change):
    paths = [_write_system(tmp_path / f"data{index}") for index in range(2)]
    original = DeepmdDataSystem(paths, 2, 1, sys_probs=[0.3, 0.7])
    original.get_batch()
    state = original.state_dict()
    kwargs = {"batch_size": 2, "test_size": 1, "sys_probs": [0.3, 0.7]}
    if change == "source":
        paths[0] = _write_system(tmp_path / "other")
    elif change == "batch_size":
        kwargs["batch_size"] = 1
    elif change == "sys_probs":
        kwargs["sys_probs"] = [0.4, 0.6]
    elif change == "order":
        paths.reverse()
    restored = DeepmdDataSystem(paths, **kwargs)
    initial = restored.state_dict()
    with pytest.raises(ValueError):
        restored.load_state_dict(state)
    assert restored.state_dict() == initial


@pytest.mark.parametrize(
    "field,value",
    [("version", 99), ("iterator", -1), ("set_count", -1), ("shuffle_state", None)],
)
def test_invalid_reader_progress_is_rejected(tmp_path, field, value):
    path = _write_system(tmp_path / "data")
    data = DeepmdData(path)
    data.get_batch(2)
    initial = data.state_dict()
    invalid = copy.deepcopy(initial)
    invalid[field] = value
    with pytest.raises(ValueError):
        data.load_state_dict(invalid)
    assert data.state_dict() == initial
