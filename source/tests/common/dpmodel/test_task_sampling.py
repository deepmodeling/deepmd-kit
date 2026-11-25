# SPDX-License-Identifier: LGPL-3.0-or-later
"""Task-stream parity and checkpoint layout without a backend or collectives."""

from copy import (
    deepcopy,
)

import numpy as np
import pytest

from deepmd.dpmodel.train import (
    TrainingTask,
    TrainingTaskCollection,
)
from deepmd.dpmodel.train.task_sampling import (
    make_task_rng,
    restore_task_rng,
    task_rng_checkpoint,
    task_rng_rank_state,
    task_sampling_mode,
    validate_task_rng_checkpoint,
)


@pytest.mark.parametrize("sampling", ["synchronized", "independent"])
@pytest.mark.parametrize("seed", [0, 19, 2**40 + 19])
@pytest.mark.parametrize("rank", [0, 1, 7])
def test_shared_selection_preserves_existing_sequence(sampling, seed, rank) -> None:
    """Compare every draw and the final RNG state against the pre-refactor code."""
    task_seed = seed % (2**32)
    if sampling == "independent":
        task_seed = [task_seed, rank]
    reference = np.random.default_rng(task_seed)
    actual = make_task_rng(sampling, seed, rank)
    tasks = TrainingTaskCollection(
        [TrainingTask(key, None) for key in ("first", "second", "third")],
        probabilities=[0.7, 0.2, 0.1],
    )
    expected = [
        tasks.keys[int(reference.choice(np.arange(len(tasks)), p=tasks.probabilities))]
        for _ in range(1000)
    ]
    assert [tasks.select(choice=actual.choice).key for _ in range(1000)] == expected
    assert actual.bit_generator.state == reference.bit_generator.state


def test_single_task_does_not_consume_task_rng() -> None:
    assert task_sampling_mode({}, 1) == "synchronized"
    rng = make_task_rng("synchronized", 19, 0)
    before = deepcopy(rng.bit_generator.state)
    tasks = TrainingTaskCollection.single(None)
    for _ in range(10):
        assert tasks.select(choice=rng.choice) is tasks[tasks.keys[0]]
    assert rng.bit_generator.state == before


@pytest.mark.parametrize("seed", [19, None])
@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize("sampling", ["synchronized", "independent"])
def test_task_rng_checkpoint_roundtrip(sampling, seed, world_size) -> None:
    generators = [make_task_rng(sampling, seed, rank) for rank in range(world_size)]
    for rng in generators:
        rng.random(13)
    ranks = [task_rng_rank_state(sampling, rng) for rng in generators]
    progress = task_rng_checkpoint(sampling, seed, generators[0], ranks)
    if sampling == "synchronized":
        # Exactly the old keys, values and insertion order, with no new defaults.
        old_record = {
            "version": 1,
            "world_size": world_size,
            "configured_seed": seed,
            "task_rng": generators[0].bit_generator.state,
            "ranks": ranks,
        }
        assert progress == old_record
        assert list(progress) == list(old_record)
    else:
        assert progress["format"] == "per_rank_task_rng"
        assert "version" not in progress
        assert "task_rng" not in progress
    assert validate_task_rng_checkpoint(progress, sampling=sampling, reset=False) == (
        world_size,
        ranks,
    )
    for rank in range(world_size):
        restored = restore_task_rng(progress, seed=seed, rank=rank)
        reference = generators[rank] if sampling == "independent" else generators[0]
        # Do not advance a synchronized reference before the next rank compares.
        expected = np.random.default_rng()
        expected.bit_generator.state = reference.bit_generator.state
        np.testing.assert_array_equal(restored.random(100), expected.random(100))


@pytest.mark.parametrize("changed", [{"version": 1}, {"format": "unknown"}])
def test_ambiguous_or_unknown_layout_is_rejected(changed) -> None:
    rng = make_task_rng("independent", 19, 0)
    progress = task_rng_checkpoint(
        "independent", 19, rng, [task_rng_rank_state("independent", rng)]
    )
    with pytest.raises(ValueError, match="version or format"):
        validate_task_rng_checkpoint(
            progress | changed, sampling="independent", reset=True
        )
