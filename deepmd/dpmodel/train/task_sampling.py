# SPDX-License-Identifier: LGPL-3.0-or-later
"""Task-selection random streams, independent of backend communication."""

from collections.abc import (
    Mapping,
)
from typing import (
    Any,
)

import numpy as np


def validate_task_sampling_backend(training: Mapping[str, Any], backend: str) -> None:
    """Reject unsupported explicit options instead of silently ignoring them."""
    if "task_sampling" in training and backend != "pt_expt":
        raise ValueError(
            f"training.task_sampling is not supported by the {backend} backend; "
            "omit this field or use dp --pt-expt train."
        )


def task_sampling_mode(training: Mapping[str, Any], num_tasks: int) -> str:
    """Resolve the default while rejecting explicit options without task choices."""
    if "task_sampling" in training and num_tasks < 2:
        raise ValueError("training.task_sampling requires at least two tasks.")
    mode = training.get("task_sampling", "synchronized")
    if mode not in ("synchronized", "independent"):
        raise ValueError("training.task_sampling must be synchronized or independent")
    return mode


def make_task_rng(mode: str, seed: int | None, rank: int) -> np.random.Generator:
    """Keep the established seeded task sequence for each sampling mode."""
    task_seed = None if seed is None else seed % (2**32)
    if mode == "independent" and task_seed is not None:
        task_seed = [task_seed, rank]
    return np.random.default_rng(task_seed)


def task_rng_rank_state(mode: str, rng: np.random.Generator) -> dict[str, Any]:
    """Return the selection state to gather with each rank's data cursors."""
    return {"task_rng": rng.bit_generator.state} if mode == "independent" else {}


def task_rng_checkpoint(
    mode: str,
    seed: int | None,
    rng: np.random.Generator,
    ranks: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Keep the old record intact; identify the new layout without numeric versions."""
    if mode == "independent":
        return {
            "format": "per_rank_task_rng",
            "task_sampling": mode,
            "world_size": len(ranks),
            "configured_seed": seed,
            "ranks": ranks,
        }
    return {
        "version": 1,
        "world_size": len(ranks),
        "configured_seed": seed,
        "task_rng": rng.bit_generator.state,
        "ranks": ranks,
    }


def validate_task_rng_checkpoint(
    progress: Mapping[str, Any], *, sampling: str, reset: bool
) -> tuple[int, list[Mapping[str, Any]]]:
    """Validate the layout and policy before reset or topology fallbacks."""
    if not isinstance(progress, Mapping):
        raise ValueError("Data-progress checkpoint must be a mapping.")
    if "format" in progress:
        valid = (
            progress["format"] == "per_rank_task_rng"
            and "version" not in progress
            and progress.get("task_sampling") == "independent"
        )
    else:
        valid = (
            type(progress.get("version")) is int
            and progress["version"] == 1
            and progress.get("task_sampling", "synchronized") == "synchronized"
        )
    if not valid:
        raise ValueError("Unsupported distributed data-progress version or format.")
    world_size = progress.get("world_size")
    if type(world_size) is not int or world_size < 1:
        raise ValueError("Data-progress checkpoint world size must be positive.")
    ranks = progress.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != world_size:
        raise ValueError("Data-progress checkpoint must contain every rank.")
    if not all(isinstance(state, Mapping) for state in ranks):
        raise ValueError("Data-progress checkpoint rank states must be mappings.")
    if not reset and progress.get("task_sampling", "synchronized") != sampling:
        raise ValueError(
            "Task sampling changed; set training.reset_data_progress=true."
        )
    return world_size, ranks


def restore_task_rng(
    progress: Mapping[str, Any], *, seed: int | None, rank: int
) -> np.random.Generator:
    """Rebuild the stream without applying any state to a running trainer."""
    per_rank = progress.get("format") == "per_rank_task_rng"
    if "configured_seed" not in progress or (
        not per_rank and "task_rng" not in progress
    ):
        raise ValueError(
            "Data-progress checkpoint lacks its seed or task RNG identity; "
            "set training.reset_data_progress=true to restart data sampling."
        )
    if progress["configured_seed"] != seed:
        raise ValueError(
            "Training data progress is incompatible: training.seed changed; "
            "set training.reset_data_progress=true to restart data sampling."
        )
    if per_rank:
        if any("task_rng" not in state for state in progress["ranks"]):
            raise ValueError(
                "Independent task sampling requires every rank's RNG state."
            )
        state = progress["ranks"][rank]["task_rng"]
    else:
        state = progress["task_rng"]
    rng = np.random.default_rng(0)
    rng.bit_generator.state = state
    return rng
