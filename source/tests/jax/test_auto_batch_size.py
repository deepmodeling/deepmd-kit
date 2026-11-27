# SPDX-License-Identifier: LGPL-3.0-or-later
"""JAX atom budgets preserve inference results and memory backoff."""

import numpy as np
import pytest

from deepmd.jax.utils.auto_batch_size import (
    AutoBatchSize,
)
from deepmd.utils.errors import (
    OutOfMemoryError,
)


@pytest.fixture(autouse=True)
def _gpu_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DP_INFER_BATCH_SIZE", raising=False)
    monkeypatch.setattr(AutoBatchSize, "is_gpu_available", lambda self: True)


def test_default_budget_preserves_gpu_growth() -> None:
    auto = AutoBatchSize(initial_batch_size=16, silent=True)
    attempts = []

    def evaluate(nframes: int, start: int) -> tuple[int, None]:
        del start
        attempts.append(nframes)
        return nframes, None

    for _ in range(3):
        auto.execute(evaluate, 0, 1)
    assert attempts == [16, 32, 64]


def test_maximum_budget_stays_stable_across_atom_counts() -> None:
    auto = AutoBatchSize(maximum_batch_size=1024, silent=True)
    for natoms in (2, 3, 4, 5, 8, 12, 32, 100):
        values = np.arange(2 * (1024 // natoms) + 1).reshape(-1, 1)
        atoms_per_batch = []

        def evaluate(batch: np.ndarray) -> np.ndarray:
            atoms_per_batch.append(batch.shape[0] * natoms)
            return batch

        result = auto.execute_all(evaluate, len(values), natoms, values)
        np.testing.assert_array_equal(result, values)
        assert max(atoms_per_batch) <= 1024
        assert auto.current_batch_size == 1024


def test_maximum_budget_caps_the_initial_size() -> None:
    auto = AutoBatchSize(initial_batch_size=2048, maximum_batch_size=512, silent=True)
    assert auto.current_batch_size == 512


@pytest.mark.parametrize("explicit_size", [256, 4096])
def test_explicit_environment_budget_takes_precedence(
    explicit_size: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DP_INFER_BATCH_SIZE", str(explicit_size))
    auto = AutoBatchSize(initial_batch_size=128, maximum_batch_size=1024, silent=True)
    values = np.arange(explicit_size).reshape(-1, 1)
    attempts = []

    def evaluate(batch: np.ndarray) -> np.ndarray:
        attempts.append(batch.shape[0] * 2)
        return batch

    result = auto.execute_all(evaluate, len(values), 2, values)
    np.testing.assert_array_equal(result, values)
    assert attempts == [explicit_size, explicit_size]
    assert auto.current_batch_size == explicit_size


def test_maximum_budget_bounds_recovery_after_oom() -> None:
    auto = AutoBatchSize(maximum_batch_size=1024, silent=True)
    attempts = []
    for natoms in (2, 4):
        values = np.arange(1024).reshape(-1, 1)

        def evaluate(batch: np.ndarray) -> np.ndarray:
            attempts.append(batch.shape[0] * natoms)
            if batch.shape[0] * natoms > 256:
                raise RuntimeError(
                    "RESOURCE_EXHAUSTED: batch workspace exceeds capacity"
                )
            return batch

        result = auto.execute_all(evaluate, len(values), natoms, values)
        np.testing.assert_array_equal(result, values)
    assert attempts[:3] == [1024, 512, 256]
    assert max(attempts) <= 1024


@pytest.mark.parametrize("out_of_memory", [False, True])
def test_maximum_budget_keeps_indivisible_frame_boundary(out_of_memory: bool) -> None:
    auto = AutoBatchSize(maximum_batch_size=16, silent=True)
    values = np.arange(2).reshape(-1, 1)
    attempts = []

    def evaluate(batch: np.ndarray) -> np.ndarray:
        attempts.append(batch.shape[0])
        if out_of_memory:
            raise RuntimeError("RESOURCE_EXHAUSTED: a single frame exceeds capacity")
        return batch

    if out_of_memory:
        with pytest.raises(OutOfMemoryError, match="batch size is 1"):
            auto.execute_all(evaluate, len(values), 32, values)
        assert attempts == [1]
    else:
        np.testing.assert_array_equal(
            auto.execute_all(evaluate, len(values), 32, values),
            values,
        )
        assert attempts == [1, 1]


def test_maximum_budget_does_not_enable_cpu_growth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(AutoBatchSize, "is_gpu_available", lambda self: False)
    auto = AutoBatchSize(initial_batch_size=128, maximum_batch_size=1024, silent=True)
    for _ in range(3):
        nframes, _ = auto.execute(lambda nframes, start: (nframes, None), 0, 1)
        assert nframes == 128
    assert auto.current_batch_size == 128


@pytest.mark.parametrize("maximum_batch_size", [0, -1])
def test_maximum_budget_requires_positive_size(maximum_batch_size: int) -> None:
    with pytest.raises(ValueError, match="maximum_batch_size must be positive"):
        AutoBatchSize(maximum_batch_size=maximum_batch_size)
