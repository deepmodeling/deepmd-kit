# SPDX-License-Identifier: LGPL-3.0-or-later
"""Competing processes must not reclaim a live conversion writer's lock."""

import json
import multiprocessing
from pathlib import (
    Path,
)
from unittest.mock import (
    patch,
)

import pytest

from deepmd.utils import (
    data_conversion,
)


def _paused_recovery(path, checked, resume):
    """Pause at the vulnerable boundary after the real inode comparison."""
    same_lock_file = data_conversion._same_lock_file

    def pause_after_check(lock_path, expected):
        same = same_lock_file(lock_path, expected)
        checked.set()
        assert resume.wait(20), "parent did not resume stale-lock recovery"
        return same

    with patch.object(data_conversion, "_same_lock_file", pause_after_check):
        assert data_conversion._recover_stale_conversion_lock(Path(path))


def _successor(path, attempting, acquired, release):
    """Recover the dead owner and hold a real successor lock until released."""
    path = Path(path)
    attempting.set()
    assert data_conversion._recover_stale_conversion_lock(path)
    lock = data_conversion._ConversionLock(path)
    try:
        acquired.set()
        assert release.wait(20), "parent did not release successor"
    finally:
        lock.release()


@pytest.mark.timeout(40)
def test_competing_recovery_cannot_unlink_live_successor(tmp_path):
    """A successor cannot enter the inode-check/unlink critical section."""
    path = tmp_path / "conversion.lock"
    path.write_text(
        json.dumps(
            {
                "hostname": data_conversion.socket.gethostname(),
                "pid": 2147483647,
                "process_start": "missing",
            }
        )
    )
    ctx = multiprocessing.get_context("spawn")
    checked, resume, attempting, acquired, release = [ctx.Event() for _ in range(5)]
    first = ctx.Process(target=_paused_recovery, args=(str(path), checked, resume))
    second = ctx.Process(
        target=_successor, args=(str(path), attempting, acquired, release)
    )
    processes = []
    try:
        first.start()
        processes.append(first)
        assert checked.wait(15)
        second.start()
        processes.append(second)
        assert attempting.wait(15)
        # The old protocol allowed B to acquire while A was paused, after
        # which A deleted B's live owner file. The guard forbids that schedule.
        assert not acquired.wait(0.5)
        resume.set()
        first.join(10)
        assert first.exitcode == 0
        assert acquired.wait(10)
        owner = json.loads(path.read_text())
        assert owner["pid"] == second.pid
        assert data_conversion._lock_owner_is_alive(owner)
        assert not data_conversion._recover_stale_conversion_lock(path)
        with pytest.raises(FileExistsError):
            data_conversion._ConversionLock(path)
        assert json.loads(path.read_text()) == owner
        release.set()
        second.join(10)
        assert second.exitcode == 0
        assert not path.exists()
    finally:
        resume.set()
        release.set()
        for process in processes:
            process.join(5)
            if process.is_alive():
                process.terminate()
                process.join(5)
