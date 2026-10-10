# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the backend-independent checkpoint layout and retention."""

import platform
from pathlib import (
    Path,
)

import pytest

from deepmd.dpmodel.train import (
    CheckpointStore,
    build_checkpoint_stores,
    resolve_checkpoint_path,
    resolve_keep_ckpt_count,
)


def _write(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    return path


def _assert_prefix_alias(alias: Path, target: Path, relative_target: str) -> None:
    assert alias.exists()
    if platform.system() == "Windows":
        assert alias.read_bytes() == target.read_bytes()
        return
    assert alias.is_symlink()
    assert alias.resolve() == target
    assert alias.readlink().as_posix() == relative_target


def test_numbered_paths_follow_save_dir(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "run" / "model.ckpt")
    assert store.path_for(7) == tmp_path / "run" / "model.ckpt-7.pt"

    relocated = CheckpointStore(
        tmp_path / "run" / "model.ckpt", save_dir=tmp_path / "ckpts"
    )
    assert relocated.path_for(7) == tmp_path / "ckpts" / "model.ckpt-7.pt"


def test_publish_links_the_prefix_relative_to_its_directory(tmp_path: Path) -> None:
    store = CheckpointStore(
        tmp_path / "model.ckpt", pointer_file=tmp_path / "checkpoint"
    )
    path = _write(store.path_for(3))

    store.publish(path)

    latest = tmp_path / "model.ckpt.pt"
    _assert_prefix_alias(latest, path, "model.ckpt-3.pt")
    assert (tmp_path / "checkpoint").read_text() == str(path)


def test_publish_reaches_across_save_dir(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "model.ckpt", save_dir=tmp_path / "ckpts")
    store.prepare()
    path = _write(store.path_for(3))

    store.publish(path)

    latest = tmp_path / "model.ckpt.pt"
    _assert_prefix_alias(latest, path, "ckpts/model.ckpt-3.pt")


def test_prune_keeps_the_newest_checkpoints(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "model.ckpt", max_keep=2)
    for step in (1, 2, 3):
        _write(store.path_for(step))
    store.publish(store.path_for(3))

    store.prune(store.path_for(3))

    assert not store.path_for(1).exists()
    assert store.path_for(2).exists()
    assert store.path_for(3).exists()
    assert (tmp_path / "model.ckpt.pt").exists()


def test_prune_keeps_every_checkpoint_below_the_window(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "model.ckpt", max_keep=10)
    for step in range(1, 10):
        _write(store.path_for(step))

    store.prune(store.path_for(9))

    assert all(store.path_for(step).exists() for step in range(1, 10))


def test_prune_drops_checkpoints_left_by_a_longer_run(tmp_path: Path) -> None:
    """A rerun in a finished directory keeps its own checkpoint.

    Without dropping the higher-numbered remnants first, the retention window
    would discard the checkpoint that was just written and leave the run with
    no result at all.
    """
    store = CheckpointStore(tmp_path / "model.ckpt", max_keep=2)
    for step in (900, 950, 1000):
        _write(store.path_for(step))
    current = _write(store.path_for(10))

    store.prune(current)

    assert current.exists()
    assert not store.path_for(900).exists()
    assert not store.path_for(950).exists()
    assert not store.path_for(1000).exists()


def test_prune_ignores_foreign_names_and_symlinks(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "model.ckpt", max_keep=1)
    other = _write(tmp_path / "best.ckpt-5.pt")
    unnumbered = _write(tmp_path / "model.ckpt-final.pt")
    current = _write(store.path_for(2))
    store.publish(current)

    store.prune(current)

    assert other.exists()
    assert unnumbered.exists()
    _assert_prefix_alias(tmp_path / "model.ckpt.pt", current, "model.ckpt-2.pt")


def test_prune_without_a_window_keeps_every_checkpoint(tmp_path: Path) -> None:
    """A disabled window deletes nothing, not even higher-numbered remnants."""
    store = CheckpointStore(tmp_path / "model.ckpt", max_keep=0)
    for step in (1, 2, 900):
        _write(store.path_for(step))

    store.prune(store.path_for(2))

    assert all(store.path_for(step).exists() for step in (1, 2, 900))


def test_prune_from_a_foreign_path_spares_the_window(tmp_path: Path) -> None:
    """A checkpoint outside the store neither dates files nor claims a slot.

    The name alone cannot decide membership: a validation checkpoint written
    elsewhere may well parse as ``<prefix>-<step>``.
    """
    store = CheckpointStore(tmp_path / "model.ckpt", max_keep=2)
    for step in (100, 200):
        _write(store.path_for(step))
    elsewhere = _write(tmp_path / "best" / "model.ckpt-150.pt")

    store.prune(elsewhere)

    assert store.path_for(100).exists()
    assert store.path_for(200).exists()
    assert elsewhere.exists()


def test_retention_ratio_maps_to_a_window_count() -> None:
    assert resolve_keep_ckpt_count(None, 1000, 10) is None
    # 1000 / 10 = 100 periodic checkpoints; 40% keeps the most recent 40.
    assert resolve_keep_ckpt_count(0.4, 1000, 10) == 40
    # 4 periodic checkpoints; ceil(0.4 * 4) = ceil(1.6) = 2.
    assert resolve_keep_ckpt_count(0.4, 4, 1) == 2
    # A save frequency above the run length yields a single, final checkpoint.
    assert resolve_keep_ckpt_count(0.4, 5, 100) == 1


def test_retention_ratio_handles_disabled_periodic_saving() -> None:
    """Without periodic saving the run produces one checkpoint, not zero."""
    assert resolve_keep_ckpt_count(0.5, 1000, 0) == 1


def test_built_stores_share_a_directory_and_split_the_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the periodic family owns the pointer file."""
    monkeypatch.chdir(tmp_path)
    store, ema_store = build_checkpoint_stores(
        {
            "save_ckpt": "model.ckpt",
            "save_dir": "ckpts",
            "save_freq": 2,
            "ckpt_keep_ratio": 0.5,
        },
        num_steps=8,
        ema_prefix="model_ema.ckpt",
    )

    # 4 periodic checkpoints; ceil(0.5 * 4) = 2 for both families.
    assert (store.max_keep, ema_store.max_keep) == (2, 2)
    assert store.directory == ema_store.directory == Path("ckpts")
    assert store.directory.is_dir()

    ema_store.publish(_write(ema_store.path_for(1)))
    assert Path("model_ema.ckpt.pt").is_symlink()
    assert not Path("checkpoint").exists()

    store.publish(_write(store.path_for(2)))
    assert Path("checkpoint").read_text() == str(store.path_for(2))


def test_ema_store_inherits_regular_retention_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    store, ema_store = build_checkpoint_stores(
        {
            "save_ckpt": "model.ckpt",
            "max_ckpt_keep": 7,
            "ema_ckpt_keep": None,
        },
        num_steps=10,
        ema_prefix="model_ema.ckpt",
    )

    assert store.max_keep == ema_store.max_keep == 7


def test_ema_store_accepts_an_explicit_retention_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    store, ema_store = build_checkpoint_stores(
        {
            "save_ckpt": "model.ckpt",
            "max_ckpt_keep": 7,
            "ema_ckpt_keep": 2,
        },
        num_steps=10,
        ema_prefix="model_ema.ckpt",
    )

    assert store.max_keep == 7
    assert ema_store.max_keep == 2


def test_nested_and_absolute_prefix_resolution(tmp_path: Path) -> None:
    """Only the prefix file name seeds numbered files under save_dir."""
    nested = CheckpointStore(tmp_path / "run" / "nested" / "model.ckpt")
    assert nested.path_for(1) == tmp_path / "run" / "nested" / "model.ckpt-1.pt"
    assert nested.alias_path == tmp_path / "run" / "nested" / "model.ckpt.pt"

    relocated = CheckpointStore(
        tmp_path / "run" / "nested" / "model.ckpt",
        save_dir=tmp_path / "models",
    )
    assert relocated.path_for(1) == tmp_path / "models" / "model.ckpt-1.pt"
    assert relocated.alias_path == tmp_path / "run" / "nested" / "model.ckpt.pt"

    absolute = CheckpointStore(
        tmp_path / "abs" / "model.ckpt",
        save_dir=tmp_path / "models",
    )
    assert absolute.path_for(2) == tmp_path / "models" / "model.ckpt-2.pt"
    assert absolute.alias_path == tmp_path / "abs" / "model.ckpt.pt"


def test_prune_never_deletes_the_latest_alias_target(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "model.ckpt", max_keep=1)
    older = _write(store.path_for(1))
    newer = _write(store.path_for(2))
    store.publish(older)

    # Retention is asked to keep only the fresh write, but the published latest
    # still names the older file and must survive.
    store.prune(newer)

    assert older.exists()
    assert newer.exists()
    _assert_prefix_alias(tmp_path / "model.ckpt.pt", older, "model.ckpt-1.pt")


def test_prune_before_publish_counts_previous_latest_in_budget(
    tmp_path: Path,
) -> None:
    """PT writes prune before publish; the prior alias target still counts.

    Skipping every protected path without budgeting it would leave
    ``max_keep + 1`` files when the fresh write and the previous latest are
    distinct. Publish does not prune again, so the completed save must already
    honor the configured window.
    """
    store = CheckpointStore(tmp_path / "model.ckpt", max_keep=2)
    for step in (2, 3, 4):
        _write(store.path_for(step))
    store.publish(store.path_for(3))

    store.prune(store.path_for(4))

    assert not store.path_for(2).exists()
    assert store.path_for(3).exists()
    assert store.path_for(4).exists()
    _assert_prefix_alias(
        tmp_path / "model.ckpt.pt", store.path_for(3), "model.ckpt-3.pt"
    )


def test_resolve_latest_prefers_pointer_then_alias(tmp_path: Path) -> None:
    store = CheckpointStore(
        tmp_path / "model.ckpt",
        save_dir=tmp_path / "ckpts",
        pointer_file=tmp_path / "checkpoint",
    )
    store.prepare()
    path = _write(store.path_for(4))
    store.publish(path)

    assert store.resolve_latest() == path

    assert resolve_checkpoint_path("latest", store=store) == path
    assert resolve_checkpoint_path(tmp_path, store=store) == path
    assert resolve_checkpoint_path(tmp_path / "model.ckpt") == path


def test_resolve_directory_ignores_foreign_store(tmp_path: Path) -> None:
    """A directory spec must read THAT directory, even when a store is passed.

    ``store`` is only for the literal ``latest`` handle. Consulting it for a
    directory would return another run's checkpoint (e.g. resolving run B's
    directory while holding store A).
    """
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    store_a = CheckpointStore(
        run_a / "model.ckpt",
        save_dir=run_a / "ckpts",
        pointer_file=run_a / "checkpoint",
    )
    store_b = CheckpointStore(
        run_b / "model.ckpt",
        save_dir=run_b / "ckpts",
        pointer_file=run_b / "checkpoint",
    )
    store_a.prepare()
    store_b.prepare()
    path_a = _write(store_a.path_for(1))
    path_b = _write(store_b.path_for(2))
    store_a.publish(path_a)
    store_b.publish(path_b)

    assert resolve_checkpoint_path("latest", store=store_a) == path_a
    assert resolve_checkpoint_path(run_b, store=store_a) == path_b
    assert resolve_checkpoint_path(run_b) == path_b
    assert resolve_checkpoint_path(run_a, store=store_b) == path_a
