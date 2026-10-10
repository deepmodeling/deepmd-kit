# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the pt_expt Torch checkpoint manager."""

from __future__ import (
    annotations,
)

from pathlib import (
    Path,
)

import pytest
import torch

from deepmd.pt_expt.train.checkpoint import (
    TorchCheckpointManager,
)


def test_save_dir_holds_periodic_files_and_cwd_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    manager = TorchCheckpointManager.from_training_params(
        {
            "save_ckpt": "model.ckpt",
            "save_dir": "models",
            "max_ckpt_keep": 2,
            "save_freq": 1,
        },
        num_steps=4,
        ema_prefix="model.ckpt_ema",
        rank=0,
    )

    for step in (1, 2, 3):
        manager.save_step(step, {"model": {"step": step}, "optimizer": {}})

    assert sorted(path.name for path in Path("models").glob("model.ckpt-*.pt")) == [
        "model.ckpt-2.pt",
        "model.ckpt-3.pt",
    ]
    assert Path("model.ckpt.pt").resolve() == (tmp_path / "models" / "model.ckpt-3.pt")
    assert Path("checkpoint").read_text() == str(Path("models") / "model.ckpt-3.pt")
    assert not (Path("models") / "model.ckpt.pt").exists()


def test_interrupted_write_leaves_previous_latest_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    manager = TorchCheckpointManager.from_training_params(
        {"save_ckpt": "model.ckpt", "save_dir": "models", "max_ckpt_keep": 5},
        num_steps=2,
        ema_prefix="model.ckpt_ema",
    )
    first = manager.save_step(1, {"model": {"step": 1}})
    assert torch.load(first, weights_only=True)["model"]["step"] == 1

    real_save = torch.save

    def boom(obj, f, *args, **kwargs):
        if hasattr(f, "name") and str(f.name).endswith(".tmp"):
            raise RuntimeError("disk full")
        return real_save(obj, f, *args, **kwargs)

    monkeypatch.setattr(torch, "save", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        manager.save_step(2, {"model": {"step": 2}})

    assert first.exists()
    assert Path("model.ckpt.pt").resolve() == first.resolve()
    assert Path("checkpoint").read_text() == str(first)
    assert not (tmp_path / "models" / "model.ckpt-2.pt").exists()
    assert list(Path("models").glob(".model.ckpt-2.pt.*.tmp")) == []


def test_non_chief_rank_performs_no_filesystem_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    manager = TorchCheckpointManager.from_training_params(
        {"save_ckpt": "model.ckpt", "save_dir": "models"},
        num_steps=1,
        ema_prefix="model.ckpt_ema",
        rank=1,
    )
    path = manager.save_step(1, {"model": {}})
    assert path == Path("models") / "model.ckpt-1.pt"
    assert not path.exists()
    assert not Path("model.ckpt.pt").exists()
    assert not Path("checkpoint").exists()


def test_resolve_restart_from_explicit_and_latest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    manager = TorchCheckpointManager.from_training_params(
        {"save_ckpt": "nested/model.ckpt", "save_dir": "models"},
        num_steps=1,
        ema_prefix="nested/model.ckpt_ema",
    )
    written = manager.save_step(7, {"model": {"ok": True}})

    assert manager.resolve_restart(written) == written
    assert manager.resolve_restart("latest") == written
    assert manager.resolve_restart(".") == written
    assert manager.resolve_restart("nested/model.ckpt").resolve() == written.resolve()
    with pytest.raises(FileNotFoundError):
        manager.resolve_restart("typo.ckpt")


def test_validation_save_does_not_move_latest_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    manager = TorchCheckpointManager.from_training_params(
        {"save_ckpt": "model.ckpt", "save_dir": "models"},
        num_steps=1,
        ema_prefix="model.ckpt_ema",
    )
    periodic = manager.save_step(1, {"model": {"kind": "periodic"}})
    best = tmp_path / "best" / "model.ckpt-1.pt"
    manager.save_at(best, {"model": {"kind": "best"}})

    assert torch.load(best, weights_only=True)["model"]["kind"] == "best"
    assert Path("model.ckpt.pt").resolve() == periodic.resolve()
