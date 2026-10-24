# SPDX-License-Identifier: LGPL-3.0-or-later
"""Training-stream continuity through native pt_expt checkpoints."""

from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)
from shutil import (
    copyfile,
    copytree,
)
from unittest.mock import (
    Mock,
    patch,
)

import h5py
import lmdb
import msgpack
import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from deepmd.dpmodel.train import (
    RankContext,
)
from deepmd.dpmodel.train.data_progress import (
    DATA_PROGRESS_CHECKPOINT_KEY,
    collect_training_data_progress,
    preserve_training_data_progress,
)
from deepmd.pt_expt.entrypoints.main import (
    get_trainer,
)
from deepmd.pt_expt.train.ema import (
    EMA_CHECKPOINT_KEY,
)
from deepmd.pt_expt.train.training import (
    Trainer,
)
from deepmd.pt_expt.utils.env import (
    DEVICE,
)
from deepmd.pt_expt.utils.lmdb_dataset import (
    LmdbDataSystem,
)
from deepmd.utils import random as dp_random
from deepmd.utils.argcheck import (
    normalize,
)
from deepmd.utils.compat import (
    update_deepmd_input,
)
from deepmd.utils.data import (
    DataRequirementItem,
)

from ..common.dpmodel.test_lmdb_data import (
    _create_lmdb,
)


def _write_directory(path: Path) -> None:
    path.mkdir()
    np.savetxt(path / "type.raw", [0, 1, 1], fmt="%d")
    (path / "type_map.raw").write_text("O H\n")
    rng = np.random.RandomState(24)
    for index in range(2):
        target = path / f"set.{index:03d}"
        target.mkdir()
        np.save(target / "coord.npy", rng.uniform(0.5, 2.5, (2, 9)))
        np.save(target / "box.npy", np.tile(np.eye(3).reshape(1, 9) * 8, (2, 1)))
        np.save(target / "energy.npy", np.arange(index * 2, index * 2 + 2) * 0.01)
        np.save(target / "force.npy", rng.normal(size=(2, 9)) * 0.01)


def _write_lmdb(path: Path, *, mixed: bool) -> None:
    def encode(array: np.ndarray) -> dict:
        return {
            "nd": None,
            "type": str(array.dtype),
            "kind": "",
            "shape": list(array.shape),
            "data": array.tobytes(),
        }

    nlocs = [3, 4, 3, 4, 3, 4] if mixed else [3] * 4
    rng = np.random.RandomState(24)
    with lmdb.open(str(path), map_size=10 * 1024 * 1024) as env:
        with env.begin(write=True) as txn:
            txn.put(
                b"__metadata__",
                msgpack.packb(
                    {
                        "nframes": len(nlocs),
                        "frame_idx_fmt": "012d",
                        "frame_nlocs": nlocs,
                        "type_map": ["O", "H"],
                        "system_info": {"natoms": [1, 2], "nframes": len(nlocs)},
                    },
                    use_bin_type=True,
                ),
            )
            for index, nloc in enumerate(nlocs):
                frame = {
                    "atom_numbs": [1, nloc - 1],
                    "atom_names": ["O", "H"],
                    "atom_types": encode(np.array([0] + [1] * (nloc - 1))),
                    "orig": encode(np.zeros(3)),
                    "cells": encode(np.eye(3) * 8),
                    "coords": encode(rng.uniform(0.5, 2.5, (nloc, 3))),
                    "energies": encode(np.array(index * 0.01)),
                    "forces": encode(rng.normal(size=(nloc, 3)) * 0.01),
                }
                txn.put(
                    f"{index:012d}".encode(),
                    msgpack.packb(frame, use_bin_type=True),
                )


def _write_hdf5(path: Path) -> None:
    """Store the directory fixture in a single HDF5 system."""
    directory = path.with_suffix("")
    _write_directory(directory)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("system/type.raw", data=[0, 1, 1])
        handle.create_dataset("system/type_map.raw", data=[b"O", b"H"])
        for array in directory.glob("set.*/*.npy"):
            handle.create_dataset(
                "system/" + array.relative_to(directory).as_posix(), data=np.load(array)
            )


def _config(path: str | Path, *, mode: str, multitask: bool = True) -> dict:
    descriptor = {
        "type": "se_e2_a",
        "sel": [4, 4],
        "rcut_smth": 0.5,
        "rcut": 3.0,
        "neuron": [4, 8],
        "axis_neuron": 2,
        "seed": 1,
    }
    if mode == "ragged":
        descriptor.update(type="dpa1", sel=8, attn_layer=0)
    model = {
        "type_map": ["O", "H"],
        "descriptor": descriptor,
        "fitting_net": {"neuron": [4], "seed": 1},
        "data_stat_nbatch": 1,
    }
    loss = {
        "type": "ener",
        "start_pref_e": 1,
        "limit_pref_e": 1,
        "start_pref_f": 1,
        "limit_pref_f": 1,
        "start_pref_v": 0,
        "limit_pref_v": 0,
    }
    dataset = {
        "systems": [str(path)] if mode == "directory" else str(path),
        "batch_size": "mix:7" if mode == "ragged" else 1,
    }
    task_data = {
        "training_data": deepcopy(dataset),
        "validation_data": {**dataset, "numb_btch": 2},
    }
    training = {
        "numb_steps": 20,
        "seed": 19,
        "disp_freq": 3,
        "save_freq": 10,
    }
    config = {
        "learning_rate": {
            "type": "exp",
            "start_lr": 1e-3,
            "stop_lr": 1e-5,
            "decay_steps": 5,
        },
        "training": training,
    }
    if multitask:
        keys = ["first", "second"]
        config["model"] = {"model_dict": {key: deepcopy(model) for key in keys}}
        config["loss_dict"] = {key: deepcopy(loss) for key in keys}
        training["data_dict"] = {key: deepcopy(task_data) for key in keys}
        training["model_prob"] = {"first": 0.5, "second": 0.5}
    else:
        config.update(model=model, loss=loss)
        training.update(task_data)
    return normalize(update_deepmd_input(config, warning=False), multi_task=multitask)


def _record_training(trainer: Trainer, *, data_rng_draws: int = 0) -> list:
    """Record real optimizer inputs, excluding diagnostic reads of the data."""
    records = []
    get_data = trainer.get_data
    train_step = trainer.train_step
    in_train_step = False

    def record_data(is_train: bool = True, task_key: str = "Default") -> tuple:
        inputs, labels = get_data(is_train=is_train, task_key=task_key)
        if is_train and in_train_step:
            if data_rng_draws:
                dp_random.random(data_rng_draws)
            records.append(
                (
                    task_key,
                    {
                        key: value.detach().cpu().clone()
                        for key, value in inputs.items()
                        if torch.is_tensor(value)
                    },
                    labels["energy"].detach().cpu().clone(),
                )
            )
        return inputs, labels

    def record_step(task, step):
        nonlocal in_train_step
        in_train_step = True
        try:
            return train_step(task, step)
        finally:
            in_train_step = False

    trainer.get_data = record_data
    trainer.train_step = record_step
    trainer.run()
    return records


def _assert_records_equal(expected: list, actual: list) -> None:
    assert len(expected) == len(actual)
    for (task, inputs, energy), (other_task, other_inputs, other_energy) in zip(
        expected, actual, strict=True
    ):
        assert task == other_task
        assert inputs.keys() == other_inputs.keys()
        for key, value in inputs.items():
            torch.testing.assert_close(value, other_inputs[key], rtol=0, atol=0)
        torch.testing.assert_close(energy, other_energy, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["directory", "lmdb", "ragged"])
def test_restart_matches_uninterrupted_training(tmp_path, monkeypatch, mode) -> None:
    """Real training matches across passes, display, validation and setup."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DP_LMDB_NUM_WORKERS", "0")
    path = tmp_path / ("data" if mode == "directory" else "data.lmdb")
    if mode == "directory":
        _write_directory(path)
    else:
        _write_lmdb(path, mixed=mode == "ragged")
    config = _config(path, mode=mode)
    dp_random.seed(19)
    uninterrupted = get_trainer(deepcopy(config))
    if mode == "ragged":
        assert all(
            source._reader.ragged_batches
            for source in uninterrupted.training_data_by_task.values()
        )
    expected = _record_training(uninterrupted)
    checkpoint = tmp_path / "model.ckpt-10.pt"
    state = torch.load(checkpoint, weights_only=True)
    assert DATA_PROGRESS_CHECKPOINT_KEY in state
    # Restart construction and fresh validation readers may consume different
    # random draws; none may leak into the resumed training stream.
    dp_random.seed(9876)
    restarted = get_trainer(deepcopy(config), restart_model=str(checkpoint))
    assert restarted.start_step == 10
    actual = _record_training(restarted)
    _assert_records_equal(expected[10:], actual)
    for key, value in uninterrupted._unwrapped.state_dict().items():
        if torch.is_tensor(value):
            torch.testing.assert_close(
                value, restarted._unwrapped.state_dict()[key], rtol=1e-12, atol=1e-12
            )


def test_best_validation_checkpoint_resumes_training(tmp_path, monkeypatch) -> None:
    """The real full-validation callback writes a restartable snapshot."""
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "data"
    _write_directory(path)
    config = _config(path, mode="directory", multitask=False)
    config["training"].update(numb_steps=6, save_freq=0)
    config["validating"] = {
        "full_validation": True,
        "full_val_start": 3,
        "validation_freq": 3,
        "save_best": True,
        "save_best_dir": str(tmp_path / "best"),
        "max_best_ckpt": 2,
    }
    original = get_trainer(deepcopy(config))
    expected = _record_training(original)
    checkpoint = next((tmp_path / "best").glob("*-3.t-*.pt"))
    restarted = get_trainer(deepcopy(config), restart_model=str(checkpoint))
    assert restarted.start_step == 3
    _assert_records_equal(expected[3:], _record_training(restarted))


@pytest.mark.parametrize("reset", [False, True])
def test_legacy_init_finetune_and_deployment(
    tmp_path, monkeypatch, caplog, reset
) -> None:
    """Only a restart restores progress; deployment snapshots omit it."""
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "data"
    _write_directory(path)
    config = _config(path, mode="directory", multitask=False)
    config["training"].update(
        numb_steps=3, save_freq=3, enable_ema=True, reset_data_progress=reset
    )
    trainer = get_trainer(deepcopy(config))
    assert trainer.start_step == 0
    trainer.run()
    checkpoint = tmp_path / "model.ckpt-3.pt"
    state = torch.load(checkpoint, weights_only=True)
    assert DATA_PROGRESS_CHECKPOINT_KEY in state
    ema_state = torch.load(tmp_path / "model_ema.ckpt-3.pt", weights_only=True)
    assert DATA_PROGRESS_CHECKPOINT_KEY not in ema_state
    assert "optimizer" not in ema_state
    legacy = tmp_path / "legacy.pt"
    state.pop(DATA_PROGRESS_CHECKPOINT_KEY)
    torch.save(state, legacy)
    restarted = get_trainer(deepcopy(config), restart_model=str(legacy))
    assert restarted.start_step == 3
    restarted._close_data_systems()
    # A deliberately unsupported version proves these paths do not restore it.
    state[DATA_PROGRESS_CHECKPOINT_KEY] = {"version": -1}
    torch.save(state, checkpoint)
    for option in ("init_model", "finetune_model"):
        fresh = get_trainer(deepcopy(config), **{option: str(checkpoint)})
        assert fresh.start_step == 0
        fresh._close_data_systems()
    assert "training.reset_data_progress is true" not in caplog.text


@pytest.mark.parametrize("mode", ["directory", "hdf5", "lmdb"])
def test_relocated_restart_can_reset_only_data_progress(
    tmp_path, monkeypatch, caplog, mode
) -> None:
    """Opting out after a move has exactly the legacy restart semantics."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DP_LMDB_NUM_WORKERS", "0")
    if mode == "lmdb":
        path = tmp_path / "data.lmdb"
        _write_lmdb(path, mixed=False)
    elif mode == "hdf5":
        path = tmp_path / "data.h5"
        _write_hdf5(path)
    else:
        path = tmp_path / "data"
        _write_directory(path)
    system = str(path) + ("#/system" if mode == "hdf5" else "")
    config_mode = "lmdb" if mode == "lmdb" else "directory"
    config = _config(system, mode=config_mode)
    config["training"].update(numb_steps=3, save_freq=3, enable_ema=True)
    original = get_trainer(deepcopy(config))
    original.run()
    checkpoint = tmp_path / "model.ckpt-3.pt"
    state = torch.load(checkpoint, weights_only=True)
    # Native restart maps every optimizer tensor, including Adam's step.
    checkpoint_optimizer = torch.load(
        checkpoint, map_location=DEVICE, weights_only=True
    )["optimizer"]["state"]
    assert DATA_PROGRESS_CHECKPOINT_KEY in state
    legacy_state = deepcopy(state)
    del legacy_state[DATA_PROGRESS_CHECKPOINT_KEY]
    legacy_path = tmp_path / "legacy.pt"
    torch.save(legacy_state, legacy_path)

    relocated = path.with_name("relocated" + path.suffix)
    if mode == "hdf5":
        copyfile(path, relocated)
    else:
        copytree(path, relocated)
    system = str(relocated) + ("#/system" if mode == "hdf5" else "")
    config = _config(system, mode=config_mode)
    config["training"].update(numb_steps=9, save_freq=3, enable_ema=True)
    assert config["training"]["reset_data_progress"] is False
    with pytest.raises(ValueError, match=r"[Pp]rogress"):
        get_trainer(deepcopy(config), restart_model=str(checkpoint))
    assert "training.reset_data_progress is true" not in caplog.text

    dp_random.seed(9876)
    legacy = get_trainer(deepcopy(config), restart_model=str(legacy_path))
    expected_lr = legacy.scheduler.get_last_lr()
    expected_optimizer = deepcopy(legacy.optimizer.state_dict())
    expected_progress = collect_training_data_progress(legacy.training_data_by_task)
    expected = _record_training(legacy)
    dp_random.seed(9876)
    config["training"]["reset_data_progress"] = True
    reset = get_trainer(deepcopy(config), restart_model=str(checkpoint))
    assert "training.reset_data_progress is true" in caplog.text
    assert "Batches may repeat" in caplog.text
    assert "exact data-order continuation is unavailable" in caplog.text
    assert reset.start_step == legacy.start_step == 3
    assert reset.scheduler.get_last_lr() == expected_lr
    assert (
        collect_training_data_progress(reset.training_data_by_task) == expected_progress
    )
    torch.testing.assert_close(reset.optimizer.state_dict(), expected_optimizer)
    torch.testing.assert_close(
        reset.optimizer.state_dict()["state"], checkpoint_optimizer
    )
    torch.testing.assert_close(reset.model_ema.state_dict(), state[EMA_CHECKPOINT_KEY])
    for key, value in state["model"].items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, reset._unwrapped.state_dict()[key])
    _assert_records_equal(expected, _record_training(reset))
    assert reset.scheduler.get_last_lr() == legacy.scheduler.get_last_lr()
    torch.testing.assert_close(
        reset.optimizer.state_dict(), legacy.optimizer.state_dict()
    )
    torch.testing.assert_close(
        reset.model_ema.state_dict(), legacy.model_ema.state_dict()
    )
    for key, value in legacy._unwrapped.state_dict().items():
        if torch.is_tensor(value):
            torch.testing.assert_close(
                value, reset._unwrapped.state_dict()[key], rtol=1e-12, atol=1e-12
            )


@pytest.mark.parametrize("mode", ["directory", "hdf5", "lmdb"])
def test_configured_seed_is_format_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Fresh streams are seeded consistently; exact restarts reject seed changes."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DP_LMDB_NUM_WORKERS", "0")
    path = (
        tmp_path / {"directory": "data", "hdf5": "data.h5", "lmdb": "data.lmdb"}[mode]
    )
    if mode == "lmdb":
        _write_lmdb(path, mixed=False)
    elif mode == "hdf5":
        _write_hdf5(path)
    else:
        _write_directory(path)
    system = str(path) + ("#/system" if mode == "hdf5" else "")
    config = _config(
        system, mode="lmdb" if mode == "lmdb" else "directory", multitask=False
    )
    config["training"].update(seed=7, numb_steps=1, save_freq=1)
    states = []
    for external_seed in (1, 999):
        dp_random.seed(external_seed)
        trainer = get_trainer(deepcopy(config))
        try:
            states.append(collect_training_data_progress(trainer.training_data_by_task))
        finally:
            trainer._close_data_systems()
    assert states[0] == states[1]

    original = get_trainer(deepcopy(config))
    original.run()
    checkpoint = str(tmp_path / "model.ckpt-1.pt")
    config["training"]["seed"] = 12
    with pytest.raises(ValueError, match=r"training.seed changed"):
        get_trainer(deepcopy(config), restart_model=checkpoint)
    config["training"]["reset_data_progress"] = True
    restarted = get_trainer(config, restart_model=checkpoint)
    try:
        assert restarted.start_step == 1
        assert restarted._collect_data_progress()["configured_seed"] == 12
    finally:
        restarted._close_data_systems()


@pytest.mark.parametrize("mode", ["directory", "lmdb"])
def test_changed_world_size_retains_legacy_optimizer_restart(
    tmp_path, monkeypatch, caplog, mode
) -> None:
    """A changed topology keeps native restart state, but not saved data order."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DP_LMDB_NUM_WORKERS", "0")
    path = tmp_path / ("data" if mode == "directory" else "data.lmdb")
    if mode == "directory":
        _write_directory(path)
    else:
        _write_lmdb(path, mixed=False)
    config = _config(path, mode=mode, multitask=False)
    config["training"].update(numb_steps=3, save_freq=3)
    original = get_trainer(deepcopy(config))
    original.run()
    checkpoint = tmp_path / "model.ckpt-3.pt"
    state = torch.load(checkpoint, weights_only=True)
    checkpoint_optimizer = torch.load(
        checkpoint, map_location=DEVICE, weights_only=True
    )["optimizer"]["state"]
    progress = state.pop(DATA_PROGRESS_CHECKPOINT_KEY)
    legacy_path = tmp_path / "legacy.pt"
    torch.save(state, legacy_path)
    # Exercise the topology fallback at the native checkpoint-load boundary.
    # The real 1-to-2-rank integration case remains in TestDDPRestart.
    progress.update(world_size=2, ranks=progress["ranks"] * 2)
    state[DATA_PROGRESS_CHECKPOINT_KEY] = progress
    changed_path = tmp_path / "changed-world.pt"
    torch.save(state, changed_path)
    config["training"]["numb_steps"] = 6

    dp_random.seed(9876)
    legacy = get_trainer(deepcopy(config), restart_model=str(legacy_path))
    expected_lr = legacy.scheduler.get_last_lr()
    expected = _record_training(legacy)
    dp_random.seed(9876)
    changed = get_trainer(deepcopy(config), restart_model=str(changed_path))
    assert "world size changed from 2 to 1" in caplog.text
    assert "exact data-order continuation is unavailable" in caplog.text
    assert changed.start_step == 3
    assert changed.scheduler.get_last_lr() == expected_lr
    torch.testing.assert_close(
        changed.optimizer.state_dict()["state"], checkpoint_optimizer
    )
    for key, value in state["model"].items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, changed._unwrapped.state_dict()[key])
    _assert_records_equal(expected, _record_training(changed))
    for key, value in legacy._unwrapped.state_dict().items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, changed._unwrapped.state_dict()[key])


def test_chief_display_preserves_training_progress(tmp_path, monkeypatch) -> None:
    """Chief-only inactive-task and validation reads cannot move rank cursors."""
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "data"
    _write_directory(path)
    trainer = get_trainer(_config(path, mode="directory"))
    active = trainer.training_tasks["first"]
    result = trainer.train_step(active, 0)
    before = collect_training_data_progress(trainer.training_data_by_task)
    trainer.collect_display_results(
        trainer.training_tasks, active_task=active, step=0, step_result=result
    )
    assert collect_training_data_progress(trainer.training_data_by_task) == before
    trainer._close_data_systems()


def test_ddp_warmup_preserves_training_progress(tmp_path, monkeypatch) -> None:
    """Exercise the warmup body on CPU without invoking a compiler or NCCL."""
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "data"
    _write_directory(path)
    trainer = get_trainer(_config(path, mode="directory"))
    before = collect_training_data_progress(trainer.training_data_by_task)

    class DDPWrapper:
        def __init__(self, module):
            self.module = module

    trainer.wrapper = DDPWrapper(trainer.wrapper)
    trainer.enable_compile = True
    with (
        patch.object(torch.nn.parallel, "DistributedDataParallel", DDPWrapper),
        patch.object(dist, "is_initialized", return_value=True),
        patch.object(dist, "get_world_size", return_value=2),
        patch.object(dist.distributed_c10d, "_get_default_store") as store,
    ):
        store.return_value.add.return_value = 2
        trainer._precompile_outside_collectives()
    assert collect_training_data_progress(trainer.training_data_by_task) == before
    trainer._close_data_systems()


def test_full_validation_checkpoint_uses_training_rng() -> None:
    """A best-checkpoint callback runs before validation returns to the loop."""
    trainer = Trainer.__new__(Trainer)
    dp_random.seed(42)
    before = dp_random.get_state()
    saved = []

    def save(*args, **kwargs):
        saved.append(dp_random.get_state())

    def validate(*, save_checkpoint, **kwargs):
        dp_random.random(100)
        save_checkpoint(Path("best.pt"), lr=1e-3, step=1)
        dp_random.random(100)

    trainer._collect_data_progress = Mock(return_value={"saved_rng": before})
    trainer.full_validator = Mock(run=validate, save_best=True)
    trainer.ema_full_validator = None
    trainer._save_full_validation_checkpoint = save
    trainer.run_full_validation(step=0, display_step=1, learning_rate=1e-3)
    assert saved == [before]
    assert dp_random.get_state() == before


def _set_mock_rank(trainer: Trainer, rank: int) -> None:
    trainer.is_distributed = True
    trainer.rank = rank
    trainer.world_size = 2
    trainer.rank_context = RankContext(rank=rank, world_size=2)


@pytest.mark.parametrize("rank", [0, 1])
def test_every_rank_joins_periodic_final_and_bias_saves(
    tmp_path, monkeypatch, rank
) -> None:
    """Run the native loop; only checkpoint boundaries gather rank cursors."""
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "data"
    _write_directory(path)
    config = _config(path, mode="directory", multitask=False)
    config["training"].update(
        numb_steps=3,
        save_freq=2,
        disp_training=False,
        change_bias_after_training=True,
        enable_ema=True,
    )
    trainer = get_trainer(config)
    _set_mock_rank(trainer, rank)
    trainer._change_bias_after_training = Mock()
    gathered = []
    collectives = []

    def gather(ranks, local):
        if isinstance(local, bool):
            collectives.append("preflight")
        else:
            collectives.append("progress")
            gathered.append(deepcopy(local))
        ranks[:] = [deepcopy(local), deepcopy(local)]

    monkeypatch.setattr(dist, "all_gather_object", gather)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    steps = []
    save = trainer.save_checkpoint

    def record_save(step):
        steps.append(step)
        save(step)

    trainer.save_checkpoint = record_save
    trainer.run()
    assert steps == [2, 3, 3]
    assert len(gathered) == 3  # No per-batch or EMA-snapshot gather.
    assert collectives == ["preflight", "progress"] * 3
    trainer._change_bias_after_training.assert_called_once()
    checkpoint = tmp_path / "model.ckpt-3.pt"
    assert checkpoint.exists() == (rank == 0)
    if rank == 0:
        progress = torch.load(checkpoint, weights_only=True)[
            DATA_PROGRESS_CHECKPOINT_KEY
        ]
        assert progress == {
            "version": 1,
            "world_size": 2,
            "configured_seed": config["training"]["seed"],
            "task_rng": trainer._task_rng.bit_generator.state,
            "ranks": [gathered[-1]] * 2,
        }


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("due,save_best", [(True, True), (True, False), (False, True)])
def test_full_validation_precollects_before_chief_only_save(
    tmp_path, monkeypatch, rank, due, save_best
) -> None:
    """The chief callback writes a complete envelope without a nested gather."""
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "data"
    _write_directory(path)
    config = _config(path, mode="directory", multitask=False)
    config["training"]["enable_ema"] = True
    trainer = get_trainer(config)
    _set_mock_rank(trainer, rank)
    before = collect_training_data_progress(trainer.training_data_by_task)
    gathered = []
    calls = []

    def gather(ranks, local):
        assert not calls
        gathered.append(deepcopy(local))
        ranks[:] = [deepcopy(local), deepcopy(local)]

    monkeypatch.setattr(dist, "all_gather_object", gather)

    def validate(*, save_checkpoint, **kwargs):
        calls.append("live")
        assert len(gathered) == int(save_best)
        dp_random.random(30)
        if rank == 0 and save_best:
            save_checkpoint(tmp_path / "best.pt", step=1)

    def validate_ema(*, save_checkpoint, **kwargs):
        calls.append("ema")
        if rank == 0 and save_best:
            save_checkpoint(tmp_path / "best_ema.pt", step=1)

    trainer.full_validator = Mock(
        should_run=Mock(return_value=due), save_best=save_best, run=validate
    )
    trainer.ema_full_validator = Mock(
        should_run=Mock(return_value=due), save_best=save_best, run=validate_ema
    )
    trainer.run_full_validation(step=0, display_step=1, learning_rate=1e-3)
    assert len(gathered) == int(due and save_best)
    assert calls == (["live", "ema"] if due else [])
    assert dp_random.get_state() == before["rng"]
    checkpoint = tmp_path / "best.pt"
    assert checkpoint.exists() == (rank == 0 and due and save_best)
    if checkpoint.exists():
        state = torch.load(checkpoint, weights_only=True)
        assert state[DATA_PROGRESS_CHECKPOINT_KEY] == {
            "version": 1,
            "world_size": 2,
            "configured_seed": config["training"]["seed"],
            "task_rng": trainer._task_rng.bit_generator.state,
            "ranks": [before, before],
        }
        ema = torch.load(tmp_path / "best_ema.pt", weights_only=True)
        assert DATA_PROGRESS_CHECKPOINT_KEY not in ema
    trainer._close_data_systems()


def test_filtered_rank_progress_restores_its_own_cursor(tmp_path, monkeypatch) -> None:
    """Local filter retries make rank zero's cursor wrong for another rank."""
    path = _create_lmdb(str(tmp_path / "filtered.lmdb"), nframes=12, natoms=2)

    def make_rank(rank):
        data = LmdbDataSystem(
            path,
            ["O", "H"],
            batch_size=1,
            seed=7,
            num_workers=0,
            rank=rank,
            world_size=2,
        )
        data.add_data_requirements(
            [
                DataRequirementItem(
                    "pair_margin", 1, default=2.0, source_policy="derived"
                )
            ]
        )
        trainer = Trainer.__new__(Trainer)
        _set_mock_rank(trainer, rank)
        trainer._configured_data_seed = 7
        trainer._task_rng = np.random.default_rng(7)
        trainer.training_data_by_task = {"Default": data}
        trainer.min_pair_dist_by_task = {"Default": 2.0}
        return trainer, data

    originals = [make_rank(rank) for rank in range(2)]
    for trainer, data in originals:
        trainer._next_training_batch(data, "Default")
    progress = [
        collect_training_data_progress(trainer.training_data_by_task)
        for trainer, _ in originals
    ]
    assert progress[0]["tasks"] != progress[1]["tasks"]

    def gather(ranks, local):
        ranks[:] = deepcopy(progress)

    monkeypatch.setattr(dist, "all_gather_object", gather)
    checkpoint = tmp_path / "progress.pt"
    torch.save(originals[0][0]._collect_data_progress(), checkpoint)
    envelope = torch.load(checkpoint, weights_only=True)
    for rank, (trainer, data) in enumerate(originals):
        expected = [
            trainer._next_training_batch(data, "Default")["coord"] for _ in range(4)
        ]
        restored, restored_data = make_rank(rank)
        restored._restore_data_progress(envelope)
        actual = [
            restored._next_training_batch(restored_data, "Default")["coord"]
            for _ in range(4)
        ]
        for want, got in zip(expected, actual, strict=True):
            np.testing.assert_array_equal(want, got)
        data.close()
        restored_data.close()


@pytest.mark.parametrize(
    "saved_size,current_size,rank", [(1, 2, 0), (1, 2, 1), (2, 1, 0), (2, 3, 2)]
)
def test_changed_world_size_leaves_data_and_rng_untouched(
    caplog, saved_size, current_size, rank
) -> None:
    """Topology fallback neither remaps a cursor nor restores another rank's RNG."""
    trainer = Trainer.__new__(Trainer)
    trainer.rank = rank
    trainer.world_size = current_size
    trainer._task_rng = np.random.default_rng(7)
    task_rng_state = deepcopy(trainer._task_rng.bit_generator.state)
    source = Mock()
    source.state_dict.return_value = {"cursor": 11}
    trainer.training_data_by_task = {"Default": source}
    local = collect_training_data_progress(trainer.training_data_by_task)
    local["tasks"]["Default"] = {"cursor": 99}
    progress = {"version": 1, "world_size": saved_size, "ranks": [local] * saved_size}
    dp_random.random(5)
    before = collect_training_data_progress(trainer.training_data_by_task)
    with patch(
        "deepmd.pt_expt.train.training.restore_training_data_progress"
    ) as restore:
        trainer._restore_data_progress(progress)
    restore.assert_not_called()
    source.load_state_dict.assert_not_called()
    assert collect_training_data_progress(trainer.training_data_by_task) == before
    assert trainer._task_rng.bit_generator.state == task_rng_state
    assert ("Checkpoint world size changed" in caplog.text) == (rank == 0)


@pytest.mark.parametrize("rank", [0, 1])
def test_explicit_reset_leaves_every_task_and_rng_untouched(caplog, rank) -> None:
    """An explicit reset does not even invoke a task or shared RNG restore."""
    trainer = Trainer.__new__(Trainer)
    _set_mock_rank(trainer, rank)
    trainer._task_rng = np.random.default_rng(7)
    task_rng_state = deepcopy(trainer._task_rng.bit_generator.state)
    sources = {"first": Mock(), "second": Mock()}
    for index, source in enumerate(sources.values()):
        source.state_dict.return_value = {"cursor": index + 10}
    trainer.training_data_by_task = sources
    dp_random.seed(42)
    local = collect_training_data_progress(sources)
    for task in local["tasks"]:
        local["tasks"][task] = {"cursor": 99}
    progress = {"version": 1, "world_size": 2, "ranks": [local, local]}
    dp_random.random(5)
    before = collect_training_data_progress(sources)
    with (
        patch(
            "deepmd.pt_expt.train.training.restore_training_data_progress"
        ) as restore,
        patch.object(dp_random, "set_state") as set_rng,
    ):
        trainer._restore_data_progress(progress, reset=True)
    restore.assert_not_called()
    set_rng.assert_not_called()
    for source in sources.values():
        source.load_state_dict.assert_not_called()
    assert collect_training_data_progress(sources) == before
    assert trainer._task_rng.bit_generator.state == task_rng_state
    assert ("training.reset_data_progress is true" in caplog.text) == (rank == 0)
    assert "Checkpoint world size changed" not in caplog.text


@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize(
    "changed",
    [
        {"version": 2},
        {"world_size": 3},
        {"world_size": 0},
        {"world_size": -1},
        {"world_size": True},
        {"world_size": 2.0},
        {"world_size": "2"},
        {"ranks": []},
        {"ranks": {}},
        {"ranks": [None, None]},
    ],
)
def test_malformed_rank_envelope_is_rejected(
    changed, world_size, caplog, reset
) -> None:
    trainer = Trainer.__new__(Trainer)
    trainer.rank = 0
    trainer.world_size = world_size
    trainer.training_data_by_task = {}
    local = collect_training_data_progress({})
    progress = {"version": 1, "world_size": 2, "ranks": [local, local]}
    with pytest.raises(ValueError, match=r"[Dd]ata-progress"):
        trainer._restore_data_progress(progress | changed, reset=reset)
    assert "Checkpoint world size changed" not in caplog.text
    assert "training.reset_data_progress is true" not in caplog.text
    trainer._restore_data_progress(None, reset=reset)
    assert "training.reset_data_progress is true" not in caplog.text


def _ddp_restart_worker(rank: int, directory: str, rendezvous: str, mode: str) -> None:
    torch.set_num_threads(1)
    root = Path(directory)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        path = root / ("data" if mode == "directory" else "data.lmdb")
        config = _config(path, mode=mode)
        config["training"].update(
            save_ckpt=str(root / "model.ckpt"),
            disp_file=str(root / "lcurve.out"),
        )
        if mode == "directory":
            unseeded_config = deepcopy(config)
            unseeded_config["training"]["seed"] = None
            unseeded = get_trainer(unseeded_config)
            choices = [
                unseeded.select_task(unseeded.training_tasks).key for _ in range(8)
            ]
            ranks = [None, None]
            dist.all_gather_object(ranks, choices)
            assert ranks[0] == ranks[1]
            unseeded._close_data_systems()
        dp_random.seed(19)
        trainer = get_trainer(deepcopy(config))
        with preserve_training_data_progress(trainer.training_data_by_task):
            first = next(iter(trainer.training_data_by_task.values()))
            frames = [np.asarray(first.get_batch()["coord"]).tolist() for _ in range(8)]
        ranks = [None, None]
        dist.all_gather_object(ranks, frames)
        assert ranks[0] != ranks[1]
        expected = _record_training(trainer, data_rng_draws=rank + 1)
        dist.all_gather_object(ranks, [record[0] for record in expected])
        assert ranks[0] == ranks[1]
        dist.barrier()
        dp_random.seed(9876)
        restarted = get_trainer(
            deepcopy(config), restart_model=str(root / "model.ckpt-10.pt")
        )
        actual = _record_training(restarted, data_rng_draws=rank + 1)
        _assert_records_equal(expected[10:], actual)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Requires Gloo")
@pytest.mark.parametrize("mode", ["directory", "lmdb"])
def test_two_rank_restart_preserves_each_rank(tmp_path, monkeypatch, mode) -> None:
    """Chief-only statistics/display do not desynchronize saved rank cursors."""
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("DP_LMDB_NUM_WORKERS", "0")
    if mode == "directory":
        _write_directory(tmp_path / "data")
    else:
        _write_lmdb(tmp_path / "data.lmdb", mixed=False)
    mp.spawn(
        _ddp_restart_worker,
        args=(str(tmp_path), (tmp_path / "rendezvous").as_uri(), mode),
        nprocs=2,
        join=True,
    )
