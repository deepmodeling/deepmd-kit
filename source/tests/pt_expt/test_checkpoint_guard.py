# SPDX-License-Identifier: LGPL-3.0-or-later
"""Checkpoint preflights exercise real trainer writers without a training dataset."""

from copy import (
    deepcopy,
)
from datetime import (
    timedelta,
)
from functools import (
    partial,
)
from pathlib import (
    Path,
)
from types import (
    SimpleNamespace,
)
from unittest.mock import (
    patch,
)

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.optim import (
    ZeroRedundancyOptimizer,
)
from torch.nn.parallel import (
    DistributedDataParallel,
)

from deepmd.dpmodel.train import (
    ShardingPolicy,
)
from deepmd.dpmodel.train.checkpoint import (
    CheckpointStore,
)
from deepmd.pt.train.training import Trainer as PTTrainer
from deepmd.pt.train.wrapper import ModelWrapper as PTModelWrapper
from deepmd.pt_expt.train.ema import (
    EMA_CHECKPOINT_KEY,
    EMA_VALIDATION_STATE_KEY,
    ModelEMA,
)
from deepmd.pt_expt.train.gradient import (
    NonFiniteGradGuard,
)
from deepmd.pt_expt.train.training import Trainer as PTExptTrainer
from deepmd.pt_expt.train.validation import (
    TOPK_RECORDS_INFO_KEY,
    build_full_validators,
)
from deepmd.pt_expt.train.wrapper import ModelWrapper as PTExptModelWrapper


def _make_trainer(backend, directory, *, rank=0, stage=0, distributed=False):
    """Initialize only checkpoint dependencies, retaining production methods."""
    trainer_class, wrapper_class = (
        (PTTrainer, PTModelWrapper)
        if backend == "pt"
        else (PTExptTrainer, PTExptModelWrapper)
    )
    trainer = trainer_class.__new__(trainer_class)
    trainer.rank = rank
    trainer.is_distributed = distributed
    if backend == "pt_expt":
        trainer.training_data_by_task = {}
        trainer._configured_data_seed = None
        trainer._task_rng = np.random.default_rng(0)
        trainer.world_size = dist.get_world_size() if distributed else 1
    trainer.sharding = ShardingPolicy(stage=stage)
    trainer.model = torch.nn.Linear(2, 1, device="cpu")
    with torch.no_grad():
        trainer.model.weight.fill_(2.0)
        trainer.model.bias.fill_(2.0)
    trainer.model_ema = ModelEMA(trainer.model, decay=0.9)
    with torch.no_grad():
        trainer.model.weight.fill_(3.0)
        trainer.model.bias.fill_(3.0)
    wrapper = wrapper_class(trainer.model)
    trainer.wrapper = DistributedDataParallel(wrapper) if distributed else wrapper
    trainer.optimizer = (
        ZeroRedundancyOptimizer(
            trainer.wrapper.parameters(), optimizer_class=torch.optim.Adam, lr=1e-3
        )
        if stage == 1
        else torch.optim.Adam(trainer.wrapper.parameters(), lr=1e-3)
    )
    trainer.nonfinite_grad_guard = NonFiniteGradGuard()
    trainer.ckpt_store = CheckpointStore(directory / "model.ckpt")
    trainer.ema_ckpt_store = CheckpointStore(directory / "model_ema.ckpt")
    directory.mkdir(parents=True, exist_ok=True)
    return trainer


def _make_validators(trainer, directory, *, save_best=True, restart=False):
    wrapper = trainer.wrapper.module if trainer.is_distributed else trainer.wrapper
    return build_full_validators(
        validating_params={
            "full_validation": True,
            "ema_full_validation": True,
            "validation_freq": 1,
            "full_val_start": 0.0,
            "save_best": save_best,
            "max_best_ckpt": 1,
            "validation_metric": "E:MAE",
            "full_val_file": str(directory / "val.log"),
        },
        validation_data=SimpleNamespace(systems=[]),
        model=trainer.model,
        state_store=wrapper.train_infos,
        num_steps=10,
        rank=trainer.rank,
        restart_training=restart,
        checkpoint_dir=directory,
        ensure_supported=lambda: None,
        model_ema=trainer.model_ema,
        sharding=trainer.sharding,
    )


def _best_writer(trainer, ema):
    if isinstance(trainer, PTTrainer):
        return trainer.save_ema_model if ema else trainer.save_model
    return (
        trainer._save_full_validation_ema_checkpoint
        if ema
        else trainer._save_full_validation_checkpoint
    )


def _validate(trainer, validator, *, step, metric, ema=False):
    save_checkpoint = _best_writer(trainer, ema)
    if isinstance(trainer, PTExptTrainer) and not ema:
        save_checkpoint = partial(
            save_checkpoint, data_progress=trainer._collect_data_progress()
        )
    with patch.object(
        validator, "evaluate_all_systems", return_value={validator.metric_key: metric}
    ):
        return validator.run(
            step_id=step,
            display_step=step,
            lr=1e-3,
            save_checkpoint=save_checkpoint,
            prepare_checkpoint=trainer.ensure_finite_gradients_for_checkpoint,
        )


def _record_recovered_gradients(trainer):
    """A later finite gradient must not hide the earlier failed training step."""
    trainer.nonfinite_grad_guard.update(torch.tensor(float("nan"), device="cpu"))
    for parameter in trainer.wrapper.parameters():
        parameter.grad = torch.ones_like(parameter)
    trainer.nonfinite_grad_guard.update(torch.tensor(1.0, device="cpu"))


@pytest.mark.parametrize("backend", ["pt", "pt_expt"])
@pytest.mark.parametrize("ema", [False, True])
def test_best_checkpoint_rejects_prior_nan_without_publication(backend, ema, tmp_path):
    trainer = _make_trainer(backend, tmp_path)
    validator = _make_validators(trainer, tmp_path)[int(ema)]
    previous = _validate(trainer, validator, step=1, metric=2.0, ema=ema)
    old_path = Path(previous.saved_best_path)
    old_bytes = old_path.read_bytes()
    old_records = list(validator.topk_records)
    old_state = deepcopy(validator.state_store)
    _record_recovered_gradients(trainer)

    with (
        patch.object(
            trainer,
            "_collect_checkpoint_states",
            wraps=trainer._collect_checkpoint_states,
        ) as collect,
        patch("torch.save", wraps=torch.save) as serialize,
        pytest.raises(RuntimeError, match="Non-finite gradient norm"),
    ):
        _validate(trainer, validator, step=2, metric=1.0, ema=ema)

    collect.assert_not_called()
    serialize.assert_not_called()
    assert validator.topk_records == old_records
    assert validator.state_store == old_state
    assert old_path.read_bytes() == old_bytes
    assert list(tmp_path.glob(f"{validator.best_checkpoint_prefix}-*.pt")) == [old_path]


@pytest.mark.parametrize("ema", [False, True])
def test_pt_expt_run_full_validation_wires_checkpoint_preflight(ema, tmp_path):
    trainer = _make_trainer("pt_expt", tmp_path)
    live_validator, ema_validator = _make_validators(trainer, tmp_path)
    trainer.full_validator = None if ema else live_validator
    trainer.ema_full_validator = ema_validator if ema else None
    validator = ema_validator if ema else live_validator
    previous_state = deepcopy(validator.state_store)
    _record_recovered_gradients(trainer)
    with (
        patch.object(
            validator,
            "evaluate_all_systems",
            return_value={validator.metric_key: 1.0},
        ),
        patch("torch.save", wraps=torch.save) as serialize,
        pytest.raises(RuntimeError, match="Non-finite gradient norm"),
    ):
        trainer.run_full_validation(step=0, display_step=1, learning_rate=1e-3)
    serialize.assert_not_called()
    assert validator.state_store == previous_state
    assert not validator.topk_records
    assert not list(tmp_path.glob("*.pt"))


@pytest.mark.parametrize("backend", ["pt", "pt_expt"])
def test_finite_best_checkpoints_serialize_topk_and_restart(backend, tmp_path):
    trainer = _make_trainer(backend, tmp_path)
    validators = _make_validators(trainer, tmp_path)
    for step, metric in ((1, 2.0), (2, 1.0)):
        for ema, validator in enumerate(validators):
            trainer.nonfinite_grad_guard.update(torch.tensor(1.0, device="cpu"))
            result = _validate(
                trainer, validator, step=step, metric=metric, ema=bool(ema)
            )
            checkpoint = torch.load(result.saved_best_path, weights_only=True)
            expected_weight = 2.0 if ema else 3.0
            assert torch.all(
                checkpoint["model"]["model.Default.weight"] == expected_weight
            )
            assert torch.all(trainer.model.weight == 3.0)
            if not ema:
                assert checkpoint["model"]["_extra_state"]["train_infos"][
                    TOPK_RECORDS_INFO_KEY
                ] == [{"metric": metric, "step": step}]
            else:
                assert "optimizer" not in checkpoint
                assert EMA_CHECKPOINT_KEY not in checkpoint

    # A regular live checkpoint carries both validators' restart bookkeeping;
    # EMA-only deployment snapshots deliberately omit EMA/optimizer state.
    restart_path = tmp_path / "restart.pt"
    _best_writer(trainer, False)(restart_path, step=2)
    checkpoint = torch.load(restart_path, weights_only=True)
    expected_records = [{"metric": 1.0, "step": 2}]
    assert (
        checkpoint[EMA_CHECKPOINT_KEY][EMA_VALIDATION_STATE_KEY][TOPK_RECORDS_INFO_KEY]
        == expected_records
    )
    restarted = _make_trainer(backend, tmp_path)
    restarted.wrapper.load_state_dict(checkpoint["model"])
    restarted.model_ema.load_state_dict(checkpoint[EMA_CHECKPOINT_KEY])
    for validator in _make_validators(restarted, tmp_path, restart=True):
        assert validator.state_store[TOPK_RECORDS_INFO_KEY] == expected_records
        assert [(record.metric, record.step) for record in validator.topk_records] == [
            (1.0, 2)
        ]
        paths = list(tmp_path.glob(f"{validator.best_checkpoint_prefix}-*.pt"))
        assert len(paths) == 1
        assert paths[0].name == f"{validator.best_checkpoint_prefix}-2.t-1.pt"


@pytest.mark.parametrize("backend", ["pt", "pt_expt"])
@pytest.mark.parametrize("ema", [False, True])
@pytest.mark.parametrize(
    "no_save_reason", ["not_selected", "save_best_false", "not_due"]
)
def test_validation_without_save_keeps_accumulated_failure(
    backend, ema, no_save_reason, tmp_path
):
    trainer = _make_trainer(backend, tmp_path)
    validator = _make_validators(
        trainer, tmp_path, save_best=no_save_reason != "save_best_false"
    )[int(ema)]
    if no_save_reason == "not_selected":
        _validate(trainer, validator, step=1, metric=1.0, ema=ema)
    elif no_save_reason == "not_due":
        validator.validation_freq = 3
    _record_recovered_gradients(trainer)
    flag = trainer.nonfinite_grad_guard._nonfinite
    with (
        patch.object(
            trainer,
            "ensure_finite_gradients_for_checkpoint",
            wraps=trainer.ensure_finite_gradients_for_checkpoint,
        ) as prepare,
        patch("torch.save", wraps=torch.save) as serialize,
    ):
        result = _validate(trainer, validator, step=2, metric=2.0, ema=ema)
    assert result is None or result.saved_best_path is None
    prepare.assert_not_called()
    serialize.assert_not_called()
    assert trainer.nonfinite_grad_guard._nonfinite is flag
    with pytest.raises(RuntimeError, match="Non-finite gradient norm"):
        trainer.ensure_finite_gradients_for_checkpoint()


def _save_regular_checkpoint(trainer, step):
    if isinstance(trainer, PTTrainer):
        trainer._save_training_checkpoint(step=step, display_step=step, lr=1e-3)
    else:
        trainer.save_checkpoint(step)


@pytest.mark.parametrize("backend", ["pt", "pt_expt"])
def test_regular_checkpoint_checks_once_for_live_and_ema(backend, tmp_path):
    trainer = _make_trainer(backend, tmp_path)
    trainer.nonfinite_grad_guard.update(torch.tensor(1.0, device="cpu"))
    with patch.object(
        trainer.nonfinite_grad_guard,
        "raise_if_nonfinite",
        wraps=trainer.nonfinite_grad_guard.raise_if_nonfinite,
    ) as check:
        _save_regular_checkpoint(trainer, 1)
    check.assert_called_once()
    assert trainer.ckpt_store.path_for(1).is_file()
    assert trainer.ema_ckpt_store.path_for(1).is_file()
    _record_recovered_gradients(trainer)
    with (
        patch.object(trainer, "_collect_checkpoint_states") as collect,
        pytest.raises(RuntimeError, match="Non-finite gradient norm"),
    ):
        _save_regular_checkpoint(trainer, 2)
    collect.assert_not_called()
    assert not trainer.ckpt_store.path_for(2).exists()
    assert not trainer.ema_ckpt_store.path_for(2).exists()


def _distributed_checkpoint_worker(rank, rendezvous, directory):
    """Run real Gloo and ZeRO-1 collection after a rank-local sticky failure."""
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        for backend in ("pt", "pt_expt"):
            for stage in (0, 1):
                for boundary in ("live_best", "ema_best", "regular"):
                    path = Path(directory) / f"{backend}-{stage}-{boundary}"
                    trainer = _make_trainer(
                        backend, path, rank=rank, stage=stage, distributed=True
                    )
                    if backend == "pt_expt":
                        assert trainer.checkpoint_is_collective
                    ema = boundary == "ema_best"
                    validator = _make_validators(trainer, path)[int(ema)]
                    old_state = deepcopy(validator.state_store)

                    def save(step):
                        if boundary == "regular":
                            _save_regular_checkpoint(trainer, step)
                        else:
                            _validate(
                                trainer, validator, step=step, metric=0.5, ema=ema
                            )

                    # Rank 0 has no accumulated norm at all. It must still join
                    # the preflight and see rank 1's historical failure before
                    # any checkpoint state collection (in particular ZeRO).
                    if rank == 1:
                        _record_recovered_gradients(trainer)
                    with (
                        patch.object(
                            trainer,
                            "_collect_checkpoint_states",
                            wraps=trainer._collect_checkpoint_states,
                        ) as collect,
                        patch("torch.save", wraps=torch.save) as serialize,
                        pytest.raises(RuntimeError, match="Non-finite gradient norm"),
                    ):
                        save(1)
                    collect.assert_not_called()
                    serialize.assert_not_called()
                    assert validator.state_store == old_state
                    assert not validator.topk_records
                    assert not list(path.glob("*.pt"))

                    # A finite interval still saves normally. Materialize Adam
                    # state so live saves exercise actual ZeRO consolidation.
                    for parameter in trainer.wrapper.parameters():
                        parameter.grad = torch.ones_like(parameter)
                    trainer.optimizer.step()
                    trainer.nonfinite_grad_guard.update(torch.tensor(1.0, device="cpu"))
                    with patch.object(
                        trainer,
                        "_collect_checkpoint_states",
                        wraps=trainer._collect_checkpoint_states,
                    ) as collect:
                        save(2)
                    expected_calls = 2 if boundary == "regular" else 1
                    if stage == 0 and rank != 0:
                        expected_calls = 0
                    assert collect.call_count == expected_calls
                    if rank == 0:
                        checkpoint_path = (
                            trainer.ckpt_store.path_for(2)
                            if boundary == "regular"
                            else path / f"{validator.best_checkpoint_prefix}-2.t-1.pt"
                        )
                        checkpoint = torch.load(checkpoint_path, weights_only=True)
                        if ema:
                            assert "optimizer" not in checkpoint
                        else:
                            assert len(checkpoint["optimizer"]["state"]) == 2
                        if boundary != "regular":
                            assert validator.state_store[TOPK_RECORDS_INFO_KEY] == [
                                {"metric": 0.5, "step": 2}
                            ]
                    dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(), reason="Gloo is required"
)
def test_rank_local_failure_precedes_zero_checkpoint_collectives(tmp_path):
    context = mp.spawn(
        _distributed_checkpoint_worker,
        args=(str(tmp_path / "rendezvous"), str(tmp_path)),
        nprocs=2,
        join=False,
    )
    try:
        # A mismatched collective fails its process-group timeout; this outer
        # deadline also prevents a regression from wedging the test runner.
        if not context.join(timeout=90):
            if not context.join(timeout=30):
                pytest.fail("Checkpoint workers did not finish within 120 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=10)
