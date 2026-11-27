# SPDX-License-Identifier: LGPL-3.0-or-later
"""JAX LMDB training uses the shared sampler, statistics, and validation data."""

from pathlib import (
    Path,
)

import lmdb
import msgpack
import numpy as np
import pytest

from deepmd.dpmodel.train import (
    TrainEntrypointOptions,
)
from deepmd.jax.entrypoints.train import (
    JAXTrainEntrypoint,
    _build_data_system,
)
from deepmd.jax.env import (
    jax,
    nnx,
)
from deepmd.utils.argcheck import (
    normalize,
)
from deepmd.utils.compat import (
    update_deepmd_input,
)

from ..common.dpmodel.test_lmdb_data import (
    _make_frame,
)
from .test_dpa4_model import (
    model_config,
)


def _write_lmdb(path: Path) -> None:
    counts = [2, 3, 2, 3]
    environment = lmdb.open(str(path), map_size=10 * 1024 * 1024)
    with environment.begin(write=True) as transaction:
        metadata = {
            "nframes": len(counts),
            "frame_idx_fmt": "012d",
            "frame_nlocs": counts,
            "type_map": ["O", "H"],
            "system_info": {"natoms": [1, 1]},
        }
        transaction.put(b"__metadata__", msgpack.packb(metadata, use_bin_type=True))
        for index, count in enumerate(counts):
            transaction.put(
                f"{index:012d}".encode(),
                msgpack.packb(_make_frame(count, index), use_bin_type=True),
            )
    environment.close()


def test_lmdb_epoch_statistics_and_full_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An LMDB sampling pass trains, writes shared statistics, and validates."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DP_LMDB_NUM_WORKERS", "0")
    path = tmp_path / "data.lmdb"
    _write_lmdb(path)
    data = _build_data_system(
        {"systems": str(path), "batch_size": "mix:8"}, 4.0, ["O", "H"], seed=10
    )
    data.use_ragged_batches(True)
    device_count = jax.local_device_count()
    expected_steps = (data.get_batch_pass_length() + device_count - 1) // device_count
    data.close()
    config = {
        "model": {**model_config(), "data_stat_nbatch": 1},
        "learning_rate": {
            "type": "exp",
            "start_lr": 1e-3,
            "stop_lr": 1e-4,
            "decay_steps": 1,
        },
        "loss": {
            "type": "ener",
            "start_pref_e": 1.0,
            "limit_pref_e": 1.0,
            "start_pref_f": 1.0,
            "limit_pref_f": 1.0,
        },
        "training": {
            "training_data": {"systems": str(path), "batch_size": "mix:8"},
            "validation_data": {
                "systems": str(path),
                "batch_size": "mix:8",
                "numb_btch": 1,
            },
            "numb_epoch": 1,
            "seed": 10,
            "disp_freq": 1,
            "save_freq": 1,
            "save_ckpt": "model",
            "stat_file": "stats.h5",
        },
        "validating": {
            "full_validation": True,
            "full_val_start": 0.0,
            "validation_freq": 1,
            "save_best": False,
            "full_val_file": "validation.log",
        },
    }
    config = normalize(update_deepmd_input(config, warning=False), backend="jax")
    JAXTrainEntrypoint().run_training(
        config,
        TrainEntrypointOptions(input_file="unused.json"),
        None,
    )
    curve = np.loadtxt(tmp_path / "lcurve.out")
    assert int(np.atleast_2d(curve)[-1, 0]) == expected_steps
    assert np.isfinite(np.atleast_2d(curve)).all()
    assert (tmp_path / "stats.h5").exists()
    assert (tmp_path / "model.jax").exists()
    assert "step" in (tmp_path / "validation.log").read_text().lower()


def test_lmdb_restart_continues_the_data_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A restart reads the batches that the uninterrupted run reads."""
    from deepmd.jax.train.trainer import (
        DPTrainer,
    )
    from deepmd.utils import random as dp_random

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DP_LMDB_NUM_WORKERS", "0")
    path = tmp_path / "data.lmdb"
    _write_lmdb(path)

    def run(name: str, steps: int, restart: str | None = None) -> DPTrainer:
        config = {
            "model": {**model_config(), "data_stat_nbatch": 1},
            "learning_rate": {
                "type": "exp",
                "start_lr": 1e-2,
                "stop_lr": 1e-2,
                "decay_steps": 1,
            },
            "loss": {
                "type": "ener",
                "start_pref_e": 1.0,
                "limit_pref_e": 1.0,
                "start_pref_f": 1.0,
                "limit_pref_f": 1.0,
            },
            "optimizer": {"type": "Adam"},
            "training": {
                "numb_steps": steps,
                "disp_freq": 1,
                "save_freq": 1,
                "save_ckpt": str(tmp_path / name / "model"),
                "disp_file": str(tmp_path / name / "lcurve.out"),
            },
        }
        dp_random.seed(10)
        trainer = DPTrainer(config, restart=restart)
        data = _build_data_system(
            {"systems": str(path), "batch_size": 1}, 4.0, ["O", "H"], seed=10
        )
        trainer.train(data, None)
        data.close()
        return trainer

    continuous = run("continuous", 3)
    run("resumed", 2)
    resumed = run("resumed", 3, restart=str(tmp_path / "resumed" / "model.jax"))
    assert resumed.start_step == 2
    expected = jax.tree.leaves(nnx.state(continuous.models, nnx.Param))
    actual = jax.tree.leaves(nnx.state(resumed.models, nnx.Param))
    for want, got in zip(expected, actual, strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
