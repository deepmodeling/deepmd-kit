# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import tempfile
import unittest
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)
from unittest.mock import (
    patch,
)

import lmdb
import msgpack
import numpy as np
import torch
from dargs.dargs import (
    ArgumentValueError,
)

from deepmd.pt.train.validation import (
    BEST_METRIC_NAME_INFO_KEY,
    TOPK_RECORDS_INFO_KEY,
    FullValidator,
    resolve_full_validation_start_step,
)
from deepmd.pt.utils.lmdb_dataset import (
    LmdbDataset,
)
from deepmd.utils.argcheck import (
    normalize,
)

from .model.test_permutation import (
    model_se_e2_a,
)


class _DummyValidationData:
    def __init__(self) -> None:
        self.systems = []


class _DummyModel(torch.nn.Module):
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def get_dim_fparam(self) -> int:
        return 0

    def get_dim_aparam(self) -> int:
        return 0


def _make_lmdb_frame(natoms: int, seed: int) -> dict:
    """Create one synthetic LMDB frame for full-validation tests."""
    rng = np.random.RandomState(seed)
    n_type0 = max(1, natoms // 3)
    n_type1 = natoms - n_type0
    atype = np.array([0] * n_type0 + [1] * n_type1, dtype=np.int64)
    return {
        "atom_names": ["O", "H"],
        "atom_numbs": [
            {
                "type": "<i8",
                "shape": (1,),
                "data": np.array([n_type0], dtype=np.int64).tobytes(),
            },
            {
                "type": "<i8",
                "shape": (1,),
                "data": np.array([n_type1], dtype=np.int64).tobytes(),
            },
        ],
        "atom_types": {
            "type": "<i8",
            "shape": (natoms,),
            "data": atype.tobytes(),
        },
        "coords": {
            "type": "<f8",
            "shape": (natoms, 3),
            "data": rng.randn(natoms, 3).astype(np.float64).tobytes(),
        },
        "cells": {
            "type": "<f8",
            "shape": (3, 3),
            "data": (np.eye(3) * 10.0).astype(np.float64).tobytes(),
        },
        "energies": {
            "type": "<f8",
            "shape": (1,),
            "data": rng.randn(1).astype(np.float64).tobytes(),
        },
        "forces": {
            "type": "<f8",
            "shape": (natoms, 3),
            "data": rng.randn(natoms, 3).astype(np.float64).tobytes(),
        },
    }


def _create_mixed_nloc_lmdb(path: str) -> str:
    """Create a mixed-nloc LMDB dataset with 6, 9, and 12-atom frames."""
    frame_specs = [(6, 4), (9, 4), (12, 2)]
    total_frames = sum(count for _, count in frame_specs)
    env = lmdb.open(path, map_size=10 * 1024 * 1024)
    with env.begin(write=True) as txn:
        metadata = {
            "nframes": total_frames,
            "frame_idx_fmt": "012d",
            "type_map": ["O", "H"],
            "system_info": {
                "natoms": [2, 4],
                "formula": "mixed",
            },
        }
        txn.put(b"__metadata__", msgpack.packb(metadata, use_bin_type=True))
        frame_idx = 0
        for natoms, count in frame_specs:
            for _ in range(count):
                txn.put(
                    format(frame_idx, "012d").encode(),
                    msgpack.packb(
                        _make_lmdb_frame(natoms=natoms, seed=frame_idx),
                        use_bin_type=True,
                    ),
                )
                frame_idx += 1
    env.close()
    return path


def _make_single_task_config() -> dict:
    return {
        "model": deepcopy(model_se_e2_a),
        "learning_rate": {
            "type": "exp",
            "start_lr": 0.001,
            "stop_lr": 1e-8,
            "decay_steps": 10,
        },
        "optimizer": {
            "type": "Adam",
        },
        "loss": {
            "type": "ener",
            "start_pref_e": 1.0,
            "limit_pref_e": 1.0,
            "start_pref_f": 1.0,
            "limit_pref_f": 1.0,
            "start_pref_v": 1.0,
            "limit_pref_v": 1.0,
        },
        "training": {
            "training_data": {"systems": ["train_system"]},
            "validation_data": {"systems": ["valid_system"]},
            "numb_steps": 10,
        },
        "validating": {
            "full_validation": True,
            "validation_freq": 2,
            "save_best": True,
            "max_best_ckpt": 1,
            "validation_metric": "E:MAE",
            "full_val_file": "val.log",
            "full_val_start": 0.0,
        },
    }


class TestValidationHelpers(unittest.TestCase):
    def test_resolve_full_validation_start_step(self) -> None:
        self.assertEqual(resolve_full_validation_start_step(0, 2000000), 0)
        self.assertEqual(resolve_full_validation_start_step(0.1, 2000000), 200000)
        self.assertEqual(resolve_full_validation_start_step(5000, 2000000), 5000)
        self.assertIsNone(resolve_full_validation_start_step(1, 2000000))

    def test_full_validator_rotates_best_checkpoint(self) -> None:
        train_infos = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                validator = FullValidator(
                    validating_params={
                        "full_validation": True,
                        "validation_freq": 1,
                        "save_best": True,
                        "max_best_ckpt": 2,
                        "validation_metric": "E:MAE",
                        "full_val_file": "val.log",
                        "full_val_start": 0.0,
                    },
                    validation_data=_DummyValidationData(),
                    model=_DummyModel(),
                    train_infos=train_infos,
                    num_steps=10,
                    rank=0,
                    zero_stage=0,
                    restart_training=False,
                )
                new_best_path = validator._update_best_state(
                    display_step=1,
                    selected_metric_value=2.0,
                )
                Path(new_best_path).touch()
                validator._reconcile_best_checkpoints()

                new_best_path = validator._update_best_state(
                    display_step=2,
                    selected_metric_value=1.0,
                )
                Path(new_best_path).touch()
                validator._reconcile_best_checkpoints()

                new_best_path = validator._update_best_state(
                    display_step=3,
                    selected_metric_value=1.5,
                )
                Path(new_best_path).touch()
                validator._reconcile_best_checkpoints()
            finally:
                os.chdir(old_cwd)

            self.assertEqual(new_best_path, "best.ckpt-3.t-2.pt")
            self.assertEqual(
                sorted(path.name for path in Path(tmpdir).glob("best.ckpt-*.pt")),
                ["best.ckpt-2.t-1.pt", "best.ckpt-3.t-2.pt"],
            )
            self.assertEqual(
                train_infos[TOPK_RECORDS_INFO_KEY],
                [
                    {"metric": 1.0, "step": 2},
                    {"metric": 1.5, "step": 3},
                ],
            )
            self.assertEqual(train_infos[BEST_METRIC_NAME_INFO_KEY], "e:mae")

    def test_full_validator_restores_top_k_checkpoints(self) -> None:
        train_infos = {
            BEST_METRIC_NAME_INFO_KEY: "e:mae",
            TOPK_RECORDS_INFO_KEY: [
                {"metric": 1.0, "step": 20},
                {"metric": 2.0, "step": 10},
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                Path("best.ckpt-20.t-9.pt").touch()
                Path("best.ckpt-10.t-8.pt").touch()
                Path("best.ckpt-999.t-1.pt").touch()
                FullValidator(
                    validating_params={
                        "full_validation": True,
                        "validation_freq": 1,
                        "save_best": True,
                        "max_best_ckpt": 2,
                        "validation_metric": "E:MAE",
                        "full_val_file": "val.log",
                        "full_val_start": 0.0,
                    },
                    validation_data=_DummyValidationData(),
                    model=_DummyModel(),
                    train_infos=train_infos,
                    num_steps=10,
                    rank=0,
                    zero_stage=0,
                    restart_training=True,
                )
            finally:
                os.chdir(old_cwd)

            self.assertEqual(
                sorted(path.name for path in Path(tmpdir).glob("best.ckpt-*.pt")),
                ["best.ckpt-10.t-2.pt", "best.ckpt-20.t-1.pt"],
            )

    def test_full_validator_lmdb_full_validation_iterates_nloc_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lmdb_path = _create_mixed_nloc_lmdb(f"{tmpdir}/mixed.lmdb")
            validation_data = LmdbDataset(
                lmdb_path,
                type_map=["O", "H"],
                batch_size=2,
            )
            validator = FullValidator(
                validating_params={
                    "full_validation": True,
                    "validation_freq": 1,
                    "save_best": False,
                    "max_best_ckpt": 1,
                    "validation_metric": "E:MAE",
                    "full_val_file": "val.log",
                    "full_val_start": 0.0,
                },
                validation_data=validation_data,
                model=_DummyModel(),
                train_infos={},
                num_steps=10,
                rank=0,
                zero_stage=0,
                restart_training=False,
            )
            observed_natoms = []

            def fake_evaluate_system(data_system):
                test_data = data_system.get_test()
                natoms = int(test_data["type"].shape[1])
                nframes = int(test_data["coord"].shape[0])
                observed_natoms.append(natoms)
                return {
                    "mae_e_per_atom": (float(natoms), nframes),
                    "rmse_e_per_atom": (float(natoms), nframes),
                }

            with patch.object(
                validator,
                "_evaluate_system",
                side_effect=fake_evaluate_system,
            ) as evaluate_system:
                metrics = validator.evaluate_all_systems()

        self.assertEqual(observed_natoms, [6, 9, 12])
        self.assertEqual(evaluate_system.call_count, 3)
        self.assertAlmostEqual(metrics["mae_e_per_atom"], 8.4)
        self.assertAlmostEqual(metrics["rmse_e_per_atom"], np.sqrt(75.6))


class TestValidationArgcheck(unittest.TestCase):
    def test_normalize_rejects_missing_validation_data(self) -> None:
        config = _make_single_task_config()
        del config["training"]["validation_data"]
        with self.assertRaisesRegex(ValueError, "training.validation_data"):
            normalize(config)

    def test_normalize_rejects_inactive_prefactor_metric(self) -> None:
        for start_pref_f, limit_pref_f in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)):
            with self.subTest(
                start_pref_f=start_pref_f,
                limit_pref_f=limit_pref_f,
            ):
                config = _make_single_task_config()
                config["validating"]["validation_metric"] = "F:RMSE"
                config["loss"]["start_pref_f"] = start_pref_f
                config["loss"]["limit_pref_f"] = limit_pref_f
                with self.assertRaisesRegex(ValueError, "start_pref_f"):
                    normalize(config)

    def test_normalize_rejects_invalid_metric(self) -> None:
        config = _make_single_task_config()
        config["validating"]["validation_metric"] = "X:MAE"
        with self.assertRaisesRegex(ArgumentValueError, "validation_metric"):
            normalize(config)

    def test_normalize_rejects_invalid_metric_with_num_epoch_schedule(self) -> None:
        config = _make_single_task_config()
        del config["training"]["numb_steps"]
        config["training"]["numb_epoch"] = 1.0
        config["validating"]["validation_metric"] = "F:RMSE"
        config["validating"]["full_val_start"] = 2
        config["loss"]["limit_pref_f"] = 0.0
        with self.assertRaisesRegex(ValueError, "start_pref_f"):
            normalize(config)

    def test_normalize_rejects_nonpositive_max_best_ckpt(self) -> None:
        config = _make_single_task_config()
        config["validating"]["max_best_ckpt"] = 0
        with self.assertRaisesRegex(ArgumentValueError, "max_best_ckpt"):
            normalize(config)
