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
    MagicMock,
    patch,
)

import lmdb
import msgpack
import numpy as np
import torch
from dargs.dargs import (
    ArgumentValueError,
)

from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.utils.env import (
    DEVICE,
)
from deepmd.pt.utils.lmdb_dataset import (
    LmdbDataset,
)
from deepmd.pt_expt.train.validation import (
    BEST_METRIC_NAME_INFO_KEY,
    TOPK_RECORDS_INFO_KEY,
    BestCheckpointRecord,
    FullValidator,
    resolve_full_validation_start_step,
)
from deepmd.utils.argcheck import (
    is_valid_full_validation_metric,
    normalize,
)
from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.eval_metrics import (
    FULL_VALIDATION_PROFILES,
    SPIN_FULL_VALIDATION_PROFILE,
    compute_full_validation_energy_metrics,
    compute_full_validation_spin_metrics,
)

from .model.test_permutation import (
    model_se_e2_a,
    model_spin,
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


class _LmdbDatasetWithoutTypeMap:
    lmdb_path = "missing-type-map.lmdb"


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


def _create_partially_labeled_lmdb(path: str) -> str:
    """Create same-nloc validation frames with complementary labels."""
    nframes = 4
    natoms = 6
    env = lmdb.open(path, map_size=10 * 1024 * 1024)
    with env.begin(write=True) as txn:
        metadata = {
            "nframes": nframes,
            "frame_idx_fmt": "012d",
            "type_map": ["O", "H"],
            "system_info": {"natoms": [2, 4]},
            "frame_nlocs": [natoms] * nframes,
        }
        txn.put(b"__metadata__", msgpack.packb(metadata, use_bin_type=True))
        for frame_idx in range(nframes):
            frame = _make_lmdb_frame(natoms=natoms, seed=frame_idx)
            if frame_idx % 2 == 0:
                frame.pop("forces")
                frame["energies"]["data"] = np.array([2.0], dtype=np.float64).tobytes()
            else:
                frame.pop("energies")
                frame["forces"]["data"] = np.ones(
                    (natoms, 3), dtype=np.float64
                ).tobytes()
            txn.put(
                format(frame_idx, "012d").encode(),
                msgpack.packb(frame, use_bin_type=True),
            )
    env.close()
    return path


def _create_mixed_nloc_partially_labeled_lmdb(path: str) -> str:
    """Create frames varying atom count and complementary label availability."""
    frame_specs = [(6, True), (6, False), (9, True), (9, False)]
    env = lmdb.open(path, map_size=10 * 1024 * 1024)
    with env.begin(write=True) as txn:
        metadata = {
            "nframes": len(frame_specs),
            "frame_idx_fmt": "012d",
            "type_map": ["O", "H"],
            "system_info": {"natoms": [2, 4]},
            "frame_nlocs": [natoms for natoms, _ in frame_specs],
        }
        txn.put(b"__metadata__", msgpack.packb(metadata, use_bin_type=True))
        for frame_idx, (natoms, has_energy) in enumerate(frame_specs):
            frame = _make_lmdb_frame(natoms=natoms, seed=frame_idx)
            if has_energy:
                frame.pop("forces")
            else:
                frame.pop("energies")
            txn.put(
                format(frame_idx, "012d").encode(),
                msgpack.packb(frame, use_bin_type=True),
            )
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


def _make_spin_task_config() -> dict:
    config = _make_single_task_config()
    config["loss"] = {
        "type": "ener_spin",
        "start_pref_e": 1.0,
        "limit_pref_e": 1.0,
        "start_pref_fr": 1.0,
        "limit_pref_fr": 1.0,
        "start_pref_fm": 1.0,
        "limit_pref_fm": 1.0,
    }
    config["validating"]["validation_metric"] = "FR:MAE"
    return config


class TestValidationHelpers(unittest.TestCase):
    def test_resolve_full_validation_start_step(self) -> None:
        self.assertEqual(resolve_full_validation_start_step(0, 2000000), 0)
        self.assertEqual(resolve_full_validation_start_step(0.1, 2000000), 200000)
        self.assertEqual(resolve_full_validation_start_step(5000, 2000000), 5000)
        self.assertIsNone(resolve_full_validation_start_step(1, 2000000))

    def test_full_validator_releases_workspace_after_restoring_model(self) -> None:
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as tmpdir:
                model = _DummyModel()
                eval_context = MagicMock()
                validator = FullValidator(
                    validating_params={
                        "full_validation": True,
                        "validation_freq": 1,
                        "full_val_start": 0.0,
                        "save_best": False,
                        "full_val_file": str(Path(tmpdir) / "val.log"),
                    },
                    validation_data=_DummyValidationData(),
                    model=model,
                    state_store={},
                    num_steps=10,
                    rank=0,
                    restart_training=False,
                    checkpoint_dir=Path(tmpdir),
                    model_eval_context=lambda: eval_context,
                )

                def evaluate() -> dict[str, float]:
                    self.assertFalse(model.training)
                    if fail:
                        raise RuntimeError("validation failure")
                    return {validator.metric_key: 1.0}

                def release() -> None:
                    self.assertTrue(model.training)
                    eval_context.__exit__.assert_called_once()

                with (
                    patch.object(
                        validator, "evaluate_all_systems", side_effect=evaluate
                    ),
                    patch("torch.cuda.is_available", return_value=True),
                    patch("torch.cuda.empty_cache", side_effect=release) as empty_cache,
                ):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "validation failure"):
                            validator.run(
                                step_id=1, display_step=1, lr=1e-3, save_checkpoint=None
                            )
                    else:
                        result = validator.run(
                            step_id=1, display_step=1, lr=1e-3, save_checkpoint=None
                        )
                        self.assertEqual(result.metrics[validator.metric_key], 1.0)
                empty_cache.assert_called_once()

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
                    state_store=train_infos,
                    num_steps=10,
                    rank=0,
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
                    state_store=train_infos,
                    num_steps=10,
                    rank=0,
                    restart_training=True,
                )
            finally:
                os.chdir(old_cwd)

            self.assertEqual(
                sorted(path.name for path in Path(tmpdir).glob("best.ckpt-*.pt")),
                ["best.ckpt-10.t-2.pt", "best.ckpt-20.t-1.pt"],
            )

    def test_full_validator_writes_best_into_custom_checkpoint_dir(self) -> None:
        train_infos = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                best_dir = Path("nested/best")
                validator = FullValidator(
                    validating_params={
                        "full_validation": True,
                        "validation_freq": 1,
                        "save_best": True,
                        "max_best_ckpt": 1,
                        "validation_metric": "E:MAE",
                        "full_val_file": "val.log",
                        "full_val_start": 0.0,
                    },
                    validation_data=_DummyValidationData(),
                    model=_DummyModel(),
                    state_store=train_infos,
                    num_steps=10,
                    rank=0,
                    restart_training=False,
                    checkpoint_dir=best_dir,
                )
                # The directory is created recursively at construction time.
                self.assertTrue(best_dir.is_dir())
                new_best_path = validator._update_best_state(
                    display_step=1,
                    selected_metric_value=2.0,
                )
            finally:
                os.chdir(old_cwd)

            self.assertEqual(new_best_path, str(best_dir / "best.ckpt-1.t-1.pt"))

    def test_full_validator_reconciles_directory_checkpoints(self) -> None:
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
                    state_store=train_infos,
                    num_steps=10,
                    rank=0,
                    restart_training=False,
                    best_checkpoint_suffix=".jax",
                )
                new_best_path = validator._update_best_state(
                    display_step=1,
                    selected_metric_value=2.0,
                )
                Path(new_best_path).mkdir()
                validator._reconcile_best_checkpoints()

                new_best_path = validator._update_best_state(
                    display_step=2,
                    selected_metric_value=1.0,
                )
                Path(new_best_path).mkdir()
                validator._reconcile_best_checkpoints()

                new_best_path = validator._update_best_state(
                    display_step=3,
                    selected_metric_value=1.5,
                )
                Path(new_best_path).mkdir()
                validator._reconcile_best_checkpoints()
            finally:
                os.chdir(old_cwd)

            self.assertEqual(new_best_path, "best.ckpt-3.t-2.jax")
            self.assertEqual(
                sorted(path.name for path in Path(tmpdir).glob("best.ckpt-*.jax")),
                ["best.ckpt-2.t-1.jax", "best.ckpt-3.t-2.jax"],
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
                state_store={},
                num_steps=10,
                rank=0,
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

    def test_full_validator_lmdb_excludes_default_filled_partial_labels(self) -> None:
        """Each optional-label metric must see only frames that provide it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lmdb_path = _create_partially_labeled_lmdb(f"{tmpdir}/partial.lmdb")
            validation_data = LmdbDataset(
                lmdb_path,
                type_map=["O", "H"],
                batch_size=2,
            )
            validation_data.add_data_requirement(
                [
                    DataRequirementItem(
                        "energy", 1, atomic=False, must=False, default=7.0
                    ),
                    DataRequirementItem(
                        "force", 3, atomic=True, must=False, default=11.0
                    ),
                ]
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
                state_store={},
                num_steps=10,
                rank=0,
                restart_training=False,
            )
            observed_flags = []

            def evaluate_label_group(data_system):
                test_data = data_system.get_test()
                natoms = int(test_data["type"].shape[1])
                nframes = int(test_data["coord"].shape[0])
                observed_flags.append(
                    (
                        float(test_data["find_energy"]),
                        float(test_data["find_force"]),
                        nframes,
                    )
                )
                prediction = {
                    "energy": np.zeros((nframes, 1)),
                    "force": np.zeros((nframes, natoms, 3)),
                }
                return validator.profile.compute_system_metrics(
                    prediction, test_data, natoms, True
                )

            with patch.object(
                validator,
                "_evaluate_system",
                side_effect=evaluate_label_group,
            ):
                metrics = validator.evaluate_all_systems()

        self.assertCountEqual(observed_flags, [(1.0, 0.0, 2), (0.0, 1.0, 2)])
        self.assertAlmostEqual(metrics["mae_e_per_atom"], 2.0 / 6.0)
        self.assertAlmostEqual(metrics["rmse_e_per_atom"], 2.0 / 6.0)
        self.assertAlmostEqual(metrics["mae_f"], 1.0)
        self.assertAlmostEqual(metrics["rmse_f"], 1.0)

    def test_full_validator_lmdb_groups_nloc_and_label_availability(self) -> None:
        """Full validation must preserve both dimensions of its tuple key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lmdb_path = _create_mixed_nloc_partially_labeled_lmdb(
                f"{tmpdir}/mixed-partial.lmdb"
            )
            validation_data = LmdbDataset(
                lmdb_path,
                type_map=["O", "H"],
                batch_size=2,
            )
            validation_data.add_data_requirement(
                [
                    DataRequirementItem(
                        "energy", 1, atomic=False, must=False, default=7.0
                    ),
                    DataRequirementItem(
                        "force", 3, atomic=True, must=False, default=11.0
                    ),
                ]
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
                state_store={},
                num_steps=10,
                rank=0,
                restart_training=False,
            )
            observed_groups = []

            def record_group(data_system):
                test_data = data_system.get_test()
                observed_groups.append(
                    (
                        int(test_data["type"].shape[1]),
                        float(test_data["find_energy"]),
                        float(test_data["find_force"]),
                    )
                )
                return {}

            with patch.object(validator, "_evaluate_system", side_effect=record_group):
                validator.evaluate_all_systems()

        self.assertCountEqual(
            observed_groups,
            [(6, 1.0, 0.0), (6, 0.0, 1.0), (9, 1.0, 0.0), (9, 0.0, 1.0)],
        )

    def test_full_validator_lmdb_snapshot_requires_type_map(self) -> None:
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
            validation_data=_DummyValidationData(),
            model=_DummyModel(),
            state_store={},
            num_steps=10,
            rank=0,
            restart_training=False,
        )

        with self.assertRaisesRegex(TypeError, "LMDB type_map"):
            validator._get_lmdb_test_data_snapshot(_LmdbDatasetWithoutTypeMap())


class TestValidationArgcheck(unittest.TestCase):
    def test_normalize_accepts_amp_infer(self) -> None:
        config = _make_single_task_config()
        normalized = normalize(config)
        self.assertFalse(normalized["validating"]["amp_infer"])

        config["validating"]["amp_infer"] = True
        normalized = normalize(config)
        self.assertTrue(normalized["validating"]["amp_infer"])

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

    def test_normalize_accepts_spin_force_metric(self) -> None:
        config = _make_spin_task_config()
        normalized = normalize(config)
        self.assertEqual(normalized["validating"]["validation_metric"], "FR:MAE")

    def test_normalize_rejects_energy_force_metric_for_spin(self) -> None:
        config = _make_spin_task_config()
        config["validating"]["validation_metric"] = "F:MAE"
        with self.assertRaisesRegex(ValueError, "spin training"):
            normalize(config)

    def test_normalize_rejects_spin_force_metric_for_energy(self) -> None:
        config = _make_single_task_config()
        config["validating"]["validation_metric"] = "FR:MAE"
        with self.assertRaisesRegex(ValueError, "energy training"):
            normalize(config)

    def test_normalize_rejects_inactive_spin_prefactor_metric(self) -> None:
        config = _make_spin_task_config()
        config["validating"]["validation_metric"] = "FM:RMSE"
        config["loss"]["limit_pref_fm"] = 0.0
        with self.assertRaisesRegex(ValueError, "start_pref_fm"):
            normalize(config)


class TestFullValidationMetricProfiles(unittest.TestCase):
    def test_spin_profile_splits_real_and_magnetic_forces(self) -> None:
        prediction = {
            "energy": np.array([[6.0]]),
            "force": np.array([[1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0]]),
            "force_mag": np.array(
                [[10.0, 10.0, 10.0, 99.0, 99.0, 99.0, 20.0, 20.0, 20.0]]
            ),
            "mask_mag": np.array([[True, False, True]]),
        }
        test_data = {
            "find_energy": 1.0,
            "find_force": 1.0,
            "find_force_mag": 1.0,
            "energy": np.array([[0.0]]),
            "force": np.zeros((1, 9)),
            "force_mag": np.zeros((1, 9)),
        }
        metrics = compute_full_validation_spin_metrics(
            prediction, test_data, natoms=3, has_pbc=False
        )
        # Energy is normalized per atom: |6| / 3 = 2.
        self.assertAlmostEqual(metrics["mae_e_per_atom"][0], 2.0)
        self.assertAlmostEqual(metrics["rmse_e_per_atom"][0], 2.0)
        # Real force spans all three atoms (nine components).
        self.assertAlmostEqual(metrics["mae_fr"][0], 2.0)
        self.assertAlmostEqual(metrics["rmse_fr"][0], np.sqrt(42.0 / 9.0))
        self.assertEqual(metrics["mae_fr"][1], 9.0)
        # Magnetic force only sees masked atoms 0 and 2 (six components).
        self.assertAlmostEqual(metrics["mae_fm"][0], 15.0)
        self.assertAlmostEqual(metrics["rmse_fm"][0], np.sqrt(250.0))
        self.assertEqual(metrics["mae_fm"][1], 6.0)

    def test_profile_tables_derive_consistently_from_families(self) -> None:
        for profile in FULL_VALIDATION_PROFILES.values():
            with self.subTest(profile=profile.name):
                self.assertEqual(
                    set(profile.metric_key_map), set(profile.prefactor_by_metric)
                )
                self.assertEqual(
                    set(profile.metric_family_by_key.values()),
                    set(profile.unit_by_family),
                )
                for metric, key in profile.metric_key_map.items():
                    self.assertEqual(
                        profile.metric_family_by_key[key], metric.split(":")[0]
                    )

    def test_second_rank_column_follows_the_selected_metric(self) -> None:
        for profile in FULL_VALIDATION_PROFILES.values():
            with self.subTest(profile=profile.name):
                default = [label for label, _ in profile.columns("e:mae")]
                self.assertIn("S_MAE", default)
                self.assertNotIn("V_MAE", default)
                selected = [label for label, _ in profile.columns("v:rmse")]
                self.assertIn("V_RMSE", selected)
                self.assertNotIn("S_RMSE", selected)
                # Every selectable metric stays reachable by the
                # best-checkpoint selector, and one trained quantity never
                # occupies two columns.
                quantities = {family.prefactors for family in profile.families}
                for metric, key in profile.metric_key_map.items():
                    columns = profile.columns(metric)
                    self.assertIn(key, [column_key for _, column_key in columns])
                    self.assertEqual(len(columns), 2 * len(quantities))

    def test_virial_metric_selectors_remain_accepted(self) -> None:
        for metric in ("V:MAE", "V:RMSE", "v:mae", "v:rmse"):
            self.assertTrue(is_valid_full_validation_metric(metric))

    def test_energy_profile_reports_stress_and_per_atom_virial(self) -> None:
        # A 2 Angstrom cubic cell has volume 8, so a unit virial error becomes
        # 1/8 in stress and 1/natoms in per-atom virial.
        prediction = {
            "energy": np.zeros((1, 1)),
            "force": np.zeros((1, 12)),
            "virial": np.ones((1, 9)),
        }
        test_data = {
            "find_energy": 1.0,
            "find_force": 1.0,
            "find_virial": 1.0,
            "energy": np.zeros((1, 1)),
            "force": np.zeros((1, 12)),
            "virial": np.zeros((1, 9)),
            "box": np.tile((np.eye(3) * 2.0).reshape(9), (1, 1)),
        }
        metrics = compute_full_validation_energy_metrics(
            prediction, test_data, natoms=4, has_pbc=True
        )
        self.assertAlmostEqual(metrics["mae_v_per_atom"][0], 0.25)
        self.assertAlmostEqual(metrics["rmse_v_per_atom"][0], 0.25)
        self.assertAlmostEqual(metrics["mae_s"][0], 0.125)
        self.assertAlmostEqual(metrics["rmse_s"][0], 0.125)
        # The two presentations carry the same virial error under different
        # normalizations.
        self.assertAlmostEqual(
            metrics["mae_s"][0] * 8.0, metrics["mae_v_per_atom"][0] * 4.0
        )

    def test_singular_cell_drops_stress_but_keeps_virial(self) -> None:
        prediction = {
            "energy": np.zeros((1, 1)),
            "force": np.zeros((1, 12)),
            "virial": np.ones((1, 9)),
        }
        test_data = {
            "find_energy": 1.0,
            "find_force": 1.0,
            "find_virial": 1.0,
            "energy": np.zeros((1, 1)),
            "force": np.zeros((1, 12)),
            "virial": np.zeros((1, 9)),
            "box": np.zeros((1, 9)),
        }
        metrics = compute_full_validation_energy_metrics(
            prediction, test_data, natoms=4, has_pbc=True
        )
        self.assertIn("mae_v_per_atom", metrics)
        self.assertNotIn("mae_s", metrics)

    def test_spin_profile_omits_magnetic_force_when_unavailable(self) -> None:
        prediction = {
            "energy": np.array([[3.0]]),
            "force": np.zeros((1, 9)),
            "force_mag": np.zeros((1, 9)),
            "mask_mag": np.array([[True, False, True]]),
        }
        test_data = {
            "find_energy": 1.0,
            "find_force": 1.0,
            "find_force_mag": 0.0,
            "energy": np.array([[0.0]]),
            "force": np.zeros((1, 9)),
        }
        metrics = compute_full_validation_spin_metrics(
            prediction, test_data, natoms=3, has_pbc=False
        )
        self.assertIn("mae_fr", metrics)
        self.assertNotIn("mae_fm", metrics)

    def test_predict_outputs_emits_real_and_magnetic_forces(self) -> None:
        model = get_model(deepcopy(model_spin)).to(DEVICE)
        nframes = 2
        natoms = 5
        rng = np.random.default_rng(0)
        coord = 3.0 * rng.random((nframes, natoms * 3))
        atom_types = np.tile(np.array([0, 0, 0, 1, 1]), (nframes, 1))
        box = np.tile((np.eye(3) * 6.0).reshape(9), (nframes, 1))
        spin = 0.5 * rng.random((nframes, natoms * 3))
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                validator = FullValidator(
                    validating_params={
                        "full_validation": True,
                        "validation_freq": 1,
                        "save_best": False,
                        "max_best_ckpt": 1,
                        "validation_metric": "FR:MAE",
                        "full_val_file": "val.log",
                        "full_val_start": 0.0,
                    },
                    validation_data=_DummyValidationData(),
                    model=model,
                    state_store={},
                    num_steps=10,
                    rank=0,
                    restart_training=False,
                )
                self.assertIs(validator.profile, SPIN_FULL_VALIDATION_PROFILE)
                prediction = validator._predict_outputs(
                    coord=coord,
                    atom_types=atom_types,
                    box=box,
                    fparam=None,
                    aparam=None,
                    spin=spin,
                    include_virial=False,
                    natoms=natoms,
                    nframes=nframes,
                )
            finally:
                os.chdir(old_cwd)

        self.assertEqual(prediction["energy"].shape, (nframes, 1))
        self.assertEqual(prediction["force"].shape, (nframes, natoms * 3))
        self.assertEqual(prediction["force_mag"].shape, (nframes, natoms * 3))
        self.assertEqual(prediction["mask_mag"].shape, (nframes, natoms))
        self.assertNotIn("virial", prediction)
        # use_spin=[True, False, False] makes only type-0 atoms magnetic.
        expected_mask = np.tile(
            np.array([True, True, True, False, False]), (nframes, 1)
        )
        np.testing.assert_array_equal(
            prediction["mask_mag"].astype(bool), expected_mask
        )


class TestFullValidatorCheckpointGate(unittest.TestCase):
    def _make_validator(self, tmpdir: str) -> FullValidator:
        return FullValidator(
            validating_params={
                "full_validation": True,
                "validation_freq": 1,
                "save_best": True,
                "max_best_ckpt": 1,
                "validation_metric": "E:MAE",
                "full_val_file": str(Path(tmpdir) / "val.log"),
                "full_val_start": 0.0,
            },
            validation_data=_DummyValidationData(),
            model=_DummyModel(),
            state_store={},
            num_steps=10,
            rank=0,
            restart_training=False,
            checkpoint_dir=Path(tmpdir),
        )

    def test_full_validator_defers_topk_until_successful_save(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = self._make_validator(tmpdir)
            writes: list[Path] = []
            serialized_topk: list[list[dict]] = []

            def save_checkpoint(path: Path, lr: float = 0.0, step: int = 0) -> None:
                del lr, step
                # The checkpoint bytes must already see the proposed top-K.
                serialized_topk.append(
                    list(validator.state_store[validator.topk_records_info_key])
                )
                Path(path).write_text("ok")
                writes.append(Path(path))

            with patch.object(
                validator,
                "evaluate_all_systems",
                return_value={validator.metric_key: 0.5},
            ):
                result = validator.run(
                    step_id=1,
                    display_step=1,
                    lr=1e-3,
                    save_checkpoint=save_checkpoint,
                )

            self.assertIsNotNone(result)
            self.assertEqual(len(writes), 1)
            self.assertTrue(writes[0].exists())
            self.assertEqual(len(validator.topk_records), 1)
            self.assertEqual(validator.topk_records[0].step, 1)
            self.assertIsNone(validator._pending_topk_records)
            self.assertIsNone(validator._rollback_topk_records)
            self.assertEqual(serialized_topk, [[{"metric": 0.5, "step": 1}]])

    def test_full_validator_discards_topk_when_save_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = self._make_validator(tmpdir)

            def save_checkpoint(path: Path, lr: float = 0.0, step: int = 0) -> None:
                del path, lr, step
                raise RuntimeError("Non-finite gradient norm; training has diverged.")

            with patch.object(
                validator,
                "evaluate_all_systems",
                return_value={validator.metric_key: 0.25},
            ):
                with self.assertRaisesRegex(RuntimeError, "Non-finite"):
                    validator.run(
                        step_id=2,
                        display_step=2,
                        lr=1e-3,
                        save_checkpoint=save_checkpoint,
                    )

            self.assertEqual(validator.topk_records, [])
            self.assertIsNone(validator._pending_topk_records)
            self.assertIsNone(validator._rollback_topk_records)
            self.assertEqual(
                validator.state_store.get(validator.topk_records_info_key, []),
                [],
            )
            self.assertEqual(
                list(Path(tmpdir).glob("best.ckpt-*.pt")),
                [],
            )

    def test_full_validator_serialized_topk_survives_restart_reconcile(self) -> None:
        """Improved best ckpt must serialize its own record so restart keeps it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store: dict = {}
            validator = FullValidator(
                validating_params={
                    "full_validation": True,
                    "validation_freq": 1,
                    "save_best": True,
                    "max_best_ckpt": 1,
                    "validation_metric": "E:MAE",
                    "full_val_file": str(Path(tmpdir) / "val.log"),
                    "full_val_start": 0.0,
                },
                validation_data=_DummyValidationData(),
                model=_DummyModel(),
                state_store=state_store,
                num_steps=10,
                rank=0,
                restart_training=False,
                checkpoint_dir=Path(tmpdir),
            )
            serialized_stores: list[dict] = []

            def save_checkpoint(path: Path, lr: float = 0.0, step: int = 0) -> None:
                del lr, step
                # Snapshot what a real torch.save would capture from train_infos.
                serialized_stores.append(
                    {
                        validator.metric_name_info_key: state_store[
                            validator.metric_name_info_key
                        ],
                        validator.topk_records_info_key: [
                            dict(record)
                            for record in state_store[validator.topk_records_info_key]
                        ],
                    }
                )
                Path(path).write_text(f"ckpt-{path.name}")
                # Leave older best files in place; reconcile decides retention.

            metrics = {1: 0.5, 2: 0.1}
            for display_step, metric in metrics.items():
                with patch.object(
                    validator,
                    "evaluate_all_systems",
                    return_value={validator.metric_key: metric},
                ):
                    validator.run(
                        step_id=display_step,
                        display_step=display_step,
                        lr=1e-3,
                        save_checkpoint=save_checkpoint,
                    )

            self.assertEqual(len(serialized_stores), 2)
            self.assertEqual(
                serialized_stores[1][validator.topk_records_info_key],
                [{"metric": 0.1, "step": 2}],
            )
            # Restart from the improved checkpoint's serialized train_infos.
            restarted = FullValidator(
                validating_params={
                    "full_validation": True,
                    "validation_freq": 1,
                    "save_best": True,
                    "max_best_ckpt": 1,
                    "validation_metric": "E:MAE",
                    "full_val_file": str(Path(tmpdir) / "val.log"),
                    "full_val_start": 0.0,
                },
                validation_data=_DummyValidationData(),
                model=_DummyModel(),
                state_store=dict(serialized_stores[1]),
                num_steps=10,
                rank=0,
                restart_training=True,
                checkpoint_dir=Path(tmpdir),
            )
            self.assertEqual(len(restarted.topk_records), 1)
            self.assertEqual(restarted.topk_records[0].step, 2)
            remaining = sorted(Path(tmpdir).glob("best.ckpt-*.pt"))
            self.assertEqual(len(remaining), 1)
            self.assertIn("best.ckpt-2", remaining[0].name)

    def test_full_validator_rolls_back_previous_topk_on_failed_save(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = self._make_validator(tmpdir)

            def save_ok(path: Path, lr: float = 0.0, step: int = 0) -> None:
                del lr, step
                Path(path).write_text("ok")

            def save_fail(path: Path, lr: float = 0.0, step: int = 0) -> None:
                del path, lr, step
                raise RuntimeError("Non-finite gradient norm; training has diverged.")

            with patch.object(
                validator,
                "evaluate_all_systems",
                return_value={validator.metric_key: 0.5},
            ):
                validator.run(
                    step_id=1,
                    display_step=1,
                    lr=1e-3,
                    save_checkpoint=save_ok,
                )
            self.assertEqual(validator.topk_records[0].step, 1)

            with patch.object(
                validator,
                "evaluate_all_systems",
                return_value={validator.metric_key: 0.1},
            ):
                with self.assertRaisesRegex(RuntimeError, "Non-finite"):
                    validator.run(
                        step_id=2,
                        display_step=2,
                        lr=1e-3,
                        save_checkpoint=save_fail,
                    )

            self.assertEqual(len(validator.topk_records), 1)
            self.assertEqual(validator.topk_records[0].step, 1)
            self.assertEqual(
                validator.state_store[validator.topk_records_info_key],
                [{"metric": 0.5, "step": 1}],
            )
            self.assertIsNone(validator._pending_topk_records)
            self.assertIsNone(validator._rollback_topk_records)

    def test_local_write_then_fail_reconciles_orphan(self) -> None:
        """Local save that writes then raises must delete the orphan file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = self._make_validator(tmpdir)

            def save_ok(path: Path, lr: float = 0.0, step: int = 0) -> None:
                del lr, step
                Path(path).write_text("ok")

            def save_write_then_fail(
                path: Path, lr: float = 0.0, step: int = 0
            ) -> None:
                del lr, step
                Path(path).write_text("orphan")
                raise RuntimeError("post-write failure")

            with patch.object(
                validator,
                "evaluate_all_systems",
                return_value={validator.metric_key: 0.5},
            ):
                validator.run(
                    step_id=1,
                    display_step=1,
                    lr=1e-3,
                    save_checkpoint=save_ok,
                )
            kept = sorted(Path(tmpdir).glob("best.ckpt-*.pt"))
            self.assertEqual(len(kept), 1)
            self.assertIn("best.ckpt-1", kept[0].name)

            with patch.object(
                validator,
                "evaluate_all_systems",
                return_value={validator.metric_key: 0.1},
            ):
                with self.assertRaisesRegex(RuntimeError, "post-write"):
                    validator.run(
                        step_id=2,
                        display_step=2,
                        lr=1e-3,
                        save_checkpoint=save_write_then_fail,
                    )

            self.assertEqual(len(validator.topk_records), 1)
            self.assertEqual(validator.topk_records[0].step, 1)
            remaining = sorted(Path(tmpdir).glob("best.ckpt-*.pt"))
            self.assertEqual(len(remaining), 1)
            self.assertIn("best.ckpt-1", remaining[0].name)
            self.assertFalse(any("best.ckpt-2" in p.name for p in remaining))

    def test_distributed_remote_save_failure_rolls_back_committed_topk(self) -> None:
        """Rank 0 must not keep top-K when another rank aborts the save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = self._make_validator(tmpdir)
            validator.is_distributed = True
            writes: list[Path] = []

            def save_checkpoint(path: Path, lr: float = 0.0, step: int = 0) -> None:
                del lr, step
                Path(path).write_text("ok")
                writes.append(Path(path))

            gather_calls = {"n": 0}

            def gather_remote_error(out: list, obj: object) -> None:
                # Succeed on the post-evaluate sync; fail on the post-save sync.
                gather_calls["n"] += 1
                if gather_calls["n"] == 2:
                    out[:] = [
                        obj,
                        "Full validation failed while saving the best checkpoint:\nremote",
                    ]
                else:
                    out[:] = [obj, None]

            with (
                patch.object(
                    validator,
                    "evaluate_all_systems",
                    return_value={validator.metric_key: 0.5},
                ),
                patch("deepmd.pt_expt.train.validation.dist.barrier"),
                patch(
                    "deepmd.pt_expt.train.validation.dist.broadcast_object_list",
                ),
                patch(
                    "deepmd.pt_expt.train.validation.dist.get_world_size",
                    return_value=2,
                ),
                patch(
                    "deepmd.pt_expt.train.validation.dist.all_gather_object",
                    side_effect=gather_remote_error,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "remote"):
                    validator.run(
                        step_id=3,
                        display_step=3,
                        lr=1e-3,
                        save_checkpoint=save_checkpoint,
                    )

            self.assertEqual(len(writes), 1)
            self.assertEqual(validator.topk_records, [])
            self.assertIsNone(validator._pending_topk_records)
            self.assertIsNone(validator._rollback_topk_records)
            self.assertEqual(
                validator.state_store.get(validator.topk_records_info_key, []),
                [],
            )
            self.assertEqual(list(Path(tmpdir).glob("best.ckpt-*.pt")), [])

    def test_distributed_non_chief_enters_save_callback(self) -> None:
        """Non-chief ranks must enter save so the non-finite gate can reset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = self._make_validator(tmpdir)
            validator.rank = 1
            validator.is_distributed = True
            writes: list[Path] = []

            def save_checkpoint(path: Path, lr: float = 0.0, step: int = 0) -> None:
                del lr, step
                Path(path).write_text("ok")
                writes.append(Path(path))

            def broadcast_save_path(holder: list, src: int = 0) -> None:
                del src
                holder[0] = str(Path(tmpdir) / "best.ckpt-7-1.pt")

            with (
                patch(
                    "deepmd.pt_expt.train.validation.dist.barrier",
                ),
                patch(
                    "deepmd.pt_expt.train.validation.dist.broadcast_object_list",
                    side_effect=broadcast_save_path,
                ),
                patch(
                    "deepmd.pt_expt.train.validation.dist.get_world_size",
                    return_value=2,
                ),
                patch(
                    "deepmd.pt_expt.train.validation.dist.all_gather_object",
                    side_effect=lambda out, obj: out.__setitem__(
                        slice(None), [obj, None]
                    ),
                ),
            ):
                result = validator.run(
                    step_id=7,
                    display_step=7,
                    lr=1e-3,
                    save_checkpoint=save_checkpoint,
                )

            self.assertIsNone(result)
            self.assertEqual(len(writes), 1)
            self.assertTrue(writes[0].exists())

    def test_reconcile_renames_before_deleting_stales(self) -> None:
        """Ranked renames must finish before any stale prune."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = FullValidator(
                validating_params={
                    "full_validation": True,
                    "validation_freq": 1,
                    "save_best": True,
                    "max_best_ckpt": 2,
                    "validation_metric": "E:MAE",
                    "full_val_file": str(Path(tmpdir) / "val.log"),
                    "full_val_start": 0.0,
                },
                validation_data=_DummyValidationData(),
                model=_DummyModel(),
                state_store={},
                num_steps=10,
                rank=0,
                restart_training=False,
                checkpoint_dir=Path(tmpdir),
            )
            # Previous top-K on disk: steps 10 (rank1) and 20 (rank2).
            Path(tmpdir, "best.ckpt-10.t-1.pt").write_text("10")
            Path(tmpdir, "best.ckpt-20.t-2.pt").write_text("20")
            # New top-K displaces 10; newly written 30 already at proposed rank.
            Path(tmpdir, "best.ckpt-30.t-2.pt").write_text("30")
            validator.topk_records = [
                BestCheckpointRecord(metric=1.0, step=20),
                BestCheckpointRecord(metric=1.5, step=30),
            ]
            validator._sync_state_store()

            events: list[str] = []
            real_rename = Path.rename
            real_remove = FullValidator._remove_checkpoint_path

            def tracking_rename(self_path: Path, target: Path) -> None:
                events.append(f"rename:{self_path.name}->{target.name}")
                real_rename(self_path, target)

            def tracking_remove(path: Path) -> None:
                events.append(f"remove:{path.name}")
                real_remove(path)

            with (
                patch.object(Path, "rename", tracking_rename),
                patch.object(
                    FullValidator,
                    "_remove_checkpoint_path",
                    staticmethod(tracking_remove),
                ),
            ):
                validator._reconcile_best_checkpoints()

            remove_stale_idxs = [
                i for i, e in enumerate(events) if e.startswith("remove:best.ckpt-10")
            ]
            final_rename_idxs = [
                i
                for i, e in enumerate(events)
                if e.startswith("rename:") and e.endswith(".tmp->best.ckpt-20.t-1.pt")
            ]
            self.assertTrue(final_rename_idxs, events)
            self.assertTrue(remove_stale_idxs, events)
            self.assertLess(max(final_rename_idxs), min(remove_stale_idxs), events)
            self.assertEqual(
                sorted(path.name for path in Path(tmpdir).glob("best.ckpt-*.pt")),
                ["best.ckpt-20.t-1.pt", "best.ckpt-30.t-2.pt"],
            )

    def test_prune_failure_does_not_restore_bookkeeping_naming_deleted_stales(
        self,
    ) -> None:
        """After ranked renames, prune failure must not roll top-K back to deleted files.

        Reproduces the pre-fix hazard: deleting ``best.ckpt-10…`` then failing
        left rollback restoring steps ``{10,20}`` while disk only had ``{20,30}``.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = FullValidator(
                validating_params={
                    "full_validation": True,
                    "validation_freq": 1,
                    "save_best": True,
                    "max_best_ckpt": 2,
                    "validation_metric": "E:MAE",
                    "full_val_file": str(Path(tmpdir) / "val.log"),
                    "full_val_start": 0.0,
                },
                validation_data=_DummyValidationData(),
                model=_DummyModel(),
                state_store={},
                num_steps=10,
                rank=0,
                restart_training=False,
                checkpoint_dir=Path(tmpdir),
            )
            old_records = [
                BestCheckpointRecord(metric=1.0, step=10),
                BestCheckpointRecord(metric=2.0, step=20),
            ]
            new_records = [
                BestCheckpointRecord(metric=1.0, step=20),
                BestCheckpointRecord(metric=1.5, step=30),
            ]
            Path(tmpdir, "best.ckpt-10.t-1.pt").write_text("10")
            Path(tmpdir, "best.ckpt-20.t-2.pt").write_text("20")
            Path(tmpdir, "best.ckpt-30.t-2.pt").write_text("30")
            validator._rollback_topk_records = list(old_records)
            validator.topk_records = list(new_records)
            validator._sync_state_store()

            real_remove = FullValidator._remove_checkpoint_path

            def fail_on_stale_step10(path: Path) -> None:
                if path.name.startswith("best.ckpt-10."):
                    # Delete first (as a real OS would), then fail — matching the
                    # mid-prune window the old ordering exposed to rollback.
                    real_remove(path)
                    raise OSError("simulated prune failure")
                real_remove(path)

            with patch.object(
                FullValidator,
                "_remove_checkpoint_path",
                staticmethod(fail_on_stale_step10),
            ):
                with self.assertRaises(OSError):
                    validator._reconcile_best_checkpoints()

            # Snapshot cleared after renames; caller rollback is a no-op.
            validator._rollback_pending_best_state()
            named_steps = {record.step for record in validator.topk_records}
            on_disk_steps = set()
            for path in Path(tmpdir).glob("best.ckpt-*.pt"):
                match = validator.best_checkpoint_pattern.match(path.name)
                self.assertIsNotNone(match, path.name)
                on_disk_steps.add(int(match.group(1)))
            self.assertEqual(named_steps, {20, 30})
            self.assertTrue(
                named_steps.issubset(on_disk_steps),
                f"bookkeeping {named_steps} missing on disk {on_disk_steps}",
            )
            self.assertEqual(
                sorted(path.name for path in Path(tmpdir).glob("best.ckpt-*.pt")),
                ["best.ckpt-20.t-1.pt", "best.ckpt-30.t-2.pt"],
            )
            self.assertIsNone(validator._rollback_topk_records)

    def test_rename_phase_failure_keeps_rollback_and_restores_temps(self) -> None:
        """Failure before ranked renames finish still allows bookkeeping rollback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = FullValidator(
                validating_params={
                    "full_validation": True,
                    "validation_freq": 1,
                    "save_best": True,
                    "max_best_ckpt": 2,
                    "validation_metric": "E:MAE",
                    "full_val_file": str(Path(tmpdir) / "val.log"),
                    "full_val_start": 0.0,
                },
                validation_data=_DummyValidationData(),
                model=_DummyModel(),
                state_store={},
                num_steps=10,
                rank=0,
                restart_training=False,
                checkpoint_dir=Path(tmpdir),
            )
            old_records = [
                BestCheckpointRecord(metric=1.0, step=10),
                BestCheckpointRecord(metric=2.0, step=20),
            ]
            new_records = [
                BestCheckpointRecord(metric=1.0, step=20),
                BestCheckpointRecord(metric=1.5, step=30),
            ]
            Path(tmpdir, "best.ckpt-10.t-1.pt").write_text("10")
            Path(tmpdir, "best.ckpt-20.t-2.pt").write_text("20")
            Path(tmpdir, "best.ckpt-30.t-2.pt").write_text("30")
            validator._rollback_topk_records = list(old_records)
            validator.topk_records = list(new_records)
            validator._sync_state_store()

            real_rename = Path.rename

            def fail_final_rename(self_path: Path, target: Path) -> None:
                # Fail only the ranked finalization, not the .tmp→original undo.
                if self_path.name.endswith(".tmp"):
                    original_name = self_path.name.removesuffix(".tmp")
                    if target.name != original_name:
                        raise OSError("simulated ranked rename failure")
                real_rename(self_path, target)

            with patch.object(Path, "rename", fail_final_rename):
                with self.assertRaises(OSError):
                    validator._reconcile_best_checkpoints()

            # Temps undone; stale 10 still present; rollback snapshot retained.
            self.assertIsNotNone(validator._rollback_topk_records)
            validator._rollback_pending_best_state()
            named_steps = {record.step for record in validator.topk_records}
            on_disk_steps = set()
            for path in Path(tmpdir).glob("best.ckpt-*.pt"):
                match = validator.best_checkpoint_pattern.match(path.name)
                self.assertIsNotNone(match, path.name)
                on_disk_steps.add(int(match.group(1)))
            self.assertEqual(named_steps, {10, 20})
            self.assertTrue(
                named_steps.issubset(on_disk_steps),
                f"bookkeeping {named_steps} missing on disk {on_disk_steps}",
            )
            self.assertFalse(list(Path(tmpdir).glob("*.tmp")))

    def test_commit_path_prune_failure_keeps_named_checkpoints(self) -> None:
        """run() commit/reconcile prune failure must not orphan named top-K steps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = FullValidator(
                validating_params={
                    "full_validation": True,
                    "validation_freq": 1,
                    "save_best": True,
                    "max_best_ckpt": 2,
                    "validation_metric": "E:MAE",
                    "full_val_file": str(Path(tmpdir) / "val.log"),
                    "full_val_start": 0.0,
                },
                validation_data=_DummyValidationData(),
                model=_DummyModel(),
                state_store={},
                num_steps=10,
                rank=0,
                restart_training=False,
                checkpoint_dir=Path(tmpdir),
            )

            def save_checkpoint(path: Path, lr: float = 0.0, step: int = 0) -> None:
                del lr, step
                Path(path).write_text(f"ckpt-{path.name}")

            # Seed previous best pair via a successful run, then force a third
            # candidate so reconcile must prune the displaced step.
            with patch.object(
                validator,
                "evaluate_all_systems",
                return_value={validator.metric_key: 2.0},
            ):
                validator.run(
                    step_id=10,
                    display_step=10,
                    lr=1e-3,
                    save_checkpoint=save_checkpoint,
                )
            with patch.object(
                validator,
                "evaluate_all_systems",
                return_value={validator.metric_key: 1.0},
            ):
                validator.run(
                    step_id=20,
                    display_step=20,
                    lr=1e-3,
                    save_checkpoint=save_checkpoint,
                )
            self.assertEqual(
                {record.step for record in validator.topk_records},
                {10, 20},
            )

            real_remove = FullValidator._remove_checkpoint_path

            def fail_on_stale_step10(path: Path) -> None:
                if path.name.startswith("best.ckpt-10."):
                    real_remove(path)
                    raise OSError("simulated prune failure")
                real_remove(path)

            with (
                patch.object(
                    validator,
                    "evaluate_all_systems",
                    return_value={validator.metric_key: 1.5},
                ),
                patch.object(
                    FullValidator,
                    "_remove_checkpoint_path",
                    staticmethod(fail_on_stale_step10),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "committing the best"):
                    validator.run(
                        step_id=30,
                        display_step=30,
                        lr=1e-3,
                        save_checkpoint=save_checkpoint,
                    )

            named_steps = {record.step for record in validator.topk_records}
            on_disk_steps = set()
            for path in Path(tmpdir).glob("best.ckpt-*.pt"):
                match = validator.best_checkpoint_pattern.match(path.name)
                self.assertIsNotNone(match, path.name)
                on_disk_steps.add(int(match.group(1)))
            self.assertEqual(named_steps, {20, 30})
            self.assertTrue(
                named_steps.issubset(on_disk_steps),
                f"bookkeeping {named_steps} missing on disk {on_disk_steps}",
            )
            self.assertIsNone(validator._rollback_topk_records)
            self.assertIsNone(validator._pending_topk_records)
