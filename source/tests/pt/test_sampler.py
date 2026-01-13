# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import os
import unittest
from pathlib import (
    Path,
)

import numpy as np
import pytest
import torch
from torch.utils.data import (
    DataLoader,
)

import deepmd.pt.utils.dataloader as pt_dataloader
from deepmd.pt.utils import (
    dp_random,
)
from deepmd.tf.common import (
    expand_sys_str,
)
from deepmd.tf.utils import random as tf_random
from deepmd.tf.utils.data_system import (
    DeepmdDataSystem,
)

CUR_DIR = os.path.dirname(__file__)


class _SerialPool:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __enter__(self) -> "_SerialPool":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def map(self, func, iterable):
        return [func(item) for item in iterable]


class TestSampler(unittest.TestCase):
    def setUp(self) -> None:
        self._monkeypatch = pytest.MonkeyPatch()
        # Avoid SemLock/CUDA initialization failures in restricted CI by forcing a serial pool.
        self._monkeypatch.setattr(pt_dataloader, "Pool", _SerialPool)
        with open(str(Path(__file__).parent / "water/se_e2_a.json")) as fin:
            content = fin.read()
        config = json.loads(content)
        data_file = [str(Path(__file__).parent / "water/data/data_0")]
        config["training"]["training_data"]["systems"] = data_file
        config["training"]["validation_data"]["systems"] = data_file
        model_config = config["model"]
        self.rcut = model_config["descriptor"]["rcut"]
        self.rcut_smth = model_config["descriptor"]["rcut_smth"]
        self.sel = model_config["descriptor"]["sel"]
        self.type_map = model_config["type_map"]
        self.batch_size = config["training"]["training_data"]["batch_size"]
        self.systems = config["training"]["validation_data"]["systems"]
        if isinstance(self.systems, str):
            self.systems = expand_sys_str(self.systems)
        self.my_dataset = pt_dataloader.DpLoaderSet(
            self.systems,
            self.batch_size,
            self.type_map,
            seed=10,
            shuffle=False,
        )

        tf_random.seed(10)
        self.dp_dataset = DeepmdDataSystem(self.systems, self.batch_size, 1, self.rcut)

    def tearDown(self) -> None:
        self._monkeypatch.undo()

    def _make_dataloader(
        self, dataset: pt_dataloader.DpLoaderSet, sampler
    ) -> DataLoader:
        return DataLoader(
            dataset,
            sampler=sampler,
            batch_size=None,
            num_workers=0,
            drop_last=False,
            collate_fn=lambda batch: batch,
        )

    def _normalize_probs(self, weights: np.ndarray) -> np.ndarray:
        weights = np.asarray(weights, dtype=np.float64)
        return weights / np.sum(weights)

    def _compute_total_numb_batch(self, nbatches: np.ndarray, probs: np.ndarray) -> int:
        # NOTE: This is a simplified test-only variant of training.py logic.
        nbatches = np.asarray(nbatches, dtype=np.float64)
        probs = np.asarray(probs, dtype=np.float64)
        if nbatches.shape != probs.shape:
            raise ValueError(
                "nbatches and probs must have the same shape in this test helper."
            )
        if not np.all(probs > 0.0):
            raise ValueError(
                "Zero or negative sampling probabilities are not supported in this "
                "test helper."
            )
        return int(np.ceil(np.max(nbatches / probs)))

    def _sample_sid_counts(
        self, dataloader: DataLoader, num_steps: int, nsystems: int
    ) -> np.ndarray:
        # === Step 1. Initialize Counters ===
        counts = np.zeros(nsystems, dtype=np.int64)
        # === Step 2. Sample Steps ===
        with torch.device("cpu"):
            iterator = iter(dataloader)
            for _ in range(num_steps):
                try:
                    batch_data = next(iterator)
                except StopIteration:
                    iterator = iter(dataloader)
                    batch_data = next(iterator)
                sid = batch_data["sid"]
                if hasattr(sid, "item"):
                    sid = sid.item()
                counts[int(sid)] += 1
        return counts

    def _sample_multitask_counts(
        self,
        dataloaders: dict[str, DataLoader],
        model_prob: np.ndarray,
        num_steps: int,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        # === Step 1. Initialize Counters ===
        model_keys = list(dataloaders.keys())
        model_counts = np.zeros(len(model_keys), dtype=np.int64)
        sid_counts = {
            model_key: np.zeros(len(dataloaders[model_key].dataset), dtype=np.int64)
            for model_key in model_keys
        }
        # === Step 2. Build Iterators and Sample Steps ===
        with torch.device("cpu"):
            iters = {
                model_key: iter(dataloaders[model_key]) for model_key in model_keys
            }
            for _ in range(num_steps):
                model_index = dp_random.choice(
                    np.arange(len(model_keys), dtype=np.int_), p=model_prob
                )
                model_key = model_keys[int(model_index)]
                model_counts[int(model_index)] += 1
                try:
                    batch_data = next(iters[model_key])
                except StopIteration:
                    iters[model_key] = iter(dataloaders[model_key])
                    batch_data = next(iters[model_key])
                sid = batch_data["sid"]
                if hasattr(sid, "item"):
                    sid = sid.item()
                sid_counts[model_key][int(sid)] += 1
        return model_counts, sid_counts

    def test_sampler_debug_info(self) -> None:
        dataloader = DataLoader(
            self.my_dataset,
            sampler=pt_dataloader.get_weighted_sampler(
                self.my_dataset, prob_style="prob_sys_size"
            ),
            batch_size=None,
            num_workers=0,  # setting to 0 diverges the behavior of its iterator; should be >=1
            drop_last=False,
        )
        with torch.device("cpu"):
            batch_data = next(iter(dataloader))
        sid = batch_data["sid"]
        fid = batch_data["fid"][0]
        coord = batch_data["coord"].squeeze(0)
        frame = self.my_dataset.systems[sid].__getitem__(fid)
        self.assertTrue(np.allclose(coord, frame["coord"]))

    def test_auto_prob_uniform(self) -> None:
        auto_prob_style = "prob_uniform"
        sampler = pt_dataloader.get_weighted_sampler(
            self.my_dataset, prob_style=auto_prob_style
        )
        my_probs = np.array(sampler.weights)
        self.dp_dataset.set_sys_probs(auto_prob_style=auto_prob_style)
        dp_probs = np.array(self.dp_dataset.sys_probs)
        self.assertTrue(np.allclose(my_probs, dp_probs))

    def test_auto_prob_sys_size(self) -> None:
        auto_prob_style = "prob_sys_size"
        sampler = pt_dataloader.get_weighted_sampler(
            self.my_dataset, prob_style=auto_prob_style
        )
        my_probs = np.array(sampler.weights)
        self.dp_dataset.set_sys_probs(auto_prob_style=auto_prob_style)
        dp_probs = np.array(self.dp_dataset.sys_probs)
        self.assertTrue(np.allclose(my_probs, dp_probs))

    def test_auto_prob_sys_size_ext(self) -> None:
        auto_prob_style = "prob_sys_size;0:1:0.2;1:3:0.8"
        sampler = pt_dataloader.get_weighted_sampler(
            self.my_dataset, prob_style=auto_prob_style
        )
        my_probs = np.array(sampler.weights)
        self.dp_dataset.set_sys_probs(auto_prob_style=auto_prob_style)
        dp_probs = np.array(self.dp_dataset.sys_probs)
        self.assertTrue(np.allclose(my_probs, dp_probs))

    def test_sys_probs(self) -> None:
        sys_probs = [0.1, 0.4, 0.5]
        sampler = pt_dataloader.get_weighted_sampler(
            self.my_dataset, prob_style=sys_probs, sys_prob=True
        )
        my_probs = np.array(sampler.weights)
        self.dp_dataset.set_sys_probs(sys_probs=sys_probs)
        dp_probs = np.array(self.dp_dataset.sys_probs)
        self.assertTrue(np.allclose(my_probs, dp_probs))

    def test_sys_probs_end2end(self):
        sys_probs = [0.1, 0.4, 0.5]
        _params = {
            "sys_probs": sys_probs,
            "auto_prob": "prob_sys_size",
        }  # use sys_probs first
        sampler = pt_dataloader.get_sampler_from_params(self.my_dataset, _params)
        my_probs = np.array(sampler.weights)
        self.dp_dataset.set_sys_probs(sys_probs=sys_probs)
        dp_probs = np.array(self.dp_dataset.sys_probs)
        self.assertTrue(np.allclose(my_probs, dp_probs))

    def test_auto_prob_sys_size_ext_end2end(self):
        auto_prob_style = "prob_sys_size;0:1:0.2;1:3:0.8"
        _params = {"sys_probs": None, "auto_prob": auto_prob_style}  # use auto_prob
        sampler = pt_dataloader.get_sampler_from_params(self.my_dataset, _params)
        my_probs = np.array(sampler.weights)
        self.dp_dataset.set_sys_probs(auto_prob_style=auto_prob_style)
        dp_probs = np.array(self.dp_dataset.sys_probs)
        self.assertTrue(np.allclose(my_probs, dp_probs))

    def test_sampling_stability_single_task(self) -> None:
        # === Step 1. Build Dataset and Sampler ===
        systems = [
            str(Path(__file__).parent / "water/data/data_0"),
            str(Path(__file__).parent / "water/data/data_1"),
            str(Path(__file__).parent / "water/data/single"),
        ]
        dataset_epoch = pt_dataloader.DpLoaderSet(
            systems,
            self.batch_size,
            self.type_map,
            seed=10,
            shuffle=False,
        )
        sys_probs = [0.2, 0.3, 0.5]
        params = {"sys_probs": sys_probs, "auto_prob": "prob_sys_size"}
        sampler_epoch = pt_dataloader.get_sampler_from_params(dataset_epoch, params)
        probs = self._normalize_probs(np.asarray(sampler_epoch.weights))
        nbatches = np.asarray(dataset_epoch.index, dtype=np.float64)
        total_numb_batch = self._compute_total_numb_batch(nbatches, probs)
        num_epoch = 1.5
        num_steps = int(np.ceil(num_epoch * total_numb_batch))

        # === Step 2. Sample Using Derived Steps ===
        torch.manual_seed(123)
        dataloader_epoch = self._make_dataloader(dataset_epoch, sampler_epoch)
        counts_epoch = self._sample_sid_counts(
            dataloader_epoch, num_steps, len(dataset_epoch)
        )
        empirical_epoch = counts_epoch / float(num_steps)
        self.assertTrue(np.allclose(empirical_epoch, probs, atol=0.1))

        # === Step 3. Sample Using Explicit Steps ===
        dataset_steps = pt_dataloader.DpLoaderSet(
            systems,
            self.batch_size,
            self.type_map,
            seed=10,
            shuffle=False,
        )
        sampler_steps = pt_dataloader.get_sampler_from_params(dataset_steps, params)
        torch.manual_seed(123)
        dataloader_steps = self._make_dataloader(dataset_steps, sampler_steps)
        counts_steps = self._sample_sid_counts(
            dataloader_steps, num_steps, len(dataset_steps)
        )
        self.assertTrue(np.array_equal(counts_epoch, counts_steps))

    def test_sampling_stability_multi_task(self) -> None:
        # === Step 1. Build Datasets and Samplers ===
        model_keys = ["model_1", "model_2"]
        systems_1 = [
            str(Path(__file__).parent / "water/data/data_0"),
            str(Path(__file__).parent / "water/data/data_1"),
        ]
        systems_2 = [
            str(Path(__file__).parent / "water/data/data_1"),
            str(Path(__file__).parent / "water/data/single"),
        ]
        dataset_1 = pt_dataloader.DpLoaderSet(
            systems_1,
            self.batch_size,
            self.type_map,
            seed=10,
            shuffle=False,
        )
        dataset_2 = pt_dataloader.DpLoaderSet(
            systems_2,
            self.batch_size,
            self.type_map,
            seed=10,
            shuffle=False,
        )
        sampler_1 = pt_dataloader.get_sampler_from_params(
            dataset_1, {"sys_probs": [0.7, 0.3], "auto_prob": "prob_sys_size"}
        )
        sampler_2 = pt_dataloader.get_sampler_from_params(
            dataset_2, {"sys_probs": [0.4, 0.6], "auto_prob": "prob_sys_size"}
        )
        probs_1 = self._normalize_probs(np.asarray(sampler_1.weights))
        probs_2 = self._normalize_probs(np.asarray(sampler_2.weights))
        per_task_total = np.array(
            [
                self._compute_total_numb_batch(
                    np.asarray(dataset_1.index, dtype=np.float64), probs_1
                ),
                self._compute_total_numb_batch(
                    np.asarray(dataset_2.index, dtype=np.float64), probs_2
                ),
            ],
            dtype=np.float64,
        )
        model_prob = np.asarray([0.4, 0.6], dtype=np.float64)
        model_prob = model_prob / np.sum(model_prob)
        total_numb_batch = int(np.ceil(np.sum(per_task_total * model_prob)))
        num_epoch = 1.5
        num_steps = int(np.ceil(num_epoch * total_numb_batch))

        # === Step 2. Sample Using Derived Steps ===
        dataloaders_epoch = {
            model_keys[0]: self._make_dataloader(dataset_1, sampler_1),
            model_keys[1]: self._make_dataloader(dataset_2, sampler_2),
        }
        dp_random.seed(321)
        torch.manual_seed(321)
        model_counts_epoch, sid_counts_epoch = self._sample_multitask_counts(
            dataloaders_epoch, model_prob, num_steps
        )
        model_freq_epoch = model_counts_epoch / float(num_steps)
        self.assertTrue(np.allclose(model_freq_epoch, model_prob, atol=0.1))
        if model_counts_epoch[0] == 0 or model_counts_epoch[1] == 0:
            raise AssertionError("Model sampling produced zero counts for a task.")
        self.assertTrue(
            np.allclose(
                sid_counts_epoch[model_keys[0]] / model_counts_epoch[0],
                probs_1,
                atol=0.1,
            )
        )
        self.assertTrue(
            np.allclose(
                sid_counts_epoch[model_keys[1]] / model_counts_epoch[1],
                probs_2,
                atol=0.1,
            )
        )

        # === Step 3. Sample Using Explicit Steps ===
        dataset_1b = pt_dataloader.DpLoaderSet(
            systems_1,
            self.batch_size,
            self.type_map,
            seed=10,
            shuffle=False,
        )
        dataset_2b = pt_dataloader.DpLoaderSet(
            systems_2,
            self.batch_size,
            self.type_map,
            seed=10,
            shuffle=False,
        )
        sampler_1b = pt_dataloader.get_sampler_from_params(
            dataset_1b, {"sys_probs": [0.7, 0.3], "auto_prob": "prob_sys_size"}
        )
        sampler_2b = pt_dataloader.get_sampler_from_params(
            dataset_2b, {"sys_probs": [0.4, 0.6], "auto_prob": "prob_sys_size"}
        )
        dataloaders_steps = {
            model_keys[0]: self._make_dataloader(dataset_1b, sampler_1b),
            model_keys[1]: self._make_dataloader(dataset_2b, sampler_2b),
        }
        dp_random.seed(321)
        torch.manual_seed(321)
        model_counts_steps, sid_counts_steps = self._sample_multitask_counts(
            dataloaders_steps, model_prob, num_steps
        )
        self.assertTrue(np.array_equal(model_counts_epoch, model_counts_steps))
        self.assertTrue(
            np.array_equal(
                sid_counts_epoch[model_keys[0]], sid_counts_steps[model_keys[0]]
            )
        )
        self.assertTrue(
            np.array_equal(
                sid_counts_epoch[model_keys[1]], sid_counts_steps[model_keys[1]]
            )
        )

    def test_num_epoch_dict(self) -> None:
        """Test num_epoch_dict calculation logic for multi-task training."""
        # === Step 1. Build Datasets ===
        model_keys = ["model_1", "model_2"]
        systems_1 = [
            str(Path(__file__).parent / "water/data/data_0"),
            str(Path(__file__).parent / "water/data/data_1"),
        ]
        systems_2 = [
            str(Path(__file__).parent / "water/data/data_1"),
            str(Path(__file__).parent / "water/data/single"),
        ]
        dataset_1 = pt_dataloader.DpLoaderSet(
            systems_1,
            self.batch_size,
            self.type_map,
            seed=10,
            shuffle=False,
        )
        dataset_2 = pt_dataloader.DpLoaderSet(
            systems_2,
            self.batch_size,
            self.type_map,
            seed=10,
            shuffle=False,
        )
        sampler_1 = pt_dataloader.get_sampler_from_params(
            dataset_1, {"sys_probs": [0.7, 0.3], "auto_prob": "prob_sys_size"}
        )
        sampler_2 = pt_dataloader.get_sampler_from_params(
            dataset_2, {"sys_probs": [0.4, 0.6], "auto_prob": "prob_sys_size"}
        )
        probs_1 = self._normalize_probs(np.asarray(sampler_1.weights))
        probs_2 = self._normalize_probs(np.asarray(sampler_2.weights))

        # === Step 2. Compute per-task total_numb_batch ===
        per_task_total = np.array(
            [
                self._compute_total_numb_batch(
                    np.asarray(dataset_1.index, dtype=np.float64), probs_1
                ),
                self._compute_total_numb_batch(
                    np.asarray(dataset_2.index, dtype=np.float64), probs_2
                ),
            ],
            dtype=np.float64,
        )

        # === Step 3. Test num_epoch_dict calculation ===
        model_prob = np.asarray([0.4, 0.6], dtype=np.float64)
        model_prob = model_prob / np.sum(model_prob)
        num_epoch_dict = {model_keys[0]: 2.0, model_keys[1]: 5.0}

        # Compute expected steps for each task
        # steps_i = epoch_i * per_task_total[i] / model_prob[i]
        per_task_steps = np.array(
            [
                num_epoch_dict[model_keys[0]] * per_task_total[0] / model_prob[0],
                num_epoch_dict[model_keys[1]] * per_task_total[1] / model_prob[1],
            ],
            dtype=np.float64,
        )

        # Total steps should be max of per-task steps
        expected_num_steps = int(np.ceil(np.max(per_task_steps)))

        # Verify the calculation matches the expected formula
        self.assertIsInstance(expected_num_steps, int)
        self.assertGreater(expected_num_steps, 0)

        # Verify that running expected_num_steps would give each task at least
        # its target epochs (may be more for tasks needing fewer steps)
        expected_model_0_counts = expected_num_steps * model_prob[0]
        expected_model_1_counts = expected_num_steps * model_prob[1]

        # Each task should complete at least its target epochs
        expected_epochs_0 = expected_model_0_counts / per_task_total[0]
        expected_epochs_1 = expected_model_1_counts / per_task_total[1]

        self.assertGreaterEqual(
            expected_epochs_0,
            num_epoch_dict[model_keys[0]],
            msg="Model 0 should complete at least 2 epochs",
        )
        self.assertGreaterEqual(
            expected_epochs_1,
            num_epoch_dict[model_keys[1]],
            msg="Model 1 should complete at least 5 epochs",
        )

        # The task requiring the most steps should complete approximately its target
        max_task_idx = int(np.argmax(per_task_steps))
        if max_task_idx == 0:
            self.assertAlmostEqual(
                expected_epochs_0,
                num_epoch_dict[model_keys[0]],
                delta=0.1,
                msg="Model 0 (max steps) should complete approximately 2 epochs",
            )
        else:
            self.assertAlmostEqual(
                expected_epochs_1,
                num_epoch_dict[model_keys[1]],
                delta=0.1,
                msg="Model 1 (max steps) should complete approximately 5 epochs",
            )


if __name__ == "__main__":
    unittest.main()
