# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import os
import unittest
from collections.abc import (
    Iterator,
)
from pathlib import (
    Path,
)
from threading import (
    Event,
)

import numpy as np
import paddle
from paddle.io import (
    BatchSampler,
    DataLoader,
)

from deepmd.common import (
    expand_sys_str,
)
from deepmd.pd.utils.dataloader import (
    QUEUESIZE,
    BufferedIterator,
    DpLoaderSet,
    get_weighted_sampler,
)
from deepmd.tf.utils import random as tf_random
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)

CUR_DIR = os.path.dirname(__file__)


class TestBufferedIterator(unittest.TestCase):
    def test_close_stops_production_with_a_full_queue(self) -> None:
        full = Event()
        extra_read = Event()

        class Source:
            def __len__(self) -> int:
                return 10000

            def __iter__(self) -> Iterator[int]:
                for index in range(len(self)):
                    if index == QUEUESIZE + 1:
                        full.set()
                    if index == QUEUESIZE + 2:
                        extra_read.set()
                    yield index

        buffered = BufferedIterator(Source())
        self.addCleanup(buffered.close)
        self.assertEqual(next(buffered), 0)
        self.assertTrue(full.wait(timeout=5))
        self.assertTrue(buffered._queue.full())
        producer = buffered._consumer
        buffered.close()
        self.assertFalse(producer.is_alive())
        self.assertFalse(extra_read.is_set())
        self.assertTrue(buffered._queue.empty())
        with self.assertRaises(StopIteration):
            next(buffered)
        buffered.close()

    def test_exhaustion_releases_the_producer(self) -> None:
        buffered = BufferedIterator(range(3))
        self.assertEqual(list(buffered), [0, 1, 2])
        self.assertIsNone(buffered._consumer)
        self.assertIsNone(buffered._iterable)


class TestSampler(unittest.TestCase):
    def setUp(self):
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
        self.batch_size = config["training"]["training_data"]["batch_size"]
        self.systems = config["training"]["validation_data"]["systems"]
        if isinstance(self.systems, str):
            self.systems = expand_sys_str(self.systems)
        self.my_dataset = DpLoaderSet(
            self.systems,
            self.batch_size,
            model_config["type_map"],
            seed=10,
            shuffle=False,
        )

        tf_random.seed(10)
        self.dp_dataset = DeepmdDataSystem(self.systems, self.batch_size, 1, self.rcut)

    @unittest.skip("Temporarily skipped due to changes in numpy 2.4")
    def test_sampler_debug_info(self):
        dataloader = DataLoader(
            self.my_dataset,
            batch_sampler=BatchSampler(
                get_weighted_sampler(self.my_dataset, prob_style="prob_sys_size"),
                drop_last=False,
            ),
            num_workers=0,  # setting to 0 diverges the behavior of its iterator; should be >=1
            # pin_memory=True,
        )
        device = paddle.get_device()
        paddle.set_device("cpu")
        batch_data = next(iter(dataloader))
        paddle.set_device(device)
        sid = batch_data["sid"]
        fid = batch_data["fid"][0]
        coord = batch_data["coord"].squeeze(0)
        frame = self.my_dataset.systems[sid].__getitem__(fid)
        self.assertTrue(np.allclose(coord, frame["coord"]))

    def test_auto_prob_uniform(self):
        auto_prob_style = "prob_uniform"
        sampler = get_weighted_sampler(self.my_dataset, prob_style=auto_prob_style)
        my_probs = np.array(sampler.weights)
        self.dp_dataset.set_sys_probs(auto_prob_style=auto_prob_style)
        dp_probs = np.array(self.dp_dataset.sys_probs)
        self.assertTrue(np.allclose(my_probs, dp_probs))

    def test_auto_prob_sys_size(self):
        auto_prob_style = "prob_sys_size"
        sampler = get_weighted_sampler(self.my_dataset, prob_style=auto_prob_style)
        my_probs = np.array(sampler.weights)
        self.dp_dataset.set_sys_probs(auto_prob_style=auto_prob_style)
        dp_probs = np.array(self.dp_dataset.sys_probs)
        self.assertTrue(np.allclose(my_probs, dp_probs))

    def test_auto_prob_sys_size_ext(self):
        auto_prob_style = "prob_sys_size;0:1:0.2;1:3:0.8"
        sampler = get_weighted_sampler(self.my_dataset, prob_style=auto_prob_style)
        my_probs = np.array(sampler.weights)
        self.dp_dataset.set_sys_probs(auto_prob_style=auto_prob_style)
        dp_probs = np.array(self.dp_dataset.sys_probs)
        self.assertTrue(np.allclose(my_probs, dp_probs))

    def test_sys_probs(self):
        sys_probs = [0.1, 0.4, 0.5]
        sampler = get_weighted_sampler(
            self.my_dataset, prob_style=sys_probs, sys_prob=True
        )
        my_probs = np.array(sampler.weights)
        self.dp_dataset.set_sys_probs(sys_probs=sys_probs)
        dp_probs = np.array(self.dp_dataset.sys_probs)
        self.assertTrue(np.allclose(my_probs, dp_probs))


if __name__ == "__main__":
    unittest.main()
