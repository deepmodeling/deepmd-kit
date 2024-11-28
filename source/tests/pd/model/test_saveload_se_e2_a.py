# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import json
import os
import unittest
from pathlib import (
    Path,
)

import numpy as np
import paddle
from paddle.io import (
    DataLoader,
)

from deepmd.pd.loss import (
    EnergyStdLoss,
)
from deepmd.pd.model.model import (
    get_model,
)
from deepmd.pd.train.wrapper import (
    ModelWrapper,
)
from deepmd.pd.utils import (
    env,
)
from deepmd.pd.utils.dataloader import (
    BufferedIterator,
    DpLoaderSet,
)
from deepmd.pd.utils.stat import (
    make_stat_input,
)
from deepmd.tf.common import (
    expand_sys_str,
)


def get_dataset(config):
    model_config = config["model"]
    rcut = model_config["descriptor"]["rcut"]
    sel = model_config["descriptor"]["sel"]
    systems = config["training"]["validation_data"]["systems"]
    if isinstance(systems, str):
        systems = expand_sys_str(systems)
    batch_size = config["training"]["training_data"]["batch_size"]
    type_map = model_config["type_map"]

    dataset = DpLoaderSet(systems, batch_size, type_map)
    data_stat_nbatch = model_config.get("data_stat_nbatch", 10)
    sampled = make_stat_input(dataset.systems, dataset.dataloaders, data_stat_nbatch)
    return dataset, sampled


class TestSaveLoadSeA(unittest.TestCase):
    def setUp(self):
        input_json = str(Path(__file__).parent / "water/se_e2_a.json")
        with open(input_json) as fin:
            self.config = json.load(fin)
        data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file
        self.config["loss"]["starter_learning_rate"] = self.config["learning_rate"][
            "start_lr"
        ]
        self.dataset, self.sampled = get_dataset(self.config)
        self.training_dataloader = DataLoader(
            self.dataset,
            batch_sampler=paddle.io.BatchSampler(
                sampler=paddle.io.RandomSampler(self.dataset),
                drop_last=False,
            ),
            num_workers=0,  # setting to 0 diverges the behavior of its iterator; should be >=1
            collate_fn=lambda batch: batch[0],
        )
        device = paddle.get_device()
        paddle.set_device("cpu")
        self.training_data = BufferedIterator(iter(self.training_dataloader))
        paddle.set_device(device)
        self.loss = EnergyStdLoss(**self.config["loss"])
        self.cur_lr = 1
        self.task_key = "Default"
        self.input_dict, self.label_dict = self.get_data()
        self.start_lr = self.config["learning_rate"]["start_lr"]

    def get_model_result(self, read=False, model_file="tmp_model.pd"):
        wrapper = self.create_wrapper()
        optimizer = paddle.optimizer.Adam(
            learning_rate=self.start_lr, parameters=wrapper.parameters()
        )
        optimizer.clear_grad()
        if read:
            wrapper.set_state_dict(paddle.load(model_file))
            os.remove(model_file)
        else:
            paddle.save(wrapper.state_dict(), model_file)
        result = wrapper(
            **self.input_dict,
            cur_lr=self.cur_lr,
            label=self.label_dict,
            task_key=self.task_key,
        )[0]
        return result

    def create_wrapper(self):
        model_config = copy.deepcopy(self.config["model"])
        model = get_model(model_config).to(env.DEVICE)
        return ModelWrapper(model, self.loss)

    def get_data(self):
        try:
            batch_data = next(iter(self.training_data))
        except StopIteration:
            # Refresh the status of the dataloader to start from a new epoch
            self.training_data = BufferedIterator(iter(self.training_dataloader))
            batch_data = next(iter(self.training_data))
        input_dict = {}
        for item in ["coord", "atype", "box"]:
            if item in batch_data:
                input_dict[item] = batch_data[item].to(env.DEVICE)
            else:
                input_dict[item] = None
        label_dict = {}
        for item in ["energy", "force", "virial"]:
            if item in batch_data:
                label_dict[item] = batch_data[item].to(env.DEVICE)
        return input_dict, label_dict

    def test_saveload(self):
        result1 = self.get_model_result()
        result2 = self.get_model_result(read=True)
        for item in result1:
            np.testing.assert_allclose(result1[item].numpy(), result2[item].numpy())


if __name__ == "__main__":
    unittest.main()
