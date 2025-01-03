# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
)

from deepmd.pt.utils.dataset import (
    DeepmdDataSetForLoader,
)
from deepmd.pt.utils.stat import (
    make_stat_input,
)


def collate_fn(batch):
    if isinstance(batch, dict):
        batch = [batch]
    collated_batch = {}
    for key in batch[0].keys():
        data_list = [d[key] for d in batch]
        if isinstance(data_list[0], np.ndarray):
            data_np = np.stack(data_list)
            collated_batch[key] = torch.from_numpy(data_np)
        else:
            collated_batch[key] = torch.tensor(data_list)
    return collated_batch


class TestMakeStatInput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        system_path = "mixed_type_data/sys.000000"
        cls.alltype = {19, 6, 17, 12, 30, 36}
        cls.datasets = [DeepmdDataSetForLoader(system=system_path)]
        weights = torch.tensor([0.1] * len(cls.datasets))
        sampler = torch.utils.data.WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )
        cls.dataloaders = []
        for dataset in cls.datasets:
            dataloader = DataLoader(
                dataset,
                sampler=sampler,
                batch_size=1,
                num_workers=0,
                drop_last=False,
                collate_fn=collate_fn,
                pin_memory=True,
            )
            cls.dataloaders.append(dataloader)

    def test_make_stat_input(self):
        nbatches = 1
        lst = make_stat_input(
            datasets=self.datasets,
            dataloaders=self.dataloaders,
            nbatches=nbatches,
            min_frames_per_element_forstat=1,
            enable_element_completion=True,
        )
        coll_ele = set()
        for i in lst:
            ele = np.unique(i["atype"].cpu().numpy())
            coll_ele.update(ele)
        if not coll_ele == self.alltype:
            self.assertFalse("Wrong")

    def test_make_stat_input_nocomplete(self):
        nbatches = 1
        lst = make_stat_input(
            datasets=self.datasets,
            dataloaders=self.dataloaders,
            nbatches=nbatches,
            min_frames_per_element_forstat=1,
            enable_element_completion=False,
        )
        coll_ele = set()
        for i in lst:
            ele = np.unique(i["atype"].cpu().numpy())
            coll_ele.update(ele)
        if coll_ele == self.alltype:
            self.assertFalse("Wrong")


if __name__ == "__main__":
    unittest.main()
