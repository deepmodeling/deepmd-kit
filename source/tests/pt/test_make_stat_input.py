# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import torch
from torch.utils.data import (
    DataLoader,
)

from deepmd.pt.utils.stat import (
    make_stat_input,
)


class TestDataset:
    def __init__(self, samples):
        self.samples = samples
        self.element_to_frames = {}
        for idx, sample in enumerate(samples):
            atypes = sample["atype"]
            for atype in atypes:
                if atype not in self.element_to_frames:
                    self.element_to_frames[atype] = []
                self.element_to_frames[atype].append(idx)

    @property
    def get_all_atype(self):
        return set(self.element_to_frames.keys())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "atype": torch.tensor(sample["atype"], dtype=torch.long),
            "energy": torch.tensor(sample["energy"], dtype=torch.float32),
        }


class TestMakeStatInput(unittest.TestCase):
    def setUp(self):
        self.system = TestDataset(
            [
                {"atype": [1], "energy": -1.0},
                {"atype": [2], "energy": -2.0},
            ]
        )
        self.datasets = [self.system]
        self.dataloaders = [
            DataLoader(self.system, batch_size=1, shuffle=False),
        ]

    def test_make_stat_input(self):
        nbatches = 1
        lst = make_stat_input(self.datasets, self.dataloaders, nbatches=nbatches)
        all_elements = self.system.get_all_atype
        unique_elements = {1, 2}
        self.assertEqual(unique_elements, all_elements, "make_stat_input miss elements")


if __name__ == "__main__":
    unittest.main()
