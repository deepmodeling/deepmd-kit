# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
import torch
from torch.utils.data import DataLoader
from deepmd.pt.utils.stat import make_stat_input
import numpy as np
import os
import glob
from collections import defaultdict

class TestDataset:
    def __init__(self, samples):
        self.samples = samples
        self.element_to_frames = defaultdict(list)
        self.mixed_type = True
        for idx, sample in enumerate(samples):
            atypes = sample["atype"]
            for atype in atypes:
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

    def true_types(self):
        element_counts = defaultdict(lambda: {"count": 0, "frames": 0})
        for idx, sample in enumerate(self.samples):
            atypes = sample["atype"]
            unique_atypes = set(atypes)
            for atype in atypes:
                element_counts[atype]["count"] += 1 
            for atype in unique_atypes:
                element_counts[atype]["frames"] += 1 
        return dict(element_counts)


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
        lst = make_stat_input(
            self.datasets,
            self.dataloaders,
            nbatches=nbatches,
            min_frames_per_element_forstat=1,
        )
        all_elements = self.system.get_all_atype
        unique_elements = {1, 2}
        self.assertEqual(unique_elements, all_elements, "make_stat_input miss elements")

        expected_true_types = {
            1: {"count": 1, "frames": 1},  
            2: {"count": 1, "frames": 1},  
        }
        actual_true_types = self.system.true_types()
        self.assertEqual(
            expected_true_types, actual_true_types, "true_types is wrong"
        )


if __name__ == "__main__":
    unittest.main()
