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
    compute_output_stats,
    make_stat_input,
)
from deepmd.utils.data import (
    DataRequirementItem,
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
        cls.datasets = DeepmdDataSetForLoader(system=system_path)
        data_requirements = [
            DataRequirementItem(
                "energy",
                ndof=1,
                atomic=False,
            ),
        ]
        cls.datasets.add_data_requirement(data_requirements)
        cls.datasets = [cls.datasets]
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

    def count_non_zero_elements(self, tensor, threshold=1e-8):
        return torch.sum(torch.abs(tensor) > threshold).item()

    def test_make_stat_input(self):
        lst = make_stat_input(
            datasets=self.datasets,
            dataloaders=self.dataloaders,
            nbatches=1,
            min_frames_per_element_forstat=1,
            enable_element_completion=True,
        )
        bias, _ = compute_output_stats(lst, ntypes=57)
        energy = bias.get("energy")
        self.assertIsNotNone(energy, "'energy' key not found in bias dictionary.")
        non_zero_count = self.count_non_zero_elements(energy)
        self.assertEqual(
            non_zero_count,
            6,
            f"Expected exactly 7 non-zero elements, but got {non_zero_count}.",
        )

    def test_make_stat_input_nocomplete(self):
        lst = make_stat_input(
            datasets=self.datasets,
            dataloaders=self.dataloaders,
            nbatches=1,
            min_frames_per_element_forstat=1,
            enable_element_completion=False,
        )
        bias, _ = compute_output_stats(lst, ntypes=57)
        energy = bias.get("energy")
        self.assertIsNotNone(energy, "'energy' key not found in bias dictionary.")
        non_zero_count = self.count_non_zero_elements(energy)
        self.assertLess(
            non_zero_count,
            6,
            f"Expected fewer than 7 non-zero elements, but got {non_zero_count}.",
        )


if __name__ == "__main__":
    unittest.main()
