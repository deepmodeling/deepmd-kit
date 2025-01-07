# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from pathlib import (
    Path,
)

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
        with torch.device("cpu"):
            system_path = str(Path(__file__).parent / "mixed_type_data/sys.000000")
            cls.real_ntypes = 6
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
            weights_tensor = torch.tensor(
                [0.1] * len(cls.datasets), dtype=torch.float64, device="cpu"
            )
            sampler = torch.utils.data.WeightedRandomSampler(
                weights_tensor,
                num_samples=len(cls.datasets),
                replacement=True,
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
                    pin_memory=False,
                )
                cls.dataloaders.append(dataloader)

    def count_non_zero_elements(self, tensor, threshold=1e-8):
        return torch.sum(torch.abs(tensor) > threshold).item()

    def test_make_stat_input(self):
        # 3 frames would be count
        lst = make_stat_input(
            datasets=self.datasets,
            dataloaders=self.dataloaders,
            nbatches=1,
            min_frames_per_element_forstat=1,
            enable_element_completion=True,
        )
        bias, _ = compute_output_stats(lst, ntypes=57)
        energy = bias.get("energy")
        non_zero_count = self.count_non_zero_elements(energy)
        self.assertEqual(
            non_zero_count,
            self.real_ntypes,
            f"Expected exactly {self.real_ntypes} non-zero elements, but got {non_zero_count}.",
        )

    def test_make_stat_input_nocomplete(self):
        # missing element:13,31,37
        # only one frame would be count

        lst = make_stat_input(
            datasets=self.datasets,
            dataloaders=self.dataloaders,
            nbatches=1,
            min_frames_per_element_forstat=1,
            enable_element_completion=False,
        )
        bias, _ = compute_output_stats(lst, ntypes=57)
        energy = bias.get("energy")
        non_zero_count = self.count_non_zero_elements(energy)
        self.assertLess(
            non_zero_count,
            self.real_ntypes,
            f"Expected fewer than {self.real_ntypes} non-zero elements, but got {non_zero_count}.",
        )

    def test_bias(self):
        lst_ori = make_stat_input(
            datasets=self.datasets,
            dataloaders=self.dataloaders,
            nbatches=1,
            min_frames_per_element_forstat=1,
            enable_element_completion=False,
        )
        lst_all = make_stat_input(
            datasets=self.datasets,
            dataloaders=self.dataloaders,
            nbatches=1,
            min_frames_per_element_forstat=1,
            enable_element_completion=True,
        )
        bias_ori, _ = compute_output_stats(lst_ori, ntypes=57)
        bias_all, _ = compute_output_stats(lst_all, ntypes=57)
        energy_ori = np.array(bias_ori.get("energy").cpu()).flatten()
        energy_all = np.array(bias_all.get("energy").cpu()).flatten()

        for i, (e_ori, e_all) in enumerate(zip(energy_ori, energy_all)):
            if e_all == 0:
                self.assertEqual(
                    e_ori, 0, f"Index {i}: energy_all=0, but energy_ori={e_ori}"
                )
            else:
                if e_ori != 0:
                    diff = abs(e_ori - e_all)
                    rel_diff = diff / abs(e_ori)
                    self.assertTrue(
                        rel_diff < 0.4,
                        f"Index {i}: energy_ori={e_ori}, energy_all={e_all}, "
                        f"relative difference {rel_diff:.2%} is too large",
                    )
