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

    out = {}
    for key in batch[0].keys():
        items = [sample[key] for sample in batch]

        if isinstance(items[0], torch.Tensor):
            out[key] = torch.stack(items, dim=0)
        elif isinstance(items[0], np.ndarray):
            out[key] = torch.from_numpy(np.stack(items, axis=0))
        else:
            try:
                out[key] = torch.tensor(items)
            except Exception:
                out[key] = items

    return out


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
            cls.dataloaders = []
            for dataset in cls.datasets:
                dataloader = DataLoader(
                    dataset,
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

    def test_with_nomissing(self):
        lst_ori = make_stat_input(
            datasets=self.datasets,
            dataloaders=self.dataloaders,
            nbatches=10,
            min_frames_per_element_forstat=1,
            enable_element_completion=False,
        )
        for dct in lst_ori:
            for key in ["find_box", "find_coord", "find_numb_copy", "find_energy"]:
                if key in dct:
                    val = dct[key]
                    if val.numel() > 1:
                        dct[key] = val[0]
        lst_new = make_stat_input(
            datasets=self.datasets,
            dataloaders=self.dataloaders,
            nbatches=10,
            min_frames_per_element_forstat=1,
            enable_element_completion=True,
        )
        for dct in lst_new:
            for key in ["find_box", "find_coord", "find_numb_copy", "find_energy"]:
                if key in dct:
                    val = dct[key]
                    if val.numel() > 1:
                        dct[key] = val[0]
        bias_ori, _ = compute_output_stats(lst_ori, ntypes=57)
        bias_new, _ = compute_output_stats(lst_new, ntypes=57)
        energy_ori = np.array(bias_ori.get("energy").cpu()).flatten()
        energy_new = np.array(bias_new.get("energy").cpu()).flatten()
        self.assertTrue(
            np.array_equal(energy_ori, energy_new),
            msg=f"energy_ori and energy_new are not exactly the same!\n"
            f"energy_ori = {energy_ori}\nenergy_new = {energy_new}",
        )
