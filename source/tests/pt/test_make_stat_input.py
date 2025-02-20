# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from collections import (
    defaultdict,
)
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
    finalize_stats,
    make_stat_input,
    process_batches,
    process_element_counts,
    process_missing_elements,
    process_with_new_frame,
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

    def test_make_stat_input_with_element_counts(self):
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
            f"Expected exactly {self.real_ntypes} non-zero elements in energy, but got {non_zero_count}.",
        )

    def test_with_missing_elements_and_new_frames(self):
        """
        Test handling missing elements and processing new frames.
        Verify if the system is correctly processing new frames to compensate for missing elements.
        Ensure that the statistics for energy have been processed correctly and missing elements are completed.
        """
        lst = make_stat_input(
            datasets=self.datasets,
            dataloaders=self.dataloaders,
            nbatches=1,
            min_frames_per_element_forstat=1,
            enable_element_completion=False,
        )

        missing_elements = []
        for sys_stat in lst:
            if "energy" in sys_stat:
                energy = sys_stat["energy"]
                missing_elements.append(self.count_non_zero_elements(energy))

        self.assertGreater(
            len(missing_elements), 0, "Expected missing elements to be processed."
        )

        lst_new = make_stat_input(
            datasets=self.datasets,
            dataloaders=self.dataloaders,
            nbatches=1,
            min_frames_per_element_forstat=1,
            enable_element_completion=True,
        )

        for original, new in zip(lst, lst_new):
            energy_ori = np.array(original["energy"].cpu()).flatten()
            energy_new = np.array(new["energy"].cpu()).flatten()
            self.assertTrue(
                np.allclose(energy_ori, energy_new),
                msg=f"Energy values don't match. Original: {energy_ori}, New: {energy_new}",
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
        # missing element:13,31,37
        # only one frame would be count
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

    def test_process_batches(self):
        with torch.device("cpu"):
            dataloader = self.dataloaders[0]
            sys_stat = {}
            process_batches(dataloader, sys_stat, nbatches=1)
            self.assertTrue("energy" in sys_stat, "Expected 'energy' to be in sys_stat")
            self.assertGreater(
                len(sys_stat["energy"]), 0, "Expected non-zero 'energy' values"
            )

    def test_finalize_stats(self):
        with torch.device("cpu"):
            sys_stat1 = {"param1": np.float32(1.23)}
            finalize_stats(sys_stat1)
            assert sys_stat1["param1"] == np.float32(1.23), "Test Case 1 Failed"

            sys_stat2 = {
                "param2": None,
                "param3": [],
                "param4": [None],
            }
            finalize_stats(sys_stat2)
            assert sys_stat2["param2"] is None, "Test Case 2a Failed"
            assert sys_stat2["param3"] is None, "Test Case 2b Failed"
            assert sys_stat2["param4"] is None, "Test Case 2c Failed"

            tensor1 = torch.tensor([1.0, 2.0])
            tensor2 = torch.tensor([3.0, 4.0])
            sys_stat3 = {
                "param5": [tensor1, tensor2],
            }
            finalize_stats(sys_stat3)
            assert torch.equal(
                sys_stat3["param5"], torch.tensor([1.0, 2.0, 3.0, 4.0])
            ), "Test Case 3 Failed"

    def test_process_element_counts(self):
        dataset = self.datasets[0]
        global_element_counts = {}
        global_type_name = {}
        total_element_types = set()
        process_element_counts(
            0,
            dataset,
            min_frames=1,
            global_element_counts=global_element_counts,
            global_type_name=global_type_name,
            total_element_types=total_element_types,
        )
        self.assertGreater(
            len(global_element_counts),
            0,
            "Expected global_element_counts to contain elements",
        )

    def test_process_with_new_frame(self):
        sys_indices = [{"sys_index": 0, "frames": [0, 1]}]
        newele_counter = 0
        collect_ele = defaultdict(int)
        datasets = self.datasets
        lst = []
        miss = 1
        process_with_new_frame(
            sys_indices, newele_counter, 1, datasets, lst, collect_ele, miss
        )
        self.assertGreater(
            len(lst), 0, "Expected lst to contain new frames after processing"
        )

    def test_process_missing_elements(self):
        # if miss 30
        min_frames = 1
        dataset = self.datasets[0]
        global_element_counts = {}
        global_type_name = {}
        total_element_types = set()
        process_element_counts(
            0,
            dataset,
            min_frames,
            global_element_counts=global_element_counts,
            global_type_name=global_type_name,
            total_element_types=total_element_types,
        )
        collect_ele = {
            np.int32(key): value
            for key, value in {"36": 1, "6": 1, "12": 1, "17": 1, "19": 1}.items()
        }
        lst = []
        process_missing_elements(
            min_frames,
            global_element_counts,
            total_element_types,
            collect_ele,
            dataset,
            lst,
        )
        assert 30 in lst[0]["atype"], "Error: 30 not found in lst[0]['atype']"
