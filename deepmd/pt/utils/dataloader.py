# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
import os
from multiprocessing.dummy import (
    Pool,
)

import h5py
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing
from torch.utils.data import (
    DataLoader,
    Dataset,
    WeightedRandomSampler,
)
from torch.utils.data._utils.collate import (
    collate_tensor_fn,
)
from torch.utils.data.distributed import (
    DistributedSampler,
)

from deepmd.pt.utils import (
    dp_random,
    env,
)
from deepmd.pt.utils.dataset import (
    DeepmdDataSetForLoader,
)
from deepmd.pt.utils.utils import (
    mix_entropy,
)
from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.data_system import (
    print_summary,
    prob_sys_size_ext,
    process_sys_probs,
)

log = logging.getLogger(__name__)
torch.multiprocessing.set_sharing_strategy("file_system")


def setup_seed(seed) -> None:
    if isinstance(seed, (list, tuple)):
        mixed_seed = mix_entropy(seed)
    else:
        mixed_seed = seed
    torch.manual_seed(mixed_seed)
    torch.cuda.manual_seed_all(mixed_seed)
    torch.backends.cudnn.deterministic = True
    dp_random.seed(seed)


class DpLoaderSet(Dataset):
    """A dataset for storing DataLoaders to multiple Systems.

    Parameters
    ----------
    sys_path
            Path to the data system
    batch_size
            Max frame count in a batch.
    type_map
            Gives the name of different atom types
    seed
            Random seed for dataloader
    shuffle
            If the data are shuffled (Only effective in serial mode. Always shuffle in distributed data parallelism)
    """

    def __init__(
        self,
        systems,
        batch_size,
        type_map,
        seed=None,
        shuffle=True,
    ) -> None:
        if seed is not None:
            setup_seed(seed)
        if isinstance(systems, str):
            with h5py.File(systems) as file:
                systems = [os.path.join(systems, item) for item in file.keys()]

        def construct_dataset(system):
            return DeepmdDataSetForLoader(
                system=system,
                type_map=type_map,
            )

        self.systems: list[DeepmdDataSetForLoader] = []
        global_rank = dist.get_rank() if dist.is_initialized() else 0
        if global_rank == 0:
            log.info(f"Constructing DataLoaders from {len(systems)} systems")
            with Pool(max(1, env.NUM_WORKERS)) as pool:
                self.systems = pool.map(construct_dataset, systems)
        else:
            self.systems = [None] * len(systems)  # type: ignore
        if dist.is_initialized():
            dist.broadcast_object_list(self.systems)
            assert self.systems[-1] is not None
        self.sampler_list: list[DistributedSampler] = []
        self.index = []
        self.total_batch = 0

        self.dataloaders = []
        self.batch_sizes = []
        if isinstance(batch_size, str):
            if batch_size == "auto":
                rule = 32
                ceiling = True
            elif batch_size.startswith("auto:"):
                rule = int(batch_size.split(":")[1])
                ceiling = True
            elif batch_size.startswith("max:"):
                rule = int(batch_size.split(":")[1])
                ceiling = False
            elif batch_size.startswith("filter:"):
                # remove system with more than `filter` atoms
                rule = int(batch_size.split(":")[1])
                len_before = len(self.systems)
                self.systems = [
                    system for system in self.systems if system._natoms <= rule
                ]
                len_after = len(self.systems)
                if len_before != len_after:
                    log.warning(
                        f"Remove {len_before - len_after} systems with more than {rule} atoms"
                    )
                if len(self.systems) == 0:
                    raise ValueError(
                        f"No system left after removing systems with more than {rule} atoms"
                    )
                ceiling = False
            else:
                raise ValueError(f"Unsupported batch size rule: {batch_size}")
            for ii in self.systems:
                ni = ii._natoms
                bsi = rule // ni
                if ceiling:
                    if bsi * ni < rule:
                        bsi += 1
                else:
                    if bsi == 0:
                        bsi = 1
                self.batch_sizes.append(bsi)
        elif isinstance(batch_size, list):
            self.batch_sizes = batch_size
        else:
            self.batch_sizes = batch_size * np.ones(len(systems), dtype=int)
        assert len(self.systems) == len(self.batch_sizes)
        for system, batch_size in zip(self.systems, self.batch_sizes):
            if dist.is_available() and dist.is_initialized():
                system_sampler = DistributedSampler(system)
                self.sampler_list.append(system_sampler)
            else:
                system_sampler = None
            system_dataloader = DataLoader(
                dataset=system,
                batch_size=int(batch_size),
                num_workers=0,  # Should be 0 to avoid too many threads forked
                sampler=system_sampler,
                collate_fn=collate_batch,
                shuffle=(
                    not (dist.is_available() and dist.is_initialized())
                )  # distributed sampler will do the shuffling by default
                and shuffle,
            )
            self.dataloaders.append(system_dataloader)
            self.index.append(len(system_dataloader))
            self.total_batch += len(system_dataloader)
        # Initialize iterator instances for DataLoader
        self.iters = []
        with torch.device("cpu"):
            for item in self.dataloaders:
                self.iters.append(iter(item))

    def set_noise(self, noise_settings) -> None:
        # noise_settings['noise_type'] # "trunc_normal", "normal", "uniform"
        # noise_settings['noise'] # float, default 1.0
        # noise_settings['noise_mode'] # "prob", "fix_num"
        # noise_settings['mask_num'] # if "fix_num", int
        # noise_settings['mask_prob'] # if "prob", float
        # noise_settings['same_mask'] # coord and type same mask?
        for system in self.systems:
            system.set_noise(noise_settings)

    def __len__(self) -> int:
        return len(self.dataloaders)

    def __getitem__(self, idx):
        # log.warning(str(torch.distributed.get_rank())+" idx: "+str(idx)+" index: "+str(self.index[idx]))
        with torch.device("cpu"):
            try:
                batch = next(self.iters[idx])
            except StopIteration:
                self.iters[idx] = iter(self.dataloaders[idx])
                batch = next(self.iters[idx])
        batch["sid"] = idx
        return batch

    def add_data_requirement(self, data_requirement: list[DataRequirementItem]) -> None:
        """Add data requirement for each system in multiple systems."""
        for system in self.systems:
            system.add_data_requirement(data_requirement)

    def print_summary(
        self,
        name: str,
        prob: list[float],
    ) -> None:
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print_summary(
                name,
                len(self.systems),
                [ss.system for ss in self.systems],
                [ss._natoms for ss in self.systems],
                self.batch_sizes,
                [
                    ss._data_system.get_sys_numb_batch(self.batch_sizes[ii])
                    for ii, ss in enumerate(self.systems)
                ],
                prob,
                [ss._data_system.pbc for ss in self.systems],
            )


def collate_batch(batch):
    example = batch[0]
    result = {}
    for key in example.keys():
        if "find_" in key:
            result[key] = batch[0][key]
        else:
            if batch[0][key] is None:
                result[key] = None
            elif key == "fid":
                result[key] = [d[key] for d in batch]
            elif key == "type":
                continue
            else:
                result[key] = collate_tensor_fn(
                    [torch.as_tensor(d[key]) for d in batch]
                )
    return result


def get_weighted_sampler(training_data, prob_style, sys_prob=False):
    if sys_prob is False:
        if prob_style == "prob_uniform":
            prob_v = 1.0 / float(training_data.__len__())
            probs = [prob_v for ii in range(training_data.__len__())]
        else:  # prob_sys_size;A:B:p1;C:D:p2 or prob_sys_size = prob_sys_size;0:nsys:1.0
            if prob_style == "prob_sys_size":
                style = f"prob_sys_size;0:{len(training_data)}:1.0"
            else:
                style = prob_style
            probs = prob_sys_size_ext(style, len(training_data), training_data.index)
    else:
        probs = process_sys_probs(prob_style, training_data.index)
    log.debug("Generated weighted sampler with prob array: " + str(probs))
    # training_data.total_batch is the size of one epoch, you can increase it to avoid too many  rebuilding of iterators
    len_sampler = training_data.total_batch * max(env.NUM_WORKERS, 1)
    with torch.device("cpu"):
        sampler = WeightedRandomSampler(
            probs,
            len_sampler,
            replacement=True,
        )
    return sampler


def get_sampler_from_params(_data, _params):
    if (
        "sys_probs" in _params and _params["sys_probs"] is not None
    ):  # use sys_probs first
        _sampler = get_weighted_sampler(
            _data,
            _params["sys_probs"],
            sys_prob=True,
        )
    elif "auto_prob" in _params:
        _sampler = get_weighted_sampler(_data, _params["auto_prob"])
    else:
        _sampler = get_weighted_sampler(_data, "prob_sys_size")
    return _sampler
