# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
import os
from collections.abc import (
    Iterator,
)
from multiprocessing.dummy import (
    Pool,
)
from typing import (
    Any,
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

from deepmd.pt.modifier import (
    BaseModifier,
)
from deepmd.pt.utils import (
    dp_random,
    env,
)
from deepmd.pt.utils.dataset import (
    DeepmdDataSetForLoader,
)
from deepmd.pt.utils.env import (
    DEVICE,
)
from deepmd.pt.utils.lmdb_dataset import (
    LmdbBatchDataLoader,
    LmdbDataset,
)
from deepmd.pt.utils.utils import (
    mix_entropy,
    to_numpy_array,
)
from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.data_config import (
    DataConfig,
    data_group_configs,
    resolve_data_groups,
    resolve_data_sources,
)
from deepmd.utils.data_group import (
    GroupedDataset,
    GroupedTrainingDataLoader,
    TrainingDataLoader,
)
from deepmd.utils.data_sampling import (
    data_child_seeds,
    global_batch_counts,
    system_probabilities,
)
from deepmd.utils.data_system import (
    print_summary,
)

log = logging.getLogger(__name__)
torch.multiprocessing.set_sharing_strategy("file_system")


def setup_seed(seed: int | list[int] | tuple[int, ...]) -> None:
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
        systems: str | list[str],
        batch_size: int,
        type_map: list[str] | None,
        seed: int | None = None,
        shuffle: bool = True,
        modifier: BaseModifier | None = None,
    ) -> None:
        if seed is not None:
            setup_seed(seed)
        if isinstance(systems, str):
            with h5py.File(systems) as file:
                systems = [os.path.join(systems, item) for item in file.keys()]

        def construct_dataset(system: str) -> DeepmdDataSetForLoader:
            return DeepmdDataSetForLoader(
                system=system,
                type_map=type_map,
                modifier=modifier,
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

    def __len__(self) -> int:
        return len(self.dataloaders)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
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

    def preload_and_modify_all_data_torch(self) -> None:
        for system in self.systems:
            system.preload_and_modify_all_data_torch()


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
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


def get_weighted_sampler(
    training_data: Any, prob_style: str | list[float], sys_prob: bool = False
) -> WeightedRandomSampler:
    probs = system_probabilities(
        training_data.index,
        sys_probs=prob_style if sys_prob else None,
        auto_prob="prob_sys_size" if sys_prob else prob_style,
    )
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


def get_sampler_from_params(_data: Any, _params: dict[str, Any]) -> Any:
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


def create_dataset(
    config: DataConfig,
    type_map: list[str],
    *,
    seed: int | list[int] | None = None,
    modifier: BaseModifier | None = None,
) -> DpLoaderSet | LmdbDataset | GroupedDataset:
    """Construct native readers from the shared recursive group configuration."""
    groups = resolve_data_groups(config)
    configs = data_group_configs(config)
    datasets = []
    for group, params in zip(groups, configs, strict=True):
        sources = resolve_data_sources(group)
        has_lmdb = any(source.format == "lmdb" for source in sources)
        children = []
        child_configs = []
        for source in sources:
            if source.format == "deepmd":
                child = DpLoaderSet(
                    list(source.systems),
                    source.batch_size,
                    type_map,
                    seed=seed,
                    modifier=modifier,
                )
            else:
                child = LmdbDataset(
                    source.systems[0],
                    type_map,
                    source.batch_size,
                    auto_prob_style=group.auto_prob,
                )
            children.append(child)
            child_configs.append(
                {**params, "sys_probs": group.sys_probs if not has_lmdb else None}
            )
        datasets.append(
            children[0]
            if len(children) == 1
            else GroupedDataset(children, child_configs, [None] * len(children))
        )
    return (
        datasets[0]
        if len(datasets) == 1
        else GroupedDataset(datasets, configs, [group.prob for group in groups])
    )


def create_dataloader(
    data: DpLoaderSet | LmdbDataset | GroupedDataset,
    config: DataConfig,
    *,
    seed: int | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> TrainingDataLoader | GroupedTrainingDataLoader:
    """Bind the native batch pipelines and their local sampling metadata."""
    if isinstance(data, GroupedDataset):
        seeds = data_child_seeds(
            0 if seed is None and world_size > 1 else seed, len(data.children) + 1
        )
        children = [
            create_dataloader(
                child, params, seed=child_seed, rank=rank, world_size=world_size
            )
            for child, params, child_seed in zip(
                data.children, data.configs, seeds[1:], strict=True
            )
        ]
        return GroupedTrainingDataLoader(data, children, seeds[0])
    if isinstance(data, LmdbDataset):
        from deepmd.dpmodel.utils.lmdb_data import (
            DistributedLmdbBatchSampler,
            LmdbBatchSampler,
        )

        options = {"shuffle": True, "seed": seed, "block_targets": data._block_targets}
        sampler = (
            DistributedLmdbBatchSampler(
                data._reader, rank=rank, world_size=world_size, **options
            )
            if world_size > 1
            else LmdbBatchSampler(data._reader, **options)
        )
        loader = LmdbBatchDataLoader(data, sampler, pin_memory=DEVICE.type != "cpu")
        return TrainingDataLoader(
            data,
            loader,
            np.ones(1, dtype=np.float64),
            lambda: [len(loader)],
            sampling_sizes=lambda: [sampler.total_batches],
        )
    params = config[0] if isinstance(config, list) else config
    sampler = get_sampler_from_params(data, params)
    loader = DataLoader(
        data,
        sampler=sampler,
        batch_size=None,
        num_workers=env.NUM_WORKERS if dist.is_available() else 0,
        drop_last=False,
        collate_fn=lambda batch: batch,
        pin_memory=DEVICE.type != "cpu",
    )

    def iterate() -> Iterator[dict[str, Any]]:
        with torch.device("cpu"):
            return iter(loader)

    return TrainingDataLoader(
        data,
        loader,
        to_numpy_array(sampler.weights),
        lambda: data.index,
        sampling_sizes=lambda: global_batch_counts(
            [len(system) for system in data.systems], data.batch_sizes
        ),
        iterator_factory=iterate,
    )
