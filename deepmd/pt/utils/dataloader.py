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
    Sampler,
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
from deepmd.pt.utils.grouped import (
    distributed_grouped_frame_batches,
    grouped_frame_batches,
    has_group_requirement,
    load_group_ids_for_system,
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
        self._group_complete_batches = False
        self._shuffle = shuffle
        self._seed = seed
        self._build_dataloaders()

    def _build_dataloaders(self) -> None:
        """Build per-system dataloaders."""
        self.sampler_list = []
        self.dataloaders = []
        self.index = []
        self.total_batch = 0
        for system, batch_size in zip(self.systems, self.batch_sizes):
            distributed = dist.is_available() and dist.is_initialized()
            system_sampler = None
            batch_sampler = None
            if self._group_complete_batches:
                group_ids = load_group_ids_for_system(system.data_system)
                if group_ids is None:
                    # GroupPropertyLoss falls back to one group per frame
                    # (torch.arange(nframes)) when group_id is absent, i.e.
                    # "no explicit grouping" means every frame stands alone.
                    # Mirror that here: treating a whole system as a single
                    # group instead would silently merge unrelated frames
                    # into one oversized batch (ignoring batch_size) and can
                    # break DDP by handing an implicit single group to more
                    # ranks than it has frames for.
                    group_ids = np.arange(len(system), dtype=np.int64)
                if distributed:
                    batch_sampler = GroupDistributedBatchSampler(
                        group_ids,
                        max_frames=int(batch_size),
                        num_replicas=dist.get_world_size(),
                        rank=dist.get_rank(),
                        shuffle=self._shuffle,
                        seed=self._seed,
                    )
                else:
                    batch_sampler = GroupCompleteBatchSampler(
                        group_ids,
                        max_frames=int(batch_size),
                        shuffle=self._shuffle,
                        seed=self._seed,
                    )
            elif distributed:
                system_sampler = DistributedSampler(system)
                self.sampler_list.append(system_sampler)
            if batch_sampler is None:
                system_dataloader = DataLoader(
                    dataset=system,
                    batch_size=int(batch_size),
                    num_workers=0,  # Should be 0 to avoid too many threads forked
                    sampler=system_sampler,
                    collate_fn=collate_batch,
                    shuffle=(
                        not (dist.is_available() and dist.is_initialized())
                    )  # distributed sampler will do the shuffling by default
                    and self._shuffle,
                )
            else:
                system_dataloader = DataLoader(
                    dataset=system,
                    batch_sampler=batch_sampler,
                    num_workers=0,
                    collate_fn=collate_batch,
                )
            self.dataloaders.append(system_dataloader)
            self.index.append(len(system_dataloader))
            self.total_batch += len(system_dataloader)
        # Initialize iterator instances for DataLoader
        self.iters = []
        with torch.device("cpu"):
            for item in self.dataloaders:
                self.iters.append(iter(item))

    def set_noise(self, noise_settings: dict[str, Any]) -> None:
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
        if has_group_requirement(data_requirement):
            self._group_complete_batches = True
            self._build_dataloaders()

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


def _normalize_group_sampler_seed(
    seed: int | list[int] | tuple[int, ...] | None,
) -> int | None:
    if seed is None:
        return None
    if isinstance(seed, (list, tuple)):
        if not seed:
            return None
        # DDP group batching first builds a global group order, then slices it
        # by rank.  Every rank must therefore use the same base seed.
        return int(seed[0])
    return int(seed)


class GroupDistributedBatchSampler(Sampler[list[int]]):
    """Yield group-complete frame batches for one distributed rank."""

    def __init__(
        self,
        group_ids: np.ndarray,
        max_frames: int,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int | list[int] | tuple[int, ...] | None = None,
    ) -> None:
        self.group_ids = np.asarray(group_ids, dtype=np.int64)
        self.max_frames = max(int(max_frames), 1)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = shuffle
        self.seed = _normalize_group_sampler_seed(seed)
        self._epoch = 0
        self._batches = self._make_batches()

    def _make_batches(self) -> list[list[int]]:
        base_seed = 0 if self.seed is None else self.seed
        # Partition groups across ranks with an epoch-INDEPENDENT seed so every
        # rank always agrees on the split.  If the group shuffle depended on a
        # per-rank ``_epoch`` (which drifts when ranks restart ``cycle_iterator``
        # at different times), the ``group_items[rank::num_replicas]`` slices
        # would come from different shuffles and duplicate/drop groups.
        rng = np.random.default_rng(base_seed)
        batches = distributed_grouped_frame_batches(
            self.group_ids,
            self.max_frames,
            num_replicas=self.num_replicas,
            rank=self.rank,
            shuffle=self.shuffle,
            rng=rng,
        )
        # Vary only THIS rank's own batch order per epoch.  Reordering a rank's
        # batches never moves groups between ranks, so the split stays intact
        # while training still sees a fresh batch order each epoch even if the
        # per-rank epoch counters are not in lock-step.
        if self.shuffle and self._epoch > 0:
            local = np.random.default_rng(
                base_seed + 1 + self.rank + self._epoch * (self.num_replicas + 1)
            )
            order = local.permutation(len(batches))
            batches = [batches[int(ii)] for ii in order]
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        self._batches = self._make_batches()
        self._epoch += 1
        yield from self._batches

    def __len__(self) -> int:
        return len(self._batches)


class GroupCompleteBatchSampler(Sampler[list[int]]):
    """Yield frame batches that never split a group inside one system."""

    def __init__(
        self,
        group_ids: np.ndarray,
        max_frames: int,
        shuffle: bool = True,
        seed: int | list[int] | tuple[int, ...] | None = None,
    ) -> None:
        self.group_ids = np.asarray(group_ids, dtype=np.int64)
        self.max_frames = max(int(max_frames), 1)
        self.shuffle = shuffle
        self.seed = _normalize_group_sampler_seed(seed)
        self._epoch = 0
        self._batches = self._make_batches()

    def _make_batches(self) -> list[list[int]]:
        rng = np.random.default_rng(
            None if self.seed is None else self.seed + self._epoch
        )
        return grouped_frame_batches(
            self.group_ids,
            self.max_frames,
            shuffle=self.shuffle,
            rng=rng,
        )

    def __iter__(self) -> Iterator[list[int]]:
        self._batches = self._make_batches()
        self._epoch += 1
        yield from self._batches

    def __len__(self) -> int:
        return len(self._batches)


def get_weighted_sampler(
    training_data: Any, prob_style: str, sys_prob: bool = False
) -> WeightedRandomSampler:
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
