# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
import os
import queue
import time
from collections.abc import (
    Iterable,
    Iterator,
)
from functools import (
    cached_property,
)
from multiprocessing.dummy import (
    Pool,
)
from threading import (
    Event,
    Thread,
)
from typing import (
    Any,
)

import h5py
import numpy as np
import paddle
import paddle.distributed as dist

# import paddle.multiprocessing
from paddle.io import (
    BatchSampler,
    DataLoader,
    Dataset,
    DistributedBatchSampler,
    WeightedRandomSampler,
)
from paddle.io.dataloader.collate import (
    default_collate_fn,
)

from deepmd.pd.utils import (
    env,
)
from deepmd.pd.utils.dataset import (
    DeepmdDataSetForLoader,
)
from deepmd.pd.utils.utils import (
    mix_entropy,
    to_numpy_array,
)
from deepmd.utils import random as dp_random
from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.data_config import (
    DataConfig,
    data_group_configs,
    resolve_data_groups,
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
# paddle.multiprocessing.set_sharing_strategy("file_system")


def setup_seed(seed: int | list | tuple) -> None:
    if isinstance(seed, (list, tuple)):
        mixed_seed = mix_entropy(seed)
    else:
        mixed_seed = seed
    paddle.seed(mixed_seed)
    os.environ["FLAGS_cudnn_deterministic"] = "True"
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
        type_map: list[str],
        seed: int | None = None,
        shuffle: bool = True,
    ) -> None:
        if seed is not None:
            setup_seed(seed)
        if isinstance(systems, str):
            with h5py.File(systems) as file:
                systems = [os.path.join(systems, item) for item in file.keys()]

        self.systems: list[DeepmdDataSetForLoader] = []
        if len(systems) >= 100:
            log.info(f"Constructing DataLoaders from {len(systems)} systems")

        def construct_dataset(system: str) -> DeepmdDataSetForLoader:
            return DeepmdDataSetForLoader(
                system=system,
                type_map=type_map,
            )

        MAX_PROCESSES_NUM = 4
        processes = min(
            os.cpu_count()
            // (
                dist.get_world_size()
                if dist.is_available() and dist.is_initialized()
                else 1
            ),
            MAX_PROCESSES_NUM,
        )
        with Pool(processes) as pool:
            self.systems = pool.map(construct_dataset, systems)

        self.sampler_list: list[DistributedBatchSampler] = []
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
                system_batch_sampler = DistributedBatchSampler(
                    system,
                    shuffle=(
                        (not (dist.is_available() and dist.is_initialized()))
                        and shuffle
                    ),
                    batch_size=int(batch_size),
                )
                self.sampler_list.append(system_batch_sampler)
            else:
                system_batch_sampler = BatchSampler(
                    system,
                    shuffle=(
                        (not (dist.is_available() and dist.is_initialized()))
                        and shuffle
                    ),
                    batch_size=int(batch_size),
                )
                self.sampler_list.append(system_batch_sampler)
            system_dataloader = DataLoader(
                dataset=system,
                num_workers=0,  # Should be 0 to avoid too many threads forked
                batch_sampler=system_batch_sampler,
                collate_fn=collate_batch,
                use_buffer_reader=False,
                places=["cpu"],
            )
            self.dataloaders.append(system_dataloader)
            self.index.append(len(system_dataloader))
            self.total_batch += len(system_dataloader)

        class LazyIter:
            """Lazy iterator to prevent fetching data when iter(item)."""

            def __init__(self, item: Any) -> None:
                self.item = item

            def __iter__(self) -> "LazyIter":
                # directly return
                return self

            def __next__(self) -> Any:
                if not isinstance(self.item, Iterator):
                    # make iterator here lazily
                    self.item = iter(self.item)
                return next(self.item)

        self.iters = []

        for item in self.dataloaders:
            self.iters.append(LazyIter(item))

    def __len__(self) -> int:
        return len(self.dataloaders)

    @cached_property
    def mean_batch_nframes(self) -> np.ndarray:
        """Count frames per native batch, including tails and distributed padding."""
        return np.asarray(
            [
                (
                    sampler.num_samples
                    if isinstance(sampler, DistributedBatchSampler)
                    else len(sampler.sampler)
                )
                / count
                for sampler, count in zip(self.sampler_list, self.index, strict=True)
            ],
            dtype=np.float64,
        )

    def __getitem__(self, idx: int) -> dict:
        # log.warning(str(paddle.distributed.get_rank())+" idx: "+str(idx)+" index: "+str(self.index[idx]))
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


_sentinel = object()
QUEUESIZE = 32


class BackgroundConsumer(Thread):
    def __init__(
        self, queue: "queue.Queue[Any]", source: Iterable, max_len: int
    ) -> None:
        Thread.__init__(self)
        self._queue = queue
        self._source = source  # Main DL iterator
        self._max_len = max_len
        self._stop_requested = Event()

    def run(self) -> None:
        for item in self._source:
            if self._stop_requested.is_set():
                return
            if not self._put(item):
                return

        # Signal the consumer we are done.
        self._put(_sentinel)

    def _put(self, item: Any) -> bool:
        """Wait for buffer capacity while allowing the owner to stop loading."""
        while not self._stop_requested.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def close(self) -> None:
        """Stop producing batches and wait for the active read to complete."""
        self._stop_requested.set()
        self.join()


class BufferedIterator:
    def __init__(self, iterable: Iterable) -> None:
        self._queue = queue.Queue(QUEUESIZE)
        self._iterable = iter(iterable)
        self._consumer = None

        self.start_time = time.time()
        self.warning_time = None
        self.total = len(iterable)

    def _create_consumer(self) -> None:
        self._consumer = BackgroundConsumer(self._queue, self._iterable, self.total)
        self._consumer.daemon = True
        self._consumer.start()

    def __iter__(self) -> "BufferedIterator":
        return self

    def __len__(self) -> int:
        return self.total

    def __next__(self) -> Any:
        if self._iterable is None:
            raise StopIteration
        # Create consumer if not created yet
        if self._consumer is None:
            self._create_consumer()
        # Notify the user if there is a data loading bottleneck
        if self._queue.qsize() < min(2, max(1, self._queue.maxsize // 2)):
            if time.time() - self.start_time > 5 * 60:
                if (
                    self.warning_time is None
                    or time.time() - self.warning_time > 15 * 60
                ):
                    log.warning(
                        "Data loading buffer is empty or nearly empty. This may "
                        "indicate a data loading bottleneck, and increasing the "
                        "number of workers (--num-workers) may help."
                    )
                    self.warning_time = time.time()

        # Get next example
        item = self._queue.get()
        if isinstance(item, Exception):
            raise item
        if item is _sentinel:
            self.close()
            raise StopIteration
        return item

    def close(self) -> None:
        """Release buffered batches before the owner closes the native loader."""
        if self._consumer is not None:
            self._consumer.close()
            self._consumer = None
        self._iterable = None
        while not self._queue.empty():
            self._queue.get_nowait()


def collate_batch(batch: list[dict]) -> dict:
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
                result[key] = default_collate_fn([d[key] for d in batch])
    return result


class DpDataLoader(DataLoader):
    """Own native reader threads and persistent per-system worker cursors."""

    def __iter__(self) -> Iterator[dict[str, Any]]:
        self._active_iterator = super().__iter__()
        return self._active_iterator

    def close(self) -> None:
        """Release native threads or workers, including a partially consumed pass."""
        iterator = getattr(self, "_active_iterator", None)
        if iterator is not None:
            iterator._try_shutdown_all()
            self._active_iterator = None
            self._iterator = None

    def __del__(self) -> None:
        """Release workers if construction or initialization ends before training."""
        self.close()


def get_weighted_sampler(
    training_data: DpLoaderSet, prob_style: str | list[float], sys_prob: bool = False
) -> WeightedRandomSampler:
    probs = system_probabilities(
        training_data.index,
        sys_probs=prob_style if sys_prob else None,
        auto_prob="prob_sys_size" if sys_prob else prob_style,
        batch_nframes=(
            training_data.mean_batch_nframes
            if not sys_prob and prob_style == "prob_uniform"
            else None
        ),
    )
    log.debug("Generated weighted sampler with prob array: " + str(probs))
    # training_data.total_batch is the size of one epoch, you can increase it to avoid too many  rebuilding of iteraters
    len_sampler = training_data.total_batch * max(env.NUM_WORKERS, 1)
    sampler = WeightedRandomSampler(probs, len_sampler, replacement=True)
    return sampler


def get_sampler_from_params(
    _data: DpLoaderSet, _params: dict
) -> WeightedRandomSampler | BatchSampler | DistributedBatchSampler:
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
    config: DataConfig, type_map: list[str], *, seed: int | list[int] | None = None
) -> DpLoaderSet | GroupedDataset:
    """Construct NumPy loader groups through the shared path resolver."""
    groups = resolve_data_groups(config, formats=("deepmd",))
    datasets = [
        DpLoaderSet(list(group.systems), group.batch_size, type_map, seed=seed)
        for group in groups
    ]
    return (
        datasets[0]
        if len(datasets) == 1
        else GroupedDataset(
            datasets, data_group_configs(config), [group.prob for group in groups]
        )
    )


def create_dataloader(
    data: DpLoaderSet | GroupedDataset,
    config: DataConfig,
    *,
    seed: int | None = None,
) -> TrainingDataLoader | GroupedTrainingDataLoader:
    """Bind native Paddle loaders to the shared batch-composition interface."""
    if isinstance(data, GroupedDataset):
        seeds = data_child_seeds(
            0 if seed is None and dist.get_world_size() > 1 else seed,
            len(data.children) + 1,
        )
        children = [
            create_dataloader(child, params, seed=child_seed)
            for child, params, child_seed in zip(
                data.children, data.configs, seeds[1:], strict=True
            )
        ]
        return GroupedTrainingDataLoader(data, children, seeds[0])
    params = config[0] if isinstance(config, list) else config
    sampler = get_sampler_from_params(data, params)
    num_workers = env.NUM_WORKERS if dist.is_available() else 0
    loader = DpDataLoader(
        data,
        batch_sampler=BatchSampler(sampler=sampler, drop_last=False),
        num_workers=num_workers,
        persistent_workers=(
            num_workers > 0
            and params.get("auto_prob") == "prob_uniform"
            and params.get("sys_probs") is None
        ),
        collate_fn=lambda batch: batch[0],
    )
    return TrainingDataLoader(
        data,
        loader,
        to_numpy_array(sampler.weights),
        lambda: data.index,
        sampling_sizes=lambda: global_batch_counts(
            [len(system) for system in data.systems], data.batch_sizes
        ),
    )
