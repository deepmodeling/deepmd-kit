# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
import os
import queue
import time
from multiprocessing.dummy import (
    Pool,
)
from threading import (
    Thread,
)
from typing import (
    List,
)

import h5py
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
    env,
)
from deepmd.pt.utils.dataset import (
    DeepmdDataSetForLoader,
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


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class DpLoaderSet(Dataset):
    """A dataset for storing DataLoaders to multiple Systems."""

    def __init__(
        self,
        systems,
        batch_size,
        model_params,
        seed=10,
        shuffle=True,
    ):
        setup_seed(seed)
        if isinstance(systems, str):
            with h5py.File(systems) as file:
                systems = [os.path.join(systems, item) for item in file.keys()]

        self.systems: List[DeepmdDataSetForLoader] = []
        if len(systems) >= 100:
            log.info(f"Constructing DataLoaders from {len(systems)} systems")

        def construct_dataset(system):
            return DeepmdDataSetForLoader(
                system=system,
                type_map=model_params["type_map"],
                shuffle=shuffle,
            )

        with Pool(
            os.cpu_count()
            // (int(os.environ["LOCAL_WORLD_SIZE"]) if dist.is_initialized() else 1)
        ) as pool:
            self.systems = pool.map(construct_dataset, systems)

        self.sampler_list: List[DistributedSampler] = []
        self.index = []
        self.total_batch = 0

        self.dataloaders = []
        self.batch_sizes = []
        for system in self.systems:
            if dist.is_initialized():
                system_sampler = DistributedSampler(system)
                self.sampler_list.append(system_sampler)
            else:
                system_sampler = None
            if isinstance(batch_size, str):
                if batch_size == "auto":
                    rule = 32
                elif batch_size.startswith("auto:"):
                    rule = int(batch_size.split(":")[1])
                else:
                    rule = None
                    log.error("Unsupported batch size type")
                self.batch_size = rule // system._natoms
                if self.batch_size * system._natoms < rule:
                    self.batch_size += 1
            else:
                self.batch_size = batch_size
            self.batch_sizes.append(self.batch_size)
            system_dataloader = DataLoader(
                dataset=system,
                batch_size=self.batch_size,
                num_workers=0,  # Should be 0 to avoid too many threads forked
                sampler=system_sampler,
                collate_fn=collate_batch,
                shuffle=(not dist.is_initialized()) and shuffle,
            )
            self.dataloaders.append(system_dataloader)
            self.index.append(len(system_dataloader))
            self.total_batch += len(system_dataloader)
        # Initialize iterator instances for DataLoader
        self.iters = []
        with torch.device("cpu"):
            for item in self.dataloaders:
                self.iters.append(iter(item))

    def set_noise(self, noise_settings):
        # noise_settings['noise_type'] # "trunc_normal", "normal", "uniform"
        # noise_settings['noise'] # float, default 1.0
        # noise_settings['noise_mode'] # "prob", "fix_num"
        # noise_settings['mask_num'] # if "fix_num", int
        # noise_settings['mask_prob'] # if "prob", float
        # noise_settings['same_mask'] # coord and type same mask?
        for system in self.systems:
            system.set_noise(noise_settings)

    def __len__(self):
        return len(self.dataloaders)

    def __getitem__(self, idx):
        # log.warning(str(torch.distributed.get_rank())+" idx: "+str(idx)+" index: "+str(self.index[idx]))
        try:
            batch = next(self.iters[idx])
        except StopIteration:
            self.iters[idx] = iter(self.dataloaders[idx])
            batch = next(self.iters[idx])
        batch["sid"] = idx
        return batch

    def add_data_requirement(self, data_requirement: List[DataRequirementItem]):
        """Add data requirement for each system in multiple systems."""
        for system in self.systems:
            system.add_data_requirement(data_requirement)

    def print_summary(
        self,
        name: str,
        prob: List[float],
    ):
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
    def __init__(self, queue, source, max_len):
        Thread.__init__(self)
        self._queue = queue
        self._source = source  # Main DL iterator
        self._max_len = max_len  #

    def run(self):
        for item in self._source:
            self._queue.put(item)  # Blocking if the queue is full

        # Signal the consumer we are done.
        self._queue.put(_sentinel)


class BufferedIterator:
    def __init__(self, iterable):
        self._queue = queue.Queue(QUEUESIZE)
        self._iterable = iterable
        self._consumer = None

        self.start_time = time.time()
        self.warning_time = None
        self.total = len(iterable)

    def _create_consumer(self):
        self._consumer = BackgroundConsumer(self._queue, self._iterable, self.total)
        self._consumer.daemon = True
        self._consumer.start()

    def __iter__(self):
        return self

    def __len__(self):
        return self.total

    def __next__(self):
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
            raise StopIteration
        return item


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
    # training_data.total_batch is the size of one epoch, you can increase it to avoid too many  rebuilding of iteraters
    len_sampler = training_data.total_batch * max(env.NUM_WORKERS, 1)
    with torch.device("cpu"):
        sampler = WeightedRandomSampler(probs, len_sampler, replacement=True)
    return sampler
