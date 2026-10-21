# SPDX-License-Identifier: LGPL-3.0-or-later
"""Batch-level composition of independently configured data systems."""

from __future__ import (
    annotations,
)

import logging
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

from deepmd.dpmodel.utils.training_utils import (
    compute_total_numb_batch,
)
from deepmd.utils.data_config import (
    check_type_map_consistency,
)
from deepmd.utils.data_sampling import (
    data_child_seeds,
    group_probabilities,
)
from deepmd.utils.model_stat import (
    _get_stat_batch,
    _get_stat_nsystems,
    _get_stat_pass_length,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterator,
        Mapping,
        Sequence,
    )

    from deepmd.utils.data import (
        DataRequirementItem,
    )

log = logging.getLogger(__name__)


class DataBatch(dict[str, Any]):
    """Carry source-specific execution context without adding model input fields."""

    def __init__(self, values: Mapping[str, Any], pair_filter_enabled: bool) -> None:
        super().__init__(values)
        self.pair_filter_enabled = pair_filter_enabled


def iter_data_sources(data: Any) -> Iterator[Any]:
    """Visit each native data source of a composed dataset in order."""
    children = getattr(data, "children", None)
    if children is not None:
        for child in children:
            yield from iter_data_sources(child)
    else:
        yield data


def epoch_batch_counts(data: Any, world_size: int) -> np.ndarray:
    """Resolve local workloads while preserving the replicated-reader contract."""
    get_counts = getattr(data, "get_epoch_batch_counts", None)
    if get_counts is not None:
        return get_counts(world_size)
    return np.asarray(data.nbatches, dtype=np.float64) / world_size


class GroupedDataSystem:
    """Select whole batches while preserving every child's batching rules.

    The child owns its within-system probabilities, batching, and I/O. This
    node owns only the probability of selecting that child. Training and
    statistical system indices are separate flattened views: an LMDB is one
    training source but may expose several homogeneous statistical systems.
    """

    def __init__(
        self,
        children: Sequence[Any],
        probabilities: Sequence[float | None],
        *,
        seed: int | None = None,
    ) -> None:
        if not children or len(children) != len(probabilities):
            raise ValueError("Data groups require one probability per nonempty child.")
        self.children = tuple(children)
        self._assigned_probabilities = tuple(probabilities)
        self._rng = np.random.default_rng(data_child_seeds(seed, 1)[0])
        self._probabilities: np.ndarray | None = None
        self._cdf: np.ndarray | None = None
        self._filter_enabled: tuple[bool, ...] | None = None
        self._remaining = 0
        self._pass_length = 0
        self._align_metadata()

    def _align_metadata(self) -> None:
        """Preserve dataset-wide NumPy type-map and atom-count contracts."""
        from deepmd.utils.data_system import (
            DeepmdDataSystem,
        )

        sources = list(iter_data_sources(self))
        self.type_map = check_type_map_consistency(
            [source.get_type_map() for source in sources]
        )
        self._ntypes = max(source.get_ntypes() for source in sources)
        numpy_sources = [
            source for source in sources if isinstance(source, DeepmdDataSystem)
        ]
        if len({source.mixed_type for source in numpy_sources}) > 1:
            raise ValueError("All NumPy data systems must agree on mixed_type.")
        for source in numpy_sources:
            source.sys_ntypes = self._ntypes
            source.natoms_vec = [
                system.get_natoms_vec(self._ntypes).astype(int)
                for system in source.data_systems
            ]
        self.mixed_type = any(source.mixed_type for source in sources)

    @property
    def component_probabilities(self) -> np.ndarray:
        """Return this sampling pass's probabilities over child datasets."""
        if self._probabilities is None:
            self._probabilities = group_probabilities(
                [sum(child.nbatches) for child in self.children],
                self._assigned_probabilities,
            )
        return self._probabilities

    @property
    def nbatches(self) -> list[int]:
        """Return global batch counts in the flattened training-system view."""
        return [count for child in self.children for count in child.nbatches]

    @property
    def sys_probs(self) -> np.ndarray:
        """Return global probabilities without duplicating child sampling state."""
        return np.concatenate(
            [
                probability * np.asarray(child.sys_probs)
                for child, probability in zip(
                    self.children, self.component_probabilities, strict=True
                )
            ]
        )

    def set_filter_enabled(self, enabled: Sequence[bool]) -> None:
        """Bind the configured frame-filter policy to each child stream."""
        if len(enabled) != len(self.children):
            raise ValueError("Each data group requires one frame-filter policy.")
        self._filter_enabled = tuple(enabled)

    def _batch_context(self, batch: dict[str, Any], child: int) -> dict[str, Any]:
        if self._filter_enabled is None:
            return batch
        return DataBatch(batch, self._filter_enabled[child])

    def _invalidate_sampling(self) -> None:
        self._probabilities = None
        self._cdf = None
        self._remaining = 0

    def _start_pass(self) -> None:
        self._probabilities = None
        probabilities = self.component_probabilities
        self._cdf = np.cumsum(probabilities)
        self._cdf[-1] = 1.0
        self._pass_length = self._reader_pass_length()
        self._remaining = self._pass_length

    @staticmethod
    def _locate(index: int, sizes: Sequence[int]) -> tuple[int, int]:
        if index < 0:
            raise IndexError(f"System index {index} must be non-negative.")
        for child, size in enumerate(sizes):
            if index < size:
                return child, index
            index -= size
        raise IndexError("System index is outside the grouped dataset.")

    def get_batch(self, sys_idx: int | None = None) -> dict[str, Any]:
        """Draw a child batch, or address one flattened training system."""
        if sys_idx is not None:
            child, local = self._locate(
                sys_idx, [source.get_nsystems() for source in self.children]
            )
            return self._batch_context(
                self.children[child].get_batch(sys_idx=local), child
            )
        if self._remaining == 0:
            self._start_pass()
        assert self._cdf is not None
        index = int(np.searchsorted(self._cdf, self._rng.random(), side="right"))
        self._remaining -= 1
        return self._batch_context(self.children[index].get_batch(), index)

    def _reader_pass_length(self) -> int:
        return compute_total_numb_batch(
            [child.get_batch_pass_length() for child in self.children],
            self.component_probabilities,
        )

    def get_batch_pass_length(self) -> int:
        """Return the local reader-pass length belonging to the current draw."""
        return self._pass_length or self._reader_pass_length()

    def get_epoch_batch_counts(self, world_size: int) -> np.ndarray:
        """Expose local optimizer workloads, including each leaf's sharding."""
        return np.concatenate(
            [epoch_batch_counts(child, world_size) for child in self.children]
        )

    def get_stat_nsystems(self) -> int:
        return sum(_get_stat_nsystems(child) for child in self.children)

    def get_stat_numb_batches(self, sys_idx: int) -> int:
        child, local = self._locate(
            sys_idx, [_get_stat_nsystems(source) for source in self.children]
        )
        return _get_stat_pass_length(self.children[child], local)

    def get_stat_batch(self, sys_idx: int) -> dict[str, Any]:
        child, local = self._locate(
            sys_idx, [_get_stat_nsystems(source) for source in self.children]
        )
        return self._batch_context(_get_stat_batch(self.children[child], local), child)

    def add_data_requirements(self, requirements: list[DataRequirementItem]) -> None:
        for child in self.children:
            child.add_data_requirements(requirements)
        self._invalidate_sampling()

    def use_ragged_batches(self, ragged: bool) -> None:
        for child in self.children:
            configure = getattr(child, "use_ragged_batches", None)
            if configure is not None:
                configure(ragged)
        self._invalidate_sampling()

    def set_frame_transform(self, transform: Any) -> None:
        for child in self.children:
            configure = getattr(child, "set_frame_transform", None)
            if configure is None:
                raise ValueError("Frame transforms require LMDB data in every group.")
            configure(transform)

    def close(self) -> None:
        for child in self.children:
            close = getattr(child, "close", None)
            if close is not None:
                close()

    def get_nsystems(self) -> int:
        return sum(child.get_nsystems() for child in self.children)

    def get_ntypes(self) -> int:
        return self._ntypes

    def get_type_map(self) -> list[str]:
        return self.type_map

    def get_nbatches(self) -> list[int]:
        return self.nbatches

    def get_batch_size(self) -> list[int]:
        return [size for child in self.children for size in child.get_batch_size()]

    def get_sys(self, index: int) -> Any:
        child, local = self._locate(
            index, [source.get_nsystems() for source in self.children]
        )
        return self.children[child].get_sys(local)

    def get_data_dict(self, ii: int = 0) -> dict[str, Any]:
        child, local = self._locate(
            ii, [source.get_nsystems() for source in self.children]
        )
        return self.children[child].get_data_dict(local)

    @property
    def data_systems(self) -> list[Any]:
        return [system for child in self.children for system in child.data_systems]

    @property
    def system_dirs(self) -> list[str]:
        return [path for child in self.children for path in child.system_dirs]

    def add_dict(self, requirements: dict[str, dict[str, Any]]) -> None:
        for child in self.children:
            child.add_dict(requirements)

    def add(self, *args: Any, **kwargs: Any) -> None:
        for child in self.children:
            child.add(*args, **kwargs)

    def reduce(self, key_out: str, key_in: str) -> None:
        for child in self.children:
            child.reduce(key_out, key_in)

    def print_summary(self, name: str, prob: Any = None) -> None:
        for index, (child, probability) in enumerate(
            zip(self.children, self.component_probabilities, strict=True)
        ):
            label = f"{name}/group_{index}"
            log.info("%s: batch probability %.6g", label, probability)
            child.print_summary(label)


class GroupedDataset:
    """Own native loader datasets and expose their combined statistics view."""

    def __init__(
        self,
        children: Sequence[Any],
        configs: Sequence[Mapping[str, Any]],
        probabilities: Sequence[float | None],
    ) -> None:
        if not children or not len(children) == len(configs) == len(probabilities):
            raise ValueError(
                "A dataset group requires matching datasets, configs, and probabilities."
            )
        self.children = tuple(children)
        self.configs = tuple(configs)
        self.probability_specs = tuple(probabilities)
        self.filter_enabled: tuple[bool, ...] | None = None

    def __len__(self) -> int:
        """Preserve native dataset sizes for default multi-task probabilities."""
        return sum(len(child) for child in self.children)

    def set_filter_enabled(self, enabled: Sequence[bool]) -> None:
        """Bind the configured frame-filter policy to each child stream."""
        if len(enabled) != len(self.children):
            raise ValueError("Each data group requires one frame-filter policy.")
        self.filter_enabled = tuple(enabled)

    @property
    def stat_filter_enabled(self) -> list[bool] | None:
        if self.filter_enabled is None:
            return None
        return [
            enabled
            for child, enabled in zip(self.children, self.filter_enabled, strict=True)
            for _ in child.systems
        ]

    @property
    def systems(self) -> list[Any]:
        return [system for child in self.children for system in child.systems]

    @property
    def dataloaders(self) -> list[Any]:
        return [loader for child in self.children for loader in child.dataloaders]

    @property
    def sampler_list(self) -> list[Any]:
        return [sampler for child in self.children for sampler in child.sampler_list]

    @property
    def index(self) -> list[int]:
        return [count for child in self.children for count in child.index]

    @property
    def total_batch(self) -> int:
        return sum(self.index)

    def add_data_requirement(self, requirements: list[DataRequirementItem]) -> None:
        for child in self.children:
            child.add_data_requirement(requirements)

    def close(self) -> None:
        for child in self.children:
            close = getattr(child, "close", None)
            if close is not None:
                close()


class TrainingDataLoader:
    """Expose native loader iteration, sampling metadata, and resource ownership."""

    def __init__(
        self,
        dataset: Any,
        loader: Any,
        probabilities: np.ndarray,
        batch_counts: Callable[[], Sequence[int]],
        *,
        sampling_sizes: Callable[[], Sequence[int]] | None = None,
        iterator_factory: Callable[[], Iterator[dict[str, Any]]] | None = None,
    ) -> None:
        self.dataset = dataset
        self.loader = loader
        self.probabilities = np.asarray(probabilities, dtype=np.float64)
        self._batch_counts = batch_counts
        self._sampling_sizes = sampling_sizes or batch_counts
        self._iterator_factory = iterator_factory or loader.__iter__

    @property
    def batch_counts(self) -> np.ndarray:
        return np.asarray(self._batch_counts(), dtype=np.float64)

    @property
    def sampling_sizes(self) -> np.ndarray:
        """Return natural source masses, independently of per-rank padding."""
        return np.asarray(self._sampling_sizes(), dtype=np.float64)

    @property
    def system_probabilities(self) -> np.ndarray:
        return self.probabilities

    @property
    def epoch_length(self) -> int:
        return compute_total_numb_batch(self.batch_counts, self.system_probabilities)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return self._iterator_factory()

    def __len__(self) -> int:
        return len(self.loader)

    def print_summary(self, name: str) -> None:
        self.dataset.print_summary(name, self.probabilities)

    def close(self) -> None:
        close_loader = getattr(self.loader, "close", None)
        if close_loader is not None:
            close_loader()
        close_dataset = getattr(self.dataset, "close", None)
        if close_dataset is not None:
            close_dataset()


class GroupedTrainingDataLoader:
    """Interleave native batch streams without moving decoding into a wrapper worker."""

    def __init__(
        self,
        dataset: GroupedDataset,
        children: Sequence[TrainingDataLoader | GroupedTrainingDataLoader],
        seed: int | None,
    ) -> None:
        self.dataset = dataset
        self.children = tuple(children)
        self._rng = np.random.default_rng(seed)
        self._iterators: dict[int, Iterator[dict[str, Any]]] = {}

    @property
    def probabilities(self) -> np.ndarray:
        return group_probabilities(
            [float(child.sampling_sizes.sum()) for child in self.children],
            self.dataset.probability_specs,
        )

    @property
    def batch_counts(self) -> np.ndarray:
        return np.concatenate([child.batch_counts for child in self.children])

    @property
    def sampling_sizes(self) -> np.ndarray:
        return np.concatenate([child.sampling_sizes for child in self.children])

    @property
    def system_probabilities(self) -> np.ndarray:
        return np.concatenate(
            [
                probability * child.system_probabilities
                for child, probability in zip(
                    self.children, self.probabilities, strict=True
                )
            ]
        )

    @property
    def epoch_length(self) -> int:
        return compute_total_numb_batch(self.batch_counts, self.system_probabilities)

    def __len__(self) -> int:
        return compute_total_numb_batch(
            [len(child) for child in self.children], self.probabilities
        )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        probabilities = self.probabilities
        cdf = np.cumsum(probabilities)
        cdf[-1] = 1.0
        count = compute_total_numb_batch(
            [len(child) for child in self.children], probabilities
        )
        offsets = np.cumsum([0, *(len(child.batch_counts) for child in self.children)])
        for _ in range(count):
            index = int(np.searchsorted(cdf, self._rng.random(), side="right"))
            if index not in self._iterators:
                self._iterators[index] = iter(self.children[index])
            batch = next(self._iterators[index], None)
            if batch is None:
                self._iterators[index] = iter(self.children[index])
                batch = next(self._iterators[index])
            result = {**batch, "sid": batch["sid"] + int(offsets[index])}
            if self.dataset.filter_enabled is not None:
                result = DataBatch(result, self.dataset.filter_enabled[index])
            elif isinstance(batch, DataBatch):
                result = DataBatch(result, batch.pair_filter_enabled)
            yield result

    def print_summary(self, name: str) -> None:
        for index, (child, probability) in enumerate(
            zip(self.children, self.probabilities, strict=True)
        ):
            label = f"{name}/group_{index}"
            log.info("%s: batch probability %.6g", label, probability)
            child.print_summary(label)

    def close(self) -> None:
        for iterator in self._iterators.values():
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
        self._iterators.clear()
        for child in self.children:
            child.close()
