# SPDX-License-Identifier: LGPL-3.0-or-later
"""Abstract data loader interface with DpLoaderSet compatibility.

This module provides an abstract interface for data loading that:
1. Is compatible with existing DpLoaderSet
2. Allows future high-performance implementations without DpLoaderSet dependency
3. Provides a clean, backend-agnostic API

Future implementations can:
- Replace DpLoaderSet with custom Dataset classes
- Implement prefetching and async data loading
- Use memory-mapped files for large datasets
- Implement custom batching strategies
"""

from __future__ import (
    annotations,
)

import logging
from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Protocol,
    runtime_checkable,
)

import torch
from torch.utils.data import (
    DataLoader,
)

from deepmd.pt.utils.env import (
    DEVICE,
)

if TYPE_CHECKING:
    from collections.abc import (
        Iterator,
    )

    from deepmd.pt.utils.dataloader import (
        DpLoaderSet,
    )

log = logging.getLogger(__name__)


@runtime_checkable
class DataLoaderInterface(Protocol):
    """Protocol defining the minimal data loader interface.

    Any data loader implementation (DpLoaderSet or future alternatives)
    must satisfy this protocol to work with the training system.

    This allows gradual migration from DpLoaderSet to new implementations.
    """

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Return iterator over batches."""
        ...

    def __next__(self) -> dict[str, Any]:
        """Get next batch."""
        ...

    def add_data_requirement(self, requirement: Any) -> None:
        """Add data requirements for labels."""
        ...

    def preload_and_modify_all_data_torch(self) -> None:
        """Preload and apply modifiers to data."""
        ...

    def print_summary(self, name: str, weights: Any = None) -> None:
        """Print dataset summary."""
        ...

    @property
    def systems(self) -> list[Any]:
        """Get list of systems/datasets."""
        ...


class BatchProcessor:
    """Processes batches: device transfer and input/label splitting.

    This class centralizes batch processing logic, making it reusable
    across different data loader implementations.
    """

    def __init__(
        self,
        device: torch.device = DEVICE,
        input_keys: list[str] | None = None,
    ) -> None:
        """Initialize batch processor.

        Parameters
        ----------
        device : torch.device
            Target device for tensors.
        input_keys : list[str] | None
            Keys that are considered model inputs.
        """
        self.device = device
        self.input_keys = input_keys or [
            "coord",
            "atype",
            "spin",
            "box",
            "fparam",
            "aparam",
        ]

    def process(
        self, batch_data: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Process a batch: transfer to device and split inputs/labels.

        Parameters
        ----------
        batch_data : dict[str, Any]
            Raw batch data.

        Returns
        -------
        tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
            (input_dict, label_dict, log_dict)
        """
        # Transfer to device
        processed = self._to_device(batch_data)

        # Split into inputs and labels
        input_dict, label_dict = self._split_inputs_labels(processed)

        # Create log dict
        log_dict = self._create_log_dict(processed)

        return input_dict, label_dict, log_dict

    def _to_device(self, batch_data: dict[str, Any]) -> dict[str, Any]:
        """Transfer batch data to target device."""
        result = {}
        for key, value in batch_data.items():
            if key in ("sid", "fid", "box") or "find_" in key:
                result[key] = value
            elif isinstance(value, list):
                result[key] = [
                    item.to(self.device, non_blocking=True)
                    if isinstance(item, torch.Tensor)
                    else item
                    for item in value
                ]
            elif isinstance(value, torch.Tensor):
                result[key] = value.to(self.device, non_blocking=True)
            else:
                result[key] = value
        return result

    def _split_inputs_labels(
        self, batch_data: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Split batch into input and label dictionaries."""
        input_dict: dict[str, Any] = {}
        label_dict: dict[str, Any] = {}

        for key, value in batch_data.items():
            if key in self.input_keys:
                # Special handling for fparam with find_fparam
                if key == "fparam" and batch_data.get("find_fparam", 0.0) == 0.0:
                    continue
                input_dict[key] = value
            elif key not in ("sid", "fid"):
                label_dict[key] = value

        return input_dict, label_dict

    def _create_log_dict(self, batch_data: dict[str, Any]) -> dict[str, Any]:
        """Create log dictionary from batch data."""
        log_dict: dict[str, Any] = {}
        if "fid" in batch_data:
            log_dict["fid"] = batch_data["fid"]
        if "sid" in batch_data:
            log_dict["sid"] = batch_data["sid"]
        return log_dict


class AbstractDataLoader(ABC):
    """Abstract base class for data loaders.

    This class defines the interface that all data loaders must implement.
    It provides a common API for training code while allowing different
    underlying implementations.

    Implementations:
    - DpLoaderSetAdapter: Wraps existing DpLoaderSet
    - Future: High-performance data loader without DpLoaderSet
    """

    def __init__(
        self,
        device: torch.device = DEVICE,
    ) -> None:
        """Initialize abstract data loader.

        Parameters
        ----------
        device : torch.device
            Target device for data.
        """
        self.device = device
        self._batch_processor = BatchProcessor(device)

    @abstractmethod
    def __iter__(
        self,
    ) -> Iterator[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
        """Return iterator yielding processed batches.

        Yields
        ------
        tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
            (input_dict, label_dict, log_dict)
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return number of batches per epoch."""
        pass

    @abstractmethod
    def add_data_requirement(self, requirement: Any) -> None:
        """Add data requirement for labels.

        Parameters
        ----------
        requirement : Any
            Data requirement specification.
        """
        pass

    @abstractmethod
    def preload_data(self) -> None:
        """Preload data into memory."""
        pass

    @abstractmethod
    def print_summary(self, name: str) -> None:
        """Print data summary.

        Parameters
        ----------
        name : str
            Name to display in summary.
        """
        pass


class DpLoaderSetAdapter(AbstractDataLoader):
    """Adapter making DpLoaderSet compatible with AbstractDataLoader.

    This adapter wraps the existing DpLoaderSet implementation,
    allowing it to be used with the new training system without
    modifying the original class.

    This is the transition solution - future implementations can
    replace this with high-performance alternatives.
    """

    def __init__(
        self,
        dp_loader_set: DpLoaderSet,
        device: torch.device = DEVICE,
    ) -> None:
        """Initialize adapter.

        Parameters
        ----------
        dp_loader_set : DpLoaderSet
            Existing DpLoaderSet instance.
        device : torch.device
            Target device.
        """
        super().__init__(device)
        self._dp_loader = dp_loader_set
        self._iterator: Iterator[dict[str, Any]] | None = None

    def __iter__(
        self,
    ) -> Iterator[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
        """Return iterator over processed batches."""
        self._iterator = self._create_iterator()
        return self

    def __next__(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Get next processed batch."""
        if self._iterator is None:
            self._iterator = self._create_iterator()

        batch = next(self._iterator)
        return self._batch_processor.process(batch)

    def _create_iterator(self) -> Iterator[dict[str, Any]]:
        """Create underlying iterator with automatic restart.

        Uses DpLoaderSet's __getitem__ method which handles:
        - System sampling according to weights
        - Iterator reset on exhaustion
        - CPU device context for data loading
        """
        # Import sampler utilities
        from deepmd.pt.utils.dataloader import (
            get_sampler_from_params,
        )

        # Get or create sampler for weighted system sampling
        if hasattr(self._dp_loader, "sampler") and self._dp_loader.sampler is not None:
            sampler = self._dp_loader.sampler
        else:
            # Create default sampler with prob_sys_size
            with torch.device("cpu"):
                sampler = get_sampler_from_params(self._dp_loader, "prob_sys_size")
            # Store sampler on dp_loader for consistency
            self._dp_loader.sampler = sampler

        # Create DataLoader that wraps DpLoaderSet (not its internal dataloaders)
        # This ensures __getitem__ is called with sampled indices
        import torch.distributed as dist

        from deepmd.pt.utils.env import (
            NUM_WORKERS,
        )

        with torch.device("cpu"):
            dataloader = DataLoader(
                self._dp_loader,
                sampler=sampler,
                batch_size=None,
                num_workers=NUM_WORKERS
                if dist.is_available() and dist.is_initialized()
                else 0,
                drop_last=False,
                collate_fn=lambda batch: batch,  # prevent extra conversion
                pin_memory=False,  # Batch processor handles device transfer
            )

        def cycle_iterator() -> Iterator[dict[str, Any]]:
            """Infinite iterator that cycles through the dataloader."""
            while True:
                with torch.device("cpu"):
                    it = iter(dataloader)
                yield from it

        return cycle_iterator()

    def __len__(self) -> int:
        """Return total number of batches."""
        return self._dp_loader.total_batch

    def add_data_requirement(self, requirement: Any) -> None:
        """Add data requirement to underlying DpLoaderSet."""
        self._dp_loader.add_data_requirement(requirement)

    def preload_data(self) -> None:
        """Preload data via DpLoaderSet."""
        self._dp_loader.preload_and_modify_all_data_torch()

    def print_summary(self, name: str) -> None:
        """Print summary via DpLoaderSet."""
        from deepmd.pt.utils.utils import (
            to_numpy_array,
        )

        weights = None
        if hasattr(self._dp_loader, "sampler_list") and self._dp_loader.sampler_list:
            # Get weights from first sampler as representative
            if hasattr(self._dp_loader.sampler_list[0], "weights"):
                weights = to_numpy_array(self._dp_loader.sampler_list[0].weights)

        # Handle case where sampler doesn't have weights (e.g., DistributedSampler)
        if weights is None and hasattr(self._dp_loader, "systems"):
            # Default: uniform weights
            import numpy as np

            weights = np.ones(len(self._dp_loader.systems), dtype=np.float32)

        self._dp_loader.print_summary(name, weights)

    @property
    def dp_loader_set(self) -> DpLoaderSet:
        """Access underlying DpLoaderSet (for backward compatibility)."""
        return self._dp_loader


class DataLoaderFactory:
    """Factory for creating data loaders.

    This factory centralizes data loader creation and allows
    easy switching between implementations.
    """

    # Registry of available implementations
    _implementations: ClassVar[dict[str, type[AbstractDataLoader]]] = {
        "dploaderset": DpLoaderSetAdapter,
    }

    @classmethod
    def register(cls, name: str, implementation: type[AbstractDataLoader]) -> None:
        """Register a new data loader implementation.

        Parameters
        ----------
        name : str
            Identifier for the implementation.
        implementation : type[AbstractDataLoader]
            Data loader class.
        """
        cls._implementations[name] = implementation
        log.info(f"Registered data loader implementation: {name}")

    @classmethod
    def create(
        cls,
        data_source: Any,
        implementation: str = "dploaderset",
        device: torch.device = DEVICE,
        **kwargs: Any,
    ) -> AbstractDataLoader:
        """Create a data loader instance.

        Parameters
        ----------
        data_source : Any
            Source data (DpLoaderSet, paths, etc.).
        implementation : str
            Which implementation to use.
        device : torch.device
            Target device.
        **kwargs : Any
            Additional arguments for the implementation.

        Returns
        -------
        AbstractDataLoader
            Configured data loader.

        Raises
        ------
        ValueError
            If implementation is not registered.
        """
        if implementation not in cls._implementations:
            raise ValueError(
                f"Unknown data loader implementation: {implementation}. "
                f"Available: {list(cls._implementations.keys())}"
            )

        impl_class = cls._implementations[implementation]
        return impl_class(data_source, device=device, **kwargs)

    @classmethod
    def get_available_implementations(cls) -> list[str]:
        """Get list of registered implementations."""
        return list(cls._implementations.keys())


# Convenience functions
def create_data_loader(
    data_source: Any,
    implementation: str = "dploaderset",
    device: torch.device = DEVICE,
    **kwargs: Any,
) -> AbstractDataLoader:
    """Create a data loader (convenience function)."""
    return DataLoaderFactory.create(data_source, implementation, device, **kwargs)


def adapt_dploader_set(
    dp_loader_set: DpLoaderSet,
    device: torch.device = DEVICE,
) -> DpLoaderSetAdapter:
    """Adapt existing DpLoaderSet (convenience function)."""
    return DpLoaderSetAdapter(dp_loader_set, device)
