# SPDX-License-Identifier: LGPL-3.0-or-later
"""Data management for training and validation.

This module handles data loading, batch iteration, and provides
a unified interface for both single-task and multi-task scenarios.

It now uses the abstract DataLoader interface, allowing future
high-performance implementations to replace DpLoaderSet.
"""

from __future__ import (
    annotations,
)

import logging
from typing import (
    TYPE_CHECKING,
    Any,
)

from deepmd.pt.train.data_loader import (
    AbstractDataLoader,
    DpLoaderSetAdapter,
)
from deepmd.pt.utils.dataloader import (
    DpLoaderSet,
)
from deepmd.pt.utils.env import (
    DEVICE,
)

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)


class DataManager:
    """Manages training and validation data.

    This class handles DataLoader creation, data iteration, and provides
    a unified interface for both single-task and multi-task scenarios.

    Attributes
    ----------
    is_multitask : bool
        Whether managing data for multiple tasks.
    """

    def __init__(
        self,
        training_data: DpLoaderSet
        | dict[str, DpLoaderSet]
        | AbstractDataLoader
        | dict[str, AbstractDataLoader],
        validation_data: DpLoaderSet
        | dict[str, DpLoaderSet]
        | AbstractDataLoader
        | dict[str, AbstractDataLoader]
        | None = None,
        training_params: dict[str, Any] | None = None,
        device: torch.device = DEVICE,
        data_loader_impl: str = "dploaderset",
    ) -> None:
        """Initialize data manager.

        Parameters
        ----------
        training_data : DpLoaderSet | dict[str, DpLoaderSet] | AbstractDataLoader | dict[str, AbstractDataLoader]
            Training dataset(s). Can be DpLoaderSet (legacy) or AbstractDataLoader.
        validation_data : DpLoaderSet | dict[str, DpLoaderSet] | AbstractDataLoader | dict[str, AbstractDataLoader] | None
            Validation dataset(s).
        training_params : dict[str, Any] | None
            Training configuration parameters (kept for API compatibility).
        device : torch.device
            Device to transfer data to.
        data_loader_impl : str
            Data loader implementation to use (for future extensions).
        """
        self.device = device
        self._data_loader_impl = data_loader_impl

        # Determine if multi-task
        self.is_multitask = isinstance(training_data, dict)

        # Convert inputs to AbstractDataLoader if needed
        if self.is_multitask:
            self.training_loaders: dict[str, AbstractDataLoader] = (
                self._ensure_data_loaders(training_data)
            )
            self.validation_loaders: dict[str, AbstractDataLoader | None] = (
                self._ensure_data_loaders(validation_data)
                if validation_data
                else dict.fromkeys(training_data)
            )
            self.model_keys = list(self.training_loaders.keys())
        else:
            self.training_loaders = self._ensure_data_loader(training_data)
            self.validation_loaders = (
                self._ensure_data_loader(validation_data) if validation_data else None
            )

        log.info(f"DataManager initialized with {data_loader_impl} implementation")

    def _ensure_data_loader(
        self, data: DpLoaderSet | AbstractDataLoader
    ) -> AbstractDataLoader:
        """Ensure data is wrapped as AbstractDataLoader.

        Parameters
        ----------
        data : DpLoaderSet | AbstractDataLoader
            Input data.

        Returns
        -------
        AbstractDataLoader
            Wrapped or original data loader.
        """
        if isinstance(data, AbstractDataLoader):
            return data
        elif isinstance(data, DpLoaderSet):
            return DpLoaderSetAdapter(data, self.device)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def _ensure_data_loaders(
        self, data: dict[str, DpLoaderSet | AbstractDataLoader]
    ) -> dict[str, AbstractDataLoader]:
        """Ensure all values in dict are AbstractDataLoader.

        Parameters
        ----------
        data : dict[str, DpLoaderSet | AbstractDataLoader]
            Input data dict.

        Returns
        -------
        dict[str, AbstractDataLoader]
            Dict with wrapped data loaders.
        """
        return {key: self._ensure_data_loader(value) for key, value in data.items()}

    def get_train_batch(
        self, task_key: str | None = None
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Get next training batch.

        Parameters
        ----------
        task_key : str | None
            Task key for multi-task training.

        Returns
        -------
        tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
            (input_dict, label_dict, log_dict)
        """
        if self.is_multitask:
            assert task_key is not None, "task_key required for multi-task"
            return next(self.training_loaders[task_key])
        return next(self.training_loaders)

    def get_valid_batch(
        self, task_key: str | None = None
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Get next validation batch.

        Parameters
        ----------
        task_key : str | None
            Task key for multi-task training.

        Returns
        -------
        tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
            (input_dict, label_dict, log_dict)
        """
        loader = self._get_valid_loader(task_key)
        if loader is None:
            return {}, {}, {}
        return next(loader)

    def _get_valid_loader(
        self, task_key: str | None = None
    ) -> AbstractDataLoader | None:
        """Get validation loader for task."""
        if self.is_multitask:
            assert task_key is not None
            return self.validation_loaders.get(task_key)
        return self.validation_loaders

    def get_valid_numb_batch(self, task_key: str | None = None) -> int:
        """Get number of validation batches.

        For now, returns a default value. Future implementations
        can derive this from the underlying data loader.

        Parameters
        ----------
        task_key : str | None
            Task key for multi-task training.

        Returns
        -------
        int
            Number of validation batches.
        """
        loader = self._get_valid_loader(task_key)
        if loader is None:
            return 1
        # Try to get length, default to 1 if not available
        try:
            return len(loader)
        except (TypeError, AttributeError):
            return 1

    def print_summary(self, rank: int = 0) -> None:
        """Print dataset summaries.

        Parameters
        ----------
        rank : int
            Current process rank (only rank 0 prints).
        """
        if rank != 0:
            return

        if self.is_multitask:
            for key in self.model_keys:
                self.training_loaders[key].print_summary(f"training in {key}")
                if self.validation_loaders.get(key):
                    self.validation_loaders[key].print_summary(f"validation in {key}")
        else:
            self.training_loaders.print_summary("training")
            if self.validation_loaders:
                self.validation_loaders.print_summary("validation")

    def add_data_requirements(
        self,
        requirements: Any,
        task_key: str | None = None,
    ) -> None:
        """Add data requirements.

        Parameters
        ----------
        requirements : Any
            Data requirements to add.
        task_key : str | None
            Task key for multi-task training.
        """
        if self.is_multitask:
            assert task_key is not None
            self.training_loaders[task_key].add_data_requirement(requirements)
            if self.validation_loaders.get(task_key):
                self.validation_loaders[task_key].add_data_requirement(requirements)
        else:
            self.training_loaders.add_data_requirement(requirements)
            if self.validation_loaders:
                self.validation_loaders.add_data_requirement(requirements)

    def preload_data(self, task_key: str | None = None) -> None:
        """Preload data into memory.

        Parameters
        ----------
        task_key : str | None
            Task key for multi-task training.
        """
        if self.is_multitask:
            assert task_key is not None
            self.training_loaders[task_key].preload_data()
            if self.validation_loaders.get(task_key):
                self.validation_loaders[task_key].preload_data()
        else:
            self.training_loaders.preload_data()
            if self.validation_loaders:
                self.validation_loaders.preload_data()

    @staticmethod
    def create_from_dploader_set(
        training_data: DpLoaderSet | dict[str, DpLoaderSet],
        validation_data: DpLoaderSet | dict[str, DpLoaderSet] | None = None,
        device: torch.device = DEVICE,
    ) -> DataManager:
        """Factory method to create from DpLoaderSet(s).

        This is the primary method for creating DataManager from
        existing DpLoaderSet instances.

        Parameters
        ----------
        training_data : DpLoaderSet | dict[str, DpLoaderSet]
            Training dataset(s).
        validation_data : DpLoaderSet | dict[str, DpLoaderSet] | None
            Validation dataset(s).
        device : torch.device
            Device to transfer data to.

        Returns
        -------
        DataManager
            Configured data manager.
        """
        return DataManager(
            training_data=training_data,
            validation_data=validation_data,
            device=device,
            data_loader_impl="dploaderset",
        )
