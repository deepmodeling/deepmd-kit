# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    abstractmethod,
)
from enum import (
    Flag,
    auto,
)
from typing import (
    TYPE_CHECKING,
    Callable,
    ClassVar,
    Dict,
    List,
    Type,
)

from deepmd.utils.plugin import (
    Plugin,
    PluginVariant,
)

if TYPE_CHECKING:
    from argparse import (
        Namespace,
    )

    from deepmd.infer.deep_eval import (
        DeepEvalBackend,
    )
    from deepmd.utils.neighbor_stat import (
        NeighborStat,
    )


class Backend(PluginVariant):
    r"""General backend class.

    Examples
    --------
    >>> @Backend.register("tf")
    >>> @Backend.register("tensorflow")
    >>> class TensorFlowBackend(Backend):
    ...     pass
    """

    __plugins = Plugin()

    @staticmethod
    def register(key: str) -> Callable[[object], object]:
        """Register a backend plugin.

        Parameters
        ----------
        key : str
            the key of a backend

        Returns
        -------
        Callable[[object], object]
            the decorator to register backend
        """
        return Backend.__plugins.register(key.lower())

    @staticmethod
    def get_backend(key: str) -> Type["Backend"]:
        """Get the backend by key.

        Parameters
        ----------
        key : str
            the key of a backend

        Returns
        -------
        Backend
            the backend
        """
        try:
            backend = Backend.__plugins.get_plugin(key.lower())
        except KeyError:
            raise KeyError(f"Backend {key} is not registered.")
        assert isinstance(backend, type)
        return backend

    @staticmethod
    def get_backends() -> Dict[str, Type["Backend"]]:
        """Get all the registered backend names.

        Returns
        -------
        list
            all the registered backends
        """
        return Backend.__plugins.plugins

    @staticmethod
    def get_backends_by_feature(
        feature: "Backend.Feature",
    ) -> Dict[str, Type["Backend"]]:
        """Get all the registered backend names with a specific feature.

        Parameters
        ----------
        feature : Backend.Feature
            the feature flag

        Returns
        -------
        list
            all the registered backends with the feature
        """
        return {
            key: backend
            for key, backend in Backend.__plugins.plugins.items()
            if backend.features & feature
        }

    @staticmethod
    def detect_backend_by_model(filename: str) -> Type["Backend"]:
        """Detect the backend of the given model file.

        Parameters
        ----------
        filename : str
            The model file name
        """
        filename = str(filename).lower()
        for backend in Backend.get_backends().values():
            if filename.endswith(backend.suffixes[0]):
                return backend
        raise ValueError(f"Cannot detect the backend of the model file {filename}.")

    class Feature(Flag):
        """Feature flag to indicate whether the backend supports certain features."""

        ENTRY_POINT = auto()
        """Support entry point hook."""
        DEEP_EVAL = auto()
        """Support Deep Eval backend."""
        NEIGHBOR_STAT = auto()
        """Support neighbor statistics."""

    name: ClassVar[str] = "Unknown"
    """The formal name of the backend.

    To be consistent, this name should be also registered in the plugin system."""

    features: ClassVar[Feature] = Feature(0)
    """The features of the backend."""
    suffixes: ClassVar[List[str]] = []
    """The supported suffixes of the saved model.

    The first element is considered as the default suffix."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is available.

        Returns
        -------
        bool
            Whether the backend is available.
        """

    @property
    @abstractmethod
    def entry_point_hook(self) -> Callable[["Namespace"], None]:
        """The entry point hook of the backend.

        Returns
        -------
        Callable[[Namespace], None]
            The entry point hook of the backend.
        """
        pass

    @property
    @abstractmethod
    def deep_eval(self) -> Type["DeepEvalBackend"]:
        """The Deep Eval backend of the backend.

        Returns
        -------
        type[DeepEvalBackend]
            The Deep Eval backend of the backend.
        """
        pass

    @property
    @abstractmethod
    def neighbor_stat(self) -> Type["NeighborStat"]:
        """The neighbor statistics of the backend.

        Returns
        -------
        type[NeighborStat]
            The neighbor statistics of the backend.
        """
        pass
