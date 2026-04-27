# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    abstractmethod,
)
from collections.abc import (
    Callable,
)
from enum import (
    Flag,
    auto,
)
from typing import (
    TYPE_CHECKING,
    ClassVar,
)

from deepmd.utils.plugin import (
    PluginVariant,
    make_plugin_registry,
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


class Backend(PluginVariant, make_plugin_registry("backend")):
    r"""General backend class.

    Examples
    --------
    >>> @Backend.register("tf")
    >>> @Backend.register("tensorflow")
    >>> class TensorFlowBackend(Backend):
    ...     pass
    """

    @staticmethod
    def get_backend(key: str) -> type["Backend"]:
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
        return Backend.get_class_by_type(key)

    @staticmethod
    def get_backends() -> dict[str, type["Backend"]]:
        """Get all the registered backend names.

        Returns
        -------
        list
            all the registered backends
        """
        return Backend.get_plugins()

    @staticmethod
    def get_backends_by_feature(
        feature: "Backend.Feature",
    ) -> dict[str, type["Backend"]]:
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
            for key, backend in Backend.get_backends().items()
            if backend.features & feature
        }

    @staticmethod
    def detect_backend_by_model(filename: str) -> type["Backend"]:
        """Detect the backend of the given model file.

        Parameters
        ----------
        filename : str
            The model file name
        """
        filename_lower = str(filename).lower()
        # `.pt` is shared between the pt and pt_expt backends. They use
        # different parameter naming (pt: `.matrix`/`.bias`, pt_expt:
        # `.w`/`.b`), so peek at the state-dict keys to disambiguate.
        if filename_lower.endswith(".pt"):
            try:
                import torch

                sd = torch.load(filename, map_location="cpu", weights_only=False)
                if isinstance(sd, dict) and "model" in sd:
                    sd = sd["model"]
                keys = list(sd.keys()) if hasattr(sd, "keys") else []
                has_pt_expt = any(k.endswith(".w") or k.endswith(".b") for k in keys)
                has_pt = any(k.endswith(".matrix") or k.endswith(".bias") for k in keys)
                if has_pt_expt and not has_pt:
                    target_name = "pt-expt"
                else:
                    target_name = "pt"
                for key, backend in Backend.get_backends().items():
                    if key == target_name:
                        return backend
            except Exception:
                # Fall through to suffix matching if sniffing fails.
                pass
        for backend in Backend.get_backends().values():
            for suffix in backend.suffixes:
                if filename_lower.endswith(suffix):
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
        IO = auto()
        """Support IO hook."""

    name: ClassVar[str] = "Unknown"
    """The formal name of the backend.

    To be consistent, this name should be also registered in the plugin system."""

    features: ClassVar[Feature] = Feature(0)
    """The features of the backend."""
    suffixes: ClassVar[list[str]] = []
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
    def deep_eval(self) -> type["DeepEvalBackend"]:
        """The Deep Eval backend of the backend.

        Returns
        -------
        type[DeepEvalBackend]
            The Deep Eval backend of the backend.
        """
        pass

    @property
    @abstractmethod
    def neighbor_stat(self) -> type["NeighborStat"]:
        """The neighbor statistics of the backend.

        Returns
        -------
        type[NeighborStat]
            The neighbor statistics of the backend.
        """
        pass

    @property
    @abstractmethod
    def serialize_hook(self) -> Callable[[str], dict]:
        """The serialize hook to convert the model file to a dictionary.

        Returns
        -------
        Callable[[str], dict]
            The serialize hook of the backend.
        """
        pass

    @property
    @abstractmethod
    def deserialize_hook(self) -> Callable[[str, dict], None]:
        """The deserialize hook to convert the dictionary to a model file.

        Returns
        -------
        Callable[[str, dict], None]
            The deserialize hook of the backend.
        """
        pass
