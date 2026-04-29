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

    @classmethod
    def match_filename(cls, filename: str) -> int:
        """Specificity score of this backend's claim on ``filename``.

        Returns a positive integer if this backend can handle the file
        (higher = stronger / more specific claim), or 0 otherwise.

        The default implementation returns 1 when ``filename`` ends with
        one of ``cls.suffixes``. Backends with overlapping suffixes can
        override this to disambiguate (e.g. by inspecting file content)
        and return a higher score so they win the tie.
        """
        fname = str(filename).lower()
        return 1 if any(fname.endswith(s) for s in cls.suffixes) else 0

    @staticmethod
    def detect_backend_by_model(filename: str) -> type["Backend"]:
        """Detect the backend of the given model file.

        Calls ``match_filename`` on every registered backend and returns
        the one with the highest specificity score (>0).

        Parameters
        ----------
        filename : str
            The model file name
        """
        best: type[Backend] | None = None
        best_score = 0
        for backend in Backend.get_backends().values():
            score = backend.match_filename(filename)
            if score > best_score:
                best_score = score
                best = backend
        if best is None:
            raise ValueError(f"Cannot detect the backend of the model file {filename}.")
        return best

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
