# SPDX-License-Identifier: LGPL-3.0-or-later
from importlib.util import (
    find_spec,
)
from typing import (
    TYPE_CHECKING,
    Callable,
    ClassVar,
)

from deepmd.backend.backend import (
    Backend,
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


@Backend.register("jax2tf")
class JAXBackend(Backend):
    """JAX to TensorFlow backend."""

    name = "JAX2TF"
    """The formal name of the backend."""
    features: ClassVar[Backend.Feature] = (
        Backend.Feature.IO
        # | Backend.Feature.ENTRY_POINT
        # | Backend.Feature.DEEP_EVAL
    )
    """The features of the backend."""
    suffixes: ClassVar[list[str]] = [".savedmodel"]
    """The suffixes of the backend."""

    def is_available(self) -> bool:
        """Check if the backend is available.

        Returns
        -------
        bool
            Whether the backend is available.
        """
        return find_spec("jax") is not None and find_spec("tensorflow") is not None

    @property
    def entry_point_hook(self) -> Callable[["Namespace"], None]:
        """The entry point hook of the backend.

        Returns
        -------
        Callable[[Namespace], None]
            The entry point hook of the backend.
        """
        raise NotImplementedError

    @property
    def deep_eval(self) -> type["DeepEvalBackend"]:
        """The Deep Eval backend of the backend.

        Returns
        -------
        type[DeepEvalBackend]
            The Deep Eval backend of the backend.
        """
        raise NotImplementedError
        # from deepmd.jax.infer.deep_eval import (
        #     DeepEval,
        # )

        # return DeepEval

    @property
    def neighbor_stat(self) -> type["NeighborStat"]:
        """The neighbor statistics of the backend.

        Returns
        -------
        type[NeighborStat]
            The neighbor statistics of the backend.
        """
        raise NotImplementedError

    @property
    def serialize_hook(self) -> Callable[[str], dict]:
        """The serialize hook to convert the model file to a dictionary.

        Returns
        -------
        Callable[[str], dict]
            The serialize hook of the backend.
        """
        raise NotImplementedError

    @property
    def deserialize_hook(self) -> Callable[[str, dict], None]:
        """The deserialize hook to convert the dictionary to a model file.

        Returns
        -------
        Callable[[str, dict], None]
            The deserialize hook of the backend.
        """
        from deepmd.jax.jax2tf.serialization import (
            deserialize_to_file,
        )

        return deserialize_to_file
