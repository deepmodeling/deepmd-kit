# SPDX-License-Identifier: LGPL-3.0-or-later
from importlib.util import (
    find_spec,
)
from typing import (
    TYPE_CHECKING,
    Callable,
    ClassVar,
    List,
    Type,
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


@Backend.register("tf")
@Backend.register("tensorflow")
class TensorFlowBackend(Backend):
    """TensorFlow backend."""

    name = "TensorFlow"
    """The formal name of the backend."""
    features: ClassVar[Backend.Feature] = (
        Backend.Feature.ENTRY_POINT
        | Backend.Feature.DEEP_EVAL
        | Backend.Feature.NEIGHBOR_STAT
    )
    """The features of the backend."""
    suffixes: ClassVar[List[str]] = [".pb"]
    """The suffixes of the backend."""

    def is_available(self) -> bool:
        """Check if the backend is available.

        Returns
        -------
        bool
            Whether the backend is available.
        """
        # deepmd.env imports expensive numpy
        # avoid import outside the method
        from deepmd.env import (
            GLOBAL_CONFIG,
        )

        return (
            find_spec("tensorflow") is not None
            and GLOBAL_CONFIG["enable_tensorflow"] != "0"
        )

    @property
    def entry_point_hook(self) -> Callable[["Namespace"], None]:
        """The entry point hook of the backend.

        Returns
        -------
        Callable[[Namespace], None]
            The entry point hook of the backend.
        """
        from deepmd.tf.entrypoints.main import main as deepmd_main

        return deepmd_main

    @property
    def deep_eval(self) -> Type["DeepEvalBackend"]:
        """The Deep Eval backend of the backend.

        Returns
        -------
        type[DeepEvalBackend]
            The Deep Eval backend of the backend.
        """
        from deepmd.tf.infer.deep_eval import DeepEval as DeepEvalTF

        return DeepEvalTF

    @property
    def neighbor_stat(self) -> Type["NeighborStat"]:
        """The neighbor statistics of the backend.

        Returns
        -------
        type[NeighborStat]
            The neighbor statistics of the backend.
        """
        from deepmd.tf.utils.neighbor_stat import (
            NeighborStat,
        )

        return NeighborStat
