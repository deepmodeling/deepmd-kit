# SPDX-License-Identifier: LGPL-3.0-or-later
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


@Backend.register("dp")
@Backend.register("dpmodel")
@Backend.register("np")
@Backend.register("numpy")
class DPModelBackend(Backend):
    """DPModel backend that uses NumPy as the reference implementation."""

    name = "DPModel"
    """The formal name of the backend."""
    features: ClassVar[Backend.Feature] = (
        Backend.Feature.DEEP_EVAL | Backend.Feature.NEIGHBOR_STAT | Backend.Feature.IO
    )
    """The features of the backend."""
    suffixes: ClassVar[List[str]] = [".dp"]
    """The suffixes of the backend."""

    def is_available(self) -> bool:
        """Check if the backend is available.

        Returns
        -------
        bool
            Whether the backend is available.
        """
        return True

    @property
    def entry_point_hook(self) -> Callable[["Namespace"], None]:
        """The entry point hook of the backend.

        Returns
        -------
        Callable[[Namespace], None]
            The entry point hook of the backend.
        """
        raise NotImplementedError(f"Unsupported backend: {self.name}")

    @property
    def deep_eval(self) -> Type["DeepEvalBackend"]:
        """The Deep Eval backend of the backend.

        Returns
        -------
        type[DeepEvalBackend]
            The Deep Eval backend of the backend.
        """
        from deepmd.dpmodel.infer.deep_eval import (
            DeepEval,
        )

        return DeepEval

    @property
    def neighbor_stat(self) -> Type["NeighborStat"]:
        """The neighbor statistics of the backend.

        Returns
        -------
        type[NeighborStat]
            The neighbor statistics of the backend.
        """
        from deepmd.dpmodel.utils.neighbor_stat import (
            NeighborStat,
        )

        return NeighborStat

    @property
    def serialize_hook(self) -> Callable[[str], dict]:
        """The serialize hook to convert the model file to a dictionary.

        Returns
        -------
        Callable[[str], dict]
            The serialize hook of the backend.
        """
        from deepmd.dpmodel.utils.network import (
            load_dp_model,
        )

        return load_dp_model

    @property
    def deserialize_hook(self) -> Callable[[str, dict], None]:
        """The deserialize hook to convert the dictionary to a model file.

        Returns
        -------
        Callable[[str, dict], None]
            The deserialize hook of the backend.
        """
        from deepmd.dpmodel.utils.network import (
            save_dp_model,
        )

        return save_dp_model
