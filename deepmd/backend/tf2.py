# SPDX-License-Identifier: LGPL-3.0-or-later
from collections.abc import (
    Callable,
)
from importlib.util import (
    find_spec,
)
from typing import (
    TYPE_CHECKING,
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


@Backend.register("tf2")
@Backend.register("tensorflow2")
class TensorFlow2Backend(Backend):
    """TensorFlow 2 eager backend."""

    name = "TensorFlow2"
    features: ClassVar[Backend.Feature] = Backend.Feature.DEEP_EVAL | Backend.Feature.IO
    suffixes: ClassVar[list[str]] = [".savedmodeltf"]

    @classmethod
    def match_filename(cls, filename: str) -> int:
        return 2 if str(filename).lower().endswith(".savedmodeltf") else 0

    def is_available(self) -> bool:
        return find_spec("tensorflow") is not None

    @property
    def entry_point_hook(self) -> Callable[["Namespace"], None]:
        raise NotImplementedError("Training entry point is not implemented for TF2")

    @property
    def deep_eval(self) -> type["DeepEvalBackend"]:
        from deepmd.tf2.infer.deep_eval import (
            DeepEval,
        )

        return DeepEval

    @property
    def neighbor_stat(self) -> type["NeighborStat"]:
        raise NotImplementedError("Neighbor statistics are not implemented for TF2")

    @property
    def serialize_hook(self) -> Callable[[str], dict]:
        from deepmd.tf2.utils.serialization import (
            serialize_from_file,
        )

        return serialize_from_file

    @property
    def deserialize_hook(self) -> Callable[[str, dict], None]:
        from deepmd.tf2.utils.serialization import (
            deserialize_to_file,
        )

        return deserialize_to_file
