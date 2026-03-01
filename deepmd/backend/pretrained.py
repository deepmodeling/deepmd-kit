# SPDX-License-Identifier: LGPL-3.0-or-later
from collections.abc import (
    Callable,
)
from typing import (
    TYPE_CHECKING,
    ClassVar,
)

from deepmd.backend.backend import (
    Backend,
)
from deepmd.pretrained.deep_eval import (
    get_pretrained_deep_eval_backend,
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


@Backend.register("pretrained")
class PretrainedBackend(Backend):
    """Backend for ``*.pretrained`` model aliases."""

    name = "Pretrained"
    features: ClassVar[Backend.Feature] = Backend.Feature.DEEP_EVAL
    suffixes: ClassVar[list[str]] = [".pretrained"]

    def is_available(self) -> bool:
        return True

    @property
    def entry_point_hook(self) -> Callable[["Namespace"], None]:
        raise NotImplementedError("Unsupported backend: pretrained")

    @property
    def deep_eval(self) -> type["DeepEvalBackend"]:
        return get_pretrained_deep_eval_backend()

    @property
    def neighbor_stat(self) -> type["NeighborStat"]:
        raise NotImplementedError("Unsupported backend: pretrained")

    @property
    def serialize_hook(self) -> Callable[[str], dict]:
        raise NotImplementedError("Unsupported backend: pretrained")

    @property
    def deserialize_hook(self) -> Callable[[str, dict], None]:
        raise NotImplementedError("Unsupported backend: pretrained")
