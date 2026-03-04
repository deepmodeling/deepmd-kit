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
from deepmd.pretrained.registry import (
    available_model_names,
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
    """Internal virtual backend for pretrained model-name alias dispatch.

    This backend is not intended to be selected explicitly by users as a real
    compute backend (such as TensorFlow/PyTorch/Paddle/JAX). It only bridges
    built-in pretrained model names into the regular deep-eval loading path.

    For convenience, all built-in pretrained model names are registered as
    suffix-like aliases, so users can pass model names directly, e.g.
    ``DeepPot("DPA-3.2-5M")``.
    """

    name = "Pretrained"
    features: ClassVar[Backend.Feature] = Backend.Feature.DEEP_EVAL
    suffixes: ClassVar[list[str]] = [
        *[model_name.lower() for model_name in available_model_names()],
    ]

    def is_available(self) -> bool:
        return True

    @property
    def entry_point_hook(self) -> Callable[["Namespace"], None]:
        raise NotImplementedError("Unsupported backend: pretrained")

    @property
    def deep_eval(self) -> type["DeepEvalBackend"]:
        from deepmd.pretrained.deep_eval import (
            PretrainedDeepEvalBackend,
        )

        return PretrainedDeepEvalBackend

    @property
    def neighbor_stat(self) -> type["NeighborStat"]:
        raise NotImplementedError("Unsupported backend: pretrained")

    @property
    def serialize_hook(self) -> Callable[[str], dict]:
        raise NotImplementedError("Unsupported backend: pretrained")

    @property
    def deserialize_hook(self) -> Callable[[str, dict], None]:
        raise NotImplementedError("Unsupported backend: pretrained")
