# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.utils.type_embed import TypeEmbedNet as TypeEmbedNetDP

from ..common import (
    to_array_api_strict_array,
)
from ..utils.network import (
    EmbeddingNet,
)


class TypeEmbedNet(TypeEmbedNetDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"econf_tebd"}:
            value = to_array_api_strict_array(value)
        if name in {"embedding_net"}:
            value = EmbeddingNet.deserialize(value.serialize())
        return super().__setattr__(name, value)
