# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.descriptor.dpa3 import DescrptDPA3 as DescrptDPA3DP

from ..common import (
    to_array_api_strict_array,
)
from ..utils.type_embed import (
    TypeEmbedNet,
)
from .base_descriptor import (
    BaseDescriptor,
)
from .repflows import (
    DescrptBlockRepflows,
)


@BaseDescriptor.register("dpa3")
class DescrptDPA3(DescrptDPA3DP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"mean", "stddev"}:
            value = to_array_api_strict_array(value)
        elif name in {"repflows"}:
            value = DescrptBlockRepflows.deserialize(value.serialize())
        elif name in {"type_embedding"}:
            value = TypeEmbedNet.deserialize(value.serialize())
        else:
            pass
        return super().__setattr__(name, value)
