# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.descriptor.dpa3 import DescrptDPA3 as DescrptDPA3DP
from deepmd.jax.common import (
    ArrayAPIVariable,
    flax_module,
    to_jax_array,
)
from deepmd.jax.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.jax.descriptor.repflows import (
    DescrptBlockRepflows,
)
from deepmd.jax.utils.type_embed import (
    TypeEmbedNet,
)


@BaseDescriptor.register("dpa3")
@flax_module
class DescrptDPA3(DescrptDPA3DP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"mean", "stddev"}:
            value = to_jax_array(value)
            if value is not None:
                value = ArrayAPIVariable(value)
        elif name in {"repflows"}:
            value = DescrptBlockRepflows.deserialize(value.serialize())
        elif name in {"type_embedding"}:
            value = TypeEmbedNet.deserialize(value.serialize())
        else:
            pass
        return super().__setattr__(name, value)
