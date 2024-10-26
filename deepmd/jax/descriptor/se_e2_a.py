# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.descriptor.se_e2_a import DescrptSeAArrayAPI as DescrptSeADP
from deepmd.jax.common import (
    ArrayAPIVariable,
    flax_module,
    to_jax_array,
)
from deepmd.jax.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.jax.utils.exclude_mask import (
    PairExcludeMask,
)
from deepmd.jax.utils.network import (
    NetworkCollection,
)


@BaseDescriptor.register("se_e2_a")
@BaseDescriptor.register("se_a")
@flax_module
class DescrptSeA(DescrptSeADP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"dstd", "davg"}:
            value = to_jax_array(value)
            if value is not None:
                value = ArrayAPIVariable(value)
        elif name in {"embeddings"}:
            if value is not None:
                value = NetworkCollection.deserialize(value.serialize())
        elif name == "env_mat":
            # env_mat doesn't store any value
            pass
        elif name == "emask":
            value = PairExcludeMask(value.ntypes, value.exclude_types)

        return super().__setattr__(name, value)
