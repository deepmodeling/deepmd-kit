# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.descriptor.se_t_tebd import (
    DescrptBlockSeTTebd as DescrptBlockSeTTebdDP,
)
from deepmd.dpmodel.descriptor.se_t_tebd import DescrptSeTTebd as DescrptSeTTebdDP
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
from deepmd.jax.utils.type_embed import (
    TypeEmbedNet,
)


@flax_module
class DescrptBlockSeTTebd(DescrptBlockSeTTebdDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"mean", "stddev"}:
            value = to_jax_array(value)
            if value is not None:
                value = ArrayAPIVariable(value)
        elif name in {"embeddings", "embeddings_strip"}:
            if value is not None:
                value = NetworkCollection.deserialize(value.serialize())
        elif name == "env_mat":
            # env_mat doesn't store any value
            pass
        elif name == "emask":
            value = PairExcludeMask(value.ntypes, value.exclude_types)

        return super().__setattr__(name, value)


@BaseDescriptor.register("se_e3_tebd")
@flax_module
class DescrptSeTTebd(DescrptSeTTebdDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name == "se_ttebd":
            value = DescrptBlockSeTTebd.deserialize(value.serialize())
        elif name == "type_embedding":
            value = TypeEmbedNet.deserialize(value.serialize())
        return super().__setattr__(name, value)
