# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.descriptor.dpa1 import DescrptBlockSeAtten as DescrptBlockSeAttenDP
from deepmd.dpmodel.descriptor.dpa1 import DescrptDPA1 as DescrptDPA1DP
from deepmd.dpmodel.descriptor.dpa1 import GatedAttentionLayer as GatedAttentionLayerDP
from deepmd.dpmodel.descriptor.dpa1 import (
    NeighborGatedAttention as NeighborGatedAttentionDP,
)
from deepmd.dpmodel.descriptor.dpa1 import (
    NeighborGatedAttentionLayer as NeighborGatedAttentionLayerDP,
)
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
    LayerNorm,
    NativeLayer,
    NetworkCollection,
)
from deepmd.jax.utils.type_embed import (
    TypeEmbedNet,
)


@flax_module
class GatedAttentionLayer(GatedAttentionLayerDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"in_proj", "out_proj"}:
            value = NativeLayer.deserialize(value.serialize())
        return super().__setattr__(name, value)


@flax_module
class NeighborGatedAttentionLayer(NeighborGatedAttentionLayerDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name == "attention_layer":
            value = GatedAttentionLayer.deserialize(value.serialize())
        elif name == "attn_layer_norm":
            value = LayerNorm.deserialize(value.serialize())
        return super().__setattr__(name, value)


@flax_module
class NeighborGatedAttention(NeighborGatedAttentionDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name == "attention_layers":
            value = [
                NeighborGatedAttentionLayer.deserialize(ii.serialize()) for ii in value
            ]
        return super().__setattr__(name, value)


@flax_module
class DescrptBlockSeAtten(DescrptBlockSeAttenDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"mean", "stddev"}:
            value = to_jax_array(value)
            if value is not None:
                value = ArrayAPIVariable(value)
        elif name in {"embeddings", "embeddings_strip"}:
            if value is not None:
                value = NetworkCollection.deserialize(value.serialize())
        elif name == "dpa1_attention":
            value = NeighborGatedAttention.deserialize(value.serialize())
        elif name == "env_mat":
            # env_mat doesn't store any value
            pass
        elif name == "emask":
            value = PairExcludeMask(value.ntypes, value.exclude_types)

        return super().__setattr__(name, value)


@BaseDescriptor.register("dpa1")
@BaseDescriptor.register("se_atten")
@flax_module
class DescrptDPA1(DescrptDPA1DP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name == "se_atten":
            value = DescrptBlockSeAtten.deserialize(value.serialize())
        elif name == "type_embedding":
            value = TypeEmbedNet.deserialize(value.serialize())
        return super().__setattr__(name, value)
