# SPDX-License-Identifier: LGPL-3.0-or-later
import array_api_compat

import deepmd.jax.utils.network as _jax_network  # noqa: F401
from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.utils.type_embed import TypeEmbedNet as TypeEmbedNetDP
from deepmd.jax.common import (
    flax_module,
)


@flax_module
class TypeEmbedNet(TypeEmbedNetDP):
    def call(self) -> Array:
        """Compute type embeddings without querying tracer devices."""
        sample_array = self.embedding_net[0]["w"]
        xp = array_api_compat.array_namespace(sample_array)
        if not self.use_econf_tebd:
            embed = self.embedding_net(
                xp.eye(
                    self.ntypes,
                    dtype=sample_array.dtype,
                    device=None,
                )
            )
        else:
            embed = self.embedding_net(self.econf_tebd)
        if self.padding:
            embed_pad = xp.zeros(
                (1, embed.shape[-1]),
                dtype=embed.dtype,
                device=None,
            )
            embed = xp.concat([embed, embed_pad], axis=0)
        return embed
