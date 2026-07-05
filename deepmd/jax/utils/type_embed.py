# SPDX-License-Identifier: LGPL-3.0-or-later
import deepmd.jax.utils.network as _jax_network  # noqa: F401
from deepmd.dpmodel.utils.type_embed import TypeEmbedNet as TypeEmbedNetDP
from deepmd.jax.common import (
    flax_module,
)


@flax_module
class TypeEmbedNet(TypeEmbedNetDP):
    pass
