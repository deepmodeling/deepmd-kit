# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.utils.type_embed import TypeEmbedNet as TypeEmbedNetDP

from ..common import (
    tf2_module,
)
from . import network as _tf2_network  # noqa: F401


@tf2_module
class TypeEmbedNet(TypeEmbedNetDP):
    pass
