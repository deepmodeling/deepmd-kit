# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.utils.type_embed import TypeEmbedNet as TypeEmbedNetDP

from ..common import (
    array_api_strict_module,
)
from . import network as _strict_network  # noqa: F401


@array_api_strict_module
class TypeEmbedNet(TypeEmbedNetDP):
    pass
