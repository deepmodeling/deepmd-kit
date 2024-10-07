# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
    ClassVar,
)

from deepmd.dpmodel.common import (
    NativeOP,
)
from deepmd.dpmodel.utils.network import LayerNorm as LayerNormDP
from deepmd.dpmodel.utils.network import NativeLayer as NativeLayerDP
from deepmd.dpmodel.utils.network import NetworkCollection as NetworkCollectionDP
from deepmd.dpmodel.utils.network import (
    make_embedding_network,
    make_fitting_network,
    make_multilayer_network,
)

from ..common import (
    to_array_api_strict_array,
)


class NativeLayer(NativeLayerDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"w", "b", "idt"}:
            value = to_array_api_strict_array(value)
        return super().__setattr__(name, value)


NativeNet = make_multilayer_network(NativeLayer, NativeOP)
EmbeddingNet = make_embedding_network(NativeNet, NativeLayer)
FittingNet = make_fitting_network(EmbeddingNet, NativeNet, NativeLayer)


class NetworkCollection(NetworkCollectionDP):
    NETWORK_TYPE_MAP: ClassVar[dict[str, type]] = {
        "network": NativeNet,
        "embedding_network": EmbeddingNet,
        "fitting_network": FittingNet,
    }


class LayerNorm(LayerNormDP, NativeLayer):
    pass
