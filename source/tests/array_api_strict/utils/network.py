# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
    ClassVar,
)

from deepmd.dpmodel.common import (
    NativeOP,
)
from deepmd.dpmodel.utils.network import EmbeddingNet as EmbeddingNetDP
from deepmd.dpmodel.utils.network import FittingNet as FittingNetDP
from deepmd.dpmodel.utils.network import Identity as IdentityDP
from deepmd.dpmodel.utils.network import LayerNorm as LayerNormDP
from deepmd.dpmodel.utils.network import NativeLayer as NativeLayerDP
from deepmd.dpmodel.utils.network import NativeNet as NativeNetDP
from deepmd.dpmodel.utils.network import NetworkCollection as NetworkCollectionDP
from deepmd.dpmodel.utils.network import (
    make_embedding_network,
    make_fitting_network,
    make_multilayer_network,
)

from ..common import (
    array_api_strict_module,
    register_dpmodel_mapping,
    to_array_api_strict_array,
)


@array_api_strict_module
class NativeLayer(NativeLayerDP):
    _array_api_strict_skip_auto_convert_attrs: ClassVar[set[str]] = {"w", "b", "idt"}

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"w", "b", "idt"}:
            value = to_array_api_strict_array(value)
        return super().__setattr__(name, value)


@array_api_strict_module
class NativeNet(make_multilayer_network(NativeLayer, NativeOP)):
    pass


class EmbeddingNet(make_embedding_network(NativeNet, NativeLayer)):
    pass


class FittingNet(make_fitting_network(EmbeddingNet, NativeNet, NativeLayer)):
    pass


@array_api_strict_module
class NetworkCollection(NetworkCollectionDP):
    NETWORK_TYPE_MAP: ClassVar[dict[str, type]] = {
        "network": NativeNet,
        "embedding_network": EmbeddingNet,
        "fitting_network": FittingNet,
    }


class LayerNorm(LayerNormDP, NativeLayer):
    pass


@array_api_strict_module
class Identity(IdentityDP):
    pass


register_dpmodel_mapping(
    NativeNetDP,
    lambda v: NativeNet.deserialize(v.serialize()),
)

register_dpmodel_mapping(
    EmbeddingNetDP,
    lambda v: EmbeddingNet.deserialize(v.serialize()),
)

register_dpmodel_mapping(
    FittingNetDP,
    lambda v: FittingNet.deserialize(v.serialize()),
)

register_dpmodel_mapping(
    NativeLayerDP,
    lambda v: NativeLayer.deserialize(v.serialize()),
)

register_dpmodel_mapping(
    LayerNormDP,
    lambda v: LayerNorm.deserialize(v.serialize()),
)

register_dpmodel_mapping(
    NetworkCollectionDP,
    lambda v: NetworkCollection.deserialize(v.serialize()),
)

register_dpmodel_mapping(
    IdentityDP,
    lambda v: Identity(),
)
