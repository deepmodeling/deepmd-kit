# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
    ClassVar,
)

import numpy as np

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
from deepmd.jax.common import (
    ArrayAPIVariable,
    flax_module,
    register_dpmodel_mapping,
    to_jax_array,
)
from deepmd.jax.env import (
    nnx,
)


class ArrayAPIParam(nnx.Param):
    def __array__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return self.value.__array__(*args, **kwargs)

    def __array_namespace__(self, *args: Any, **kwargs: Any) -> Any:
        return self.value.__array_namespace__(*args, **kwargs)

    def __dlpack__(self, *args: Any, **kwargs: Any) -> Any:
        return self.value.__dlpack__(*args, **kwargs)

    def __dlpack_device__(self, *args: Any, **kwargs: Any) -> Any:
        return self.value.__dlpack_device__(*args, **kwargs)


@flax_module
class NativeLayer(NativeLayerDP):
    _jax_skip_auto_convert_attrs: ClassVar[set[str]] = {"w", "b", "idt"}

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"w", "b", "idt"}:
            value = to_jax_array(value)
            if value is not None:
                if self.trainable:
                    value = ArrayAPIParam(value)
                else:
                    value = ArrayAPIVariable(value)
        return super().__setattr__(name, value)


@flax_module
class NativeNet(make_multilayer_network(NativeLayer, NativeOP)):
    pass


class EmbeddingNet(make_embedding_network(NativeNet, NativeLayer)):
    pass


class FittingNet(make_fitting_network(EmbeddingNet, NativeNet, NativeLayer)):
    pass


@flax_module
class NetworkCollection(NetworkCollectionDP):
    _jax_data_list_attrs: ClassVar[set[str]] = {"_networks"}

    NETWORK_TYPE_MAP: ClassVar[dict[str, type]] = {
        "network": NativeNet,
        "embedding_network": EmbeddingNet,
        "fitting_network": FittingNet,
    }


class LayerNorm(LayerNormDP, NativeLayer):
    pass


@flax_module
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
