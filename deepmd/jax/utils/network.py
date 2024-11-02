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
from deepmd.jax.common import (
    flax_module,
    to_jax_array,
)
from deepmd.jax.env import (
    nnx,
)


class ArrayAPIParam(nnx.Param):
    def __array__(self, *args, **kwargs):
        return self.value.__array__(*args, **kwargs)

    def __array_namespace__(self, *args, **kwargs):
        return self.value.__array_namespace__(*args, **kwargs)

    def __dlpack__(self, *args, **kwargs):
        return self.value.__dlpack__(*args, **kwargs)

    def __dlpack_device__(self, *args, **kwargs):
        return self.value.__dlpack_device__(*args, **kwargs)


@flax_module
class NativeLayer(NativeLayerDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"w", "b", "idt"}:
            value = to_jax_array(value)
            if value is not None:
                value = ArrayAPIParam(value)
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
    NETWORK_TYPE_MAP: ClassVar[dict[str, type]] = {
        "network": NativeNet,
        "embedding_network": EmbeddingNet,
        "fitting_network": FittingNet,
    }


class LayerNorm(LayerNormDP, NativeLayer):
    pass
