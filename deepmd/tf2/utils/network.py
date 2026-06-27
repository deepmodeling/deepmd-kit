# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
    ClassVar,
)

from deepmd.dpmodel.common import (
    NativeOP,
    PRECISION_DICT,
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
    to_tf_tensor,
    tf2_module,
    register_dpmodel_mapping,
    to_tensorflow_array,
)


@tf2_module
class NativeLayer(NativeLayerDP):
    _tf2_skip_auto_convert_attrs: ClassVar[set[str]] = {"w", "b", "idt"}

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"w", "b", "idt"}:
            value = to_tensorflow_array(value)
        return super().__setattr__(name, value)

    def check_type_consistency(self) -> None:
        precision = self.precision

        def check_var(var: Any | None) -> None:
            if var is not None:
                dtype_name = getattr(var.dtype, "name", str(var.dtype).split(".")[-1])
                assert PRECISION_DICT[dtype_name] is PRECISION_DICT[precision]

        check_var(self.w)
        check_var(self.b)
        check_var(self.idt)

    def serialize(self) -> dict:
        data = super().serialize()

        def to_numpy(var: Any | None) -> Any | None:
            tensor = to_tf_tensor(var)
            return None if tensor is None else tensor.numpy()

        data["@variables"] = {
            "w": to_numpy(self.w),
            "b": to_numpy(self.b),
            "idt": to_numpy(self.idt),
        }
        return data


@tf2_module
class NativeNet(make_multilayer_network(NativeLayer, NativeOP)):
    pass


class EmbeddingNet(make_embedding_network(NativeNet, NativeLayer)):
    pass


class FittingNet(make_fitting_network(EmbeddingNet, NativeNet, NativeLayer)):
    pass


@tf2_module
class NetworkCollection(NetworkCollectionDP):
    NETWORK_TYPE_MAP: ClassVar[dict[str, type]] = {
        "network": NativeNet,
        "embedding_network": EmbeddingNet,
        "fitting_network": FittingNet,
    }


class LayerNorm(LayerNormDP, NativeLayer):
    pass


@tf2_module
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
