# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
    ClassVar,
)

import tensorflow as tf

from deepmd.dpmodel.common import (
    PRECISION_DICT,
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
    register_dpmodel_mapping,
    tf2_module,
    to_tensorflow_array,
    to_tf_tensor,
)


class NativeLayer(NativeLayerDP, tf.Module):
    _tf2_variable_attrs: ClassVar[set[str]] = {"w", "b", "idt"}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        tf.Module.__init__(self)
        NativeLayerDP.__init__(self, *args, **kwargs)

    @staticmethod
    def _tf2_variable_storage_name(name: str) -> str:
        return f"_tf2_{name}_variable"

    def _get_tf2_variable(self, name: str) -> tf.Variable | None:
        return getattr(self, self._tf2_variable_storage_name(name), None)

    def _get_tf2_variable_array(self, name: str) -> Any | None:
        variable = self._get_tf2_variable(name)
        return None if variable is None else to_tensorflow_array(variable)

    def _set_tf2_variable(self, name: str, value: Any) -> None:
        storage_name = self._tf2_variable_storage_name(name)
        if value is None:
            tf.Module.__setattr__(self, storage_name, None)
            return
        tensor = to_tf_tensor(value)
        variable = tf.Variable(
            tensor,
            trainable=bool(getattr(self, "trainable", True)),
            name=name,
        )
        tf.Module.__setattr__(self, storage_name, variable)

    def _refresh_tf2_variable_trainability(self) -> None:
        for name in self._tf2_variable_attrs:
            variable = self._get_tf2_variable(name)
            if variable is not None and variable.trainable != self.trainable:
                self._set_tf2_variable(name, variable.read_value())

    @property
    def w(self) -> Any | None:
        return self._get_tf2_variable_array("w")

    @property
    def b(self) -> Any | None:
        return self._get_tf2_variable_array("b")

    @property
    def idt(self) -> Any | None:
        return self._get_tf2_variable_array("idt")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._tf2_variable_attrs:
            self._set_tf2_variable(name, value)
            return
        tf.Module.__setattr__(self, name, value)
        if name == "trainable":
            self._refresh_tf2_variable_trainability()

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
