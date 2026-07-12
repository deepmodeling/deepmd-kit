# SPDX-License-Identifier: LGPL-3.0-or-later
from collections.abc import (
    Mapping,
    Sequence,
)
from typing import (
    Any,
)

import numpy as np

from deepmd._vendors import ndtensorflow as xp
from deepmd.dpmodel.common import (
    NativeOP,
)
from deepmd.dpmodel.descriptor.dpa4 import DescrptDPA4 as DescrptDPA4DP
from deepmd.dpmodel.descriptor.dpa4_nn.activation import SwiGLU as SwiGLUDP
from deepmd.dpmodel.descriptor.dpa4_nn.grid_net import GridProduct as GridProductDP
from deepmd.dpmodel.descriptor.dpa4_nn.radial import (
    C3CutoffEnvelope as C3CutoffEnvelopeDP,
)
from deepmd.dpmodel.descriptor.dpa4_nn.radial import RadialMLP as RadialMLPDP
from deepmd.dpmodel.descriptor.dpa4_nn.so2 import SO2Linear as SO2LinearDP
from deepmd.dpmodel.descriptor.dpa4_nn.so2 import (
    DynamicRadialDegreeMixer as DynamicRadialDegreeMixerDP,
)
from deepmd.dpmodel.descriptor.dpa4_nn.wignerd import (
    WignerDCalculator as WignerDCalculatorDP,
)
from deepmd.tf2.common import (
    register_dpmodel_mapping,
    tf,
    tf2_module,
    to_tf_tensor,
    try_convert_module,
)
from deepmd.tf2.descriptor.base_descriptor import (
    BaseDescriptor,
)


@tf2_module
class SwiGLU(SwiGLUDP):
    pass


register_dpmodel_mapping(SwiGLUDP, lambda v: SwiGLU())


@tf2_module
class C3CutoffEnvelope(C3CutoffEnvelopeDP):
    pass


register_dpmodel_mapping(
    C3CutoffEnvelopeDP,
    lambda v: C3CutoffEnvelope(v.rcut, v.p, precision=v.precision),
)


@tf2_module
class RadialMLP(RadialMLPDP):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.net = [self._convert_layer(layer) for layer in self.net]
        self._tracked_net_modules = [
            layer for layer in self.net if isinstance(layer, tf.Module)
        ]

    @staticmethod
    def _convert_layer(layer: Any) -> Any:
        if isinstance(layer, tf.Module):
            return layer
        if isinstance(layer, NativeOP):
            converted = try_convert_module(layer)
            if converted is not None:
                return converted
        return layer


register_dpmodel_mapping(
    RadialMLPDP,
    lambda v: RadialMLP.deserialize(v.serialize()),
)


@tf2_module
class GridProduct(GridProductDP):
    pass


register_dpmodel_mapping(GridProductDP, lambda v: GridProduct())


@tf2_module
class WignerDCalculator(WignerDCalculatorDP):
    pass


register_dpmodel_mapping(
    WignerDCalculatorDP,
    lambda v: WignerDCalculator(v.lmax, eps=v.eps, precision=v.precision),
)


_TRAINABLE_ATTRS: dict[str, tuple[str, ...]] = {
    "RMSNorm": ("adam_scale",),
    "EquivariantRMSNorm": ("adam_scale", "bias"),
    "ReducedEquivariantRMSNorm": ("adam_scale", "bias0"),
    "ScalarRMSNorm": ("adam_scale",),
    "RadialBasis": ("adam_freqs",),
    "SO3Linear": ("weight", "bias"),
    "FocusLinear": ("weight", "bias"),
    "ChannelLinear": ("weight", "bias"),
    "FrameContract": ("weight",),
    "FrameExpand": ("weight",),
    "SO2Linear": ("weight_m0", "bias0"),
    "DynamicRadialDegreeMixer": ("weight", "channel_basis"),
    "SO2Convolution": (
        "adamw_attn_logit_w",
        "adamw_attn_z_bias_raw",
        "adamw_attn_gate_w",
        "adamw_focus_compete_w",
        "focus_compete_bias",
    ),
    "SeZMTypeEmbedding": ("adam_type_embedding",),
    "SpinEmbedding": ("adam_spin_vec_weight", "adam_spin_nbr_weight"),
    "EnvironmentInitialEmbedding": ("spin_scale",),
    "DepthAttnRes": ("adamw_pseudo_query",),
    "S2GridNet": ("residual_scale",),
    "SO3GridNet": ("residual_scale",),
    "DescrptDPA4": ("film_scale_strength_log", "film_shift_strength_log"),
}

_TRAINABLE_LIST_ATTRS: dict[str, tuple[str, ...]] = {
    "SeZMInteractionBlock": ("adam_ffn_layer_scales",),
    "SO2Linear": ("weight_m",),
    "SO2Convolution": ("adam_so2_layer_scales",),
}


def _is_array_like(value: Any) -> bool:
    return isinstance(value, (np.ndarray, tf.Tensor, tf.Variable, xp.Array))


def _is_floating_array(value: Any) -> bool:
    tensor = to_tf_tensor(value)
    return tensor is not None and tensor.dtype.is_floating


def _iter_object_tree(root: Any) -> Any:
    seen: set[int] = set()

    def visit(value: Any) -> Any:
        if value is None or isinstance(value, (str, bytes, int, float, bool)):
            return
        if _is_array_like(value):
            return
        value_id = id(value)
        if value_id in seen:
            return
        seen.add(value_id)

        if isinstance(value, Mapping):
            for item in value.values():
                yield from visit(item)
            return
        if isinstance(value, Sequence):
            for item in value:
                yield from visit(item)
            return
        try:
            value_dict = object.__getattribute__(value, "__dict__")
        except AttributeError:
            return

        yield value
        for item in value_dict.values():
            yield from visit(item)

    yield from visit(root)


def _enable_tf2_parameter_attr(module: Any, name: str, *, trainable: bool) -> None:
    attrs = set(getattr(module, "_tf2_array_variable_attrs", ()))
    if name not in attrs:
        tf.Module.__setattr__(module, "_tf2_array_variable_attrs", attrs | {name})
    policies = dict(getattr(module, "_tf2_array_variable_trainable", {}))
    policies[name] = trainable
    tf.Module.__setattr__(module, "_tf2_array_variable_trainable", policies)


def _enable_tf2_parameter_list_attr(module: Any, name: str, *, trainable: bool) -> None:
    attrs = set(getattr(module, "_tf2_array_variable_list_attrs", ()))
    if name not in attrs:
        tf.Module.__setattr__(
            module,
            "_tf2_array_variable_list_attrs",
            attrs | {name},
        )
    policies = dict(getattr(module, "_tf2_array_variable_list_trainable", {}))
    policies[name] = trainable
    tf.Module.__setattr__(module, "_tf2_array_variable_list_trainable", policies)


def _promote_parameters(
    module: Any, names: tuple[str, ...], *, trainable: bool
) -> None:
    for name in names:
        if not hasattr(module, name):
            continue
        value = getattr(module, name)
        if not _is_floating_array(value):
            continue
        _enable_tf2_parameter_attr(module, name, trainable=trainable)
        setattr(module, name, value)


def _promote_parameter_lists(
    module: Any, names: tuple[str, ...], *, trainable: bool
) -> None:
    for name in names:
        if not hasattr(module, name):
            continue
        value = getattr(module, name)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            continue
        if not value or not all(_is_floating_array(item) for item in value):
            continue
        _enable_tf2_parameter_list_attr(module, name, trainable=trainable)
        setattr(module, name, value)


def _promote_trainable_tree(module: Any) -> Any:
    root_trainable = bool(getattr(module, "trainable", True))
    for submodule in _iter_object_tree(module):
        trainable = root_trainable and bool(getattr(submodule, "trainable", True))
        names = _TRAINABLE_ATTRS.get(type(submodule).__name__)
        if names is not None:
            _promote_parameters(submodule, names, trainable=trainable)
        list_names = _TRAINABLE_LIST_ATTRS.get(type(submodule).__name__)
        if list_names is not None:
            _promote_parameter_lists(submodule, list_names, trainable=trainable)
    return module


@tf2_module
class SO2Linear(SO2LinearDP):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        _promote_parameter_lists(self, ("weight_m",), trainable=bool(self.trainable))

    @classmethod
    def deserialize(cls, data: dict) -> "SO2Linear":
        obj = super().deserialize(data)
        _promote_parameter_lists(obj, ("weight_m",), trainable=bool(obj.trainable))
        return obj


register_dpmodel_mapping(
    SO2LinearDP,
    lambda v: SO2Linear.deserialize(v.serialize()),
)


@tf2_module
class DynamicRadialDegreeMixer(DynamicRadialDegreeMixerDP):
    """TF2 mixer with runtime shape checks for generalized edge counts."""

    def call(self, x_local: Any, radial_feat: Any) -> Any:
        x_tensor = to_tf_tensor(x_local)
        radial_tensor = to_tf_tensor(radial_feat)
        assertion = tf.debugging.assert_equal(
            tf.shape(x_tensor),
            tf.shape(radial_tensor),
            message="x_local and radial_feat must have the same shape",
        )
        with tf.control_dependencies([assertion]):
            return super().call(
                xp.asarray(tf.identity(x_tensor)),
                xp.asarray(tf.identity(radial_tensor)),
            )


register_dpmodel_mapping(
    DynamicRadialDegreeMixerDP,
    lambda v: DynamicRadialDegreeMixer.deserialize(v.serialize()),
)


@BaseDescriptor.register("SeZM")
@BaseDescriptor.register("sezm")
@BaseDescriptor.register("DPA4")
@BaseDescriptor.register("dpa4")
@tf2_module
class DescrptDPA4(DescrptDPA4DP):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.random_gamma:
            raise NotImplementedError(
                "TF2 DPA4 does not yet support graph-safe random_gamma; "
                "set descriptor.random_gamma to false."
            )
        _promote_trainable_tree(self)

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptDPA4":
        obj = super().deserialize(data)
        return _promote_trainable_tree(obj)
