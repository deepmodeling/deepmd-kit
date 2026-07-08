# SPDX-License-Identifier: LGPL-3.0-or-later
from collections.abc import (
    Mapping,
    Sequence,
)
from typing import (
    Any,
)

import numpy as np

import deepmd.jax.utils.exclude_mask as _jax_exclude_mask  # noqa: F401
import deepmd.jax.utils.network as _jax_network  # noqa: F401
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
from deepmd.dpmodel.descriptor.dpa4_nn.wignerd import (
    WignerDCalculator as WignerDCalculatorDP,
)
from deepmd.jax.common import (
    flax_module,
    register_dpmodel_mapping,
    to_jax_array,
    try_convert_module,
)
from deepmd.jax.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.jax.env import (
    jnp,
    nnx,
)
from deepmd.jax.utils.network import (
    ArrayAPIParam,
)


@flax_module
class SwiGLU(SwiGLUDP):
    pass


register_dpmodel_mapping(SwiGLUDP, lambda v: SwiGLU())


@flax_module
class C3CutoffEnvelope(C3CutoffEnvelopeDP):
    pass


register_dpmodel_mapping(
    C3CutoffEnvelopeDP,
    lambda v: C3CutoffEnvelope(v.rcut, v.p, precision=v.precision),
)


@flax_module
class RadialMLP(RadialMLPDP):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        converted = [self._convert_layer(layer) for layer in self.net]
        self.net = nnx.List(converted) if hasattr(nnx, "List") else converted

    @staticmethod
    def _convert_layer(layer: Any) -> Any:
        if isinstance(layer, nnx.Module):
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


@flax_module
class GridProduct(GridProductDP):
    pass


register_dpmodel_mapping(GridProductDP, lambda v: GridProduct())


@flax_module
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
    "SO2Linear": ("weight_m",),
    "SO2Convolution": ("adam_so2_layer_scales",),
}


def _is_array_like(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype")


def _array_value(value: Any) -> Any:
    if isinstance(value, nnx.Variable):
        return value.value
    return value


def _is_floating_array(value: Any) -> bool:
    value = _array_value(value)
    if value is None or not _is_array_like(value):
        return False
    return bool(jnp.issubdtype(value.dtype, jnp.floating))


def _as_param(value: Any) -> Any:
    if isinstance(value, ArrayAPIParam):
        return value
    if not _is_floating_array(value):
        return value
    if isinstance(value, nnx.Variable):
        return ArrayAPIParam(value.value)
    if isinstance(value, np.ndarray):
        return ArrayAPIParam(to_jax_array(value))
    return ArrayAPIParam(value)


def _as_param_list(value: Any) -> Any:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return value
    promoted = []
    changed = False
    for item in value:
        new_item = _as_param(item)
        promoted.append(new_item)
        changed = changed or new_item is not item
    if not changed:
        return value
    return nnx.List(promoted) if hasattr(nnx, "List") else promoted


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


def _promote_trainable(module: Any, names: tuple[str, ...]) -> None:
    if not getattr(module, "trainable", True):
        return
    for name in names:
        if not hasattr(module, name):
            continue
        value = getattr(module, name)
        new_value = _as_param(value)
        if new_value is not value:
            setattr(module, name, new_value)


def _promote_trainable_lists(module: Any, names: tuple[str, ...]) -> None:
    if not getattr(module, "trainable", True):
        return
    for name in names:
        if not hasattr(module, name):
            continue
        value = getattr(module, name)
        new_value = _as_param_list(value)
        if new_value is not value:
            setattr(module, name, new_value)


def _promote_trainable_tree(module: Any) -> Any:
    for submodule in _iter_object_tree(module):
        names = _TRAINABLE_ATTRS.get(type(submodule).__name__)
        if names is not None:
            _promote_trainable(submodule, names)
        list_names = _TRAINABLE_LIST_ATTRS.get(type(submodule).__name__)
        if list_names is not None:
            _promote_trainable_lists(submodule, list_names)
    return module


@flax_module
class SO2Linear(SO2LinearDP):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.weight_m = _as_param_list(self.weight_m)

    @classmethod
    def deserialize(cls, data: dict) -> "SO2Linear":
        obj = super().deserialize(data)
        obj.weight_m = _as_param_list(obj.weight_m)
        return obj


register_dpmodel_mapping(
    SO2LinearDP,
    lambda v: SO2Linear.deserialize(v.serialize()),
)


@BaseDescriptor.register("SeZM")
@BaseDescriptor.register("sezm")
@BaseDescriptor.register("DPA4")
@BaseDescriptor.register("dpa4")
@flax_module
class DescrptDPA4(DescrptDPA4DP):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        _promote_trainable_tree(self)

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptDPA4":
        obj = super().deserialize(data)
        return _promote_trainable_tree(obj)
