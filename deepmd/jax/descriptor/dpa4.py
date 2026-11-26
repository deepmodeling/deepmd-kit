# SPDX-License-Identifier: LGPL-3.0-or-later
import math
from collections.abc import (
    Mapping,
    Sequence,
)
from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel.common import (
    NativeOP,
)
from deepmd.dpmodel.descriptor.dpa4 import DescrptDPA4 as DescrptDPA4DP
from deepmd.dpmodel.descriptor.dpa4_nn.activation import (
    GatedActivation as GatedActivationDP,
)
from deepmd.dpmodel.descriptor.dpa4_nn.activation import SwiGLU as SwiGLUDP
from deepmd.dpmodel.descriptor.dpa4_nn.grid_net import GridProduct as GridProductDP
from deepmd.dpmodel.descriptor.dpa4_nn.radial import (
    C3CutoffEnvelope as C3CutoffEnvelopeDP,
)
from deepmd.dpmodel.descriptor.dpa4_nn.radial import RadialMLP as RadialMLPDP
from deepmd.dpmodel.descriptor.dpa4_nn.so2 import SO2Convolution as SO2ConvolutionDP
from deepmd.dpmodel.descriptor.dpa4_nn.so2 import SO2Linear as SO2LinearDP
from deepmd.dpmodel.descriptor.dpa4_nn.wignerd import (
    WignerDCalculator as WignerDCalculatorDP,
)
from deepmd.dpmodel.train.inference import (
    infer_amp_enabled,
)
from deepmd.dpmodel.utils.expert_route import (
    reject_moe,
)
from deepmd.dpmodel.utils.network import (
    softplus_t,
)
from deepmd.jax.common import (
    ArrayAPIVariable,
    flax_module,
    register_dpmodel_mapping,
    set_variable_value,
    to_jax_array,
    try_convert_module,
)
from deepmd.jax.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.jax.env import (
    jax,
    jnp,
    nnx,
)
from deepmd.jax.utils.network import (
    ArrayAPIParam,
)
from deepmd.jax.utils.precision import (
    autocast_matmul,
)


@jax.custom_jvp
def _sigmoid_gate(values: jax.Array, logits: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return gated values and the sigmoid state used by their derivatives."""
    weights = jax.nn.sigmoid(logits)
    return values * weights, weights


@_sigmoid_gate.defjvp
def _sigmoid_gate_jvp(
    primals: tuple[jax.Array, jax.Array],
    tangents: tuple[jax.Array, jax.Array],
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    values, logits = primals
    values_dot, logits_dot = tangents
    output, weights = _sigmoid_gate(values, logits)
    # For y = x * sigmoid(z), dy = sigmoid(z) * dx + y * (1-sigmoid(z)) * dz.
    # Expressing the tangent through y reuses the output activation instead
    # of retaining another edge-sized value array for higher-order derivatives.
    factor = (1.0 - weights) * logits_dot
    return (output, weights), (
        weights * values_dot + output * factor,
        weights * factor,
    )


@flax_module
class GatedActivation(GatedActivationDP):
    def _apply_vector_gate(
        self, values: jax.Array, logits: jax.Array, dtype: Any
    ) -> jax.Array:
        if values.dtype == logits.dtype == dtype and dtype in (
            jnp.float32,
            jnp.float64,
        ):
            return _sigmoid_gate(values, self._expand_gates(logits))[0]
        return super()._apply_vector_gate(values, logits, dtype)


register_dpmodel_mapping(
    GatedActivationDP, lambda value: GatedActivation.deserialize(value.serialize())
)


@flax_module
class SO2Convolution(SO2ConvolutionDP):
    def _attention_softmax(
        self, attn_logits: jax.Array, edge_cache: Any, n_nodes: int
    ) -> jax.Array:
        """Normalize edge and null masses with a constant stabilization shift."""
        shape = attn_logits.shape
        n_edge, n_channel = shape[0], math.prod(shape[1:])
        dtype = jnp.dtype(self.compute_precision)
        logits = attn_logits.astype(dtype).reshape(n_edge, n_channel)
        envelope = edge_cache.edge_env.astype(dtype).reshape(n_edge)
        dst = edge_cache.dst.astype(jnp.int64)

        # === Step 1. Physical edge and null log masses ===
        active = envelope > 0.0
        log_weight = 2.0 * jnp.log(jnp.where(active, envelope, jnp.ones_like(envelope)))
        if edge_cache.edge_mask is not None:
            active = active & (edge_cache.edge_mask.reshape(n_edge) > 0.0)
        effective = jnp.where(active[:, None], logits + log_weight[:, None], -jnp.inf)
        null_mass = softplus_t(self.adamw_attn_z_bias_raw[...].astype(dtype)) + self.eps
        null_logit = jnp.log(null_mass.reshape(1, n_channel))

        # === Step 2. Destination normalization in a shared shifted frame ===
        zeros = jnp.zeros((n_nodes, n_channel), dtype=dtype)
        maximum = (zeros + null_logit).at[dst].max(effective)
        # The same shift multiplies every edge and the null mass. Its
        # derivative cancels from their ratio, including higher derivatives.
        maximum = jax.lax.stop_gradient(maximum)
        numerator = jnp.exp(effective - jnp.take(maximum, dst, axis=0))
        denominator = zeros.at[dst].add(numerator) + jnp.exp(null_logit - maximum)
        return (numerator / jnp.take(denominator, dst, axis=0)).reshape(shape)


register_dpmodel_mapping(
    SO2ConvolutionDP, lambda value: SO2Convolution.deserialize(value.serialize())
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


def _as_parameter_variable(value: Any, *, trainable: bool) -> Any:
    """Track a floating parameter with the requested optimizer visibility."""
    variable_type = ArrayAPIParam if trainable else ArrayAPIVariable
    if type(value) is variable_type:
        return value
    if not _is_floating_array(value):
        return value
    if isinstance(value, nnx.Variable):
        value = value.value
    if isinstance(value, np.ndarray):
        value = to_jax_array(value)
    return variable_type(value)


def _as_parameter_variable_list(value: Any, *, trainable: bool) -> Any:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return value
    promoted = []
    changed = False
    for item in value:
        new_item = _as_parameter_variable(item, trainable=trainable)
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


def _promote_parameters(
    module: Any, names: tuple[str, ...], *, trainable: bool
) -> None:
    for name in names:
        if not hasattr(module, name):
            continue
        value = getattr(module, name)
        new_value = _as_parameter_variable(value, trainable=trainable)
        if new_value is not value:
            setattr(module, name, new_value)


def _promote_parameter_lists(
    module: Any, names: tuple[str, ...], *, trainable: bool
) -> None:
    for name in names:
        if not hasattr(module, name):
            continue
        value = getattr(module, name)
        new_value = _as_parameter_variable_list(value, trainable=trainable)
        if new_value is not value:
            setattr(module, name, new_value)


def _promote_trainable_tree(module: Any) -> Any:
    root_trainable = bool(getattr(module, "trainable", True))
    for submodule in _iter_object_tree(module):
        # A frozen descriptor freezes every descendant regardless of its local flag.
        trainable = root_trainable and bool(getattr(submodule, "trainable", True))
        names = _TRAINABLE_ATTRS.get(type(submodule).__name__)
        if names is not None:
            _promote_parameters(submodule, names, trainable=trainable)
        list_names = _TRAINABLE_LIST_ATTRS.get(type(submodule).__name__)
        if list_names is not None:
            _promote_parameter_lists(submodule, list_names, trainable=trainable)
    return module


@flax_module
class SO2Linear(SO2LinearDP):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.weight_m = _as_parameter_variable_list(
            self.weight_m, trainable=bool(self.trainable)
        )

    @classmethod
    def deserialize(cls, data: dict) -> "SO2Linear":
        obj = super().deserialize(data)
        obj.weight_m = _as_parameter_variable_list(
            obj.weight_m, trainable=bool(obj.trainable)
        )
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
        reject_moe(kwargs, "DPA4 descriptor")
        super().__init__(*args, **kwargs)
        self.deterministic = True
        rng_seed = int(np.random.SeedSequence(self.seed).generate_state(1)[0])
        self.rotation_key = ArrayAPIVariable(jax.random.PRNGKey(rng_seed))
        self.use_amp_infer = infer_amp_enabled()
        _promote_trainable_tree(self)

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptDPA4":
        from deepmd.jax.descriptor.lora import (
            restore_lora_from_variables,
        )

        obj = super().deserialize(data)
        _promote_trainable_tree(obj)
        restore_lora_from_variables(obj, data["@variables"])
        return obj

    def serialize(self) -> dict[str, Any]:
        from deepmd.jax.descriptor.lora import (
            add_lora_scaling,
        )

        data = super().serialize()
        add_lora_scaling(self, data["@variables"])
        return data

    def adam_route_patterns(self) -> list[str]:
        """Route the sparsely sampled radial-input projections through AdamW."""
        return [
            "radial_embedding.net.0.",
            "env_seed_embedding.rbf_proj_layer1.",
        ]

    def _in_training_mode(self) -> bool:
        return not self.deterministic

    def advance_random_state(self) -> None:
        """Advance augmentation once before differentiating a training step.

        All coordinate and parameter derivatives in that step read the same
        key. Sampling therefore remains pure inside nested JAX transforms,
        including differentiation through forces and Hessians.
        """
        if self.random_gamma and not self.deterministic:
            set_variable_value(
                self.rotation_key, jax.random.split(self.rotation_key[...])[0]
            )

    def _rotation_angles(self, edge_vec: jax.Array) -> jax.Array:
        return jax.random.uniform(
            self.rotation_key[...],
            (edge_vec.shape[0],),
            dtype=edge_vec.dtype,
            maxval=2.0 * math.pi,
        )

    def _forward_blocks(
        self,
        x: jax.Array,
        edge_cache: Any,
        radial_feat_per_block: list[jax.Array],
        comm_dict: dict[str, jax.Array] | None = None,
        unit_routes: list[Any] | None = None,
    ) -> jax.Array:
        """Use BF16 multiplicands with FP32 training activations.

        Master parameters retain their declared precision. Normalization
        accumulates in its declared precision and restores the input dtype;
        FP32 activations preserve precision through force-loss derivatives.
        Residual addition follows its operand dtypes. Geometry, radial
        features and the readout lie outside this region. Inference retains
        its BF16 activation-storage policy when AMP inference is requested.
        Float64 models retain float64 arithmetic in every stage.
        """
        enabled = self.use_amp_infer if self.deterministic else self.use_amp
        if enabled and x.dtype == jnp.float32:
            with autocast_matmul(fp32_output=not self.deterministic):
                return super()._forward_blocks(
                    x,
                    edge_cache,
                    radial_feat_per_block,
                    comm_dict=comm_dict,
                    unit_routes=unit_routes,
                )
        return super()._forward_blocks(
            x,
            edge_cache,
            radial_feat_per_block,
            comm_dict=comm_dict,
            unit_routes=unit_routes,
        )
