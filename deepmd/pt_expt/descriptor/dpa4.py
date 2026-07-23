# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import torch

from deepmd.dpmodel.descriptor.dpa4 import DescrptDPA4 as DescrptDPA4DP
from deepmd.dpmodel.descriptor.dpa4_nn.activation import SwiGLU as SwiGLUDP
from deepmd.dpmodel.descriptor.dpa4_nn.grid_net import GridProduct as GridProductDP
from deepmd.dpmodel.descriptor.dpa4_nn.radial import (
    C3CutoffEnvelope as C3CutoffEnvelopeDP,
)
from deepmd.kernels.utils import (
    use_amp_infer,
)
from deepmd.pt_expt.common import (
    register_dpmodel_mapping,
    torch_module,
)
from deepmd.pt_expt.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.pt_expt.utils.update_sel import (
    UpdateSel,
)


@torch_module
class SwiGLU(SwiGLUDP):
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.call(*args, **kwargs)


# SwiGLU is parameter-free (no serialize); rebuild fresh.
register_dpmodel_mapping(SwiGLUDP, lambda v: SwiGLU())


@torch_module
class C3CutoffEnvelope(C3CutoffEnvelopeDP):
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.call(*args, **kwargs)


# C3CutoffEnvelope carries only scalar configuration (cutoff radius and
# polynomial exponent) and holds no trainable arrays, so it implements no
# serialize()/deserialize() that the generic auto-wrap path relies on; rebuild
# it directly from the stored constructor arguments (``p`` is the exponent).
register_dpmodel_mapping(
    C3CutoffEnvelopeDP,
    lambda v: C3CutoffEnvelope(v.rcut, v.p, precision=v.precision),
)


@torch_module
class GridProduct(GridProductDP):
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.call(*args, **kwargs)


# GridProduct is a parameter-free quadratic grid product with no constructor
# arguments and no serialize()/deserialize(); rebuild a fresh instance.
register_dpmodel_mapping(GridProductDP, lambda v: GridProduct())


# ---------------------------------------------------------------------------
# Trainable-weight promotion
#
# ``dpmodel_setattr`` registers every numpy attribute as a torch *buffer*, so
# the auto-wrapped dpa4_nn sub-modules would otherwise expose their trainable
# weights as non-trainable buffers (no autograd, invisible to the optimizer).
# The table below lists, per dpmodel class name, the attributes that are
# ``torch.nn.Parameter`` in the reference pt SeZM implementation
# (deepmd/pt/model/descriptor/sezm_nn).  ``_promote_trainable_tree`` walks the
# fully-built module tree and re-registers those buffers as Parameters.
#
# Constant float buffers (e.g. ``balance_weight``, ``rotate_inv_rescale_full``,
# ``mean``/``stddev``) are intentionally NOT listed: they are buffers in pt
# too.  Lists of weights (e.g. ``SO2Linear.weight_m``) are already converted
# to trainable ``ParameterList`` by ``_try_convert_list``.
# ---------------------------------------------------------------------------
_TRAINABLE_ATTRS: dict[str, tuple[str, ...]] = {
    # dpa4_nn.norm
    "RMSNorm": ("adam_scale",),
    "EquivariantRMSNorm": ("adam_scale", "bias"),
    "ReducedEquivariantRMSNorm": ("adam_scale", "bias0"),
    "ScalarRMSNorm": ("adam_scale",),
    # dpa4_nn.radial
    "RadialBasis": ("adam_freqs",),
    # dpa4_nn.so3
    "SO3Linear": ("weight", "bias"),
    "FocusLinear": ("weight", "bias"),
    "ChannelLinear": ("weight", "bias"),
    # dpa4_nn.so2
    "SO2Linear": ("weight_m0", "bias0"),
    "DynamicRadialDegreeMixer": ("weight", "channel_basis"),
    "SO2Convolution": (
        "adamw_attn_logit_w",
        "adamw_attn_z_bias_raw",
        "adamw_attn_gate_w",
        "adamw_focus_compete_w",
        "focus_compete_bias",
    ),
    # dpa4_nn.embedding
    "SeZMTypeEmbedding": ("adam_type_embedding",),
    # dpa4_nn.embedding (native spin): these are nn.Parameter in pt but land as
    # numpy->buffer in dpmodel; mag_layer1/2 are NativeLayer and auto-promote,
    # and _promote_trainable skips a missing buffer, so no-spin configs (where
    # spin_scale is absent) stay safe.
    "SpinEmbedding": ("adam_spin_vec_weight", "adam_spin_nbr_weight"),
    "EnvironmentInitialEmbedding": ("spin_scale",),
    # dpa4_nn.attn_res
    "DepthAttnRes": ("adamw_pseudo_query",),
    # dpa4_nn.grid_net (residual_scale is None when disabled; _promote_trainable
    # skips the missing buffer, so listing both concrete subclasses is safe)
    "S2GridNet": ("residual_scale",),
    "SO3GridNet": ("residual_scale",),
    # descriptor-level FiLM strengths
    "DescrptDPA4": ("film_scale_strength_log", "film_shift_strength_log"),
}


def _promote_trainable(module: torch.nn.Module, names: tuple[str, ...]) -> None:
    """Re-register the given float buffers of *module* as Parameters."""
    if not getattr(module, "trainable", True):
        return
    for name in names:
        buf = module._buffers.get(name)
        if buf is None or not buf.is_floating_point():
            continue
        del module._buffers[name]
        setattr(module, name, torch.nn.Parameter(buf, requires_grad=True))


def _promote_trainable_tree(module: torch.nn.Module) -> torch.nn.Module:
    """Promote trainable buffers to Parameters across the whole module tree.

    Must run after the tree is fully built (post ``__init__`` /
    ``deserialize``): dpmodel deserialize may assign numpy arrays onto nested
    attributes, which ``dpmodel_setattr`` would re-register as buffers.
    """
    for sub in module.modules():
        names = _TRAINABLE_ATTRS.get(type(sub).__name__)
        if names is not None:
            _promote_trainable(sub, names)
    # Freeze every Parameter under a ``trainable=False`` module.  This covers
    # parameters that exist regardless of the promotion table above, e.g. the
    # ``SO2Linear.weight_m`` list, which ``_try_convert_list`` converts to a
    # ParameterList with ``requires_grad=True`` unconditionally.
    for sub in module.modules():
        if getattr(sub, "trainable", True) is False:
            for p in sub.parameters(recurse=True):
                p.requires_grad_(False)
    return module


@BaseDescriptor.register("SeZM")
@BaseDescriptor.register("sezm")
@BaseDescriptor.register("DPA4")
@BaseDescriptor.register("dpa4")
@torch_module
class DescrptDPA4(DescrptDPA4DP):
    _update_sel_cls = UpdateSel

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Persisted graph-routing knob (first-class training configuration):
        # ``disable_graph_lower()`` used to flip only the plain dpmodel bool,
        # which a Trainer checkpoint restart silently reset (the fresh model
        # is rebuilt from config before ``load_state_dict``, and neither the
        # state-dict keys nor ``_extra_state.model_params`` carried the
        # choice) -- on a binding-sel system that switched the training
        # equation and gradients without warning.  A persistent buffer rides
        # every pt_expt state_dict, so save/restart round-trips it.
        torch.nn.Module.register_buffer(
            self,
            "graph_lower_disabled",
            torch.zeros((), dtype=torch.bool, device="cpu"),
        )
        self.use_amp_infer = use_amp_infer()
        _promote_trainable_tree(self)

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptDPA4":
        # deserialize assigns numpy arrays after __init__, which demotes
        # promoted Parameters back to buffers; re-promote at the end.
        obj = super().deserialize(data)
        return _promote_trainable_tree(obj)

    def _in_training_mode(self) -> bool:
        """Torch runtime hook for the training-only random local-Z roll.

        Overrides the dpmodel default (``False``) with the torch module's
        ``training`` flag, restoring pt's ``random_gamma=self.random_gamma
        and self.training`` semantics: train-mode forwards draw a fresh
        gamma per call, eval/export forwards fix gamma (the export path
        calls ``model.eval()`` before tracing).
        """
        return bool(self.training)

    def disable_graph_lower(self) -> None:
        """Persisted variant of the dpmodel escape hatch (see base class).

        The buffer (and the routing bool) are PER-TASK state: multi-task
        ``share_params`` shares network submodules, not this buffer, so
        disabling the graph lower on one task branch does not propagate to
        branches sharing the same descriptor weights -- each branch owns
        its routing decision.
        """
        super().disable_graph_lower()
        self.graph_lower_disabled.fill_(True)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # Back-compat: checkpoints written before the knob was persisted lack
        # the buffer; default to the fresh module's value (graph enabled)
        # instead of failing the strict load.
        key = prefix + "graph_lower_disabled"
        if key not in state_dict:
            state_dict[key] = self.graph_lower_disabled.detach().clone()
        else:
            # Re-sync the dpmodel-side routing bool from the RESTORED value
            # here, at load time, where the incoming tensor is real.  The
            # routing predicate itself must stay a plain python bool:
            # ``uses_graph_lower()`` runs inside traced forwards (the dense
            # adapter gate), and reading the buffer there would emit a
            # data-dependent ``bool(FakeTensor)`` guard that breaks
            # torch.export (GuardOnDataDependentSymNode Eq(u0, 1)).
            self._graph_lower_disabled = bool(state_dict[key])
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.call(*args, **kwargs)

    def _forward_blocks(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        """Run the interaction blocks under the pt_expt AMP policy.

        This is the torch (pt_expt) implementation of the descriptor's
        ``use_amp`` switch, mirroring the reference pt ``_compute_mode_ctx``:
        bfloat16 autocast wraps only the interaction-block region, while the
        geometry, edge cache, radial, env-seed, GIE and output FFN stages stay
        in fp32 (or higher). The dpmodel base stores ``use_amp`` only as a
        config flag and never autocasts (array-API has no autocast), so the
        real automatic mixed precision lives here. ``x`` is the node-feature
        tensor entering the blocks; its device equals the working device, so
        autocast engages when ``self.use_amp`` is set, the inputs live on a
        CUDA device, and either the module is training or eval-time AMP was
        opted in through ``DP_AMP_INFER`` (captured once at construction as
        ``self.use_amp_infer``).
        """
        if (
            self.use_amp
            and x.device.type == "cuda"
            and (self.training or self.use_amp_infer)
        ):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                return super()._forward_blocks(x, *args, **kwargs)
        return super()._forward_blocks(x, *args, **kwargs)

    def share_params(
        self,
        base_class: "DescrptDPA4",
        shared_level: int,
        model_prob: float = 1.0,
        resume: bool = False,
    ) -> None:
        # Multi-task parameter sharing for DPA4 is out of scope for this PR.
        raise NotImplementedError("share_params is not yet implemented for DescrptDPA4")
