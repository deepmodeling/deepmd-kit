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
from deepmd.dpmodel.descriptor.dpa4_nn.wignerd import (
    WignerDCalculator as WignerDCalculatorDP,
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
class WignerDCalculator(WignerDCalculatorDP):
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.call(*args, **kwargs)


# WignerDCalculator.deserialize raises NotImplementedError by design (its
# tables are derived constants); rebuild from the stored constructor args.
register_dpmodel_mapping(
    WignerDCalculatorDP,
    lambda v: WignerDCalculator(v.lmax, eps=v.eps, precision=v.precision),
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
        _promote_trainable_tree(self)

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptDPA4":
        # deserialize assigns numpy arrays after __init__, which demotes
        # promoted Parameters back to buffers; re-promote at the end.
        obj = super().deserialize(data)
        return _promote_trainable_tree(obj)

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
        autocast engages only when ``self.use_amp`` is set, the module is in
        training mode, and the inputs live on a CUDA device.
        """
        if self.use_amp and self.training and x.device.type == "cuda":
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
