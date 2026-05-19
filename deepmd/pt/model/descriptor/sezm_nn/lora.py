# SPDX-License-Identifier: LGPL-3.0-or-later
"""LoRA low-rank fine-tuning support for SeZM.

This module adds two things:

* ``LoRASO3`` and ``LoRASO2`` subclasses that wrap the corresponding base
  equivariant linear operators (``SO3Linear`` / ``SO2Linear``).  Each one
  freezes the pre-trained weights and registers rank-``R`` adapter
  parameters ``A``/``B`` whose shapes share the base's batch layout
  (per-``l`` for SO(3), per-``|m|``-group for SO(2)).  The LoRA delta is
  folded into the *effective* weight before the single large einsum that
  already exists in the base module; forward FLOPs are therefore identical
  to the base, and the overhead comes only from an ``O(R)`` weight-side
  matmul that does not depend on the number of edges or nodes.

* ``apply_lora_to_sezm``, ``merge_lora_into_base`` and a few helpers that
  drive the fine-tune policy (which submodules stay trainable, which ones
  remain frozen) and the merged-checkpoint export used by
  ``Trainer.save_model_merged``.

Naming convention: the LoRA parameter names -- ``A_by_l``, ``B_by_l``,
``A_m0``, ``B_m0``, ``A_m``, ``B_m`` -- intentionally do **not** start with
``adam_`` / ``adamw_`` and do not contain ``bias``.  ``HybridMuon.get_adam_route``
therefore classifies them as ``muon`` and, because the tensors have the
same rank structure as the corresponding base weights, the slice-mode
matrix view gives per-``l`` / per-``|m|``-group Newton-Schulz updates that
match the base training recipe.
"""

from __future__ import (
    annotations,
)

import math
from copy import (
    deepcopy,
)
from typing import (
    Any,
)

import torch
import torch.nn as nn

from .activation import (
    GatedActivation,
)
from .so2 import (
    SO2Linear,
)
from .so3 import (
    SO3Linear,
)

# ---------------------------------------------------------------------------
# LoRA adapter modules
# ---------------------------------------------------------------------------


class LoRASO3(SO3Linear):
    """
    Per-l ELoRA adapter for ``SO3Linear``.

    The pre-trained weight ``self.weight`` (``(lmax+1, C_in, F*C_out)``) is
    frozen.  Two new 3D parameters ``A_by_l`` (``(lmax+1, rank, C_in)``) and
    ``B_by_l`` (``(lmax+1, F*C_out, rank)``) share the same ``lmax+1`` batch
    axis as the base so that ``muon_mode="slice"`` updates every ``l``-block
    independently.  SO(3) equivariance is preserved because the per-``l``
    delta only rotates within each ``l``-block (no cross-``l`` mixing).

    Parameters
    ----------
    base
        Pre-trained ``SO3Linear`` to adapt.  Its weights are copied and then
        frozen.
    rank
        LoRA rank.  Must satisfy ``rank >= 1``.
    alpha
        Scaling numerator; the effective scaling is ``alpha / rank``.
        ``None`` defaults to ``alpha = rank`` (scaling ``1.0``).
    """

    def __init__(
        self,
        base: SO3Linear,
        *,
        rank: int,
        alpha: float | None = None,
    ) -> None:
        if rank < 1:
            raise ValueError(f"LoRASO3 requires rank >= 1, got {rank}")
        # Construct a same-shape SO3Linear, then overwrite its weight with
        # base's state.  ``init_std=0.0`` skips the expensive random init.
        super().__init__(
            lmax=base.lmax,
            in_channels=base.in_channels,
            out_channels=base.out_channels,
            n_focus=base.n_focus,
            dtype=base.dtype,
            mlp_bias=base.mlp_bias,
            trainable=False,
            seed=None,
            init_std=0.0,
        )
        self.load_state_dict(base.state_dict())
        # Defensive: ensure the base weight is frozen even if ``base`` was
        # trainable at serialize time.  ``self.bias`` is intentionally *kept*
        # trainable — ``apply_lora_to_sezm`` re-enables every leaf whose name
        # contains ``"bias"`` so the LoRA-preserved bias can absorb the
        # downstream mean shift alongside the low-rank ``ΔW``.  The assignment
        # here is only a "known starting state" before that policy step runs.
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

        self.rank = int(rank)
        alpha_value = float(alpha) if alpha is not None else float(rank)
        self.scaling = alpha_value / float(rank)
        self.register_buffer(
            "lora_scaling",
            torch.tensor(self.scaling, dtype=self.dtype, device=self.device),
            persistent=True,
        )

        num_l = self.lmax + 1
        self.A_by_l = nn.Parameter(
            torch.empty(
                num_l,
                self.rank,
                self.in_channels,
                dtype=self.dtype,
                device=self.device,
            )
        )
        # B is zero-initialised so that the initial forward is an exact
        # identity to the base module; training backprop updates B first
        # (gradA is zero while B is zero), which is the standard LoRA
        # two-step unlock pattern and is compatible with Newton-Schulz on
        # rectangular matrices.
        self.B_by_l = nn.Parameter(
            torch.zeros(
                num_l,
                self.n_focus * self.out_channels,
                self.rank,
                dtype=self.dtype,
                device=self.device,
            )
        )
        nn.init.normal_(self.A_by_l, mean=0.0, std=1.0 / math.sqrt(self.rank))

    def extra_repr(self) -> str:
        return f"rank={self.rank}, scaling={self.scaling}"

    def _compute_delta_weight(self) -> torch.Tensor:
        """Return ``ΔW`` with shape ``(lmax+1, C_in, F*C_out)``."""
        return torch.einsum("lor,lri->lio", self.B_by_l, self.A_by_l) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Input features with shape ``(N, D, F, C_in)`` where ``D=(lmax+1)^2``.

        Returns
        -------
        torch.Tensor
            Output features with shape ``(N, D, F, C_out)``.
        """
        delta_w = self._compute_delta_weight()
        weight = (self.weight + delta_w).view(
            self.lmax + 1,
            self.in_channels,
            self.n_focus,
            self.out_channels,
        )
        weight_expanded = torch.index_select(weight, dim=0, index=self.expand_index)
        out = torch.einsum("ndfi,difo->ndfo", x, weight_expanded)
        if self.mlp_bias:
            bias = self.bias.view(self.n_focus, self.out_channels)
            out[:, 0, :, :] = out[:, 0, :, :] + bias.unsqueeze(0)
        return out

    def merge_into_base(self) -> SO3Linear:
        """Build a plain ``SO3Linear`` whose weight has absorbed the LoRA delta."""
        base = SO3Linear(
            lmax=self.lmax,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            n_focus=self.n_focus,
            dtype=self.dtype,
            mlp_bias=self.mlp_bias,
            trainable=True,
            seed=None,
            init_std=0.0,
        )
        with torch.no_grad():
            merged = self.weight.detach() + self._compute_delta_weight().detach()
            base.weight.copy_(merged)
            if self.bias is not None:
                assert base.bias is not None
                base.bias.copy_(self.bias.detach())
        return base


class LoRASO2(SO2Linear):
    """
    Per-``|m|``-group LoRA adapter for ``SO2Linear``.

    ``weight_m0`` (``(num_in_m0, F*num_out_m0)``) and each
    ``weight_m[i]`` (``(num_in_m, F*2*num_out_m)``) get an independent 2D
    LoRA pair ``A``/``B``.  SO(2) equivariance is preserved because the
    ``|m|>0`` 2x2 complex block ``[[W_u, -W_v], [W_v, W_u]]`` stays intact
    when ``ΔW_m`` is absorbed into the concatenated ``[W_u | W_v]`` layout
    before ``_build_so2_weight`` splits it (the shared input basis ``A``
    splits naturally into ``ΔW_u = B_u A`` and ``ΔW_v = B_v A``).

    The base ``forward``/``_cached_weight``/``train`` logic is inherited
    unchanged; only ``_build_so2_weight`` is overridden to fold the LoRA
    delta into each base block prior to assembling the block-diagonal
    weight.  The ``ΔW_m`` construction does not depend on the edge count
    ``E``, so the forward FLOPs remain identical to the base.

    Parameters
    ----------
    base
        Pre-trained ``SO2Linear`` to adapt.
    rank
        LoRA rank.
    alpha
        Scaling numerator; scaling is ``alpha / rank``.  ``None`` defaults
        to ``alpha = rank`` (scaling ``1.0``).
    """

    def __init__(
        self,
        base: SO2Linear,
        *,
        rank: int,
        alpha: float | None = None,
    ) -> None:
        if rank < 1:
            raise ValueError(f"LoRASO2 requires rank >= 1, got {rank}")
        super().__init__(
            lmax=base.lmax,
            mmax=base.mmax,
            in_channels=base.in_channels,
            out_channels=base.out_channels,
            n_focus=base.n_focus,
            dtype=base.dtype,
            mlp_bias=base.mlp_bias,
            seed=None,
            trainable=False,
        )
        self.load_state_dict(base.state_dict())
        # Defensive: the base matrices are frozen here, but ``self.bias0`` is
        # intentionally re-enabled later by ``apply_lora_to_sezm`` via the
        # "any leaf containing 'bias' is trainable" rule (``"bias" in "bias0"``
        # is ``True``) so the LoRA-preserved scalar offset can absorb the
        # downstream mean shift alongside the low-rank ``ΔW``.
        self.weight_m0.requires_grad_(False)
        if self.bias0 is not None:
            self.bias0.requires_grad_(False)
        for w in self.weight_m:
            w.requires_grad_(False)
        # Any cached block-diagonal from the base is stale now; force rebuild.
        self._cached_weight = None

        self.rank = int(rank)
        alpha_value = float(alpha) if alpha is not None else float(rank)
        self.scaling = alpha_value / float(rank)
        self.register_buffer(
            "lora_scaling",
            torch.tensor(self.scaling, dtype=self.dtype, device=self.device),
            persistent=True,
        )

        num_in_m0 = (self.lmax + 1) * self.in_channels
        num_out_m0_per_focus = (self.lmax + 1) * self.out_channels
        focus_num_out_m0 = self.n_focus * num_out_m0_per_focus
        self.A_m0 = nn.Parameter(
            torch.empty(
                self.rank,
                num_in_m0,
                dtype=self.dtype,
                device=self.device,
            )
        )
        self.B_m0 = nn.Parameter(
            torch.zeros(
                focus_num_out_m0,
                self.rank,
                dtype=self.dtype,
                device=self.device,
            )
        )
        nn.init.normal_(self.A_m0, mean=0.0, std=1.0 / math.sqrt(self.rank))

        self.A_m = nn.ParameterList()
        self.B_m = nn.ParameterList()
        for w in self.weight_m:
            num_in, focus_two_num_out = w.shape
            a_m = nn.Parameter(
                torch.empty(
                    self.rank,
                    num_in,
                    dtype=self.dtype,
                    device=self.device,
                )
            )
            b_m = nn.Parameter(
                torch.zeros(
                    focus_two_num_out,
                    self.rank,
                    dtype=self.dtype,
                    device=self.device,
                )
            )
            nn.init.normal_(a_m, mean=0.0, std=1.0 / math.sqrt(self.rank))
            self.A_m.append(a_m)
            self.B_m.append(b_m)

    def extra_repr(self) -> str:
        return f"rank={self.rank}, scaling={self.scaling}"

    def _compute_delta_m0(self) -> torch.Tensor:
        """Return ``ΔW_m0`` with shape ``(num_in_m0, F*num_out_m0)``."""
        return torch.einsum("ri,or->io", self.A_m0, self.B_m0) * self.scaling

    def _compute_delta_m(self, m_idx: int) -> torch.Tensor:
        """Return ``ΔW_m[m_idx]`` with the same shape as ``weight_m[m_idx]``."""
        return (
            torch.einsum("ri,or->io", self.A_m[m_idx], self.B_m[m_idx]) * self.scaling
        )

    def _build_so2_weight(self) -> torch.Tensor:
        """Assemble the block-diagonal weight with LoRA delta folded in."""
        in_total = self.reduced_dim * self.in_channels
        out_total = self.reduced_dim * self.out_channels
        weight = self.weight_m0.new_zeros(in_total, self.n_focus, out_total)
        num_in_m0 = (self.lmax + 1) * self.in_channels
        num_out_m0 = (self.lmax + 1) * self.out_channels

        # m=0 block: fold ΔW_m0 into the base weight before the view.
        w_m0_eff = (self.weight_m0 + self._compute_delta_m0()).view(
            num_in_m0, self.n_focus, num_out_m0
        )
        weight[: self._m0_in, :, : self._m0_out] = w_m0_eff

        # |m|>0 blocks: same 2x2 coupling assembly as the base, but with
        # ΔW_m folded into the concatenated [W_u | W_v] layout first.
        for m_idx, w_base in enumerate(self.weight_m):
            ni0, ni1, pi0, pi1, no0, no1, po0, po1 = self._block_slices[m_idx]
            ib = ni1 - ni0
            ob = no1 - no0
            w_eff = (w_base + self._compute_delta_m(m_idx)).view(
                ib, self.n_focus, 2 * ob
            )
            w_u = w_eff[:, :, :ob]
            w_v = w_eff[:, :, ob:]
            weight[ni0:ni1, :, no0:no1] = w_u
            weight[ni0:ni1, :, po0:po1] = w_v
            weight[pi0:pi1, :, no0:no1] = -w_v
            weight[pi0:pi1, :, po0:po1] = w_u
        return weight

    def merge_into_base(self) -> SO2Linear:
        """Build a plain ``SO2Linear`` whose weights have absorbed every LoRA delta."""
        base = SO2Linear(
            lmax=self.lmax,
            mmax=self.mmax,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            n_focus=self.n_focus,
            dtype=self.dtype,
            mlp_bias=self.mlp_bias,
            seed=None,
            trainable=True,
        )
        with torch.no_grad():
            base.weight_m0.copy_(
                self.weight_m0.detach() + self._compute_delta_m0().detach()
            )
            if self.bias0 is not None:
                assert base.bias0 is not None
                base.bias0.copy_(self.bias0.detach())
            for m_idx, w in enumerate(self.weight_m):
                base.weight_m[m_idx].copy_(
                    w.detach() + self._compute_delta_m(m_idx).detach()
                )
        return base


# ---------------------------------------------------------------------------
# Fine-tune policy: freeze / unfreeze rules
# ---------------------------------------------------------------------------

# Leaf parameter names that stay trainable during LoRA fine-tune.  These are small
# scalar / per-l scales / attention gating weights whose full-rank update costs
# are negligible but directly absorb the domain shift of the downstream dataset.
_UNFREEZE_LEAF_NAMES: frozenset[str] = frozenset(
    {
        "adam_scale",
        "adam_so2_layer_scales",
        "adam_ffn_layer_scales",
        "film_scale_strength_log",
        "film_shift_strength_log",
        "adamw_attn_logit_w",
        "adamw_attn_z_bias_raw",
        "adamw_attn_gate_w",
        "adamw_focus_compete_w",
        "adamw_pseudo_query",
        "focus_compete_bias",
    }
)

# Leaf names that stay frozen (override any unfreeze rule above).  The backbone
# pre-training has already converged on these quantities for all-element
# datasets; downstream fine-tuning should keep them fixed.
_OVERRIDE_FREEZE_LEAF_NAMES: frozenset[str] = frozenset(
    {
        "adam_type_embedding",
        "adam_freqs",
    }
)

# Submodule paths (rooted at the SeZMModel) that get fully unfrozen.
_UNFREEZE_SUBMODULE_PATHS: tuple[str, ...] = (
    "atomic_model.fitting_net",
    "atomic_model.dens_fitting_net",
    "atomic_model.descriptor.radial_embedding",
    "atomic_model.descriptor.env_seed_embedding",
    "atomic_model.descriptor.film_scale_norm",
    "atomic_model.descriptor.film_shift_norm",
    "atomic_model.descriptor.final_full_attn_res",
    "atomic_model.descriptor.final_block_attn_res",
)

# Per-interaction-block submodule paths that get fully unfrozen.  The
# descriptor stores the block list at ``atomic_model.descriptor.blocks``.
_UNFREEZE_PER_BLOCK_SUBPATHS: tuple[str, ...] = (
    "full_attn_res_so2",
    "full_attn_res_ffns",
    "block_attn_res_so2",
    "block_attn_res_ffns",
    "so2_conv.attn_q_proj",
    "so2_conv.attn_k_proj",
    "so2_conv.attn_qk_norm",
    "so2_conv.attn_output_gate_norm",
    "so2_conv.focus_compete_norm",
    "so2_conv.radial_hidden_proj",
    "so2_conv.so2_layer_attn_res",
)

_BLOCKS_PATH: str = "atomic_model.descriptor.blocks"


def _leaf_name(param_name: str) -> str:
    """Return the trailing non-numeric segment of a parameter name.

    ``nn.ParameterList`` children show up as ``foo.0``, ``foo.1``, ...;
    ``get_adam_route`` strips those numeric indices before routing, so this
    helper keeps the policy in sync.
    """
    parts = param_name.split(".")
    i = len(parts) - 1
    while i > 0 and parts[i].isdigit():
        i -= 1
    return parts[i]


def _get_submodule_or_none(root: nn.Module, path: str) -> nn.Module | None:
    if not path:
        return root
    try:
        return root.get_submodule(path)
    except AttributeError:
        return None


def _clear_sezm_compile_cache(model: nn.Module) -> None:
    """Invalidate any ``compiled_core_compute_cache`` / ``compiled_dens_compute``.

    LoRA injection or merge replaces submodules, which changes the Python
    object graph that ``torch.compile`` had captured.  Without clearing the
    cache the next forward would reuse the stale compiled callable and
    crash or silently skip LoRA parameters.  Mirrors the pattern used in
    :meth:`SeZMModel.reset_head_for_mode`.
    """
    for m in model.modules():
        core_cache = getattr(m, "compiled_core_compute_cache", None)
        if isinstance(core_cache, dict):
            core_cache.clear()
            if hasattr(m, "_core_compute_pending_compile_t0"):
                m._core_compute_pending_compile_t0 = None
            if hasattr(m, "_core_compute_pending_compile_key"):
                m._core_compute_pending_compile_key = None
        if hasattr(m, "compiled_dens_compute"):
            object.__setattr__(m, "compiled_dens_compute", None)
            if hasattr(m, "_dens_compiled"):
                m._dens_compiled = False
            if hasattr(m, "_dens_pending_compile_t0"):
                m._dens_pending_compile_t0 = None


def _swap_submodule(parent: nn.Module, attr: str, new_module: nn.Module) -> None:
    """Replace ``parent.attr`` with ``new_module``.

    Uses ``parent._modules[attr]`` so that numeric attribute names (for
    ``nn.ModuleList`` / ``nn.ParameterList`` children) work as well.
    """
    parent._modules[attr] = new_module


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def has_lora(module: nn.Module) -> bool:
    """Return ``True`` iff any submodule is a LoRA adapter."""
    return any(isinstance(m, (LoRASO3, LoRASO2)) for m in module.modules())


def apply_lora_to_sezm(
    model: nn.Module,
    *,
    rank: int,
    alpha: float | None = None,
) -> nn.Module:
    """
    Inject LoRA adapters into every ``SO3Linear`` / ``SO2Linear`` of a SeZM
    model and apply the SeZM fine-tune freeze/unfreeze policy in place.

    This function is idempotent-safe: the ``type(mod) is SO3Linear`` (exact
    type) test prevents re-wrapping a LoRASO3 that is already present.

    Parameters
    ----------
    model
        A ``SeZMModel`` instance (or any ``nn.Module`` containing SeZM
        ``SO3Linear`` / ``SO2Linear`` submodules).
    rank
        LoRA rank applied uniformly to every adapter.
    alpha
        LoRA scaling numerator; scaling is ``alpha / rank``.  ``None``
        defaults to ``alpha = rank`` (scaling ``1.0``).

    Returns
    -------
    nn.Module
        The same ``model`` after injection (returned for chaining).
    """
    # === Step 1. Freeze all parameters ===
    for p in model.parameters():
        p.requires_grad_(False)

    # === Step 2. Replace SO3Linear / SO2Linear with LoRA subclasses ===
    # Snapshot named_modules() first so the later in-place replacement does
    # not invalidate the iterator.  ``type(...) is ...`` is deliberate: it
    # matches only the exact base class, skipping any pre-existing LoRA
    # adapter so apply_lora_to_sezm remains idempotent.
    replacements: list[tuple[nn.Module, str, nn.Module]] = []
    for name, mod in list(model.named_modules()):
        if type(mod) is SO3Linear:
            parent_name, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            replacements.append((parent, attr, LoRASO3(mod, rank=rank, alpha=alpha)))
        elif type(mod) is SO2Linear:
            parent_name, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            replacements.append((parent, attr, LoRASO2(mod, rank=rank, alpha=alpha)))
    for parent, attr, new_mod in replacements:
        _swap_submodule(parent, attr, new_mod)

    # === Step 3. Unfreeze whole submodules (descriptor-level and per-block) ===
    for path in _UNFREEZE_SUBMODULE_PATHS:
        sub = _get_submodule_or_none(model, path)
        if sub is None:
            continue
        for p in sub.parameters():
            p.requires_grad_(True)

    blocks = _get_submodule_or_none(model, _BLOCKS_PATH)
    if blocks is not None:
        for block in blocks:
            for subpath in _UNFREEZE_PER_BLOCK_SUBPATHS:
                sub = _get_submodule_or_none(block, subpath)
                if sub is None:
                    continue
                for p in sub.parameters():
                    p.requires_grad_(True)

    # === Step 4. Unfreeze small parameters by leaf name ===
    # Any name ending in a LoRA-listed leaf or containing ``bias`` becomes
    # trainable.  The ``"bias" in leaf`` rule deliberately also re-enables the
    # base biases that ``LoRASO3.__init__`` / ``LoRASO2.__init__`` had frozen
    # (``SO3Linear.bias``, ``SO2Linear.bias0``); keeping those trainable lets
    # the LoRA-preserved offsets absorb the downstream mean shift alongside
    # the low-rank ``ΔW``.  The same rule also unfreezes norm biases
    # (``EquivariantRMSNorm.bias``, ``ReducedEquivariantRMSNorm.bias0``)
    # anywhere in the model -- tiny parameter counts, large domain-shift
    # headroom.  ``adam_scale`` is listed similarly: every RMSNorm scale in
    # the backbone (per-block ``pre/post_so2_norm``, ``pre/post_ffn_norms``,
    # ``so2_inter_norms``, etc.) becomes trainable, again at negligible cost.
    for name, p in model.named_parameters():
        leaf = _leaf_name(name)
        if leaf in _UNFREEZE_LEAF_NAMES or "bias" in leaf:
            p.requires_grad_(True)

    # === Step 5. Override-freeze converged parameters by leaf name ===
    # Must run after steps 3/4 because earlier whole-module unfreezes may
    # have turned them back on (e.g. ``adam_type_embedding`` inside the
    # unfrozen ``env_seed_embedding``).
    for name, p in model.named_parameters():
        leaf = _leaf_name(name)
        if leaf in _OVERRIDE_FREEZE_LEAF_NAMES:
            p.requires_grad_(False)

    # === Step 6. Override-freeze every GatedActivation submodule ===
    # Stable gate patterns; avoids turning on gate_linear.bias via the
    # step-4 "bias" rule.
    for mod in model.modules():
        if isinstance(mod, GatedActivation):
            for p in mod.parameters():
                p.requires_grad_(False)

    return model


def fold_lora_state_dict_keys(state_dict: dict[str, torch.Tensor], prefix: str) -> None:
    """Fold LoRA adapter keys into base weight keys in *state_dict* (in-place).

    Scans for SO3-style ``A_by_l``/``B_by_l`` pairs and SO2-style
    ``A_m0``/``B_m0``/``A_m.*``/``B_m.*`` groups under *prefix*.  For each
    pair whose corresponding base weight key also exists, the delta
    ``einsum(B, A) * scaling`` is added to the weight and the adapter keys
    are popped.  ``lora_scaling`` is read from *state_dict* when present;
    otherwise ``1.0`` is assumed (the default when ``alpha == rank``).

    Called by ``DescrptSeZM._load_from_state_dict`` so that a LoRA-trained
    checkpoint can be loaded into a plain (non-LoRA) descriptor transparently.

    Parameters
    ----------
    state_dict
        Flat state dict to mutate in place.
    prefix
        Key prefix that scopes the scan (e.g. ``"model.Default.atomic_model.descriptor."``).
    """
    # === SO3: fold A_by_l / B_by_l into weight ===
    so3_prefixes = [
        k[: -len("A_by_l")]
        for k in list(state_dict)
        if k.startswith(prefix) and k.endswith(".A_by_l")
    ]
    for sp in so3_prefixes:
        a_key, b_key, w_key = sp + "A_by_l", sp + "B_by_l", sp + "weight"
        if b_key not in state_dict or w_key not in state_dict:
            continue
        a = state_dict.pop(a_key)
        b = state_dict.pop(b_key)
        scaling_tensor = state_dict.pop(sp + "lora_scaling", None)
        scaling = float(scaling_tensor) if scaling_tensor is not None else 1.0
        state_dict[w_key] = (
            state_dict[w_key] + torch.einsum("lor,lri->lio", b, a) * scaling
        )

    # === SO2: fold A_m0 / B_m0 and A_m.* / B_m.* into weight_m0 / weight_m.* ===
    so2_prefixes = [
        k[: -len("A_m0")]
        for k in list(state_dict)
        if k.startswith(prefix) and k.endswith(".A_m0")
    ]
    for sp in so2_prefixes:
        a0_key, b0_key, w0_key = sp + "A_m0", sp + "B_m0", sp + "weight_m0"
        if b0_key not in state_dict or w0_key not in state_dict:
            continue
        scaling_tensor = state_dict.pop(sp + "lora_scaling", None)
        scaling = float(scaling_tensor) if scaling_tensor is not None else 1.0
        a0 = state_dict.pop(a0_key)
        b0 = state_dict.pop(b0_key)
        state_dict[w0_key] = (
            state_dict[w0_key] + torch.einsum("ri,or->io", a0, b0) * scaling
        )
        m_idx = 0
        while True:
            a_key = sp + f"A_m.{m_idx}"
            b_key = sp + f"B_m.{m_idx}"
            w_key = sp + f"weight_m.{m_idx}"
            if a_key not in state_dict:
                break
            a_m = state_dict.pop(a_key)
            b_m = state_dict.pop(b_key)
            state_dict[w_key] = (
                state_dict[w_key] + torch.einsum("ri,or->io", a_m, b_m) * scaling
            )
            m_idx += 1


def build_merged_state_dict(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor] | None = None,
    *,
    prefix: str = "",
) -> dict[str, torch.Tensor]:
    """
    Produce a plain (LoRA-free) state dict from a LoRA-augmented module.

    Walks ``module.named_modules()`` and, for every ``LoRASO3`` /
    ``LoRASO2`` submodule, folds ``ΔW = BA·scaling`` into the base weight
    key and removes the ``A``/``B`` keys.  The returned dict has the same
    key set as a same-topology SeZM that has never been LoRA-wrapped, and
    is suitable for loading into a plain SeZM model with ``strict=True``.

    Non-destructive: when ``state_dict`` is ``None`` a deep copy of
    ``module.state_dict()`` is taken; when the caller provides a
    ``state_dict`` it is assumed to already be a detached copy (e.g. the
    full-gathered state dict from FSDP2) and is *mutated in place* for
    efficiency.

    Parameters
    ----------
    module
        The LoRA-augmented module tree.  Only used for structural
        information (LoRA submodule prefixes, ``scaling``, ``weight_m``
        length); its parameters are not modified.
    state_dict
        Optional pre-collected state dict (e.g. gathered from FSDP2).  If
        ``None``, ``deepcopy(module.state_dict())`` is used.
    prefix
        Prefix to prepend to every LoRA submodule name when looking keys
        up in ``state_dict``.  Use this when the caller has state keyed
        under an outer wrapper (for example ``"model.Default."``).

    Returns
    -------
    dict
        Flat state dict with LoRA adapters folded into base weights.
    """
    state = deepcopy(module.state_dict()) if state_dict is None else state_dict
    for name, mod in module.named_modules():
        key_prefix = prefix + name + "." if name else prefix
        if isinstance(mod, LoRASO3):
            a = state.pop(key_prefix + "A_by_l")
            b = state.pop(key_prefix + "B_by_l")
            state.pop(key_prefix + "lora_scaling", None)
            weight_key = key_prefix + "weight"
            delta = torch.einsum("lor,lri->lio", b, a) * mod.scaling
            state[weight_key] = state[weight_key] + delta
        elif isinstance(mod, LoRASO2):
            a_m0 = state.pop(key_prefix + "A_m0")
            b_m0 = state.pop(key_prefix + "B_m0")
            state.pop(key_prefix + "lora_scaling", None)
            w_m0_key = key_prefix + "weight_m0"
            state[w_m0_key] = (
                state[w_m0_key] + torch.einsum("ri,or->io", a_m0, b_m0) * mod.scaling
            )
            for m_idx in range(len(mod.weight_m)):
                a_i = state.pop(key_prefix + f"A_m.{m_idx}")
                b_i = state.pop(key_prefix + f"B_m.{m_idx}")
                w_i_key = key_prefix + f"weight_m.{m_idx}"
                state[w_i_key] = (
                    state[w_i_key] + torch.einsum("ri,or->io", a_i, b_i) * mod.scaling
                )
    return state


def strip_lora_from_extra_state(extra_state: dict[str, Any]) -> dict[str, Any]:
    """
    Drop any ``lora`` entry from ``_extra_state["model_params"]``.

    Handles both single-task (``model_params`` is the model config) and
    multi-task (``model_params["model_dict"][<branch>]`` is each branch's
    config).  Returns a deep-copied dict; the input is not mutated.
    """
    out = deepcopy(extra_state)
    model_params = out.get("model_params")
    if not isinstance(model_params, dict):
        return out
    model_params.pop("lora", None)
    model_dict = model_params.get("model_dict")
    if isinstance(model_dict, dict):
        for branch_cfg in model_dict.values():
            if isinstance(branch_cfg, dict):
                branch_cfg.pop("lora", None)
    return out


def merge_lora_into_base(model: nn.Module) -> nn.Module:
    """
    Destructively replace every ``LoRASO3`` / ``LoRASO2`` with its merged
    plain base module.

    After this call the model no longer contains LoRA submodules: the
    optimizer, EMA state, and any compiled callables that reference the old
    submodules become invalid.  Prefer :func:`build_merged_state_dict` for
    non-destructive checkpoint export during or after training; this function
    is primarily useful in tests and offline scripts.
    """
    replacements: list[tuple[nn.Module, str, nn.Module]] = []
    for name, mod in list(model.named_modules()):
        if isinstance(mod, (LoRASO3, LoRASO2)):
            parent_name, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            replacements.append((parent, attr, mod.merge_into_base()))
    for parent, attr, new_mod in replacements:
        _swap_submodule(parent, attr, new_mod)
    _clear_sezm_compile_cache(model)
    return model
