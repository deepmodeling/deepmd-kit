# SPDX-License-Identifier: LGPL-3.0-or-later
"""LoRA low-rank fine-tuning support for DPA4/SeZM.

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

This module is the dpmodel (array-API) port of
``deepmd.pt.model.descriptor.sezm_nn.lora``.
"""

from __future__ import (
    annotations,
)

import math
from copy import (
    deepcopy,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
    NativeOP,
)
from deepmd.dpmodel.array_api import (
    xp_asarray_nodetach,
)
from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .activation import (
    GatedActivation,
)
from .so2 import (
    SO2Linear,
)
from .so3 import (
    SO3Linear,
)

if TYPE_CHECKING:
    from collections.abc import (
        Iterator,
    )

    from deepmd.dpmodel.array_api import (
        Array,
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
    lmax, in_channels, out_channels, n_focus, precision, mlp_bias, trainable, seed
        Forwarded to ``SO3Linear`` to build the frozen base weight.
    lora_rank
        LoRA rank.  Must satisfy ``lora_rank >= 1``.
    lora_alpha
        Scaling numerator; the effective scaling is ``lora_alpha / lora_rank``.
        ``None`` defaults to ``lora_alpha = lora_rank`` (scaling ``1.0``).
    """

    def __init__(
        self,
        *,
        lmax: int,
        in_channels: int,
        out_channels: int,
        n_focus: int = 1,
        precision: str = DEFAULT_PRECISION,
        mlp_bias: bool = False,
        trainable: bool = False,
        seed: int | list[int] | None = None,
        lora_rank: int,
        lora_alpha: float | None = None,
    ) -> None:
        if lora_rank < 1:
            raise ValueError(f"LoRASO3 requires rank >= 1, got {lora_rank}")
        # Build a same-shape SO3Linear base; the pre-trained weight is restored
        # by ``deserialize`` afterwards.
        super().__init__(
            lmax=lmax,
            in_channels=in_channels,
            out_channels=out_channels,
            n_focus=n_focus,
            precision=precision,
            mlp_bias=mlp_bias,
            trainable=False,
            seed=seed,
        )
        self.trainable = bool(trainable)
        prec = PRECISION_DICT[self.precision.lower()]

        self.lora_rank = int(lora_rank)
        alpha_value = float(lora_alpha) if lora_alpha is not None else float(lora_rank)
        self.lora_alpha = alpha_value
        self.scaling = alpha_value / float(lora_rank)
        self.lora_scaling = np.array(self.scaling, dtype=prec)

        num_l = self.lmax + 1
        rng = np.random.default_rng(seed)
        self.A_by_l = rng.normal(
            0.0,
            1.0 / math.sqrt(self.lora_rank),
            size=(num_l, self.lora_rank, self.in_channels),
        ).astype(prec)
        # B is zero-initialised so that the initial forward is an exact
        # identity to the base module; training backprop updates B first
        # (gradA is zero while B is zero), which is the standard LoRA
        # two-step unlock pattern and is compatible with Newton-Schulz on
        # rectangular matrices.
        self.B_by_l = np.zeros(
            (num_l, self.n_focus * self.out_channels, self.lora_rank), dtype=prec
        )

    def _compute_delta_weight(self, xp: Any, device: Any) -> Array:
        """Return ``ΔW`` with shape ``(lmax+1, C_in, F*C_out)``."""
        B_by_l = xp_asarray_nodetach(xp, self.B_by_l[...], device=device)
        A_by_l = xp_asarray_nodetach(xp, self.A_by_l[...], device=device)
        # einsum "lor,lri->lio" as a per-l batched matmul (B @ A) then transpose:
        # (L, F*Cout, R) @ (L, R, Cin) -> (L, F*Cout, Cin) -> (L, Cin, F*Cout)
        return xp.permute_dims(xp.matmul(B_by_l, A_by_l), (0, 2, 1)) * self.scaling

    def call(self, x: Array) -> Array:
        """
        Parameters
        ----------
        x
            Input features with shape ``(N, D, F, C_in)`` where ``D=(lmax+1)^2``.

        Returns
        -------
        Array
            Output features with shape ``(N, D, F, C_out)``.
        """
        xp = array_api_compat.array_namespace(x)
        device = array_api_compat.device(x)
        delta_w = self._compute_delta_weight(xp, device)
        weight = xp.reshape(
            xp_asarray_nodetach(xp, self.weight[...], device=device) + delta_w,
            (self.lmax + 1, self.in_channels, self.n_focus, self.out_channels),
        )
        expand_index = xp_asarray_nodetach(xp, self.expand_index, device=device)
        weight_expanded = xp.take(weight, expand_index, axis=0)
        # einsum "ndfi,difo->ndfo" as a broadcast batched matmul:
        # (N, D, F, 1, Cin) @ (1, D, F, Cin, Cout) -> (N, D, F, 1, Cout)
        weight_expanded = xp.permute_dims(weight_expanded, (0, 2, 1, 3))
        out = xp.matmul(x[:, :, :, None, :], weight_expanded[None, ...])[..., 0, :]
        if self.mlp_bias:
            bias = xp.reshape(
                xp_asarray_nodetach(xp, self.bias[...], device=device),
                (self.n_focus, self.out_channels),
            )
            out = xp.concat(
                [out[:, :1, :, :] + bias[None, None, ...], out[:, 1:, :, :]], axis=1
            )
        return out

    def merge_into_base(self) -> SO3Linear:
        """Build a plain ``SO3Linear`` whose weight has absorbed the LoRA delta."""
        base = SO3Linear(
            lmax=self.lmax,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            n_focus=self.n_focus,
            precision=self.precision,
            mlp_bias=self.mlp_bias,
            trainable=True,
            seed=None,
            init_std=0.0,
        )
        xp = array_api_compat.array_namespace(self.B_by_l)
        device = array_api_compat.device(self.B_by_l)
        base.weight = to_numpy_array(
            self.weight + self._compute_delta_weight(xp, device)
        )
        if self.bias is not None:
            base.bias = to_numpy_array(self.bias)
        return base

    def serialize(self) -> dict[str, Any]:
        """Serialize the LoRASO3 to a dict."""
        data = super().serialize()
        data["@class"] = "LoRASO3"
        data["config"]["lora_rank"] = self.lora_rank
        data["config"]["lora_alpha"] = self.lora_alpha
        data["@variables"]["A_by_l"] = to_numpy_array(self.A_by_l)
        data["@variables"]["B_by_l"] = to_numpy_array(self.B_by_l)
        return data

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> LoRASO3:
        """Deserialize a LoRASO3 from a dict."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "LoRASO3":
            raise ValueError(f"Invalid class for LoRASO3: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(**config)
        prec = PRECISION_DICT[obj.precision.lower()]
        obj.expand_index = np.asarray(variables["expand_index"], dtype=np.int64)
        obj.weight = np.asarray(variables["weight"], dtype=prec)
        if obj.mlp_bias:
            obj.bias = np.asarray(variables["bias"], dtype=prec)
        obj.A_by_l = np.asarray(variables["A_by_l"], dtype=prec)
        obj.B_by_l = np.asarray(variables["B_by_l"], dtype=prec)
        return obj


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

    The base ``call`` logic is inherited unchanged; only ``_build_so2_weight``
    is overridden to fold the LoRA delta into each base block prior to
    assembling the block-diagonal weight.  The ``ΔW_m`` construction does not
    depend on the edge count ``E``, so the forward FLOPs remain identical to
    the base.

    Parameters
    ----------
    lmax, mmax, in_channels, out_channels, n_focus, precision, mlp_bias, trainable, seed
        Forwarded to ``SO2Linear`` to build the frozen base weights.
    lora_rank
        LoRA rank.
    lora_alpha
        Scaling numerator; scaling is ``lora_alpha / lora_rank``.  ``None``
        defaults to ``lora_alpha = lora_rank`` (scaling ``1.0``).
    """

    def __init__(
        self,
        *,
        lmax: int,
        mmax: int | None = None,
        in_channels: int,
        out_channels: int,
        n_focus: int = 1,
        precision: str = DEFAULT_PRECISION,
        mlp_bias: bool = False,
        seed: int | list[int] | None = None,
        trainable: bool = False,
        lora_rank: int,
        lora_alpha: float | None = None,
    ) -> None:
        if lora_rank < 1:
            raise ValueError(f"LoRASO2 requires rank >= 1, got {lora_rank}")
        super().__init__(
            lmax=lmax,
            mmax=mmax,
            in_channels=in_channels,
            out_channels=out_channels,
            n_focus=n_focus,
            precision=precision,
            mlp_bias=mlp_bias,
            seed=seed,
            trainable=False,
        )
        self.trainable = bool(trainable)
        prec = PRECISION_DICT[self.precision.lower()]

        self.lora_rank = int(lora_rank)
        alpha_value = float(lora_alpha) if lora_alpha is not None else float(lora_rank)
        self.lora_alpha = alpha_value
        self.scaling = alpha_value / float(lora_rank)
        self.lora_scaling = np.array(self.scaling, dtype=prec)

        rng = np.random.default_rng(seed)
        num_in_m0 = (self.lmax + 1) * self.in_channels
        num_out_m0_per_focus = (self.lmax + 1) * self.out_channels
        focus_num_out_m0 = self.n_focus * num_out_m0_per_focus
        self.A_m0 = rng.normal(
            0.0, 1.0 / math.sqrt(self.lora_rank), size=(self.lora_rank, num_in_m0)
        ).astype(prec)
        self.B_m0 = np.zeros((focus_num_out_m0, self.lora_rank), dtype=prec)

        self.A_m: list[np.ndarray] = []
        self.B_m: list[np.ndarray] = []
        for w in self.weight_m:
            num_in, focus_two_num_out = w.shape
            a_m = rng.normal(
                0.0, 1.0 / math.sqrt(self.lora_rank), size=(self.lora_rank, num_in)
            ).astype(prec)
            b_m = np.zeros((focus_two_num_out, self.lora_rank), dtype=prec)
            self.A_m.append(a_m)
            self.B_m.append(b_m)

    def _compute_delta_m0(self, xp: Any, device: Any) -> Array:
        """Return ``ΔW_m0`` with shape ``(num_in_m0, F*num_out_m0)``."""
        A_m0 = xp_asarray_nodetach(xp, self.A_m0[...], device=device)
        B_m0 = xp_asarray_nodetach(xp, self.B_m0[...], device=device)
        # einsum "ri,or->io" as a matmul (B @ A) then transpose:
        # (F*num_out_m0, R) @ (R, num_in_m0) -> (F*num_out_m0, num_in_m0)
        #   -> (num_in_m0, F*num_out_m0)
        return xp.permute_dims(xp.matmul(B_m0, A_m0), (1, 0)) * self.scaling

    def _compute_delta_m(self, m_idx: int, xp: Any, device: Any) -> Array:
        """Return ``ΔW_m[m_idx]`` with the same shape as ``weight_m[m_idx]``."""
        A_m = xp_asarray_nodetach(xp, self.A_m[m_idx][...], device=device)
        B_m = xp_asarray_nodetach(xp, self.B_m[m_idx][...], device=device)
        return xp.permute_dims(xp.matmul(B_m, A_m), (1, 0)) * self.scaling

    def _build_so2_weight(self, xp: Any, device: Any) -> Array:
        """Assemble the block-diagonal weight with LoRA delta folded in."""
        out_total = self.reduced_dim * self.out_channels
        num_in_m0 = (self.lmax + 1) * self.in_channels
        num_out_m0 = (self.lmax + 1) * self.out_channels

        # m=0 block: fold ΔW_m0 into the base weight before the view.
        weight_m0 = xp.reshape(
            xp_asarray_nodetach(xp, self.weight_m0[...], device=device)
            + self._compute_delta_m0(xp, device),
            (num_in_m0, self.n_focus, num_out_m0),
        )
        row_blocks = [
            xp.concat(
                [
                    weight_m0,
                    xp.zeros(
                        (self._m0_in, self.n_focus, out_total - self._m0_out),
                        dtype=weight_m0.dtype,
                        device=device,
                    ),
                ],
                axis=2,
            )
        ]

        # |m|>0 blocks: same 2x2 coupling assembly as the base, but with
        # ΔW_m folded into the concatenated [W_u | W_v] layout first.
        for m_idx, w in enumerate(self.weight_m):
            ni0, ni1, pi0, pi1, no0, no1, po0, po1 = self._block_slices[m_idx]
            ib = ni1 - ni0
            ob = no1 - no0
            w = xp.reshape(
                xp_asarray_nodetach(xp, w[...], device=device)
                + self._compute_delta_m(m_idx, xp, device),
                (ib, self.n_focus, 2 * ob),
            )
            w_u = w[:, :, :ob]
            w_v = w[:, :, ob:]
            left_pad = xp.zeros((ib, self.n_focus, no0), dtype=w.dtype, device=device)
            right_pad = xp.zeros(
                (ib, self.n_focus, out_total - po1), dtype=w.dtype, device=device
            )
            neg_row = xp.concat([left_pad, w_u, w_v, right_pad], axis=2)
            pos_row = xp.concat([left_pad, -w_v, w_u, right_pad], axis=2)
            row_blocks.append(neg_row)
            row_blocks.append(pos_row)
        return xp.concat(row_blocks, axis=0)

    def merge_into_base(self) -> SO2Linear:
        """Build a plain ``SO2Linear`` whose weights have absorbed every LoRA delta."""
        base = SO2Linear(
            lmax=self.lmax,
            mmax=self.mmax,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            n_focus=self.n_focus,
            precision=self.precision,
            mlp_bias=self.mlp_bias,
            seed=None,
            trainable=True,
        )
        xp = array_api_compat.array_namespace(self.weight_m0)
        device = array_api_compat.device(self.weight_m0)
        base.weight_m0 = to_numpy_array(
            self.weight_m0 + self._compute_delta_m0(xp, device)
        )
        if self.bias0 is not None:
            base.bias0 = to_numpy_array(self.bias0)
        for m_idx, w in enumerate(self.weight_m):
            base.weight_m[m_idx] = to_numpy_array(
                w + self._compute_delta_m(m_idx, xp, device)
            )
        return base

    def serialize(self) -> dict[str, Any]:
        data = super().serialize()
        data["@class"] = "LoRASO2"
        data["config"]["lora_rank"] = self.lora_rank
        data["config"]["lora_alpha"] = self.lora_alpha
        variables = data["@variables"]
        variables["A_m0"] = to_numpy_array(self.A_m0)
        variables["B_m0"] = to_numpy_array(self.B_m0)
        for i, (a, b) in enumerate(zip(self.A_m, self.B_m, strict=True)):
            variables[f"A_m.{i}"] = to_numpy_array(a)
            variables[f"B_m.{i}"] = to_numpy_array(b)
        return data

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> LoRASO2:
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "LoRASO2":
            raise ValueError(f"Invalid class for LoRASO2: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(**config)
        prec = PRECISION_DICT[obj.precision.lower()]
        obj.m0_idx = np.asarray(variables["m0_idx"], dtype=np.int64)
        obj.pos_indices = np.asarray(variables["pos_indices"], dtype=np.int64)
        obj.neg_indices = np.asarray(variables["neg_indices"], dtype=np.int64)
        obj.weight_m0 = np.asarray(variables["weight_m0"], dtype=prec)
        if obj.mlp_bias:
            obj.bias0 = np.asarray(variables["bias0"], dtype=prec)
        obj.weight_m = [
            np.asarray(variables[f"weight_m.{i}"], dtype=prec)
            for i in range(len(obj.weight_m))
        ]
        obj.A_m0 = np.asarray(variables["A_m0"], dtype=prec)
        obj.B_m0 = np.asarray(variables["B_m0"], dtype=prec)
        obj.A_m = [
            np.asarray(variables[f"A_m.{i}"], dtype=prec) for i in range(len(obj.A_m))
        ]
        obj.B_m = [
            np.asarray(variables[f"B_m.{i}"], dtype=prec) for i in range(len(obj.B_m))
        ]
        return obj


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
        # LoRA adapter deltas: pt makes them trainable ``nn.Parameter`` at
        # construction, but the dpmodel tracks trainability per module, so the
        # owning ``LoRASO3``/``LoRASO2`` must be marked trainable here for its
        # low-rank delta to receive gradients (the frozen base is restored by
        # the backend, e.g. pt_expt ``_LORA_FROZEN_BASE``).
        "A_by_l",
        "B_by_l",
        "A_m0",
        "B_m0",
        "A_m",
        "B_m",
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


# ---------------------------------------------------------------------------
# NativeOP tree traversal
# ---------------------------------------------------------------------------
# Children of a ``NativeOP`` are stored as plain attributes and as
# ``list``/``tuple`` of ``NativeOP`` (the equivalent of ``nn.ModuleList``).
# The helpers below walk that object graph to enumerate modules, parameters
# and a flat weight dictionary.


def _iter_named_modules(
    root: NativeOP, prefix: str = "", memo: set[int] | None = None
) -> Iterator[tuple[str, NativeOP]]:
    """Yield ``(dotted_name, module)`` for *root* and every nested ``NativeOP``.

    ``root`` is yielded first under *prefix*, then the walk descends into every
    attribute value that is a ``NativeOP`` and into every ``NativeOP`` element
    of a ``list``/``tuple``, building dotted paths (``attr`` and ``attr.{i}``).
    A shared-module memo de-duplicates repeated references.
    """
    if memo is None:
        memo = set()
    if id(root) in memo:
        return
    memo.add(id(root))
    yield prefix, root
    for attr, value in vars(root).items():
        if isinstance(value, NativeOP):
            child = f"{prefix}.{attr}" if prefix else attr
            yield from _iter_named_modules(value, child, memo)
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                if isinstance(item, NativeOP):
                    child = f"{prefix}.{attr}.{i}" if prefix else f"{attr}.{i}"
                    yield from _iter_named_modules(item, child, memo)


def _iter_named_parameters(
    root: NativeOP,
) -> Iterator[tuple[str, NativeOP, np.ndarray]]:
    """Yield ``(dotted_name, owner, array)`` for every numpy-array parameter.

    A dpmodel "parameter" is a ``numpy`` array stored as a module attribute (or
    a ``numpy`` element of a ``list``/``tuple`` attribute, the equivalent of an
    ``nn.ParameterList``).  ``owner`` is the module holding the array; because
    the dpmodel tracks trainability per module (``module.trainable``) rather
    than per tensor, callers toggle ``owner.trainable`` where the PyTorch code
    toggles ``param.requires_grad``.
    """
    for mod_name, mod in _iter_named_modules(root):
        base = mod_name + "." if mod_name else ""
        for attr, value in vars(mod).items():
            if isinstance(value, np.ndarray):
                yield base + attr, mod, value
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, np.ndarray):
                        yield f"{base}{attr}.{i}", mod, item


def _module_state_dict(root: NativeOP) -> dict[str, np.ndarray]:
    """Flat dotted ``{name: array}`` dict over the whole module tree."""
    return {name: value for name, _owner, value in _iter_named_parameters(root)}


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


def _get_submodule_or_none(root: NativeOP, path: str) -> Any:
    if not path:
        return root
    obj: Any = root
    for part in path.split("."):
        if part.isdigit() and isinstance(obj, (list, tuple)):
            index = int(part)
            obj = obj[index] if index < len(obj) else None
        else:
            obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _clear_sezm_compile_cache(model: NativeOP) -> None:
    """No-op retained for parity with the PyTorch backend.

    In PyTorch, LoRA injection or merge replaces submodules and therefore
    invalidates any ``torch.compile`` / inductor callable captured on the
    module graph, which must be cleared before the next forward.  The dpmodel
    (array-API) backend compiles nothing, so there is no cache to clear and
    this function intentionally does nothing.
    """
    return


def _swap_submodule(parent: Any, attr: str, new_module: NativeOP) -> None:
    """Replace ``parent.attr`` with ``new_module``.

    Numeric attribute names address ``list``/``tuple`` children (the dpmodel
    analogue of ``nn.ModuleList`` / ``nn.ParameterList`` elements) and are
    assigned by index; every other name is a plain attribute assignment.
    """
    if attr.isdigit() and isinstance(parent, (list, tuple)):
        parent[int(attr)] = new_module
    else:
        setattr(parent, attr, new_module)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def has_lora(module: NativeOP) -> bool:
    """Return ``True`` iff any submodule is a LoRA adapter."""
    return any(
        isinstance(m, (LoRASO3, LoRASO2)) for _name, m in _iter_named_modules(module)
    )


def apply_lora_to_sezm(
    model: NativeOP,
    *,
    rank: int,
    alpha: float | None = None,
) -> NativeOP:
    """
    Inject LoRA adapters into every ``SO3Linear`` / ``SO2Linear`` of a SeZM
    model and apply the SeZM fine-tune freeze/unfreeze policy in place.

    This function is idempotent-safe: the ``type(mod) is SO3Linear`` (exact
    type) test prevents re-wrapping a LoRASO3 that is already present.

    Parameters
    ----------
    model
        A ``SeZMModel`` instance (or any ``NativeOP`` containing SeZM
        ``SO3Linear`` / ``SO2Linear`` submodules).
    rank
        LoRA rank applied uniformly to every adapter.
    alpha
        LoRA scaling numerator; scaling is ``alpha / rank``.  ``None``
        defaults to ``alpha = rank`` (scaling ``1.0``).

    Returns
    -------
    NativeOP
        The same ``model`` after injection (returned for chaining).
    """
    # === Step 1. Freeze all parameters ===
    for _name, mod in _iter_named_modules(model):
        mod.trainable = False

    # === Step 2. Replace SO3Linear / SO2Linear with LoRA subclasses ===
    # Snapshot named_modules() first so the later in-place replacement does
    # not invalidate the iterator.  ``type(...) is ...`` is deliberate: it
    # matches only the exact base class, skipping any pre-existing LoRA
    # adapter so apply_lora_to_sezm remains idempotent.
    replacements: list[tuple[Any, str, NativeOP]] = []
    for name, mod in list(_iter_named_modules(model)):
        if type(mod) is SO3Linear:
            parent_name, _, attr = name.rpartition(".")
            parent = (
                _get_submodule_or_none(model, parent_name) if parent_name else model
            )
            new_mod = LoRASO3(
                **mod.serialize()["config"], lora_rank=rank, lora_alpha=alpha
            )
            new_mod.weight = mod.weight
            new_mod.bias = mod.bias
            replacements.append((parent, attr, new_mod))
        elif type(mod) is SO2Linear:
            parent_name, _, attr = name.rpartition(".")
            parent = (
                _get_submodule_or_none(model, parent_name) if parent_name else model
            )
            new_mod = LoRASO2(
                **mod.serialize()["config"], lora_rank=rank, lora_alpha=alpha
            )
            new_mod.weight_m0 = mod.weight_m0
            new_mod.bias0 = mod.bias0
            new_mod.weight_m = list(mod.weight_m)
            replacements.append((parent, attr, new_mod))
    for parent, attr, new_mod in replacements:
        _swap_submodule(parent, attr, new_mod)

    # === Step 3. Unfreeze whole submodules (descriptor-level and per-block) ===
    for path in _UNFREEZE_SUBMODULE_PATHS:
        sub = _get_submodule_or_none(model, path)
        if sub is None:
            continue
        for _name, mod in _iter_named_modules(sub):
            mod.trainable = True

    blocks = _get_submodule_or_none(model, _BLOCKS_PATH)
    if blocks is not None:
        for block in blocks:
            for subpath in _UNFREEZE_PER_BLOCK_SUBPATHS:
                sub = _get_submodule_or_none(block, subpath)
                if sub is None:
                    continue
                for _name, mod in _iter_named_modules(sub):
                    mod.trainable = True

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
    for name, owner, _value in _iter_named_parameters(model):
        leaf = _leaf_name(name)
        if leaf in _UNFREEZE_LEAF_NAMES or "bias" in leaf:
            owner.trainable = True

    # === Step 5. Override-freeze converged parameters by leaf name ===
    # Must run after steps 3/4 because earlier whole-module unfreezes may
    # have turned them back on (e.g. ``adam_type_embedding`` inside the
    # unfrozen ``env_seed_embedding``).
    for name, owner, _value in _iter_named_parameters(model):
        leaf = _leaf_name(name)
        if leaf in _OVERRIDE_FREEZE_LEAF_NAMES:
            owner.trainable = False

    # === Step 6. Override-freeze every GatedActivation submodule ===
    # Stable gate patterns; avoids turning on gate_linear.bias via the
    # step-4 "bias" rule.
    for _name, mod in _iter_named_modules(model):
        if isinstance(mod, GatedActivation):
            for _sub_name, sub_mod in _iter_named_modules(mod):
                sub_mod.trainable = False

    return model


def fold_lora_state_dict_keys(state_dict: dict[str, np.ndarray], prefix: str) -> None:
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
            state_dict[w_key] + np.transpose(np.matmul(b, a), (0, 2, 1)) * scaling
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
            state_dict[w0_key] + np.transpose(np.matmul(b0, a0), (1, 0)) * scaling
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
                state_dict[w_key] + np.transpose(np.matmul(b_m, a_m), (1, 0)) * scaling
            )
            m_idx += 1


def build_merged_state_dict(
    module: NativeOP,
    state_dict: dict[str, np.ndarray] | None = None,
    *,
    prefix: str = "",
) -> dict[str, np.ndarray]:
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
    state = deepcopy(_module_state_dict(module)) if state_dict is None else state_dict
    for name, mod in _iter_named_modules(module):
        key_prefix = prefix + name + "." if name else prefix
        if isinstance(mod, LoRASO3):
            a = state.pop(key_prefix + "A_by_l")
            b = state.pop(key_prefix + "B_by_l")
            state.pop(key_prefix + "lora_scaling", None)
            weight_key = key_prefix + "weight"
            delta = np.transpose(np.matmul(b, a), (0, 2, 1)) * mod.scaling
            state[weight_key] = state[weight_key] + delta
        elif isinstance(mod, LoRASO2):
            a_m0 = state.pop(key_prefix + "A_m0")
            b_m0 = state.pop(key_prefix + "B_m0")
            state.pop(key_prefix + "lora_scaling", None)
            w_m0_key = key_prefix + "weight_m0"
            state[w_m0_key] = (
                state[w_m0_key]
                + np.transpose(np.matmul(b_m0, a_m0), (1, 0)) * mod.scaling
            )
            for m_idx in range(len(mod.weight_m)):
                a_i = state.pop(key_prefix + f"A_m.{m_idx}")
                b_i = state.pop(key_prefix + f"B_m.{m_idx}")
                w_i_key = key_prefix + f"weight_m.{m_idx}"
                state[w_i_key] = (
                    state[w_i_key]
                    + np.transpose(np.matmul(b_i, a_i), (1, 0)) * mod.scaling
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


def merge_lora_into_base(model: NativeOP) -> NativeOP:
    """
    Destructively replace every ``LoRASO3`` / ``LoRASO2`` with its merged
    plain base module.

    After this call the model no longer contains LoRA submodules: the
    optimizer, EMA state, and any compiled callables that reference the old
    submodules become invalid.  Prefer :func:`build_merged_state_dict` for
    non-destructive checkpoint export during or after training; this function
    is primarily useful in tests and offline scripts.
    """
    replacements: list[tuple[Any, str, NativeOP]] = []
    for name, mod in list(_iter_named_modules(model)):
        if isinstance(mod, (LoRASO3, LoRASO2)):
            parent_name, _, attr = name.rpartition(".")
            parent = (
                _get_submodule_or_none(model, parent_name) if parent_name else model
            )
            replacements.append((parent, attr, mod.merge_into_base()))
    for parent, attr, new_mod in replacements:
        _swap_submodule(parent, attr, new_mod)
    _clear_sezm_compile_cache(model)
    return model
