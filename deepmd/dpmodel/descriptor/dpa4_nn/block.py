# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Interaction blocks for DPA4/SeZM.

This module is the dpmodel port of ``deepmd.pt.model.descriptor.sezm_nn.block``.
It defines the SeZM interaction block that combines SO(2) message passing and
equivariant feed-forward subblocks with residual shortcuts.

Branches guarded with ``NotImplementedError`` at this level (flags consumed by
block.py itself, all unused by the core DPA4 config):

- ``full_attn_res != "none"`` / ``block_attn_res != "none"`` — the pt block
  builds ``DepthAttnRes`` aggregators and switches the forward implementation
  (pt block.py:514/541); only the baseline residual-shortcut path
  (pt block.py:756) is ported.
- ``layer_scale=True`` — the pt block builds per-channel
  ``adam_ffn_layer_scales`` on the FFN residual branches (pt block.py:500)
  in addition to the SO(2)-internal scales; not ported.

Flags merely forwarded to sub-components keep their guards there (delegated,
not duplicated here): ``so2_attn_res``, ``so2_s2_activation``,
``node_wise_s2/so3``, ``message_node_s2/so3``, ``atten_f_mix``,
``atten_v_proj``, ``atten_o_proj`` (raised by ``SO2Convolution``) and
``ffn_so3_grid`` with the grid path active (raised by ``EquivariantFFN``).

The pt eval-time activation-checkpoint / nvtx instrumentation
(``DP_ACT_INFER``, ``DP_COMPILE_INFER``, ``nvtx_range``) is pt-runtime-only
and intentionally not ported.
"""

from __future__ import (
    annotations,
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
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .ffn import (
    EquivariantFFN,
)
from .norm import (
    EquivariantRMSNorm,
)
from .so2 import (
    SO2Convolution,
    _compute_precision,
)
from .utils import (
    ATTN_RES_MODES,
)

if TYPE_CHECKING:
    from .edge_cache import (
        EdgeCache,
    )


class SeZMInteractionBlock(NativeOP):
    """
    SeZM interaction block with SO(2) message passing and equivariant FFN stack.

    Branch order:
    1. SO(2) branch: optional pre-norm -> `SO2Convolution` -> optional post-norm.
    2. FFN branch: repeated subblocks of
       optional pre-norm -> `EquivariantFFN` -> optional post-norm.

    Outer residual shortcuts are applied around the SO(2) unit and each FFN
    subblock (the pt AttnRes paths are not ported; see the module docstring).

    `SO2Convolution` internally handles the real multi-focus expansion, so this
    block keeps a singleton-focus backbone layout `(N, D, 1, C)` at boundaries.

    Parameters mirror the pt ``SeZMInteractionBlock`` (pt block.py:227) with
    ``precision`` replacing ``dtype``; see the pt docstring for the full
    per-parameter description.
    """

    def __init__(
        self,
        *,
        lmax: int,
        node_lmax: int | None = None,
        mmax: int | None = None,
        kmax: int = 1,
        channels: int,
        n_focus: int = 1,
        focus_dim: int = 0,
        focus_compete: bool = True,
        so2_norm: bool = False,
        so2_layers: int = 4,
        so2_attn_res: str = "none",
        radial_so2_mode: str = "none",
        radial_so2_rank: int = 0,
        n_atten_head: int = 1,
        atten_f_mix: bool = False,
        atten_v_proj: bool = False,
        atten_o_proj: bool = False,
        so2_pre_norm: bool = True,
        so2_post_norm: bool = False,
        ffn_pre_norm: bool = True,
        ffn_post_norm: bool = False,
        ffn_neurons: int = 96,
        node_wise_grid_mlp: bool = False,
        node_wise_grid_branch: int = 0,
        message_node_grid_mlp: bool = False,
        message_node_grid_branch: int = 0,
        ffn_grid_mlp: bool = False,
        ffn_grid_branch: int = 0,
        ffn_blocks: int = 1,
        layer_scale: bool = False,
        full_attn_res: str = "none",
        block_attn_res: str = "none",
        so2_s2_activation: bool = False,
        node_wise_s2: bool = False,
        node_wise_so3: bool = False,
        message_node_s2: bool = False,
        message_node_so3: bool = False,
        ffn_s2_activation: bool = False,
        ffn_so3_grid: bool = False,
        so2_lebedev_quadrature: bool = False,
        ffn_lebedev_quadrature: bool = False,
        so2_activation_function: str = "silu",
        ffn_activation_function: str,
        ffn_glu_activation: bool = True,
        mlp_bias: bool = False,
        eps: float = 1e-7,
        precision: str = DEFAULT_PRECISION,
        seed: int | list[int] | None = None,
        trainable: bool = True,
    ) -> None:
        self.lmax = int(lmax)
        self.node_lmax = self.lmax if node_lmax is None else int(node_lmax)
        if self.node_lmax < self.lmax:
            raise ValueError("`node_lmax` must be >= `lmax`")
        self.mp_ebed_dim = (self.lmax + 1) ** 2
        self.node_ebed_dim = (self.node_lmax + 1) ** 2
        self.mmax = int(self.lmax if mmax is None else mmax)
        if self.mmax < 0:
            raise ValueError("`mmax` must be non-negative")
        if self.mmax > self.lmax:
            raise ValueError("`mmax` must be <= `lmax`")
        self.kmax = int(kmax)
        if self.kmax < 0:
            raise ValueError("`kmax` must be non-negative")
        self.channels = int(channels)
        self.n_focus = int(n_focus)
        if self.n_focus < 1:
            raise ValueError("`n_focus` must be >= 1")
        self.focus_dim = int(focus_dim)
        if self.focus_dim < 0:
            raise ValueError("`focus_dim` must be >= 0")
        self.focus_compete = bool(focus_compete)
        self.so2_norm = bool(so2_norm)
        self.so2_layers = int(so2_layers)
        self.so2_attn_res_mode = str(so2_attn_res).lower()
        if self.so2_attn_res_mode not in ATTN_RES_MODES:
            raise ValueError(
                "`so2_attn_res` must be one of 'none', 'independent', or 'dependent'"
            )
        self.radial_so2_mode = str(radial_so2_mode).lower()
        self.radial_so2_rank = int(radial_so2_rank)
        self.n_atten_head = int(n_atten_head)
        self.atten_f_mix = bool(atten_f_mix)
        self.use_atten_v_proj = bool(atten_v_proj)
        self.use_atten_o_proj = bool(atten_o_proj)
        self.so2_pre_norm = bool(so2_pre_norm)
        self.so2_post_norm = bool(so2_post_norm)
        self.ffn_pre_norm = bool(ffn_pre_norm)
        self.ffn_post_norm = bool(ffn_post_norm)
        self.ffn_neurons = int(ffn_neurons)
        self.node_wise_grid_mlp = bool(node_wise_grid_mlp)
        self.node_wise_grid_branch = int(node_wise_grid_branch)
        self.message_node_grid_mlp = bool(message_node_grid_mlp)
        self.message_node_grid_branch = int(message_node_grid_branch)
        self.ffn_grid_mlp = bool(ffn_grid_mlp)
        self.ffn_grid_branch = int(ffn_grid_branch)
        if (
            min(
                self.node_wise_grid_branch,
                self.message_node_grid_branch,
                self.ffn_grid_branch,
            )
            < 0
        ):
            raise ValueError("grid branch counts must be non-negative")
        self.ffn_blocks = int(ffn_blocks)
        if self.ffn_blocks < 1:
            raise ValueError("`ffn_blocks` must be >= 1")
        self.layer_scale = bool(layer_scale)
        if self.layer_scale:
            # consumed by block.py itself (FFN-branch adam_ffn_layer_scales)
            raise NotImplementedError("layer_scale=True is not ported to dpmodel")
        self.full_attn_res_mode = str(full_attn_res).lower()
        if self.full_attn_res_mode not in ATTN_RES_MODES:
            raise ValueError(
                "`full_attn_res` must be one of 'none', 'independent', or 'dependent'"
            )
        self.block_attn_res_mode = str(block_attn_res).lower()
        if self.block_attn_res_mode not in ATTN_RES_MODES:
            raise ValueError(
                "`block_attn_res` must be one of 'none', 'independent', or 'dependent'"
            )
        if self.full_attn_res_mode != "none":
            raise NotImplementedError(
                "full_attn_res != 'none' (DepthAttnRes) is not ported to dpmodel"
            )
        if self.block_attn_res_mode != "none":
            raise NotImplementedError(
                "block_attn_res != 'none' (DepthAttnRes) is not ported to dpmodel"
            )
        self.so2_s2_activation = bool(so2_s2_activation)
        self.node_wise_s2 = bool(node_wise_s2)
        self.node_wise_so3 = bool(node_wise_so3)
        self.message_node_s2 = bool(message_node_s2)
        self.message_node_so3 = bool(message_node_so3)
        self.ffn_s2_activation = bool(ffn_s2_activation)
        self.ffn_so3_grid = bool(ffn_so3_grid)
        self.so2_lebedev_quadrature = bool(so2_lebedev_quadrature)
        self.ffn_lebedev_quadrature = bool(ffn_lebedev_quadrature)
        self.so2_activation_function = str(so2_activation_function)
        self.ffn_activation_function = str(ffn_activation_function)
        self.ffn_glu_activation = bool(ffn_glu_activation)
        self.mlp_bias = bool(mlp_bias)
        self.eps = float(eps)
        self.precision = precision
        self.compute_precision = _compute_precision(precision)
        self.trainable = bool(trainable)

        # === Step 0. Split deterministic seeds at the block top-level ===
        # pt also splits seed_full_attn / seed_block_attn (block.py:378-379);
        # those consumers are guarded above, so the splits are unused here.
        seed_so2_conv = child_seed(seed, 0)
        seed_ffn = child_seed(seed, 1)

        # === Step 1. SO(2) convolution branch norms ===
        # pt uses nn.Identity() for disabled norms (parameter-free); the
        # dpmodel equivalent is None.
        self.pre_so2_norm: EquivariantRMSNorm | None = (
            EquivariantRMSNorm(
                self.lmax,
                self.channels,
                n_focus=1,
                precision=self.compute_precision,
                trainable=self.trainable,
            )
            if self.so2_pre_norm
            else None
        )
        self.post_so2_norm: EquivariantRMSNorm | None = (
            EquivariantRMSNorm(
                self.lmax,
                self.channels,
                n_focus=1,
                precision=self.compute_precision,
                trainable=self.trainable,
            )
            if self.so2_post_norm
            else None
        )

        self.so2_conv = SO2Convolution(
            lmax=self.lmax,
            mmax=self.mmax,
            kmax=self.kmax,
            channels=self.channels,
            n_focus=self.n_focus,
            focus_dim=self.focus_dim,
            focus_compete=self.focus_compete,
            so2_norm=self.so2_norm,
            so2_layers=self.so2_layers,
            so2_attn_res=self.so2_attn_res_mode,
            radial_so2_mode=self.radial_so2_mode,
            radial_so2_rank=self.radial_so2_rank,
            layer_scale=self.layer_scale,
            n_atten_head=self.n_atten_head,
            atten_f_mix=self.atten_f_mix,
            atten_v_proj=self.use_atten_v_proj,
            atten_o_proj=self.use_atten_o_proj,
            s2_activation=self.so2_s2_activation,
            node_wise_grid_mlp=self.node_wise_grid_mlp,
            node_wise_grid_branch=self.node_wise_grid_branch,
            message_node_grid_mlp=self.message_node_grid_mlp,
            message_node_grid_branch=self.message_node_grid_branch,
            node_wise_s2=self.node_wise_s2,
            node_wise_so3=self.node_wise_so3,
            message_node_s2=self.message_node_s2,
            message_node_so3=self.message_node_so3,
            lebedev_quadrature=self.so2_lebedev_quadrature,
            activation_function=self.so2_activation_function,
            mlp_bias=self.mlp_bias,
            eps=self.eps,
            precision=self.precision,
            seed=seed_so2_conv,
            trainable=self.trainable,
        )

        # === Step 2. FFN subblock sequence ===
        pre_ffn_norms: list[EquivariantRMSNorm | None] = []
        post_ffn_norms: list[EquivariantRMSNorm | None] = []
        ffns: list[EquivariantFFN] = []

        for i in range(self.ffn_blocks):
            seed_ffn_i = child_seed(seed_ffn, i)
            pre_ffn_norms.append(
                EquivariantRMSNorm(
                    self.node_lmax,
                    self.channels,
                    n_focus=1,
                    precision=self.compute_precision,
                    trainable=self.trainable,
                )
                if self.ffn_pre_norm
                else None
            )
            post_ffn_norms.append(
                EquivariantRMSNorm(
                    self.node_lmax,
                    self.channels,
                    n_focus=1,
                    precision=self.compute_precision,
                    trainable=self.trainable,
                )
                if self.ffn_post_norm
                else None
            )
            ffns.append(
                EquivariantFFN(
                    lmax=self.node_lmax,
                    channels=self.channels,
                    hidden_channels=self.ffn_neurons,
                    kmax=self.kmax,
                    grid_mlp=self.ffn_grid_mlp,
                    grid_branch=self.ffn_grid_branch,
                    s2_activation=self.ffn_s2_activation,
                    ffn_so3_grid=self.ffn_so3_grid,
                    lebedev_quadrature=self.ffn_lebedev_quadrature,
                    activation_function=self.ffn_activation_function,
                    glu_activation=self.ffn_glu_activation,
                    mlp_bias=self.mlp_bias,
                    precision=self.precision,
                    trainable=self.trainable,
                    seed=seed_ffn_i,
                )
            )
        self.pre_ffn_norms = pre_ffn_norms
        self.post_ffn_norms = post_ffn_norms
        self.ffns = ffns

    def _run_so2_unit(
        self,
        x: Any,
        edge_cache: EdgeCache,
        radial_feat: Any,
    ) -> Any:
        """
        Run the SO(2) unit without an outer block-level residual shortcut.

        Parameters
        ----------
        x
            Canonical node features with shape `(N, D, 1, C)`.
        edge_cache
            Edge cache (padded layout; see ``edge_cache.EdgeCache``).
        radial_feat
            Per-edge radial features with shape (E, lmax+1, C).

        Returns
        -------
        Array
            SO(2) unit output with shape `(N, D, 1, C)`.
        """
        xp = array_api_compat.array_namespace(x)
        n_node = x.shape[0]
        channels = self.channels
        use_full_node = self.node_lmax == self.lmax
        x_so2 = x if use_full_node else x[:, : self.mp_ebed_dim, :, :]
        x_pre = x_so2 if self.pre_so2_norm is None else self.pre_so2_norm(x_so2)
        so2_unit_output = self.so2_conv(
            xp.reshape(x_pre, (n_node, x_so2.shape[1], channels)),
            edge_cache,
            radial_feat,
        )
        so2_unit_output = so2_unit_output[:, :, None, :]
        if self.post_so2_norm is not None:
            so2_unit_output = self.post_so2_norm(so2_unit_output)
        if use_full_node:
            return so2_unit_output
        # zero-pad the degrees above lmax (pt writes into x.new_zeros)
        pad = xp.zeros(
            (n_node, self.node_ebed_dim - self.mp_ebed_dim, 1, channels),
            dtype=x.dtype,
            device=array_api_compat.device(x),
        )
        return xp.concat([so2_unit_output, pad], axis=1)

    def _run_ffn_unit(self, x: Any, unit_idx: int) -> Any:
        """
        Run one FFN subblock without the outer unit-level residual shortcut.

        Parameters
        ----------
        x
            Canonical node features with shape `(N, D, 1, C)`.
        unit_idx
            FFN subblock index.

        Returns
        -------
        Array
            FFN unit output with shape `(N, D, 1, C)`.
        """
        pre_norm = self.pre_ffn_norms[unit_idx]
        post_norm = self.post_ffn_norms[unit_idx]
        x_pre = x if pre_norm is None else pre_norm(x)
        y = self.ffns[unit_idx](x_pre)
        if post_norm is not None:
            y = post_norm(y)
        return y

    def call(
        self,
        x: Any,
        edge_cache: EdgeCache,
        radial_feat: Any,
        unit_history: list[Any] | None = None,
    ) -> tuple[Any, None, None, None]:
        """
        Run the residual-connected block path (pt baseline path).

        Parameters
        ----------
        x
            Features with shape `(N, D, 1, C)`.
        edge_cache
            Edge cache (padded layout).
        radial_feat
            Per-edge radial features with shape (E, lmax+1, C).
        unit_history
            Unused in the residual-connected path (the pt AttnRes paths that
            consume it are not ported).

        Returns
        -------
        tuple[Array, None, None, None]
            Tuple `(block_output, None, None, None)` matching the pt
            baseline-path return convention.
        """
        so2_unit_output = self._run_so2_unit(x, edge_cache, radial_feat)
        ffn_state = x + so2_unit_output
        for i in range(self.ffn_blocks):
            ffn_state = ffn_state + self._run_ffn_unit(ffn_state, i)
        return ffn_state, None, None, None

    def _sub_modules(self) -> list[tuple[str, NativeOP | None]]:
        """Sub-modules with their pt module names (None = pt nn.Identity)."""
        subs: list[tuple[str, NativeOP | None]] = [
            ("pre_so2_norm", self.pre_so2_norm),
            ("post_so2_norm", self.post_so2_norm),
            ("so2_conv", self.so2_conv),
        ]
        for i in range(self.ffn_blocks):
            subs.append((f"pre_ffn_norms.{i}", self.pre_ffn_norms[i]))
            subs.append((f"post_ffn_norms.{i}", self.post_ffn_norms[i]))
            subs.append((f"ffns.{i}", self.ffns[i]))
        return subs

    def _variables(self) -> dict[str, Any]:
        """Variables keyed by the pt ``state_dict`` key names."""
        variables: dict[str, Any] = {}
        for prefix, sub in self._sub_modules():
            if sub is None:
                continue
            if isinstance(sub, SO2Convolution):
                sub_vars = sub._variables()
            else:
                sub_vars = sub.serialize()["@variables"]
            for key, value in sub_vars.items():
                variables[f"{prefix}.{key}"] = value
        return variables

    def _load_variables(self, variables: dict[str, Any]) -> None:
        """Load variables keyed by the pt ``state_dict`` key names."""
        variables = dict(variables)
        for name, sub in self._sub_modules():
            if sub is None:
                continue
            full = f"{name}."
            sv = {
                key[len(full) :]: value
                for key, value in variables.items()
                if key.startswith(full)
            }
            for key in list(variables):
                if key.startswith(full):
                    del variables[key]
            if not sv:
                raise KeyError(f"Missing variables with prefix: {full}")
            if isinstance(sub, SO2Convolution):
                sub._load_variables(sv)
            elif isinstance(sub, EquivariantFFN):
                sub._load_variables(sv)
            else:
                # norms: rebuild through the shape-checking deserialize
                data = sub.serialize()
                data["@variables"] = sv
                new_sub = type(sub).deserialize(data)
                attr, _, idx = name.partition(".")
                if idx:
                    getattr(self, attr)[int(idx)] = new_sub
                else:
                    setattr(self, attr, new_sub)
        if variables:
            raise KeyError(f"Unknown variables: {sorted(variables)}")

    def serialize(self) -> dict[str, Any]:
        """Serialize the SeZMInteractionBlock to a dict (pt-compatible format)."""
        return {
            "@class": "SeZMInteractionBlock",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "node_lmax": self.node_lmax,
                "mmax": self.mmax,
                "kmax": self.kmax,
                "channels": self.channels,
                "n_focus": self.n_focus,
                "focus_dim": self.focus_dim,
                "focus_compete": self.focus_compete,
                "so2_norm": self.so2_norm,
                "so2_layers": self.so2_layers,
                "so2_attn_res": self.so2_attn_res_mode,
                "radial_so2_mode": self.radial_so2_mode,
                "radial_so2_rank": self.radial_so2_rank,
                "n_atten_head": self.n_atten_head,
                "atten_f_mix": self.atten_f_mix,
                "atten_v_proj": self.use_atten_v_proj,
                "atten_o_proj": self.use_atten_o_proj,
                "so2_pre_norm": self.so2_pre_norm,
                "so2_post_norm": self.so2_post_norm,
                "ffn_pre_norm": self.ffn_pre_norm,
                "ffn_post_norm": self.ffn_post_norm,
                "ffn_neurons": self.ffn_neurons,
                "node_wise_grid_mlp": self.node_wise_grid_mlp,
                "node_wise_grid_branch": self.node_wise_grid_branch,
                "message_node_grid_mlp": self.message_node_grid_mlp,
                "message_node_grid_branch": self.message_node_grid_branch,
                "ffn_grid_mlp": self.ffn_grid_mlp,
                "ffn_grid_branch": self.ffn_grid_branch,
                "ffn_blocks": self.ffn_blocks,
                "full_attn_res": self.full_attn_res_mode,
                "block_attn_res": self.block_attn_res_mode,
                "so2_s2_activation": self.so2_s2_activation,
                "node_wise_s2": self.node_wise_s2,
                "node_wise_so3": self.node_wise_so3,
                "message_node_s2": self.message_node_s2,
                "message_node_so3": self.message_node_so3,
                "ffn_s2_activation": self.ffn_s2_activation,
                "ffn_so3_grid": self.ffn_so3_grid,
                "so2_lebedev_quadrature": self.so2_lebedev_quadrature,
                "ffn_lebedev_quadrature": self.ffn_lebedev_quadrature,
                "so2_activation_function": self.so2_activation_function,
                "ffn_activation_function": self.ffn_activation_function,
                "ffn_glu_activation": self.ffn_glu_activation,
                "mlp_bias": self.mlp_bias,
                "layer_scale": self.layer_scale,
                "eps": self.eps,
                "precision": np.dtype(PRECISION_DICT[self.precision]).name,
                "trainable": self.trainable,
                "seed": None,
            },
            "@variables": self._variables(),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> SeZMInteractionBlock:
        """Deserialize a SeZMInteractionBlock from a dict."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "SeZMInteractionBlock":
            raise ValueError(f"Invalid class for SeZMInteractionBlock: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        config = dict(data.pop("config"))
        variables = data.pop("@variables")
        config["precision"] = str(config.pop("precision"))
        obj = cls(**config)
        obj._load_variables(variables)
        return obj
