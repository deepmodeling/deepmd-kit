# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Interaction blocks for the SeZM descriptor.

This module defines the SeZM interaction block that combines SO(2)
message passing, equivariant feed-forward subblocks, and optional
attention-residual history aggregation.
"""

from __future__ import (
    annotations,
)

from typing import (
    TYPE_CHECKING,
    Any,
)

import torch
import torch.nn as nn

from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
    RESERVED_PRECISION_DICT,
)

from .attn_res import (
    DepthAttnRes,
)
from .ffn import (
    EquivariantFFN,
)
from .norm import (
    EquivariantRMSNorm,
)
from .so2 import (
    SO2Convolution,
)
from .utils import (
    ATTN_RES_MODES,
    get_promoted_dtype,
    np_safe,
    nvtx_range,
    safe_numpy_to_tensor,
)

if TYPE_CHECKING:
    from .edge_cache import (
        EdgeFeatureCache,
    )


class SeZMInteractionBlock(nn.Module):
    """
    SeZM interaction block with SO(2) message passing and equivariant FFN stack.

    Branch order:
    1. SO(2) branch: optional pre-norm -> `SO2Convolution` -> optional post-norm.
    2. FFN branch: repeated subblocks of
       optional pre-norm -> `EquivariantFFN` -> optional post-norm.

    In the baseline path, outer residual shortcuts are applied around the SO(2)
    unit and each FFN subblock. In AttnRes paths, these shortcuts are replaced by
    selective depth-wise aggregation before each unit.

    `SO2Convolution` internally handles the real multi-focus expansion, so this
    block keeps a singleton-focus backbone layout `(N, D, 1, C)` at boundaries.

    Parameters
    ----------
    lmax
        Maximum spherical harmonic degree.
    mmax
        Maximum SO(2) order (|m|) mixed inside SO(2) convolution.
    channels
        Total channels per (l, m) coefficient.
    n_focus
        Number of multi-focus streams used only by the internal SO(2) branch.
    focus_dim
        Hidden width per focus stream used inside the SO(2) branch.
        ``focus_dim=0`` means using ``channels``.
    focus_compete
        If True, enable cross-focus softmax competition in SO(2) convolution.
    so2_norm
        If True, apply intermediate ReducedEquivariantRMSNorm between SO(2) mixing layers.
        When False (default), no normalization is applied between layers.
    so2_layers
        Number of SO(2) mixing layers.
    so2_attn_res
        Depth-wise attention residual mode across the internal SO(2) layer
        history. Must be one of ``"none"``, ``"independent"``, or
        ``"dependent"``.
    radial_so2_mode
        Dynamic radial degree mixer mode inside SO(2) convolution. ``"none"``
        applies elementwise radial modulation, ``"degree"`` uses a
        channel-shared edge-conditioned cross-degree kernel, and
        ``"degree_channel"`` uses a per-channel cross-degree kernel.
    radial_so2_rank
        Low-rank channel factorization rank for
        ``radial_so2_mode="degree_channel"``. ``0`` uses the full
        per-channel dynamic degree kernel.
    n_atten_head
        Number of attention heads when aggregating messages in SO(2) convolution.
        0 means no attention is used; >0 enables envelope-gated grouped softmax
        attention with output-side head gate.
    atten_f_mix
        If True, merge SO(2) focus streams into one attention stream after
        rotate-back. This gives each attention head access to the full
        multi-focus hidden width.
    atten_v_proj
        If True, apply an explicit degree-aware value projection inside SO(2)
        attention.
    atten_o_proj
        If True, apply an explicit degree-aware output projection inside SO(2)
        attention.
    so2_pre_norm
        If True, apply pre-norm before SO(2) convolution.
    so2_post_norm
        If True, apply post-norm on SO(2) output before the residual add.
    ffn_pre_norm
        If True, apply pre-norm before each FFN subblock.
    ffn_post_norm
        If True, apply post-norm on each FFN subblock output before the residual add.
    ffn_neurons
        Hidden dimension for each FFN subblock.
    grid_mlp
        If True, use the optional grid-MLP structure for the block-internal FFN
        units. The final descriptor output head is unaffected.
    ffn_blocks
        Number of FFN subblocks per block.
    layer_scale
        If True, apply learnable LayerScale (init 1e-3) on residual branches:
        - SO(2) branch: per-focus-channel scales `(n_focus, focus_dim)`
          on each SO(2) mixing layer.
        - FFN branch: per-channel scales `(channels,)` on each FFN subblock.
    full_attn_res
        Descriptor-level full attention residual mode for this block wrapper.
        When enabled, the block uses external unit history to build the SO(2)
        input and the input of each FFN unit.
    block_attn_res
        Descriptor-level block attention residual mode for this block wrapper.
        When enabled, the block uses external block history plus an intra-block
        partial sum to build the SO(2) input and the input of each FFN unit.
    so2_s2_activation
        If True, enable the merged scalar/grid SwiGLU-S2 activation in the SO(2)
        branch.
    ffn_s2_activation
        If True, enable the merged scalar/grid SwiGLU-S2 activation in the
        default FFN activation path.
    so2_lebedev_quadrature
        If True, use Lebedev quadrature for the SO(2) S2 activation projector.
    ffn_lebedev_quadrature
        If True, use Lebedev quadrature for the FFN S2 activation projector.
    so2_activation_function
        Activation function for the block-internal SO(2) l=0 gated activation
        path when ``so2_s2_activation=False``.
    ffn_activation_function
        Activation function for the block-internal FFN l=0 components.
    ffn_glu_activation
        If True, use GLU-style gating in the block-internal FFN
        (e.g., silu -> swiglu, gelu -> geglu).
    mlp_bias
        Whether to use bias in equivariant layers. Controls:
        - SO3Linear: l=0 bias
        - SO2Linear: l=0 bias
        - GatedActivation: gate linear bias
    use_triton
        If True, opt into fused Triton SO(2) rotation kernels inside
        ``SO2Convolution`` when the runtime supports them.
    eps
        Small epsilon for numerical stability.
    dtype
        Parameter dtype.
    seed
        Random seed for weight initialization.
    trainable
        Whether parameters are trainable.
    """

    def __init__(
        self,
        *,
        lmax: int,
        mmax: int | None = None,
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
        grid_mlp: bool = False,
        ffn_blocks: int = 1,
        layer_scale: bool = False,
        full_attn_res: str = "none",
        block_attn_res: str = "none",
        so2_s2_activation: bool = False,
        ffn_s2_activation: bool = False,
        so2_lebedev_quadrature: bool = False,
        ffn_lebedev_quadrature: bool = False,
        so2_activation_function: str = "silu",
        ffn_activation_function: str,
        ffn_glu_activation: bool = True,
        mlp_bias: bool = False,
        use_triton: bool = False,
        eps: float = 1e-7,
        dtype: torch.dtype,
        seed: int | list[int] | None,
        trainable: bool,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.mmax = int(self.lmax if mmax is None else mmax)
        if self.mmax < 0:
            raise ValueError("`mmax` must be non-negative")
        if self.mmax > self.lmax:
            raise ValueError("`mmax` must be <= `lmax`")
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
        self.grid_mlp = bool(grid_mlp)
        self.ffn_blocks = int(ffn_blocks)
        if self.ffn_blocks < 1:
            raise ValueError("`ffn_blocks` must be >= 1")
        self.layer_scale = bool(layer_scale)
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
        self.use_full_attn_res = self.full_attn_res_mode != "none"
        self.use_block_attn_res = self.block_attn_res_mode != "none"
        if self.use_full_attn_res and self.use_block_attn_res:
            raise ValueError(
                "`full_attn_res` and `block_attn_res` cannot both be enabled"
            )
        self.so2_s2_activation = bool(so2_s2_activation)
        self.ffn_s2_activation = bool(ffn_s2_activation)
        self.so2_lebedev_quadrature = bool(so2_lebedev_quadrature)
        self.ffn_lebedev_quadrature = bool(ffn_lebedev_quadrature)
        self.so2_activation_function = str(so2_activation_function)
        self.ffn_activation_function = str(ffn_activation_function)
        self.ffn_glu_activation = bool(ffn_glu_activation)
        self.mlp_bias = bool(mlp_bias)
        self.use_triton = bool(use_triton)
        self.eps = float(eps)
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[dtype]
        self.compute_dtype = get_promoted_dtype(self.dtype)

        # === Step 0. Split deterministic seeds at the block top-level ===
        seed_so2_conv = child_seed(seed, 0)
        seed_ffn = child_seed(seed, 1)
        seed_full_attn = child_seed(seed, 2)
        seed_block_attn = child_seed(seed, 3)

        # === Step 1. SO(2) convolution branch norms ===
        if self.so2_pre_norm:
            self.pre_so2_norm: nn.Module = EquivariantRMSNorm(
                self.lmax,
                self.channels,
                n_focus=1,
                dtype=self.compute_dtype,
                trainable=trainable,
            )
        else:
            self.pre_so2_norm = nn.Identity()

        if self.so2_post_norm:
            self.post_so2_norm: nn.Module = EquivariantRMSNorm(
                self.lmax,
                self.channels,
                n_focus=1,
                dtype=self.compute_dtype,
                trainable=trainable,
            )
        else:
            self.post_so2_norm = nn.Identity()

        self.so2_conv = SO2Convolution(
            lmax=self.lmax,
            mmax=self.mmax,
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
            n_atten_head=n_atten_head,
            atten_f_mix=self.atten_f_mix,
            atten_v_proj=self.use_atten_v_proj,
            atten_o_proj=self.use_atten_o_proj,
            s2_activation=self.so2_s2_activation,
            lebedev_quadrature=self.so2_lebedev_quadrature,
            activation_function=self.so2_activation_function,
            mlp_bias=self.mlp_bias,
            use_triton=self.use_triton,
            eps=self.eps,
            dtype=dtype,
            seed=seed_so2_conv,
            trainable=trainable,
        )

        # === Step 2. FFN subblock sequence ===
        pre_ffn_norms: list[nn.Module] = []
        post_ffn_norms: list[nn.Module] = []
        ffns: list[EquivariantFFN] = []

        for i in range(self.ffn_blocks):
            seed_ffn_i = child_seed(seed_ffn, i)

            if self.ffn_pre_norm:
                pre_ffn_norms.append(
                    EquivariantRMSNorm(
                        self.lmax,
                        self.channels,
                        n_focus=1,
                        dtype=self.compute_dtype,
                        trainable=trainable,
                    )
                )
            else:
                pre_ffn_norms.append(nn.Identity())

            if self.ffn_post_norm:
                post_ffn_norms.append(
                    EquivariantRMSNorm(
                        self.lmax,
                        self.channels,
                        n_focus=1,
                        dtype=self.compute_dtype,
                        trainable=trainable,
                    )
                )
            else:
                post_ffn_norms.append(nn.Identity())

            ffns.append(
                EquivariantFFN(
                    lmax=self.lmax,
                    channels=self.channels,
                    hidden_channels=ffn_neurons,
                    grid_mlp=self.grid_mlp,
                    dtype=dtype,
                    s2_activation=self.ffn_s2_activation,
                    lebedev_quadrature=self.ffn_lebedev_quadrature,
                    activation_function=self.ffn_activation_function,
                    glu_activation=self.ffn_glu_activation,
                    mlp_bias=self.mlp_bias,
                    trainable=trainable,
                    seed=seed_ffn_i,
                )
            )

        self.pre_ffn_norms = nn.ModuleList(pre_ffn_norms)
        self.post_ffn_norms = nn.ModuleList(post_ffn_norms)
        self.ffns = nn.ModuleList(ffns)

        # Optional per-channel LayerScale on each FFN residual branch
        if self.layer_scale:
            self.adam_ffn_layer_scales = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.ones(self.channels, dtype=self.dtype, device=self.device)
                        * 1e-3,
                        requires_grad=trainable,
                    )
                    for _ in range(self.ffn_blocks)
                ]
            )
        else:
            self.adam_ffn_layer_scales = None

        # === Step 3. Optional full attention residuals for block inputs ===
        if self.use_full_attn_res:
            self.full_attn_res_so2: DepthAttnRes | None = DepthAttnRes(
                channels=self.channels,
                input_dependent=self.full_attn_res_mode == "dependent",
                eps=self.eps,
                bias=self.mlp_bias,
                dtype=self.compute_dtype,
                trainable=trainable,
                seed=child_seed(seed_full_attn, 0),
            )
            self.full_attn_res_ffns: nn.ModuleList | None = nn.ModuleList(
                [
                    DepthAttnRes(
                        channels=self.channels,
                        input_dependent=self.full_attn_res_mode == "dependent",
                        eps=self.eps,
                        bias=self.mlp_bias,
                        dtype=self.compute_dtype,
                        trainable=trainable,
                        seed=child_seed(seed_full_attn, i + 1),
                    )
                    for i in range(self.ffn_blocks)
                ]
            )
            self.block_attn_res_so2 = None
            self.block_attn_res_ffns = None
            self._forward_impl = self._forward_with_full_attn_res
        elif self.use_block_attn_res:
            self.full_attn_res_so2 = None
            self.full_attn_res_ffns = None
            self.block_attn_res_so2: DepthAttnRes | None = DepthAttnRes(
                channels=self.channels,
                input_dependent=self.block_attn_res_mode == "dependent",
                eps=self.eps,
                bias=self.mlp_bias,
                dtype=self.compute_dtype,
                trainable=trainable,
                seed=child_seed(seed_block_attn, 0),
            )
            self.block_attn_res_ffns: nn.ModuleList | None = nn.ModuleList(
                [
                    DepthAttnRes(
                        channels=self.channels,
                        input_dependent=self.block_attn_res_mode == "dependent",
                        eps=self.eps,
                        bias=self.mlp_bias,
                        dtype=self.compute_dtype,
                        trainable=trainable,
                        seed=child_seed(seed_block_attn, i + 1),
                    )
                    for i in range(self.ffn_blocks)
                ]
            )
            self._forward_impl = self._forward_with_block_attn_res
        else:
            self.full_attn_res_so2 = None
            self.full_attn_res_ffns = None
            self.block_attn_res_so2 = None
            self.block_attn_res_ffns = None
            self._forward_impl = self._forward_with_residual_shortcuts

    def forward(
        self,
        x: torch.Tensor,
        edge_cache: EdgeFeatureCache,
        radial_feat: torch.Tensor,
        unit_history: list[torch.Tensor] | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        list[torch.Tensor] | None,
    ]:
        """
        Parameters
        ----------
        x
            Features with shape `(N, D, 1, C)`.
        edge_cache
            Edge cache.
        radial_feat
            Per-edge radial features with shape (E, lmax+1, C).
        unit_history
            Optional truncated depth history in canonical node layout. When
            `full_attn_res != "none"`, it is interpreted as completed unit
            history. When `block_attn_res != "none"`, it is interpreted as
            completed block history.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, list[torch.Tensor] | None]
            Tuple `(block_output, block_summary, so2_unit_output, ffn_unit_outputs)`
            in canonical node layout. `block_output` is always returned.
            Auxiliary outputs are mode-dependent and may be `None` when the
            current caller does not need them:

            - baseline path returns `(block_output, None, None, None)`
            - full AttnRes path returns `(block_output, None, so2_unit_output, ffn_unit_outputs)`
            - block AttnRes path returns `(block_output, block_summary, None, None)`
        """
        return self._forward_impl(x, edge_cache, radial_feat, unit_history)

    def _extract_l0_from_canonical(self, value: torch.Tensor) -> torch.Tensor:
        """
        Extract scalar channels from canonical node layout.

        Parameters
        ----------
        value
            Canonical node features with shape `(N, D, 1, C)`.

        Returns
        -------
        torch.Tensor
            Scalar channels with shape (N, channels).
        """
        return value[:, 0, :, :].reshape(value.shape[0], self.channels)

    def _run_so2_unit(
        self,
        x: torch.Tensor,
        edge_cache: EdgeFeatureCache,
        radial_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the SO(2) unit without an outer block-level residual shortcut.

        Parameters
        ----------
        x
            Canonical node features with shape `(N, D, 1, C)`.
        edge_cache
            Edge cache.
        radial_feat
            Per-edge radial features with shape (E, lmax+1, C).

        Returns
        -------
        torch.Tensor
            SO(2) unit output with shape `(N, D, 1, C)`.
        """
        n_node = x.shape[0]
        ebed_dim = x.shape[1]
        channels = self.channels
        x_pre = self.pre_so2_norm(x)
        so2_unit_output = self.so2_conv(
            x_pre.reshape(n_node, ebed_dim, channels), edge_cache, radial_feat
        )
        return self.post_so2_norm(so2_unit_output.unsqueeze(2))

    def _run_ffn_unit(self, x: torch.Tensor, unit_idx: int) -> torch.Tensor:
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
        torch.Tensor
            FFN unit output with shape `(N, D, 1, C)`.
        """
        n_node = x.shape[0]
        ebed_dim = x.shape[1]
        channels = self.channels
        x_ffn = x.reshape(n_node, ebed_dim, 1, channels)  # (N, D, 1, C)
        x_pre = self.pre_ffn_norms[unit_idx](x_ffn)
        y: torch.Tensor = self.ffns[unit_idx](x_pre)
        y = self.post_ffn_norms[unit_idx](y)
        if self.layer_scale:
            y = y * self.adam_ffn_layer_scales[unit_idx]
        return y

    def _forward_with_residual_shortcuts(
        self,
        x: torch.Tensor,
        edge_cache: EdgeFeatureCache,
        radial_feat: torch.Tensor,
        unit_history: list[torch.Tensor] | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        list[torch.Tensor] | None,
    ]:
        """
        Run the original residual-connected block path.

        Parameters
        ----------
        x
            Canonical node features with shape `(N, D, 1, C)`.
        edge_cache
            Edge cache.
        radial_feat
            Per-edge radial features with shape (E, lmax+1, C).
        unit_history
            Unused in the residual-connected path.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, list[torch.Tensor] | None]
            Tuple `(block_output, None, None, None)`.
        """
        with nvtx_range("so2_conv"):
            so2_unit_output = self._run_so2_unit(x, edge_cache, radial_feat)
            so2_state = x + so2_unit_output

        with nvtx_range("ffn"):
            ffn_state = so2_state
            for i in range(self.ffn_blocks):
                ffn_unit_output = self._run_ffn_unit(ffn_state, i)
                ffn_state = ffn_state + ffn_unit_output

        block_output = ffn_state
        return block_output, None, None, None

    def _forward_with_full_attn_res(
        self,
        x: torch.Tensor,
        edge_cache: EdgeFeatureCache,
        radial_feat: torch.Tensor,
        unit_history: list[torch.Tensor] | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        list[torch.Tensor] | None,
    ]:
        """
        Run the block with full attention residuals over unit history.

        Parameters
        ----------
        x
            Current block input with shape `(N, D, 1, C)`.
        edge_cache
            Edge cache.
        radial_feat
            Per-edge radial features with shape (E, lmax+1, C).
        unit_history
            Truncated history in canonical node layout. Each source has shape
            `(N, D, 1, C)`.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, list[torch.Tensor] | None]
            Tuple `(block_output, None, so2_unit_output, ffn_unit_outputs)`.
        """
        with nvtx_range("so2_conv"):
            with nvtx_range("full_attn_res"):
                so2_input = self.full_attn_res_so2(
                    sources=unit_history,
                    scalar_extractor=self._extract_l0_from_canonical,
                    current_x=x,
                )
            so2_unit_output = self._run_so2_unit(so2_input, edge_cache, radial_feat)

        with nvtx_range("ffn"):
            completed_units = [*unit_history, so2_unit_output]
            current_x = so2_unit_output
            ffn_unit_outputs: list[torch.Tensor] = []
            for i in range(self.ffn_blocks):
                with nvtx_range("full_attn_res"):
                    ffn_input: torch.Tensor = self.full_attn_res_ffns[i](
                        sources=completed_units,
                        scalar_extractor=self._extract_l0_from_canonical,
                        current_x=current_x,
                    )
                ffn_unit_output = self._run_ffn_unit(ffn_input, i)
                ffn_unit_outputs.append(ffn_unit_output)
                completed_units.append(ffn_unit_output)
                current_x = ffn_unit_output

        block_output = current_x
        return block_output, None, so2_unit_output, ffn_unit_outputs

    def _forward_with_block_attn_res(
        self,
        x: torch.Tensor,
        edge_cache: EdgeFeatureCache,
        radial_feat: torch.Tensor,
        unit_history: list[torch.Tensor] | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        list[torch.Tensor] | None,
    ]:
        """
        Run the block with block attention residuals over block history.

        Parameters
        ----------
        x
            Current block input with shape `(N, D, 1, C)`.
        edge_cache
            Edge cache.
        radial_feat
            Per-edge radial features with shape (E, lmax+1, C).
        unit_history
            Truncated block history in canonical node layout. Each source has shape
            `(N, D, 1, C)`.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, list[torch.Tensor] | None]
            Tuple `(block_output, block_summary, None, None)`.
        """
        with nvtx_range("so2_conv"):
            with nvtx_range("block_attn_res"):
                so2_input = self.block_attn_res_so2(
                    sources=unit_history,
                    scalar_extractor=self._extract_l0_from_canonical,
                    current_x=x,
                )
            so2_unit_output = self._run_so2_unit(so2_input, edge_cache, radial_feat)

        with nvtx_range("ffn"):
            partial_block = so2_unit_output
            current_x = so2_unit_output
            for i in range(self.ffn_blocks):
                with nvtx_range("block_attn_res"):
                    ffn_input: torch.Tensor = self.block_attn_res_ffns[i](
                        sources=[*unit_history, partial_block],
                        scalar_extractor=self._extract_l0_from_canonical,
                        current_x=current_x,
                    )
                ffn_unit_output = self._run_ffn_unit(ffn_input, i)
                partial_block = partial_block + ffn_unit_output
                current_x = ffn_unit_output

        block_output = current_x
        block_summary = partial_block
        return block_output, block_summary, None, None

    def serialize(self) -> dict[str, Any]:
        trainable = all(p.requires_grad for p in self.parameters())
        state = self.state_dict()
        return {
            "@class": "SeZMInteractionBlock",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "mmax": self.mmax,
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
                "grid_mlp": self.grid_mlp,
                "ffn_blocks": self.ffn_blocks,
                "full_attn_res": self.full_attn_res_mode,
                "block_attn_res": self.block_attn_res_mode,
                "so2_s2_activation": self.so2_s2_activation,
                "ffn_s2_activation": self.ffn_s2_activation,
                "so2_lebedev_quadrature": self.so2_lebedev_quadrature,
                "ffn_lebedev_quadrature": self.ffn_lebedev_quadrature,
                "so2_activation_function": self.so2_activation_function,
                "ffn_activation_function": self.ffn_activation_function,
                "ffn_glu_activation": self.ffn_glu_activation,
                "mlp_bias": self.mlp_bias,
                "layer_scale": self.layer_scale,
                "eps": self.eps,
                "precision": RESERVED_PRECISION_DICT[self.dtype],
                "trainable": trainable,
                "seed": None,
            },
            "@variables": {key: np_safe(value) for key, value in state.items()},
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> SeZMInteractionBlock:
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "SeZMInteractionBlock":
            raise ValueError(f"Invalid class for SeZMInteractionBlock: {data_cls}")
        version = int(data.pop("@version"))
        if version != 1:
            raise ValueError(f"Unsupported SeZMInteractionBlock version: {version}")
        config = data.pop("config")
        variables = data.pop("@variables")
        precision = config.pop("precision")
        config["dtype"] = PRECISION_DICT[precision]
        obj = cls(**config)
        template = obj.state_dict()
        state = {
            key: safe_numpy_to_tensor(
                value, device=template[key].device, dtype=template[key].dtype
            )
            for key, value in variables.items()
        }
        obj.load_state_dict(state)
        return obj
