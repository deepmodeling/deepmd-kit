# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Interaction blocks for the DPA4/SeZM descriptor.

This module defines the SeZM interaction block that combines SO(2)
message passing, equivariant feed-forward subblocks, and optional
attention-residual history aggregation.

This module is the dpmodel (array-API) port of
``deepmd.pt.model.descriptor.sezm_nn.block``.
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
from deepmd.dpmodel.array_api import (
    xp_asarray_nodetach,
)
from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.utils.network import (
    Identity,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.utils.version import (
    check_version_compatibility,
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
)

if TYPE_CHECKING:
    from deepmd.dpmodel.array_api import (
        Array,
    )

    from .edge_cache import (
        EdgeFeatureCache,
    )


def exchange_ghost_features(
    x: Array,
    comm_dict: dict[str, Array],
) -> Array:
    """
    Refresh ghost-node features from their owner ranks via MPI border exchange.

    SeZM node features are SO(3) coefficients expressed in the shared global
    frame, so a ghost atom and its owner carry identical features and the
    per-row owner->ghost copy is exact and equivariance-preserving. The opaque
    ``deepmd_export::border_op`` performs the exchange and carries a registered
    backward (reverse communication of gradients), so a single
    ``autograd.grad(energy, edge_vec)`` accumulates cross-rank force
    contributions when every rank runs the exchange in lockstep.

    This is applied to the SO(2) convolution input — the descriptor's only
    cross-node operation — so ghost rows are correct exactly where message
    passing reads them, regardless of how the (per-node) attention-residual
    history that produced the input populated its ghost rows.

    Parameters
    ----------
    x
        Extended node features with shape (nall, D, 1, channels). Owned-atom rows
        hold up-to-date values; ghost rows are overwritten by this call.
    comm_dict
        Border-exchange tensors ``send_list``, ``send_proc``, ``recv_proc``,
        ``send_num``, ``recv_num``, ``communicator``, ``nlocal``, ``nghost``.

    Returns
    -------
    Array
        Node features with ghost rows filled, same shape as ``x``.
    """
    raise NotImplementedError(
        "Multi-rank border exchange (comm_dict) is not supported in the "
        "dpmodel backend."
    )


class SeZMInteractionBlock(NativeOP):
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
        Maximum message-passing spherical harmonic degree.
    node_lmax
        Maximum node representation degree. If None, equals `lmax`.
    mmax
        Maximum SO(2) order (|m|) mixed inside SO(2) convolution.
    kmax
        Maximum Wigner-D frame order (|k|) used by SO(3) grid branches.
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
    mixing_layers
        Number of learnable mixing layers in the per-edge message core. ``0``
        applies only the edge-condition modulation.
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
    edge_cartesian
        If True, replace the per-edge SO(2) rotation-frame tensor product inside
        ``SO2Convolution`` with the global-frame Cartesian rank-2 tensor
        product. Requires ``lmax`` in ``{1, 2}``.
    node_cartesian
        Per-node global-frame Cartesian rank-2 tensor product on the aggregated
        message inside ``SO2Convolution``, configured by a ``"<mode>:<layers>"``
        string (``mode`` is ``"default"`` or ``"parity"``); a bare integer ``N``
        is shorthand for ``"default:N"``, and ``"none"`` disables it. Requires
        ``lmax`` in ``{1, 2}`` and is orthogonal to ``edge_cartesian``.
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
    node_wise_grid_mlp
        If True, select the polynomial grid MLP operation for the SO(2)
        convolution node-wise cross-grid path.
    node_wise_grid_branch
        Number of scalar-routed polynomial product branches for the node-wise
        cross-grid path. ``0`` disables branch mixing; positive values take
        precedence over ``node_wise_grid_mlp``.
    message_node_grid_mlp
        If True, select the polynomial grid MLP operation for the SO(2)
        convolution message-node cross-grid path.
    message_node_grid_branch
        Number of scalar-routed polynomial product branches for the
        message-node cross-grid path. ``0`` disables branch mixing; positive
        values take precedence over ``message_node_grid_mlp``.
    ffn_grid_mlp
        If True, select the polynomial grid MLP operation for the
        block-internal FFN grid path.
    ffn_grid_branch
        Number of scalar-routed polynomial product branches for the FFN grid
        path. ``0`` disables branch mixing; positive values take precedence
        over ``ffn_grid_mlp``.
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
    node_wise_s2
        If True, enable the edge-local source-destination S2 product branch in
        the SO(2) convolution.
    node_wise_so3
        If True, enable the corresponding edge-local SO(3) Wigner-D grid branch
        in the SO(2) convolution.
    message_node_s2
        If True, enable the post-aggregation message-node S2 product branch in
        the SO(2) convolution.
    message_node_so3
        If True, enable the corresponding post-aggregation SO(3) Wigner-D grid
        branch in the SO(2) convolution.
    ffn_s2_activation
        If True, enable the merged scalar/grid SwiGLU-S2 activation in the
        default FFN activation path.
    ffn_so3_grid
        If True, use the SO(3) Wigner-D grid in the block-internal FFN. This
        takes precedence over ``ffn_s2_activation``.
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
    eps
        Small epsilon for numerical stability.
    precision
        Parameter precision.
    seed
        Random seed for weight initialization.
    trainable
        Whether parameters are trainable.
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
        mixing_layers: int = 4,
        so2_attn_res: str = "none",
        radial_so2_mode: str = "none",
        radial_so2_rank: int = 0,
        edge_cartesian: bool = False,
        node_cartesian: str | int = "none",
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
        seed: int | list[int] | None,
        trainable: bool,
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
        self.mixing_layers = int(mixing_layers)
        self.so2_attn_res_mode = str(so2_attn_res).lower()
        if self.so2_attn_res_mode not in ATTN_RES_MODES:
            raise ValueError(
                "`so2_attn_res` must be one of 'none', 'independent', or 'dependent'"
            )
        self.radial_so2_mode = str(radial_so2_mode).lower()
        self.radial_so2_rank = int(radial_so2_rank)
        self.edge_cartesian = bool(edge_cartesian)
        self.node_cartesian = str(node_cartesian)
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
        self.compute_precision = np.dtype(
            get_promoted_dtype(PRECISION_DICT[self.precision])
        ).name

        # === Step 0. Split deterministic seeds at the block top-level ===
        seed_so2_conv = child_seed(seed, 0)
        seed_ffn = child_seed(seed, 1)
        seed_full_attn = child_seed(seed, 2)
        seed_block_attn = child_seed(seed, 3)

        # === Step 1. SO(2) convolution branch norms ===
        if self.so2_pre_norm:
            self.pre_so2_norm = EquivariantRMSNorm(
                self.lmax,
                self.channels,
                n_focus=1,
                precision=self.compute_precision,
                trainable=trainable,
            )
        else:
            self.pre_so2_norm = Identity()

        if self.so2_post_norm:
            self.post_so2_norm = EquivariantRMSNorm(
                self.lmax,
                self.channels,
                n_focus=1,
                precision=self.compute_precision,
                trainable=trainable,
            )
        else:
            self.post_so2_norm = Identity()

        self.so2_conv = SO2Convolution(
            lmax=self.lmax,
            mmax=self.mmax,
            kmax=self.kmax,
            channels=self.channels,
            n_focus=self.n_focus,
            focus_dim=self.focus_dim,
            focus_compete=self.focus_compete,
            so2_norm=self.so2_norm,
            mixing_layers=self.mixing_layers,
            so2_attn_res=self.so2_attn_res_mode,
            radial_so2_mode=self.radial_so2_mode,
            radial_so2_rank=self.radial_so2_rank,
            edge_cartesian=self.edge_cartesian,
            node_cartesian=self.node_cartesian,
            layer_scale=self.layer_scale,
            n_atten_head=n_atten_head,
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
            precision=precision,
            seed=seed_so2_conv,
            trainable=trainable,
        )

        # === Step 2. FFN subblock sequence ===
        pre_ffn_norms: list = []
        post_ffn_norms: list = []
        ffns: list[EquivariantFFN] = []

        for i in range(self.ffn_blocks):
            seed_ffn_i = child_seed(seed_ffn, i)

            if self.ffn_pre_norm:
                pre_ffn_norms.append(
                    EquivariantRMSNorm(
                        self.node_lmax,
                        self.channels,
                        n_focus=1,
                        precision=self.compute_precision,
                        trainable=trainable,
                    )
                )
            else:
                pre_ffn_norms.append(Identity())

            if self.ffn_post_norm:
                post_ffn_norms.append(
                    EquivariantRMSNorm(
                        self.node_lmax,
                        self.channels,
                        n_focus=1,
                        precision=self.compute_precision,
                        trainable=trainable,
                    )
                )
            else:
                post_ffn_norms.append(Identity())

            ffns.append(
                EquivariantFFN(
                    lmax=self.node_lmax,
                    channels=self.channels,
                    hidden_channels=ffn_neurons,
                    kmax=self.kmax,
                    grid_mlp=self.ffn_grid_mlp,
                    grid_branch=self.ffn_grid_branch,
                    precision=precision,
                    s2_activation=self.ffn_s2_activation,
                    ffn_so3_grid=self.ffn_so3_grid,
                    lebedev_quadrature=self.ffn_lebedev_quadrature,
                    activation_function=self.ffn_activation_function,
                    glu_activation=self.ffn_glu_activation,
                    mlp_bias=self.mlp_bias,
                    trainable=trainable,
                    seed=seed_ffn_i,
                )
            )

        self.pre_ffn_norms = pre_ffn_norms
        self.post_ffn_norms = post_ffn_norms
        self.ffns = ffns

        # Optional per-channel LayerScale on each FFN residual branch
        if self.layer_scale:
            self.adam_ffn_layer_scales = [
                np.ones((self.channels,), dtype=PRECISION_DICT[self.precision]) * 1e-3
                for _ in range(self.ffn_blocks)
            ]
        else:
            self.adam_ffn_layer_scales = None

        # === Step 3. Optional full attention residuals for block inputs ===
        if self.use_full_attn_res:
            self.full_attn_res_so2: DepthAttnRes | None = DepthAttnRes(
                channels=self.channels,
                input_dependent=self.full_attn_res_mode == "dependent",
                eps=self.eps,
                bias=self.mlp_bias,
                precision=self.compute_precision,
                trainable=trainable,
                seed=child_seed(seed_full_attn, 0),
            )
            self.full_attn_res_ffns: list | None = [
                DepthAttnRes(
                    channels=self.channels,
                    input_dependent=self.full_attn_res_mode == "dependent",
                    eps=self.eps,
                    bias=self.mlp_bias,
                    precision=self.compute_precision,
                    trainable=trainable,
                    seed=child_seed(seed_full_attn, i + 1),
                )
                for i in range(self.ffn_blocks)
            ]
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
                precision=self.compute_precision,
                trainable=trainable,
                seed=child_seed(seed_block_attn, 0),
            )
            self.block_attn_res_ffns: list | None = [
                DepthAttnRes(
                    channels=self.channels,
                    input_dependent=self.block_attn_res_mode == "dependent",
                    eps=self.eps,
                    bias=self.mlp_bias,
                    precision=self.compute_precision,
                    trainable=trainable,
                    seed=child_seed(seed_block_attn, i + 1),
                )
                for i in range(self.ffn_blocks)
            ]
            self._forward_impl = self._forward_with_block_attn_res
        else:
            self.full_attn_res_so2 = None
            self.full_attn_res_ffns = None
            self.block_attn_res_so2 = None
            self.block_attn_res_ffns = None
            self._forward_impl = self._forward_with_residual_shortcuts

        self.trainable = bool(trainable)

    def call(
        self,
        x: Array,
        edge_cache: EdgeFeatureCache,
        radial_feat: Array,
        unit_history: list[Array] | None = None,
        comm_dict: dict[str, Array] | None = None,
    ) -> tuple[
        Array,
        Array | None,
        Array | None,
        list[Array] | None,
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
        comm_dict
            Border-exchange tensors for parallel (LAMMPS multi-rank) inference.
            When provided, the SO(2) convolution input has its ghost rows
            refreshed from owner ranks; the depth-attention history may carry
            stale ghost rows because the exchange happens at the convolution
            input, after the (per-node) aggregation that consumes it.

        Returns
        -------
        tuple[Array, Array | None, Array | None, list[Array] | None]
            Tuple `(block_output, block_summary, so2_unit_output, ffn_unit_outputs)`
            in canonical node layout. `block_output` is always returned.
            Auxiliary outputs are mode-dependent and may be `None` when the
            current caller does not need them:

            - baseline path returns `(block_output, None, None, None)`
            - full AttnRes path returns `(block_output, None, so2_unit_output, ffn_unit_outputs)`
            - block AttnRes path returns `(block_output, block_summary, None, None)`
        """
        return self._forward_impl(x, edge_cache, radial_feat, unit_history, comm_dict)

    def _extract_l0_from_canonical(self, value: Array) -> Array:
        """
        Extract scalar channels from canonical node layout.

        Parameters
        ----------
        value
            Canonical node features with shape `(N, D, 1, C)`.

        Returns
        -------
        Array
            Scalar channels with shape (N, channels).
        """
        xp = array_api_compat.array_namespace(value)
        return xp.reshape(value[:, 0, :, :], (value.shape[0], self.channels))

    def _run_so2_unit(
        self,
        x: Array,
        edge_cache: EdgeFeatureCache,
        radial_feat: Array,
        comm_dict: dict[str, Array] | None = None,
    ) -> Array:
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
        comm_dict
            Border-exchange tensors for parallel inference. When provided, the
            convolution input's ghost rows are refreshed from owner ranks
            immediately before the only cross-node operation in the block, so
            owned destinations gather up-to-date neighbour features.

        Returns
        -------
        Array
            SO(2) unit output with shape `(N, D, 1, C)`.
        """
        if comm_dict is not None:
            x = exchange_ghost_features(x, comm_dict)
        return self._run_so2_unit_impl(x, edge_cache, radial_feat)

    def _run_so2_unit_impl(
        self,
        x: Array,
        edge_cache: EdgeFeatureCache,
        radial_feat: Array,
    ) -> Array:
        """Run the SO(2) unit implementation."""
        xp = array_api_compat.array_namespace(x)
        n_node = x.shape[0]
        channels = self.channels
        use_full_node = self.node_lmax == self.lmax
        x_so2 = x if use_full_node else x[:, : self.mp_ebed_dim, :, :]
        x_pre = self.pre_so2_norm(x_so2)
        so2_unit_output = self.so2_conv(
            xp.reshape(x_pre, (n_node, x_so2.shape[1], channels)),
            edge_cache,
            radial_feat,
        )
        so2_unit_output = self.post_so2_norm(so2_unit_output[:, :, None, :])
        if use_full_node:
            return so2_unit_output
        output = xp.zeros(x.shape, dtype=x.dtype, device=array_api_compat.device(x))
        output = xp.concat(
            [so2_unit_output, output[:, self.mp_ebed_dim :, :, :]], axis=1
        )
        return output

    def _run_ffn_unit(self, x: Array, unit_idx: int) -> Array:
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
        return self._run_ffn_unit_impl(x, unit_idx)

    def _run_ffn_unit_impl(self, x: Array, unit_idx: int) -> Array:
        """Run one FFN subblock implementation."""
        xp = array_api_compat.array_namespace(x)
        n_node = x.shape[0]
        ebed_dim = x.shape[1]
        channels = self.channels
        x_ffn = xp.reshape(x, (n_node, ebed_dim, 1, channels))  # (N, D, 1, C)
        x_pre = self.pre_ffn_norms[unit_idx](x_ffn)
        y: Array = self.ffns[unit_idx](x_pre)
        y = self.post_ffn_norms[unit_idx](y)
        if self.layer_scale:
            device = array_api_compat.device(x)
            y = y * xp_asarray_nodetach(
                xp, self.adam_ffn_layer_scales[unit_idx][...], device=device
            )
        return y

    def _forward_with_residual_shortcuts(
        self,
        x: Array,
        edge_cache: EdgeFeatureCache,
        radial_feat: Array,
        unit_history: list[Array] | None = None,
        comm_dict: dict[str, Array] | None = None,
    ) -> tuple[
        Array,
        Array | None,
        Array | None,
        list[Array] | None,
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
        comm_dict
            Border-exchange tensors for parallel inference, forwarded to the
            SO(2) unit. The owned-atom residual reads the original ``x``, which
            is already correct on owned rows.

        Returns
        -------
        tuple[Array, Array | None, Array | None, list[Array] | None]
            Tuple `(block_output, None, None, None)`.
        """
        so2_unit_output = self._run_so2_unit(x, edge_cache, radial_feat, comm_dict)
        so2_state = x + so2_unit_output

        ffn_state = so2_state
        for i in range(self.ffn_blocks):
            ffn_unit_output = self._run_ffn_unit(ffn_state, i)
            ffn_state = ffn_state + ffn_unit_output

        block_output = ffn_state
        return block_output, None, None, None

    def _forward_with_full_attn_res(
        self,
        x: Array,
        edge_cache: EdgeFeatureCache,
        radial_feat: Array,
        unit_history: list[Array] | None = None,
        comm_dict: dict[str, Array] | None = None,
    ) -> tuple[
        Array,
        Array | None,
        Array | None,
        list[Array] | None,
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
        comm_dict
            Border-exchange tensors for parallel inference, forwarded to the
            SO(2) unit. The attention-residual aggregation is per-node, so the
            ghost exchange at the convolution input restores ghost correctness
            even when the history sources carry stale ghost rows.

        Returns
        -------
        tuple[Array, Array | None, Array | None, list[Array] | None]
            Tuple `(block_output, None, so2_unit_output, ffn_unit_outputs)`.
        """
        so2_input = self.full_attn_res_so2(
            sources=unit_history,
            scalar_extractor=self._extract_l0_from_canonical,
            current_x=x,
        )
        so2_unit_output = self._run_so2_unit(
            so2_input, edge_cache, radial_feat, comm_dict
        )

        completed_units = [*unit_history, so2_unit_output]
        current_x = so2_unit_output
        ffn_unit_outputs: list[Array] = []
        for i in range(self.ffn_blocks):
            ffn_input: Array = self.full_attn_res_ffns[i](
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
        x: Array,
        edge_cache: EdgeFeatureCache,
        radial_feat: Array,
        unit_history: list[Array] | None = None,
        comm_dict: dict[str, Array] | None = None,
    ) -> tuple[
        Array,
        Array | None,
        Array | None,
        list[Array] | None,
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
        comm_dict
            Border-exchange tensors for parallel inference, forwarded to the
            SO(2) unit. The attention-residual aggregation is per-node, so the
            ghost exchange at the convolution input restores ghost correctness
            even when the history sources carry stale ghost rows.

        Returns
        -------
        tuple[Array, Array | None, Array | None, list[Array] | None]
            Tuple `(block_output, block_summary, None, None)`.
        """
        so2_input = self.block_attn_res_so2(
            sources=unit_history,
            scalar_extractor=self._extract_l0_from_canonical,
            current_x=x,
        )
        so2_unit_output = self._run_so2_unit(
            so2_input, edge_cache, radial_feat, comm_dict
        )

        partial_block = so2_unit_output
        current_x = so2_unit_output
        for i in range(self.ffn_blocks):
            ffn_input: Array = self.block_attn_res_ffns[i](
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

    def _variables(self) -> dict[str, np.ndarray]:
        variables: dict[str, np.ndarray] = {}
        if self.pre_so2_norm is not None:
            pre_so2_vars = self.pre_so2_norm.serialize().get("@variables", {})
            for key, value in pre_so2_vars.items():
                variables[f"pre_so2_norm.{key}"] = value
        if self.post_so2_norm is not None:
            post_so2_vars = self.post_so2_norm.serialize().get("@variables", {})
            for key, value in post_so2_vars.items():
                variables[f"post_so2_norm.{key}"] = value
        for key, value in self.so2_conv.serialize()["@variables"].items():
            variables[f"so2_conv.{key}"] = value
        for i, ffn in enumerate(self.ffns):
            for key, value in ffn.serialize()["@variables"].items():
                variables[f"ffns.{i}.{key}"] = value
        for i, norm in enumerate(self.pre_ffn_norms):
            if norm is not None:
                for key, value in norm.serialize().get("@variables", {}).items():
                    variables[f"pre_ffn_norms.{i}.{key}"] = value
        for i, norm in enumerate(self.post_ffn_norms):
            if norm is not None:
                for key, value in norm.serialize().get("@variables", {}).items():
                    variables[f"post_ffn_norms.{i}.{key}"] = value
        if self.adam_ffn_layer_scales is not None:
            for i, scale in enumerate(self.adam_ffn_layer_scales):
                variables[f"adam_ffn_layer_scales.{i}"] = to_numpy_array(scale)
        if self.full_attn_res_so2 is not None:
            for key, value in self.full_attn_res_so2.serialize()["@variables"].items():
                variables[f"full_attn_res_so2.{key}"] = value
        if self.full_attn_res_ffns is not None:
            for i, attn in enumerate(self.full_attn_res_ffns):
                for key, value in attn.serialize()["@variables"].items():
                    variables[f"full_attn_res_ffns.{i}.{key}"] = value
        if self.block_attn_res_so2 is not None:
            for key, value in self.block_attn_res_so2.serialize()["@variables"].items():
                variables[f"block_attn_res_so2.{key}"] = value
        if self.block_attn_res_ffns is not None:
            for i, attn in enumerate(self.block_attn_res_ffns):
                for key, value in attn.serialize()["@variables"].items():
                    variables[f"block_attn_res_ffns.{i}.{key}"] = value
        return variables

    def _load_variables(self, variables: dict[str, np.ndarray]) -> None:
        def load(module: NativeOP, prefix: str) -> NativeOP:
            data = module.serialize()
            data["@variables"] = {
                key[len(prefix) :]: value
                for key, value in variables.items()
                if key.startswith(prefix)
            }
            return type(module).deserialize(data)

        if self.pre_so2_norm is not None:
            self.pre_so2_norm = load(self.pre_so2_norm, "pre_so2_norm.")
        if self.post_so2_norm is not None:
            self.post_so2_norm = load(self.post_so2_norm, "post_so2_norm.")
        self.so2_conv = load(self.so2_conv, "so2_conv.")
        self.ffns = [load(ffn, f"ffns.{i}.") for i, ffn in enumerate(self.ffns)]
        self.pre_ffn_norms = [
            load(norm, f"pre_ffn_norms.{i}.") if norm is not None else None
            for i, norm in enumerate(self.pre_ffn_norms)
        ]
        self.post_ffn_norms = [
            load(norm, f"post_ffn_norms.{i}.") if norm is not None else None
            for i, norm in enumerate(self.post_ffn_norms)
        ]
        if self.adam_ffn_layer_scales is not None:
            self.adam_ffn_layer_scales = [
                np.asarray(
                    variables[f"adam_ffn_layer_scales.{i}"],
                    dtype=PRECISION_DICT[self.precision],
                )
                for i in range(len(self.adam_ffn_layer_scales))
            ]
        if self.full_attn_res_so2 is not None:
            self.full_attn_res_so2 = load(self.full_attn_res_so2, "full_attn_res_so2.")
        if self.full_attn_res_ffns is not None:
            self.full_attn_res_ffns = [
                load(attn, f"full_attn_res_ffns.{i}.")
                for i, attn in enumerate(self.full_attn_res_ffns)
            ]
        if self.block_attn_res_so2 is not None:
            self.block_attn_res_so2 = load(
                self.block_attn_res_so2, "block_attn_res_so2."
            )
        if self.block_attn_res_ffns is not None:
            self.block_attn_res_ffns = [
                load(attn, f"block_attn_res_ffns.{i}.")
                for i, attn in enumerate(self.block_attn_res_ffns)
            ]

    def serialize(self) -> dict[str, Any]:
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
                "mixing_layers": self.mixing_layers,
                "so2_attn_res": self.so2_attn_res_mode,
                "radial_so2_mode": self.radial_so2_mode,
                "radial_so2_rank": self.radial_so2_rank,
                "edge_cartesian": self.edge_cartesian,
                "node_cartesian": self.node_cartesian,
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
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "SeZMInteractionBlock":
            raise ValueError(f"Invalid class for SeZMInteractionBlock: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(**config)
        obj._load_variables(variables)
        return obj
