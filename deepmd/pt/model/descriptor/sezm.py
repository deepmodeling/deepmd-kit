# SPDX-License-Identifier: LGPL-3.0-or-later
"""
SeZM descriptor: Smooth Equivariant Zone-bridging Model.

PyTorch backend

This implementation is designed around two goals:

1) Conservative forces: the descriptor is computed from differentiable energy.
2) Efficient inference: edge geometry and Wigner-D rotation blocks are computed
   exactly once per `forward()` and reused by all interaction blocks.

Shared descriptor building blocks are re-exported by `sezm_nn/__init__.py`.

Runtime flow at a glance:
1) Build edge cache and radial features once.
2) Run interaction blocks with shared geometric caches.
3) Return scalar (`l=0`) descriptor channels for fitting.

Layout notes
------------
- Node-level backbone features use contiguous `(N, D_node, 1, C)` where
  `D_node=(l_schedule[i]+extra_node_l+1)^2` and `C=channels`.
- The singleton focus axis is kept only to reuse the existing equivariant
  operators; real multi-focus structure lives strictly inside `SO2Convolution`.
- Edge-level SO(2) internal operators keep m-major reduced layout
  `(E, F, D_m_trunc, Cf)` with `F=n_focus` and `Cf=focus_dim` inside the
  SO(2) branch only.
"""

from __future__ import (
    annotations,
)

import math
from contextlib import (
    contextmanager,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

import torch
import torch.nn as nn
from einops import (
    rearrange,
)

from deepmd.dpmodel.utils import EnvMat as DPEnvMat
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
from deepmd.pt.utils.exclude_mask import (
    PairExcludeMask,
)
from deepmd.pt.utils.update_sel import (
    UpdateSel,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .base_descriptor import (
    BaseDescriptor,
)
from .sezm_nn import (
    ATTN_RES_MODES,
    BridgingSwitch,
    C3CutoffEnvelope,
    ChargeSpinEmbedding,
    DepthAttnRes,
    EdgeFeatureCache,
    EnvironmentInitialEmbedding,
    EquivariantFFN,
    GeometricInitialEmbedding,
    InnerClamp,
    RadialBasis,
    RadialMLP,
    ScalarRMSNorm,
    SeZMInteractionBlock,
    SeZMTypeEmbedding,
    WignerDCalculator,
    build_edge_cache,
    build_edge_cache_from_edges,
    build_edge_quaternion,
    edge_cache_to_dtype,
    fold_lora_state_dict_keys,
    get_promoted_dtype,
    get_so3_dim_of_lmax,
    has_lora,
    np_safe,
    nvtx_range,
    safe_norm,
    safe_numpy_to_tensor,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Generator,
    )

    from deepmd.utils.data_system import (
        DeepmdDataSystem,
    )
    from deepmd.utils.path import (
        DPPath,
    )


@BaseDescriptor.register("SeZM")
@BaseDescriptor.register("sezm")
@BaseDescriptor.register("DPA4")
@BaseDescriptor.register("dpa4")
class DescrptSeZM(BaseDescriptor, nn.Module):
    """
    SeZM descriptor.

    Execution outline
    -----------------
    1. Build a per-forward `EdgeFeatureCache` (geometry, envelope, Wigner-D).
    2. Build radial/type edge features once and reuse across blocks.
    3. Run `SeZMInteractionBlock` stack with optional l/m schedules.
    4. Extract scalar channels and apply the final scalar FFN.

    Parameters
    ----------
    ntypes
        Number of element types.
    sel
        Maximum number of neighbors per type within `rcut`.
        - int: broadcast to all types, e.g. sel=100 with ntypes=2 → [100, 100]
        - list[int]: sel[i] is the maximum number of type i atoms within `rcut`
    rcut
        Cutoff radius in Å.
    env_exp
        C^3 cutoff envelope exponents `[rbf_env_exp, edge_env_exp]`.
        - `rbf_env_exp`: Controls radial basis function envelope decay.
        - `edge_env_exp`: Controls message passing edge weight envelope decay.
        Larger values give weaker suppression (values stay near 1.0 longer).
    channels
        Total channels per (l,m) coefficient.
    basis_type
        Radial basis type. Supported values are ``"bessel"`` and ``"gaussian"``.
    n_radial
        Number of radial basis functions.
    radial_mlp
        Hidden layer sizes for radial networks. An output layer of size
        `(l_schedule[0]+extra_node_l+1)*channels` will be automatically appended.
    use_env_seed
        If True, seed the initial node state with local-environment information:
        apply environment matrix FiLM conditioning on l=0 features using 4D
        `[s, s*r_hat]` representation, and enable the non-scalar geometric
        initial embedding when `l_schedule[0] + extra_node_l > 0`. If False, the initial state
        contains only atom-local scalar features before message passing. FiLM
        deltas are normalized and scaled with learnable strengths initialized
        to small values. Internal dimensions are derived from `channels`:
        `embed_dim=min(channels, 128)`,
        `axis_dim=min(4 if embed_dim < 64 else 8, embed_dim-1)`,
        `type_dim=clamp(channels//4, 8, 32)`,
        `rbf_out_dim=max(32, embed_dim-2*type_dim)`,
        `hidden_dim=min(256, max(2*embed_dim, rbf_out_dim+2*type_dim))`.
    random_gamma
        If True, apply a random roll about the edge-aligned local ``+Z`` axis
        before building the Wigner-D blocks. The roll is sampled independently
        per edge and per forward call.
    lmax
        Maximum degree, only used when `l_schedule` is None.
    l_schedule
        Pyramid schedule of lmax per block, e.g. [3, 3, 2]. Must be non-increasing.
        If set, lmax and n_blocks will be ignored.
    mmax
        Maximum SO(2) order (|m|), only used when `m_schedule` is None.
        If None, defaults to the per-block `lmax` (i.e. `m_schedule = l_schedule`).
    kmax
        Maximum Wigner-D frame order (|k|) used by SO(3) grid nets. The frame set
        is built as ``[0, -1, 1, ..., -kmax, kmax]``. ``kmax=0`` recovers the
        S2-like k=0 slice, while ``kmax=1`` is the default low-cost setting that
        opens odd/antisymmetric coupling paths.
    m_schedule
        Schedule of mmax per block, e.g. [2, 2, 1, 0]. Must satisfy
        `m_schedule[i] <= l_schedule[i]` for every block. A non-increasing schedule is
        recommended but not required. If set, `mmax` will be ignored.
    extra_node_l
        Extra node representation degree above each message-passing degree.
        The node degree of block `i` is `l_schedule[i] + extra_node_l`, while
        SO(2) message passing still uses `l_schedule[i]`.
    n_blocks
        Number of blocks (only used when `l_schedule` is None).
    so2_norm
        If True, apply intermediate ReducedEquivariantRMSNorm between SO(2) mixing layers.
        When False (default), no normalization is applied between layers.
    so2_layers
        Number of SO(2) mixing layers per block.
    so2_attn_res
        SO(2)-internal depth-wise attention residual mode inside each interaction
        block. Must be one of ``"none"``, ``"independent"``, or ``"dependent"``.
    radial_so2_mode
        Dynamic radial degree mixer mode inside SO(2) convolution. ``"none"``
        applies elementwise radial modulation, ``"degree"`` uses a
        channel-shared edge-conditioned cross-degree kernel, and
        ``"degree_channel"`` uses a per-channel cross-degree kernel.
    radial_so2_rank
        Low-rank channel factorization rank for
        ``radial_so2_mode="degree_channel"``. ``0`` uses the full
        per-channel dynamic degree kernel.
    n_focus
        Number of parallel focus streams used only inside the SO(2) convolution.
        Node-level backbone tensors still keep a singleton focus axis.
    focus_dim
        Hidden width per focus stream inside the SO(2) convolution.
        ``focus_dim=0`` means using ``channels``.
    n_atten_head
        Number of attention heads when aggregating messages in SO(2) convolution.
        0 applies a plain envelope-weighted scatter-sum; >0 enables
        envelope-gated grouped softmax attention with output-side head gate.
        Attention uses ``w**2 * exp(logit)`` in the numerator and
        ``zeta + sum(w**2 * exp(logit))`` in the denominator.
    atten_f_mix
        If True, merge all SO(2) focus streams into one attention stream after
        rotate-back. Attention heads split ``n_focus * focus_dim`` instead of
        each focus stream independently.
    atten_v_proj
        If True, apply an explicit degree-aware value projection inside SO(2)
        attention.
    atten_o_proj
        If True, apply an explicit degree-aware output projection inside SO(2)
        attention.
    ffn_neurons
        Hidden width for block FFNs and the final scalar output FFN.
        If ``>0``, both paths use this width.
        If ``=0``, each path resolves its own width from ``channels`` and its
        effective GLU setting: ``4 * channels`` without GLU, ``(8 / 3) * channels``
        with GLU, then round up to a multiple of 32.
    grid_mlp
        Either one boolean applied to every grid path, or three booleans
        ``[node_wise, message_node, ffn]`` selecting the polynomial point-wise
        grid MLP operation per grid path. On any path whose ``grid_branch``
        entry is positive it is overridden by branch mixing, and it has no
        effect on the final ``l=0`` output head.
    grid_branch
        Either one non-negative integer applied to every grid path, or three
        integers ``[node_wise, message_node, ffn]`` setting the number of
        scalar-routed polynomial product branches per grid path. ``0`` disables
        branch mixing on that path; positive values select branch mixing and
        take precedence over ``grid_mlp``. Branch weights are computed from
        ``l=0`` scalar features only, while each branch is a quadratic product
        of channel-mixed grid fields. The ``node_wise`` and ``message_node``
        entries control the SO(2) convolution cross-grid paths, and the ``ffn``
        entry controls the block-internal FFN grid path.
    ffn_blocks
        Number of FFN subblocks per interaction block.
    sandwich_norm
        Pre/post-norm switches for [SO(2), FFN] residual branches in order:
        [so2_pre, so2_post, ffn_pre, ffn_post], shared across all blocks.
    mlp_bias
        Whether to use bias in equivariant layers. When False, removes bias from:
        - SO3Linear: l=0 bias
        - SO2Linear: l=0 bias
        - GatedActivation: gate linear bias
        - DepthAttnRes: input-dependent query projection
        - EnvironmentInitialEmbedding:
          rbf_proj_layer1/2 and g_layer1/2
        Attention logit and output-gate parameters in SO(2) convolution are
        always bias-free.
    layer_scale
        If True, apply learnable LayerScale (init 1e-3) on residual branches:
        - SO(2) branch: per-focus-channel scales `(n_focus, focus_dim)`
          on each SO(2) mixing layer.
        - FFN branch: per-channel scales `(channels,)` on each FFN subblock.
    full_attn_res
        Descriptor-level full attention residual mode over the unit history
        `[x0, so2_0, ffn_0_0, ffn_0_1, ..., so2_1, ffn_1_0, ffn_1_1, ...]`,
        where each FFN subblock contributes its own completed unit
        representation. `independent` uses learned query vectors, while
        `dependent` derives queries from the current SeZM state before the
        SO(2) unit, before each FFN unit, and before the final aggregation.
        Must be one of ``"none"``, ``"independent"``, or ``"dependent"``.
    block_attn_res
        Descriptor-level block attention residual mode over the block history
        `[x0, b1, b2, ...]`, where each `b_i` is the sum of all unit outputs
        inside one `SeZMInteractionBlock`. `independent` uses learned query
        vectors, while `dependent` derives queries from the current SeZM state
        before the SO(2) unit, before each FFN unit, and before the final block
        aggregation. Must be one of ``"none"``, ``"independent"``, or
        ``"dependent"``. Cannot be enabled together with `full_attn_res`.
    s2_activation
        Two booleans ``[so2_enabled, ffn_enabled]``.
        ``so2_enabled=True`` makes the SO(2) gated activation path use
        ``activation_function="silu"``.
        ``ffn_enabled=True`` makes the block-internal FFN path use
        ``activation_function="silu"`` and ``glu_activation=True``.
        S2-grid resolutions are resolved automatically per block. The
        tensor-product grid uses ``[2 * mmax + 4, ceil_even(3 * lmax + 2)]``
        in the SO(2) branch, and the FFN branch lifts it to a square
        ``[max(R_phi, R_theta), max(R_phi, R_theta)]`` grid. Lebedev branches
        use the smallest packaged rule with precision at least ``3 * lmax``.
        The final ``l=0`` output FFN is unchanged.
    ffn_so3_grid
        If True, use the SO(3) Wigner-D grid in the block-internal FFN. This
        option takes precedence over the FFN grid path and ignores
        ``s2_activation[1]``. The final ``l=0`` output FFN is unchanged.
    node_wise_s2
        If True, add an edge-local S2 product branch between source and
        destination node features inside the SO(2) convolution.
    node_wise_so3
        If True, use the corresponding edge-local SO(3) Wigner-D grid-net branch.
        The source side is the query and the destination side is the context.
    message_node_s2
        If True, add a post-aggregation S2 product branch between hidden messages
        and destination node features before the SO(2) output projection.
    message_node_so3
        If True, use the corresponding post-aggregation SO(3) Wigner-D grid-net
        branch. The message is the query and the node state is the context.
    so3_readout
        Read-out FFN mode for the final ``l=0`` descriptor. ``"none"`` applies a
        degree-0 scalar FFN to the ``l=0`` slice only; ``l>0`` coefficients are
        discarded before the read-out. ``"glu"`` and ``"mlp"`` apply a full
        equivariant FFN whose degree equals the node degree of the last
        interaction block, driven by the SO(3) Wigner-D grid, so ``l>0`` geometry
        is folded into ``l=0`` before the scalar is extracted. The value selects
        the quadratic grid product (``"glu"``) or the polynomial point-wise grid
        MLP (``"mlp"``). The Wigner-D frame order follows ``kmax``. The residual
        stays on the ``l=0`` channel.
    lebedev_quadrature
        Either one boolean applied to both S2 branches, or two booleans
        ``[so2_enabled, ffn_enabled]`` aligned with ``s2_activation``. If
        enabled for a branch, that branch uses packaged Lebedev quadrature
        instead of the tensor-product sphere grid in its S2 projector.
    activation_function
        Base activation function for helper MLPs, the SO(2) gated activation
        path, and the final ``l=0`` output FFN.
        It is overridden to ``"silu"`` only on paths whose ``s2_activation``
        switch is enabled.
    glu_activation
        Base GLU switch for FFN. The block-internal FFN path overrides it to
        ``True`` only when ``s2_activation[1]=True``. The final ``l=0`` output
        FFN always keeps this user-provided value.
    use_amp
        If True, use automatic mixed precision (AMP) with bfloat16 on CUDA
        during training. This can improve speed and reduce memory usage.
        Enabling this option is recommended on GPUs with native bfloat16 support.
        Disable it on GPUs without native bfloat16 support to avoid runtime
        errors or additional conversion overhead.
    exclude_types
        List of excluded type pairs.
    precision
        Precision for neural network parameters and computations. Geometry computations
        (edge distances, Wigner-D matrices, rotations, and enabled env seeds) always
        run in fp32+ to provide accurate geometric information for better convergence.
        Only the interaction blocks use this precision.
    eps
        Small epsilon for numerical stability in division and normalization.
    trainable
        Whether parameters are trainable.
    seed
        Random seed(s).
    type_map
        Type names.
    inner_clamp_r_inner
        Inner radius for distance saturation in Å. If both inner and outer radii
        are set, the descriptor freezes short-range descriptor geometry inside
        the zone-bridging window.
    inner_clamp_r_outer
        Outer radius for distance saturation in Å.
    add_chg_spin_ebd
        If True, add frame-level charge/spin condition embedding to scalar type
        features before edge features are built.
    default_chg_spin
        Default frame-level charge/spin condition `[charge, spin]`. This value is
        used when `add_chg_spin_ebd=True` and no explicit `charge_spin` tensor is
        provided at the descriptor or SeZM model boundary.

    Notes
    -----
    SeZM does not use the traditional environment matrix (r, a_x, a_y, a_z).
    Instead, it uses radial basis functions and spherical harmonics directly.
    The mean/stddev statistics are kept for interface compatibility but are not
    actively used in the forward pass.
    """

    _ENV_DIM: int = 1  # Use se_r style (radial only) for EnvMatStatSe compatibility
    LATEST_VERSION: float = 1.1

    def __init__(
        self,
        ntypes: int,
        sel: list[int] | int,
        rcut: float = 6.0,
        env_exp: list[int] | None = None,
        channels: int = 64,
        basis_type: str = "bessel",
        n_radial: int = 16,
        radial_mlp: list[int] | None = None,
        use_env_seed: bool = True,
        random_gamma: bool = True,
        lmax: int = 3,
        l_schedule: list[int] | None = None,
        mmax: int | None = 1,
        kmax: int = 1,
        m_schedule: list[int] | None = None,
        extra_node_l: int = 0,
        n_blocks: int = 3,
        so2_norm: bool = False,
        so2_layers: int = 4,
        so2_attn_res: str = "none",
        radial_so2_mode: str = "degree_channel",
        radial_so2_rank: int = 1,
        n_focus: int = 1,
        focus_dim: int = 0,
        n_atten_head: int = 1,
        atten_f_mix: bool = False,
        atten_v_proj: bool = False,
        atten_o_proj: bool = False,
        ffn_neurons: int = 0,
        grid_mlp: bool | list[bool] = False,
        grid_branch: int | list[int] = 0,
        ffn_blocks: int = 1,
        sandwich_norm: list[bool] | None = None,
        mlp_bias: bool = False,
        layer_scale: bool = False,
        full_attn_res: str = "none",
        block_attn_res: str = "none",
        s2_activation: list[bool] | None = None,
        ffn_so3_grid: bool = False,
        node_wise_s2: bool = False,
        node_wise_so3: bool = False,
        message_node_s2: bool = False,
        message_node_so3: bool = False,
        so3_readout: str = "none",
        lebedev_quadrature: bool | list[bool] | None = True,
        activation_function: str = "silu",
        glu_activation: bool = True,
        use_amp: bool = True,
        exclude_types: list[tuple[int, int]] | None = None,
        precision: str = "float32",
        eps: float = 1e-7,
        trainable: bool = True,
        seed: int | list[int] | None = None,
        type_map: list[str] | None = None,
        inner_clamp_r_inner: float | None = None,
        inner_clamp_r_outer: float | None = None,
        add_chg_spin_ebd: bool = False,
        default_chg_spin: list[float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.register_buffer(
            "version_tensor",
            torch.tensor(self.LATEST_VERSION, dtype=torch.float64, device=env.DEVICE),
            persistent=True,
        )
        self.version = float(self.version_tensor.item())
        self.rcut = float(rcut)
        if env_exp is None:
            env_exp = [7, 5]
        if len(env_exp) != 2:
            raise ValueError(
                "`env_exp` must be a list of two integers: [rbf_env_exp, edge_env_exp]"
            )
        self.env_exp = [int(x) for x in env_exp]
        self.eps = float(eps)
        # Floor for the envelope-squared degree normalization (GIE / env_seed).
        # version < 1.1 keeps the tiny ``eps`` floor (legacy path, untouched);
        # version >= 1.1 swaps in this O(1) value so sparse-neighborhood (e.g.
        # dimer) features vanish smoothly at rcut instead of saturating and
        # kinking just inside the cutoff.
        self.deg_norm_floor = 0.25

        if isinstance(sel, int):
            sel = [sel]
        self.ntypes = int(ntypes)
        self.sel = [int(x) for x in sel]
        self.type_map = type_map
        self.nnei = int(sum(self.sel))
        self.ndescrpt = int(self.nnei * self._ENV_DIM)

        self.channels = int(channels)
        self.n_focus = int(n_focus)
        if self.n_focus < 1:
            raise ValueError("`n_focus` must be >= 1")
        self.focus_dim = int(focus_dim)
        if self.focus_dim < 0:
            raise ValueError("`focus_dim` must be >= 0")
        self.basis_type = str(basis_type).lower()
        self.n_radial = int(n_radial)
        if radial_mlp is None:
            radial_mlp = [0]
        self.radial_mlp = [self.channels if x == 0 else int(x) for x in radial_mlp]
        if sandwich_norm is None:
            sandwich_norm = [False, True, True, False]
        if not isinstance(sandwich_norm, (list, tuple)) or len(sandwich_norm) != 4:
            raise ValueError(
                "sandwich_norm must be a list[bool] of length 4: [so2_pre, so2_post, ffn_pre, ffn_post]"
            )
        self.sandwich_norm = [bool(x) for x in sandwich_norm]
        self.so2_pre_norm = self.sandwich_norm[0]
        self.so2_post_norm = self.sandwich_norm[1]
        self.ffn_pre_norm = self.sandwich_norm[2]
        self.ffn_post_norm = self.sandwich_norm[3]
        if s2_activation is None:
            s2_activation = [False, True]
        if not isinstance(s2_activation, list) or len(s2_activation) != 2:
            raise ValueError(
                "`s2_activation` must be a list[bool] of length 2: [so2_activation, ffn_activation]"
            )
        if any(not isinstance(flag, bool) for flag in s2_activation):
            raise ValueError(
                "`s2_activation` must be a list[bool] of length 2: [so2_activation, ffn_activation]"
            )
        self.s2_activation = list(s2_activation)
        self.ffn_so3_grid = bool(ffn_so3_grid)
        self.node_wise_s2 = bool(node_wise_s2)
        self.node_wise_so3 = bool(node_wise_so3)
        self.message_node_s2 = bool(message_node_s2)
        self.message_node_so3 = bool(message_node_so3)
        self.so3_readout = str(so3_readout).lower()
        if self.so3_readout not in {"none", "glu", "mlp"}:
            raise ValueError("`so3_readout` must be one of 'none', 'glu', or 'mlp'")
        if lebedev_quadrature is None:
            lebedev_quadrature = [True, True]
        elif isinstance(lebedev_quadrature, bool):
            lebedev_quadrature = [lebedev_quadrature, lebedev_quadrature]
        if not isinstance(lebedev_quadrature, list) or len(lebedev_quadrature) != 2:
            raise ValueError(
                "`lebedev_quadrature` must be a bool or a list[bool] of length 2: [so2_quadrature, ffn_quadrature]"
            )
        if any(not isinstance(flag, bool) for flag in lebedev_quadrature):
            raise ValueError(
                "`lebedev_quadrature` must be a bool or a list[bool] of length 2: [so2_quadrature, ffn_quadrature]"
            )
        self.lebedev_quadrature = list(lebedev_quadrature)
        self.activation_function = str(activation_function)
        self.glu_activation = bool(glu_activation)

        # === Split effective activation config by branch ===
        self.so2_s2_activation = self.s2_activation[0]
        self.ffn_s2_activation = False if self.ffn_so3_grid else self.s2_activation[1]
        self.so2_lebedev_quadrature = self.lebedev_quadrature[0]
        self.ffn_lebedev_quadrature = self.lebedev_quadrature[1]
        self.so2_activation_function = (
            "silu" if self.so2_s2_activation else self.activation_function
        )
        self.ffn_activation_function = (
            "silu" if self.ffn_s2_activation else self.activation_function
        )
        self.ffn_glu_activation = (
            True
            if (self.ffn_s2_activation or self.ffn_so3_grid)
            else self.glu_activation
        )
        self.out_activation_function = self.activation_function
        self.out_glu_activation = self.glu_activation
        self.precision = str(precision)
        self.dtype = PRECISION_DICT[self.precision]
        self.device = env.DEVICE
        self.compute_dtype = get_promoted_dtype(self.dtype)
        self.mlp_bias = bool(mlp_bias)
        self.layer_scale = bool(layer_scale)
        self.use_amp = bool(use_amp)  # and self.training
        self.trainable = bool(trainable)
        self.seed = seed
        self.random_gamma = bool(random_gamma)
        self.add_chg_spin_ebd = bool(add_chg_spin_ebd)
        if default_chg_spin is not None and len(default_chg_spin) != 2:
            raise ValueError("`default_chg_spin` must contain [charge, spin].")
        self.default_chg_spin = (
            None if default_chg_spin is None else [float(x) for x in default_chg_spin]
        )

        # === Zone bridging: InnerClamp + Source Freeze Propagation Gate ===
        # Both the geometry clamp (``InnerClamp``) and the message-passing
        # switch (``BridgingSwitch``) are activated together on the same
        # ``[r_inner, r_outer]`` window. The clamp freezes scalar distance
        # on every ``(j, k)`` edge with ``r_{jk} < r_inner``; the switch
        # feeds a per-edge C3 amplitude into ``compute_edge_src_gate`` so
        # that any node with a frozen neighbor cannot propagate
        # information through the GNN, closing the direction / multi-hop
        # leakage channels that a pure ``InnerClamp`` cannot reach. Both
        # modules are parameter-free, so enabling bridging does not add
        # any keys to the descriptor's state dict.
        self.inner_clamp_r_inner = (
            float(inner_clamp_r_inner) if inner_clamp_r_inner is not None else None
        )
        self.inner_clamp_r_outer = (
            float(inner_clamp_r_outer) if inner_clamp_r_outer is not None else None
        )
        if (
            self.inner_clamp_r_inner is not None
            and self.inner_clamp_r_outer is not None
        ):
            self.inner_clamp: InnerClamp | None = InnerClamp(
                self.inner_clamp_r_inner, self.inner_clamp_r_outer
            )
            self.bridging_switch: BridgingSwitch | None = BridgingSwitch(
                self.inner_clamp_r_inner, self.inner_clamp_r_outer
            )
        else:
            self.inner_clamp = None
            self.bridging_switch = None

        # === Env seed parameters ===
        self.use_env_seed = bool(use_env_seed)
        self.env_seed_embed_dim = min(self.channels, 128)
        self.env_seed_type_dim = min(32, max(8, self.channels // 4))
        axis_dim = 4 if self.env_seed_embed_dim < 64 else 8
        self.env_seed_axis_dim = min(axis_dim, max(1, self.env_seed_embed_dim - 1))
        rbf_out_dim = max(32, self.env_seed_embed_dim - 2 * self.env_seed_type_dim)
        g_in_dim = rbf_out_dim + 2 * self.env_seed_type_dim
        self.env_seed_hidden_dim = min(256, max(2 * self.env_seed_embed_dim, g_in_dim))

        # === Split deterministic seeds at the descriptor top-level ===
        seed_type_embedding = child_seed(self.seed, 0)
        seed_blocks = child_seed(self.seed, 1)
        seed_out = child_seed(self.seed, 2)
        seed_radial_embedding = child_seed(self.seed, 3)
        seed_env_seed = child_seed(self.seed, 4)
        seed_full_attn = child_seed(self.seed, 5)
        seed_block_attn = child_seed(self.seed, 6)
        seed_charge_spin = child_seed(self.seed, 7)

        # === L/M schedules ===
        self._init_lm_schedules(lmax, n_blocks, l_schedule, mmax, m_schedule)
        self.kmax = int(kmax)
        if self.kmax < 0:
            raise ValueError("`kmax` must be non-negative")
        if self.kmax > self.lmax:
            raise ValueError("`kmax` must be <= `lmax`")
        self.ebed_dims = [get_so3_dim_of_lmax(l) for l in self.l_schedule]
        self._init_node_l_schedules(extra_node_l)
        self.rad_sizes_per_block = [l + 1 for l in self.l_schedule]

        self.so2_norm = bool(so2_norm)
        self.so2_layers = int(so2_layers)
        self.so2_attn_res_mode = str(so2_attn_res).lower()
        if self.so2_attn_res_mode not in ATTN_RES_MODES:
            raise ValueError(
                "`so2_attn_res` must be one of 'none', 'independent', or 'dependent'"
            )
        self.radial_so2_mode = str(radial_so2_mode).lower()
        if self.radial_so2_mode not in {"none", "degree", "degree_channel"}:
            raise ValueError(
                "`radial_so2_mode` must be one of 'none', 'degree', or 'degree_channel'"
            )
        self.radial_so2_rank = int(radial_so2_rank)
        if self.radial_so2_rank < 0:
            raise ValueError("`radial_so2_rank` must be non-negative")
        self.ffn_neurons = int(ffn_neurons)
        self.block_ffn_neurons = self._resolve_ffn_neurons(
            self.ffn_neurons,
            glu_activation=self.ffn_glu_activation,
        )
        self.out_ffn_neurons = self._resolve_ffn_neurons(
            self.ffn_neurons,
            glu_activation=self.out_glu_activation,
        )
        self.grid_mlp = self._broadcast_grid_setting(
            grid_mlp,
            name="grid_mlp",
            cast=bool,
        )
        self.grid_branch = self._broadcast_grid_setting(
            grid_branch,
            name="grid_branch",
            cast=int,
            non_negative=True,
        )
        (
            self.node_wise_grid_mlp,
            self.message_node_grid_mlp,
            self.ffn_grid_mlp,
        ) = self.grid_mlp
        (
            self.node_wise_grid_branch,
            self.message_node_grid_branch,
            self.ffn_grid_branch,
        ) = self.grid_branch
        self.ffn_blocks = int(ffn_blocks)
        if self.ffn_blocks < 1:
            raise ValueError("`ffn_blocks` must be >= 1")
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
        self.n_atten_head = int(n_atten_head)
        self.atten_f_mix = bool(atten_f_mix)
        self.use_atten_v_proj = bool(atten_v_proj)
        self.use_atten_o_proj = bool(atten_o_proj)
        so2_focus_dim = self.channels if self.focus_dim == 0 else self.focus_dim
        attn_focus_dim = (
            self.n_focus * so2_focus_dim if self.atten_f_mix else so2_focus_dim
        )
        if self.n_atten_head > 0 and attn_focus_dim % self.n_atten_head != 0:
            raise ValueError(
                "`n_atten_head` must divide the attention width "
                "(`focus_dim` or `n_focus * focus_dim` when `atten_f_mix=True`)"
            )

        # === Excluded type pairs ===
        self.reinit_exclude(exclude_types)

        # === Type embedding ===
        self.type_embedding = SeZMTypeEmbedding(
            ntypes=self.ntypes,
            embed_dim=self.channels,
            dtype=self.compute_dtype,  # force fp32+
            seed=seed_type_embedding,
            trainable=self.trainable,
        )
        if self.add_chg_spin_ebd:
            self.charge_spin_embedding: ChargeSpinEmbedding | None = (
                ChargeSpinEmbedding(
                    embed_dim=self.channels,
                    activation_function=self.activation_function,
                    dtype=self.compute_dtype,
                    seed=seed_charge_spin,
                    trainable=self.trainable,
                )
            )
        else:
            self.charge_spin_embedding = None

        # === Env FiLM embedding (optional) ===
        if self.use_env_seed:
            self.env_seed_embedding: EnvironmentInitialEmbedding | None = (
                EnvironmentInitialEmbedding(
                    ntypes=self.ntypes,
                    n_radial=self.n_radial,
                    channels=self.channels,
                    embed_dim=self.env_seed_embed_dim,
                    axis_dim=self.env_seed_axis_dim,
                    type_dim=self.env_seed_type_dim,
                    hidden_dim=self.env_seed_hidden_dim,
                    mlp_bias=self.mlp_bias,
                    activation_function=self.activation_function,
                    eps=self.eps,
                    dtype=self.compute_dtype,  # force fp32+
                    trainable=self.trainable,
                    seed=seed_env_seed,
                )
            )
            self.film_scale_norm = ScalarRMSNorm(
                channels=self.channels,
                n_focus=1,
                eps=self.eps,
                dtype=self.compute_dtype,
                trainable=self.trainable,
            )
            self.film_shift_norm = ScalarRMSNorm(
                channels=self.channels,
                n_focus=1,
                eps=self.eps,
                dtype=self.compute_dtype,
                trainable=self.trainable,
            )
            film_strength_init = 0.01
            # Use 1D tensor (not scalar) for FSDP2 compatibility
            self.film_scale_strength_log = nn.Parameter(
                torch.full(
                    (1,),
                    math.log(film_strength_init),
                    dtype=self.compute_dtype,
                    device=self.device,
                ),
                requires_grad=self.trainable,
            )
            self.film_shift_strength_log = nn.Parameter(
                torch.full(
                    (1,),
                    math.log(film_strength_init),
                    dtype=self.compute_dtype,
                    device=self.device,
                ),
                requires_grad=self.trainable,
            )
        else:
            self.env_seed_embedding = None
            self.film_scale_norm = None
            self.film_shift_norm = None
            self.film_scale_strength_log = None
            self.film_shift_strength_log = None

        self.radial_basis = RadialBasis(
            rcut=self.rcut,
            basis_type=self.basis_type,
            n_radial=self.n_radial,
            dtype=self.compute_dtype,  # force fp32+
            exponent=self.env_exp[0],
        )

        # === Shared radial embedding: RBF -> per-l radial features ===
        # Output dimension follows the first node degree, directly usable by
        # GIE and truncated for each SO2Conv block.
        # radial_mlp specifies hidden layer sizes; input/output layers are prepended/appended.
        # Use fp32+ precision (same as RBF output) for numerical stability.
        radial_out_dim = (self.node_l_schedule[0] + 1) * self.channels
        radial_mlp_layers = [self.n_radial, *self.radial_mlp, radial_out_dim]
        self.radial_embedding = RadialMLP(
            radial_mlp_layers,
            activation_function=self.activation_function,
            dtype=self.compute_dtype,  # force fp32+
            trainable=self.trainable,
            seed=seed_radial_embedding,
        )

        # === C^3 cutoff envelope for edge weight ===
        self.edge_envelope = C3CutoffEnvelope(rcut=self.rcut, exponent=self.env_exp[1])

        wigner_lmax = self.l_schedule[0]
        # force fp32+
        self.wigner_calc = WignerDCalculator(
            lmax=wigner_lmax,
            eps=self.eps,
            dtype=self.compute_dtype,
        )

        self.use_gie = self.use_env_seed and self.node_l_schedule[0] > 0
        if self.use_gie:
            self.gie = GeometricInitialEmbedding(
                lmax=self.node_l_schedule[0],
                channels=self.channels,
                dtype=self.compute_dtype,  # force fp32+
            )
            if self.extra_node_l > 0:
                self.gie_zonal_wigner_calc: WignerDCalculator | None = (
                    WignerDCalculator(
                        lmax=self.node_l_schedule[0],
                        eps=self.eps,
                        dtype=self.compute_dtype,
                    )
                )
            else:
                self.gie_zonal_wigner_calc = None
        else:
            self.gie = None
            self.gie_zonal_wigner_calc = None

        blocks: list[SeZMInteractionBlock] = []
        for block_idx, (l_b, node_l_b, m_b) in enumerate(
            zip(
                self.l_schedule,
                self.node_l_schedule,
                self.m_schedule,
                strict=True,
            )
        ):
            k_b = min(self.kmax, l_b)
            blocks.append(
                SeZMInteractionBlock(
                    lmax=l_b,
                    node_lmax=node_l_b,
                    mmax=m_b,
                    channels=self.channels,
                    n_focus=self.n_focus,
                    focus_dim=self.focus_dim,
                    so2_norm=self.so2_norm,
                    so2_layers=self.so2_layers,
                    so2_attn_res=self.so2_attn_res_mode,
                    radial_so2_mode=self.radial_so2_mode,
                    radial_so2_rank=self.radial_so2_rank,
                    ffn_neurons=self.block_ffn_neurons,
                    node_wise_grid_mlp=self.node_wise_grid_mlp,
                    node_wise_grid_branch=self.node_wise_grid_branch,
                    message_node_grid_mlp=self.message_node_grid_mlp,
                    message_node_grid_branch=self.message_node_grid_branch,
                    ffn_grid_mlp=self.ffn_grid_mlp,
                    ffn_grid_branch=self.ffn_grid_branch,
                    ffn_blocks=self.ffn_blocks,
                    layer_scale=self.layer_scale,
                    full_attn_res=self.full_attn_res_mode,
                    block_attn_res=self.block_attn_res_mode,
                    so2_s2_activation=self.so2_s2_activation,
                    node_wise_s2=self.node_wise_s2,
                    node_wise_so3=self.node_wise_so3,
                    message_node_s2=self.message_node_s2,
                    message_node_so3=self.message_node_so3,
                    ffn_s2_activation=self.ffn_s2_activation,
                    ffn_so3_grid=self.ffn_so3_grid,
                    kmax=k_b,
                    so2_lebedev_quadrature=self.so2_lebedev_quadrature,
                    ffn_lebedev_quadrature=self.ffn_lebedev_quadrature,
                    n_atten_head=self.n_atten_head,
                    atten_f_mix=self.atten_f_mix,
                    atten_v_proj=self.use_atten_v_proj,
                    atten_o_proj=self.use_atten_o_proj,
                    so2_pre_norm=self.so2_pre_norm,
                    so2_post_norm=self.so2_post_norm,
                    so2_activation_function=self.so2_activation_function,
                    ffn_pre_norm=self.ffn_pre_norm,
                    ffn_post_norm=self.ffn_post_norm,
                    ffn_activation_function=self.ffn_activation_function,
                    ffn_glu_activation=self.ffn_glu_activation,
                    mlp_bias=self.mlp_bias,
                    eps=self.eps,
                    dtype=self.dtype,
                    seed=child_seed(seed_blocks, block_idx),
                    trainable=self.trainable,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        # === Optional descriptor-level attention residuals ===
        self.final_block_attn_res = None
        if self.use_full_attn_res:
            self.final_full_attn_res: DepthAttnRes | None = DepthAttnRes(
                channels=self.channels,
                input_dependent=self.full_attn_res_mode == "dependent",
                eps=self.eps,
                bias=self.mlp_bias,
                dtype=self.compute_dtype,
                trainable=self.trainable,
                seed=child_seed(seed_full_attn, 2000),
            )
        else:
            self.final_full_attn_res = None
        if self.use_block_attn_res:
            self.final_block_attn_res: DepthAttnRes | None = DepthAttnRes(
                channels=self.channels,
                input_dependent=self.block_attn_res_mode == "dependent",
                eps=self.eps,
                bias=self.mlp_bias,
                dtype=self.compute_dtype,
                trainable=self.trainable,
                seed=child_seed(seed_block_attn, 2000),
            )

        # === Final FFN for l=0 output mixing ===
        # ``so3_readout="none"`` runs a degree-0 scalar FFN on the l=0 slice.
        # ``"glu"``/``"mlp"`` run a full FFN at the last block's node degree whose
        # SO(3) Wigner-D grid folds l>0 geometry into l=0; the value selects the
        # quadratic grid product or the point-wise grid MLP.
        readout_lmax = self.node_l_schedule[-1]
        self.output_ffn = EquivariantFFN(
            lmax=0 if self.so3_readout == "none" else readout_lmax,
            channels=self.channels,
            hidden_channels=self.out_ffn_neurons,
            kmax=min(self.kmax, readout_lmax),
            grid_mlp=self.so3_readout == "mlp",
            grid_branch=0,
            dtype=self.compute_dtype,
            s2_activation=False,
            ffn_so3_grid=self.so3_readout != "none",
            activation_function=self.out_activation_function,
            glu_activation=self.out_glu_activation,
            mlp_bias=self.mlp_bias,
            trainable=self.trainable,
            seed=seed_out,
        )

        for p in self.parameters():
            p.requires_grad = self.trainable

        # Pre-allocate empty tensor for interface compatibility (torch.compile + DDP)
        self.register_buffer(
            "_empty_tensor",
            torch.empty(0, device=env.DEVICE, dtype=env.GLOBAL_PT_FLOAT_PRECISION),
            persistent=True,
        )

        # === Statistics buffers (interface compatibility) ===
        self.stats: dict[str, Any] | None = None
        self.register_buffer(
            "mean",
            torch.zeros(0, dtype=self.dtype, device=self.device),
            persistent=True,
        )
        self.register_buffer(
            "stddev",
            torch.ones(0, dtype=self.dtype, device=self.device),
            persistent=True,
        )

    def forward(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        edge_vec: torch.Tensor | None = None,
        edge_mask: torch.Tensor | None = None,
        comm_dict: dict[str, torch.Tensor] | None = None,
        fparam: torch.Tensor | None = None,
        force_embedding: torch.Tensor | None = None,
        charge_spin: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Compute the descriptor.

        Parameters
        ----------
        extended_coord
            Extended coordinates of atoms with shape (nf, nall*3) or (nf, nall, 3) in Å.
        extended_atype
            Extended atom types with shape (nf, nall).
        nlist
            Neighbor list with shape (nf, nloc, nnei).
        mapping
            Extended-to-local mapping with shape (nf, nall), or None.
        edge_index
            Fixed-shape edge indices with shape (2, E). If provided, the descriptor
            uses the edge-list path and ignores `nlist` and `mapping`.
        edge_vec
            Fixed-shape edge vectors with shape (E, 3) in Å. Required when
            `edge_index` is provided.
        edge_mask
            Fixed-shape edge mask with shape (E,). Required when `edge_index`
            is provided.
        comm_dict
            Communication dictionary for parallel inference (unused).
        fparam
            Frame parameters with shape (nf, nfp). Not used by SeZM, kept for
            interface compatibility.
        force_embedding
            Optional precomputed equivariant force embedding with shape
            ``(nf * nloc, D, 1, channels)``, where
            ``D = (node_l_schedule[0] + 1) ** 2``. This tensor is added to the
            initial SO(3) backbone state before the interaction blocks.
        charge_spin
            Frame-level charge and spin conditions with shape (nf, 2).

        Returns
        -------
        descriptor
            Descriptor with shape (nf, nloc, channels). Only l=0 is returned.
        rot_mat
            Empty tensor (not used).
        g2
            Empty tensor (not used).
        h2
            Empty tensor (not used).
        sw
            Empty tensor (not used).
        """
        if extended_coord.ndim == 2:
            extended_coord = rearrange(extended_coord, "nf (nall c) -> nf nall c", c=3)
        elif extended_coord.ndim != 3:
            raise ValueError(
                "extended_coord must have shape (nf, nall*3) or (nf, nall, 3)"
            )

        if edge_index is not None:
            nf_edge = extended_atype.shape[0]
            charge_spin = self._canonicalize_charge_spin(
                charge_spin,
                nf=nf_edge,
                dtype=extended_coord.dtype,
                device=extended_coord.device,
            )
            descriptor, _ = self.forward_with_edges(
                extended_coord=extended_coord,
                extended_atype=extended_atype,
                edge_index=edge_index,
                edge_vec=edge_vec,
                edge_mask=edge_mask,
                force_embedding=force_embedding,
                charge_spin=charge_spin,
            )
            return (
                descriptor,
                self._empty_tensor,
                self._empty_tensor,
                self._empty_tensor,
                self._empty_tensor,
            )

        # === Step 1. Setup dimensions ===
        extended_coord = extended_coord.to(self.compute_dtype)
        nf, nloc, nnei = nlist.shape
        nall = extended_coord.shape[1]
        n_nodes = nf * nloc
        charge_spin = self._canonicalize_charge_spin(
            charge_spin,
            nf=nf,
            dtype=extended_coord.dtype,
            device=extended_coord.device,
        )

        # === Step 2. Excluded type pairs ===
        if self.exclude_types:
            # (nf, nloc, nnei), True means keep.
            pair_keep_mask = self.emask(nlist, extended_atype).to(dtype=torch.bool)
        else:
            pair_keep_mask = torch.ones_like(nlist, dtype=torch.bool)

        # === Step 3. Type embedding (l=0) ===
        with nvtx_range("type_embedding"):
            atype_loc = extended_atype[:, :nloc]  # (nf, nloc)
            type_ebed = self.type_embedding(atype_loc).reshape(
                n_nodes, self.channels
            )  # (N, C)
            if self.charge_spin_embedding is not None:
                type_ebed = self._apply_charge_spin_embedding(
                    type_ebed,
                    charge_spin,
                    nf=nf,
                    nloc=nloc,
                )

        # === Step 4. Build edge cache once (geometry + RBF + Wigner-D) ===
        # Zone bridging (InnerClamp + SFPG + ZBL) is not routed through the
        # standard DeePMD path: bridging only makes physical sense when
        # paired with the ZBL energy that ``SeZMModel`` injects on the
        # sparse-edge path, so ``forward`` keeps the original
        # bridging-free aggregation semantics.
        with nvtx_range("build_edge_cache"):
            edge_cache = build_edge_cache(
                type_ebed=type_ebed,
                extended_coord=extended_coord,
                nlist=nlist,
                mapping=mapping,
                pair_keep_mask=pair_keep_mask,
                eps=self.eps,
                deg_norm_floor=(
                    self.deg_norm_floor if self.version >= 1.1 else self.eps
                ),
                edge_envelope=self.edge_envelope,
                radial_basis=self.radial_basis,
                n_radial=self.radial_basis.n_radial,
                # Random local-Z roll is a training-only augmentation;
                # the model is roll-equivariant, so inference fixes gamma.
                random_gamma=self.random_gamma and self.training,
                wigner_calc=self.wigner_calc,
            )

        ebed_dim_0 = self.node_ebed_dims[0]  # (node_lmax+1)^2
        x0 = type_ebed  # (N, C)
        x0_out = x0  # (N, C)

        # === Step 5. Compute radial features once (fp32+) ===
        # Shape: (E, (node_lmax+1)*C) -> (E, node_lmax+1, C)
        radial_feat = None
        with nvtx_range("radial_embedding"):
            if edge_cache.src.numel() > 0:
                radial_feat = rearrange(
                    self.radial_embedding(edge_cache.edge_rbf),
                    "E (L C) -> E L C",
                    L=self.node_l_schedule[0] + 1,
                    C=self.channels,
                )  # (E, lmax+1, C)
                if self.version >= 1.1:
                    radial_feat = radial_feat * edge_cache.edge_env.reshape(-1, 1, 1)

        # === Step 6. Env FiLM conditioning (optional, fp32+) ===
        with nvtx_range("env_film"):
            if self.use_env_seed and edge_cache.src.numel() > 0:
                atype_flat = atype_loc.reshape(-1)  # (N,)
                film = self.env_seed_embedding(
                    edge_cache=edge_cache,
                    atype_flat=atype_flat,
                    n_nodes=n_nodes,
                )  # (N, 2*C)
                scale_logits = film[:, : self.channels]  # (N, C)
                shift_logits = film[:, self.channels :]  # (N, C)
                scale_hat = self.film_scale_norm(scale_logits)  # (N, C)
                shift_hat = self.film_shift_norm(shift_logits)  # (N, C)
                scale_strength = torch.exp(self.film_scale_strength_log)
                shift_strength = torch.exp(self.film_shift_strength_log)
                scale = 1.0 + scale_strength * torch.tanh(scale_hat)  # (N, C)
                shift = shift_strength * torch.tanh(shift_hat)  # (N, C)
                x0_out = x0 * scale + shift

        # === Step 7. Build backbone l=0 features ===
        x = type_ebed.new_zeros(n_nodes, ebed_dim_0, 1, self.channels)  # (N, D, 1, C)
        x[:, 0, 0, :] = x0_out

        # === Step 8. Geometric Initial Embedding (fp32+) ===
        with nvtx_range("gie"):
            if self.use_gie and radial_feat is not None:
                # GIE only needs l>=1, slice radial_feat[:, 1:, :]
                zonal_coupling = self._build_gie_zonal_coupling(edge_cache)
                x = x + self.gie(
                    n_nodes=n_nodes,
                    edge_cache=edge_cache,
                    radial_feat=radial_feat[:, 1:, :],
                    zonal_coupling=zonal_coupling,
                ).unsqueeze(2)

        # === Step 9. Fuse edge type features into radial features (fp32+) ===
        with nvtx_range("radial_fuse"):
            if radial_feat is not None:
                radial_feat = radial_feat + rearrange(
                    edge_cache.edge_type_feat, "E C -> E 1 C"
                )
                radial_feat = radial_feat.to(dtype=self.dtype)
                rad_feat_per_block = [
                    radial_feat[:, :rad_len, :] for rad_len in self.rad_sizes_per_block
                ]  # list of (E, lmax+1, C)
            else:
                rad_feat_per_block = []

        # === Step 10. Convert to self.dtype and run blocks ===
        with nvtx_range("blocks"):
            x = x.to(dtype=self.dtype)  # (N, D, 1, C)
            if force_embedding is not None:
                x = x + force_embedding.to(dtype=self.dtype)
            if edge_cache.src.numel() > 0:
                edge_cache = edge_cache_to_dtype(edge_cache, self.dtype)
                with self._compute_mode_ctx(extended_coord.device):
                    x = self._forward_blocks(x, edge_cache, rad_feat_per_block)

        # === Step 11. Final l=0 output mixing ===
        # ``none`` feeds the l=0 slice only; ``glu``/``mlp`` feed the full
        # (N, D, 1, C) node tensor so the SO(3) grid folds l>0 into l=0. The
        # residual is added on the full coefficient tensor before extracting
        # l=0: slicing the summed tensor rather than the FFN output keeps the
        # saved degree-axis stride static under torch.compile dynamic shapes.
        with nvtx_range("output_ffn"):
            ffn_in = (
                x[:, 0:1, :, :]
                .reshape(n_nodes, 1, 1, self.channels)
                .to(dtype=self.compute_dtype)
                if self.so3_readout == "none"
                # truncate to the final node degree: the empty-edge path
                # skips the blocks, leaving x at node_ebed_dims[0]; output_ffn
                # is built for node_ebed_dims[-1]. No-op when blocks ran.
                else x[:, : self.node_ebed_dims[-1], :, :].to(dtype=self.compute_dtype)
            )
            x_scalar = (ffn_in + self.output_ffn(ffn_in))[:, 0:1, :, :]

        # === Step 12. Reshape to (nf, nloc, channels) and return ===
        descriptor = rearrange(
            x_scalar, "(nf nloc) 1 1 C -> nf nloc C", nf=nf, nloc=nloc
        )  # (nf, nloc, C)
        return (
            descriptor.to(dtype=env.GLOBAL_PT_FLOAT_PRECISION),
            self._empty_tensor,
            self._empty_tensor,
            self._empty_tensor,
            self._empty_tensor,
        )

    def forward_with_edges(
        self,
        *,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_mask: torch.Tensor,
        force_embedding: torch.Tensor | None = None,
        charge_spin: torch.Tensor | None = None,
        comm_dict: dict[str, torch.Tensor] | None = None,
        nloc: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the descriptor from a sparse edge list.

        Two node-set conventions share this path. In the single-domain path
        (``comm_dict`` is ``None``) the nodes are exactly the local atoms and
        ``edge_index`` source/destination both index ``[0, nf*nloc)``. In the
        parallel (LAMMPS multi-rank) path the nodes span the extended region
        (local owners followed by ghosts), ``edge_index`` indexes the extended
        atoms directly, and each interaction block refreshes ghost-node features
        from their owner ranks at the SO(2) convolution input (see
        :func:`~deepmd.pt.model.descriptor.sezm_nn.block.exchange_ghost_features`).

        Parameters
        ----------
        extended_coord
            Coordinates with shape (nf, n*3) or (nf, n, 3) in Å, where ``n`` is
            ``nloc`` in the single-domain path and ``nall`` in the parallel path.
        extended_atype
            Atom types with shape (nf, n). In the parallel path this spans the
            extended region so ghost type embeddings are available for the
            edge-type and environment-seed features.
        edge_index
            Edge indices with shape (2, E).
        edge_vec
            Edge vectors with shape (E, 3) in Å.
        edge_mask
            Edge mask with shape (E,).
        force_embedding
            Optional precomputed equivariant force embedding with shape
            ``(nf * nloc, D, 1, channels)``, where
            ``D = (node_l_schedule[0] + 1) ** 2``. This tensor is added to the
            initial SO(3) backbone state before the interaction blocks.
        charge_spin
            Frame-level charge and spin conditions with shape (nf, 2).
        comm_dict
            Border-exchange tensors for parallel inference. When provided, the
            node set spans the extended region and ghost features are exchanged
            via ``deepmd_export::border_op`` between interaction blocks.
        nloc
            Number of owned (local) atoms per frame. Required when ``comm_dict``
            is provided; the final scalar read-out is restricted to these atoms.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The scalar descriptor with shape ``(nf, nloc, channels)`` and the
            final equivariant latent with shape ``(nf * nloc, D_final, 1, channels)``.
        """
        # === Step 1. Setup dimensions ===
        # ``n_per_frame`` is the per-frame node count: ``nloc`` in the
        # single-domain path and ``nall`` in the parallel path. ``out_nloc`` is
        # the owned-atom count used for the final local read-out.
        extended_coord = extended_coord.to(self.compute_dtype)
        nf, n_per_frame = extended_atype.shape[:2]
        parallel = comm_dict is not None
        if parallel:
            # The border exchange and the owned-atom read-out assume one MPI
            # rank's single-frame extended layout (LAMMPS, the with-comm export
            # trace, and the parity tests all provide it). nf > 1 would silently
            # mix frames into wrong forces, so it is rejected outright.
            if nf != 1:
                raise ValueError("parallel `comm_dict` inference requires nf == 1")
            # Imported lazily so plain pt inference never pulls the custom-op
            # registration module onto its import path.
            from deepmd.pt_expt.utils.comm import (
                ensure_comm_registered,
            )

            ensure_comm_registered()
        out_nloc = nloc if parallel else n_per_frame
        atype_flat = extended_atype.reshape(-1)  # (N,)

        # === Step 2. Type embedding (l=0) ===
        with nvtx_range("type_embedding"):
            type_ebed = self.type_embedding(extended_atype).reshape(
                -1, self.channels
            )  # (N, C)
            if self.charge_spin_embedding is not None:
                type_ebed = self._apply_charge_spin_embedding(
                    type_ebed,
                    charge_spin,
                    nf=nf,
                    nloc=n_per_frame,
                )
            n_nodes = type_ebed.shape[0]

        # === Step 3. Build edge cache once (sparse edges) ===
        with nvtx_range("build_edge_cache"):
            edge_cache = build_edge_cache_from_edges(
                type_ebed=type_ebed,
                atype_flat=atype_flat,
                edge_index=edge_index,
                edge_vec=edge_vec,
                edge_mask=edge_mask,
                compute_dtype=self.compute_dtype,
                eps=self.eps,
                deg_norm_floor=(
                    self.deg_norm_floor if self.version >= 1.1 else self.eps
                ),
                inner_clamp=self.inner_clamp,
                bridging_switch=self.bridging_switch,
                edge_envelope=self.edge_envelope,
                radial_basis=self.radial_basis,
                has_exclude_types=bool(self.exclude_types),
                edge_type_keep_mask=self._edge_type_keep_mask,
                # Random local-Z roll is a training-only augmentation;
                # the model is roll-equivariant, so inference fixes gamma.
                random_gamma=self.random_gamma and self.training,
                wigner_calc=self.wigner_calc,
            )

        ebed_dim_0 = self.node_ebed_dims[0]  # (node_lmax+1)^2
        x0 = type_ebed  # (N, C)
        x0_out = x0  # (N, C)

        # === Step 4. Compute radial features once (fp32+) ===
        with nvtx_range("radial_embedding"):
            radial_feat_flat = self.radial_embedding(edge_cache.edge_rbf)
            radial_feat = radial_feat_flat.reshape(
                radial_feat_flat.shape[0],
                self.node_l_schedule[0] + 1,
                self.channels,
            )  # (E, lmax+1, C)
            if self.version >= 1.1:
                radial_feat = radial_feat * edge_cache.edge_env.reshape(-1, 1, 1)

        # === Step 5. Env FiLM conditioning (optional, fp32+) ===
        with nvtx_range("env_film"):
            if self.use_env_seed:
                film = self.env_seed_embedding(
                    edge_cache=edge_cache,
                    atype_flat=atype_flat,
                    n_nodes=n_nodes,
                )  # (N, 2*C)
                scale_logits = film[:, : self.channels]  # (N, C)
                shift_logits = film[:, self.channels :]  # (N, C)
                scale_hat = self.film_scale_norm(scale_logits)  # (N, C)
                shift_hat = self.film_shift_norm(shift_logits)  # (N, C)
                scale_strength = torch.exp(self.film_scale_strength_log)
                shift_strength = torch.exp(self.film_shift_strength_log)
                scale = 1.0 + scale_strength * torch.tanh(scale_hat)  # (N, C)
                shift = shift_strength * torch.tanh(shift_hat)  # (N, C)
                x0_out = x0 * scale + shift

        # === Step 6. Build backbone l=0 features ===
        x = type_ebed.new_zeros(n_nodes, ebed_dim_0, 1, self.channels)  # (N, D, 1, C)
        x[:, 0, 0, :] = x0_out

        # === Step 7. Geometric Initial Embedding (fp32+) ===
        with nvtx_range("gie"):
            if self.use_gie:
                zonal_coupling = self._build_gie_zonal_coupling(edge_cache)
                x = x + self.gie(
                    n_nodes=n_nodes,
                    edge_cache=edge_cache,
                    radial_feat=radial_feat[:, 1:, :],
                    zonal_coupling=zonal_coupling,
                ).unsqueeze(2)

        # === Step 8. Fuse edge type features into radial features (fp32+) ===
        with nvtx_range("radial_fuse"):
            radial_feat = radial_feat.to(dtype=self.dtype)
            radial_feat = radial_feat + rearrange(
                edge_cache.edge_type_feat.to(dtype=self.dtype), "E C -> E 1 C"
            )
            rad_feat_per_block = [
                radial_feat[:, :rad_len, :] for rad_len in self.rad_sizes_per_block
            ]

        # === Step 9. Convert to self.dtype and run blocks ===
        with nvtx_range("blocks"):
            x = x.to(dtype=self.dtype)  # (N, D, 1, C)
            if force_embedding is not None:
                x = x + force_embedding.to(dtype=self.dtype)
            edge_cache = edge_cache_to_dtype(edge_cache, self.dtype)
            with self._compute_mode_ctx(extended_coord.device):
                x = self._forward_blocks(
                    x, edge_cache, rad_feat_per_block, comm_dict=comm_dict
                )

        # === Step 10. Keep the owned-atom rows for the read-out ===
        # ``n_out_nodes`` is the owned-node count in the flattened layout
        # (``nf * nloc``). Single-domain: ``out_nloc == n_per_frame``, so this
        # equals the whole node set and the slice is a no-op. Parallel
        # (single-frame): it drops the trailing ghost rows that only fed message
        # passing -- LAMMPS orders owned atoms before ghosts, so they lead.
        n_out_nodes = nf * out_nloc
        x = x[:n_out_nodes]

        # === Step 11. Final l=0 output mixing ===
        # ``none`` feeds the l=0 slice only; ``glu``/``mlp`` feed the full
        # (N, D, 1, C) node tensor so the SO(3) grid folds l>0 into l=0. The
        # residual is added on the full coefficient tensor before extracting
        # l=0: slicing the summed tensor rather than the FFN output keeps the
        # saved degree-axis stride static under torch.compile dynamic shapes.
        with nvtx_range("output_ffn"):
            ffn_in = (
                x[:, 0:1, :, :]
                .reshape(n_out_nodes, 1, 1, self.channels)
                .to(dtype=self.compute_dtype)
                if self.so3_readout == "none"
                # truncate to the final node degree: the empty-edge path
                # skips the blocks, leaving x at node_ebed_dims[0]; output_ffn
                # is built for node_ebed_dims[-1]. No-op when blocks ran.
                else x[:, : self.node_ebed_dims[-1], :, :].to(dtype=self.compute_dtype)
            )
            x_scalar = (ffn_in + self.output_ffn(ffn_in))[:, 0:1, :, :]

        # === Step 12. Reshape to (nf, nloc, channels) and return ===
        descriptor = x_scalar.reshape(nf, out_nloc, self.channels)  # (nf, nloc, C)
        return descriptor.to(dtype=env.GLOBAL_PT_FLOAT_PRECISION), x.contiguous()

    def _forward_blocks(
        self,
        x: torch.Tensor,
        edge_cache: EdgeFeatureCache,
        radial_feat_per_block: list[torch.Tensor],
        comm_dict: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Run the interaction blocks with optional depth attention.

        Parameters
        ----------
        x
            Initial node features with shape (N, D, 1, C).
        edge_cache
            Per-edge cache.
        radial_feat_per_block
            List of per-block radial features already truncated to l_schedule[i]+1.
        comm_dict
            Border-exchange tensors for parallel inference, forwarded to each
            block. The block refreshes ghost rows at the SO(2) convolution
            input — the descriptor's only cross-node operation — so message
            passing always reads up-to-date neighbours regardless of the
            (per-node) attention-residual history.

        Returns
        -------
        torch.Tensor
            Output features with shape (N, D, 1, C).
        """
        if not self.use_full_attn_res and not self.use_block_attn_res:
            # === Fast path without descriptor-level attention residuals ===
            for i, block in enumerate(self.blocks):
                x = x[:, : self.node_ebed_dims[i], :, :]
                blk_radial = radial_feat_per_block[i]
                with nvtx_range(f"block_{i}"):
                    x, _, _, _ = block(
                        x,
                        edge_cache,
                        blk_radial,
                        comm_dict=self._block_comm(i, comm_dict),
                    )
            return x

        n_node = x.shape[0]

        def node_l0_extractor(v: torch.Tensor) -> torch.Tensor:
            """Extract scalar features from global SO(3) layout."""
            return v[:, 0, :, :].reshape(n_node, self.channels)

        if self.use_full_attn_res:
            # === Step 1. Maintain descriptor-level unit history ===
            unit_history = [x]

            # === Step 2. Run each block with selective unit-history aggregation ===
            for i, block in enumerate(self.blocks):
                current_dim = self.node_ebed_dims[i]
                current_x = x[:, :current_dim, :, :]
                truncated_unit_history = [
                    source[:, :current_dim, :, :] for source in unit_history
                ]
                blk_radial = radial_feat_per_block[i]
                with nvtx_range(f"block_{i}"):
                    block_output, _, so2_unit_output, ffn_unit_outputs = block(
                        current_x,
                        edge_cache,
                        blk_radial,
                        unit_history=truncated_unit_history,
                        comm_dict=self._block_comm(i, comm_dict),
                    )
                unit_history.append(so2_unit_output)
                unit_history.extend(ffn_unit_outputs)
                x = block_output

            # === Step 3. Final aggregation over all completed unit representations ===
            final_dim = self.node_ebed_dims[-1]
            final_sources = [source[:, :final_dim, :, :] for source in unit_history]
            x = self.final_full_attn_res(
                sources=final_sources,
                scalar_extractor=node_l0_extractor,
                current_x=x,
            ).to(dtype=self.dtype)
            return x

        # === Step 1. Maintain descriptor-level block history ===
        block_history = [x]

        # === Step 2. Run each block with selective block-history aggregation ===
        for i, block in enumerate(self.blocks):
            current_dim = self.node_ebed_dims[i]
            current_x = x[:, :current_dim, :, :]
            truncated_block_history = [
                source[:, :current_dim, :, :] for source in block_history
            ]
            blk_radial = radial_feat_per_block[i]
            with nvtx_range(f"block_{i}"):
                block_output, block_summary, _, _ = block(
                    current_x,
                    edge_cache,
                    blk_radial,
                    unit_history=truncated_block_history,
                    comm_dict=self._block_comm(i, comm_dict),
                )
            block_history.append(block_summary)
            x = block_output

        # === Step 3. Final aggregation over all completed block summaries ===
        final_dim = self.node_ebed_dims[-1]
        final_sources = [source[:, :final_dim, :, :] for source in block_history]
        x = self.final_block_attn_res(
            sources=final_sources,
            scalar_extractor=node_l0_extractor,
            current_x=x,
        ).to(dtype=self.dtype)
        return x

    def _build_gie_zonal_coupling(
        self,
        edge_cache: EdgeFeatureCache,
    ) -> torch.Tensor | None:
        """
        Build node-level zonal coupling for GIE when node degrees exceed MP degrees.

        Returns
        -------
        torch.Tensor or None
            Coupling with shape ``(E, D_node - 1)`` when ``extra_node_l > 0``;
            otherwise None, letting GIE gather from the MP Wigner-D cache.
        """
        if self.gie_zonal_wigner_calc is None:
            return None
        mp_row_count = self.ebed_dims[0] - 1
        mp_row_index = self.gie.non_scalar_row_index[:mp_row_count]
        mp_m0_col_index = self.gie.zonal_m0_col_index_for_row[:mp_row_count]
        mp_coupling = edge_cache.Dt_full[
            :,
            mp_row_index,
            mp_m0_col_index,
        ]
        edge_quat = edge_cache.edge_quat
        if edge_quat is None:
            edge_len = safe_norm(edge_cache.edge_vec, self.eps)
            edge_quat = build_edge_quaternion(
                edge_cache.edge_vec,
                edge_len=edge_len,
                eps=self.eps,
            )
        extra_coupling = self.gie_zonal_wigner_calc.forward_zonal(
            edge_quat,
            lmin=self.lmax + 1,
        )
        return torch.cat([mp_coupling, extra_coupling], dim=1)

    def _apply_charge_spin_embedding(
        self,
        type_ebed: torch.Tensor,
        charge_spin: torch.Tensor,
        *,
        nf: int,
        nloc: int,
    ) -> torch.Tensor:
        """
        Add frame-level charge and spin conditions to scalar type features.

        Parameters
        ----------
        type_ebed
            Flattened type embeddings with shape (nf * nloc, channels).
        charge_spin
            Frame-level charge and spin conditions with shape (nf, 2).
        nf
            Number of frames.
        nloc
            Number of local atoms.

        Returns
        -------
        torch.Tensor
            Conditioned type embeddings with shape (nf * nloc, channels).
        """
        condition = self.charge_spin_embedding(charge_spin.to(dtype=type_ebed.dtype))
        condition = condition[:, None, :].expand(nf, nloc, self.channels)
        return type_ebed + condition.reshape_as(type_ebed)

    def _edge_type_keep_mask(
        self,
        atype_flat: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build keep mask for edge pairs based on excluded type pairs.

        Parameters
        ----------
        atype_flat
            Flattened local atom types with shape (N,).
        src
            Source indices with shape (E,).
        dst
            Destination indices with shape (E,).

        Returns
        -------
        torch.Tensor
            Boolean mask with shape (E,), True means keep.
        """
        if self.emask.no_exclusion:
            return torch.ones_like(src, dtype=torch.bool, device=src.device)
        type_i = atype_flat.index_select(0, dst)
        type_j = atype_flat.index_select(0, src)
        type_i = torch.where(type_i >= 0, type_i, self.ntypes)
        type_j = torch.where(type_j >= 0, type_j, self.ntypes)
        type_ij = type_i * (self.ntypes + 1) + type_j
        type_mask = self.emask.type_mask.to(device=atype_flat.device)
        keep = type_mask.index_select(0, type_ij.to(dtype=torch.long))
        return keep.to(dtype=torch.bool)

    @staticmethod
    def _broadcast_grid_setting(
        value: bool | int | list[bool] | list[int],
        *,
        name: str,
        cast: type,
        non_negative: bool = False,
    ) -> list:
        """Normalize a grid-path setting to ``[node_wise, message_node, ffn]``.

        A scalar is broadcast to all three grid paths, while a length-three
        list is validated element-wise. When ``non_negative`` is set, every
        entry must be ``>= 0``.
        """
        entries = list(value) if isinstance(value, list) else [value, value, value]
        if len(entries) != 3:
            raise ValueError(
                f"`{name}` must be a {cast.__name__} or a list[{cast.__name__}] "
                "of length 3: [node_wise, message_node, ffn]"
            )
        normalized = [cast(entry) for entry in entries]
        if non_negative and any(entry < 0 for entry in normalized):
            raise ValueError(f"`{name}` entries must be non-negative")
        return normalized

    def _resolve_ffn_neurons(
        self,
        ffn_neurons: int,
        *,
        glu_activation: bool,
    ) -> int:
        """Resolve one FFN hidden width from the descriptor config."""
        resolved = int(ffn_neurons)
        if resolved < 0:
            raise ValueError("`ffn_neurons` must be >= 0")
        if resolved > 0:
            return resolved
        base_width = (
            (8.0 * float(self.channels) / 3.0)
            if glu_activation
            else (4.0 * float(self.channels))
        )
        return int(32 * math.ceil(base_width / 32.0))

    def _init_lm_schedules(
        self,
        lmax: int,
        n_blocks: int,
        l_schedule: list[int] | None,
        mmax: int | None,
        m_schedule: list[int] | None,
    ) -> None:
        """Parse and validate L/M schedules, setting self.l_schedule/m_schedule/lmax/mmax."""
        # === L schedule ===
        if l_schedule is None:
            self.l_schedule = [int(lmax)] * int(n_blocks)
        else:
            self.l_schedule = [int(x) for x in l_schedule]
        if len(self.l_schedule) == 0:
            raise ValueError("`l_schedule` must be non-empty")
        if any(x < 0 for x in self.l_schedule):
            raise ValueError("`l_schedule` entries must be non-negative")
        if any(
            self.l_schedule[i] < self.l_schedule[i + 1]
            for i in range(len(self.l_schedule) - 1)
        ):
            raise ValueError("`l_schedule` must be non-increasing (pyramid schedule)")

        self.lmax = int(self.l_schedule[0])
        self.n_blocks = len(self.l_schedule)

        # === M schedule ===
        if m_schedule is None:
            if mmax is None:
                self.m_schedule = [int(l) for l in self.l_schedule]
            else:
                mmax_i = int(mmax)
                if mmax_i < 0:
                    raise ValueError("`mmax` must be non-negative")
                self.m_schedule = [min(mmax_i, int(l)) for l in self.l_schedule]
        else:
            self.m_schedule = [int(x) for x in m_schedule]
        if len(self.m_schedule) == 0:
            raise ValueError("`m_schedule` must be non-empty")
        if len(self.m_schedule) != len(self.l_schedule):
            raise ValueError("`m_schedule` must have the same length as `l_schedule`")
        if any(x < 0 for x in self.m_schedule):
            raise ValueError("`m_schedule` entries must be non-negative")
        if any(m > l for m, l in zip(self.m_schedule, self.l_schedule, strict=True)):
            raise ValueError(
                "`m_schedule` entries must satisfy `m_schedule[i] <= l_schedule[i]`"
            )

        self.mmax = int(self.m_schedule[0])

    def _init_node_l_schedules(self, extra_node_l: int) -> None:
        """Parse node degree schedules derived from message-passing schedules."""
        self.extra_node_l = int(extra_node_l)
        if self.extra_node_l < 0:
            raise ValueError("`extra_node_l` must be non-negative")
        self.node_l_schedule = [
            int(l_value) + self.extra_node_l for l_value in self.l_schedule
        ]
        self.node_ebed_dims = [
            get_so3_dim_of_lmax(l_value) for l_value in self.node_l_schedule
        ]
        self.node_lmax = int(self.node_l_schedule[0])
        self.node_ebed_dim = int(self.node_ebed_dims[0])

    def _canonicalize_charge_spin(
        self,
        charge_spin: torch.Tensor | None,
        *,
        nf: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        """
        Canonicalize charge/spin conditions for the public descriptor path.

        Parameters
        ----------
        charge_spin
            Optional frame-level charge and spin conditions.
        nf
            Number of frames.
        dtype
            Target floating-point dtype.
        device
            Target device.

        Returns
        -------
        torch.Tensor or None
            Tensor with shape (nf, 2) when condition embedding is enabled.
        """
        if self.charge_spin_embedding is None:
            return None
        if charge_spin is None:
            if self.default_chg_spin is None:
                raise ValueError("`charge_spin` is required for this SeZM descriptor.")
            charge_spin = torch.tensor(
                self.default_chg_spin,
                dtype=dtype,
                device=device,
            ).view(1, 2)
        else:
            charge_spin = charge_spin.to(dtype=dtype, device=device)

        if charge_spin.ndim == 1:
            if charge_spin.numel() != 2:
                raise ValueError("`charge_spin` must contain [charge, spin].")
            charge_spin = charge_spin.view(1, 2)
        elif charge_spin.ndim != 2 or charge_spin.shape[-1] != 2:
            raise ValueError("`charge_spin` must have shape (nf, 2).")

        if charge_spin.shape[0] == 1 and nf != 1:
            charge_spin = charge_spin.expand(nf, -1)
        elif charge_spin.shape[0] != nf:
            raise ValueError("`charge_spin` first dimension must match nframes.")
        return charge_spin

    def _block_comm(
        self,
        block_idx: int,
        comm_dict: dict[str, torch.Tensor] | None,
    ) -> dict[str, torch.Tensor] | None:
        """Return the border-exchange tensors block ``block_idx`` actually needs.

        Only the SO(2) convolution reads neighbour features, so a block needs the
        ghost exchange exactly when its neighbour rows cannot be rebuilt locally.
        Block 0 reads the initial node state: a rank reproduces its ghost rows
        from ``extended_atype`` (type embedding) unless env-seed / GIE folds
        neighbour-environment information into them. Every later block reads a
        previous block's output, which a rank cannot reproduce for ghosts (they
        receive no messages locally). Returning ``None`` skips the exchange, so a
        purely local model (``use_env_seed=False`` with a single block) runs
        multi-rank with no communication at all.
        """
        if comm_dict is None:
            return None
        if block_idx == 0 and not self.use_env_seed:
            return None
        return comm_dict

    @contextmanager
    def _compute_mode_ctx(self, device: torch.device) -> Generator[None, None, None]:
        """
        Context manager that applies automatic mixed precision (AMP) for forward().

        Parameters
        ----------
        device
            The device of the input tensors (used to determine if CUDA ops apply).

        Notes
        -----
        - When `use_amp=True` and the model is in training mode, enables
          torch.autocast with bfloat16 on CUDA. This can improve speed and
          reduce memory usage on GPUs with native bfloat16 support.
          Disable AMP on GPUs without native bfloat16 support to avoid runtime
          errors or additional conversion overhead.
        - Only affects autocast-eligible operations.
        - Does nothing during inference (`self.training=False`), on non-CUDA
          devices, or when `use_amp=False`.

        Yields
        ------
        None
            Runs the wrapped region under the configured AMP setting.
        """
        if not self.use_amp or device.type != "cuda" or not self.training:
            yield
            return

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            yield

    # === DeePMD descriptor interface ===
    def get_rcut(self) -> float:
        return self.rcut

    def get_rcut_smth(self) -> float:
        return self.rcut

    def get_sel(self) -> list[int]:
        return self.sel

    def get_nsel(self) -> int:
        return sum(self.sel)

    def get_ntypes(self) -> int:
        return self.ntypes

    def get_type_map(self) -> list[str]:
        return self.type_map if self.type_map is not None else []

    def get_dim_chg_spin(self) -> int:
        """Return the charge/spin condition width."""
        return 2 if self.add_chg_spin_ebd else 0

    def has_default_chg_spin(self) -> bool:
        """Return whether default charge/spin conditions are configured."""
        return self.default_chg_spin is not None

    def get_default_chg_spin(self) -> list[float] | None:
        """Return default charge/spin conditions."""
        return self.default_chg_spin

    def get_dim_out(self) -> int:
        return self.channels

    def get_dim_emb(self) -> int:
        return self.get_dim_out()

    def mixed_types(self) -> bool:
        """
        If true, the descriptor
        1. assumes total number of atoms aligned across frames;
        2. requires a neighbor list that does not distinguish different atomic types.

        If false, the descriptor
        1. assumes total number of atoms of each atom type aligned across frames;
        2. requires a neighbor list that distinguishes different atomic types.

        SeZM uses SeZMTypeEmbedding for type handling, so it does not require
        a type-distinguished neighbor list.
        """
        return True

    def has_message_passing(self) -> bool:
        # SeZM resolves ghost neighbours through the atom-map fold (single
        # domain) or border_op exchange (parallel) instead of reading them
        # directly, so its lower path always needs message-passing handling.
        return True

    def has_message_passing_across_ranks(self) -> bool:
        """Whether multi-rank inference needs cross-rank ghost-feature exchange.

        SeZM reads ghost-neighbour features at every interaction block, so a
        domain-decomposed run must exchange them through ``border_op``. Source
        Freeze Propagation bridging is excluded: its per-node gate folds a
        node's entire outgoing-edge set, which a single rank cannot observe for
        ghost owners, so the edge-based with-comm artifact is not exported for
        bridging models and multi-rank inference fails fast instead.
        """
        return self.bridging_switch is None

    def need_sorted_nlist_for_lower(self) -> bool:
        return False

    def get_env_protection(self) -> float:
        return self.eps

    @property
    def dim_out(self) -> int:
        return self.get_dim_out()

    @property
    def dim_emb(self) -> int:
        return self.get_dim_emb()

    def share_params(
        self, base_class: Any, shared_level: int, resume: bool = False
    ) -> None:
        """
        Share the parameters of self to the base_class with shared_level during multitask training.

        SeZM does not rely on running mean/stddev statistics in ``forward``
        (``EquivariantRMSNorm`` is used instead), so only submodules and
        the optional FiLM strength parameters need to be linked.

        Parameters
        ----------
        base_class
            The base class to share parameters with. Must be the same class as self.

        shared_level
            The level of sharing.

            - ``0``: share every learnable submodule and FiLM strength parameter
              (type_embedding, env_seed_embedding, film_*_norm,
              film_*_strength_log, radial_basis, radial_embedding,
              edge_envelope, wigner_calc, gie, blocks, final_*_attn_res,
              output_ffn).
            - ``1``: share ``type_embedding`` and optional condition embedding.

        resume
            Unused for SeZM; kept for interface compatibility.

        Raises
        ------
        NotImplementedError
            If ``shared_level`` is not ``0`` or ``1``.
        """
        del resume
        assert self.__class__ == base_class.__class__, (
            "Only descriptors of the same type can share params!"
        )
        if shared_level == 0:
            # NOTE: ``nn.Module.__setattr__`` routes plain assignment of a
            # child ``nn.Module`` through the ``_modules`` dict, so iterating
            # that dict covers every learnable submodule registered by
            # ``__init__`` (type_embedding, env_seed_embedding, film norms,
            # radial_*, edge_envelope, wigner_calc, gie, blocks, final attn
            # residuals, output_ffn).  Raw ``nn.Parameter`` attributes
            # (``film_*_strength_log``) live in ``_parameters`` instead and
            # are linked explicitly below.
            for item in self._modules:
                self._modules[item] = base_class._modules[item]
            for name in ("film_scale_strength_log", "film_shift_strength_log"):
                if self._parameters.get(name) is not None:
                    self._parameters[name] = base_class._parameters[name]
        elif shared_level == 1:
            self._modules["type_embedding"] = base_class._modules["type_embedding"]
            if self.charge_spin_embedding is not None:
                self._modules["charge_spin_embedding"] = base_class._modules[
                    "charge_spin_embedding"
                ]
        else:
            raise NotImplementedError

    def enable_compression(
        self,
        min_nbor_dist: float,
        table_extrapolate: float = 5,
        table_stride_1: float = 0.01,
        table_stride_2: float = 0.1,
        check_frequency: int = -1,
    ) -> None:
        """Receive the statistics (distance, max_nbor_size and env_mat_range) of the training data.

        Parameters
        ----------
        min_nbor_dist
            The nearest distance between atoms
        table_extrapolate
            The scale of model extrapolation
        table_stride_1
            The uniform stride of the first table
        table_stride_2
            The uniform stride of the second table
        check_frequency
            The overflow check frequency
        """
        raise NotImplementedError("Compression is unsupported for SeZM.")

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat: Any | None = None
    ) -> None:
        raise NotImplementedError("change_type_map is not supported for SeZM")

    def reinit_exclude(
        self, exclude_types: list[tuple[int, int]] | None = None
    ) -> None:
        if exclude_types is None:
            exclude_types = []
        self.exclude_types = exclude_types
        self.emask = PairExcludeMask(self.ntypes, exclude_types=exclude_types)

    # =========================================================================
    # Statistics interface (interface compatibility only)
    # -------------------------------------------------------------------------
    # SeZM uses EquivariantRMSNorm inside blocks for feature normalization,
    # so mean/stddev are NOT used in forward(). These methods are kept for:
    #   1. Interface compatibility with BaseDescriptor
    #   2. Consistent serialization format (davg/dstd in checkpoint)
    # =========================================================================

    def set_stat_mean_and_stddev(
        self, mean: torch.Tensor, stddev: torch.Tensor
    ) -> None:
        """Set mean and stddev (interface compatibility, not used in forward)."""
        self.mean = mean
        self.stddev = stddev

    def get_stat_mean_and_stddev(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get mean and stddev (interface compatibility, not used in forward)."""
        return self.mean, self.stddev

    def compute_input_stats(
        self,
        merged: Callable[[], list[dict]] | list[dict],
        path: DPPath | None = None,
    ) -> None:
        """
        Compute statistics (interface compatibility, not used in forward).

        SeZM uses learnable EquivariantRMSNorm for normalization, so these
        statistics do not affect the forward pass. This is a no-op that keeps
        mean/stddev at their initialized values (zero/one) for interface consistency.
        """
        # No-op: mean and stddev are already initialized to zero/one in __init__
        # and are not used in forward() due to EquivariantRMSNorm.

    def serialize(self) -> dict[str, Any]:
        state = self.state_dict()
        return {
            "@class": "Descriptor",
            "type": "SeZM",
            "@version": self.version,
            "config": {
                "ntypes": self.ntypes,
                "sel": self.sel,
                "rcut": self.rcut,
                "env_exp": self.env_exp,
                "type_map": self.type_map,
                "lmax": self.lmax,
                "n_blocks": self.n_blocks,
                "l_schedule": self.l_schedule,
                "mmax": self.mmax,
                "kmax": self.kmax,
                "m_schedule": self.m_schedule,
                "extra_node_l": self.extra_node_l,
                "channels": self.channels,
                "basis_type": self.basis_type,
                "n_radial": self.n_radial,
                "radial_mlp": self.radial_mlp,
                "use_env_seed": self.use_env_seed,
                "random_gamma": self.random_gamma,
                "so2_norm": self.so2_norm,
                "so2_layers": self.so2_layers,
                "so2_attn_res": self.so2_attn_res_mode,
                "radial_so2_mode": self.radial_so2_mode,
                "radial_so2_rank": self.radial_so2_rank,
                "n_focus": self.n_focus,
                "focus_dim": self.focus_dim,
                "ffn_neurons": self.ffn_neurons,
                "grid_mlp": self.grid_mlp,
                "grid_branch": self.grid_branch,
                "ffn_blocks": self.ffn_blocks,
                "layer_scale": self.layer_scale,
                "n_atten_head": self.n_atten_head,
                "atten_f_mix": self.atten_f_mix,
                "atten_v_proj": self.use_atten_v_proj,
                "atten_o_proj": self.use_atten_o_proj,
                "sandwich_norm": self.sandwich_norm,
                "full_attn_res": self.full_attn_res_mode,
                "block_attn_res": self.block_attn_res_mode,
                "s2_activation": self.s2_activation,
                "ffn_so3_grid": self.ffn_so3_grid,
                "node_wise_s2": self.node_wise_s2,
                "node_wise_so3": self.node_wise_so3,
                "message_node_s2": self.message_node_s2,
                "message_node_so3": self.message_node_so3,
                "so3_readout": self.so3_readout,
                "lebedev_quadrature": self.lebedev_quadrature,
                "activation_function": self.activation_function,
                "glu_activation": self.glu_activation,
                "precision": RESERVED_PRECISION_DICT[self.dtype],
                "mlp_bias": self.mlp_bias,
                "exclude_types": self.exclude_types,
                "eps": self.eps,
                "trainable": self.trainable,
                "seed": self.seed,
                "inner_clamp_r_inner": self.inner_clamp_r_inner,
                "inner_clamp_r_outer": self.inner_clamp_r_outer,
                "add_chg_spin_ebd": self.add_chg_spin_ebd,
                "default_chg_spin": self.default_chg_spin,
            },
            "@variables": {key: np_safe(value) for key, value in state.items()},
            "env_mat": DPEnvMat(self.rcut, self.rcut, self.eps).serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> DescrptSeZM:
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "Descriptor":
            raise ValueError(f"Invalid class for DescrptSeZM: {data_cls}")
        type_val = data.pop("type")
        if type_val not in ("SeZM", "sezm", "dpa4"):
            raise ValueError(f"Invalid type for DescrptSeZM: {type_val}")
        version = float(data.pop("@version"))
        check_version_compatibility(version, cls.LATEST_VERSION, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        data.pop("env_mat", None)
        config.pop("s2_grid_resolution", None)
        obj = cls(**config)
        obj.version = version
        obj.version_tensor.fill_(version)
        template = obj.state_dict()
        state = {
            key: safe_numpy_to_tensor(
                value, device=template[key].device, dtype=template[key].dtype
            )
            for key, value in variables.items()
        }
        obj.load_state_dict(state)
        return obj

    @classmethod
    def update_sel(
        cls,
        train_data: DeepmdDataSystem,
        type_map: list[str] | None,
        local_jdata: dict,
    ) -> tuple[dict, float | None]:
        """
        Update the selection and perform neighbor statistics.

        Parameters
        ----------
        train_data : DeepmdDataSystem
            Data used to do neighbor statistics.
        type_map : list[str] | None
            The name of each type of atoms.
        local_jdata : dict
            The local data refer to the current class.

        Returns
        -------
        dict
            The updated local data.
        float | None
            The minimum distance between two atoms.
        """
        local_jdata_cpy = local_jdata.copy()
        min_nbor_dist, sel = UpdateSel().update_one_sel(
            train_data,
            type_map,
            local_jdata_cpy["rcut"],
            local_jdata_cpy["sel"],
            True,  # mixed_type=True for unified sel
        )
        local_jdata_cpy["sel"] = sel[0]
        return local_jdata_cpy, min_nbor_dist

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Fold LoRA adapters and drop transient state before loading.

        When a LoRA-trained checkpoint is loaded into a plain (non-LoRA)
        descriptor, any ``A_by_l``/``B_by_l`` (SO3) and
        ``A_m0``/``B_m0``/``A_m.*``/``B_m.*`` (SO2) keys are folded into
        their corresponding base weight keys (``weight``, ``weight_m0``,
        ``weight_m.*``) using ``ΔW = einsum(B, A) * scaling``.  The LoRA
        keys are then removed so the load proceeds as if the checkpoint
        were a plain SeZM.  This enables resume, finetune, and full-train
        from any LoRA checkpoint without manual merging.

        When the current descriptor is itself LoRA-injected, however, the
        incoming ``A_*`` / ``B_*`` / ``lora_scaling`` tensors are
        first-class parameters this descriptor already owns, *not*
        redundant adapters to be folded away.  Folding in that case would
        consume the LoRA keys and then ``super()._load_from_state_dict``
        would report them as ``Missing key(s)`` against the target
        module.  ``has_lora(self)`` gates the fold step so the
        LoRA-to-plain merge still runs when appropriate, while
        LoRA-to-LoRA loads (full training, ckpt resume, tests, and
        cross-instance copies via ``model_a.load_state_dict(
        model_b.state_dict())``) pass the adapter keys through
        unchanged.
        """
        # === Step 1. Fold any LoRA keys into base weights ===
        # Only fold when the current descriptor has no LoRA adapters
        # (see docstring).
        if not has_lora(self):
            fold_lora_state_dict_keys(state_dict, prefix)

        # === Step 2. Backfill descriptor version for legacy checkpoints ===
        version_key = prefix + "version_tensor"
        if version_key not in state_dict:
            state_dict[version_key] = self.version_tensor.new_tensor(1.0)

        # === Step 3. Drop transient descriptor state rebuilt at construction ===
        expected_keys = {prefix + key for key in self.state_dict().keys()}
        for full_key in list(state_dict.keys()):
            if full_key.startswith(prefix) and full_key not in expected_keys:
                state_dict.pop(full_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self.version = float(self.version_tensor.item())
