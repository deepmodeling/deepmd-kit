# SPDX-License-Identifier: LGPL-3.0-or-later
"""
DPA4/SeZM descriptor: Smooth Equivariant Zone-bridging Model.

dpmodel (array-API) backend

This implementation is designed around two goals:

1) Conservative forces: the descriptor is computed from differentiable energy.
2) Efficient inference: edge geometry and Wigner-D rotation blocks are computed
   exactly once per `call()` and reused by all interaction blocks.

Shared descriptor building blocks live in the `dpa4_nn` subpackage.

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

This module is the dpmodel (array-API) port of
``deepmd.pt.model.descriptor.sezm``.
"""

from __future__ import (
    annotations,
)

import math
from typing import (
    TYPE_CHECKING,
    Any,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel import (
    NativeOP,
)
from deepmd.dpmodel.array_api import (
    xp_asarray_nodetach,
    xp_scatter_sum,
    xp_take_first_n,
)
from deepmd.dpmodel.common import (
    PRECISION_DICT,
    get_xp_precision,
    to_numpy_array,
)
from deepmd.dpmodel.utils import (
    EnvMat,
)
from deepmd.dpmodel.utils.exclude_mask import (
    PairExcludeMask,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.dpmodel.utils.update_sel import (
    UpdateSel,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .base_descriptor import (
    BaseDescriptor,
)
from .dpa4_nn.attn_res import (
    DepthAttnRes,
)
from .dpa4_nn.block import (
    SeZMInteractionBlock,
)
from .dpa4_nn.edge_cache import (
    EdgeCache,
    build_edge_cache,
    build_edge_cache_from_edges,
    edge_cache_to_dtype,
)
from .dpa4_nn.embedding import (
    ChargeSpinEmbedding,
    EnvironmentInitialEmbedding,
    GeometricInitialEmbedding,
    SeZMTypeEmbedding,
    SpinEmbedding,
)
from .dpa4_nn.ffn import (
    EquivariantFFN,
)
from .dpa4_nn.indexing import (
    get_so3_dim_of_lmax,
)
from .dpa4_nn.norm import (
    ScalarRMSNorm,
)
from .dpa4_nn.radial import (
    BridgingSwitch,
    C3CutoffEnvelope,
    InnerClamp,
    RadialBasis,
    RadialMLP,
)
from .dpa4_nn.utils import (
    ATTN_RES_MODES,
    get_promoted_dtype,
    safe_norm,
)
from .dpa4_nn.wignerd import (
    WignerDCalculator,
    build_edge_quaternion,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )

    from deepmd.dpmodel.array_api import (
        Array,
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
class DescrptDPA4(NativeOP, BaseDescriptor):
    r"""
    SeZM descriptor.

    DPA4 stores the state of atom :math:`i` at layer :math:`l` as SO(3)
    coefficients :math:`\mathbf h_i^{(l,\ell,m)}`.  For an edge :math:`j\to i`,
    the source state is rotated into an edge-aligned frame, processed by an
    SO(2)-equivariant convolution, and rotated back:

    .. math::
        \mathbf q_{ji}^{(l)} =
        \mathbf D(\hat{\mathbf r}_{ji})^{-1}\mathbf h_j^{(l)},
        \qquad
        \mathbf m_{ji}^{(l)} =
        \mathbf D(\hat{\mathbf r}_{ji})
        \operatorname{SO2Conv}\!\left(
        \mathbf q_{ji}^{(l)},\boldsymbol\rho(r_{ji})\right),

    where :math:`\mathbf D` contains Wigner-D rotation blocks and
    :math:`\boldsymbol\rho` is the radial embedding multiplied by a smooth
    cutoff envelope.  In the baseline residual path, the aggregated message is
    first added directly to the node state, after which every equivariant FFN
    subblock applies its own residual update:

    .. math::

        \mathbf M_i^{(l)}=\sum_{j\in\mathcal N(i)}
        w_{ji}\mathbf m_{ji}^{(l)},\qquad
        \mathbf u_i^{(l,0)}=\mathbf h_i^{(l)}+\mathbf M_i^{(l)},

    .. math::

        \mathbf u_i^{(l,r)}=\mathbf u_i^{(l,r-1)}+
        \operatorname{FFN}_{\mathrm{eq},r}
        \!\left(\mathbf u_i^{(l,r-1)}\right),\qquad
        \mathbf h_i^{(l+1)}=\mathbf u_i^{(l,B)}.

    Consequently, one FFN subblock gives
    :math:`\mathbf h_i^{(l+1)}=\mathbf h_i^{(l)}+\mathbf M_i^{(l)}+
    \operatorname{FFN}_{\mathrm{eq}}(\mathbf h_i^{(l)}+\mathbf M_i^{(l)})`.
    The AttnRes modes replace these baseline shortcuts with selective
    depth-wise aggregation before the SO(2) and/or FFN units.

    The final read-out applies the configured scalar/equivariant read-out to
    the last interaction state and then keeps its invariant scalar output:

    .. math::
        \mathcal D_i = \operatorname{ScalarReadout}_{\mathrm{mode}}
        \left(\mathbf h_i^{(L)}\right).

    In ``so3_readout="none"`` mode, coefficients with :math:`\ell>0` are
    discarded and the :math:`\ell=0` slice is processed by the configured
    learned scalar residual FFN stack.  In ``"glu"`` and ``"mlp"`` modes,
    equivariant residual read-out blocks first fold higher-degree coefficients
    into the scalar channel before extraction.

    The weights :math:`w_{ji}` are either cutoff-envelope weights or normalized
    attention weights, depending on ``n_atten_head``.

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
    edge_norm
        Whether to apply channel RMSNorm on the descriptor's cutoff-vanishing
        branches: the radial network hidden layers, the environment-seed FiLM
        scale/shift logits, the cross-focus competition scalars, and the
        post-SO(2) residual messages. ``False`` replaces the first three norms
        with identity and changes only the post-SO(2) norm to unit-floor residual
        scaling. The unit floor uses ``sqrt(1 + variance)`` so small messages
        retain their cutoff envelope instead of receiving the standard
        ``1/sqrt(eps)`` small-signal gain.
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
    edge_cartesian
        If True, every block whose message-passing degree is ``1`` or ``2``
        replaces its per-edge SO(2) rotation-frame tensor product with the
        equivalent global-frame Cartesian rank-2 tensor product, removing the two
        per-edge Wigner-D rotations. Blocks with degree ``0`` or ``>= 3`` keep
        the SO(2) path. When every block takes the Cartesian path the full
        Wigner-D construction is skipped automatically, and the geometric initial
        embedding falls back to the zonal coupling.
    node_cartesian
        Per-node global-frame Cartesian rank-2 tensor product on the aggregated
        message, applied in every block whose message-passing degree is ``1`` or
        ``2``. Configured by a ``"<mode>:<layers>"`` string where ``mode`` is
        ``"default"`` (one-sided product) or ``"parity"`` (symmetrized product)
        and ``layers`` is the stack depth; a bare integer ``N`` is shorthand for
        ``"default:N"``, and ``"none"`` disables it. Orthogonal to
        ``edge_cartesian``: either, both, or neither may be set. Unlike
        ``edge_cartesian`` it does not affect the Wigner-D construction, since the
        per-edge message path is left unchanged.
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
        Number of blocks (only used when `l_schedule` is None). ``0`` disables
        the interaction blocks and builds the zero-block descriptor: type
        embedding, optional env FiLM and geometric initial embedding, then the
        final SO(3) read-out. The backbone degree is taken from `lmax`
        (plus `extra_node_l`). Geometry then enters only through the GIE, which
        is active when `use_env_seed=True` and `lmax + extra_node_l > 0`.
    so2_norm
        If True, apply intermediate ReducedEquivariantRMSNorm between SO(2) mixing layers.
        When False (default), no normalization is applied between layers.
    mixing_layers
        Number of learnable mixing layers in the per-edge message core of each
        block (legacy alias: ``so2_layers``). ``0`` applies only the
        edge-condition modulation: the rotation-free per-degree radial scaling on
        the SO(2) path, or a single ``x @ T_e`` when ``edge_cartesian`` applies.
        The per-node ``node_cartesian`` stack carries its own independent depth.
    so2_attn_res
        SO(2)-internal depth-wise attention residual mode inside each interaction
        block. Must be one of ``"none"``, ``"independent"``, or ``"dependent"``.
    radial_so2_mode
        Dynamic radial degree mixer mode inside SO(2) convolution. ``"none"``
        applies elementwise radial modulation, ``"degree"`` uses a
        channel-shared edge-conditioned cross-degree kernel, and
        ``"degree_channel"`` uses a per-channel cross-degree kernel. Has no
        effect on blocks taking the Cartesian path (``edge_cartesian`` with
        degree 1 or 2), where the dynamic radial degree mixer is bypassed.
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
        MLP (``"mlp"``). The Wigner-D frame order follows ``kmax``.
    readout_layers
        Number of stacked equivariant residual read-out FFNs (default ``1``).
        Every layer is an ``x + FFN(x)`` residual block sharing the read-out
        degree; intermediate layers keep the full SO(3) tensor so high-degree
        geometry is folded into ``l=0`` repeatedly, and only the final layer
        slices the ``l=0`` channel from its residual sum. With ``so3_readout`` of
        ``"none"`` the stack is a degree-0 scalar residual MLP on the ``l=0``
        slice.
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
        edge_norm: bool = True,
        use_env_seed: bool = True,
        random_gamma: bool = True,
        edge_cartesian: bool = False,
        node_cartesian: str | int = "none",
        lmax: int = 3,
        l_schedule: list[int] | None = None,
        mmax: int | None = 1,
        kmax: int = 1,
        m_schedule: list[int] | None = None,
        extra_node_l: int = 0,
        n_blocks: int = 3,
        so2_norm: bool = False,
        mixing_layers: int = 4,
        so2_layers: int | None = None,
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
        readout_layers: int = 1,
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
        use_spin: list[bool] | None = None,
        **kwargs: Any,
    ) -> None:
        self.version = float(self.LATEST_VERSION)
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
        self.edge_norm = bool(edge_norm)
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
        self.readout_layers = int(readout_layers)
        if self.readout_layers < 1:
            raise ValueError("`readout_layers` must be >= 1")
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
        self.compute_precision = str(
            np.dtype(get_promoted_dtype(PRECISION_DICT[self.precision])).name
        )
        self.mlp_bias = bool(mlp_bias)
        self.layer_scale = bool(layer_scale)
        self.use_amp = bool(use_amp)
        self.trainable = bool(trainable)
        self.seed = seed
        self.random_gamma = bool(random_gamma)
        self.edge_cartesian = bool(edge_cartesian)
        self.node_cartesian = str(node_cartesian)
        self.add_chg_spin_ebd = bool(add_chg_spin_ebd)
        if default_chg_spin is not None and len(default_chg_spin) != 2:
            raise ValueError("`default_chg_spin` must contain [charge, spin].")
        self.default_chg_spin = (
            None if default_chg_spin is None else [float(x) for x in default_chg_spin]
        )

        # === Native per-atom spin embedding ===
        # The spin vector enters the descriptor as an l=0 magnitude scalar plus
        # an l=1 direction feature (see ``SpinEmbedding``). Providing per-type
        # ``use_spin`` flags enables the native spin embedding.
        self.use_spin = None if use_spin is None else [bool(x) for x in use_spin]

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
        seed_spin_embedding = child_seed(self.seed, 8)

        # === L/M schedules ===
        self._init_lm_schedules(lmax, n_blocks, l_schedule, mmax, m_schedule)
        self.kmax = int(kmax)
        if self.kmax < 0:
            raise ValueError("`kmax` must be non-negative")
        if self.kmax > self.lmax:
            raise ValueError("`kmax` must be <= `lmax`")
        self._init_node_l_schedules(extra_node_l)
        self.rad_sizes_per_block = [l + 1 for l in self.l_schedule]

        self.so2_norm = bool(so2_norm)
        # ``so2_layers`` is the legacy alias for ``mixing_layers``; when supplied
        # it takes precedence so existing configs keep working.
        self.mixing_layers = int(mixing_layers if so2_layers is None else so2_layers)
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
            precision=self.compute_precision,  # force fp32+
            seed=seed_type_embedding,
            trainable=self.trainable,
        )
        if self.add_chg_spin_ebd:
            self.charge_spin_embedding: ChargeSpinEmbedding | None = (
                ChargeSpinEmbedding(
                    embed_dim=self.channels,
                    activation_function=self.activation_function,
                    precision=self.compute_precision,
                    seed=seed_charge_spin,
                    trainable=self.trainable,
                )
            )
        else:
            self.charge_spin_embedding = None

        if self.use_spin is not None:
            if self.node_init_lmax < 1:
                raise ValueError(
                    "`use_spin` requires a node degree >= 1 "
                    "(lmax + extra_node_l) to host the l=1 spin feature."
                )
            self.spin_embedding: SpinEmbedding | None = SpinEmbedding(
                ntypes=self.ntypes,
                channels=self.channels,
                use_spin=self.use_spin,
                activation_function=self.activation_function,
                precision=self.compute_precision,  # force fp32+
                seed=seed_spin_embedding,
                trainable=self.trainable,
            )
            # Packed rows hosting the l=1 spin coefficients (m = -1, 0, +1).
            self._spin_l1_rows = np.arange(1, 4, dtype=np.int64)
        else:
            self.spin_embedding = None

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
                    use_spin=self.use_spin,
                    precision=self.compute_precision,  # force fp32+
                    trainable=self.trainable,
                    seed=seed_env_seed,
                )
            )
            # The FiLM logits derive from the env-seed matrix D = envᵀenv, which
            # vanishes at rcut; normalizing them shares the radial network's
            # cutoff-smoothness issue, so ``edge_norm=False`` also drops these
            # norms (identity pass-through) to keep the FiLM scale/shift smooth.
            if self.edge_norm:
                self.film_scale_norm = ScalarRMSNorm(
                    channels=self.channels,
                    n_focus=1,
                    eps=self.eps,
                    precision=self.compute_precision,
                    trainable=self.trainable,
                )
                self.film_shift_norm = ScalarRMSNorm(
                    channels=self.channels,
                    n_focus=1,
                    eps=self.eps,
                    precision=self.compute_precision,
                    trainable=self.trainable,
                )
            else:
                self.film_scale_norm = None
                self.film_shift_norm = None
            film_strength_init = 0.01
            # Use 1D tensor (not scalar) for FSDP2 compatibility
            self.film_scale_strength_log = np.full(
                (1,),
                math.log(film_strength_init),
                dtype=PRECISION_DICT[self.compute_precision],
            )
            self.film_shift_strength_log = np.full(
                (1,),
                math.log(film_strength_init),
                dtype=PRECISION_DICT[self.compute_precision],
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
            precision=self.compute_precision,  # force fp32+
            exponent=self.env_exp[0],
        )

        # === Shared radial embedding: RBF -> per-l radial features ===
        # Output dimension follows the first node degree, directly usable by
        # GIE and truncated for each SO2Conv block.
        # radial_mlp specifies hidden layer sizes; input/output layers are prepended/appended.
        # Use fp32+ precision (same as RBF output) for numerical stability.
        radial_out_dim = (self.node_init_lmax + 1) * self.channels
        radial_mlp_layers = [self.n_radial, *self.radial_mlp, radial_out_dim]
        self.radial_embedding = RadialMLP(
            radial_mlp_layers,
            activation_function=self.activation_function,
            precision=self.compute_precision,  # force fp32+
            trainable=self.trainable,
            radial_norm=self.edge_norm,
            seed=seed_radial_embedding,
        )

        # === C^3 cutoff envelope for edge weight ===
        self.edge_envelope = C3CutoffEnvelope(rcut=self.rcut, exponent=self.env_exp[1])

        # === Edge-aligned Wigner-D calculator ===
        # Cartesian blocks (degree 1 or 2) skip the SO(2) rotations, so the full
        # per-edge Wigner-D blocks are built only when a block keeps the SO(2)
        # path (tracked by ``_need_full_wigner``).
        block_edge_cartesian = [
            self.edge_cartesian and l_b in (1, 2) for l_b in self.l_schedule
        ]
        block_node_cartesian = [
            self.node_cartesian if l_b in (1, 2) else "none" for l_b in self.l_schedule
        ]
        self._need_full_wigner = not all(block_edge_cartesian)
        self.wigner_calc = WignerDCalculator(
            lmax=self.mp_init_lmax,
            eps=self.eps,
            precision=self.compute_precision,  # force fp32+
        )

        self.use_gie = self.use_env_seed and self.node_init_lmax > 0
        if self.use_gie:
            self.gie = GeometricInitialEmbedding(
                lmax=self.node_init_lmax,
                channels=self.channels,
                precision=self.compute_precision,  # force fp32+
            )
            if self.extra_node_l > 0:
                self.gie_zonal_wigner_calc: WignerDCalculator | None = (
                    WignerDCalculator(
                        lmax=self.node_init_lmax,
                        eps=self.eps,
                        precision=self.compute_precision,
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
                    focus_norm=self.edge_norm,
                    so2_norm=self.so2_norm,
                    mixing_layers=self.mixing_layers,
                    so2_attn_res=self.so2_attn_res_mode,
                    radial_so2_mode=self.radial_so2_mode,
                    radial_so2_rank=self.radial_so2_rank,
                    edge_cartesian=block_edge_cartesian[block_idx],
                    node_cartesian=block_node_cartesian[block_idx],
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
                    so2_post_norm_eps=1.0e-5 if self.edge_norm else 1.0,
                    so2_activation_function=self.so2_activation_function,
                    ffn_pre_norm=self.ffn_pre_norm,
                    ffn_post_norm=self.ffn_post_norm,
                    ffn_activation_function=self.ffn_activation_function,
                    ffn_glu_activation=self.ffn_glu_activation,
                    mlp_bias=self.mlp_bias,
                    eps=self.eps,
                    precision=self.precision,
                    seed=child_seed(seed_blocks, block_idx),
                    trainable=self.trainable,
                )
            )
        self.blocks = blocks

        # === Optional descriptor-level attention residuals ===
        self.final_block_attn_res = None
        if self.use_full_attn_res:
            self.final_full_attn_res: DepthAttnRes | None = DepthAttnRes(
                channels=self.channels,
                input_dependent=self.full_attn_res_mode == "dependent",
                eps=self.eps,
                bias=self.mlp_bias,
                precision=self.compute_precision,
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
                precision=self.compute_precision,
                trainable=self.trainable,
                seed=child_seed(seed_block_attn, 2000),
            )

        # === Final FFN stack for l=0 output mixing ===
        # ``readout_layers`` residual blocks run in sequence (see
        # ``_apply_readout``): ``readout_pre_layers`` keep the full SO(3) tensor
        # and only the final ``output_ffn`` slices l=0. The final layer keeps the
        # ``output_ffn`` name and ``seed_out`` so a single-layer read-out matches
        # the single-module checkpoint layout.
        readout_lmax = self.node_readout_lmax
        readout_ffn_kwargs = {
            "lmax": 0 if self.so3_readout == "none" else readout_lmax,
            "channels": self.channels,
            "hidden_channels": self.out_ffn_neurons,
            "kmax": min(self.kmax, readout_lmax),
            "grid_mlp": self.so3_readout == "mlp",
            "grid_branch": 0,
            "precision": self.compute_precision,
            "s2_activation": False,
            "ffn_so3_grid": self.so3_readout != "none",
            "activation_function": self.out_activation_function,
            "glu_activation": self.out_glu_activation,
            "mlp_bias": self.mlp_bias,
            "trainable": self.trainable,
        }
        self.readout_pre_layers = [
            EquivariantFFN(**readout_ffn_kwargs, seed=child_seed(seed_out, layer_index))
            for layer_index in range(self.readout_layers - 1)
        ]
        self.output_ffn = EquivariantFFN(**readout_ffn_kwargs, seed=seed_out)

        # === Statistics buffers (interface compatibility) ===
        self.stats: dict[str, Any] | None = None
        self.mean = np.zeros(0, dtype=PRECISION_DICT[self.precision])
        self.stddev = np.ones(0, dtype=PRECISION_DICT[self.precision])

    def call(
        self,
        coord_ext: Array,
        atype_ext: Array,
        nlist: Array,
        mapping: Array | None = None,
        edge_index: Array | None = None,
        edge_vec: Array | None = None,
        edge_mask: Array | None = None,
        comm_dict: dict[str, Array] | None = None,
        fparam: Array | None = None,
        force_embedding: Array | None = None,
        charge_spin: Array | None = None,
        spin: Array | None = None,
    ) -> tuple[
        Array,
        Array | None,
        Array | None,
        Array | None,
        Array | None,
    ]:
        """
        Compute the descriptor.

        Parameters
        ----------
        coord_ext
            Extended coordinates of atoms with shape (nf, nall*3) or (nf, nall, 3) in Å.
        atype_ext
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
            ``D = (node_init_lmax + 1) ** 2``. This tensor is added to the
            initial SO(3) backbone state before the interaction blocks.
        charge_spin
            Frame-level charge and spin conditions with shape (nf, 2).

        Returns
        -------
        descriptor
            Descriptor with shape (nf, nloc, channels). Only l=0 is returned.
        rot_mat
            None (not used).
        g2
            None (not used).
        h2
            None (not used).
        sw
            None (not used).
        """
        xp = array_api_compat.array_namespace(coord_ext, atype_ext)
        device = array_api_compat.device(coord_ext)
        if coord_ext.ndim == 2:
            coord_ext = xp.reshape(coord_ext, (coord_ext.shape[0], -1, 3))
        elif coord_ext.ndim != 3:
            raise ValueError("coord_ext must have shape (nf, nall*3) or (nf, nall, 3)")

        if edge_index is not None:
            nf_edge = atype_ext.shape[0]
            charge_spin = self._canonicalize_charge_spin(
                charge_spin,
                nf=nf_edge,
                dtype=coord_ext.dtype,
                device=device,
            )
            descriptor, _ = self.call_with_edges(
                coord_ext=coord_ext,
                atype_ext=atype_ext,
                edge_index=edge_index,
                edge_vec=edge_vec,
                edge_mask=edge_mask,
                force_embedding=force_embedding,
                charge_spin=charge_spin,
                spin=spin,
            )
            return (
                descriptor,
                None,
                None,
                None,
                None,
            )

        # === Step 1. Setup dimensions ===
        coord_ext = xp.astype(coord_ext, get_xp_precision(xp, self.compute_precision))
        nf, nloc, nnei = nlist.shape
        nall = coord_ext.shape[1]
        n_nodes = nf * nloc
        charge_spin = self._canonicalize_charge_spin(
            charge_spin,
            nf=nf,
            dtype=coord_ext.dtype,
            device=device,
        )

        # === Step 2. Excluded type pairs ===
        if self.exclude_types:
            # (nf, nloc, nnei), True means keep.
            pair_keep_mask = xp.astype(
                self.emask.build_type_exclude_mask(nlist, atype_ext), xp.bool
            )
        else:
            pair_keep_mask = xp.ones_like(nlist, dtype=xp.bool)

        # === Step 3. Type embedding (l=0) ===
        atype_loc = xp_take_first_n(atype_ext, 1, nloc)  # (nf, nloc)
        type_ebed = xp.reshape(
            self.type_embedding(atype_loc), (n_nodes, self.channels)
        )  # (N, C)
        if self.charge_spin_embedding is not None:
            type_ebed = self._apply_charge_spin_embedding(
                type_ebed,
                charge_spin,
                nf=nf,
                nloc=nloc,
            )

        # Native spin: condition the l=0 type features on the spin magnitude
        # and hold the l=1 direction coefficients for the backbone seed.
        spin_vec = None
        if self.spin_embedding is not None and spin is not None:
            type_ebed, spin_vec = self._apply_spin_embedding(
                type_ebed, spin, xp.reshape(atype_loc, (-1,)), n_nodes=n_nodes
            )

        # === Step 4. Build edge cache once (geometry + RBF + Wigner-D) ===
        # Zone bridging (InnerClamp + SFPG + ZBL) is not routed through the
        # standard DeePMD path: bridging only makes physical sense when
        # paired with the ZBL energy that ``SeZMModel`` injects on the
        # sparse-edge path, so ``forward`` keeps the original
        # bridging-free aggregation semantics.
        edge_cache = build_edge_cache(
            type_ebed=type_ebed,
            extended_coord=coord_ext,
            nlist=nlist,
            mapping=mapping,
            pair_keep_mask=pair_keep_mask,
            eps=self.eps,
            deg_norm_floor=(self.deg_norm_floor if self.version >= 1.1 else self.eps),
            edge_envelope=self.edge_envelope,
            radial_basis=self.radial_basis,
            n_radial=self.radial_basis.n_radial,
            # Random local-Z roll is a training-only augmentation;
            # the model is roll-equivariant, so inference fixes gamma.
            random_gamma=False,
            wigner_calc=self.wigner_calc,
            build_wigner=self._need_full_wigner,
        )

        ebed_dim_0 = self.node_init_dim  # (node_init_lmax+1)^2
        x0 = type_ebed  # (N, C)
        x0_out = x0  # (N, C)

        # === Step 5. Compute radial features once (fp32+) ===
        # Shape: (E, (node_init_lmax+1)*C) -> (E, node_init_lmax+1, C)
        radial_feat = xp.reshape(
            self.radial_embedding(edge_cache.edge_rbf),
            (-1, self.node_init_lmax + 1, self.channels),
        )  # (E, node_init_lmax+1, C)
        if self.version >= 1.1:
            radial_feat = radial_feat * xp.reshape(edge_cache.edge_env, (-1, 1, 1))

        # === Step 6. Env FiLM conditioning (optional, fp32+) ===
        if self.use_env_seed:
            atype_flat = xp.reshape(atype_loc, (-1,))  # (N,)
            spin_flat = (
                xp.reshape(spin, (n_nodes, 3))
                if (self.spin_embedding is not None and spin is not None)
                else None
            )
            film = self.env_seed_embedding(
                edge_cache=edge_cache,
                atype_flat=atype_flat,
                n_nodes=n_nodes,
                spin=spin_flat,
            )  # (N, 2*C)
            scale_logits = film[:, : self.channels]  # (N, C)
            shift_logits = film[:, self.channels :]  # (N, C)
            scale_hat = (
                self.film_scale_norm(scale_logits) if self.edge_norm else scale_logits
            )  # (N, C)
            shift_hat = (
                self.film_shift_norm(shift_logits) if self.edge_norm else shift_logits
            )  # (N, C)
            scale_strength = xp.exp(
                xp_asarray_nodetach(
                    xp, self.film_scale_strength_log[...], device=device
                )
            )
            shift_strength = xp.exp(
                xp_asarray_nodetach(
                    xp, self.film_shift_strength_log[...], device=device
                )
            )
            scale = 1.0 + scale_strength * xp.tanh(scale_hat)  # (N, C)
            shift = shift_strength * xp.tanh(shift_hat)  # (N, C)
            x0_out = x0 * scale + shift

        # === Step 7. Build backbone l=0 features ===
        x = xp.concat(
            [
                xp.reshape(x0_out, (n_nodes, 1, 1, self.channels)),
                xp.zeros(
                    (n_nodes, ebed_dim_0 - 1, 1, self.channels),
                    dtype=type_ebed.dtype,
                    device=device,
                ),
            ],
            axis=1,
        )  # (N, D, 1, C)

        # === Step 8. Geometric Initial Embedding (+ neighbor spin l=1) ===
        if self.use_gie:
            # GIE only needs l>=1, slice radial_feat[:, 1:, :]
            zonal_coupling = self._build_gie_zonal_coupling(edge_cache)
            spin_l1_message = (
                self.spin_embedding.edge_l1(
                    xp.reshape(spin, (n_nodes, 3)),
                    xp.reshape(atype_loc, (-1,)),
                    edge_cache,
                )
                if (self.spin_embedding is not None and spin is not None)
                else None
            )
            x = (
                x
                + self.gie(
                    n_nodes=n_nodes,
                    edge_cache=edge_cache,
                    radial_feat=radial_feat[:, 1:, :],
                    zonal_coupling=zonal_coupling,
                    spin_l1_message=spin_l1_message,
                )[:, :, None, :]
            )

        # === Step 9. Add the on-site native spin l=1 to the backbone ===
        # The neighbor-spin l=1 is aggregated inside GIE (degree-normalized like
        # the geometry); the atom's own spin direction is added here, un-normalized.
        if spin_vec is not None:
            spin_l1_rows = xp_asarray_nodetach(xp, self._spin_l1_rows, device=device)
            spin_l1_src = spin_vec[:, :, None, :]  # (N, 3, 1, C)
            scatter_index = xp.broadcast_to(
                xp.reshape(spin_l1_rows, (1, 3, 1, 1)), spin_l1_src.shape
            )
            x = xp_scatter_sum(x, 1, scatter_index, spin_l1_src)

        # === Step 10. Fuse edge type features into radial features (fp32+) ===
        radial_feat = radial_feat + xp.reshape(
            edge_cache.edge_type_feat, (-1, 1, self.channels)
        )
        radial_feat = xp.astype(radial_feat, get_xp_precision(xp, self.precision))
        rad_feat_per_block = [
            radial_feat[:, :rad_len, :] for rad_len in self.rad_sizes_per_block
        ]  # list of (E, lmax+1, C)

        # === Step 11. Convert to self.dtype and run blocks ===
        # The block stage is skipped entirely for the zero-block descriptor.
        # Array operations in the blocks also support an empty edge axis; avoid
        # inspecting that dynamic dimension in Python so TF graphs can retrace
        # across different atom counts.
        x = xp.astype(x, get_xp_precision(xp, self.precision))  # (N, D, 1, C)
        if force_embedding is not None:
            x = x + xp.astype(force_embedding, get_xp_precision(xp, self.precision))
        if self.blocks:
            edge_cache = edge_cache_to_dtype(
                edge_cache, get_xp_precision(xp, self.precision)
            )
            x = self._forward_blocks(x, edge_cache, rad_feat_per_block)

        # === Step 12. Final l=0 output mixing ===
        x_scalar = self._apply_readout(x, n_nodes)

        # === Step 13. Reshape to (nf, nloc, channels) and return ===
        descriptor = xp.reshape(x_scalar, (nf, nloc, self.channels))  # (nf, nloc, C)
        return (
            xp.astype(descriptor, get_xp_precision(xp, "global")),
            None,
            None,
            None,
            None,
        )

    def call_with_edges(
        self,
        *,
        coord_ext: Array,
        atype_ext: Array,
        edge_index: Array,
        edge_vec: Array,
        edge_mask: Array,
        force_embedding: Array | None = None,
        charge_spin: Array | None = None,
        spin: Array | None = None,
        comm_dict: dict[str, Array] | None = None,
        nloc: int | None = None,
    ) -> tuple[Array, Array]:
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
        coord_ext
            Coordinates with shape (nf, n*3) or (nf, n, 3) in Å, where ``n`` is
            ``nloc`` in the single-domain path and ``nall`` in the parallel path.
        atype_ext
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
            ``D = (node_init_lmax + 1) ** 2``. This tensor is added to the
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
        tuple[Array, Array]
            The scalar descriptor with shape ``(nf, nloc, channels)`` and the
            final equivariant latent with shape ``(nf * nloc, D_final, 1, channels)``.
        """
        xp = array_api_compat.array_namespace(coord_ext, atype_ext, edge_vec)
        device = array_api_compat.device(coord_ext)
        # === Step 1. Setup dimensions ===
        # ``n_per_frame`` is the per-frame node count: ``nloc`` in the
        # single-domain path and ``nall`` in the parallel path. ``out_nloc`` is
        # the owned-atom count used for the final local read-out.
        coord_ext = xp.astype(coord_ext, get_xp_precision(xp, self.compute_precision))
        nf, n_per_frame = atype_ext.shape[:2]
        parallel = comm_dict is not None
        if parallel:
            # Multi-rank parallel inference requires a custom border-exchange
            # communication op that is not available in the dpmodel backend.
            raise NotImplementedError(
                "multi-rank comm_dict inference is not supported in the dpmodel backend"
            )
        out_nloc = nloc if parallel else n_per_frame
        atype_flat = xp.reshape(atype_ext, (-1,))  # (N,)

        # === Step 2. Type embedding (l=0) ===
        type_ebed = xp.reshape(
            self.type_embedding(atype_ext), (-1, self.channels)
        )  # (N, C)
        if self.charge_spin_embedding is not None:
            type_ebed = self._apply_charge_spin_embedding(
                type_ebed,
                charge_spin,
                nf=nf,
                nloc=n_per_frame,
            )
        n_nodes = type_ebed.shape[0]

        # Native spin: condition the l=0 type features on the spin magnitude
        # and hold the l=1 direction coefficients for the backbone seed.
        spin_vec = None
        if self.spin_embedding is not None and spin is not None:
            type_ebed, spin_vec = self._apply_spin_embedding(
                type_ebed, spin, atype_flat, n_nodes=n_nodes
            )

        # === Step 3. Build edge cache once (sparse edges) ===
        edge_cache = build_edge_cache_from_edges(
            type_ebed=type_ebed,
            atype_flat=atype_flat,
            edge_index=edge_index,
            edge_vec=edge_vec,
            edge_mask=edge_mask,
            compute_dtype=get_xp_precision(xp, self.compute_precision),
            eps=self.eps,
            deg_norm_floor=(self.deg_norm_floor if self.version >= 1.1 else self.eps),
            inner_clamp=self.inner_clamp,
            bridging_switch=self.bridging_switch,
            edge_envelope=self.edge_envelope,
            radial_basis=self.radial_basis,
            has_exclude_types=bool(self.exclude_types),
            edge_type_keep_mask=self._edge_type_keep_mask,
            # Random local-Z roll is a training-only augmentation;
            # the model is roll-equivariant, so inference fixes gamma.
            random_gamma=False,
            wigner_calc=self.wigner_calc,
            build_wigner=self._need_full_wigner,
        )

        ebed_dim_0 = self.node_init_dim  # (node_init_lmax+1)^2
        x0 = type_ebed  # (N, C)
        x0_out = x0  # (N, C)

        # === Step 4. Compute radial features once (fp32+) ===
        radial_feat_flat = self.radial_embedding(edge_cache.edge_rbf)
        radial_feat = xp.reshape(
            radial_feat_flat,
            (
                radial_feat_flat.shape[0],
                self.node_init_lmax + 1,
                self.channels,
            ),
        )  # (E, node_init_lmax+1, C)
        if self.version >= 1.1:
            radial_feat = radial_feat * xp.reshape(edge_cache.edge_env, (-1, 1, 1))

        # === Step 5. Env FiLM conditioning (optional, fp32+) ===
        if self.use_env_seed:
            spin_flat = (
                xp.reshape(spin, (n_nodes, 3))
                if (self.spin_embedding is not None and spin is not None)
                else None
            )
            film = self.env_seed_embedding(
                edge_cache=edge_cache,
                atype_flat=atype_flat,
                n_nodes=n_nodes,
                spin=spin_flat,
            )  # (N, 2*C)
            scale_logits = film[:, : self.channels]  # (N, C)
            shift_logits = film[:, self.channels :]  # (N, C)
            scale_hat = (
                self.film_scale_norm(scale_logits) if self.edge_norm else scale_logits
            )  # (N, C)
            shift_hat = (
                self.film_shift_norm(shift_logits) if self.edge_norm else shift_logits
            )  # (N, C)
            scale_strength = xp.exp(
                xp_asarray_nodetach(
                    xp, self.film_scale_strength_log[...], device=device
                )
            )
            shift_strength = xp.exp(
                xp_asarray_nodetach(
                    xp, self.film_shift_strength_log[...], device=device
                )
            )
            scale = 1.0 + scale_strength * xp.tanh(scale_hat)  # (N, C)
            shift = shift_strength * xp.tanh(shift_hat)  # (N, C)
            x0_out = x0 * scale + shift

        # === Step 6. Build backbone l=0 features ===
        x = xp.concat(
            [
                xp.reshape(x0_out, (n_nodes, 1, 1, self.channels)),
                xp.zeros(
                    (n_nodes, ebed_dim_0 - 1, 1, self.channels),
                    dtype=type_ebed.dtype,
                    device=device,
                ),
            ],
            axis=1,
        )  # (N, D, 1, C)

        # === Step 7. Geometric Initial Embedding (+ neighbor spin l=1) ===
        if self.use_gie:
            zonal_coupling = self._build_gie_zonal_coupling(edge_cache)
            spin_l1_message = (
                self.spin_embedding.edge_l1(
                    xp.reshape(spin, (n_nodes, 3)), atype_flat, edge_cache
                )
                if (self.spin_embedding is not None and spin is not None)
                else None
            )
            x = (
                x
                + self.gie(
                    n_nodes=n_nodes,
                    edge_cache=edge_cache,
                    radial_feat=radial_feat[:, 1:, :],
                    zonal_coupling=zonal_coupling,
                    spin_l1_message=spin_l1_message,
                )[:, :, None, :]
            )

        # === Step 8. Add the on-site native spin l=1 to the backbone ===
        # The neighbor-spin l=1 is aggregated inside GIE; the
        # atom's own spin direction is added here, un-normalized.
        if spin_vec is not None:
            spin_l1_rows = xp_asarray_nodetach(xp, self._spin_l1_rows, device=device)
            spin_l1_src = spin_vec[:, :, None, :]  # (N, 3, 1, C)
            scatter_index = xp.broadcast_to(
                xp.reshape(spin_l1_rows, (1, 3, 1, 1)), spin_l1_src.shape
            )
            x = xp_scatter_sum(x, 1, scatter_index, spin_l1_src)

        # === Step 9. Fuse edge type features into radial features (fp32+) ===
        radial_feat = xp.astype(radial_feat, get_xp_precision(xp, self.precision))
        radial_feat = radial_feat + xp.reshape(
            xp.astype(edge_cache.edge_type_feat, get_xp_precision(xp, self.precision)),
            (-1, 1, self.channels),
        )
        rad_feat_per_block = [
            radial_feat[:, :rad_len, :] for rad_len in self.rad_sizes_per_block
        ]

        # === Step 10. Convert to self.dtype and run blocks ===
        # The block stage is skipped entirely for the zero-block descriptor,
        # sparing the working edge-cache dtype cast that only the blocks consume.
        x = xp.astype(x, get_xp_precision(xp, self.precision))  # (N, D, 1, C)
        if force_embedding is not None:
            x = x + xp.astype(force_embedding, get_xp_precision(xp, self.precision))
        if self.blocks:
            edge_cache = edge_cache_to_dtype(
                edge_cache, get_xp_precision(xp, self.precision)
            )
            x = self._forward_blocks(
                x, edge_cache, rad_feat_per_block, comm_dict=comm_dict
            )

        # === Step 11. Keep the owned-atom rows for the read-out ===
        # ``n_out_nodes`` is the owned-node count in the flattened layout
        # (``nf * nloc``). Single-domain: ``out_nloc == n_per_frame``, so this
        # equals the whole node set and the slice is a no-op. Parallel
        # (single-frame): it drops the trailing ghost rows that only fed message
        # passing -- LAMMPS orders owned atoms before ghosts, so they lead.
        n_out_nodes = nf * out_nloc
        x = x[:n_out_nodes]

        # === Step 12. Final l=0 output mixing ===
        x_scalar = self._apply_readout(x, n_out_nodes)

        # === Step 13. Reshape to (nf, nloc, channels) and return ===
        descriptor = xp.reshape(
            x_scalar, (nf, out_nloc, self.channels)
        )  # (nf, nloc, C)
        return xp.astype(descriptor, get_xp_precision(xp, "global")), x

    def _forward_blocks(
        self,
        x: Array,
        edge_cache: EdgeCache,
        radial_feat_per_block: list[Array],
        comm_dict: dict[str, Array] | None = None,
    ) -> Array:
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
        Array
            Output features with shape (N, D, 1, C).
        """
        if not self.use_full_attn_res and not self.use_block_attn_res:
            # === Fast path without descriptor-level attention residuals ===
            for i, block in enumerate(self.blocks):
                x = x[:, : self.node_ebed_dims[i], :, :]
                blk_radial = radial_feat_per_block[i]
                x, _, _, _ = block(
                    x,
                    edge_cache,
                    blk_radial,
                    comm_dict=self._block_comm(i, comm_dict),
                )
            return x

        n_node = x.shape[0]
        xp = array_api_compat.array_namespace(x)

        def node_l0_extractor(v: Array) -> Array:
            """Extract scalar features from global SO(3) layout."""
            return xp.reshape(v[:, 0, :, :], (n_node, self.channels))

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
            final_dim = self.node_readout_dim
            final_sources = [source[:, :final_dim, :, :] for source in unit_history]
            x = xp.astype(
                self.final_full_attn_res(
                    sources=final_sources,
                    scalar_extractor=node_l0_extractor,
                    current_x=x,
                ),
                get_xp_precision(xp, self.precision),
            )
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
        final_dim = self.node_readout_dim
        final_sources = [source[:, :final_dim, :, :] for source in block_history]
        x = xp.astype(
            self.final_block_attn_res(
                sources=final_sources,
                scalar_extractor=node_l0_extractor,
                current_x=x,
            ),
            get_xp_precision(xp, self.precision),
        )
        return x

    def _apply_readout(self, x: Array, n_rows: int) -> Array:
        """Fold the node tensor into the scalar (``l=0``) descriptor.

        Runs the ``readout_layers`` stack of equivariant residual read-out FFNs.
        ``so3_readout="none"`` feeds only the ``l=0`` slice; ``"glu"``/``"mlp"``
        feed the full ``(N, D, 1, C)`` node tensor so the SO(3) grid folds
        ``l>0`` geometry into ``l=0``. Each layer is an ``x + FFN(x)`` residual:
        the ``readout_pre_layers`` keep the full tensor so the geometry keeps
        folding, while the final ``output_ffn`` slices the ``l=0`` channel from
        its residual sum. Slicing the summed tensor rather than the FFN output
        keeps the saved degree-axis stride static under ``torch.compile`` dynamic
        shapes.

        Parameters
        ----------
        x
            Node features with shape ``(n_rows, D, 1, channels)``. With the
            blocks skipped (zero-block or empty-edge path) ``D`` is the initial
            degree; otherwise the pyramid has shrunk it, so the read-out slice to
            ``node_readout_dim`` is a no-op there.
        n_rows
            Number of node rows fed to the read-out.

        Returns
        -------
        Array
            Scalar descriptor with shape ``(n_rows, 1, 1, channels)``.
        """
        xp = array_api_compat.array_namespace(x)
        if self.so3_readout == "none":
            x_ro = xp.astype(
                xp.reshape(x[:, 0:1, :, :], (n_rows, 1, 1, self.channels)),
                get_xp_precision(xp, self.compute_precision),
            )
        else:
            x_ro = xp.astype(
                x[:, : self.node_readout_dim, :, :],
                get_xp_precision(xp, self.compute_precision),
            )
        for layer in self.readout_pre_layers:
            x_ro = x_ro + layer(x_ro)
        return (x_ro + self.output_ffn(x_ro))[:, 0:1, :, :]

    def _edge_quaternion(self, edge_cache: EdgeCache) -> Array:
        """
        Return the cached global->local edge quaternion, rebuilding if absent.

        Parameters
        ----------
        edge_cache : EdgeFeatureCache
            Per-edge cache. ``edge_quat`` is populated by the cache builder; the
            fallback covers caches produced without it.

        Returns
        -------
        Array
            Unit quaternions with shape (E, 4).
        """
        edge_quat = edge_cache.edge_quat
        if edge_quat is None:
            edge_len = safe_norm(edge_cache.edge_vec, self.eps)
            edge_quat = build_edge_quaternion(
                edge_cache.edge_vec,
                edge_len=edge_len,
                eps=self.eps,
            )
        return edge_quat

    def _build_gie_zonal_coupling(
        self,
        edge_cache: EdgeCache,
    ) -> Array | None:
        """
        Build node-level zonal coupling for GIE when node degrees exceed MP degrees.

        Returns
        -------
        Array or None
            Coupling with shape ``(E, D_node - 1)``. ``None`` is returned only
            when the full Wigner-D blocks are present and ``extra_node_l == 0``,
            in which case GIE gathers the coupling from the cache directly. When
            the blocks are skipped (all-Cartesian model) the full coupling is
            reconstructed from the edge quaternion via the m=0-only path.
        """
        if edge_cache.Dt_full is None:
            calc = self.gie_zonal_wigner_calc or self.wigner_calc
            return calc.forward_zonal(self._edge_quaternion(edge_cache), lmin=1)
        if self.gie_zonal_wigner_calc is None:
            return None
        xp = array_api_compat.array_namespace(edge_cache.Dt_full)
        device = array_api_compat.device(edge_cache.Dt_full)
        mp_row_count = self.mp_init_dim - 1
        mp_row_index = self.gie.non_scalar_row_index[:mp_row_count]
        mp_m0_col_index = self.gie.zonal_m0_col_index_for_row[:mp_row_count]
        dim_full = edge_cache.Dt_full.shape[-1]
        mp_coupling = xp.take(
            xp.reshape(edge_cache.Dt_full, (-1, dim_full * dim_full)),
            xp_asarray_nodetach(
                xp, mp_row_index * dim_full + mp_m0_col_index, device=device
            ),
            axis=1,
        )
        extra_coupling = self.gie_zonal_wigner_calc.forward_zonal(
            self._edge_quaternion(edge_cache),
            lmin=self.lmax + 1,
        )
        return xp.concat([mp_coupling, extra_coupling], axis=1)

    def _apply_charge_spin_embedding(
        self,
        type_ebed: Array,
        charge_spin: Array,
        *,
        nf: int,
        nloc: int,
    ) -> Array:
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
        Array
            Conditioned type embeddings with shape (nf * nloc, channels).
        """
        xp = array_api_compat.array_namespace(type_ebed, charge_spin)
        condition = self.charge_spin_embedding(xp.astype(charge_spin, type_ebed.dtype))
        condition = xp.broadcast_to(condition[:, None, :], (nf, nloc, self.channels))
        return type_ebed + xp.reshape(condition, type_ebed.shape)

    def _apply_spin_embedding(
        self,
        type_ebed: Array,
        spin: Array,
        atype_flat: Array,
        *,
        n_nodes: int,
    ) -> tuple[Array, Array]:
        """
        Inject the per-atom spin embedding into the node features.

        The l=0 magnitude scalar is added to the flattened type embedding so it
        propagates into the scalar backbone, the per-edge type features, and
        every block's radial features (exactly like the type embedding). The l=1
        direction coefficients are returned for the caller to add to the
        equivariant backbone after the geometric initial embedding.

        Parameters
        ----------
        type_ebed
            Flattened type embedding with shape (N, channels).
        spin
            Per-atom spin vectors with shape (nf, nloc, 3) or (N, 3).
        atype_flat
            Flattened local atom types with shape (N,).
        n_nodes
            Number of local nodes ``N = nf * nloc``.

        Returns
        -------
        tuple[Array, Array]
            The l=0-conditioned type embedding with shape (N, channels) and the
            packed l=1 direction coefficients with shape (N, 3, channels).
        """
        xp = array_api_compat.array_namespace(type_ebed, spin)
        scalar, vector = self.spin_embedding(xp.reshape(spin, (n_nodes, 3)), atype_flat)
        type_ebed = type_ebed + xp.astype(scalar, type_ebed.dtype)
        return type_ebed, vector

    def _edge_type_keep_mask(
        self,
        atype_flat: Array,
        src: Array,
        dst: Array,
    ) -> Array:
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
        Array
            Boolean mask with shape (E,), True means keep.
        """
        xp = array_api_compat.array_namespace(atype_flat, src, dst)
        if len(self.emask.exclude_types) == 0:
            return xp.ones_like(src, dtype=xp.bool)
        device = array_api_compat.device(atype_flat)
        type_i = xp.take(atype_flat, dst, axis=0)
        type_j = xp.take(atype_flat, src, axis=0)
        type_i = xp.where(type_i >= 0, type_i, self.ntypes)
        type_j = xp.where(type_j >= 0, type_j, self.ntypes)
        type_ij = type_i * (self.ntypes + 1) + type_j
        type_mask = xp_asarray_nodetach(xp, self.emask.type_mask[...], device=device)
        keep = xp.take(type_mask, xp.astype(type_ij, xp.int64), axis=0)
        return xp.astype(keep, xp.bool)

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
        """Parse and validate L/M schedules, setting self.l_schedule/m_schedule/lmax/mmax.

        An empty schedule (``n_blocks=0`` or ``l_schedule=[]``) is valid and
        selects the zero-block descriptor: no interaction blocks are built, only
        the initial SO(3) backbone (type embedding, optional env FiLM and GIE)
        followed by the final read-out. The backbone degree then derives from
        the configured ``lmax``/``mmax`` instead of the schedule endpoints.
        """
        # === L schedule ===
        if l_schedule is None:
            self.l_schedule = [int(lmax)] * int(n_blocks)
        else:
            self.l_schedule = [int(x) for x in l_schedule]
        if any(x < 0 for x in self.l_schedule):
            raise ValueError("`l_schedule` entries must be non-negative")
        if any(
            self.l_schedule[i] < self.l_schedule[i + 1]
            for i in range(len(self.l_schedule) - 1)
        ):
            raise ValueError("`l_schedule` must be non-increasing (pyramid schedule)")

        # The first entry sets the maximum degree; with zero blocks the backbone
        # degree falls back to the configured ``lmax``.
        self.lmax = int(self.l_schedule[0]) if self.l_schedule else int(lmax)
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
        if len(self.m_schedule) != len(self.l_schedule):
            raise ValueError("`m_schedule` must have the same length as `l_schedule`")
        if any(x < 0 for x in self.m_schedule):
            raise ValueError("`m_schedule` entries must be non-negative")
        if any(m > l for m, l in zip(self.m_schedule, self.l_schedule, strict=True)):
            raise ValueError(
                "`m_schedule` entries must satisfy `m_schedule[i] <= l_schedule[i]`"
            )

        self.mmax = (
            int(self.m_schedule[0])
            if self.m_schedule
            else (int(mmax) if mmax is not None else int(self.lmax))
        )

    def _init_node_l_schedules(self, extra_node_l: int) -> None:
        """Parse node degree schedules and resolve the canonical backbone degrees.

        The descriptor references three backbone degrees that must stay valid
        even with zero interaction blocks, so they are resolved here into
        scalars rather than indexed off the (possibly empty) schedules:

        - ``mp_init_lmax`` : message-passing degree at initialization, driving
          the Wigner-D calculator and the GIE message-passing coupling rows.
        - ``node_init_lmax`` : node backbone degree at initialization, driving
          the radial-embedding width, the initial state dimension, and GIE.
        - ``node_readout_lmax`` : node backbone degree fed to the read-out FFN.

        With blocks these equal ``l_schedule[0]``, ``node_l_schedule[0]`` and
        ``node_l_schedule[-1]``; with zero blocks all three collapse onto the
        configured ``lmax`` (plus ``extra_node_l`` on the node side), so the
        pyramid endpoints are never read from an empty list.
        """
        self.extra_node_l = int(extra_node_l)
        if self.extra_node_l < 0:
            raise ValueError("`extra_node_l` must be non-negative")
        self.node_l_schedule = [
            int(l_value) + self.extra_node_l for l_value in self.l_schedule
        ]
        self.node_ebed_dims = [
            get_so3_dim_of_lmax(l_value) for l_value in self.node_l_schedule
        ]

        # === Canonical backbone degrees (valid for any block count) ===
        self.mp_init_lmax = int(self.lmax)
        self.node_init_lmax = int(self.lmax) + self.extra_node_l
        self.node_readout_lmax = (
            int(self.node_l_schedule[-1]) if self.n_blocks > 0 else self.node_init_lmax
        )
        self.mp_init_dim = get_so3_dim_of_lmax(self.mp_init_lmax)
        self.node_init_dim = get_so3_dim_of_lmax(self.node_init_lmax)
        self.node_readout_dim = get_so3_dim_of_lmax(self.node_readout_lmax)

    def _canonicalize_charge_spin(
        self,
        charge_spin: Array | None,
        *,
        nf: int,
        dtype: Any,
        device: Any,
    ) -> Array | None:
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
        Array or None
            Tensor with shape (nf, 2) when condition embedding is enabled.
        """
        if self.charge_spin_embedding is None:
            return None
        if charge_spin is None:
            if self.default_chg_spin is None:
                raise ValueError("`charge_spin` is required for this SeZM descriptor.")
            default_chg_spin = np.asarray(self.default_chg_spin)
            xp = array_api_compat.array_namespace(default_chg_spin)
            charge_spin = xp.reshape(
                xp_asarray_nodetach(xp, default_chg_spin, dtype=dtype, device=device),
                (1, 2),
            )
        else:
            xp = array_api_compat.array_namespace(charge_spin)
            charge_spin = xp.astype(charge_spin, dtype)

        if charge_spin.ndim == 1:
            if math.prod(charge_spin.shape) != 2:
                raise ValueError("`charge_spin` must contain [charge, spin].")
            charge_spin = xp.reshape(charge_spin, (1, 2))
        elif charge_spin.ndim != 2 or charge_spin.shape[-1] != 2:
            raise ValueError("`charge_spin` must have shape (nf, 2).")

        if charge_spin.shape[0] == 1 and nf != 1:
            charge_spin = xp.broadcast_to(charge_spin, (nf, charge_spin.shape[-1]))
        elif charge_spin.shape[0] != nf:
            raise ValueError("`charge_spin` first dimension must match nframes.")
        return charge_spin

    def _block_comm(
        self,
        block_idx: int,
        comm_dict: dict[str, Array] | None,
    ) -> dict[str, Array] | None:
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
        raise NotImplementedError("share_params is not yet implemented for DescrptDPA4")

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

    def set_stat_mean_and_stddev(self, mean: Array, stddev: Array) -> None:
        """Set mean and stddev (interface compatibility, not used in forward)."""
        self.mean = mean
        self.stddev = stddev

    def get_stat_mean_and_stddev(self) -> tuple[Array, Array]:
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

    def _variables(self) -> dict[str, np.ndarray]:
        """Variables keyed by the pt ``state_dict`` key names."""
        variables: dict[str, np.ndarray] = {}
        # === Descriptor-level version and statistics buffers ===
        variables["version_tensor"] = np.asarray(self.version, dtype=np.float64)
        variables["mean"] = to_numpy_array(self.mean)
        variables["stddev"] = to_numpy_array(self.stddev)
        # === Type embedding (always present) ===
        for key, value in self.type_embedding.serialize()["@variables"].items():
            variables[f"type_embedding.{key}"] = value
        # === Frame charge/spin embedding (optional) ===
        if self.add_chg_spin_ebd:
            for key, value in self.charge_spin_embedding.serialize()[
                "@variables"
            ].items():
                variables[f"charge_spin_embedding.{key}"] = value
        # === Native per-atom spin embedding (optional) ===
        if self.spin_embedding is not None:
            for key, value in self.spin_embedding.serialize()["@variables"].items():
                variables[f"spin_embedding.{key}"] = value
        # === Environment FiLM stack (optional) ===
        if self.use_env_seed:
            for key, value in self.env_seed_embedding.serialize()["@variables"].items():
                variables[f"env_seed_embedding.{key}"] = value
            if self.edge_norm:
                for key, value in self.film_scale_norm.serialize()[
                    "@variables"
                ].items():
                    variables[f"film_scale_norm.{key}"] = value
                for key, value in self.film_shift_norm.serialize()[
                    "@variables"
                ].items():
                    variables[f"film_shift_norm.{key}"] = value
            variables["film_scale_strength_log"] = to_numpy_array(
                self.film_scale_strength_log
            )
            variables["film_shift_strength_log"] = to_numpy_array(
                self.film_shift_strength_log
            )
        # === Radial basis and shared radial embedding ===
        for key, value in self.radial_basis.serialize()["@variables"].items():
            variables[f"radial_basis.{key}"] = value
        for key, value in self.radial_embedding.serialize()["@variables"].items():
            variables[f"radial_embedding.net.{key}"] = value
        # === Wigner-D static buffers ===
        # The Wigner-D index/sign tables are derived constants with no trainable
        # parameters. They are emitted to keep the ``state_dict`` key set complete
        # and are rebuilt at construction on load (see ``_load_variables``).
        variables["wigner_calc.l1_perm"] = to_numpy_array(self.wigner_calc.l1_perm)
        variables["wigner_calc.l1_sign_outer"] = to_numpy_array(
            self.wigner_calc.l1_sign_outer
        )
        if self.gie_zonal_wigner_calc is not None:
            variables["gie_zonal_wigner_calc.l1_perm"] = to_numpy_array(
                self.gie_zonal_wigner_calc.l1_perm
            )
            variables["gie_zonal_wigner_calc.l1_sign_outer"] = to_numpy_array(
                self.gie_zonal_wigner_calc.l1_sign_outer
            )
        # === Geometric initial embedding index buffers (optional) ===
        if self.use_gie:
            variables["gie.non_scalar_row_index"] = to_numpy_array(
                self.gie.non_scalar_row_index
            )
            variables["gie.zonal_m0_col_index_for_row"] = to_numpy_array(
                self.gie.zonal_m0_col_index_for_row
            )
            variables["gie.radial_slot_index_for_row"] = to_numpy_array(
                self.gie.radial_slot_index_for_row
            )
        # === Interaction blocks ===
        for i, block in enumerate(self.blocks):
            for key, value in block._variables().items():
                variables[f"blocks.{i}.{key}"] = value
        # === Descriptor-level attention residuals (optional, mutually exclusive) ===
        if self.use_full_attn_res:
            for key, value in self.final_full_attn_res.serialize()[
                "@variables"
            ].items():
                variables[f"final_full_attn_res.{key}"] = value
        if self.use_block_attn_res:
            for key, value in self.final_block_attn_res.serialize()[
                "@variables"
            ].items():
                variables[f"final_block_attn_res.{key}"] = value
        # === Read-out pre-layers (optional) ===
        for i, layer in enumerate(self.readout_pre_layers):
            for key, value in layer._variables().items():
                variables[f"readout_pre_layers.{i}.{key}"] = value
        # === Output FFN ===
        for key, value in self.output_ffn._variables().items():
            variables[f"output_ffn.{key}"] = value
        return variables

    def _load_variables(self, variables: dict[str, np.ndarray]) -> None:
        """Load variables keyed by the pt ``state_dict`` key names."""

        def take_prefix(prefix: str) -> dict[str, np.ndarray]:
            return {
                key[len(prefix) :]: value
                for key, value in variables.items()
                if key.startswith(prefix)
            }

        def load(module: Any, prefix: str) -> Any:
            data = module.serialize()
            data["@variables"] = take_prefix(prefix)
            return type(module).deserialize(data)

        prec = PRECISION_DICT[self.precision]
        compute_prec = PRECISION_DICT[self.compute_precision]
        # === Descriptor-level statistics buffers ===
        # ``version_tensor`` and the derived Wigner-D / GIE index tables are
        # transient: they are rebuilt at construction, so they are ignored here.
        self.mean = np.asarray(variables["mean"], dtype=prec)
        self.stddev = np.asarray(variables["stddev"], dtype=prec)
        # === Type embedding (always present) ===
        self.type_embedding = load(self.type_embedding, "type_embedding.")
        # === Frame charge/spin embedding (optional) ===
        if self.add_chg_spin_ebd:
            self.charge_spin_embedding = load(
                self.charge_spin_embedding, "charge_spin_embedding."
            )
        # === Native per-atom spin embedding (optional) ===
        if self.spin_embedding is not None:
            self.spin_embedding = load(self.spin_embedding, "spin_embedding.")
        # === Environment FiLM stack (optional) ===
        if self.use_env_seed:
            self.env_seed_embedding = load(
                self.env_seed_embedding, "env_seed_embedding."
            )
            if self.edge_norm:
                self.film_scale_norm = load(self.film_scale_norm, "film_scale_norm.")
                self.film_shift_norm = load(self.film_shift_norm, "film_shift_norm.")
            self.film_scale_strength_log = np.asarray(
                variables["film_scale_strength_log"], dtype=compute_prec
            )
            self.film_shift_strength_log = np.asarray(
                variables["film_shift_strength_log"], dtype=compute_prec
            )
        # === Radial basis and shared radial embedding ===
        self.radial_basis = load(self.radial_basis, "radial_basis.")
        self.radial_embedding = load(self.radial_embedding, "radial_embedding.net.")
        # === Interaction blocks ===
        for i, block in enumerate(self.blocks):
            block._load_variables(take_prefix(f"blocks.{i}."))
        # === Descriptor-level attention residuals (optional, mutually exclusive) ===
        if self.use_full_attn_res:
            self.final_full_attn_res = load(
                self.final_full_attn_res, "final_full_attn_res."
            )
        if self.use_block_attn_res:
            self.final_block_attn_res = load(
                self.final_block_attn_res, "final_block_attn_res."
            )
        # === Read-out pre-layers (optional) ===
        for i, layer in enumerate(self.readout_pre_layers):
            layer._load_variables(take_prefix(f"readout_pre_layers.{i}."))
        # === Output FFN ===
        self.output_ffn._load_variables(take_prefix("output_ffn."))

    def serialize(self) -> dict[str, Any]:
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
                "edge_norm": self.edge_norm,
                "use_env_seed": self.use_env_seed,
                "random_gamma": self.random_gamma,
                "edge_cartesian": self.edge_cartesian,
                "node_cartesian": self.node_cartesian,
                "so2_norm": self.so2_norm,
                "mixing_layers": self.mixing_layers,
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
                "readout_layers": self.readout_layers,
                "lebedev_quadrature": self.lebedev_quadrature,
                "activation_function": self.activation_function,
                "glu_activation": self.glu_activation,
                "precision": np.dtype(PRECISION_DICT[self.precision]).name,
                "mlp_bias": self.mlp_bias,
                "exclude_types": self.exclude_types,
                "eps": self.eps,
                "trainable": self.trainable,
                "seed": self.seed,
                "inner_clamp_r_inner": self.inner_clamp_r_inner,
                "inner_clamp_r_outer": self.inner_clamp_r_outer,
                "add_chg_spin_ebd": self.add_chg_spin_ebd,
                "default_chg_spin": self.default_chg_spin,
                "use_spin": self.use_spin,
            },
            "@variables": self._variables(),
            "env_mat": EnvMat(self.rcut, self.rcut, self.eps).serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> DescrptDPA4:
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "Descriptor":
            raise ValueError(f"Invalid class for DescrptDPA4: {data_cls}")
        type_val = data.pop("type")
        if type_val not in ("SeZM", "sezm", "dpa4"):
            raise ValueError(f"Invalid type for DescrptDPA4: {type_val}")
        version = float(data.pop("@version"))
        check_version_compatibility(version, cls.LATEST_VERSION, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        data.pop("env_mat", None)
        config.pop("s2_grid_resolution", None)
        obj = cls(**config)
        obj.version = version
        obj._load_variables(variables)
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
