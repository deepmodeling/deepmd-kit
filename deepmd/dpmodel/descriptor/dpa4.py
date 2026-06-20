# SPDX-License-Identifier: LGPL-3.0-or-later
"""
DPA4 (SeZM) descriptor: dpmodel (array-API) backend.

This is the dpmodel port of ``deepmd.pt.model.descriptor.sezm.DescrptSeZM``.
It orchestrates the dpa4_nn building blocks on the padded, frame-explicit
edge layout (``E = nf * nloc * nnei``; no ``torch.nonzero``-style sparse
edge extraction anywhere; see ``dpa4_nn.edge_cache``).

Scope notes (vs pt):

- Only the standard DeePMD ``call(coord_ext, atype_ext, nlist, mapping)``
  path is ported. The pt-only paths (sparse ``edge_index`` inputs,
  ``forward_with_edges``, zone bridging / InnerClamp, charge/spin condition
  embedding, AMP autocast) are out of core scope; out-of-core construction
  flags raise ``NotImplementedError`` at ``__init__`` (either here or in the
  owning submodule).
- ``random_gamma`` is a training-only augmentation in pt
  (``random_gamma and self.training``); dpmodel evaluates in inference mode,
  so the roll is never applied (the config value is still serialized).
- ``use_amp`` is accepted and ignored: it is a pt-runtime (CUDA autocast)
  switch with no dpmodel counterpart.
"""

from __future__ import (
    annotations,
)

import logging
import math
from typing import (
    TYPE_CHECKING,
    Any,
    NoReturn,
)

import array_api_compat
import numpy as np

log = logging.getLogger(__name__)

# Warn at most once per process for backend-ignored switches (keyed by name).
_WARNED_ONCE: set[str] = set()

from deepmd.dpmodel import (
    NativeOP,
)
from deepmd.dpmodel.array_api import (
    xp_asarray_nodetach,
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
from .dpa4_nn.block import (
    SeZMInteractionBlock,
)
from .dpa4_nn.edge_cache import (
    EdgeCache,
    build_edge_cache,
)
from .dpa4_nn.embedding import (
    EnvironmentInitialEmbedding,
    GeometricInitialEmbedding,
    SeZMTypeEmbedding,
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
    C3CutoffEnvelope,
    RadialBasis,
    RadialMLP,
)
from .dpa4_nn.utils import (
    get_promoted_dtype,
)
from .dpa4_nn.wignerd import (
    WignerDCalculator,
)

if TYPE_CHECKING:
    from deepmd.dpmodel.array_api import (
        Array,
    )
    from deepmd.utils.data_system import (
        DeepmdDataSystem,
    )
    from deepmd.utils.path import (
        DPPath,
    )

ATTN_RES_MODES = ("none", "independent", "dependent")


@BaseDescriptor.register("SeZM")
@BaseDescriptor.register("sezm")
@BaseDescriptor.register("DPA4")
@BaseDescriptor.register("dpa4")
class DescrptDPA4(NativeOP, BaseDescriptor):
    """
    DPA4 (SeZM) descriptor, dpmodel backend.

    See the pt ``DescrptSeZM`` docstring
    (``deepmd/pt/model/descriptor/sezm.py``) for the full per-parameter
    description; the constructor mirrors the pt signature and defaults
    exactly. Parameters whose machinery is not ported to dpmodel raise
    ``NotImplementedError`` at construction (some directly here, the rest
    delegated to the owning submodule, e.g. ``layer_scale`` and the
    ``*_attn_res`` / SO(2) attention projection flags).

    Execution outline (pt ``forward`` standard path):

    1. Type embedding and pair-exclusion keep mask.
    2. ``build_edge_cache`` once (geometry, envelope, RBF, Wigner-D) on the
       padded edge layout.
    3. Radial features once; optional environment FiLM seeding and geometric
       initial embedding.
    4. ``SeZMInteractionBlock`` stack with the per-block l/m schedules.
    5. Final scalar (l=0) FFN readout to ``(nf, nloc, channels)``.
    """

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
        # version >= 1.1 O(1) floor for the envelope-squared degree
        # normalization (see pt sezm.py).
        self.deg_norm_floor = 0.25

        if isinstance(sel, int):
            sel = [sel]
        self.ntypes = int(ntypes)
        self.sel = [int(x) for x in sel]
        self.type_map = type_map
        self.nnei = int(sum(self.sel))

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
                "sandwich_norm must be a list[bool] of length 4: "
                "[so2_pre, so2_post, ffn_pre, ffn_post]"
            )
        self.sandwich_norm = [bool(x) for x in sandwich_norm]
        (
            self.so2_pre_norm,
            self.so2_post_norm,
            self.ffn_pre_norm,
            self.ffn_post_norm,
        ) = self.sandwich_norm
        if s2_activation is None:
            s2_activation = [False, True]
        if not isinstance(s2_activation, list) or len(s2_activation) != 2:
            raise ValueError(
                "`s2_activation` must be a list[bool] of length 2: "
                "[so2_activation, ffn_activation]"
            )
        if any(not isinstance(flag, bool) for flag in s2_activation):
            raise ValueError(
                "`s2_activation` must be a list[bool] of length 2: "
                "[so2_activation, ffn_activation]"
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
                "`lebedev_quadrature` must be a bool or a list[bool] of length 2: "
                "[so2_quadrature, ffn_quadrature]"
            )
        if any(not isinstance(flag, bool) for flag in lebedev_quadrature):
            raise ValueError(
                "`lebedev_quadrature` must be a bool or a list[bool] of length 2: "
                "[so2_quadrature, ffn_quadrature]"
            )
        self.lebedev_quadrature = list(lebedev_quadrature)
        # The tensor-product (e3nn-style) sphere grid is not ported to
        # dpmodel; only the packaged Lebedev quadrature path exists
        # (see dpa4_nn.projection).
        if not all(self.lebedev_quadrature):
            raise NotImplementedError(
                "lebedev_quadrature entries with False (tensor-product S2 "
                "grid) are not ported to dpmodel"
            )
        self.activation_function = str(activation_function)
        self.glu_activation = bool(glu_activation)

        # === Split effective activation config by branch (pt sezm.py) ===
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
        # Geometry / seeding paths run in promoted ("fp32+") precision (pt
        # uses compute_dtype = get_promoted_dtype(dtype) there).
        self.compute_precision = str(
            np.dtype(get_promoted_dtype(PRECISION_DICT[self.precision])).name
        )
        self.mlp_bias = bool(mlp_bias)
        self.layer_scale = bool(layer_scale)
        # pt-runtime-only switch (CUDA bfloat16 autocast during training);
        # accepted for config compatibility and ignored by dpmodel.
        self.use_amp = bool(use_amp)
        if self.use_amp and "use_amp" not in _WARNED_ONCE:
            log.warning(
                "`use_amp` has no effect on the dpmodel/pt_expt backend "
                "(it is a pt-runtime CUDA autocast switch); ignoring it."
            )
            _WARNED_ONCE.add("use_amp")
        self.trainable = bool(trainable)
        self.seed = seed
        self.random_gamma = bool(random_gamma)
        self.add_chg_spin_ebd = bool(add_chg_spin_ebd)
        if self.add_chg_spin_ebd:
            raise NotImplementedError(
                "add_chg_spin_ebd=True (ChargeSpinEmbedding) is not ported to dpmodel"
            )
        if default_chg_spin is not None and len(default_chg_spin) != 2:
            raise ValueError("`default_chg_spin` must contain [charge, spin].")
        self.default_chg_spin = (
            None if default_chg_spin is None else [float(x) for x in default_chg_spin]
        )

        # === Zone bridging (InnerClamp + BridgingSwitch): not ported ===
        self.inner_clamp_r_inner = (
            float(inner_clamp_r_inner) if inner_clamp_r_inner is not None else None
        )
        self.inner_clamp_r_outer = (
            float(inner_clamp_r_outer) if inner_clamp_r_outer is not None else None
        )
        if self.inner_clamp_r_inner is not None or self.inner_clamp_r_outer is not None:
            raise NotImplementedError(
                "inner_clamp_r_inner/inner_clamp_r_outer (zone bridging) are "
                "not ported to dpmodel"
            )

        # === Env seed derived dimensions (pt sezm.py) ===
        self.use_env_seed = bool(use_env_seed)
        self.env_seed_embed_dim = min(self.channels, 128)
        self.env_seed_type_dim = min(32, max(8, self.channels // 4))
        axis_dim = 4 if self.env_seed_embed_dim < 64 else 8
        self.env_seed_axis_dim = min(axis_dim, max(1, self.env_seed_embed_dim - 1))
        rbf_out_dim = max(32, self.env_seed_embed_dim - 2 * self.env_seed_type_dim)
        g_in_dim = rbf_out_dim + 2 * self.env_seed_type_dim
        self.env_seed_hidden_dim = min(256, max(2 * self.env_seed_embed_dim, g_in_dim))

        # === Deterministic seed split (same indices as pt) ===
        seed_type_embedding = child_seed(self.seed, 0)
        seed_blocks = child_seed(self.seed, 1)
        seed_out = child_seed(self.seed, 2)
        seed_radial_embedding = child_seed(self.seed, 3)
        seed_env_seed = child_seed(self.seed, 4)

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
            self.ffn_neurons, glu_activation=self.ffn_glu_activation
        )
        self.out_ffn_neurons = self._resolve_ffn_neurons(
            self.ffn_neurons, glu_activation=self.out_glu_activation
        )
        self.grid_mlp = self._broadcast_grid_setting(
            grid_mlp, name="grid_mlp", cast=bool
        )
        self.grid_branch = self._broadcast_grid_setting(
            grid_branch, name="grid_branch", cast=int, non_negative=True
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

        # === Type embedding (fp32+) ===
        self.type_embedding = SeZMTypeEmbedding(
            ntypes=self.ntypes,
            embed_dim=self.channels,
            precision=self.compute_precision,
            seed=seed_type_embedding,
            trainable=self.trainable,
        )

        # === Env FiLM embedding (optional, fp32+) ===
        compute_np_prec = PRECISION_DICT[self.compute_precision]
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
                    precision=self.compute_precision,
                    trainable=self.trainable,
                    seed=seed_env_seed,
                )
            )
            self.film_scale_norm: ScalarRMSNorm | None = ScalarRMSNorm(
                channels=self.channels,
                n_focus=1,
                eps=self.eps,
                precision=self.compute_precision,
                trainable=self.trainable,
            )
            self.film_shift_norm: ScalarRMSNorm | None = ScalarRMSNorm(
                channels=self.channels,
                n_focus=1,
                eps=self.eps,
                precision=self.compute_precision,
                trainable=self.trainable,
            )
            film_strength_init = 0.01
            self.film_scale_strength_log: np.ndarray | None = np.full(
                (1,), math.log(film_strength_init), dtype=compute_np_prec
            )
            self.film_shift_strength_log: np.ndarray | None = np.full(
                (1,), math.log(film_strength_init), dtype=compute_np_prec
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
            precision=self.compute_precision,
            exponent=self.env_exp[0],
        )

        # === Shared radial embedding: RBF -> per-l radial features (fp32+) ===
        radial_out_dim = (self.node_l_schedule[0] + 1) * self.channels
        radial_mlp_layers = [self.n_radial, *self.radial_mlp, radial_out_dim]
        self.radial_embedding = RadialMLP(
            radial_mlp_layers,
            activation_function=self.activation_function,
            precision=self.compute_precision,
            trainable=self.trainable,
            seed=seed_radial_embedding,
        )

        # === C^3 cutoff envelope for edge weight ===
        self.edge_envelope = C3CutoffEnvelope(
            self.rcut, self.env_exp[1], precision=self.compute_precision
        )

        wigner_lmax = self.l_schedule[0]
        self.wigner_calc = WignerDCalculator(
            wigner_lmax, eps=self.eps, precision=self.compute_precision
        )

        # === Geometric initial embedding (optional, fp32+) ===
        self.use_gie = self.use_env_seed and self.node_l_schedule[0] > 0
        if self.use_gie:
            self.gie: GeometricInitialEmbedding | None = GeometricInitialEmbedding(
                lmax=self.node_l_schedule[0],
                channels=self.channels,
                precision=self.compute_precision,
            )
            if self.extra_node_l > 0:
                self.gie_zonal_wigner_calc: WignerDCalculator | None = (
                    WignerDCalculator(
                        self.node_l_schedule[0],
                        eps=self.eps,
                        precision=self.compute_precision,
                    )
                )
            else:
                self.gie_zonal_wigner_calc = None
        else:
            self.gie = None
            self.gie_zonal_wigner_calc = None

        # === Interaction blocks ===
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
                    precision=self.precision,
                    seed=child_seed(seed_blocks, block_idx),
                    trainable=self.trainable,
                )
            )
        self.blocks = blocks

        # === Final FFN for l=0 output mixing (fp32+) ===
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
            s2_activation=False,
            ffn_so3_grid=self.so3_readout != "none",
            activation_function=self.out_activation_function,
            glu_activation=self.out_glu_activation,
            mlp_bias=self.mlp_bias,
            precision=self.compute_precision,
            trainable=self.trainable,
            seed=seed_out,
        )

        # === Statistics buffers (interface compatibility, unused in call) ===
        model_np_prec = PRECISION_DICT[self.precision]
        self.mean = np.zeros((0,), dtype=model_np_prec)
        self.stddev = np.ones((0,), dtype=model_np_prec)

    # =========================================================================
    # Construction helpers (mirroring pt)
    # =========================================================================

    @staticmethod
    def _broadcast_grid_setting(
        value: bool | int | list[bool] | list[int],
        *,
        name: str,
        cast: type,
        non_negative: bool = False,
    ) -> list:
        """Normalize a grid-path setting to ``[node_wise, message_node, ffn]``."""
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

    def _resolve_ffn_neurons(self, ffn_neurons: int, *, glu_activation: bool) -> int:
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
        """Parse and validate L/M schedules (pt ``_init_lm_schedules``)."""
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

    def reinit_exclude(
        self, exclude_types: list[tuple[int, int]] | None = None
    ) -> None:
        if exclude_types is None:
            exclude_types = []
        self.exclude_types = exclude_types
        self.emask = PairExcludeMask(self.ntypes, exclude_types=exclude_types)

    # =========================================================================
    # Forward
    # =========================================================================

    def call(
        self,
        coord_ext: Array,
        atype_ext: Array,
        nlist: Array,
        mapping: Array | None = None,
        fparam: Array | None = None,
        comm_dict: dict | None = None,
        charge_spin: Array | None = None,
    ) -> tuple[Array, Any, Any, Any, Any]:
        """Compute the DPA4 descriptor.

        Parameters
        ----------
        coord_ext
            Extended coordinates with shape (nf, nall*3) or (nf, nall, 3).
        atype_ext
            Extended atom types with shape (nf, nall).
        nlist
            Neighbor list with shape (nf, nloc, nnei); -1 marks padding.
        mapping
            Extended-to-local mapping with shape (nf, nall), or None when the
            neighbor indices are already local.
        fparam
            Frame parameters; not used by DPA4 (interface compatibility).
        comm_dict
            MPI communication metadata; not used (interface compatibility).
        charge_spin
            Charge/spin embedding input; must be None since
            ``add_chg_spin_ebd=True`` is rejected at construction
            (interface compatibility with ``DPAtomicModel``).

        Returns
        -------
        descriptor
            Scalar descriptor with shape (nf, nloc, channels).
        rot_mat, g2, h2, sw
            ``None`` placeholders (pt returns empty tensors for these).
        """
        xp = array_api_compat.array_namespace(coord_ext, atype_ext, nlist)
        nf, nloc, nnei = nlist.shape
        nall = xp.reshape(coord_ext, (nf, -1)).shape[1] // 3
        extended_coord = xp.reshape(coord_ext, (nf, nall, 3))
        extended_coord = xp.astype(
            extended_coord, get_xp_precision(xp, self.compute_precision)
        )
        n_nodes = nf * nloc

        # === Step 1. Excluded type pairs (keep mask, True means keep) ===
        # The dpmodel PairExcludeMask returns an int mask; build_edge_cache
        # expects a boolean keep mask.
        pair_keep_mask = self.emask.build_type_exclude_mask(nlist, atype_ext) != 0

        # === Step 2. Type embedding (l=0) ===
        # Use ``xp_take_first_n`` (torch.index_select) rather than a plain
        # ``[:, :nloc]`` slice: the slice makes torch.export emit a spurious
        # ``Ne(nall, nloc)`` contiguity guard that breaks the ``nall == nloc``
        # (NoPBC, no ghost atoms) case in the compiled .pt2 artifact.
        atype_loc = xp_take_first_n(atype_ext, 1, nloc)
        type_ebed = xp.reshape(
            self.type_embedding(atype_loc), (n_nodes, self.channels)
        )  # (N, C)

        # === Step 3. Build edge cache once (geometry + RBF + Wigner-D) ===
        # Random local-Z roll is a training-only augmentation in pt; the
        # dpmodel descriptor evaluates in inference mode, so gamma is fixed.
        edge_cache = build_edge_cache(
            type_ebed=type_ebed,
            extended_coord=extended_coord,
            nlist=nlist,
            mapping=mapping,
            pair_keep_mask=pair_keep_mask,
            eps=self.eps,
            deg_norm_floor=(self.deg_norm_floor if self.version >= 1.1 else self.eps),
            edge_envelope=self.edge_envelope,
            radial_basis=self.radial_basis,
            n_radial=self.n_radial,
            random_gamma=False,
            wigner_calc=self.wigner_calc,
        )

        # === Step 4. Compute radial features once (fp32+) ===
        # Padded layout: E = nf * nloc * nnei is shape-determined, so there is
        # no pt-style empty-edge special case.
        radial_feat = xp.reshape(
            self.radial_embedding(edge_cache.edge_rbf),
            (-1, self.node_l_schedule[0] + 1, self.channels),
        )  # (E, node_lmax+1, C)
        if self.version >= 1.1:
            radial_feat = radial_feat * xp.reshape(edge_cache.edge_env, (-1, 1, 1))

        # === Step 5. Env FiLM conditioning (optional, fp32+) ===
        x0_out = type_ebed  # (N, C)
        if self.use_env_seed:
            atype_flat = xp.reshape(atype_loc, (-1,))
            film = self.env_seed_embedding(
                edge_cache=edge_cache,
                atype_flat=atype_flat,
                n_nodes=n_nodes,
            )  # (N, 2*C)
            scale_logits = film[:, : self.channels]
            shift_logits = film[:, self.channels :]
            scale_hat = self.film_scale_norm(scale_logits)
            shift_hat = self.film_shift_norm(shift_logits)
            device = array_api_compat.device(scale_hat)
            scale_strength = xp.exp(
                xp_asarray_nodetach(xp, self.film_scale_strength_log, device=device)
            )
            shift_strength = xp.exp(
                xp_asarray_nodetach(xp, self.film_shift_strength_log, device=device)
            )
            scale = 1.0 + scale_strength * xp.tanh(scale_hat)
            shift = shift_strength * xp.tanh(shift_hat)
            x0_out = type_ebed * scale + shift

        # === Step 6. Build backbone l=0 features ===
        # pt scatters x0_out into x[:, 0, 0, :] of a zeros tensor; here this
        # is a concat with zero rows for l >= 1 (no fancy __setitem__).
        ebed_dim_0 = self.node_ebed_dims[0]
        x = xp.concat(
            [
                x0_out[:, None, :],
                xp.zeros(
                    (n_nodes, ebed_dim_0 - 1, self.channels),
                    dtype=x0_out.dtype,
                    device=array_api_compat.device(x0_out),
                ),
            ]
            if ebed_dim_0 > 1
            else [x0_out[:, None, :]],
            axis=1,
        )  # (N, D, C)

        # === Step 7. Geometric initial embedding (fp32+) ===
        if self.use_gie:
            zonal_coupling = self._build_gie_zonal_coupling(edge_cache)
            x = x + self.gie(
                n_nodes=n_nodes,
                edge_cache=edge_cache,
                radial_feat=radial_feat[:, 1:, :],
                zonal_coupling=zonal_coupling,
            )
        x = x[:, :, None, :]  # (N, D, 1, C)

        # === Step 8. Fuse edge type features into radial features ===
        radial_feat = radial_feat + edge_cache.edge_type_feat[:, None, :]
        rad_feat_per_block = [
            radial_feat[:, :rad_len, :] for rad_len in self.rad_sizes_per_block
        ]

        # === Step 9. Run interaction blocks (residual baseline path) ===
        for i, block in enumerate(self.blocks):
            x = x[:, : self.node_ebed_dims[i], :, :]
            x = block(x, edge_cache, rad_feat_per_block[i])[0]

        # === Step 10. Final l=0 output mixing ===
        # ``none`` feeds the l=0 slice only; ``glu``/``mlp`` feed the full
        # (N, D, 1, C) node tensor so the SO(3) grid folds l>0 into l=0. The
        # residual is added on the full coefficient tensor before extracting
        # l=0 to mirror pt.
        compute_prec = get_xp_precision(xp, self.compute_precision)
        if self.so3_readout == "none":
            ffn_in = xp.astype(
                xp.reshape(x[:, 0:1, :, :], (n_nodes, 1, 1, self.channels)),
                compute_prec,
            )  # (N, 1, 1, C)
        else:
            ffn_in = xp.astype(x, compute_prec)  # (N, D, 1, C)
        x_scalar = (ffn_in + self.output_ffn(ffn_in))[:, 0:1, :, :]

        # === Step 11. Reshape and return ===
        descriptor = xp.reshape(x_scalar, (nf, nloc, self.channels))
        descriptor = xp.astype(descriptor, get_xp_precision(xp, "global"))
        return descriptor, None, None, None, None

    def _build_gie_zonal_coupling(self, edge_cache: EdgeCache) -> Any:
        """
        Build node-level zonal coupling for GIE when node degrees exceed MP
        degrees (pt ``_build_gie_zonal_coupling``).

        Returns ``None`` when ``extra_node_l == 0``, letting GIE gather from
        the MP Wigner-D cache.
        """
        if self.gie_zonal_wigner_calc is None:
            return None
        xp = array_api_compat.array_namespace(edge_cache.edge_quat)
        device = array_api_compat.device(edge_cache.edge_quat)
        n_edge = edge_cache.dst.shape[0]
        mp_row_count = self.ebed_dims[0] - 1
        mp_rows = self.gie.non_scalar_row_index[:mp_row_count]
        mp_cols = self.gie.zonal_m0_col_index_for_row[:mp_row_count]
        Dt_full = edge_cache.Dt_full
        dim_full = Dt_full.shape[-1]
        flat_index = xp_asarray_nodetach(
            xp, mp_rows * dim_full + mp_cols, device=device
        )
        mp_coupling = xp.take(
            xp.reshape(Dt_full, (n_edge, dim_full * dim_full)),
            flat_index,
            axis=1,
        )  # (E, D_mp - 1)
        extra_coupling = self.gie_zonal_wigner_calc.forward_zonal(
            edge_cache.edge_quat,
            lmin=self.lmax + 1,
        )
        return xp.concat([mp_coupling, extra_coupling], axis=1)

    # =========================================================================
    # DeePMD descriptor interface
    # =========================================================================

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

    def get_dim_out(self) -> int:
        return self.channels

    def get_dim_emb(self) -> int:
        return self.get_dim_out()

    def mixed_types(self) -> bool:
        """DPA4 uses SeZMTypeEmbedding, no type-distinguished nlist needed."""
        return True

    def has_message_passing(self) -> bool:
        return bool(len(self.blocks) > 0 and self.lmax > 0)

    def has_message_passing_across_ranks(self) -> bool:
        return self.has_message_passing()

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
    ) -> NoReturn:
        """Parameter sharing is a pt-backend training feature."""
        raise NotImplementedError

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat: Any = None
    ) -> NoReturn:
        raise NotImplementedError("change_type_map is not supported for SeZM")

    def enable_compression(
        self,
        min_nbor_dist: float,
        table_extrapolate: float = 5,
        table_stride_1: float = 0.01,
        table_stride_2: float = 0.1,
        check_frequency: int = -1,
    ) -> NoReturn:
        raise NotImplementedError("Compression is unsupported for SeZM.")

    # === Statistics interface (interface compatibility only) ===
    # SeZM normalizes with learnable RMS norms; mean/stddev are kept only for
    # interface and checkpoint-format compatibility (see pt sezm.py).

    def compute_input_stats(
        self, merged: list[dict], path: DPPath | None = None
    ) -> None:
        """No-op: statistics are not used by the DPA4 forward pass."""

    def set_stat_mean_and_stddev(self, mean: Array, stddev: Array) -> None:
        """Set mean and stddev (interface compatibility, unused in call)."""
        self.mean = mean
        self.stddev = stddev

    def get_stat_mean_and_stddev(self) -> tuple[Array, Array]:
        """Get mean and stddev (interface compatibility, unused in call)."""
        return self.mean, self.stddev

    # =========================================================================
    # Serialization (pt state_dict-key compatible)
    # =========================================================================

    def _variables(self) -> dict[str, np.ndarray]:
        """Variables keyed exactly by the pt ``state_dict()`` key names."""
        model_np_prec = PRECISION_DICT[self.precision]
        variables: dict[str, np.ndarray] = {
            # pt interface-compatibility buffers
            "version_tensor": np.asarray(self.version, dtype=np.float64),
            "_empty_tensor": np.zeros((0,), dtype=np.float64),
            "mean": to_numpy_array(self.mean).astype(model_np_prec),
            "stddev": to_numpy_array(self.stddev).astype(model_np_prec),
        }

        def add(prefix: str, sub_vars: dict[str, Any]) -> None:
            for key, value in sub_vars.items():
                variables[f"{prefix}{key}"] = to_numpy_array(value)

        add("type_embedding.", self.type_embedding.serialize()["@variables"])
        if self.use_env_seed:
            add(
                "env_seed_embedding.",
                self.env_seed_embedding.serialize()["@variables"],
            )
            add("film_scale_norm.", self.film_scale_norm.serialize()["@variables"])
            add("film_shift_norm.", self.film_shift_norm.serialize()["@variables"])
            variables["film_scale_strength_log"] = to_numpy_array(
                self.film_scale_strength_log
            )
            variables["film_shift_strength_log"] = to_numpy_array(
                self.film_shift_strength_log
            )
        add("radial_basis.", self.radial_basis.serialize()["@variables"])
        add("radial_embedding.net.", self.radial_embedding.serialize()["@variables"])

        # Static pt WignerDCalculator buffers (rebuilt at construction here;
        # emitted so pt's strict load_state_dict finds every key).
        def wigner_buffers(calc: WignerDCalculator) -> dict[str, np.ndarray]:
            return {
                "l1_perm": np.asarray([1, 2, 0], dtype=np.int64),
                "l1_sign_outer": to_numpy_array(calc.l1_sign_outer).astype(np.float64),
            }

        add("wigner_calc.", wigner_buffers(self.wigner_calc))
        if self.use_gie:
            add(
                "gie.",
                {
                    "non_scalar_row_index": self.gie.non_scalar_row_index,
                    "zonal_m0_col_index_for_row": self.gie.zonal_m0_col_index_for_row,
                    "radial_slot_index_for_row": self.gie.radial_slot_index_for_row,
                },
            )
            if self.gie_zonal_wigner_calc is not None:
                add(
                    "gie_zonal_wigner_calc.",
                    wigner_buffers(self.gie_zonal_wigner_calc),
                )
        for i, block in enumerate(self.blocks):
            add(f"blocks.{i}.", block._variables())
        add("output_ffn.", self.output_ffn._variables())
        return variables

    def _load_variables(self, variables: dict[str, Any]) -> None:
        """Load variables keyed by the pt ``state_dict()`` key names."""
        variables = dict(variables)

        def take_prefix(prefix: str) -> dict[str, Any]:
            sub = {
                key[len(prefix) :]: value
                for key, value in variables.items()
                if key.startswith(prefix)
            }
            for key in list(variables):
                if key.startswith(prefix):
                    del variables[key]
            return sub

        # Transient / static pt buffers rebuilt at construction.
        for key in ("version_tensor", "_empty_tensor"):
            variables.pop(key, None)
        take_prefix("wigner_calc.")
        take_prefix("gie.")
        take_prefix("gie_zonal_wigner_calc.")

        model_np_prec = PRECISION_DICT[self.precision]
        compute_np_prec = PRECISION_DICT[self.compute_precision]
        if "mean" in variables:
            self.mean = np.asarray(variables.pop("mean"), dtype=model_np_prec)
        if "stddev" in variables:
            self.stddev = np.asarray(variables.pop("stddev"), dtype=model_np_prec)

        def load_via_serialize(attr: str, prefix: str) -> None:
            sub = getattr(self, attr)
            sv = take_prefix(prefix)
            if sub is None:
                if sv:
                    raise KeyError(f"Unexpected variables with prefix: {prefix}")
                return
            if not sv:
                raise KeyError(f"Missing variables with prefix: {prefix}")
            data = sub.serialize()
            data["@variables"] = sv
            setattr(self, attr, type(sub).deserialize(data))

        load_via_serialize("type_embedding", "type_embedding.")
        load_via_serialize("env_seed_embedding", "env_seed_embedding.")
        load_via_serialize("film_scale_norm", "film_scale_norm.")
        load_via_serialize("film_shift_norm", "film_shift_norm.")
        load_via_serialize("radial_basis", "radial_basis.")
        load_via_serialize("radial_embedding", "radial_embedding.net.")
        if self.use_env_seed:
            for name in ("film_scale_strength_log", "film_shift_strength_log"):
                value = np.asarray(variables.pop(name), dtype=compute_np_prec)
                setattr(self, name, value.reshape((1,)))
        for i, block in enumerate(self.blocks):
            block._load_variables(take_prefix(f"blocks.{i}."))
        self.output_ffn._load_variables(take_prefix("output_ffn."))
        if variables:
            raise KeyError(f"Unknown variables: {sorted(variables)}")

    def serialize(self) -> dict[str, Any]:
        """Serialize the descriptor (pt ``DescrptSeZM.serialize`` format)."""
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
            },
            "@variables": self._variables(),
            "env_mat": EnvMat(self.rcut, self.rcut, self.eps).serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> DescrptDPA4:
        """Deserialize from a dict (accepts the pt ``serialize()`` output)."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "Descriptor":
            raise ValueError(f"Invalid class for DescrptDPA4: {data_cls}")
        type_val = data.pop("type")
        if type_val not in ("SeZM", "sezm", "dpa4"):
            raise ValueError(f"Invalid type for DescrptDPA4: {type_val}")
        version = float(data.pop("@version"))
        check_version_compatibility(version, cls.LATEST_VERSION, 1)
        config = dict(data.pop("config"))
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
        """Update the selection and perform neighbor statistics."""
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
