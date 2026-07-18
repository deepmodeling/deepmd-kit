# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Embedding layers for the DPA4/SeZM descriptor.

This module defines the type embedding, geometric initial embedding, and
environment-seed embedding used to initialize SeZM node features.

This module is the dpmodel (array-API) port of
``deepmd.pt.model.descriptor.sezm_nn.embedding``.
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
    DEFAULT_PRECISION,
    PRECISION_DICT,
    NativeOP,
)
from deepmd.dpmodel.array_api import (
    xp_add_at,
    xp_asarray_nodetach,
    xp_scatter_sum,
)
from deepmd.dpmodel.common import (
    get_xp_precision,
    to_numpy_array,
)
from deepmd.dpmodel.utils.network import (
    NativeLayer,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .cartesian import (
    build_cartesian_basis,
)
from .indexing import (
    build_gie_zonal_index,
    get_so3_dim_of_lmax,
)

if TYPE_CHECKING:
    from .edge_cache import (
        EdgeCache,
    )


class SeZMTypeEmbedding(NativeOP):
    """
    Minimal SeZM type embedding with Adam-routed parameter naming.

    Parameters
    ----------
    ntypes
        Number of atom types.
    embed_dim
        Embedding dimension.
    precision
        Parameter precision.
    seed
        Random seed for initialization.
    trainable
        Whether parameters are trainable.
    padding
        Whether to append one all-zero padding row.

    Notes
    -----
    The parameter is named with ``adam_`` prefix so HybridMuon routes it to Adam.
    """

    def __init__(
        self,
        *,
        ntypes: int,
        embed_dim: int,
        precision: str = DEFAULT_PRECISION,
        seed: int | list[int] | None = None,
        trainable: bool = True,
        padding: bool = True,
    ) -> None:
        self.ntypes = int(ntypes)
        self.embed_dim = int(embed_dim)
        self.precision = precision
        self.seed = seed
        self.trainable = bool(trainable)
        self.padding = bool(padding)
        if self.ntypes <= 0:
            raise ValueError("`ntypes` must be positive")
        if self.embed_dim <= 0:
            raise ValueError("`embed_dim` must be positive")
        prec = PRECISION_DICT[self.precision.lower()]

        # === Step 1. Build the full embedding table in a local array ===
        # The table is assembled locally and assigned to ``self`` exactly once.
        # The pt_expt backend converts ``self`` attributes into torch buffers on
        # assignment, so a later in-place slice write into
        # ``self.adam_type_embedding`` would raise; the local-then-assign pattern
        # keeps the produced values identical while staying backend-agnostic.
        n_rows = self.ntypes + int(self.padding)
        init_std = 1.0 / math.sqrt(float(self.ntypes + self.embed_dim))
        rng = np.random.default_rng(child_seed(seed, 0))
        table = np.empty((n_rows, self.embed_dim), dtype=prec)
        table[: self.ntypes] = rng.normal(
            0.0, init_std, size=(self.ntypes, self.embed_dim)
        )
        if self.padding:
            table[self.ntypes] = 0.0

        # === Step 2. Register the embedding table parameter ===
        self.adam_type_embedding = table

    def call(self, atype: Any) -> Any:
        """
        Gather type embeddings.

        Parameters
        ----------
        atype
            Atom types with shape (...,). Valid type range is [0, ntypes-1].

        Returns
        -------
        Array
            Type embeddings with shape (..., embed_dim).
        """
        xp = array_api_compat.array_namespace(atype)
        weight = xp_asarray_nodetach(
            xp, self.adam_type_embedding[...], device=array_api_compat.device(atype)
        )
        # torch.embedding gather: flatten the indices to int64, take the rows,
        # then restore the original index shape.
        index = xp.astype(xp.reshape(atype, (-1,)), xp.int64)
        out = xp.take(weight, index, axis=0)
        return xp.reshape(out, (*atype.shape, self.embed_dim))

    def serialize(self) -> dict[str, Any]:
        """Serialize the SeZMTypeEmbedding to a dict."""
        return {
            "@class": "SeZMTypeEmbedding",
            "@version": 1,
            "config": {
                "ntypes": self.ntypes,
                "embed_dim": self.embed_dim,
                "precision": np.dtype(PRECISION_DICT[self.precision]).name,
                "trainable": self.trainable,
                "padding": self.padding,
                "seed": None,
            },
            "@variables": {
                "adam_type_embedding": to_numpy_array(self.adam_type_embedding),
            },
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> SeZMTypeEmbedding:
        """Deserialize a SeZMTypeEmbedding from a dict."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "SeZMTypeEmbedding":
            raise ValueError(f"Invalid class for SeZMTypeEmbedding: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(**config)
        prec = PRECISION_DICT[obj.precision.lower()]
        obj.adam_type_embedding = np.asarray(
            variables["adam_type_embedding"], dtype=prec
        )
        return obj


class GeometricInitialEmbedding(NativeOP):
    """
    Geometric initial embedding that adds zonal (m=0) rotated features.

    This module rotates pre-computed radial features for each degree l >= 1 using the
    zonal (m=0) column of the cached inverse Wigner-D blocks (local->global).
    The l=0 component is not computed here since it comes from type embedding.

    Parameters
    ----------
    lmax
        Maximum node degree for the initial embedding.
    channels
        Number of channels per (l, m) coefficient.
    precision
        Parameter precision.
    """

    def __init__(
        self,
        *,
        lmax: int,
        channels: int,
        precision: str = DEFAULT_PRECISION,
    ) -> None:
        self.lmax = int(lmax)
        self.channels = int(channels)
        self.ebed_dim = get_so3_dim_of_lmax(self.lmax)
        self.precision = precision
        (
            node_row_index,
            node_zonal_m0_col_index,
            node_radial_l_index,
        ) = build_gie_zonal_index(self.lmax)
        # One aligned entry per non-scalar node row: output row, local m=0
        # column, and the matching radial degree slot.
        self.non_scalar_row_index = node_row_index
        self.zonal_m0_col_index_for_row = node_zonal_m0_col_index
        self.radial_slot_index_for_row = node_radial_l_index
        # The l=1 coefficients (packed rows 1..3) are the first three entries of
        # the non-scalar sequence ``node_row_index = [1, 2, ..., D-1]``, so the
        # native neighbor-spin l=1 message folds in at these local positions.
        self.l1_local_index = np.arange(3, dtype=np.int64)

    def call(
        self,
        *,
        n_nodes: int,
        edge_cache: EdgeCache,
        radial_feat: Any,
        zonal_coupling: Any = None,
        spin_l1_message: Any = None,
    ) -> Any:
        """
        Parameters
        ----------
        n_nodes
            Number of nodes (nf*nloc).
        edge_cache
            Per-edge cache containing geometry, weights, and Wigner-D blocks.
        radial_feat
            Per-edge radial features with shape (E, lmax, C) for l=1..lmax.
        zonal_coupling
            Optional precomputed zonal coupling with shape (E, D-1). If None,
            it is gathered from ``edge_cache.Dt_full``.
        spin_l1_message
            Optional per-edge neighbor-spin l=1 message with shape (E, 3, C) for
            the native spin scheme (built by ``SpinEmbedding.edge_l1``). It is
            added to the l=1 rows of the per-edge message, so it shares this
            module's source gate, scatter and degree normalization with the
            geometric message.

        Returns
        -------
        Array
            Initial features to add with shape (N, D, C). l=0 is guaranteed zero.
        """
        # === Step 1. Initialize output ===
        xp = array_api_compat.array_namespace(edge_cache.edge_vec)
        device = array_api_compat.device(edge_cache.edge_vec)
        dtype = edge_cache.edge_vec.dtype
        if self.lmax == 0:
            return xp.zeros(
                (n_nodes, self.ebed_dim, self.channels), dtype=dtype, device=device
            )  # (N, D, C)
        n_edge = edge_cache.dst.shape[0]

        # === Step 2. Gather all m=0 columns (l >= 1) in one shot ===
        # Advanced indexing pairs one packed non-scalar row with the zonal m=0 column
        # from the same degree block in Dt_full.
        if zonal_coupling is None:
            Dt_full = edge_cache.Dt_full  # (E, D, D)
            dim_full = Dt_full.shape[-1]
            flat_index = xp_asarray_nodetach(
                xp,
                self.non_scalar_row_index * dim_full + self.zonal_m0_col_index_for_row,
                device=device,
            )
            zonal_coupling = xp.take(
                xp.reshape(Dt_full, (n_edge, dim_full * dim_full)),
                flat_index,
                axis=1,
            )  # (E, D-1)

        # === Step 3. Broadcast radial features per row ===
        # Each non-scalar packed row reuses the radial feature of its degree l.
        radial_slot_index = xp_asarray_nodetach(
            xp, self.radial_slot_index_for_row, device=device
        )
        radial_value_for_row = xp.take(
            radial_feat, radial_slot_index, axis=1
        )  # (E, D-1, C)
        non_scalar_message = (
            zonal_coupling[:, :, None] * radial_value_for_row
        )  # (E, D-1, C)

        # === Step 3b. Fold in the neighbor-spin l=1 message (native spin) ===
        # The l=1 coefficients occupy the first three packed non-scalar rows, so
        # the neighbor-spin message joins the geometric message there and then
        # shares the source gate, scatter and degree normalization below.
        if spin_l1_message is not None:
            l1_local_index = xp_asarray_nodetach(xp, self.l1_local_index, device=device)
            scatter_index = xp.broadcast_to(
                xp.reshape(l1_local_index, (1, 3, 1)), spin_l1_message.shape
            )
            non_scalar_message = xp_scatter_sum(
                non_scalar_message, 1, scatter_index, spin_l1_message
            )

        # === Step 4. Source Freeze Propagation Gate (optional) ===
        # Mute messages emitted by nodes whose local neighborhood enters
        # the frozen zone. ``edge_src_gate`` is ``None`` outside bridging
        # mode so this is a no-op in normal training.
        src_gate = edge_cache.edge_src_gate
        if src_gate is not None:
            non_scalar_message = non_scalar_message * xp.astype(
                xp.reshape(src_gate, (n_edge, 1, 1)), non_scalar_message.dtype
            )

        # === Step 5. Scatter to nodes and normalize ===
        # Destination scatter-add over ``edge_cache.dst`` (pt ``index_add_``),
        # applied after the validity masking below. This reduction is
        # layout-agnostic: it is correct both for the padded ``call`` (row-major
        # ``dst`` makes the accumulation order identical to a sum over the
        # ``nnei`` axis, hence bit-exact) and for the sparse ``call_with_edges``
        # (arbitrary ``dst`` order and per-node degree). The l=0 row is left at
        # its zero initialization by concatenating it below the contiguous
        # non-scalar rows 1..D-1.
        edge_mask = edge_cache.edge_mask
        if edge_mask is not None:
            non_scalar_message = non_scalar_message * xp.astype(
                xp.reshape(edge_mask, (n_edge, 1, 1)), non_scalar_message.dtype
            )
        non_scalar_out = xp_add_at(
            xp.zeros(
                (n_nodes, self.ebed_dim - 1, self.channels),
                dtype=non_scalar_message.dtype,
                device=device,
            ),
            edge_cache.dst,
            non_scalar_message,
        )  # (N, D-1, C)
        out = xp.concat(
            [
                xp.zeros(
                    (n_nodes, 1, self.channels),
                    dtype=non_scalar_out.dtype,
                    device=device,
                ),
                non_scalar_out,
            ],
            axis=1,
        )  # (N, D, C)
        out = out * xp.astype(edge_cache.inv_sqrt_deg, out.dtype)
        return xp.astype(out, dtype)

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "GeometricInitialEmbedding",
            "@version": 1,
            "lmax": self.lmax,
            "channels": self.channels,
            "precision": np.dtype(PRECISION_DICT[self.precision]).name,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> GeometricInitialEmbedding:
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "GeometricInitialEmbedding":
            raise ValueError(f"Invalid class for GeometricInitialEmbedding: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        return cls(**data)


class EnvironmentInitialEmbedding(NativeOP):
    """
    Environment matrix initial embedding for l=0 features.

    Computes an initial embedding based on the 4D environment matrix::

        [s, s * rx, s * ry, s * rz]

    Combined with independent type embeddings (individual type embedding),
    providing physical inductive bias for l=0 features.

    The computation follows the environment matrix approach where::

        1. Build `r_tilde = [s, s*r_hat]` where `s = edge_env / r` and `r_hat = edge_vec / r`
        2. G network: `g = G(rbf_proj(edge_rbf), type_src, type_dst)` produces per-edge features
           - Uses independent `env_type_embed` instead of projecting from main type embedding
           - Uses `rbf_proj` to project edge_rbf to `rbf_out_dim`
        3. env_agg: aggregate outer product `r_tilde ⊗ g` by destination node
        4. D matrix: `D = env_agg^T @ env_agg[:, :, :axis_dim]`
        5. Output: projection of flattened D matrix into FiLM logits

    Parameters
    ----------
    ntypes : int
        Number of atom types.
    n_radial : int
        Number of radial basis functions.
    channels : int
        Output channel dimension per FiLM branch (final output is 2*channels).
    embed_dim : int
        G network output dimension (filter width).
    axis_dim : int
        D matrix axis dimension (must be < embed_dim).
    type_dim : int
        Dimension for independent type embeddings in env_seed.
    hidden_dim : int
        Hidden layer size for G network.
    mlp_bias : bool
        Whether to enable bias terms in env-seed MLP layers
        (`rbf_proj_layer1/2` and `g_layer1/2`).
    activation_function : str
        Activation function for G network hidden layer.
    eps : float
        Small epsilon for numerical stability.
    use_spin : list[bool] | None
        Per-type spin flags (native spin scheme). When provided, the neighbor
        spin is appended as extra coordinate channels of the environment matrix,
        so the inner product ``D = M^T M`` additionally yields the neighbor
        spin-spin invariants. A per-type mask gates the channel, so a
        non-magnetic neighbor contributes zero and carries zero magnetic force.
    precision : str
        Parameter precision.
    trainable : bool
        Whether parameters are trainable.
    seed : int | list[int] | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        *,
        ntypes: int,
        n_radial: int,
        channels: int,
        embed_dim: int = 64,
        axis_dim: int = 8,
        type_dim: int = 16,
        hidden_dim: int = 64,
        mlp_bias: bool = False,
        activation_function: str = "silu",
        eps: float = 1e-7,
        use_spin: list[bool] | None = None,
        precision: str = DEFAULT_PRECISION,
        trainable: bool = True,
        seed: int | list[int] | None = None,
    ) -> None:
        # === Validate parameters ===
        if axis_dim >= embed_dim:
            raise ValueError(
                f"`axis_dim` ({axis_dim}) must be < `embed_dim` ({embed_dim})"
            )

        self.ntypes = int(ntypes)
        self.n_radial = int(n_radial)
        self.channels = int(channels)
        self.embed_dim = int(embed_dim)
        self.axis_dim = int(axis_dim)
        self.type_dim = int(type_dim)
        self.hidden_dim = int(hidden_dim)
        self.mlp_bias = bool(mlp_bias)
        self.activation_function = str(activation_function)
        self.eps = float(eps)
        self.spin_flags = None if use_spin is None else [bool(x) for x in use_spin]
        if self.spin_flags is not None and len(self.spin_flags) != int(ntypes):
            raise ValueError("`use_spin` length must equal `ntypes`")
        self.precision = precision
        self.trainable = bool(trainable)
        # The environment matrix carries the 4 geometric channels ``[s, s*r_hat]``
        # plus, for the native spin scheme, the 3 envelope-gated neighbor-spin
        # components, so the inner product ``D = M^T M`` yields the neighbor
        # spin-spin invariants alongside the geometric ones.
        self.coord_dim = 4 + (3 if self.spin_flags is not None else 0)

        # === RBF projection: n_radial -> rbf_out_dim (two-layer MLP) ===
        # rbf_out_dim = max(32, embed_dim - 2*type_dim) to align G-network width to embed_dim
        # First layer: n_radial -> rbf_out_dim with activation
        # Second layer: rbf_out_dim -> rbf_out_dim linear
        self.rbf_out_dim = max(32, self.embed_dim - 2 * self.type_dim)
        seed_rbf_proj = child_seed(seed, 0)
        self.rbf_proj_layer1 = NativeLayer(
            self.n_radial,
            self.rbf_out_dim,
            bias=self.mlp_bias,
            activation_function=self.activation_function,
            precision=self.precision,
            seed=child_seed(seed_rbf_proj, 0),
            trainable=self.trainable,
        )
        self.rbf_proj_layer2 = NativeLayer(
            self.rbf_out_dim,
            self.rbf_out_dim,
            bias=self.mlp_bias,
            activation_function=None,
            precision=self.precision,
            seed=child_seed(seed_rbf_proj, 1),
            trainable=self.trainable,
        )

        # === Independent type embedding: ntypes -> type_dim ===
        # Individual type embedding
        seed_type_embed = child_seed(seed, 1)
        self.env_type_embed = SeZMTypeEmbedding(
            ntypes=self.ntypes,
            embed_dim=self.type_dim,
            precision=self.precision,
            seed=seed_type_embed,
            trainable=self.trainable,
        )

        # === G network: (rbf_out_dim + 2*type_dim) -> hidden_dim -> embed_dim ===
        seed_g_net = child_seed(seed, 2)
        g_in_dim = self.rbf_out_dim + 2 * self.type_dim
        self.g_layer1 = NativeLayer(
            g_in_dim,
            self.hidden_dim,
            bias=self.mlp_bias,
            activation_function=self.activation_function,
            precision=self.precision,
            seed=child_seed(seed_g_net, 0),
            trainable=self.trainable,
        )
        self.g_layer2 = NativeLayer(
            self.hidden_dim,
            self.embed_dim,
            bias=self.mlp_bias,
            activation_function=None,
            precision=self.precision,
            seed=child_seed(seed_g_net, 1),
            trainable=self.trainable,
        )

        # === Output projection: embed_dim * axis_dim -> 2*channels ===
        # Zero init so FiLM logits start at zero; strengths control magnitude.
        seed_out = child_seed(seed, 3)
        self.output_proj = NativeLayer(
            self.embed_dim * self.axis_dim,
            2 * self.channels,
            bias=False,
            activation_function=None,
            precision=self.precision,
            seed=seed_out,
            trainable=self.trainable,
        )
        # NativeLayer has no ``init="final"``; replicate it by zeroing the weight.
        self.output_proj.w = np.zeros(
            (self.embed_dim * self.axis_dim, 2 * self.channels),
            dtype=PRECISION_DICT[self.precision.lower()],
        )

        # === Native spin: per-type mask and isotropic channel scale ===
        # The mask gates the neighbor-spin channel by source type, so a
        # non-magnetic neighbor contributes zero and (critically) carries zero
        # magnetic force ``-dE/ds``. The single scalar scale (shared across
        # x/y/z) keeps the spin coordinates transforming with the geometry, so
        # the env-matrix invariant stays SO(3)-invariant; ``output_proj`` is
        # zero-initialized, so the spin contribution starts neutral regardless.
        if self.spin_flags is not None:
            self.spin_mask = np.array(
                [1.0 if flag else 0.0 for flag in self.spin_flags],
                dtype=PRECISION_DICT[self.precision.lower()],
            )
            self.spin_scale = np.ones(
                (1,), dtype=PRECISION_DICT[self.precision.lower()]
            )

    def call(
        self,
        *,
        edge_cache: EdgeCache,
        atype_flat: Any,
        n_nodes: int,
        spin: Any = None,
    ) -> Any:
        """
        Compute environment FiLM logits for l=0 conditioning.

        Parameters
        ----------
        edge_cache : EdgeCache
            Edge cache containing src, dst, edge_vec, edge_rbf, edge_env.
        atype_flat : Array
            Flattened atom types with shape (N,), where N = nf * nloc.
        n_nodes : int
            Number of nodes (N = nf * nloc).
        spin : Array | None
            Per-node spin vectors with shape (N, 3) for the native spin scheme.
            Used only when ``use_spin`` is set; the source (neighbor) spin is
            appended to the environment matrix as an envelope-gated coordinate
            channel. When ``None`` the spin channels are zero-padded so the
            coordinate dimension stays fixed.

        Returns
        -------
        Array
            FiLM logits with shape (N, 2*channels).
        """
        xp = array_api_compat.array_namespace(edge_cache.edge_vec)
        src, dst = edge_cache.src, edge_cache.dst
        edge_vec = edge_cache.edge_vec  # (E, 3)
        edge_rbf = edge_cache.edge_rbf  # (E, n_radial)
        edge_env = edge_cache.edge_env  # (E, 1)
        n_edge = dst.shape[0]

        # === Step 1. Construct r_tilde = [s, s*r_hat] ===
        # s = edge_env * (1/r), r_hat = edge_vec / r
        r_sq = xp.sum(edge_vec * edge_vec, axis=-1, keepdims=True)  # (E, 1)
        inv_r = 1.0 / xp.sqrt(r_sq + self.eps * self.eps)  # (E, 1)
        s = edge_env * inv_r  # (E, 1)
        r_hat = edge_vec * inv_r  # (E, 3)
        r_tilde = xp.concat([s, s * r_hat], axis=-1)  # (E, 4)

        # === Step 1b. Append neighbor spin as extra coordinate channels ===
        # The source (neighbor) spin enters the environment matrix gated by the
        # same C^3 envelope as the geometry, so it decays smoothly at rcut and a
        # non-magnetic neighbor (s_j = 0) contributes exactly zero. The linear
        # form keeps the magnetic force continuous at s = 0.
        if self.spin_flags is not None:
            device = array_api_compat.device(edge_vec)
            if spin is not None:
                src_i = xp.astype(src, xp.int64)
                spin_src = xp.astype(
                    xp.take(spin, src_i, axis=0), r_tilde.dtype
                )  # (E, 3)
                # Gate by source type: a non-magnetic neighbor must not enter
                # the energy, so its magnetic force ``-dE/ds`` stays exactly zero.
                spin_mask = xp_asarray_nodetach(xp, self.spin_mask[...], device=device)
                mask = xp.take(
                    spin_mask,
                    xp.take(xp.astype(atype_flat, xp.int64), src_i, axis=0),
                    axis=0,
                )[:, None]  # (E, 1)
                spin_scale = xp.astype(
                    xp_asarray_nodetach(xp, self.spin_scale[...], device=device),
                    r_tilde.dtype,
                )
                spin_chan = edge_env * spin_scale * spin_src * mask  # (E, 3)
            else:
                spin_chan = xp.zeros(
                    (r_tilde.shape[0], 3), dtype=r_tilde.dtype, device=device
                )
            r_tilde = xp.concat([r_tilde, spin_chan], axis=-1)  # (E, coord_dim)

        # === Step 2. Compute G network input and output ===
        # Use independent type embeddings (decoupled from main type embedding)
        atype_src = xp.take(atype_flat, xp.astype(src, xp.int64), axis=0)  # (E,)
        atype_dst = xp.take(atype_flat, xp.astype(dst, xp.int64), axis=0)  # (E,)
        type_src = self.env_type_embed(atype_src)  # (E, type_dim)
        type_dst = self.env_type_embed(atype_dst)  # (E, type_dim)

        # Project edge_rbf to rbf_out_dim (two-layer MLP)
        rbf_proj = self.rbf_proj_layer2(
            self.rbf_proj_layer1(edge_rbf)
        )  # (E, rbf_out_dim)

        # G network input: concat projected RBF and type embeddings
        g_input = xp.concat([rbf_proj, type_src, type_dst], axis=-1)  # (E, g_in_dim)
        g = self.g_layer2(self.g_layer1(g_input))  # (E, embed_dim)

        # === Step 3. Aggregate outer product by destination node ===
        # outer = r_tilde[:, :, None] * g[:, None, :], einsum "ei,ej->eij".
        outer = r_tilde[:, :, None] * g[:, None, :]  # (E, coord_dim, embed_dim)
        outer_flat = xp.reshape(
            outer, (n_edge, self.coord_dim * self.embed_dim)
        )  # (E, coord_dim*embed_dim)
        # Source Freeze Propagation Gate: mute the outer-product contribution
        # of any edge whose source node has a neighbor in the frozen zone.
        src_gate = edge_cache.edge_src_gate
        if src_gate is not None:
            outer_flat = outer_flat * xp.astype(
                xp.reshape(src_gate, (n_edge, 1)), outer_flat.dtype
            )
        # Destination scatter-add over ``dst`` (pt ``index_add_``), applied after
        # the validity masking below. Layout-agnostic: correct for the padded
        # ``call`` (row-major ``dst`` keeps the accumulation order identical to a
        # sum over the ``nnei`` axis, hence bit-exact) and for the sparse
        # ``call_with_edges`` (arbitrary ``dst`` order and per-node degree).
        edge_mask = edge_cache.edge_mask
        if edge_mask is not None:
            outer_flat = outer_flat * xp.astype(
                xp.reshape(edge_mask, (n_edge, 1)), outer_flat.dtype
            )
        env_agg = xp_add_at(
            xp.zeros(
                (n_nodes, self.coord_dim * self.embed_dim),
                dtype=outer_flat.dtype,
                device=array_api_compat.device(outer_flat),
            ),
            dst,
            outer_flat,
        )  # (N, coord_dim*embed_dim)
        env_agg = xp.reshape(
            env_agg, (n_nodes, self.coord_dim, self.embed_dim)
        )  # (N, coord_dim, embed_dim)

        # === Step 4. Smooth normalization by envelope-squared degree ===
        # Reuse the cache's inverse-sqrt degree so the version-aware
        # ``deg_norm_floor`` is applied consistently with GIE.
        env_agg = env_agg * xp.astype(edge_cache.inv_sqrt_deg, env_agg.dtype)

        # === Step 5. D matrix construction: D = env_agg^T @ env_agg[:,:,:axis_dim] ===
        # Summing over the coordinate axis makes D invariant to a joint rotation
        # of the geometry and the spin channels; with the spin channels present,
        # D additionally carries the neighbor spin-spin invariants.
        env_agg_t = xp.permute_dims(env_agg, (0, 2, 1))  # (N, embed_dim, coord_dim)
        env_agg_axis = env_agg[:, :, : self.axis_dim]  # (N, coord_dim, axis_dim)
        D = xp.matmul(env_agg_t, env_agg_axis)  # (N, embed_dim, axis_dim)

        # === Step 6. Output projection for FiLM logits ===
        D_flat = xp.reshape(
            D, (n_nodes, self.embed_dim * self.axis_dim)
        )  # (N, embed_dim*axis_dim)
        return self.output_proj(D_flat)

    def _variables(self) -> dict[str, np.ndarray]:
        """Variables keyed by the pt ``state_dict`` key names."""
        variables = {
            "rbf_proj_layer1.matrix": to_numpy_array(self.rbf_proj_layer1.w),
            "rbf_proj_layer2.matrix": to_numpy_array(self.rbf_proj_layer2.w),
            "env_type_embed.adam_type_embedding": to_numpy_array(
                self.env_type_embed.adam_type_embedding
            ),
            "g_layer1.matrix": to_numpy_array(self.g_layer1.w),
            "g_layer2.matrix": to_numpy_array(self.g_layer2.w),
            "output_proj.matrix": to_numpy_array(self.output_proj.w),
        }
        if self.mlp_bias:
            variables["rbf_proj_layer1.bias"] = to_numpy_array(self.rbf_proj_layer1.b)
            variables["rbf_proj_layer2.bias"] = to_numpy_array(self.rbf_proj_layer2.b)
            variables["g_layer1.bias"] = to_numpy_array(self.g_layer1.b)
            variables["g_layer2.bias"] = to_numpy_array(self.g_layer2.b)
        if self.spin_flags is not None:
            variables["spin_scale"] = to_numpy_array(self.spin_scale)
        return variables

    def _load_variables(self, variables: dict[str, Any]) -> None:
        """Load variables keyed by the pt ``state_dict`` key names."""
        prec = PRECISION_DICT[self.precision.lower()]
        self.rbf_proj_layer1.w = np.asarray(
            variables["rbf_proj_layer1.matrix"], dtype=prec
        )
        self.rbf_proj_layer2.w = np.asarray(
            variables["rbf_proj_layer2.matrix"], dtype=prec
        )
        self.env_type_embed.adam_type_embedding = np.asarray(
            variables["env_type_embed.adam_type_embedding"], dtype=prec
        )
        self.g_layer1.w = np.asarray(variables["g_layer1.matrix"], dtype=prec)
        self.g_layer2.w = np.asarray(variables["g_layer2.matrix"], dtype=prec)
        self.output_proj.w = np.asarray(variables["output_proj.matrix"], dtype=prec)
        if self.mlp_bias:
            self.rbf_proj_layer1.b = np.asarray(
                variables["rbf_proj_layer1.bias"], dtype=prec
            )
            self.rbf_proj_layer2.b = np.asarray(
                variables["rbf_proj_layer2.bias"], dtype=prec
            )
            self.g_layer1.b = np.asarray(variables["g_layer1.bias"], dtype=prec)
            self.g_layer2.b = np.asarray(variables["g_layer2.bias"], dtype=prec)
        if self.spin_flags is not None:
            self.spin_scale = np.asarray(variables["spin_scale"], dtype=prec)

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "EnvironmentInitialEmbedding",
            "@version": 1,
            "config": {
                "ntypes": self.ntypes,
                "n_radial": self.n_radial,
                "channels": self.channels,
                "embed_dim": self.embed_dim,
                "axis_dim": self.axis_dim,
                "type_dim": self.type_dim,
                "hidden_dim": self.hidden_dim,
                "mlp_bias": self.mlp_bias,
                "activation_function": self.activation_function,
                "eps": self.eps,
                "use_spin": self.spin_flags,
                "precision": np.dtype(PRECISION_DICT[self.precision]).name,
                "trainable": self.trainable,
                "seed": None,
            },
            "@variables": self._variables(),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> EnvironmentInitialEmbedding:
        """Deserialize from dictionary."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "EnvironmentInitialEmbedding":
            raise ValueError(f"Invalid class: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(**config)
        obj._load_variables(variables)
        return obj


class ChargeSpinEmbedding(NativeOP):
    """
    Frame-level charge and spin embedding for scalar type features.

    Parameters
    ----------
    embed_dim
        Embedding dimension.
    activation_function
        Activation function used by the mixing layer.
    precision
        Parameter precision.
    seed
        Random seed for initialization.
    trainable
        Whether parameters are trainable.
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        activation_function: str,
        precision: str = DEFAULT_PRECISION,
        seed: int | list[int] | None = None,
        trainable: bool = True,
    ) -> None:
        self.embed_dim = int(embed_dim)
        self.activation_function = str(activation_function)
        self.precision = precision
        self.trainable = bool(trainable)
        if self.embed_dim <= 0:
            raise ValueError("`embed_dim` must be positive")

        self.charge_embedding = SeZMTypeEmbedding(
            ntypes=200,
            embed_dim=self.embed_dim,
            precision=self.precision,
            seed=child_seed(seed, 0),
            trainable=self.trainable,
            padding=False,
        )
        self.spin_embedding = SeZMTypeEmbedding(
            ntypes=100,
            embed_dim=self.embed_dim,
            precision=self.precision,
            seed=child_seed(seed, 1),
            trainable=self.trainable,
            padding=False,
        )
        self.mix_layer = NativeLayer(
            2 * self.embed_dim,
            self.embed_dim,
            activation_function=self.activation_function,
            precision=self.precision,
            seed=child_seed(seed, 2),
            trainable=self.trainable,
        )

    def call(self, charge_spin: Any) -> Any:
        """
        Embed frame-level charge and spin.

        Parameters
        ----------
        charge_spin
            Frame charge and spin values with shape (nf, 2).

        Returns
        -------
        Array
            Mixed condition embedding with shape (nf, embed_dim).
        """
        xp = array_api_compat.array_namespace(charge_spin)
        charge = xp.astype(charge_spin[:, 0], xp.int64) + 100
        spin = xp.astype(charge_spin[:, 1], xp.int64)
        charge_embed = self.charge_embedding(charge)
        spin_embed = self.spin_embedding(spin)
        return self.mix_layer(xp.concat((charge_embed, spin_embed), axis=-1))

    def _variables(self) -> dict[str, np.ndarray]:
        """Variables keyed by the pt ``state_dict`` key names."""
        return {
            "charge_embedding.adam_type_embedding": to_numpy_array(
                self.charge_embedding.adam_type_embedding
            ),
            "spin_embedding.adam_type_embedding": to_numpy_array(
                self.spin_embedding.adam_type_embedding
            ),
            "mix_layer.matrix": to_numpy_array(self.mix_layer.w),
            "mix_layer.bias": to_numpy_array(self.mix_layer.b),
        }

    def _load_variables(self, variables: dict[str, Any]) -> None:
        """Load variables keyed by the pt ``state_dict`` key names."""
        prec = PRECISION_DICT[self.precision.lower()]
        self.charge_embedding.adam_type_embedding = np.asarray(
            variables["charge_embedding.adam_type_embedding"], dtype=prec
        )
        self.spin_embedding.adam_type_embedding = np.asarray(
            variables["spin_embedding.adam_type_embedding"], dtype=prec
        )
        self.mix_layer.w = np.asarray(variables["mix_layer.matrix"], dtype=prec)
        self.mix_layer.b = np.asarray(variables["mix_layer.bias"], dtype=prec)

    def serialize(self) -> dict[str, Any]:
        """Serialize the ChargeSpinEmbedding to a dict."""
        return {
            "@class": "ChargeSpinEmbedding",
            "@version": 1,
            "config": {
                "embed_dim": self.embed_dim,
                "activation_function": self.activation_function,
                "precision": np.dtype(PRECISION_DICT[self.precision]).name,
                "trainable": self.trainable,
                "seed": None,
            },
            "@variables": self._variables(),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> ChargeSpinEmbedding:
        """Deserialize a ChargeSpinEmbedding from a dict."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "ChargeSpinEmbedding":
            raise ValueError(f"Invalid class for ChargeSpinEmbedding: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(**config)
        obj._load_variables(variables)
        return obj


class SpinEmbedding(NativeOP):
    """
    Per-atom spin embedding for the native spin scheme.

    The per-atom spin vector ``s`` is injected as an equivariant extension of
    the type embedding, producing two additive contributions to the descriptor
    node features:

    - **l = 0 (invariant):** a small network of the squared magnitude ``|s|^2``
      yields a per-channel scalar added to the scalar type embedding. The
      squared magnitude is used (rather than ``|s|``) so the feature is smooth
      at ``s = 0`` and its gradient there vanishes, keeping the magnetic force
      continuous as a spin crosses zero.
    - **l = 1 (equivariant):** the Cartesian spin vector is mapped to the packed
      ``l = 1`` coefficients through the SeZM Wigner-D convention (derived from
      :func:`build_cartesian_basis`), then scaled by a per-type per-channel
      weight. The map is linear in ``s``, so the contribution vanishes at
      ``s = 0`` and rotates as an ``l = 1`` object under SO(3), i.e.
      ``cart_to_l1(R s) = D^1(R) cart_to_l1(s)``.

    Both contributions are gated by a per-type spin mask, so atom types without
    spin contribute exactly zero regardless of their (nominally zero) input.

    Parameters
    ----------
    ntypes
        Number of (real) atom types.
    channels
        Number of channels per (l, m) coefficient.
    use_spin
        Per-type boolean flags marking which atom types carry spin.
    activation_function
        Activation used by the magnitude network.
    precision
        Parameter precision.
    seed
        Random seed for initialization.
    trainable
        Whether parameters are trainable.
    """

    def __init__(
        self,
        *,
        ntypes: int,
        channels: int,
        use_spin: list[bool],
        activation_function: str = "silu",
        precision: str = DEFAULT_PRECISION,
        seed: int | list[int] | None = None,
        trainable: bool = True,
    ) -> None:
        self.ntypes = int(ntypes)
        self.channels = int(channels)
        self.activation_function = str(activation_function)
        self.precision = precision
        self.trainable = bool(trainable)
        if self.ntypes <= 0:
            raise ValueError("`ntypes` must be positive")
        if self.channels <= 0:
            raise ValueError("`channels` must be positive")
        if len(use_spin) != self.ntypes:
            raise ValueError("`use_spin` length must equal `ntypes`")
        prec = PRECISION_DICT[self.precision.lower()]
        self.spin_flags = [bool(flag) for flag in use_spin]

        # === Per-type spin gate ===
        # Non-persistent: rebuilt from config on construction and moved with the
        # module, so the deterministic mask never enters the serialized state.
        self.spin_mask = np.array(
            [1.0 if bool(flag) else 0.0 for flag in use_spin], dtype=prec
        )

        # === Cartesian -> packed l=1 projection ===
        # Derived from the SeZM packed basis so a spin vector rotates with the
        # same Wigner-D block as the geometry. Non-persistent constant.
        self.cart_to_l1 = self._build_cart_to_l1_matrix()

        # === l=0 magnitude network: |s|^2 -> channels ===
        # The leading ``1 -> channels`` layer carries a singleton input
        # dimension that HybridMuon routes to its Adam path automatically.
        seed_scalar = child_seed(seed, 0)
        self.mag_layer1 = NativeLayer(
            1,
            self.channels,
            bias=False,
            activation_function=self.activation_function,
            precision=self.precision,
            seed=child_seed(seed_scalar, 0),
            trainable=self.trainable,
        )
        self.mag_layer2 = NativeLayer(
            self.channels,
            self.channels,
            bias=False,
            activation_function=None,
            precision=self.precision,
            seed=child_seed(seed_scalar, 1),
            trainable=self.trainable,
        )

        # === l=1 per-type per-channel weight ===
        # ``adam_`` prefix routes the table to Adam in HybridMuon, matching the
        # type-embedding treatment for per-type lookup parameters.
        init_std = 1.0 / math.sqrt(float(self.ntypes + self.channels))
        rng_vec = np.random.default_rng(child_seed(seed, 1))
        self.adam_spin_vec_weight = rng_vec.normal(
            0.0, init_std, size=(self.ntypes, self.channels)
        ).astype(prec)

        # === l=1 per-source-type per-channel weight for neighbor aggregation ===
        # Separate from the on-site weight: this scales the neighbor's spin
        # direction before it is aggregated into the center node's l=1 seed.
        rng_nbr = np.random.default_rng(child_seed(seed, 2))
        self.adam_spin_nbr_weight = rng_nbr.normal(
            0.0, init_std, size=(self.ntypes, self.channels)
        ).astype(prec)

    def call(self, spin: Any, atype: Any) -> tuple[Any, Any]:
        """
        Compute the l=0 and l=1 spin contributions.

        Parameters
        ----------
        spin
            Per-atom spin vectors with shape (N, 3).
        atype
            Per-atom types with shape (N,).

        Returns
        -------
        tuple[Array, Array]
            ``(scalar, vector)`` where ``scalar`` has shape (N, channels) for
            the l=0 contribution and ``vector`` has shape (N, 3, channels) for
            the packed l=1 contribution (orders m = -1, 0, +1). Both are exactly
            zero for atom types without spin.
        """
        xp = array_api_compat.array_namespace(spin)
        device = array_api_compat.device(spin)
        dtype = get_xp_precision(xp, self.precision)
        spin = xp.astype(spin, dtype)
        index = xp.astype(atype, xp.int64)
        spin_mask = xp_asarray_nodetach(xp, self.spin_mask[...], device=device)
        mask = xp.take(spin_mask, index, axis=0)[:, None]  # (N, 1)

        # === l=0: smooth invariant magnitude embedding ===
        mag2 = xp.sum(spin * spin, axis=-1, keepdims=True)  # (N, 1)
        scalar = self.mag_layer2(self.mag_layer1(mag2)) * mask  # (N, C)

        # === l=1: equivariant direction embedding (linear in spin) ===
        cart_to_l1 = xp.astype(
            xp_asarray_nodetach(xp, self.cart_to_l1[...], device=device), dtype
        )
        # einsum "dk,nk->nd" as a matmul against the transposed projection.
        l1 = xp.matmul(spin, xp.permute_dims(cart_to_l1, (1, 0)))  # (N, 3)
        weight_table = xp_asarray_nodetach(
            xp, self.adam_spin_vec_weight[...], device=device
        )
        weight = xp.take(weight_table, index, axis=0)  # (N, C)
        vector = l1[:, :, None] * weight[:, None, :] * mask[:, :, None]  # (N, 3, C)

        return scalar, vector

    def edge_l1(
        self,
        spin: Any,
        atype: Any,
        edge_cache: EdgeCache,
    ) -> Any:
        """
        Build the per-edge neighbor-spin l=1 message for the GIE aggregation.

        Each edge carries the packed ``l = 1`` coefficients of the source
        (neighbor) spin, scaled by a per-source-type per-channel weight and
        gated by the C^3 envelope. The message is returned per edge; the
        geometric initial embedding folds it into the l=1 rows and applies the
        shared source gate, scatter and degree normalization, so a neighbor's
        spin direction enters an atom's l=1 backbone before any interaction
        block (the spin analogue of the geometric initial embedding).

        Parameters
        ----------
        spin
            Per-node spin vectors with shape (N, 3).
        atype
            Per-node types with shape (N,).
        edge_cache
            Edge cache providing ``src`` and ``edge_env``.

        Returns
        -------
        Array
            Per-edge packed l=1 message with shape (E, 3, channels), exactly
            zero for non-magnetic neighbors.
        """
        xp = array_api_compat.array_namespace(spin)
        device = array_api_compat.device(spin)
        dtype = get_xp_precision(xp, self.precision)
        spin = xp.astype(spin, dtype)
        src = xp.astype(edge_cache.src, xp.int64)
        spin_src = xp.take(spin, src, axis=0)  # (E, 3)
        atype_src = xp.take(xp.astype(atype, xp.int64), src, axis=0)  # (E,)

        # Packed l=1 of the neighbor spin; the global-frame vector needs no
        # Wigner-D rotation (it rotates with the geometry by construction).
        cart_to_l1 = xp.astype(
            xp_asarray_nodetach(xp, self.cart_to_l1[...], device=device), dtype
        )
        # einsum "dk,ek->ed" as a matmul against the transposed projection.
        l1 = xp.matmul(spin_src, xp.permute_dims(cart_to_l1, (1, 0)))  # (E, 3)
        weight_table = xp_asarray_nodetach(
            xp, self.adam_spin_nbr_weight[...], device=device
        )
        weight = xp.take(weight_table, atype_src, axis=0)  # (E, C)
        spin_mask = xp_asarray_nodetach(xp, self.spin_mask[...], device=device)
        mask = xp.take(spin_mask, atype_src, axis=0)  # (E,)
        gate = edge_cache.edge_env * mask[:, None]  # (E, 1)
        return gate[:, :, None] * l1[:, :, None] * weight[:, None, :]  # (E, 3, C)

    def _build_cart_to_l1_matrix(self) -> np.ndarray:
        """
        Build the ``(3, 3)`` Cartesian-to-packed-``l=1`` projection.

        The packed ``l = 1`` coefficient of a vector ``v`` is obtained by
        projecting the skew-symmetric matrix ``[v]_x`` onto the antisymmetric
        ``l = 1`` block of :func:`build_cartesian_basis`. With packed order
        ``m = -1, 0, +1``, row ``d`` and Cartesian component ``k`` give
        ``M[d, k] = <[e_k]_x, B[1 + d]>_F``, so ``coeff = M @ v`` and
        ``M @ (R v) = D^1(R) (M @ v)``.
        """
        prec = PRECISION_DICT[self.precision.lower()]
        basis_l1 = build_cartesian_basis(1, dtype=prec)[1:4]
        # Skew (cross-product) matrices of the Cartesian unit vectors, following
        # ``[v]_x w = v x w`` (matching ``build_edge_cartesian_tensors``).
        skew_basis = np.zeros((3, 3, 3), dtype=prec)
        skew_basis[0, 1, 2], skew_basis[0, 2, 1] = -1.0, 1.0
        skew_basis[1, 0, 2], skew_basis[1, 2, 0] = 1.0, -1.0
        skew_basis[2, 0, 1], skew_basis[2, 1, 0] = -1.0, 1.0
        return np.einsum("kij,dij->dk", skew_basis, basis_l1)

    def _variables(self) -> dict[str, np.ndarray]:
        """Variables keyed by the pt ``state_dict`` key names."""
        return {
            "mag_layer1.matrix": to_numpy_array(self.mag_layer1.w),
            "mag_layer2.matrix": to_numpy_array(self.mag_layer2.w),
            "adam_spin_vec_weight": to_numpy_array(self.adam_spin_vec_weight),
            "adam_spin_nbr_weight": to_numpy_array(self.adam_spin_nbr_weight),
        }

    def _load_variables(self, variables: dict[str, Any]) -> None:
        """Load variables keyed by the pt ``state_dict`` key names."""
        prec = PRECISION_DICT[self.precision.lower()]
        self.mag_layer1.w = np.asarray(variables["mag_layer1.matrix"], dtype=prec)
        self.mag_layer2.w = np.asarray(variables["mag_layer2.matrix"], dtype=prec)
        self.adam_spin_vec_weight = np.asarray(
            variables["adam_spin_vec_weight"], dtype=prec
        )
        self.adam_spin_nbr_weight = np.asarray(
            variables["adam_spin_nbr_weight"], dtype=prec
        )

    def serialize(self) -> dict[str, Any]:
        """Serialize the SpinEmbedding to a dict."""
        return {
            "@class": "SpinEmbedding",
            "@version": 1,
            "config": {
                "ntypes": self.ntypes,
                "channels": self.channels,
                "use_spin": self.spin_flags,
                "activation_function": self.activation_function,
                "precision": np.dtype(PRECISION_DICT[self.precision]).name,
                "trainable": self.trainable,
                "seed": None,
            },
            "@variables": self._variables(),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> SpinEmbedding:
        """Deserialize a SpinEmbedding from a dict."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "SpinEmbedding":
            raise ValueError(f"Invalid class for SpinEmbedding: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(**config)
        obj._load_variables(variables)
        return obj
