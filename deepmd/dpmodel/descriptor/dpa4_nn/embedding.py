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
from deepmd.dpmodel.utils.type_embed import (
    remap_atype_to_padding,
)
from deepmd.utils.charge_state import (
    CHARGE_OFFSET,
    CHARGE_TABLE_ROWS,
    MULTIPLICITY_TABLE_ROWS,
)
from deepmd.utils.version import (
    check_version_compatibility,
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

    def call(self, atype: Any | None = None) -> Any:
        """
        Gather type embeddings.

        Parameters
        ----------
        atype
            Atom types with shape (...). Valid type range is [0, ntypes-1].
            If omitted, return the complete embedding table, including the
            optional padding row. This form is used by graph-native descriptor
            ABIs that precompute the table once per forward call.

        Returns
        -------
        Array
            Gathered type embeddings with shape ``(..., embed_dim)`` when
            ``atype`` is provided. Otherwise, the complete table with shape
            ``(ntypes + int(padding), embed_dim)``.
        """
        # === Step 1. Return the complete graph-native lookup table ===
        if atype is None:
            xp = array_api_compat.array_namespace(self.adam_type_embedding)
            return xp_asarray_nodetach(
                xp,
                self.adam_type_embedding[...],
                device=array_api_compat.device(self.adam_type_embedding),
            )

        # === Step 2. Gather rows for an explicit atom-type tensor ===
        xp = array_api_compat.array_namespace(atype)
        weight = xp_asarray_nodetach(
            xp, self.adam_type_embedding[...], device=array_api_compat.device(atype)
        )
        # Flattening provides one backend-neutral gather while preserving every
        # leading batch or graph dimension on restoration.
        index = xp.astype(xp.reshape(atype, (-1,)), xp.int64)
        if self.padding:
            index = remap_atype_to_padding(index, self.ntypes + 1)
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

    def call(
        self,
        *,
        n_nodes: int,
        edge_cache: EdgeCache,
        radial_feat: Any,
        zonal_coupling: Any = None,
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
        # The fused operator spans this broadcast and the scatter of Step 4, so
        # it takes over whenever nothing else joins the message in between.
        if self._can_fuse_scatter(zonal_coupling) and edge_cache.csr_cache is not None:
            return self.forward_fused_scatter(
                n_nodes, edge_cache, radial_feat, zonal_coupling
            )

        radial_slot_index = xp_asarray_nodetach(
            xp, self.radial_slot_index_for_row, device=device
        )
        radial_value_for_row = xp.take(
            radial_feat, radial_slot_index, axis=1
        )  # (E, D-1, C)
        non_scalar_message = (
            zonal_coupling[:, :, None] * radial_value_for_row
        )  # (E, D-1, C)

        # === Step 4. Scatter to nodes and normalize ===
        # Destination scatter-add over ``edge_cache.dst`` (pt ``index_add_``),
        # applied after the validity masking below. This reduction is
        # layout-agnostic: it is correct both for the padded ``call`` (row-major
        # ``dst`` makes the accumulation order identical to a sum over the
        # ``nnei`` axis, hence bit-exact) and for the graph-native ``call_graph``
        # route (arbitrary ``dst`` order and per-node degree). The l=0 row is left at
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

    def _can_fuse_scatter(self, zonal_coupling: Any) -> bool:
        """Return whether a backend can run the fused scatter for this input."""
        return False

    def forward_fused_scatter(
        self,
        n_nodes: int,
        edge_cache: EdgeCache,
        radial_feat: Any,
        zonal_coupling: Any,
    ) -> Any:
        """Build and reduce the geometric message with a backend operator."""
        raise NotImplementedError(
            "The fused GIE scatter is unavailable in the dpmodel reference"
        )

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


def spin_environment_coordinates(moment: Any) -> Any:
    r"""Build homogeneous spin coordinates before applying the edge envelope.

    The output contains the vector, squared magnitude, and five normalized
    degree-two harmonics, with shape (..., 9). The harmonic normalization gives
    Q(s) . Q(t) = 3/2 (s . t)^2 - 1/2 |s|^2 |t|^2. All coordinates are
    polynomial at zero moment. The input moment already carries its type mask.
    """
    xp = array_api_compat.array_namespace(moment)
    x, y, z = moment[..., 0], moment[..., 1], moment[..., 2]
    magnitude = xp.sum(moment * moment, axis=-1, keepdims=True)
    root_three = math.sqrt(3.0)
    quadrupole = xp.stack(
        [
            root_three * x * y,
            root_three * y * z,
            0.5 * (2.0 * z * z - x * x - y * y),
            root_three * x * z,
            0.5 * root_three * (x * x - y * y),
        ],
        axis=-1,
    )
    return xp.concat([moment, magnitude, quadrupole], axis=-1)


def spin_environment_invariants(moments: Any, axis_dim: int) -> Any:
    """Contract normalized spin densities into independent invariant blocks.

    Parameters
    ----------
    moments
        Vector, magnitude and quadrupole densities with shape (N, 9, K).
    axis_dim
        Number of leading radial channels retained on the second Gram axis.

    Returns
    -------
    Array
        Concatenated magnitude density, vector Gram and quadrupole Gram with
        shape (N, K + 2 * K * axis_dim). No cross-family contractions occur.
    """
    xp = array_api_compat.array_namespace(moments)
    nodes, _, channels = moments.shape
    vector = moments[:, :3, :]
    quadrupole = moments[:, 4:, :]
    vector_gram = xp.matmul(xp.permute_dims(vector, (0, 2, 1)), vector[:, :, :axis_dim])
    quadrupole_gram = xp.matmul(
        xp.permute_dims(quadrupole, (0, 2, 1)), quadrupole[:, :, :axis_dim]
    )
    return xp.concat(
        [
            moments[:, 3, :],
            xp.reshape(vector_gram, (nodes, channels * axis_dim)),
            xp.reshape(quadrupole_gram, (nodes, channels * axis_dim)),
        ],
        axis=-1,
    )


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

    Native spin adds masked vector, squared-magnitude and homogeneous
    quadrupole densities to the same reduction. Their magnitude density and
    separate vector/quadrupole Grams feed an independent output projection;
    the geometric projection retains its own weights and contraction.

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
        Per-type spin flags. Masked neighbor vector, magnitude and quadrupole
        densities share the geometric radial/type weights and have independent
        invariant output weights. Non-magnetic moments contribute zero at every
        derivative order.
    precision : str
        Parameter precision.
    trainable : bool
        Whether parameters are trainable.
    seed : int | list[int] | None
        Random seed for reproducibility.
    """

    CONFIG_DERIVED_ARRAYS = ("spin_mask",)

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
        # Geometry and spin families share the radial/type weights and the
        # destination reduction. Independent contractions retain their distinct
        # spatial and time-reversal transformation laws.
        self.geometry_coord_dim = 4
        self.coord_dim = self.geometry_coord_dim + (
            9 if self.spin_flags is not None else 0
        )

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

        # === Native spin: masked densities and independent invariant readout ===
        if self.spin_flags is not None:
            self.spin_mask = np.array(
                [1.0 if flag else 0.0 for flag in self.spin_flags],
                dtype=PRECISION_DICT[self.precision.lower()],
            )
            spin_width = self.embed_dim * (1 + 2 * self.axis_dim)
            self.spin_output_proj = NativeLayer(
                spin_width,
                2 * self.channels,
                bias=False,
                activation_function=None,
                precision=self.precision,
                seed=child_seed(seed, 4),
                trainable=self.trainable,
            )
            self.spin_output_proj.w = np.zeros(
                (spin_width, 2 * self.channels),
                dtype=PRECISION_DICT[self.precision.lower()],
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
            Used only when ``use_spin`` is set. The masked source moment forms
            vector, magnitude and quadrupole coordinates before envelope gating.
            When ``None`` these coordinates are zero-padded.

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

        # === Step 1b. Form the masked spin families before envelope gating ===
        # Vector, magnitude and quadrupole coordinates are homogeneous in the
        # moment. The common C3 envelope multiplies each family once, retaining
        # smooth coordinate and spin derivatives at zero and at the cutoff.
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
                spin_chan = edge_env * spin_environment_coordinates(spin_src * mask)
            else:
                spin_chan = xp.zeros(
                    (r_tilde.shape[0], 9), dtype=r_tilde.dtype, device=device
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
        # Destination scatter-add over ``dst`` (pt ``index_add_``), applied after
        # the validity masking below. Layout-agnostic: correct for the padded
        # ``call`` (row-major ``dst`` keeps the accumulation order identical to a
        # sum over the ``nnei`` axis, hence bit-exact) and for the graph-native
        # ``call_graph`` route (arbitrary ``dst`` order and per-node degree).
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
        # Each family contracts independently: geometric invariants retain
        # their projection, while spin invariants receive separate output
        # weights without spin/geometry or odd-time-reversal cross terms.
        if self.spin_flags is None:
            env_agg_t = xp.permute_dims(env_agg, (0, 2, 1))
            env_agg_axis = env_agg[:, :, : self.axis_dim]
            D = xp.matmul(env_agg_t, env_agg_axis)
        else:
            geometry_agg = env_agg[:, : self.geometry_coord_dim, :]
            D = xp.matmul(
                xp.permute_dims(geometry_agg, (0, 2, 1)),
                geometry_agg[:, :, : self.axis_dim],
            )

        # === Step 6. Output projection for FiLM logits ===
        D_flat = xp.reshape(
            D, (n_nodes, self.embed_dim * self.axis_dim)
        )  # (N, embed_dim*axis_dim)
        logits = self.output_proj(D_flat)
        if self.spin_flags is not None:
            spin_features = spin_environment_invariants(
                env_agg[:, self.geometry_coord_dim :, :], self.axis_dim
            )
            logits = logits + self.spin_output_proj(spin_features)
        return logits

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
            variables["spin_output_proj.matrix"] = to_numpy_array(
                self.spin_output_proj.w
            )
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
            self.spin_output_proj.w = np.asarray(
                variables["spin_output_proj.matrix"], dtype=prec
            )

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
            ntypes=CHARGE_TABLE_ROWS,
            embed_dim=self.embed_dim,
            precision=self.precision,
            seed=child_seed(seed, 0),
            trainable=self.trainable,
            padding=False,
        )
        self.spin_embedding = SeZMTypeEmbedding(
            ntypes=MULTIPLICITY_TABLE_ROWS,
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
        charge = xp.astype(charge_spin[:, 0], xp.int64) + CHARGE_OFFSET
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


def spin_pair_invariants(
    moment: Any, src: Any, dst: Any, edge_vec: Any, spin_mode: str
) -> Any:
    r"""Form time-even pair scalars from masked moments and bond vectors.

    The exchange block is (|s_i|², |s_j|², s_i·s_j). The orbit block appends
    ((s_i·u)², (s_j·u)², (s_i·u)(s_j·u)), with u = r / sqrt(r² + 1e-20).
    The regularized direction is smooth at coincident positions and transforms
    as a polar vector. Only the simultaneous reversal of all moments is imposed;
    the signed exchange correlation distinguishes parallel and antiparallel spins.
    """
    xp = array_api_compat.array_namespace(moment)
    si = xp.take(moment, xp.astype(dst, xp.int64), axis=0)
    sj = xp.take(moment, xp.astype(src, xp.int64), axis=0)
    features = [
        xp.sum(si * si, axis=-1),
        xp.sum(sj * sj, axis=-1),
        xp.sum(si * sj, axis=-1),
    ]
    if spin_mode == "orbit":
        u = edge_vec / xp.sqrt(
            xp.sum(edge_vec * edge_vec, axis=-1, keepdims=True) + 1e-20
        )
        pi = xp.sum(si * u, axis=-1)
        pj = xp.sum(sj * u, axis=-1)
        features.extend([pi * pi, pj * pj, pi * pj])
    return xp.stack(features, axis=-1)


class SpinEmbedding(NativeOP):
    r"""Embed time-even on-site and pair spin invariants.

    ``exchange`` uses the two squared moments and their signed dot product.
    ``orbit`` also uses the two squared bond projections and their product.
    For an orthogonal spatial map R, a moment transforms as det(R) R s:
    each bond projection gains det(R), so every emitted product is invariant.
    All inputs have even total spin order; arbitrary subsequent nonlinearities
    therefore preserve global time reversal. A single-site flip is not a
    symmetry because the signed pair dot product changes sign.

    The on-site map conditions scalar type features. The pair map conditions
    the shared radial basis and hence every interaction block. Its output is
    multiplied by the edge envelope, including the exclusion mask.
    Both output maps start at zero to preserve spin-free initialization.
    """

    CONFIG_DERIVED_ARRAYS = ("spin_mask",)

    def __init__(
        self,
        *,
        ntypes: int,
        channels: int,
        n_radial: int,
        spin_mode: str = "orbit",
        use_spin: list[bool],
        activation_function: str = "silu",
        precision: str = DEFAULT_PRECISION,
        seed: int | list[int] | None = None,
        trainable: bool = True,
    ) -> None:
        self.ntypes = int(ntypes)
        self.channels = int(channels)
        self.n_radial = int(n_radial)
        self.spin_mode = spin_mode
        if spin_mode not in {"exchange", "orbit"}:
            raise ValueError("`spin_mode` must be exchange or orbit")
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
        # Configuration-derived (hence ``CONFIG_DERIVED_ARRAYS``): rebuilt on
        # construction and moved with the module, so the deterministic mask
        # never enters the serialized state.
        self.spin_mask = np.array(
            [1.0 if bool(flag) else 0.0 for flag in use_spin], dtype=prec
        )

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
        self.mag_layer2.w = np.zeros(
            (self.channels, self.channels),
            dtype=prec,
        )

        self.pair_layer = NativeLayer(
            3 if spin_mode == "exchange" else 6,
            self.n_radial,
            bias=False,
            activation_function=None,
            precision=self.precision,
            seed=child_seed(seed, 1),
            trainable=self.trainable,
        )
        self.pair_layer.w = np.zeros(
            (3 if spin_mode == "exchange" else 6, self.n_radial), dtype=prec
        )

    def conditioned_spin(self, spin: Any, atype: Any) -> Any:
        """Mask non-magnetic moments before forming any contraction."""
        xp = array_api_compat.array_namespace(spin)
        mask = xp_asarray_nodetach(
            xp, self.spin_mask, device=array_api_compat.device(spin)
        )
        return (
            xp.astype(spin, get_xp_precision(xp, self.precision))
            * xp.take(mask, xp.astype(atype, xp.int64), axis=0)[:, None]
        )

    def call(self, spin: Any, atype: Any) -> Any:
        """Return scalar type conditioning with shape (N, channels)."""
        xp = array_api_compat.array_namespace(spin)
        moment = self.conditioned_spin(spin, atype)
        magnitude = xp.sum(moment * moment, axis=-1, keepdims=True)
        # Subtract the activation at zero, including for sigmoid-like choices.
        hidden = self.mag_layer1(magnitude) - self.mag_layer1(
            xp.zeros((1, 1), dtype=moment.dtype, device=array_api_compat.device(moment))
        )
        return self.mag_layer2(hidden)

    def edge_features(self, spin: Any, atype: Any, edge_cache: EdgeCache) -> Any:
        """Return enveloped pair conditioning with shape (E, n_radial)."""
        moment = self.conditioned_spin(spin, atype)
        features = spin_pair_invariants(
            moment, edge_cache.src, edge_cache.dst, edge_cache.edge_vec, self.spin_mode
        )
        return self.pair_layer(features) * edge_cache.edge_env

    def _variables(self) -> dict[str, np.ndarray]:
        """Return variables keyed by the PyTorch state names."""
        return {
            "mag_layer1.matrix": to_numpy_array(self.mag_layer1.w),
            "mag_layer2.matrix": to_numpy_array(self.mag_layer2.w),
            "pair_layer.matrix": to_numpy_array(self.pair_layer.w),
        }

    def _load_variables(self, variables: dict[str, Any]) -> None:
        """Load the scalar and pair projections."""
        prec = PRECISION_DICT[self.precision.lower()]
        for name in ("mag_layer1", "mag_layer2", "pair_layer"):
            getattr(self, name).w = np.asarray(variables[f"{name}.matrix"], dtype=prec)

    def serialize(self) -> dict[str, Any]:
        """Serialize the SpinEmbedding to a dict."""
        return {
            "@class": "SpinEmbedding",
            "@version": 1,
            "config": {
                "ntypes": self.ntypes,
                "channels": self.channels,
                "n_radial": self.n_radial,
                "spin_mode": self.spin_mode,
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
