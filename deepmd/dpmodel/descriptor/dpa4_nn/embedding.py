# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Embedding layers for the dpmodel DPA4/SeZM descriptor.

This module is the dpmodel port of
``deepmd.pt.model.descriptor.sezm_nn.embedding``. It defines the type
embedding, geometric initial embedding, and environment-seed embedding used
to initialize SeZM node features.

Padded-edge layout
------------------
The pt implementation aggregates sparse per-edge messages into nodes with
``index_add_``. The dpmodel port uses the padded, frame-explicit edge layout
of :class:`~deepmd.dpmodel.descriptor.dpa4_nn.edge_cache.EdgeCache`
(``E = nf * nloc * nnei`` with invalid slots marked by ``edge_mask == 0``),
so every destination aggregation becomes a masked sum over the ``nnei`` axis
of the ``(N, nnei, ...)`` reshape. Each rewrite is commented with the pt
line it replaces.

Ported / skipped classes
------------------------
- ``SeZMTypeEmbedding``, ``GeometricInitialEmbedding`` and
  ``EnvironmentInitialEmbedding`` are ported (core consumers:
  ``sezm.py:710``, ``sezm.py:826`` and ``sezm.py:733`` respectively).
- ``ChargeSpinEmbedding`` (pt ``embedding.py:591``) is NOT ported: it is
  constructed only when ``add_chg_spin_ebd=True`` (``sezm.py:717``), and the
  flag defaults to ``False`` (``sezm.py:440``), so it is outside the core
  DPA4 configuration targeted by this port.
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
from deepmd.dpmodel.common import (
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

from .indexing import (
    build_gie_zonal_index,
    get_so3_dim_of_lmax,
)

if TYPE_CHECKING:
    from .edge_cache import (
        EdgeCache,
    )


def _edge_layout(n_edge: int, n_nodes: int) -> int:
    """Validate the padded-edge layout and return ``nnei = E // N``."""
    if n_nodes <= 0 or n_edge % n_nodes != 0:
        raise ValueError(
            "padded-edge layout requires E to be a multiple of N; "
            f"got E={n_edge}, N={n_nodes}"
        )
    return n_edge // n_nodes


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
        Floating-point precision of the embedding table.
    seed
        Random seed for initialization.
    trainable
        Whether parameters are trainable.
    padding
        Whether to append one all-zero padding row.

    Notes
    -----
    The parameter is named with ``adam_`` prefix so HybridMuon routes it to
    Adam (the name matches the pt ``state_dict`` key ``adam_type_embedding``).
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

        # === Step 1+2. Build the table; active rows N(0, init_std), padding
        # row zero (pt embedding.py:103-124). The numpy RNG stream differs
        # from pt's torch generator; weight values are not bit-compatible.
        init_std = 1.0 / math.sqrt(float(self.ntypes + self.embed_dim))
        rng = np.random.default_rng(child_seed(seed, 0))
        table = rng.normal(scale=init_std, size=(self.ntypes, self.embed_dim))
        if self.padding:
            table = np.concatenate([table, np.zeros((1, self.embed_dim))], axis=0)
        self.adam_type_embedding = table.astype(prec)

    def call(self, atype: Any) -> Any:
        """
        Gather type embeddings.

        Parameters
        ----------
        atype
            Atom types with shape (...,). Valid type range is [0, ntypes-1]
            (plus the padding row index ``ntypes`` when ``padding=True``).
            Negative type ids are invalid input and are NOT validated here
            (caller contract).

        Returns
        -------
        Array
            Type embeddings with shape (..., embed_dim).
        """
        xp = array_api_compat.array_namespace(atype)
        weight = xp.asarray(
            self.adam_type_embedding[...], device=array_api_compat.device(atype)
        )
        # pt embedding.py:143 torch.embedding -> flat int64 take + reshape.
        index = xp.astype(xp.reshape(atype, (-1,)), xp.int64)
        out = xp.take(weight, index, axis=0)
        return xp.reshape(out, (*atype.shape, self.embed_dim))

    def serialize(self) -> dict[str, Any]:
        """Serialize to a dict.

        The pt class has no ``serialize()``; the ``@variables`` key here
        matches the pt ``state_dict()`` key (``adam_type_embedding``).
        """
        return {
            "@class": "SeZMTypeEmbedding",
            "@version": 1,
            "config": {
                "ntypes": self.ntypes,
                "embed_dim": self.embed_dim,
                "padding": self.padding,
                "precision": self.precision.lower(),
                "trainable": self.trainable,
                "seed": None,
            },
            "@variables": {
                "adam_type_embedding": to_numpy_array(self.adam_type_embedding)
            },
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> SeZMTypeEmbedding:
        """Deserialize from a dict."""
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
        table = np.asarray(variables["adam_type_embedding"], dtype=prec)
        if table.shape != obj.adam_type_embedding.shape:
            raise ValueError(
                f"adam_type_embedding shape {table.shape} does not match "
                f"the expected shape {obj.adam_type_embedding.shape}"
            )
        obj.adam_type_embedding = table
        return obj


class GeometricInitialEmbedding(NativeOP):
    """
    Geometric initial embedding that adds zonal (m=0) rotated features.

    This module rotates pre-computed radial features for each degree l >= 1
    using the zonal (m=0) column of the cached inverse Wigner-D blocks
    (local->global). The l=0 component is not computed here since it comes
    from type embedding.

    Parameters
    ----------
    lmax
        Maximum node degree for the initial embedding.
    channels
        Number of channels per (l, m) coefficient.
    precision
        Floating-point precision label (kept for config parity with pt; the
        computation follows the input dtype).
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
        # One aligned entry per non-scalar node row: output row, local m=0
        # column, and the matching radial degree slot (static int64 tables;
        # pt registers them as persistent buffers, embedding.py:185-195).
        (
            self.non_scalar_row_index,
            self.zonal_m0_col_index_for_row,
            self.radial_slot_index_for_row,
        ) = build_gie_zonal_index(self.lmax)

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
            Per-edge cache containing geometry, weights, and Wigner-D blocks
            in the padded layout (``E = n_nodes * nnei``).
        radial_feat
            Per-edge radial features with shape (E, lmax, C) for l=1..lmax.
        zonal_coupling
            Optional precomputed zonal coupling with shape (E, D-1). If None,
            it is gathered from ``edge_cache.Dt_full``.

        Returns
        -------
        Array
            Initial features to add with shape (N, D, C). l=0 is guaranteed
            zero.
        """
        # === Step 1. Initialize output ===
        xp = array_api_compat.array_namespace(edge_cache.edge_vec)
        device = array_api_compat.device(edge_cache.edge_vec)
        dtype = edge_cache.edge_vec.dtype
        if self.lmax == 0:
            # pt embedding.py:226-230: zeros short-circuit.
            return xp.zeros(
                (n_nodes, self.ebed_dim, self.channels), dtype=dtype, device=device
            )
        n_edge = int(edge_cache.dst.shape[0])
        nnei = _edge_layout(n_edge, int(n_nodes))

        # === Step 2. Gather all m=0 columns (l >= 1) in one shot ===
        # pt embedding.py:235-241 pairs one packed non-scalar row with the
        # zonal m=0 column from the same degree block via advanced indexing
        # Dt_full[:, rows, cols]; here this becomes a flat row-major take.
        if zonal_coupling is None:
            Dt_full = edge_cache.Dt_full  # (E, D, D)
            dim_full = Dt_full.shape[-1]
            flat_index = xp.asarray(
                self.non_scalar_row_index * dim_full + self.zonal_m0_col_index_for_row,
                device=device,
            )
            zonal_coupling = xp.take(
                xp.reshape(Dt_full, (n_edge, dim_full * dim_full)),
                flat_index,
                axis=1,
            )  # (E, D-1)

        # === Step 3. Broadcast radial features per row ===
        # Each non-scalar packed row reuses the radial feature of its degree l
        # (pt embedding.py:245-250, index_select on axis 1).
        radial_slot_index = xp.asarray(self.radial_slot_index_for_row, device=device)
        radial_value_for_row = xp.take(
            radial_feat, radial_slot_index, axis=1
        )  # (E, D-1, C)
        non_scalar_message = (
            zonal_coupling[:, :, None] * radial_value_for_row
        )  # (E, D-1, C)

        # === Step 4. Source Freeze Propagation Gate (optional) ===
        # pt embedding.py:256-260: mute messages emitted by nodes whose local
        # neighborhood enters the frozen zone; ``edge_src_gate`` is ``None``
        # outside bridging mode so this is a no-op in normal training.
        src_gate = edge_cache.edge_src_gate
        if src_gate is not None:
            non_scalar_message = non_scalar_message * xp.astype(
                xp.reshape(src_gate, (n_edge, 1, 1)), non_scalar_message.dtype
            )

        # === Step 5. Aggregate to nodes and normalize ===
        # pt embedding.py:264-267: non_scalar_out.index_add_(0, dst, msg) —
        # padded-edge masked sum over the nnei axis (dst is slot-implicit).
        edge_mask = edge_cache.edge_mask
        if edge_mask is not None:
            non_scalar_message = non_scalar_message * xp.astype(
                xp.reshape(edge_mask, (n_edge, 1, 1)), non_scalar_message.dtype
            )
        non_scalar_out = xp.sum(
            xp.reshape(
                non_scalar_message,
                (n_nodes, nnei, self.ebed_dim - 1, self.channels),
            ),
            axis=1,
        )  # (N, D-1, C)
        # pt embedding.py:268: out[:, non_scalar_row_index, :] = non_scalar_out
        # with row 0 (l=0) left at its zeros init (pt embedding.py:226).
        # ``non_scalar_row_index`` is the contiguous arange(1, D), so the
        # writeback is a concat with a zero l=0 row.
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
        # pt embedding.py:269: out.mul_(inv_sqrt_deg).
        out = out * xp.astype(edge_cache.inv_sqrt_deg, out.dtype)
        return xp.astype(out, dtype)

    def serialize(self) -> dict[str, Any]:
        """Serialize to a dict (config only; same flat layout as pt)."""
        return {
            "@class": "GeometricInitialEmbedding",
            "@version": 1,
            "lmax": self.lmax,
            "channels": self.channels,
            "precision": self.precision.lower(),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> GeometricInitialEmbedding:
        """Deserialize from a dict (accepts the pt ``serialize()`` output)."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "GeometricInitialEmbedding":
            raise ValueError(f"Invalid class for GeometricInitialEmbedding: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        return cls(
            lmax=int(data.pop("lmax")),
            channels=int(data.pop("channels")),
            precision=str(data.pop("precision")),
        )


class EnvironmentInitialEmbedding(NativeOP):
    """
    Environment matrix initial embedding for l=0 features.

    Computes an initial embedding based on the 4D environment matrix::

        [s, s * rx, s * ry, s * rz]

    Combined with independent type embeddings (individual type embedding),
    providing physical inductive bias for l=0 features.

    The computation follows the environment matrix approach where::

        1. Build `r_tilde = [s, s*r_hat]` where `s = edge_env / r` and
           `r_hat = edge_vec / r`
        2. G network: `g = G(rbf_proj(edge_rbf), type_src, type_dst)` produces
           per-edge features
           - Uses independent `env_type_embed` instead of projecting from the
             main type embedding
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
    precision : str
        Floating-point precision of the parameters.
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
        self.precision = precision
        self.trainable = bool(trainable)

        # === RBF projection: n_radial -> rbf_out_dim (two-layer MLP) ===
        # rbf_out_dim = max(32, embed_dim - 2*type_dim) to align G-network
        # width to embed_dim. First layer: n_radial -> rbf_out_dim with
        # activation. Second layer: rbf_out_dim -> rbf_out_dim linear.
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
        self.env_type_embed = SeZMTypeEmbedding(
            ntypes=self.ntypes,
            embed_dim=self.type_dim,
            precision=self.precision,
            seed=child_seed(seed, 1),
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
        # Zero init so FiLM logits start at zero (pt init="final",
        # embedding.py:447-455); strengths control magnitude.
        self.output_proj = NativeLayer(
            self.embed_dim * self.axis_dim,
            2 * self.channels,
            bias=False,
            activation_function=None,
            precision=self.precision,
            seed=child_seed(seed, 3),
            trainable=self.trainable,
        )
        self.output_proj.w = np.zeros_like(self.output_proj.w)

    def call(
        self,
        *,
        edge_cache: EdgeCache,
        atype_flat: Any,
        n_nodes: int,
    ) -> Any:
        """
        Compute environment FiLM logits for l=0 conditioning.

        Parameters
        ----------
        edge_cache : EdgeCache
            Edge cache containing src, dst, edge_vec, edge_rbf, edge_env in
            the padded layout (``E = n_nodes * nnei``).
        atype_flat : Array
            Flattened atom types with shape (N,), where N = nf * nloc.
        n_nodes : int
            Number of nodes (N = nf * nloc).

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
        n_edge = int(dst.shape[0])
        nnei = _edge_layout(n_edge, int(n_nodes))

        # === Step 1. Construct r_tilde = [s, s*r_hat] ===
        # s = edge_env * (1/r), r_hat = edge_vec / r (pt embedding.py:489-495)
        r_sq = xp.sum(edge_vec * edge_vec, axis=-1, keepdims=True)  # (E, 1)
        inv_r = 1.0 / xp.sqrt(r_sq + self.eps * self.eps)  # (E, 1)
        s = edge_env * inv_r  # (E, 1)
        r_hat = edge_vec * inv_r  # (E, 3)
        r_tilde = xp.concat([s, s * r_hat], axis=-1)  # (E, 4)

        # === Step 2. Compute G network input and output ===
        # Use independent type embeddings (decoupled from main type embedding)
        src_index = xp.astype(xp.reshape(src, (n_edge,)), xp.int64)
        dst_index = xp.astype(xp.reshape(dst, (n_edge,)), xp.int64)
        atype_src = xp.take(atype_flat, src_index, axis=0)  # (E,)
        atype_dst = xp.take(atype_flat, dst_index, axis=0)  # (E,)
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
        # pt embedding.py:515 einsum("ei,ej->eij") -> broadcast product.
        outer = r_tilde[:, :, None] * g[:, None, :]  # (E, 4, embed_dim)
        outer_flat = xp.reshape(outer, (n_edge, 4 * self.embed_dim))
        # Source Freeze Propagation Gate (pt embedding.py:519-521): mute the
        # outer-product contribution of any edge whose source node has a
        # neighbor in the frozen zone.
        src_gate = edge_cache.edge_src_gate
        if src_gate is not None:
            outer_flat = outer_flat * xp.astype(
                xp.reshape(src_gate, (n_edge, 1)), outer_flat.dtype
            )
        # pt embedding.py:522-523: env_agg.index_add_(0, dst, outer_flat) —
        # padded-edge masked sum over the nnei axis (dst is slot-implicit).
        edge_mask = edge_cache.edge_mask
        if edge_mask is not None:
            outer_flat = outer_flat * xp.astype(
                xp.reshape(edge_mask, (n_edge, 1)), outer_flat.dtype
            )
        env_agg = xp.sum(
            xp.reshape(outer_flat, (n_nodes, nnei, 4 * self.embed_dim)),
            axis=1,
        )  # (N, 4*embed_dim)
        env_agg = xp.reshape(env_agg, (n_nodes, 4, self.embed_dim))

        # === Step 4. Smooth normalization by envelope-squared degree ===
        # Reuse the cache's inverse-sqrt degree so the version-aware
        # ``deg_norm_floor`` is applied consistently with GIE.
        env_agg = env_agg * xp.astype(edge_cache.inv_sqrt_deg, env_agg.dtype)

        # === Step 5. D matrix: D = env_agg^T @ env_agg[:, :, :axis_dim] ===
        env_agg_t = xp.permute_dims(env_agg, (0, 2, 1))  # (N, embed_dim, 4)
        env_agg_axis = env_agg[:, :, : self.axis_dim]  # (N, 4, axis_dim)
        mat_d = xp.matmul(env_agg_t, env_agg_axis)  # (N, embed_dim, axis_dim)

        # === Step 6. Output projection for FiLM logits ===
        d_flat = xp.reshape(
            mat_d, (n_nodes, self.embed_dim * self.axis_dim)
        )  # (N, embed_dim*axis_dim)
        return self.output_proj(d_flat)

    def _variable_slots(self) -> dict[str, tuple[Any, str]]:
        """Map pt ``state_dict`` keys to (owner object, attribute name)."""
        slots: dict[str, tuple[Any, str]] = {}
        for name in ("rbf_proj_layer1", "rbf_proj_layer2", "g_layer1", "g_layer2"):
            layer = getattr(self, name)
            slots[f"{name}.matrix"] = (layer, "w")
            if self.mlp_bias:
                slots[f"{name}.bias"] = (layer, "b")
        slots["env_type_embed.adam_type_embedding"] = (
            self.env_type_embed,
            "adam_type_embedding",
        )
        slots["output_proj.matrix"] = (self.output_proj, "w")
        return slots

    def serialize(self) -> dict[str, Any]:
        """Serialize to a dict.

        The ``@variables`` keys match the pt ``state_dict()`` key names, so
        the pt ``serialize()`` output deserializes directly into this class
        (and vice versa).
        """
        variables = {
            key: to_numpy_array(getattr(owner, attr))
            for key, (owner, attr) in self._variable_slots().items()
        }
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
                "precision": self.precision.lower(),
                "trainable": self.trainable,
                "seed": None,
            },
            "@variables": variables,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> EnvironmentInitialEmbedding:
        """Deserialize from a dict (accepts the pt ``serialize()`` output)."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "EnvironmentInitialEmbedding":
            raise ValueError(f"Invalid class: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(**config)
        prec = PRECISION_DICT[obj.precision.lower()]
        slots = obj._variable_slots()
        if set(variables) != set(slots):
            raise ValueError(
                f"variable keys {sorted(variables)} do not match the expected "
                f"keys {sorted(slots)}"
            )
        for key, (owner, attr) in slots.items():
            value = np.asarray(variables[key], dtype=prec)
            expected_shape = getattr(owner, attr).shape
            if value.shape != expected_shape:
                raise ValueError(
                    f"shape of {key} {value.shape} does not match "
                    f"the expected shape {expected_shape}"
                )
            setattr(owner, attr, value)
        return obj
