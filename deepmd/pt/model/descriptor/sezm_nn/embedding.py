# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Embedding layers for the SeZM descriptor.

This module defines the type embedding, geometric initial embedding, and
environment-seed embedding used to initialize SeZM node features.
"""

from __future__ import (
    annotations,
)

import math
from typing import (
    TYPE_CHECKING,
    Any,
)

import torch
import torch.nn as nn

from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pt.model.network.mlp import (
    MLPLayer,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
    RESERVED_PRECISION_DICT,
)
from deepmd.pt.utils.utils import (
    get_generator,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .indexing import (
    build_gie_zonal_index,
    get_so3_dim_of_lmax,
)
from .utils import (
    np_safe,
    safe_numpy_to_tensor,
)

if TYPE_CHECKING:
    from .edge_cache import (
        EdgeFeatureCache,
    )


class SeZMTypeEmbedding(nn.Module):
    """
    Minimal SeZM type embedding with Adam-routed parameter naming.

    Parameters
    ----------
    ntypes
        Number of atom types.
    embed_dim
        Embedding dimension.
    dtype
        Parameter dtype.
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
        dtype: torch.dtype,
        seed: int | list[int] | None = None,
        trainable: bool,
        padding: bool = True,
    ) -> None:
        super().__init__()
        self.ntypes = int(ntypes)
        self.embed_dim = int(embed_dim)
        self.dtype = dtype
        self.seed = seed
        self.device = env.DEVICE
        self.padding = bool(padding)
        if self.ntypes <= 0:
            raise ValueError("`ntypes` must be positive")
        if self.embed_dim <= 0:
            raise ValueError("`embed_dim` must be positive")

        # === Step 1. Build embedding table parameter ===
        n_rows = self.ntypes + int(self.padding)
        self.adam_type_embedding = nn.Parameter(
            torch.empty(
                n_rows,
                self.embed_dim,
                device=self.device,
                dtype=self.dtype,
            )
        )

        # === Step 2. Initialize active type rows with default normal scale ===
        init_std = 1.0 / math.sqrt(float(self.ntypes + self.embed_dim))
        nn.init.normal_(
            self.adam_type_embedding[: self.ntypes],
            mean=0.0,
            std=init_std,
            generator=get_generator(child_seed(seed, 0)),
        )
        if self.padding:
            with torch.no_grad():
                self.adam_type_embedding[self.ntypes].zero_()

        for p in self.parameters():
            p.requires_grad = trainable

    def forward(self, atype: torch.Tensor) -> torch.Tensor:
        """
        Gather type embeddings.

        Parameters
        ----------
        atype
            Atom types with shape (...,). Valid type range is [0, ntypes-1].

        Returns
        -------
        torch.Tensor
            Type embeddings with shape (..., embed_dim).
        """
        return torch.embedding(self.adam_type_embedding, atype)


class GeometricInitialEmbedding(nn.Module):
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
    dtype
        Parameter dtype.
    """

    def __init__(
        self,
        *,
        lmax: int,
        channels: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.channels = int(channels)
        self.ebed_dim = get_so3_dim_of_lmax(self.lmax)
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[dtype]
        (
            node_row_index,
            node_zonal_m0_col_index,
            node_radial_l_index,
        ) = build_gie_zonal_index(self.lmax, device=self.device)
        # One aligned entry per non-scalar node row: output row, local m=0
        # column, and the matching radial degree slot.
        self.register_buffer("non_scalar_row_index", node_row_index, persistent=True)
        self.register_buffer(
            "zonal_m0_col_index_for_row",
            node_zonal_m0_col_index,
            persistent=True,
        )
        self.register_buffer(
            "radial_slot_index_for_row",
            node_radial_l_index,
            persistent=True,
        )

    def forward(
        self,
        *,
        n_nodes: int,
        edge_cache: EdgeFeatureCache,
        radial_feat: torch.Tensor,
        zonal_coupling: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
        torch.Tensor
            Initial features to add with shape (N, D, C). l=0 is guaranteed zero.
        """
        # === Step 1. Initialize output ===
        device = edge_cache.edge_vec.device
        dtype = edge_cache.edge_vec.dtype
        out = torch.zeros(
            n_nodes, self.ebed_dim, self.channels, device=device, dtype=dtype
        )  # (N, D, C)
        if self.lmax == 0:
            return out

        # === Step 2. Gather all m=0 columns (l >= 1) in one shot ===
        # Advanced indexing pairs one packed non-scalar row with the zonal m=0 column
        # from the same degree block in Dt_full.
        if zonal_coupling is None:
            Dt_full = edge_cache.Dt_full  # (E, D, D)
            zonal_coupling = Dt_full[
                :,
                self.non_scalar_row_index,
                self.zonal_m0_col_index_for_row,
            ]  # (E, D-1)

        # === Step 3. Broadcast radial features per row ===
        # Each non-scalar packed row reuses the radial feature of its degree l.
        radial_value_for_row = radial_feat.index_select(
            1, self.radial_slot_index_for_row
        )  # (E, D-1, C)
        non_scalar_message = (
            zonal_coupling.unsqueeze(-1) * radial_value_for_row
        )  # (E, D-1, C)

        # === Step 4. Source Freeze Propagation Gate (optional) ===
        # Mute messages emitted by nodes whose local neighborhood enters
        # the frozen zone. ``edge_src_gate`` is ``None`` outside bridging
        # mode so this is a no-op in normal training.
        src_gate = edge_cache.edge_src_gate
        if src_gate is not None:
            non_scalar_message = non_scalar_message * src_gate.to(
                dtype=non_scalar_message.dtype
            ).unsqueeze(-1)

        # === Step 5. Scatter to nodes and normalize ===
        # Avoid advanced-index writeback (out[:, non_scalar_row_index, :]) which produces a copy.
        non_scalar_out = out.new_zeros(
            n_nodes, self.non_scalar_row_index.numel(), self.channels
        )  # (N, D-1, C)
        non_scalar_out.index_add_(0, edge_cache.dst, non_scalar_message)
        out[:, self.non_scalar_row_index, :] = non_scalar_out
        out.mul_(edge_cache.inv_sqrt_deg)
        return out

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "GeometricInitialEmbedding",
            "@version": 1,
            "lmax": self.lmax,
            "channels": self.channels,
            "precision": RESERVED_PRECISION_DICT[self.dtype],
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> GeometricInitialEmbedding:
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "GeometricInitialEmbedding":
            raise ValueError(f"Invalid class for GeometricInitialEmbedding: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        precision = data.pop("precision")
        data["dtype"] = PRECISION_DICT[precision]
        return cls(**data)


class EnvironmentInitialEmbedding(nn.Module):
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
    dtype : torch.dtype
        Parameter dtype.
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
        dtype: torch.dtype,
        trainable: bool,
        seed: int | list[int] | None = None,
    ) -> None:
        super().__init__()

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
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[dtype]
        self.register_buffer(
            "eps_sq_tensor",
            torch.tensor(self.eps * self.eps, dtype=self.dtype, device=self.device),
            persistent=False,
        )

        # === RBF projection: n_radial -> rbf_out_dim (two-layer MLP) ===
        # rbf_out_dim = max(32, embed_dim - 2*type_dim) to align G-network width to embed_dim
        # First layer: n_radial -> rbf_out_dim with activation
        # Second layer: rbf_out_dim -> rbf_out_dim linear
        self.rbf_out_dim = max(32, self.embed_dim - 2 * self.type_dim)
        seed_rbf_proj = child_seed(seed, 0)
        self.rbf_proj_layer1 = MLPLayer(
            self.n_radial,
            self.rbf_out_dim,
            bias=self.mlp_bias,
            activation_function=self.activation_function,
            precision=self.precision,
            seed=child_seed(seed_rbf_proj, 0),
        )
        self.rbf_proj_layer2 = MLPLayer(
            self.rbf_out_dim,
            self.rbf_out_dim,
            bias=self.mlp_bias,
            activation_function=None,
            precision=self.precision,
            seed=child_seed(seed_rbf_proj, 1),
        )

        # === Independent type embedding: ntypes -> type_dim ===
        # Individual type embedding
        seed_type_embed = child_seed(seed, 1)
        self.env_type_embed = SeZMTypeEmbedding(
            ntypes=self.ntypes,
            embed_dim=self.type_dim,
            dtype=self.dtype,
            seed=seed_type_embed,
            trainable=trainable,
        )

        # === G network: (rbf_out_dim + 2*type_dim) -> hidden_dim -> embed_dim ===
        seed_g_net = child_seed(seed, 2)
        g_in_dim = self.rbf_out_dim + 2 * self.type_dim
        self.g_layer1 = MLPLayer(
            g_in_dim,
            self.hidden_dim,
            bias=self.mlp_bias,
            activation_function=self.activation_function,
            precision=self.precision,
            seed=child_seed(seed_g_net, 0),
        )
        self.g_layer2 = MLPLayer(
            self.hidden_dim,
            self.embed_dim,
            bias=self.mlp_bias,
            activation_function=None,
            precision=self.precision,
            seed=child_seed(seed_g_net, 1),
        )

        # === Output projection: embed_dim * axis_dim -> 2*channels ===
        # Zero init so FiLM logits start at zero; strengths control magnitude.
        seed_out = child_seed(seed, 3)
        self.output_proj = MLPLayer(
            self.embed_dim * self.axis_dim,
            2 * self.channels,
            bias=False,
            activation_function=None,
            init="final",
            precision=self.precision,
            seed=seed_out,
        )

        for p in self.parameters():
            p.requires_grad = trainable

    def forward(
        self,
        *,
        edge_cache: EdgeFeatureCache,
        atype_flat: torch.Tensor,
        n_nodes: int,
    ) -> torch.Tensor:
        """
        Compute environment FiLM logits for l=0 conditioning.

        Parameters
        ----------
        edge_cache : EdgeFeatureCache
            Edge cache containing src, dst, edge_vec, edge_rbf, edge_env.
        atype_flat : torch.Tensor
            Flattened atom types with shape (N,), where N = nf * nloc.
        n_nodes : int
            Number of nodes (N = nf * nloc).

        Returns
        -------
        torch.Tensor
            FiLM logits with shape (N, 2*channels).
        """
        src, dst = edge_cache.src, edge_cache.dst
        edge_vec = edge_cache.edge_vec  # (E, 3)
        edge_rbf = edge_cache.edge_rbf  # (E, n_radial)
        edge_env = edge_cache.edge_env  # (E, 1)

        # === Step 1. Construct r_tilde = [s, s*r_hat] ===
        # s = edge_env * (1/r), r_hat = edge_vec / r
        r_sq = (edge_vec * edge_vec).sum(dim=-1, keepdim=True)  # (E, 1)
        inv_r = torch.rsqrt(r_sq + self.eps_sq_tensor)  # (E, 1)
        s = edge_env * inv_r  # (E, 1)
        r_hat = edge_vec * inv_r  # (E, 3)
        r_tilde = torch.cat([s, s * r_hat], dim=-1)  # (E, 4)

        # === Step 2. Compute G network input and output ===
        # Use independent type embeddings (decoupled from main type embedding)
        atype_src = atype_flat.index_select(0, src)  # (E,)
        atype_dst = atype_flat.index_select(0, dst)  # (E,)
        type_src = self.env_type_embed(atype_src)  # (E, type_dim)
        type_dst = self.env_type_embed(atype_dst)  # (E, type_dim)

        # Project edge_rbf to rbf_out_dim (two-layer MLP)
        rbf_proj = self.rbf_proj_layer2(
            self.rbf_proj_layer1(edge_rbf)
        )  # (E, rbf_out_dim)

        # G network input: concat projected RBF and type embeddings
        g_input = torch.cat([rbf_proj, type_src, type_dst], dim=-1)  # (E, g_in_dim)
        g = self.g_layer2(self.g_layer1(g_input))  # (E, embed_dim)

        # === Step 3. Aggregate outer product by destination node ===
        # outer = r_tilde[:, :, None] * g[:, None, :]  # (E, 4, embed_dim)
        outer = torch.einsum("ei,ej->eij", r_tilde, g)  # (E, 4, embed_dim)
        outer_flat = outer.reshape(-1, 4 * self.embed_dim)  # (E, 4*embed_dim)
        # Source Freeze Propagation Gate: mute the outer-product contribution
        # of any edge whose source node has a neighbor in the frozen zone.
        src_gate = edge_cache.edge_src_gate
        if src_gate is not None:
            outer_flat = outer_flat * src_gate.to(dtype=outer_flat.dtype)
        env_agg = outer_flat.new_zeros(n_nodes, 4 * self.embed_dim)  # (N, 4*embed_dim)
        env_agg.index_add_(0, dst, outer_flat)
        env_agg = env_agg.reshape(n_nodes, 4, self.embed_dim)  # (N, 4, embed_dim)

        # === Step 4. Smooth normalization by envelope-squared degree ===
        # Reuse the cache's inverse-sqrt degree so the version-aware
        # ``deg_norm_floor`` is applied consistently with GIE.
        env_agg = env_agg * edge_cache.inv_sqrt_deg

        # === Step 5. D matrix construction: D = env_agg^T @ env_agg[:,:,:axis_dim] ===
        env_agg_t = env_agg.permute(0, 2, 1)  # (N, embed_dim, 4)
        env_agg_axis = env_agg[:, :, : self.axis_dim]  # (N, 4, axis_dim)
        D = torch.bmm(env_agg_t, env_agg_axis)  # (N, embed_dim, axis_dim)

        # === Step 6. Output projection for FiLM logits ===
        D_flat = D.reshape(
            n_nodes, self.embed_dim * self.axis_dim
        )  # (N, embed_dim*axis_dim)
        return self.output_proj(D_flat)

    def serialize(self) -> dict[str, Any]:
        trainable = all(p.requires_grad for p in self.parameters())
        state = self.state_dict()
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
                "precision": self.precision,
                "trainable": trainable,
                "seed": None,
            },
            "@variables": {key: np_safe(value) for key, value in state.items()},
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


class ChargeSpinEmbedding(nn.Module):
    """
    Frame-level charge and spin embedding for scalar type features.

    Parameters
    ----------
    embed_dim
        Embedding dimension.
    activation_function
        Activation function used by the mixing layer.
    dtype
        Parameter dtype.
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
        dtype: torch.dtype,
        seed: int | list[int] | None = None,
        trainable: bool,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.activation_function = str(activation_function)
        self.dtype = dtype
        self.precision = RESERVED_PRECISION_DICT[dtype]
        if self.embed_dim <= 0:
            raise ValueError("`embed_dim` must be positive")

        self.charge_embedding = SeZMTypeEmbedding(
            ntypes=200,
            embed_dim=self.embed_dim,
            dtype=self.dtype,
            seed=child_seed(seed, 0),
            trainable=trainable,
            padding=False,
        )
        self.spin_embedding = SeZMTypeEmbedding(
            ntypes=100,
            embed_dim=self.embed_dim,
            dtype=self.dtype,
            seed=child_seed(seed, 1),
            trainable=trainable,
            padding=False,
        )
        self.mix_layer = MLPLayer(
            2 * self.embed_dim,
            self.embed_dim,
            activation_function=self.activation_function,
            precision=self.precision,
            seed=child_seed(seed, 2),
            trainable=trainable,
        )

        for p in self.parameters():
            p.requires_grad = trainable

    def forward(self, charge_spin: torch.Tensor) -> torch.Tensor:
        """
        Embed frame-level charge and spin.

        Parameters
        ----------
        charge_spin
            Frame charge and spin values with shape (nf, 2).

        Returns
        -------
        torch.Tensor
            Mixed condition embedding with shape (nf, embed_dim).
        """
        charge = charge_spin[:, 0].to(dtype=torch.int64) + 100
        spin = charge_spin[:, 1].to(dtype=torch.int64)
        charge_embed = self.charge_embedding(charge)
        spin_embed = self.spin_embedding(spin)
        return self.mix_layer(torch.cat((charge_embed, spin_embed), dim=-1))
