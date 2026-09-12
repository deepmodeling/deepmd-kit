# SPDX-License-Identifier: LGPL-3.0-or-later
"""The Uni-Mol v1 backbone as a deepmd descriptor.

Wraps the ported Uni-Mol encoder (:mod:`deepmd.dpmodel.descriptor.unimol_nn`) in
the descriptor interface. Unlike every other descriptor here, Uni-Mol is global
rather than local: it attends over all atom pairs with no cut-off and no smooth
envelope, so it is neither extensive nor periodic and its forces are not
conserved. It is meant for molecular property and pretraining work, and the
model layer refuses to pair it with an energy fitting.

Uni-Mol's own vocabulary is kept, because the released weights are indexed by
it: four special tokens, then 26 elements, then ``[MASK]``. A deepmd
``type_map`` is mapped onto those ids, and an element outside the vocabulary
becomes ``[UNK]``.
"""

import array_api_compat

from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.common import (
    NativeOP,
)
from deepmd.dpmodel.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.dpmodel.descriptor.unimol_nn import (
    GaussianLayer,
    NonLinearHead,
    TransformerEncoderWithPair,
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

# Uni-Mol's molecular dictionary, in file order. The order is load-bearing: it
# fixes the embedding rows, the edge-type ids (t_i * V + t_j) and the output
# width of the element head in the released checkpoints.
UNIMOL_SPECIAL_TOKENS = ["[PAD]", "[CLS]", "[SEP]", "[UNK]"]
UNIMOL_ELEMENTS = [
    "C", "N", "O", "S", "H", "Cl", "F", "Br", "I", "Si", "P", "B", "Na", "K",
    "Al", "Ca", "Sn", "As", "Hg", "Fe", "Zn", "Cr", "Se", "Gd", "Au", "Li",
]  # fmt: skip
UNIMOL_MASK_TOKEN = "[MASK]"


def unimol_vocabulary() -> list[str]:
    """Return the 31 Uni-Mol tokens in checkpoint order."""
    return [*UNIMOL_SPECIAL_TOKENS, *UNIMOL_ELEMENTS, UNIMOL_MASK_TOKEN]


@BaseDescriptor.register("unimol")
class DescrptUniMol(NativeOP, BaseDescriptor):
    r"""Uni-Mol v1 transformer backbone.

    Every atom attends to every other atom; geometry enters only through
    pairwise distances, expanded in a Gaussian basis whose affine parameters are
    specific to the ordered element pair, and injected as a per-head attention
    bias. Two virtual tokens wrap each molecule, and the running sum of
    per-layer attention logits is the pair representation that the pretraining
    heads read.

    Parameters
    ----------
    type_map : list[str]
        Element names, mapped onto the Uni-Mol vocabulary.
    encoder_layers : int
        Number of transformer blocks.
    encoder_embed_dim : int
        Width of the node representation.
    encoder_ffn_embed_dim : int
        Width of the feed-forward hidden layer.
    encoder_attention_heads : int
        Number of attention heads, which is also the width of the pair channel.
    max_atoms : int
        Largest molecule accepted, which fixes ``sel``.
    max_seq_len : int
        Upstream's sequence guard, kept for configuration compatibility.
    activation_function : str
        Activation of the blocks and heads. Uni-Mol uses the exact GELU.
    dropout, emb_dropout, attention_dropout, activation_dropout : float
        Dropout rates. They are inert in this backend-agnostic implementation
        and are applied by the PyTorch-Exportable wrapper during training.
    no_final_head_layer_norm : bool
        Skip the layer norm on the pair delta. Upstream builds that norm unless
        its weight is negative.
    single_precision_basis : bool
        Evaluate the Gaussian basis in fp32, which is what upstream does.
    single_precision_distance : bool
        Round the pairwise distances to fp32 before the Gaussian basis. Upstream
        precomputes its distance matrix in fp32 in the data pipeline, so this
        reproduces its numbers; the default computes them in the working
        precision, which is more accurate and is what gradients flow through.
    virtual_token_position : str
        Where the two virtual tokens sit. ``"centroid"`` places them at the
        centroid of the real atoms, which keeps the sequence translation
        invariant. ``"origin"`` places them at the origin, reproducing upstream
        exactly for data that its own pipeline has already centred.
    precision : str
        Floating-point precision of the parameters.
    seed : int, optional
        Random seed for initialization.
    """

    def __init__(
        self,
        type_map: list[str],
        encoder_layers: int = 15,
        encoder_embed_dim: int = 512,
        encoder_ffn_embed_dim: int = 2048,
        encoder_attention_heads: int = 64,
        max_atoms: int = 256,
        max_seq_len: int = 512,
        activation_function: str = "gelu_erf",
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        no_final_head_layer_norm: bool = False,
        single_precision_basis: bool = True,
        single_precision_distance: bool = False,
        virtual_token_position: str = "centroid",
        gaussian_kernels: int = 128,
        precision: str = "float64",
        seed: int | list[int] | None = None,
        type_map_tokens: list[str] | None = None,
        **kwargs: float | str | bool,
    ) -> None:
        del kwargs
        self.type_map = list(type_map)
        self.encoder_layers = encoder_layers
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_ffn_embed_dim = encoder_ffn_embed_dim
        self.encoder_attention_heads = encoder_attention_heads
        self.max_atoms = max_atoms
        self.max_seq_len = max_seq_len
        self.activation_function = activation_function
        self.dropout = dropout
        self.emb_dropout = emb_dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.no_final_head_layer_norm = no_final_head_layer_norm
        self.single_precision_basis = single_precision_basis
        self.single_precision_distance = single_precision_distance
        if virtual_token_position not in ("centroid", "origin"):
            raise ValueError(
                f"virtual_token_position must be centroid or origin, got {virtual_token_position}"
            )
        self.virtual_token_position = virtual_token_position
        self.gaussian_kernels = gaussian_kernels
        self.precision = precision
        self.vocabulary = type_map_tokens or unimol_vocabulary()
        self.ntokens = len(self.vocabulary)

        import numpy as np

        token_of = {sym: i for i, sym in enumerate(self.vocabulary)}
        self.pad_idx = token_of["[PAD]"]
        self.bos_idx = token_of["[CLS]"]
        self.eos_idx = token_of["[SEP]"]
        self.unk_idx = token_of["[UNK]"]
        # type id -> Uni-Mol token id; unknown elements fall back to [UNK].
        self.type_to_token = np.array(
            [token_of.get(sym, self.unk_idx) for sym in self.type_map], dtype=np.int64
        )

        # The token embedding is trained, so it is a layer rather than a bare
        # array: a bare array becomes a buffer on the torch backends, and a
        # buffer never receives a gradient.
        self.embed_tokens = NativeLayer(
            self.ntokens,
            encoder_embed_dim,
            bias=False,
            precision=precision,
            seed=child_seed(seed, 0),
        )
        self.gbf = GaussianLayer(
            gaussian_kernels,
            self.ntokens**2,
            single_precision_basis=single_precision_basis,
            precision=precision,
            seed=child_seed(seed, 1),
        )
        self.gbf_proj = NonLinearHead(
            gaussian_kernels,
            encoder_attention_heads,
            activation_function,
            precision=precision,
            seed=child_seed(seed, 2),
        )
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=encoder_layers,
            embed_dim=encoder_embed_dim,
            ffn_embed_dim=encoder_ffn_embed_dim,
            attention_heads=encoder_attention_heads,
            emb_dropout=emb_dropout,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            max_seq_len=max_seq_len,
            activation_function=activation_function,
            no_final_head_layer_norm=no_final_head_layer_norm,
            precision=precision,
            seed=child_seed(seed, 3),
        )

    # ------------------------------------------------------------------
    # capability queries
    # ------------------------------------------------------------------
    def get_rcut(self) -> float:
        """All pairs are neighbours, so the radius is effectively unbounded."""
        return float(self.max_seq_len) * 1e3

    def get_rcut_smth(self) -> float:
        """No smoothing exists; smoothing starts where the cut-off is."""
        return self.get_rcut()

    def get_sel(self) -> list[int]:
        """One entry, because the neighbour list is type-blind."""
        return [self.max_atoms - 1]

    def get_ntypes(self) -> int:
        """Number of element types."""
        return len(self.type_map)

    def get_type_map(self) -> list[str]:
        """Element names."""
        return self.type_map

    def get_dim_out(self) -> int:
        """Width of the node representation."""
        return self.encoder_embed_dim

    def get_dim_emb(self) -> int:
        """Width of the pair channel, which is the head count."""
        return self.encoder_attention_heads

    def mixed_types(self) -> bool:
        """The neighbour list is not split by type."""
        return True

    def has_message_passing(self) -> bool:
        """No ghost-atom exchange."""
        return False

    def need_sorted_nlist_for_lower(self) -> bool:
        """All pairs take part, so their order does not matter."""
        return False

    def get_env_protection(self) -> float:
        """No environment-matrix protection is used."""
        return 0.0

    def supports_edge_parallel(self) -> bool:
        """Global attention cannot be split across ranks."""
        return False

    def dense_lower_supports_comm(self) -> bool:
        """There is no ghost-communication path."""
        return False

    def compression_needs_min_nbor_dist(self) -> bool:
        """Compression is not supported, so no neighbour statistics are needed."""
        return False

    def compute_input_stats(self, merged, path=None) -> None:  # noqa: ANN001
        """No environment statistics: the basis is learned, not normalized."""

    def set_stat_mean_and_stddev(self, mean, stddev) -> None:  # noqa: ANN001
        """Stat-free descriptor; nothing to assign."""

    def get_stat_mean_and_stddev(self) -> tuple[list, list]:
        """Stat-free descriptor; no statistics to report."""
        return [], []

    def share_params(self, base_class, shared_level, resume=False) -> None:  # noqa: ANN001
        """Parameter sharing is implemented by the PyTorch-Exportable wrapper."""
        raise NotImplementedError

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat: object = None
    ) -> None:
        """Remap the element names onto Uni-Mol tokens.

        The token embedding itself is indexed by Uni-Mol token, not by deepmd
        type, so only the lookup table changes.
        """
        import numpy as np

        token_of = {sym: i for i, sym in enumerate(self.vocabulary)}
        self.type_map = list(type_map)
        self.type_to_token = np.array(
            [token_of.get(sym, self.unk_idx) for sym in self.type_map], dtype=np.int64
        )

    @classmethod
    def update_sel(cls, train_data, type_map, local_jdata: dict) -> tuple[dict, None]:  # noqa: ANN001
        """``sel`` follows from ``max_atoms``, so neighbour statistics are moot."""
        return local_jdata.copy(), None

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def build_tokens(self, coord_ext: Array, atype_ext: Array, nlist: Array) -> dict:
        """Turn a padded deepmd frame into Uni-Mol's token sequence.

        Real atoms are identified from the neighbour list rather than from
        ``atype``: by the time a descriptor is called, virtual atoms have been
        clamped to type 0 and are indistinguishable from a real first element,
        whereas the neighbour list still shows them as empty rows.

        Returns
        -------
        dict
            ``tokens``, ``coord``, ``padding_mask``, ``real_mask`` and
            ``n_real``; the sequence is ``[CLS] atoms [SEP] pad...``.
        """
        xp = array_api_compat.array_namespace(coord_ext)
        dev = array_api_compat.device(coord_ext)
        nf, nloc = nlist.shape[0], nlist.shape[1]
        coord = xp.reshape(coord_ext, (nf, -1, 3))
        nall = coord.shape[1]
        if nall != nloc:
            raise ValueError(
                "the unimol descriptor needs every atom to be local: it attends "
                "over all pairs, so it supports neither periodic images nor the "
                "ghost-atom layout that freezing and parallel evaluation assume "
                f"(got {nall} extended atoms for {nloc} local atoms)"
            )

        real_mask = xp.any(nlist >= 0, axis=-1)
        n_real = xp.sum(xp.astype(real_mask, xp.int64), axis=-1)
        if bool(xp.any(n_real < 2)):
            raise ValueError(
                "the unimol descriptor needs at least two real atoms per frame; "
                "single-atom frames cannot be told apart from padding"
            )

        real = xp.astype(real_mask, coord.dtype)
        # BOS and EOS sit at the centroid of the real atoms, which keeps the
        # sequence translation invariant. Upstream centres the coordinates in
        # its data pipeline and then places both at the origin, so the two agree
        # whenever the data went through that transform.
        if self.virtual_token_position == "centroid":
            centroid = (
                xp.sum(coord * real[..., None], axis=1)
                / xp.astype(n_real, coord.dtype)[:, None]
            )
        else:
            centroid = xp.zeros((nf, 3), dtype=coord.dtype, device=dev)

        # The lookup table is plain integer data rather than a parameter, so it
        # does not travel with the module and has to be placed explicitly.
        token_table = xp.asarray(self.type_to_token, device=dev)
        atom_tokens = xp.take(token_table, xp.reshape(atype_ext, (-1,)), axis=0)
        atom_tokens = xp.reshape(atom_tokens, (nf, nloc))
        atom_tokens = xp.where(
            real_mask, atom_tokens, xp.full_like(atom_tokens, self.pad_idx)
        )

        nt = nloc + 2
        positions = xp.arange(nt, device=dev)[None, :]
        eos_at = (n_real + 1)[:, None]
        bos_row = xp.full((nf, 1), self.bos_idx, dtype=atom_tokens.dtype, device=dev)
        pad_row = xp.full((nf, 1), self.pad_idx, dtype=atom_tokens.dtype, device=dev)
        tokens = xp.concat([bos_row, atom_tokens, pad_row], axis=1)
        tokens = xp.where(
            positions == eos_at, xp.full_like(tokens, self.eos_idx), tokens
        )

        virtual = centroid[:, None, :]
        coord_full = xp.concat([virtual, coord, virtual], axis=1)
        at_virtual = (positions == 0) | (positions == eos_at)
        coord_full = xp.where(
            at_virtual[..., None],
            xp.broadcast_to(virtual, coord_full.shape),
            coord_full,
        )

        padding_mask = xp.astype(tokens == self.pad_idx, coord.dtype)
        return {
            "tokens": tokens,
            "coord": coord_full,
            "padding_mask": padding_mask,
            "real_mask": real_mask,
            "n_real": n_real,
        }

    def forward_tokens(self, coord_ext: Array, atype_ext: Array, nlist: Array) -> dict:
        """Run the backbone and return everything at token resolution.

        The five-tuple of :meth:`call` cannot carry the two virtual tokens or
        the norm regularisers, which the pretraining heads need, so the
        pretraining path goes through here instead.
        """
        xp = array_api_compat.array_namespace(coord_ext)
        seq = self.build_tokens(coord_ext, atype_ext, nlist)
        tokens, coord, padding_mask = seq["tokens"], seq["coord"], seq["padding_mask"]
        nf, nt = tokens.shape

        embed = xp.asarray(
            self.embed_tokens.w, device=array_api_compat.device(coord_ext)
        )
        emb = xp.take(embed, xp.reshape(tokens, (-1,)), axis=0)
        emb = xp.reshape(emb, (nf, nt, self.encoder_embed_dim))
        emb = xp.astype(emb, coord.dtype)

        diff = coord[:, :, None, :] - coord[:, None, :, :]
        dist = xp.sqrt(xp.sum(diff**2, axis=-1))
        if self.single_precision_distance:
            # Upstream's data pipeline stores the distance matrix in fp32, and
            # the Gaussian basis is narrow enough for that rounding to matter.
            dist = xp.astype(xp.astype(dist, xp.float32), dist.dtype)
        edge_type = tokens[:, :, None] * self.ntokens + tokens[:, None, :]

        bias = self.gbf_proj(self.gbf(dist, edge_type))
        bias = xp.reshape(xp.permute_dims(bias, (0, 3, 1, 2)), (-1, nt, nt))

        # ``training`` exists only once the PyTorch-Exportable wrapper has made
        # this a torch module; on the array-API path it is always inference.
        x, pair_rep, delta_pair_rep, x_norm, delta_pair_norm = self.encoder(
            emb, bias, padding_mask, training=bool(getattr(self, "training", False))
        )
        # Upstream clears the -inf that padding leaves on the pair channel
        # before any head reads it (unimol/models/unimol.py:221).
        pair_rep = xp.where(xp.isinf(pair_rep), xp.zeros_like(pair_rep), pair_rep)
        return {
            **seq,
            "node_ebd": x,
            "pair_rep": pair_rep,
            "delta_pair_rep": delta_pair_rep,
            "x_norm": x_norm,
            "delta_pair_norm": delta_pair_norm,
        }

    def call(
        self,
        coord_ext: Array,
        atype_ext: Array,
        nlist: Array,
        mapping: Array | None = None,
        fparam: Array | None = None,
        comm_dict: dict | None = None,
        charge_spin: Array | None = None,
    ) -> tuple[Array, None, None, None, None]:
        """Return the per-atom representation.

        The two virtual tokens are dropped so that the output lines up with the
        local atoms. Slots the backbone does not produce are ``None``, which the
        atomic model accepts; the pretraining path uses :meth:`forward_tokens`.
        """
        del mapping, fparam, comm_dict, charge_spin
        out = self.forward_tokens(coord_ext, atype_ext, nlist)
        nloc = nlist.shape[1]
        return out["node_ebd"][:, 1 : nloc + 1, :], None, None, None, None

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------
    def serialize(self) -> dict:
        """Serialize the descriptor."""
        return {
            "@class": "Descriptor",
            "type": "unimol",
            "@version": 1,
            "type_map": self.type_map,
            "encoder_layers": self.encoder_layers,
            "encoder_embed_dim": self.encoder_embed_dim,
            "encoder_ffn_embed_dim": self.encoder_ffn_embed_dim,
            "encoder_attention_heads": self.encoder_attention_heads,
            "max_atoms": self.max_atoms,
            "max_seq_len": self.max_seq_len,
            "activation_function": self.activation_function,
            "dropout": self.dropout,
            "emb_dropout": self.emb_dropout,
            "attention_dropout": self.attention_dropout,
            "activation_dropout": self.activation_dropout,
            "no_final_head_layer_norm": self.no_final_head_layer_norm,
            "single_precision_basis": self.single_precision_basis,
            "single_precision_distance": self.single_precision_distance,
            "virtual_token_position": self.virtual_token_position,
            "gaussian_kernels": self.gaussian_kernels,
            "precision": self.precision,
            "type_map_tokens": self.vocabulary,
            "embed_tokens": self.embed_tokens.serialize(),
            "gbf": self.gbf.serialize(),
            "gbf_proj": self.gbf_proj.serialize(),
            "encoder": self.encoder.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptUniMol":
        """Deserialize the descriptor."""
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class", None)
        data.pop("type", None)
        embed_tokens = data.pop("embed_tokens")
        gbf = data.pop("gbf")
        gbf_proj = data.pop("gbf_proj")
        encoder = data.pop("encoder")
        obj = cls(**data)
        obj.embed_tokens = NativeLayer.deserialize(embed_tokens)
        obj.gbf = GaussianLayer.deserialize(gbf)
        obj.gbf_proj = NonLinearHead.deserialize(gbf_proj)
        obj.encoder = TransformerEncoderWithPair.deserialize(encoder)
        return obj
