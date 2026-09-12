# SPDX-License-Identifier: LGPL-3.0-or-later
"""Uni-Mol v1 transformer encoder, ported to the array-API dpmodel layer.

The maths follows Uni-Mol (https://github.com/deepmodeling/Uni-Mol) at commit
90f52c4 and the Uni-Core modules it builds on
(https://github.com/dptech-corp/Uni-Core) at commit ace6fae, both MIT licensed:

    Copyright (c) DP Technology
    This source code is licensed under the MIT license found in the LICENSE
    file in the root directory of that source tree.

Parts of Uni-Core derive in turn from fairseq (Copyright (c) Facebook, Inc. and
its affiliates, MIT licensed).

Ported files and the classes taken from them:

======================================== ==========================================
upstream                                 classes
======================================== ==========================================
unicore/modules/multihead_attention.py   :class:`SelfMultiheadAttention`
unicore/modules/transformer_encoder_layer.py :class:`TransformerEncoderLayer`
unimol/models/transformer_encoder_with_pair.py :class:`TransformerEncoderWithPair`
unimol/models/unimol.py                  :class:`GaussianLayer`, :class:`NonLinearHead`
======================================== ==========================================

The port keeps upstream's exact order of operations, so that a checkpoint
converted from Uni-Mol reproduces upstream outputs to floating-point rounding.
Two upstream quirks are reproduced deliberately and are marked in the code: the
truncated ``pi`` in the Gaussian basis, and the fp32 cast inside the norm
regularisers.
"""

import array_api_compat

from deepmd.dpmodel.common import (
    NativeOP,
)
from deepmd.dpmodel.utils.network import (
    LayerNorm,
    NativeLayer,
)

# Uni-Mol's Gaussian basis uses a truncated pi (unimol/models/unimol.py:393-397).
# Keeping it is required for bitwise agreement with the released weights.
UNIMOL_PI = 3.14159


def softmax(x, axis: int = -1):  # noqa: ANN001, ANN201
    """Numerically stable softmax over ``axis``.

    Rows that are entirely ``-inf`` would produce NaN, as they do upstream.
    Uni-Mol never builds such a row: the BOS column is never masked.
    """
    xp = array_api_compat.array_namespace(x)
    x_max = xp.max(x, axis=axis, keepdims=True)
    e = xp.exp(x - x_max)
    return e / xp.sum(e, axis=axis, keepdims=True)


def dropout(x, p: float, training: bool):  # noqa: ANN001, ANN201
    """Apply dropout, but only while training.

    deepmd has no dropout anywhere else, and the array API has no random
    numbers, so this dispatches to torch when a training step actually needs
    it. Inference, which is what the array-API backends are for, is the
    identity. Training on a non-torch backend is refused rather than silently
    dropping the regularisation, which would be a quiet parity bug.
    """
    if not training or p <= 0.0:
        return x
    if array_api_compat.is_torch_array(x):
        import torch

        return torch.nn.functional.dropout(x, p=p, training=True)
    raise NotImplementedError(
        "dropout during training is only implemented for the PyTorch backends; "
        "the array-API path is for inference"
    )


def norm_loss(x, eps: float = 1e-10, tolerance: float = 1.0):  # noqa: ANN001, ANN201
    """Hinge on the deviation of the row norm from ``sqrt(dim)``.

    Mirrors ``norm_loss`` in unimol/models/transformer_encoder_with_pair.py:101.
    Upstream evaluates this in fp32 because it pretrains a pure-fp16 model; the
    cast is reproduced so that the value matches upstream exactly.
    """
    xp = array_api_compat.array_namespace(x)
    x = xp.astype(x, xp.float32)
    max_norm = x.shape[-1] ** 0.5
    norm = xp.sqrt(xp.sum(x**2, axis=-1) + eps)
    error = xp.abs(norm - max_norm) - tolerance
    return xp.where(error > 0, error, xp.zeros_like(error))


def masked_mean(mask, value, axis=-1, eps: float = 1e-10):  # noqa: ANN001, ANN201
    """Mean of ``value`` over ``mask``, then mean over what is left.

    Mirrors ``masked_mean`` in transformer_encoder_with_pair.py:108. The ``eps``
    in the denominator is what makes an all-padding row return 0 rather than NaN.
    """
    xp = array_api_compat.array_namespace(value)
    mask = xp.astype(mask, value.dtype)
    num = xp.sum(mask * value, axis=axis)
    den = eps + xp.sum(mask, axis=axis)
    return xp.mean(num / den)


class NonLinearHead(NativeOP):
    """Two-layer head, ``linear1 -> activation -> linear2``.

    Mirrors unimol/models/unimol.py:347.
    """

    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        activation_function: str = "gelu_erf",
        hidden: int | None = None,
        precision: str = "float64",
        seed: int | list[int] | None = None,
    ) -> None:
        hidden = hidden or input_dim
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.activation_function = activation_function
        self.hidden = hidden
        self.precision = precision
        self.linear1 = NativeLayer(
            input_dim,
            hidden,
            activation_function=activation_function,
            precision=precision,
            seed=seed,
        )
        self.linear2 = NativeLayer(
            hidden, out_dim, activation_function=None, precision=precision, seed=seed
        )

    def call(self, x):  # noqa: ANN001, ANN201
        return self.linear2(self.linear1(x))

    def serialize(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "out_dim": self.out_dim,
            "hidden": self.hidden,
            "activation_function": self.activation_function,
            "precision": self.precision,
            "linear1": self.linear1.serialize(),
            "linear2": self.linear2.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "NonLinearHead":
        data = data.copy()
        linear1 = data.pop("linear1")
        linear2 = data.pop("linear2")
        obj = cls(**data)
        obj.linear1 = NativeLayer.deserialize(linear1)
        obj.linear2 = NativeLayer.deserialize(linear2)
        return obj


class GaussianLayer(NativeOP):
    """Gaussian radial basis with a per-edge-type affine on the distance.

    Mirrors unimol/models/unimol.py:400. ``means`` and ``stds`` are shared by all
    edge types; ``mul`` and ``bias`` hold one scalar per ordered element pair.
    """

    def __init__(
        self,
        k: int = 128,
        edge_types: int = 1024,
        single_precision_basis: bool = True,
        precision: str = "float64",
        seed: int | list[int] | None = None,
    ) -> None:
        import numpy as np

        self.k = k
        self.edge_types = edge_types
        # Upstream evaluates the basis in fp32 because it pretrains an fp16
        # model (unimol/models/unimol.py:418-420). Keeping that is what makes
        # the released weights reproduce; set False to keep the working dtype.
        self.single_precision_basis = single_precision_basis
        self.precision = precision
        rng = np.random.default_rng(seed if isinstance(seed, int) else None)
        # Upstream initialises means/stds as U(0, 3) and mul/bias as 1/0, but
        # init_bert_params then overwrites all four with N(0, 0.02); see
        # unicore/modules/transformer_encoder.py:16 and the note in the spec.
        self.means = rng.uniform(0.0, 3.0, size=(1, k))
        self.stds = rng.uniform(0.0, 3.0, size=(1, k))
        self.mul = np.ones((edge_types, 1))
        self.bias = np.zeros((edge_types, 1))

    def call(self, dist, edge_type):  # noqa: ANN001, ANN201
        """Expand ``dist`` (nf x nt x nt) into ``k`` Gaussians per atom pair."""
        xp = array_api_compat.array_namespace(dist)
        dev = array_api_compat.device(dist)
        mul = xp.reshape(
            xp.take(
                xp.asarray(self.mul, device=dev), xp.reshape(edge_type, (-1,)), axis=0
            ),
            (*edge_type.shape, 1),
        )
        bias = xp.reshape(
            xp.take(
                xp.asarray(self.bias, device=dev), xp.reshape(edge_type, (-1,)), axis=0
            ),
            (*edge_type.shape, 1),
        )
        mul = xp.astype(mul, dist.dtype)
        bias = xp.astype(bias, dist.dtype)
        x = mul * dist[..., None] + bias
        x = xp.repeat(x, self.k, axis=-1)
        work = xp.float32 if self.single_precision_basis else x.dtype
        x = xp.astype(x, work)
        mean = xp.astype(xp.reshape(xp.asarray(self.means, device=dev), (-1,)), work)
        std = (
            xp.abs(
                xp.astype(xp.reshape(xp.asarray(self.stds, device=dev), (-1,)), work)
            )
            + 1e-5
        )
        a = (2 * UNIMOL_PI) ** 0.5
        out = xp.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)
        return xp.astype(out, dist.dtype)

    def serialize(self) -> dict:
        from deepmd.dpmodel.common import (
            to_numpy_array,
        )

        return {
            "k": self.k,
            "edge_types": self.edge_types,
            "single_precision_basis": self.single_precision_basis,
            "precision": self.precision,
            "@variables": {
                "means": to_numpy_array(self.means),
                "stds": to_numpy_array(self.stds),
                "mul": to_numpy_array(self.mul),
                "bias": to_numpy_array(self.bias),
            },
        }

    @classmethod
    def deserialize(cls, data: dict) -> "GaussianLayer":
        data = data.copy()
        variables = data.pop("@variables")
        obj = cls(**data)
        for key, value in variables.items():
            setattr(obj, key, value)
        return obj


class SelfMultiheadAttention(NativeOP):
    """Self-attention with a fused QKV projection and an additive pair bias.

    Mirrors unicore/modules/multihead_attention.py:12. Only the ``return_attn``
    path of upstream is kept, because Uni-Mol always takes it: the pre-softmax
    logits are the pair representation that the next layer biases with.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        scaling_factor: float = 1.0,
        precision: str = "float64",
        seed: int | list[int] | None = None,
    ) -> None:
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = (self.head_dim * scaling_factor) ** -0.5
        self.precision = precision
        self.in_proj = NativeLayer(
            embed_dim, embed_dim * 3, bias=bias, precision=precision, seed=seed
        )
        self.out_proj = NativeLayer(
            embed_dim, embed_dim, bias=bias, precision=precision, seed=seed
        )

    def call(self, query, attn_bias, training: bool = False):  # noqa: ANN001, ANN201
        """Return the attended values and the biased pre-softmax logits.

        Parameters
        ----------
        query
            nf x nt x embed_dim.
        attn_bias
            (nf * num_heads) x nt x nt, already carrying ``-inf`` on padded keys.

        Returns
        -------
        tuple
            ``(output, attn_weights)``, shaped nf x nt x embed_dim and
            (nf * num_heads) x nt x nt.
        """
        xp = array_api_compat.array_namespace(query)
        nf, nt, _ = query.shape
        qkv = self.in_proj(query)
        q, k, v = (
            qkv[..., i * self.embed_dim : (i + 1) * self.embed_dim] for i in range(3)
        )

        def split_heads(t):  # noqa: ANN001, ANN202
            t = xp.reshape(t, (nf, nt, self.num_heads, self.head_dim))
            t = xp.permute_dims(t, (0, 2, 1, 3))
            return xp.reshape(t, (nf * self.num_heads, nt, self.head_dim))

        q = split_heads(q) * self.scaling
        k = split_heads(k)
        v = split_heads(v)

        attn_weights = q @ xp.permute_dims(k, (0, 2, 1))
        # Upstream adds the bias in place and returns the biased logits, which
        # become the next layer's bias (multihead_attention.py:100-103).
        attn_weights = attn_weights + attn_bias
        attn = dropout(softmax(attn_weights, axis=-1), self.dropout, training)
        o = attn @ v
        o = xp.reshape(o, (nf, self.num_heads, nt, self.head_dim))
        o = xp.permute_dims(o, (0, 2, 1, 3))
        o = xp.reshape(o, (nf, nt, self.embed_dim))
        return self.out_proj(o), attn_weights

    def serialize(self) -> dict:
        return {
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "precision": self.precision,
            "in_proj": self.in_proj.serialize(),
            "out_proj": self.out_proj.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "SelfMultiheadAttention":
        data = data.copy()
        in_proj = data.pop("in_proj")
        out_proj = data.pop("out_proj")
        obj = cls(**data)
        obj.in_proj = NativeLayer.deserialize(in_proj)
        obj.out_proj = NativeLayer.deserialize(out_proj)
        return obj


class TransformerEncoderLayer(NativeOP):
    """Pre-layer-norm transformer block that also returns attention logits.

    Mirrors unicore/modules/transformer_encoder_layer.py:15. Uni-Mol never sets
    ``post_ln``, so only the pre-LN order is implemented.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        activation_function: str = "gelu_erf",
        precision: str = "float64",
        seed: int | list[int] | None = None,
    ) -> None:
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.precision = precision
        self.self_attn = SelfMultiheadAttention(
            embed_dim,
            attention_heads,
            dropout=attention_dropout,
            precision=precision,
            seed=seed,
        )
        self.self_attn_layer_norm = LayerNorm(embed_dim, precision=precision, seed=seed)
        self.fc1 = NativeLayer(
            embed_dim,
            ffn_embed_dim,
            activation_function=activation_function,
            precision=precision,
            seed=seed,
        )
        self.fc2 = NativeLayer(
            ffn_embed_dim,
            embed_dim,
            activation_function=None,
            precision=precision,
            seed=seed,
        )
        self.final_layer_norm = LayerNorm(embed_dim, precision=precision, seed=seed)

    def call(self, x, attn_bias, training: bool = False):  # noqa: ANN001, ANN201
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn_weights = self.self_attn(x, attn_bias=attn_bias, training=training)
        x = dropout(x, self.dropout, training)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        # fc1 carries the activation, so the activation dropout sits between
        # the two linears, as upstream has it.
        x = dropout(self.fc1(x), self.activation_dropout, training)
        x = dropout(self.fc2(x), self.dropout, training)
        x = residual + x
        return x, attn_weights

    def serialize(self) -> dict:
        return {
            "embed_dim": self.embed_dim,
            "ffn_embed_dim": self.ffn_embed_dim,
            "attention_heads": self.attention_heads,
            "dropout": self.dropout,
            "attention_dropout": self.attention_dropout,
            "activation_dropout": self.activation_dropout,
            "activation_function": self.activation_function,
            "precision": self.precision,
            "self_attn": self.self_attn.serialize(),
            "self_attn_layer_norm": self.self_attn_layer_norm.serialize(),
            "fc1": self.fc1.serialize(),
            "fc2": self.fc2.serialize(),
            "final_layer_norm": self.final_layer_norm.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "TransformerEncoderLayer":
        data = data.copy()
        parts = {
            key: data.pop(key)
            for key in (
                "self_attn",
                "self_attn_layer_norm",
                "fc1",
                "fc2",
                "final_layer_norm",
            )
        }
        obj = cls(**data)
        obj.self_attn = SelfMultiheadAttention.deserialize(parts["self_attn"])
        obj.self_attn_layer_norm = LayerNorm.deserialize(parts["self_attn_layer_norm"])
        obj.fc1 = NativeLayer.deserialize(parts["fc1"])
        obj.fc2 = NativeLayer.deserialize(parts["fc2"])
        obj.final_layer_norm = LayerNorm.deserialize(parts["final_layer_norm"])
        return obj


class TransformerEncoderWithPair(NativeOP):
    """Stack of pre-LN blocks that carries a pair representation.

    Mirrors unimol/models/transformer_encoder_with_pair.py:14. Each block's
    pre-softmax logits become the next block's bias, so the pair representation
    is the running sum of per-layer logits. Returns the node representation, the
    final pair representation, the pair delta, and the two norm regularisers.
    """

    def __init__(
        self,
        encoder_layers: int = 6,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 8,
        emb_dropout: float = 0.1,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        max_seq_len: int = 256,
        activation_function: str = "gelu_erf",
        no_final_head_layer_norm: bool = False,
        precision: str = "float64",
        seed: int | list[int] | None = None,
    ) -> None:
        self.encoder_layers = encoder_layers
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.emb_dropout = emb_dropout
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.max_seq_len = max_seq_len
        self.activation_function = activation_function
        self.no_final_head_layer_norm = no_final_head_layer_norm
        self.precision = precision
        self.emb_layer_norm = LayerNorm(embed_dim, precision=precision, seed=seed)
        self.final_layer_norm = LayerNorm(embed_dim, precision=precision, seed=seed)
        self.final_head_layer_norm = (
            None
            if no_final_head_layer_norm
            else LayerNorm(attention_heads, precision=precision, seed=seed)
        )
        self.layers = [
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                ffn_embed_dim=ffn_embed_dim,
                attention_heads=attention_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
                activation_function=activation_function,
                precision=precision,
                seed=seed,
            )
            for _ in range(encoder_layers)
        ]

    def call(self, emb, attn_mask, padding_mask, training: bool = False):  # noqa: ANN001, ANN201
        """Run the stack.

        Parameters
        ----------
        emb
            nf x nt x embed_dim token embedding.
        attn_mask
            (nf * heads) x nt x nt bias from the Gaussian basis.
        padding_mask
            nf x nt, 1 on padded positions.

        Returns
        -------
        tuple
            ``(x, pair_rep, delta_pair_rep, x_norm, delta_pair_rep_norm)``.
        """
        xp = array_api_compat.array_namespace(emb)
        nf, nt = emb.shape[0], emb.shape[1]
        x = dropout(self.emb_layer_norm(emb), self.emb_dropout, training)
        pad = xp.astype(padding_mask, x.dtype)
        x = x * (1 - pad[..., None])

        # Upstream merges padding into the bias as -inf on padded key columns,
        # so the attention itself never sees a key_padding_mask. It does this
        # in place, which aliases the saved input bias; the delta below is
        # therefore taken against the already-filled bias, and the resulting
        # NaN on padded pairs is overwritten with 0 right after.
        neg_inf = xp.asarray(float("-inf"), dtype=attn_mask.dtype)
        key_pad = xp.astype(padding_mask, xp.bool)[:, None, None, :]
        attn_mask = xp.reshape(attn_mask, (nf, -1, nt, nt))
        attn_mask = xp.where(key_pad, neg_inf, attn_mask)
        input_attn_mask = attn_mask
        attn_mask = xp.reshape(attn_mask, (-1, nt, nt))

        for layer in self.layers:
            x, attn_mask = layer(x, attn_bias=attn_mask, training=training)

        token_mask = 1.0 - pad
        x_norm = masked_mean(token_mask, norm_loss(x))

        x = self.final_layer_norm(x)

        # Padded pairs are dropped from the delta anyway. Upstream reaches that
        # by computing -inf minus -inf, getting NaN, and overwriting it with 0;
        # zeroing both operands first gives the same values without the NaN.
        out_pair = xp.reshape(attn_mask, (nf, -1, nt, nt))
        zero = xp.zeros_like(out_pair)
        delta_pair_repr = xp.where(key_pad, zero, out_pair) - xp.where(
            key_pad, zero, input_attn_mask
        )
        pair_rep = xp.permute_dims(out_pair, (0, 2, 3, 1))
        delta_pair_repr = xp.permute_dims(delta_pair_repr, (0, 2, 3, 1))

        pair_mask = token_mask[..., None] * token_mask[..., None, :]
        delta_pair_repr_norm = masked_mean(
            pair_mask, norm_loss(delta_pair_repr), axis=(-1, -2)
        )

        if self.final_head_layer_norm is not None:
            delta_pair_repr = self.final_head_layer_norm(delta_pair_repr)

        return x, pair_rep, delta_pair_repr, x_norm, delta_pair_repr_norm

    def serialize(self) -> dict:
        return {
            "encoder_layers": self.encoder_layers,
            "embed_dim": self.embed_dim,
            "ffn_embed_dim": self.ffn_embed_dim,
            "attention_heads": self.attention_heads,
            "emb_dropout": self.emb_dropout,
            "dropout": self.dropout,
            "attention_dropout": self.attention_dropout,
            "activation_dropout": self.activation_dropout,
            "max_seq_len": self.max_seq_len,
            "activation_function": self.activation_function,
            "no_final_head_layer_norm": self.no_final_head_layer_norm,
            "precision": self.precision,
            "emb_layer_norm": self.emb_layer_norm.serialize(),
            "final_layer_norm": self.final_layer_norm.serialize(),
            "final_head_layer_norm": None
            if self.final_head_layer_norm is None
            else self.final_head_layer_norm.serialize(),
            "layers": [layer.serialize() for layer in self.layers],
        }

    @classmethod
    def deserialize(cls, data: dict) -> "TransformerEncoderWithPair":
        data = data.copy()
        emb_ln = data.pop("emb_layer_norm")
        final_ln = data.pop("final_layer_norm")
        head_ln = data.pop("final_head_layer_norm")
        layers = data.pop("layers")
        obj = cls(**data)
        obj.emb_layer_norm = LayerNorm.deserialize(emb_ln)
        obj.final_layer_norm = LayerNorm.deserialize(final_ln)
        obj.final_head_layer_norm = (
            None if head_ln is None else LayerNorm.deserialize(head_ln)
        )
        obj.layers = [TransformerEncoderLayer.deserialize(layer) for layer in layers]
        return obj
