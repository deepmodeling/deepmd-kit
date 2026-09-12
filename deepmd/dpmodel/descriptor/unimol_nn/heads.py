# SPDX-License-Identifier: LGPL-3.0-or-later
"""The three Uni-Mol v1 pretraining heads.

Ported from Uni-Mol (https://github.com/deepmodeling/Uni-Mol) at commit 90f52c4,
MIT licensed:

    Copyright (c) DP Technology
    This source code is licensed under the MIT license found in the LICENSE
    file in the root directory of that source tree.

``MaskLMHead`` predicts the element of every corrupted atom, ``coord_update``
denoises the coordinates through the pair channel, and ``DistanceHead``
predicts the clean pairwise distances. Together with the two norm regularisers
of the encoder these make up the five pretraining objectives.
"""

import array_api_compat

from deepmd.dpmodel.common import (
    NativeOP,
)
from deepmd.dpmodel.utils.network import (
    LayerNorm,
    NativeLayer,
)

__all__ = ["DistanceHead", "MaskLMHead", "coord_update"]


class MaskLMHead(NativeOP):
    """Element prediction for the corrupted atoms.

    Mirrors unimol/models/unimol.py:292. The output projection is a free
    parameter, not tied to the token embedding: upstream takes the weight of a
    throw-away ``nn.Linear`` and keeps it (``:301-303``).
    """

    def __init__(
        self,
        embed_dim: int,
        output_dim: int,
        activation_function: str = "gelu_erf",
        precision: str = "float64",
        seed: int | list[int] | None = None,
    ) -> None:
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.activation_function = activation_function
        self.precision = precision
        self.dense = NativeLayer(
            embed_dim,
            embed_dim,
            activation_function=activation_function,
            precision=precision,
            seed=seed,
        )
        self.layer_norm = LayerNorm(embed_dim, precision=precision, seed=seed)
        self.out_proj = NativeLayer(
            embed_dim, output_dim, bias=True, precision=precision, seed=seed
        )

    def call(self, features, masked_tokens=None):  # noqa: ANN001, ANN201
        """Project the selected positions onto the vocabulary.

        Parameters
        ----------
        features
            nf x nt x embed_dim node representation.
        masked_tokens
            nf x nt boolean mask; only these positions are projected, which is
            what upstream does to save memory.

        Returns
        -------
        Array
            n_masked x output_dim logits, or nf x nt x output_dim if no mask.
        """
        xp = array_api_compat.array_namespace(features)
        if masked_tokens is not None:
            features = features[xp.astype(masked_tokens, xp.bool), :]
        return self.out_proj(self.layer_norm(self.dense(features)))

    def serialize(self) -> dict:
        return {
            "embed_dim": self.embed_dim,
            "output_dim": self.output_dim,
            "activation_function": self.activation_function,
            "precision": self.precision,
            "dense": self.dense.serialize(),
            "layer_norm": self.layer_norm.serialize(),
            "out_proj": self.out_proj.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "MaskLMHead":
        data = data.copy()
        parts = {k: data.pop(k) for k in ("dense", "layer_norm", "out_proj")}
        obj = cls(**data)
        obj.dense = NativeLayer.deserialize(parts["dense"])
        obj.layer_norm = LayerNorm.deserialize(parts["layer_norm"])
        obj.out_proj = NativeLayer.deserialize(parts["out_proj"])
        return obj


class DistanceHead(NativeOP):
    """Pairwise distance prediction from the pair representation.

    Mirrors unimol/models/unimol.py:370. It reads the final attention logits,
    not the pair delta, and symmetrises its output.
    """

    def __init__(
        self,
        heads: int,
        activation_function: str = "gelu_erf",
        precision: str = "float64",
        seed: int | list[int] | None = None,
    ) -> None:
        self.heads = heads
        self.activation_function = activation_function
        self.precision = precision
        self.dense = NativeLayer(
            heads,
            heads,
            activation_function=activation_function,
            precision=precision,
            seed=seed,
        )
        self.layer_norm = LayerNorm(heads, precision=precision, seed=seed)
        self.out_proj = NativeLayer(heads, 1, bias=True, precision=precision, seed=seed)

    def call(self, pair_rep):  # noqa: ANN001, ANN201
        """Map nf x nt x nt x heads onto a symmetric nf x nt x nt matrix."""
        xp = array_api_compat.array_namespace(pair_rep)
        x = self.out_proj(self.layer_norm(self.dense(pair_rep)))
        x = xp.reshape(x, x.shape[:3])
        return (x + xp.matrix_transpose(x)) * 0.5

    def serialize(self) -> dict:
        return {
            "heads": self.heads,
            "activation_function": self.activation_function,
            "precision": self.precision,
            "dense": self.dense.serialize(),
            "layer_norm": self.layer_norm.serialize(),
            "out_proj": self.out_proj.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "DistanceHead":
        data = data.copy()
        parts = {k: data.pop(k) for k in ("dense", "layer_norm", "out_proj")}
        obj = cls(**data)
        obj.dense = NativeLayer.deserialize(parts["dense"])
        obj.layer_norm = LayerNorm.deserialize(parts["layer_norm"])
        obj.out_proj = NativeLayer.deserialize(parts["out_proj"])
        return obj


def coord_update(coord, delta_pair_rep, padding_mask, pair2coord_proj):  # noqa: ANN001, ANN201
    r"""Denoise coordinates through the pair channel.

    Mirrors unimol/models/unimol.py:229-245, the form introduced by upstream
    #211: the normaliser counts every non-padding token, BOS and EOS included,
    and pairs touching padding are zeroed before the sum.

    .. math::

        \hat{x}_i = x_i + \sum_j \frac{x_j - x_i}{N} c_{ij}

    Parameters
    ----------
    coord
        nf x nt x 3 input (noisy) coordinates.
    delta_pair_rep
        nf x nt x nt x heads pair delta from the encoder.
    padding_mask
        nf x nt, 1 on padded positions.
    pair2coord_proj
        Head mapping heads -> 1.

    Returns
    -------
    Array
        nf x nt x 3 updated coordinates.
    """
    xp = array_api_compat.array_namespace(coord)
    pad = xp.astype(padding_mask, coord.dtype)
    atom_num = xp.reshape(xp.sum(1 - pad, axis=1), (-1, 1, 1, 1))
    delta_pos = coord[:, None, :, :] - coord[:, :, None, :]
    attn_probs = pair2coord_proj(delta_pair_rep)
    update = delta_pos / atom_num * attn_probs
    pair_coords_mask = (1 - pad)[..., None] * (1 - pad)[:, None, :]
    update = update * pair_coords_mask[..., None]
    return coord + xp.sum(update, axis=2)
