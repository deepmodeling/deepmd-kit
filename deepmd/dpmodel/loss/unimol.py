# SPDX-License-Identifier: LGPL-3.0-or-later
"""The Uni-Mol v1 molecular-pretraining objective.

Ported from Uni-Mol (https://github.com/deepmodeling/Uni-Mol) at commit 90f52c4,
MIT licensed:

    Copyright (c) DP Technology
    This source code is licensed under the MIT license found in the LICENSE
    file in the root directory of that source tree.

Five terms, with upstream's default weights from its README pretraining recipe:
element prediction (1), coordinate denoising (5), distance prediction (10), and
the two norm regularisers (0.01 each). The regularisers are produced by the
backbone, so the loss only weights them.
"""

import array_api_compat

from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.loss.loss import (
    Loss,
)
from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

# Upstream normalises the distance target with these two constants
# (unimol/losses/unimol.py:17-18). They are hard-coded there, not fitted.
DIST_MEAN = 6.312581655060595
DIST_STD = 3.3899264663911888


def _smooth_l1(pred: Array, label: Array, beta: float = 1.0) -> Array:
    r"""Mean smooth L1, matching ``F.smooth_l1_loss(reduction="mean")``.

    .. math::

       \ell_\beta(e)=\begin{cases}
       e^2/(2\beta),&|e|<\beta,\\
       |e|-\beta/2,&|e|\ge\beta.
       \end{cases}
    """
    xp = array_api_compat.array_namespace(pred)
    diff = xp.abs(pred - label)
    elementwise = xp.where(diff < beta, 0.5 * diff**2 / beta, diff - 0.5 * beta)
    return xp.mean(elementwise)


def _masked_nll(logits: Array, target: Array, pad_idx: int) -> Array:
    """Negative log likelihood over the selected positions.

    Upstream evaluates ``log_softmax`` in fp32 (``losses/unimol.py:36``) because
    it pretrains an fp16 model; the cast is kept so the value matches.
    """
    xp = array_api_compat.array_namespace(logits)
    logits = xp.astype(logits, xp.float32)
    x_max = xp.max(logits, axis=-1, keepdims=True)
    shifted = logits - x_max
    log_probs = shifted - xp.log(xp.sum(xp.exp(shifted), axis=-1, keepdims=True))
    target = xp.reshape(target, (-1,))
    keep = target != pad_idx
    picked = xp.take_along_axis(log_probs, xp.reshape(target, (-1, 1)), axis=1)
    picked = xp.reshape(picked, (-1,))
    picked = xp.where(keep, picked, xp.zeros_like(picked))
    return -xp.sum(picked) / xp.astype(
        xp.sum(xp.astype(keep, logits.dtype)), logits.dtype
    )


@Loss.register("unimol")
class UniMolLoss(Loss):
    r"""Uni-Mol v1 self-supervised pretraining loss.

    .. math::

       L = w_t L_\text{token} + w_c L_\text{coord} + w_d L_\text{dist}
           + w_x L_{\|x\|} + w_p L_{\|\Delta p\|}

    Every term is a flat mean over the positions it covers, so molecules with
    more corrupted atoms weigh more, exactly as upstream.

    Parameters
    ----------
    masked_token_loss : float
        Weight of the element-prediction term.
    masked_coord_loss : float
        Weight of the coordinate-denoising term.
    masked_dist_loss : float
        Weight of the distance-prediction term.
    x_norm_loss : float
        Weight of the node-norm regulariser.
    delta_pair_repr_norm_loss : float
        Weight of the pair-delta-norm regulariser.
    beta : float
        The transition point of the smooth L1 used by the coordinate and
        distance terms.
    pad_idx : int
        Token id that marks padding, excluded from every term.
    """

    def __init__(
        self,
        masked_token_loss: float = 1.0,
        masked_coord_loss: float = 5.0,
        masked_dist_loss: float = 10.0,
        x_norm_loss: float = 0.01,
        delta_pair_repr_norm_loss: float = 0.01,
        beta: float = 1.0,
        pad_idx: int = 0,
        **kwargs: float,
    ) -> None:
        self.masked_token_loss = masked_token_loss
        self.masked_coord_loss = masked_coord_loss
        self.masked_dist_loss = masked_dist_loss
        self.x_norm_loss = x_norm_loss
        self.delta_pair_repr_norm_loss = delta_pair_repr_norm_loss
        self.beta = beta
        self.pad_idx = pad_idx

    def call(
        self,
        learning_rate: float,
        natoms: int,
        model_dict: dict[str, Array],
        label_dict: dict[str, Array],
        mae: bool = False,
    ) -> tuple[Array, dict[str, Array]]:
        """Evaluate the five terms and their weighted sum."""
        del learning_rate, natoms, mae
        token_target = label_dict["unimol_token_target"]
        xp = array_api_compat.array_namespace(token_target)
        masked = token_target != self.pad_idx
        more_loss = {}
        loss = None

        def add(term: Array, weight: float, name: str) -> None:
            nonlocal loss
            more_loss[name] = term
            loss = weight * term if loss is None else loss + weight * term

        if self.masked_token_loss > 0:
            add(
                _masked_nll(
                    model_dict["unimol_logits"], token_target[masked], self.pad_idx
                ),
                self.masked_token_loss,
                "token_loss",
            )
        if self.masked_coord_loss > 0:
            coord_pred = model_dict["unimol_coord"][masked]
            coord_label = label_dict["unimol_coord_target"][masked]
            add(
                _smooth_l1(
                    xp.reshape(coord_pred, (-1, 3)),
                    xp.reshape(coord_label, (-1, 3)),
                    self.beta,
                ),
                self.masked_coord_loss,
                "coord_loss",
            )
        if self.masked_dist_loss > 0:
            # Rows are the corrupted atoms; columns are every non-padding token,
            # BOS, EOS and the diagonal included (losses/unimol.py:159-180).
            token_mask = label_dict["unimol_token_mask"]
            pair_mask = masked[..., None] & xp.astype(token_mask, xp.bool)[:, None, :]
            dist_label = (
                label_dict["unimol_dist_target"][pair_mask] - DIST_MEAN
            ) / DIST_STD
            add(
                _smooth_l1(model_dict["unimol_dist"][pair_mask], dist_label, self.beta),
                self.masked_dist_loss,
                "dist_loss",
            )
        if self.x_norm_loss > 0:
            add(model_dict["unimol_x_norm"], self.x_norm_loss, "x_norm_loss")
        if self.delta_pair_repr_norm_loss > 0:
            add(
                model_dict["unimol_delta_pair_norm"],
                self.delta_pair_repr_norm_loss,
                "delta_pair_norm_loss",
            )
        return loss, more_loss

    @property
    def label_requirement(self) -> list[DataRequirementItem]:
        """Labels produced by the Uni-Mol data transform, not by a simulation."""
        return [
            DataRequirementItem("unimol_token_target", ndof=1, atomic=True, must=True),
            DataRequirementItem("unimol_token_mask", ndof=1, atomic=True, must=True),
            DataRequirementItem("unimol_coord_target", ndof=3, atomic=True, must=True),
            DataRequirementItem("unimol_dist_target", ndof=1, atomic=True, must=True),
        ]

    def serialize(self) -> dict:
        """Serialize the loss module."""
        return {
            "@class": "UniMolLoss",
            "@version": 1,
            "masked_token_loss": self.masked_token_loss,
            "masked_coord_loss": self.masked_coord_loss,
            "masked_dist_loss": self.masked_dist_loss,
            "x_norm_loss": self.x_norm_loss,
            "delta_pair_repr_norm_loss": self.delta_pair_repr_norm_loss,
            "beta": self.beta,
            "pad_idx": self.pad_idx,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "UniMolLoss":
        """Deserialize the loss module."""
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
        return cls(**data)
