# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import torch

from deepmd.pt.model.network.network import (
    MaskLMHead,
)
from deepmd.pt.model.task import (
    Fitting,
)


class TypePredictNet(Fitting):
    def __init__(
        self,
        feature_dim: int,
        ntypes: int,
        activation_function: str = "gelu",
        **kwargs: Any,
    ) -> None:
        """Construct a type predict net.

        Args:
        - feature_dim: Input dm.
        - ntypes: Number of types to predict.
        - activation_function: Activate function.
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.ntypes = ntypes
        self.lm_head = MaskLMHead(
            embed_dim=self.feature_dim,
            output_dim=ntypes,
            activation_fn=activation_function,
            weight=None,
        )

    def forward(
        self, features: torch.Tensor, masked_tokens: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Calculate the predicted logits.
        Args:
        - features: Input features with shape [nframes, nloc, feature_dim].
        - masked_tokens: Input masked tokens with shape [nframes, nloc].

        Returns
        -------
        - logits: Predicted probs with shape [nframes, nloc, ntypes].
        """
        # [nframes, nloc, ntypes]
        logits = self.lm_head(features, masked_tokens=masked_tokens)
        return logits
