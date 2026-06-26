# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Attention-residual layers for the SeZM descriptor.

This module defines the depth-wise attention residual aggregator used to
combine equivariant states across descriptor and block histories.
"""

from __future__ import (
    annotations,
)

from typing import (
    TYPE_CHECKING,
    Any,
)

import torch
import torch.nn as nn

from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
    RESERVED_PRECISION_DICT,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .norm import (
    ScalarRMSNorm,
)
from .so3 import (
    ChannelLinear,
)
from .utils import (
    np_safe,
    safe_numpy_to_tensor,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )


class DepthAttnRes(nn.Module):
    """
    Depth-wise attention residual aggregation for equivariant tensors.

    Attention logits are computed only from scalar ``l=0`` channels, while the
    resulting scalar weights are broadcast to the full equivariant value tensors.
    This keeps the aggregation equivariant as long as all sources share the same
    representation space.

    Query modes
    -----------
    - ``input_dependent=True``: query comes from the current scalar state.
    - ``input_dependent=False``: use a learned pseudo-query shared across inputs.

    Both query paths are zero-initialized so the initial aggregation is a uniform
    average over all provided sources.

    Parameters
    ----------
    channels
        Scalar feature dimension used by query and key.
    input_dependent
        Whether to project the current scalar state into a query vector.
    eps
        Small epsilon for key RMS normalization.
    bias
        Whether to use bias in the input-dependent query projection. Only
        effective when ``input_dependent=True``.
    dtype
        Parameter and compute dtype. Caller should pass compute_dtype (fp32+).
    trainable
        Whether parameters are trainable.
    seed
        Random seed reserved for consistency with other modules.
    """

    if TYPE_CHECKING:
        query_proj: ChannelLinear
        adamw_pseudo_query: torch.Tensor

    def __init__(
        self,
        *,
        channels: int,
        input_dependent: bool = True,
        eps: float = 1e-7,
        bias: bool = True,
        dtype: torch.dtype,
        trainable: bool,
        seed: int | list[int] | None = None,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.input_dependent = bool(input_dependent)
        self.eps = float(eps)
        self.query_bias = bool(bias)
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[dtype]

        self.key_norm = ScalarRMSNorm(
            channels=self.channels,
            n_focus=1,
            eps=self.eps,
            dtype=self.dtype,
            trainable=trainable,
        )
        if self.input_dependent:
            self.query_proj = ChannelLinear(
                in_channels=self.channels,
                out_channels=self.channels,
                dtype=self.dtype,
                bias=self.query_bias,
                trainable=trainable,
                seed=seed,
                init_std=0.0,
            )
        else:
            self.adamw_pseudo_query = nn.Parameter(
                torch.zeros(self.channels, dtype=self.dtype, device=self.device),
                requires_grad=trainable,
            )

        for p in self.parameters():
            p.requires_grad = trainable

    def forward(
        self,
        *,
        sources: list[torch.Tensor],
        scalar_extractor: Callable[[torch.Tensor], torch.Tensor],
        current_x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Aggregate same-shape sources with depth attention.

        Parameters
        ----------
        sources
            Source tensors with identical shape ``(B, ...)``.
        scalar_extractor
            Function that extracts scalar features from each source with shape
            ``(B, C)`` where ``C=channels``.
        current_x
            Current tensor state. Required when ``input_dependent=True`` and
            converted to scalar query features via ``scalar_extractor``.

        Returns
        -------
        torch.Tensor
            Aggregated tensor with the same shape as each source.
        """
        source0 = sources[0]
        if len(sources) == 1:
            return source0
        batch_size = int(source0.shape[0])
        value_dtype = source0.dtype

        # === Step 1. Build the query vector ===
        if self.input_dependent:
            current_x_scalar = scalar_extractor(current_x)
            query = self.query_proj(current_x_scalar.to(dtype=self.dtype))
        else:
            query = self.adamw_pseudo_query.unsqueeze(0).expand(batch_size, -1)

        # === Step 2. Extract and normalize scalar keys ===
        source_count = len(sources)
        raw_keys = torch.stack(
            [scalar_extractor(source).to(dtype=self.dtype) for source in sources],
            dim=1,
        )  # (B, S, C)
        keys = self.key_norm(raw_keys)
        logits = torch.einsum("bc,bsc->bs", query, keys)
        alpha = torch.softmax(logits, dim=1)  # (B, S)

        # === Step 3. Broadcast scalar weights to equivariant values ===
        value_stack = torch.stack(
            [source.to(dtype=self.dtype) for source in sources],
            dim=1,
        )
        alpha = alpha.reshape(
            batch_size,
            source_count,
            *([1] * (value_stack.ndim - 2)),
        )
        aggregated = (alpha * value_stack).sum(dim=1)
        return aggregated.to(dtype=value_dtype)

    def serialize(self) -> dict[str, Any]:
        trainable = all(p.requires_grad for p in self.parameters())
        state = self.state_dict()
        return {
            "@class": "DepthAttnRes",
            "@version": 1,
            "config": {
                "channels": self.channels,
                "input_dependent": self.input_dependent,
                "eps": self.eps,
                "bias": self.query_bias,
                "precision": RESERVED_PRECISION_DICT[self.dtype],
                "trainable": trainable,
                "seed": None,
            },
            "@variables": {key: np_safe(value) for key, value in state.items()},
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> DepthAttnRes:
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "DepthAttnRes":
            raise ValueError(f"Invalid class for DepthAttnRes: {data_cls}")
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
