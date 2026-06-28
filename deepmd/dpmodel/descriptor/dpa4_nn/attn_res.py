# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Attention-residual layers for the DPA4/SeZM descriptor.

This module defines the depth-wise attention residual aggregator used to
combine equivariant states across descriptor and block histories.

This module is the dpmodel (array-API) port of
``deepmd.pt.model.descriptor.sezm_nn.attn_res``.
"""

from __future__ import (
    annotations,
)

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
    xp_asarray_nodetach,
)
from deepmd.dpmodel.common import (
    get_xp_precision,
    to_numpy_array,
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

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )

    from deepmd.dpmodel.array_api import (
        Array,
    )


class DepthAttnRes(NativeOP):
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
    precision
        Parameter and compute precision. Caller should pass compute precision (fp32+).
    trainable
        Whether parameters are trainable.
    seed
        Random seed reserved for consistency with other modules.
    """

    if TYPE_CHECKING:
        query_proj: ChannelLinear
        adamw_pseudo_query: Array

    def __init__(
        self,
        *,
        channels: int,
        input_dependent: bool = True,
        eps: float = 1e-7,
        bias: bool = True,
        precision: str = DEFAULT_PRECISION,
        trainable: bool,
        seed: int | list[int] | None = None,
    ) -> None:
        self.channels = int(channels)
        self.input_dependent = bool(input_dependent)
        self.eps = float(eps)
        self.query_bias = bool(bias)
        self.precision = precision
        prec = PRECISION_DICT[self.precision.lower()]

        self.key_norm = ScalarRMSNorm(
            channels=self.channels,
            n_focus=1,
            eps=self.eps,
            precision=self.precision,
            trainable=trainable,
        )
        if self.input_dependent:
            self.query_proj = ChannelLinear(
                in_channels=self.channels,
                out_channels=self.channels,
                precision=self.precision,
                bias=self.query_bias,
                trainable=trainable,
                seed=seed,
                init_std=0.0,
            )
        else:
            self.adamw_pseudo_query = np.zeros(self.channels, dtype=prec)

        self.trainable = bool(trainable)

    def call(
        self,
        *,
        sources: list[Array],
        scalar_extractor: Callable[[Array], Array],
        current_x: Array | None = None,
    ) -> Array:
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
        Array
            Aggregated tensor with the same shape as each source.
        """
        source0 = sources[0]
        if len(sources) == 1:
            return source0
        xp = array_api_compat.array_namespace(source0)
        device = array_api_compat.device(source0)
        batch_size = int(source0.shape[0])
        value_dtype = source0.dtype

        # === Step 1. Build the query vector ===
        if self.input_dependent:
            current_x_scalar = scalar_extractor(current_x)
            query = self.query_proj(
                xp.astype(current_x_scalar, get_xp_precision(xp, self.precision))
            )
        else:
            query = xp.broadcast_to(
                xp_asarray_nodetach(xp, self.adamw_pseudo_query[...], device=device)[
                    None, :
                ],
                (batch_size, self.channels),
            )

        # === Step 2. Extract and normalize scalar keys ===
        source_count = len(sources)
        raw_keys = xp.stack(
            [
                xp.astype(
                    scalar_extractor(source), get_xp_precision(xp, self.precision)
                )
                for source in sources
            ],
            axis=1,
        )  # (B, S, C)
        keys = self.key_norm(raw_keys)
        logits = xp.sum(query[:, None, :] * keys, axis=-1)
        alpha = xp.exp(logits - xp.max(logits, axis=1, keepdims=True))
        alpha = alpha / xp.sum(alpha, axis=1, keepdims=True)  # (B, S)

        # === Step 3. Broadcast scalar weights to equivariant values ===
        value_stack = xp.stack(
            [
                xp.astype(source, get_xp_precision(xp, self.precision))
                for source in sources
            ],
            axis=1,
        )
        alpha = xp.reshape(
            alpha,
            (
                batch_size,
                source_count,
                *([1] * (value_stack.ndim - 2)),
            ),
        )
        aggregated = xp.sum(alpha * value_stack, axis=1)
        return xp.astype(aggregated, value_dtype)

    def serialize(self) -> dict[str, Any]:
        variables = {"key_norm.adam_scale": to_numpy_array(self.key_norm.adam_scale)}
        if self.input_dependent:
            variables["query_proj.weight"] = to_numpy_array(self.query_proj.weight)
            if self.query_bias:
                variables["query_proj.bias"] = to_numpy_array(self.query_proj.bias)
        else:
            variables["adamw_pseudo_query"] = to_numpy_array(self.adamw_pseudo_query)
        return {
            "@class": "DepthAttnRes",
            "@version": 1,
            "config": {
                "channels": self.channels,
                "input_dependent": self.input_dependent,
                "eps": self.eps,
                "bias": self.query_bias,
                "precision": np.dtype(PRECISION_DICT[self.precision]).name,
                "trainable": self.trainable,
                "seed": None,
            },
            "@variables": variables,
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
        obj = cls(**config)
        prec = PRECISION_DICT[obj.precision.lower()]
        obj.key_norm.adam_scale = np.asarray(
            variables["key_norm.adam_scale"], dtype=prec
        )
        if obj.input_dependent:
            obj.query_proj.weight = np.asarray(
                variables["query_proj.weight"], dtype=prec
            )
            if obj.query_bias:
                obj.query_proj.bias = np.asarray(
                    variables["query_proj.bias"], dtype=prec
                )
        else:
            obj.adamw_pseudo_query = np.asarray(
                variables["adamw_pseudo_query"], dtype=prec
            )
        return obj
