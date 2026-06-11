# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Normalization layers for the DPA4/SeZM descriptor.

This module is the dpmodel port of ``deepmd.pt.model.descriptor.sezm_nn.norm``.
Currently only ``RMSNorm`` is ported (it is required by ``radial.RadialMLP``);
``EquivariantRMSNorm`` is ported by a later task.
"""

from __future__ import (
    annotations,
)

from typing import (
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
from deepmd.utils.version import (
    check_version_compatibility,
)


class RMSNorm(NativeOP):
    """
    Generic RMSNorm on tensors with shape `(..., C)`.

    This is the plain channel-wise RMS normalization used for non-equivariant
    branches whose last axis stores feature channels. A learnable affine scale
    is applied on the channel axis only, while all leading axes are treated as
    batch dimensions.

    Parameters
    ----------
    channels : int
        Feature dimension of the last axis.
    eps : float
        Small epsilon for numerical stability.
    precision : str
        Parameter and computation precision. Caller should pass a compute
        precision (fp32+) for numerical stability.
    trainable : bool
        Whether parameters are trainable.
    """

    def __init__(
        self,
        *,
        channels: int,
        eps: float = 1e-7,
        precision: str = DEFAULT_PRECISION,
        trainable: bool = True,
    ) -> None:
        self.channels = int(channels)
        self.eps = float(eps)
        self.precision = precision
        self.trainable = bool(trainable)
        prec = PRECISION_DICT[self.precision.lower()]
        # adam_ prefix routes this to Adam (no weight decay) in HybridMuon.
        self.adam_scale = np.ones((self.channels,), dtype=prec)

    def call(self, x: Any) -> Any:
        """
        Apply RMS normalization.

        Parameters
        ----------
        x : Array
            Input array with shape `(..., C)`.

        Returns
        -------
        Array
            Normalized array with shape `(..., C)`, same dtype as input.
        """
        xp = array_api_compat.array_namespace(x)
        scale = self.adam_scale[...]
        in_dtype = x.dtype
        if in_dtype != scale.dtype:
            x = xp.astype(x, scale.dtype)
        inv_rms = 1.0 / xp.sqrt(xp.mean(x * x, axis=-1, keepdims=True) + self.eps)
        out = x * inv_rms * scale
        if out.dtype != in_dtype:
            out = xp.astype(out, in_dtype)
        return out

    def serialize(self) -> dict[str, Any]:
        """Serialize the RMSNorm to a dict."""
        return {
            "@class": "RMSNorm",
            "@version": 1,
            "config": {
                "channels": self.channels,
                "eps": self.eps,
                "precision": np.dtype(PRECISION_DICT[self.precision]).name,
                "trainable": self.trainable,
            },
            "@variables": {"adam_scale": to_numpy_array(self.adam_scale)},
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> RMSNorm:
        """Deserialize an RMSNorm from a dict."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "RMSNorm":
            raise ValueError(f"Invalid class for RMSNorm: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(
            channels=int(config["channels"]),
            eps=float(config["eps"]),
            precision=str(config["precision"]),
            trainable=bool(config["trainable"]),
        )
        prec = PRECISION_DICT[obj.precision.lower()]
        adam_scale = np.asarray(variables["adam_scale"], dtype=prec).reshape(-1)
        if adam_scale.shape != obj.adam_scale.shape:
            raise ValueError(
                f"adam_scale shape {adam_scale.shape} does not match "
                f"channels {obj.channels}"
            )
        obj.adam_scale = adam_scale
        return obj
