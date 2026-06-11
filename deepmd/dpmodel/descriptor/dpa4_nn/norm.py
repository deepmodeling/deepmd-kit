# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Normalization layers for the DPA4/SeZM descriptor.

This module is the dpmodel port of ``deepmd.pt.model.descriptor.sezm_nn.norm``.
All four pt norm classes are ported: ``RMSNorm`` (used by ``radial.RadialMLP``),
``EquivariantRMSNorm`` (used by ``block``), ``ReducedEquivariantRMSNorm`` and
``ScalarRMSNorm`` (used by ``so2``).

Serialization contract: the ``@variables`` keys of each class match the
``state_dict`` key names of its pt counterpart, so pt ``serialize()`` output
deserializes directly into the dpmodel classes (and vice versa).
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

from .indexing import (
    map_degree_idx,
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
        scale = xp.asarray(self.adam_scale[...], device=array_api_compat.device(x))
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


class EquivariantRMSNorm(NativeOP):
    """
    Degree-balanced equivariant RMS normalization on packed `(l, m)` layout.

    The scalar slice `l=0` is mean-centered across channels before the shared
    RMS is evaluated. All coefficients, including the centered scalar slice,
    contribute to the same per-sample and per-focus RMS. Degree balancing
    assigns each coefficient from degree `l` the weight
    `1 / ((2 * l + 1) * (lmax + 1))`, so each degree contributes equally
    regardless of its multiplicity. A learnable per-focus, per-degree scale is
    then expanded to all `m` coefficients, and a learnable bias is added only
    to the scalar slice.

    Parameters
    ----------
    lmax : int
        Maximum spherical harmonic degree.
    channels : int
        Channels per `(l, m)` coefficient in each focus stream.
    n_focus : int
        Number of focus streams. Affine parameters are independent per focus.
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
        lmax: int,
        channels: int,
        n_focus: int = 1,
        *,
        eps: float = 1e-5,
        precision: str = DEFAULT_PRECISION,
        trainable: bool = True,
    ) -> None:
        self.lmax = int(lmax)
        self.channels = int(channels)
        self.n_focus = int(n_focus)
        self.eps = float(eps)
        self.precision = precision
        self.trainable = bool(trainable)
        prec = PRECISION_DICT[self.precision.lower()]

        # === Step 1. Learnable Parameters ===
        # Store affine scales in degree-major layout (L, F, C). This matches the
        # packed output layout after degree expansion.
        # adam_ prefix routes this to Adam (no weight decay) in HybridMuon.
        self.adam_scale = np.ones(
            (self.lmax + 1, self.n_focus, self.channels), dtype=prec
        )
        # Bias only for l=0, independent per focus.
        self.bias = np.zeros((self.n_focus, self.channels), dtype=prec)

        # === Step 2. Index and Weight Buffers ===
        self.expand_index = map_degree_idx(self.lmax)

        # Pre-fuse degree balancing and channel averaging into a single weight:
        #   w_d = 1 / ((2l+1) * (lmax+1) * C)
        # so that the shared RMS statistic is a single weighted sum without
        # allocating an intermediate (N, D, F, C) buffer beyond x^2 itself.
        weights_list = []
        scale = 1.0 / ((self.lmax + 1) * self.channels)
        for l in range(self.lmax + 1):
            w = scale / (2 * l + 1)
            weights_list.extend([w] * (2 * l + 1))
        self.balance_weight = np.asarray(weights_list, dtype=prec)

    def call(self, x: Any) -> Any:
        """
        Apply degree-balanced equivariant RMS normalization.

        Parameters
        ----------
        x : Array
            Features with shape `(N, D, F, C)` where `D = (lmax + 1)^2`.

        Returns
        -------
        Array
            Normalized features with shape `(N, D, F, C)`, same dtype as input.
        """
        xp = array_api_compat.array_namespace(x)
        device = array_api_compat.device(x)
        scale = xp.asarray(self.adam_scale[...], device=device)
        bias = xp.asarray(self.bias[...], device=device)
        balance_weight = xp.asarray(
            self.balance_weight, device=array_api_compat.device(x)
        )
        in_dtype = x.dtype
        if in_dtype != scale.dtype:
            x = xp.astype(x, scale.dtype)
        x0 = x[:, :1, :, :]  # (N, 1, F, C)
        xt = x[:, 1:, :, :]  # (N, D-1, F, C)

        # === Step 1. Center the scalar slice ===
        x0 = x0 - xp.mean(x0, axis=-1, keepdims=True)

        # === Step 2. Compute a shared degree-balanced RMS ===
        mean_variance = xp.sum(x0 * x0, axis=(1, 3)) * balance_weight[0]
        if self.lmax > 0:
            mean_variance = mean_variance + xp.sum(
                (xt * xt) * balance_weight[1:][None, :, None, None], axis=(1, 3)
            )
        inv_rms = 1.0 / xp.sqrt(mean_variance + self.eps)
        inv_rms = inv_rms[:, None, :, None]  # (N, 1, F, 1)

        x0 = x0 * inv_rms
        if self.lmax > 0:
            xt = xt * inv_rms

        # === Step 3. Apply per-degree affine parameters ===
        expand_index = xp.asarray(self.expand_index, device=array_api_compat.device(x))
        expanded_scale = xp.take(scale, expand_index, axis=0)
        expanded_scale = expanded_scale[None, ...]  # (1, D, F, C)
        x0 = x0 * expanded_scale[:, :1, :, :]
        if self.lmax > 0:
            xt = xt * expanded_scale[:, 1:, :, :]

        # === Step 4. Add scalar bias and restore layout ===
        bias0 = xp.reshape(bias, (1, 1, self.n_focus, -1))  # (1, 1, F, C)
        x0 = x0 + bias0

        out = x0 if self.lmax == 0 else xp.concat([x0, xt], axis=1)
        if out.dtype != in_dtype:
            out = xp.astype(out, in_dtype)
        return out

    def serialize(self) -> dict[str, Any]:
        """Serialize the EquivariantRMSNorm to a dict."""
        return {
            "@class": "EquivariantRMSNorm",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "channels": self.channels,
                "n_focus": self.n_focus,
                "eps": self.eps,
                "precision": np.dtype(PRECISION_DICT[self.precision]).name,
                "trainable": self.trainable,
            },
            "@variables": {
                "adam_scale": to_numpy_array(self.adam_scale),
                "bias": to_numpy_array(self.bias),
                "expand_index": to_numpy_array(self.expand_index),
                "balance_weight": to_numpy_array(self.balance_weight),
            },
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> EquivariantRMSNorm:
        """Deserialize an EquivariantRMSNorm from a dict."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "EquivariantRMSNorm":
            raise ValueError(f"Invalid class for EquivariantRMSNorm: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(
            lmax=int(config["lmax"]),
            channels=int(config["channels"]),
            n_focus=int(config["n_focus"]),
            eps=float(config["eps"]),
            precision=str(config["precision"]),
            trainable=bool(config["trainable"]),
        )
        prec = PRECISION_DICT[obj.precision.lower()]
        expand_index = np.asarray(variables["expand_index"], dtype=np.int64)
        if not np.array_equal(expand_index, obj.expand_index):
            raise ValueError("expand_index does not match the lmax-derived table")
        for name in ("adam_scale", "bias", "balance_weight"):
            value = np.asarray(variables[name], dtype=prec)
            if value.shape != getattr(obj, name).shape:
                raise ValueError(
                    f"{name} shape {value.shape} does not match "
                    f"the expected shape {getattr(obj, name).shape}"
                )
            setattr(obj, name, value)
        return obj


class ReducedEquivariantRMSNorm(NativeOP):
    """
    Degree-balanced equivariant RMS normalization on reduced m-major layout.

    The scalar slice `l=0` is mean-centered across channels before the shared
    RMS is evaluated. All retained coefficients, including the centered scalar
    slice, contribute to the same per-edge and per-focus RMS. Degree balancing
    assigns each retained coefficient from degree `l` the weight
    `1 / (n_coeff_l * (lmax + 1))`, where
    `n_coeff_l = 2 * min(l, mmax) + 1` is the number of retained coefficients
    for that degree in the reduced layout. A learnable per-focus, per-degree
    scale is expanded with `degree_index_m`, and a learnable bias is added only
    to the scalar slice.

    Parameters
    ----------
    lmax : int
        Maximum spherical harmonic degree.
    mmax : int
        Maximum order kept in the truncated layout.
    channels : int
        Number of channels per retained coefficient.
    degree_index_m : np.ndarray
        Degree index per coefficient in m-major truncated layout, with shape
        `(D_m_trunc,)`.
    n_focus : int
        Number of focus streams.
    eps : float
        Epsilon for numerical stability.
    precision : str
        Parameter and computation precision. Caller should pass a compute
        precision (fp32+) for numerical stability.
    trainable : bool
        Whether parameters are trainable.
    """

    def __init__(
        self,
        *,
        lmax: int,
        mmax: int,
        channels: int,
        degree_index_m: np.ndarray,
        n_focus: int = 1,
        eps: float = 1e-5,
        precision: str = DEFAULT_PRECISION,
        trainable: bool = True,
    ) -> None:
        self.lmax = int(lmax)
        self.mmax = int(mmax)
        if self.mmax < 0:
            raise ValueError("`mmax` must be non-negative")
        if self.mmax > self.lmax:
            raise ValueError("`mmax` must be <= `lmax`")
        self.channels = int(channels)
        self.n_focus = int(n_focus)
        self.eps = float(eps)
        self.precision = precision
        self.trainable = bool(trainable)
        prec = PRECISION_DICT[self.precision.lower()]

        self.degree_index_m = np.asarray(degree_index_m, dtype=np.int64)

        # Pre-fuse degree balancing and channel averaging into a single weight:
        #   w_d = 1 / (n_coeff_l * (lmax+1) * C)
        # where n_coeff_l is the number of retained coefficients for degree l in
        # the reduced layout.
        weights = np.zeros(self.degree_index_m.size, dtype=prec)
        scale = 1.0 / ((self.lmax + 1) * self.channels)
        for l in range(self.lmax + 1):
            n_coeff_l = 2 * min(l, self.mmax) + 1
            w_l = scale / float(n_coeff_l)
            weights[self.degree_index_m == l] = w_l
        if np.any(weights == 0):
            raise ValueError(
                "ReducedEquivariantRMSNorm: balance_weight has zeros; "
                "degree_index_m may be invalid."
            )
        self.balance_weight = weights

        # adam_ prefix routes this to Adam (no weight decay) in HybridMuon.
        self.adam_scale = np.ones(
            (self.n_focus, self.lmax + 1, self.channels), dtype=prec
        )
        self.bias0 = np.zeros((self.n_focus, self.channels), dtype=prec)

    def call(self, x: Any) -> Any:
        """
        Apply degree-balanced reduced-layout RMS normalization.

        Parameters
        ----------
        x : Array
            Input array with shape (E, F, D_m_trunc, C).

        Returns
        -------
        Array
            Normalized array with shape `(E, F, D_m_trunc, C)`, same dtype as
            input.
        """
        xp = array_api_compat.array_namespace(x)
        device = array_api_compat.device(x)
        scale = xp.asarray(self.adam_scale[...], device=device)
        bias0_w = xp.asarray(self.bias0[...], device=device)
        balance_weight = xp.asarray(
            self.balance_weight, device=array_api_compat.device(x)
        )
        in_dtype = x.dtype
        if in_dtype != scale.dtype:
            x = xp.astype(x, scale.dtype)
        has_xt = self.degree_index_m.size > 1
        x0 = x[:, :, :1, :]  # (E, F, 1, C)
        xt = x[:, :, 1:, :]  # (E, F, D_m_trunc-1, C)

        # === Step 1. Center the scalar slice ===
        x0 = x0 - xp.mean(x0, axis=-1, keepdims=True)

        # === Step 2. Compute a shared degree-balanced RMS ===
        mean_variance = xp.sum(x0 * x0, axis=(2, 3)) * balance_weight[0]
        if has_xt:
            mean_variance = mean_variance + xp.sum(
                (xt * xt) * balance_weight[1:][None, None, :, None], axis=(2, 3)
            )
        inv_rms = 1.0 / xp.sqrt(mean_variance + self.eps)
        inv_rms = inv_rms[:, :, None, None]  # (E, F, 1, 1)

        x0 = x0 * inv_rms
        if has_xt:
            xt = xt * inv_rms

        # === Step 3. Apply per-degree affine parameters ===
        degree_index_m = xp.asarray(
            self.degree_index_m, device=array_api_compat.device(x)
        )
        expanded_scale = xp.take(scale, degree_index_m, axis=1)
        expanded_scale = expanded_scale[None, ...]  # (1, F, D_m_trunc, C)
        x0 = x0 * expanded_scale[:, :, :1, :]
        if has_xt:
            xt = xt * expanded_scale[:, :, 1:, :]

        # === Step 4. Add scalar bias and restore layout ===
        bias0 = xp.reshape(bias0_w, (1, self.n_focus, 1, -1))  # (1, F, 1, C)
        x0 = x0 + bias0

        out = xp.concat([x0, xt], axis=2) if has_xt else x0
        if out.dtype != in_dtype:
            out = xp.astype(out, in_dtype)
        return out

    def serialize(self) -> dict[str, Any]:
        """Serialize the ReducedEquivariantRMSNorm to a dict."""
        return {
            "@class": "ReducedEquivariantRMSNorm",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "mmax": self.mmax,
                "channels": self.channels,
                "degree_index_m": to_numpy_array(self.degree_index_m),
                "n_focus": self.n_focus,
                "eps": self.eps,
                "precision": np.dtype(PRECISION_DICT[self.precision]).name,
                "trainable": self.trainable,
            },
            "@variables": {
                "degree_index_m": to_numpy_array(self.degree_index_m),
                "balance_weight": to_numpy_array(self.balance_weight),
                "adam_scale": to_numpy_array(self.adam_scale),
                "bias0": to_numpy_array(self.bias0),
            },
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> ReducedEquivariantRMSNorm:
        """Deserialize a ReducedEquivariantRMSNorm from a dict."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "ReducedEquivariantRMSNorm":
            raise ValueError(f"Invalid class for ReducedEquivariantRMSNorm: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(
            lmax=int(config["lmax"]),
            mmax=int(config["mmax"]),
            channels=int(config["channels"]),
            degree_index_m=np.asarray(config["degree_index_m"], dtype=np.int64),
            n_focus=int(config["n_focus"]),
            eps=float(config["eps"]),
            precision=str(config["precision"]),
            trainable=bool(config["trainable"]),
        )
        prec = PRECISION_DICT[obj.precision.lower()]
        degree_index_m = np.asarray(variables["degree_index_m"], dtype=np.int64)
        if not np.array_equal(degree_index_m, obj.degree_index_m):
            raise ValueError("degree_index_m variable does not match the config")
        for name in ("balance_weight", "adam_scale", "bias0"):
            value = np.asarray(variables[name], dtype=prec)
            if value.shape != getattr(obj, name).shape:
                raise ValueError(
                    f"{name} shape {value.shape} does not match "
                    f"the expected shape {getattr(obj, name).shape}"
                )
            setattr(obj, name, value)
        return obj


class ScalarRMSNorm(NativeOP):
    """
    Lightweight per-focus RMSNorm for scalar branches.

    This is the unified scalar norm used by SeZM:
    - `n_focus=1` naturally degenerates to the single-stream behavior.
    - `n_focus>1` uses independent learnable scales per focus stream.
    Bias is intentionally omitted to keep the gate paths minimal.

    Parameters
    ----------
    channels : int
        Feature dimension of the last axis.
    n_focus : int
        Number of focus streams.
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
        n_focus: int = 1,
        eps: float = 1e-7,
        precision: str = DEFAULT_PRECISION,
        trainable: bool = True,
    ) -> None:
        self.channels = int(channels)
        self.n_focus = int(n_focus)
        self.eps = float(eps)
        self.precision = precision
        self.trainable = bool(trainable)
        prec = PRECISION_DICT[self.precision.lower()]
        # adam_ prefix routes this to Adam (no weight decay) in HybridMuon.
        self.adam_scale = np.ones((self.n_focus, self.channels), dtype=prec)

    def call(self, x: Any) -> Any:
        """
        Apply per-focus RMS normalization.

        Parameters
        ----------
        x : Array
            Input array with shape (B, F, C) or (B, C) when `n_focus=1`.

        Returns
        -------
        Array
            Normalized array with the same shape as input and same dtype.
        """
        xp = array_api_compat.array_namespace(x)
        scale = xp.asarray(self.adam_scale[...], device=array_api_compat.device(x))
        in_dtype = x.dtype
        if in_dtype != scale.dtype:
            x = xp.astype(x, scale.dtype)

        inv_rms = 1.0 / xp.sqrt(xp.mean(x * x, axis=-1, keepdims=True) + self.eps)
        x = x * inv_rms
        if x.ndim == 2:
            x = x * scale[0, :]
        else:
            x = x * scale[None, ...]
        if x.dtype != in_dtype:
            x = xp.astype(x, in_dtype)
        return x

    def serialize(self) -> dict[str, Any]:
        """Serialize the ScalarRMSNorm to a dict."""
        return {
            "@class": "ScalarRMSNorm",
            "@version": 1,
            "config": {
                "channels": self.channels,
                "n_focus": self.n_focus,
                "eps": self.eps,
                "precision": np.dtype(PRECISION_DICT[self.precision]).name,
                "trainable": self.trainable,
            },
            "@variables": {"adam_scale": to_numpy_array(self.adam_scale)},
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> ScalarRMSNorm:
        """Deserialize a ScalarRMSNorm from a dict."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "ScalarRMSNorm":
            raise ValueError(f"Invalid class for ScalarRMSNorm: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(
            channels=int(config["channels"]),
            n_focus=int(config["n_focus"]),
            eps=float(config["eps"]),
            precision=str(config["precision"]),
            trainable=bool(config["trainable"]),
        )
        prec = PRECISION_DICT[obj.precision.lower()]
        adam_scale = np.asarray(variables["adam_scale"], dtype=prec)
        adam_scale = adam_scale.reshape(obj.adam_scale.shape)
        obj.adam_scale = adam_scale
        return obj
