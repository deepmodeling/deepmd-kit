# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Normalization layers for the SeZM descriptor.

This module defines the packed-layout, reduced-layout, generic, and scalar
RMS normalization layers used throughout SeZM.
"""

from __future__ import (
    annotations,
)

from typing import (
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

from .indexing import (
    map_degree_idx,
)
from .utils import (
    np_safe,
    safe_numpy_to_tensor,
)


class RMSNorm(nn.Module):
    """
    Generic RMSNorm on tensors with shape `(..., C)`.

    This is the plain channel-wise RMS normalization used for non-equivariant
    branches whose last axis stores feature channels. A learnable affine scale is
    applied on the channel axis only, while all leading axes are treated as batch
    dimensions.

    Parameters
    ----------
    channels
        Feature dimension of the last axis.
    eps
        Small epsilon for numerical stability.
    dtype
        Parameter and computation dtype. Caller should pass compute_dtype (fp32+)
        for numerical stability.
    trainable
        Whether parameters are trainable.
    """

    def __init__(
        self,
        *,
        channels: int,
        eps: float = 1e-7,
        dtype: torch.dtype,
        trainable: bool,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.dtype = dtype
        self.device = env.DEVICE
        self.eps = float(eps)
        self.register_buffer(
            "eps_tensor",
            torch.tensor(self.eps, dtype=self.dtype, device=self.device),
            persistent=False,
        )

        # adam_ prefix routes this to Adam (no weight decay) in HybridMuon.
        self.adam_scale = nn.Parameter(
            torch.ones(self.channels, dtype=self.dtype, device=self.device)
        )

        for p in self.parameters():
            p.requires_grad = trainable

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Input tensor with shape `(..., C)`.

        Returns
        -------
        torch.Tensor
            Normalized tensor with shape `(..., C)`, same dtype as input.
        """
        in_dtype = x.dtype
        x = x.to(dtype=self.dtype)
        inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps_tensor)
        scale = self.adam_scale.view(*([1] * (x.ndim - 1)), self.channels)
        x = x * inv_rms * scale
        return x.to(dtype=in_dtype)

    def serialize(self) -> dict[str, Any]:
        trainable = all(p.requires_grad for p in self.parameters())
        state = self.state_dict()
        return {
            "@class": "RMSNorm",
            "@version": 1,
            "config": {
                "channels": self.channels,
                "eps": self.eps,
                "precision": RESERVED_PRECISION_DICT[self.dtype],
                "trainable": trainable,
            },
            "@variables": {key: np_safe(value) for key, value in state.items()},
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> RMSNorm:
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "RMSNorm":
            raise ValueError(f"Invalid class for RMSNorm: {data_cls}")
        version = int(data.pop("@version"))
        if version != 1:
            raise ValueError(f"Unsupported RMSNorm version: {version}")
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


class EquivariantRMSNorm(nn.Module):
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
    lmax
        Maximum spherical harmonic degree.
    channels
        Channels per `(l, m)` coefficient in each focus stream.
    n_focus
        Number of focus streams. Affine parameters are independent per focus.
    eps
        Small epsilon for numerical stability.
    dtype
        Parameter and computation dtype. Caller should pass compute_dtype (fp32+)
        for numerical stability and handle input/output conversion at boundaries.
    trainable
        Whether parameters are trainable.
    """

    def __init__(
        self,
        lmax: int,
        channels: int,
        n_focus: int = 1,
        *,
        eps: float = 1e-5,
        dtype: torch.dtype,
        trainable: bool,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.channels = int(channels)
        self.n_focus = int(n_focus)
        self.dtype = dtype
        self.device = env.DEVICE
        self.eps = float(eps)
        self.register_buffer(
            "eps_tensor",
            torch.tensor(self.eps, dtype=self.dtype, device=self.device),
            persistent=False,
        )

        # === Step 1. Learnable Parameters ===
        # Store affine scales in degree-major layout (L, F, C). This matches the
        # packed output layout after degree expansion
        # adam_ prefix routes this to Adam (no weight decay) in HybridMuon.
        self.adam_scale = nn.Parameter(
            torch.ones(
                self.lmax + 1,
                self.n_focus,
                self.channels,
                dtype=self.dtype,
                device=self.device,
            )
        )
        # Bias only for l=0, independent per focus.
        self.bias = nn.Parameter(
            torch.zeros(
                self.n_focus, self.channels, dtype=self.dtype, device=self.device
            )
        )

        # === Step 2. Index and Weight Buffers ===
        expand_index = map_degree_idx(self.lmax, device=self.device)
        self.register_buffer("expand_index", expand_index, persistent=True)

        # Pre-fuse degree balancing and channel averaging into a single weight:
        #   w_d = 1 / ((2l+1) * (lmax+1) * C)
        # so that
        #   mean_variance = einsum('ndfc,d->nf', x^2, balance_weight)
        # directly computes the shared RMS statistic without allocating an
        # intermediate (N, D, F, C) buffer beyond x^2 itself.
        weights_list = []
        scale = 1.0 / ((self.lmax + 1) * self.channels)
        for l in range(self.lmax + 1):
            w = scale / (2 * l + 1)
            weights_list.extend([w] * (2 * l + 1))
        balance_weight = torch.tensor(
            weights_list, dtype=self.dtype, device=self.device
        )
        self.register_buffer("balance_weight", balance_weight, persistent=True)

        for p in self.parameters():
            p.requires_grad = trainable

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Features with shape `(N, D, F, C)` where `D = (lmax + 1)^2`.

        Returns
        -------
        torch.Tensor
            Normalized features with shape `(N, D, F, C)`, same dtype as input.
        """
        in_dtype = x.dtype
        x = x.to(dtype=self.dtype)
        x0 = x[:, :1, :, :]  # (N, 1, F, C)
        xt = x[:, 1:, :, :]  # (N, D-1, F, C)

        # === Step 1. Center the scalar slice ===
        x0 = x0 - x0.mean(dim=-1, keepdim=True)

        # === Step 2. Compute a shared degree-balanced RMS ===
        mean_variance = x0.square().sum(dim=(1, 3)) * self.balance_weight[0]
        if xt.numel() > 0:
            mean_variance = mean_variance + torch.einsum(
                "ndfc,d->nf", xt * xt, self.balance_weight[1:]
            )
        inv_rms = (
            torch.rsqrt(mean_variance + self.eps_tensor).unsqueeze(1).unsqueeze(-1)
        )

        x0 = x0 * inv_rms
        if xt.numel() > 0:
            xt = xt * inv_rms

        # === Step 3. Apply per-degree affine parameters ===
        expanded_scale = torch.index_select(
            self.adam_scale, dim=0, index=self.expand_index
        )
        expanded_scale = expanded_scale.unsqueeze(0)  # (1, D, F, C)
        x0 = x0 * expanded_scale[:, :1, :, :]
        if xt.numel() > 0:
            xt = xt * expanded_scale[:, 1:, :, :]

        # === Step 4. Add scalar bias and restore layout ===
        bias0 = self.bias.reshape(1, 1, self.n_focus, -1)  # (1, 1, F, C)
        x0 = x0 + bias0

        out = x0 if xt.numel() == 0 else torch.cat([x0, xt], dim=1)
        out = out.to(dtype=in_dtype)
        return out

    def serialize(self) -> dict[str, Any]:
        trainable = all(p.requires_grad for p in self.parameters())
        state = self.state_dict()
        return {
            "@class": "EquivariantRMSNorm",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "channels": self.channels,
                "n_focus": self.n_focus,
                "eps": self.eps,
                "precision": RESERVED_PRECISION_DICT[self.dtype],
                "trainable": trainable,
            },
            "@variables": {key: np_safe(value) for key, value in state.items()},
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> EquivariantRMSNorm:
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "EquivariantRMSNorm":
            raise ValueError(f"Invalid class for EquivariantRMSNorm: {data_cls}")
        version = int(data.pop("@version"))
        if version != 1:
            raise ValueError(f"Unsupported EquivariantRMSNorm version: {version}")
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


class ReducedEquivariantRMSNorm(nn.Module):
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
    lmax
        Maximum spherical harmonic degree.
    mmax
        Maximum order kept in the truncated layout.
    channels
        Number of channels per retained coefficient.
    degree_index_m
        Degree index per coefficient in m-major truncated layout, with shape
        `(D_m_trunc,)`.
    n_focus
        Number of focus streams.
    eps
        Epsilon for numerical stability.
    dtype
        Parameter and computation dtype. Caller should pass compute_dtype (fp32+)
        for numerical stability.
    trainable
        Whether parameters are trainable.
    """

    def __init__(
        self,
        *,
        lmax: int,
        mmax: int,
        channels: int,
        degree_index_m: torch.Tensor,
        n_focus: int = 1,
        eps: float = 1e-5,
        dtype: torch.dtype,
        trainable: bool,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.mmax = int(mmax)
        self.channels = int(channels)
        self.n_focus = int(n_focus)
        self.eps = float(eps)
        self.dtype = dtype
        self.device = env.DEVICE
        self.register_buffer(
            "eps_tensor",
            torch.tensor(self.eps, dtype=self.dtype, device=self.device),
            persistent=False,
        )

        if degree_index_m.dtype != torch.long:
            degree_index_m = degree_index_m.to(dtype=torch.long)
        self.register_buffer("degree_index_m", degree_index_m, persistent=True)

        # Pre-fuse degree balancing and channel averaging into a single weight:
        #   w_d = 1 / (n_coeff_l * (lmax+1) * C)
        # where n_coeff_l is the number of retained coefficients for degree l in
        # the reduced layout.
        weights = torch.zeros(
            degree_index_m.numel(), dtype=self.dtype, device=self.device
        )
        scale = 1.0 / ((self.lmax + 1) * self.channels)
        for l in range(self.lmax + 1):
            n_coeff_l = 2 * min(l, self.mmax) + 1
            w_l = scale / float(n_coeff_l)
            weights[degree_index_m == l] = w_l
        if torch.any(weights == 0):
            raise ValueError(
                "ReducedEquivariantRMSNorm: balance_weight has zeros; degree_index_m may be invalid."
            )
        self.register_buffer("balance_weight", weights, persistent=True)

        # adam_ prefix routes this to Adam (no weight decay) in HybridMuon.
        self.adam_scale = nn.Parameter(
            torch.ones(
                self.n_focus,
                self.lmax + 1,
                self.channels,
                dtype=self.dtype,
                device=self.device,
            )
        )
        self.bias0 = nn.Parameter(
            torch.zeros(
                self.n_focus,
                self.channels,
                dtype=self.dtype,
                device=self.device,
            )
        )

        for p in self.parameters():
            p.requires_grad = trainable

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Input tensor with shape (E, F, D_m_trunc, C).

        Returns
        -------
        torch.Tensor
            Normalized tensor with shape `(E, F, D_m_trunc, C)`, same dtype as
            input.
        """
        in_dtype = x.dtype
        x = x.to(dtype=self.dtype)
        x0 = x[:, :, :1, :]  # (E, F, 1, C)
        xt = x[:, :, 1:, :]  # (E, F, D_m_trunc-1, C)

        # === Step 1. Center the scalar slice ===
        x0 = x0 - x0.mean(dim=-1, keepdim=True)

        # === Step 2. Compute a shared degree-balanced RMS ===
        mean_variance = x0.square().sum(dim=(2, 3)) * self.balance_weight[0]
        if xt.numel() > 0:
            mean_variance = mean_variance + torch.einsum(
                "efdc,d->ef", xt * xt, self.balance_weight[1:]
            )
        inv_rms = (
            torch.rsqrt(mean_variance + self.eps_tensor).unsqueeze(-1).unsqueeze(-1)
        )

        x0 = x0 * inv_rms
        if xt.numel() > 0:
            xt = xt * inv_rms

        # === Step 3. Apply per-degree affine parameters ===
        expanded_scale = torch.index_select(
            self.adam_scale, dim=1, index=self.degree_index_m
        )
        expanded_scale = expanded_scale.unsqueeze(0)  # (1, F, D_m_trunc, C)
        x0 = x0 * expanded_scale[:, :, :1, :]
        if xt.numel() > 0:
            xt = xt * expanded_scale[:, :, 1:, :]

        # === Step 4. Add scalar bias and restore layout ===
        bias0 = self.bias0.reshape(1, self.n_focus, 1, -1)  # (1, F, 1, C)
        x0 = x0 + bias0

        out = x0 if xt.numel() == 0 else torch.cat([x0, xt], dim=2)
        out = out.to(dtype=in_dtype)
        return out

    def serialize(self) -> dict[str, Any]:
        trainable = all(p.requires_grad for p in self.parameters())
        state = self.state_dict()
        return {
            "@class": "ReducedEquivariantRMSNorm",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "mmax": self.mmax,
                "channels": self.channels,
                "degree_index_m": np_safe(self.degree_index_m),
                "n_focus": self.n_focus,
                "eps": self.eps,
                "precision": RESERVED_PRECISION_DICT[self.dtype],
                "trainable": trainable,
            },
            "@variables": {key: np_safe(value) for key, value in state.items()},
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> ReducedEquivariantRMSNorm:
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "ReducedEquivariantRMSNorm":
            raise ValueError(f"Invalid class for ReducedEquivariantRMSNorm: {data_cls}")
        version = int(data.pop("@version"))
        if version != 1:
            raise ValueError(
                f"Unsupported ReducedEquivariantRMSNorm version: {version}"
            )
        config = data.pop("config")
        variables = data.pop("@variables")
        degree_index_m = safe_numpy_to_tensor(
            config.pop("degree_index_m"),
            device=env.DEVICE,
            dtype=torch.long,
        )
        precision = config.pop("precision")
        config["dtype"] = PRECISION_DICT[precision]
        config["degree_index_m"] = degree_index_m
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


class ScalarRMSNorm(nn.Module):
    """
    Lightweight per-focus RMSNorm for scalar branches.

    This is the unified scalar norm used by SeZM:
    - `n_focus=1` naturally degenerates to the single-stream behavior.
    - `n_focus>1` uses independent learnable scales per focus stream.
    Bias is intentionally omitted to keep the gate paths minimal.

    Parameters
    ----------
    channels
        Feature dimension of the last axis.
    n_focus
        Number of focus streams.
    eps
        Small epsilon for numerical stability.
    dtype
        Parameter and computation dtype. Caller should pass compute_dtype (fp32+)
        for numerical stability.
    trainable
        Whether parameters are trainable.
    """

    def __init__(
        self,
        *,
        channels: int,
        n_focus: int = 1,
        eps: float = 1e-7,
        dtype: torch.dtype,
        trainable: bool,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.n_focus = int(n_focus)
        self.dtype = dtype
        self.device = env.DEVICE
        self.eps = float(eps)
        self.register_buffer(
            "eps_tensor",
            torch.tensor(self.eps, dtype=self.dtype, device=self.device),
            persistent=False,
        )

        # adam_ prefix routes this to Adam (no weight decay) in HybridMuon.
        self.adam_scale = nn.Parameter(
            torch.ones(
                self.n_focus,
                self.channels,
                dtype=self.dtype,
                device=self.device,
            )
        )

        for p in self.parameters():
            p.requires_grad = trainable

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Input tensor with shape (B, F, C) or (B, C) when `n_focus=1`.

        Returns
        -------
        torch.Tensor
            Normalized tensor with the same shape as input and same dtype.
        """
        in_dtype = x.dtype
        x = x.to(dtype=self.dtype)

        if x.ndim == 2:
            inv_rms = torch.rsqrt(
                x.square().mean(dim=-1, keepdim=True) + self.eps_tensor
            )
            x = x * inv_rms
            x = x * self.adam_scale[0]
            return x.to(dtype=in_dtype)

        inv_rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps_tensor)
        x = x * inv_rms
        x = x * self.adam_scale.unsqueeze(0)
        return x.to(dtype=in_dtype)

    def serialize(self) -> dict[str, Any]:
        trainable = all(p.requires_grad for p in self.parameters())
        state = self.state_dict()
        return {
            "@class": "ScalarRMSNorm",
            "@version": 1,
            "config": {
                "channels": self.channels,
                "n_focus": self.n_focus,
                "eps": self.eps,
                "precision": RESERVED_PRECISION_DICT[self.dtype],
                "trainable": trainable,
            },
            "@variables": {key: np_safe(value) for key, value in state.items()},
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> ScalarRMSNorm:
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "ScalarRMSNorm":
            raise ValueError(f"Invalid class for ScalarRMSNorm: {data_cls}")
        version = int(data.pop("@version"))
        if version != 1:
            raise ValueError(f"Unsupported ScalarRMSNorm version: {version}")
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
