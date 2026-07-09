# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Autograd operator and entry point wiring the fused CuTe kernels into the SeZM
SO(2) convolution value path.

:class:`_SO2ValuePathFunction` runs the fused forward kernel (saving only the
small inputs ``x``, ``D_to_m``, ``Kc``) and, on the force path, recomputes the
value path in the fused backward kernel -- so the per-edge ``E x D_m x C``
intermediates stay off DRAM across the whole autograd graph. :func:`make_cute_value_path`
builds the per-convolution entry :class:`_CuteSO2ValuePath`, which computes the
radial-/scalar-only tensors (``D_to_m``, ``Kc``, ``rad_feat``) in ordinary
autograd, invokes the operator to produce the pre-focus-compete local features
and the pre-mixing ``l = 0`` scalar, and applies the focus competition -- exactly
as the reference path does. The packed weights are extracted lazily on the first
call so they reflect the loaded checkpoint.
"""

from __future__ import (
    annotations,
)

from dataclasses import (
    dataclass,
)
from typing import (
    TYPE_CHECKING,
)

import torch

from deepmd.pt.model.descriptor.sezm_nn.indexing import (
    project_D_to_m,
)

from .forward import (
    SEZM_CUTE_AVAILABLE,
)

if TYPE_CHECKING:
    from deepmd.pt.model.descriptor.sezm_nn.edge_cache import (
        EdgeFeatureCache,
    )
    from deepmd.pt.model.descriptor.sezm_nn.so2 import (
        SO2Convolution,
    )

if SEZM_CUTE_AVAILABLE:
    from .backward import (
        BackwardRunner,
    )
    from .forward import (
        ForwardRunner,
    )

# Validated configuration of the fused operator: the deployed three-layer
# ``[gated, gated, identity]`` mixing stack in the ``lmax = 3, mmax = 1`` layout.
_SUPPORTED_LMAX = 3
_SUPPORTED_MMAX = 1
_SUPPORTED_LAYERS = 3


@dataclass
class _PackedWeights:
    """Static weights of the SO(2) value path, packed for the fused kernels.

    Attributes
    ----------
    so2_w : torch.Tensor
        Assembled block-diagonal SO2Linear weight per layer with shape
        (L, F, D_m*Cf, D_m*Cf), in ``(in, out)`` convention.
    gate_w : torch.Tensor
        GatedActivation ``FocusLinear`` weight per layer with shape
        (L, Cf, F, lmax*Cf); zero for non-gated layers.
    has_gate : torch.Tensor
        Boolean per-layer flag with shape (L,).
    channel_basis : torch.Tensor
        Radial degree-mixer channel basis with shape (C_wide,).
    """

    so2_w: torch.Tensor
    gate_w: torch.Tensor
    has_gate: torch.Tensor
    channel_basis: torch.Tensor


def _pack_weights(conv: SO2Convolution) -> _PackedWeights:
    """Extract and pack the SO(2) value-path weights from a convolution block."""
    n_layers = conv.mixing_layers
    so2_w = torch.stack(
        [
            conv.so2_linears[layer]._build_so2_weight().permute(1, 0, 2).contiguous()
            for layer in range(n_layers)
        ]
    )
    gate_w, has_gate = [], []
    for layer in range(n_layers):
        non_linear = conv.non_linearities[layer]
        if type(non_linear).__name__ == "GatedActivation" and non_linear.lmax > 0:
            gate_w.append(
                non_linear.gate_linear.weight.view(
                    conv.so2_focus_dim, conv.n_focus, conv.lmax * conv.so2_focus_dim
                ).contiguous()
            )
            has_gate.append(True)
        else:
            gate_w.append(
                torch.zeros_like(gate_w[0])
                if gate_w
                else torch.zeros(
                    conv.so2_focus_dim,
                    conv.n_focus,
                    conv.lmax * conv.so2_focus_dim,
                    device=so2_w.device,
                    dtype=so2_w.dtype,
                )
            )
            has_gate.append(False)
    return _PackedWeights(
        so2_w=so2_w,
        gate_w=torch.stack(gate_w),
        has_gate=torch.tensor(has_gate, device=so2_w.device, dtype=torch.bool),
        channel_basis=conv.radial_degree_mixer.channel_basis.reshape(-1).contiguous(),
    )


class _SO2ValuePathFunction(torch.autograd.Function):
    """Fused CuTe forward with a recompute backward for the force path."""

    @staticmethod
    def forward(ctx, x, d_to_m, kc, src, fwd_runner, bwd_runner):  # noqa: ANN001, ANN205
        with torch.no_grad():
            x_local, focus_gate = fwd_runner(
                x.detach(), src, d_to_m.detach(), kc.detach()
            )
        ctx.save_for_backward(x, d_to_m, kc)
        ctx.src = src
        ctx.bwd_runner = bwd_runner
        return x_local, focus_gate

    @staticmethod
    def backward(ctx, grad_local, grad_focus_gate):  # noqa: ANN001, ANN205
        x, d_to_m, kc = ctx.saved_tensors
        need = ctx.needs_input_grad
        grad_x, grad_d_to_m, grad_kc = ctx.bwd_runner(
            x.detach(),
            ctx.src,
            d_to_m.detach(),
            kc.detach(),
            grad_local.detach().contiguous(),
            grad_focus_gate.detach().contiguous(),
        )
        return (
            grad_x if need[0] else None,
            grad_d_to_m if need[1] else None,
            grad_kc if need[2] else None,
            None,
            None,
            None,
        )


class _CuteSO2ValuePath:
    """Per-convolution entry that runs the value path through the fused kernels.

    The convolution is held by reference so the packed weights are extracted
    lazily on the first call (after the checkpoint is loaded) and the kernels are
    compiled on first use.

    Parameters
    ----------
    conv : SO2Convolution
        The owning convolution block.
    bucket_fwd, threads_fwd, bucket_bwd, threads_bwd, rb, rn : int
        Fused-kernel launch configuration.
    """

    def __init__(
        self,
        conv: SO2Convolution,
        *,
        bucket_fwd: int = 32,
        threads_fwd: int = 1024,
        bucket_bwd: int = 16,
        threads_bwd: int = 512,
        rb: int = 4,
        rn: int = 4,
    ) -> None:
        self._conv = conv
        self._cfg = {
            "lmax": conv.lmax,
            "mmax": conv.mmax,
            "cf": conv.so2_focus_dim,
            "n_focus": conv.n_focus,
            "n_layers": conv.mixing_layers,
            "rb": rb,
            "rn": rn,
        }
        self._launch = {
            "bucket_fwd": bucket_fwd,
            "threads_fwd": threads_fwd,
            "bucket_bwd": bucket_bwd,
            "threads_bwd": threads_bwd,
        }
        self._fwd_runner = None
        self._bwd_runner = None

    def _build(self) -> None:
        weights = _pack_weights(self._conv)
        self._fwd_runner = ForwardRunner(
            weights,
            bucket=self._launch["bucket_fwd"],
            threads=self._launch["threads_fwd"],
            **self._cfg,
        )
        self._bwd_runner = BackwardRunner(
            weights,
            bucket=self._launch["bucket_bwd"],
            threads=self._launch["threads_bwd"],
            **self._cfg,
        )

    def __call__(
        self,
        x: torch.Tensor,
        edge_cache: EdgeFeatureCache,
        radial_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the SO(2) local features and radial features via the fused op.

        Parameters
        ----------
        x : torch.Tensor
            Node features with shape (N, D, C_wide).
        edge_cache : EdgeFeatureCache
            Precomputed edge cache (provides ``src`` and the Wigner ``D_full``).
        radial_feat : torch.Tensor
            Per-edge radial features with shape (E, lmax+1, C).

        Returns
        -------
        x_local : torch.Tensor
            Post-focus-compete local features with shape (E, F, D_m, Cf).
        rad_feat : torch.Tensor
            Projected radial features with shape (E, D_m, C_wide); its ``l = 0``
            slice is consumed by the attention aggregation.
        """
        if self._fwd_runner is None:
            self._build()
        conv = self._conv
        src = edge_cache.src

        # === Step 1. Radial-/scalar-only tensors (kept in ordinary autograd) ===
        d_to_m = project_D_to_m(
            edge_cache.D_full,
            conv.coeff_index_m,
            conv.ebed_dim_full,
            None,
            conv.lmax,
            conv.mmax,
        )
        rad_feat = radial_feat[:, conv.degree_index_m, :]
        rad_feat = conv.radial_hidden_proj(rad_feat)
        mixer = conv.radial_degree_mixer
        kernel_flat = mixer._project_radial(rad_feat)
        compact = kernel_flat.view(src.shape[0], mixer.degree_kernel_size, mixer.rank)
        kc = mixer._scatter_rank_kernel(compact).squeeze(-1)

        # === Step 2. Fused value path -> pre-focus-compete local + l=0 scalar ===
        x_local, focus_gate = _SO2ValuePathFunction.apply(
            x, d_to_m, kc, src, self._fwd_runner, self._bwd_runner
        )

        # === Step 3. Cross-focus softmax competition (rotation-free scalars) ===
        if conv.focus_compete and conv.n_focus > 1:
            alpha = conv._focus_alpha(focus_gate)
            x_local = x_local * alpha.to(dtype=x_local.dtype).unsqueeze(-1).unsqueeze(
                -1
            )
        return x_local, rad_feat


def _is_supported(conv: SO2Convolution) -> bool:
    """Return whether ``conv`` matches the validated fused-operator configuration."""
    if (
        conv.lmax != _SUPPORTED_LMAX
        or conv.mmax != _SUPPORTED_MMAX
        or conv.mixing_layers != _SUPPORTED_LAYERS
        or conv.node_wise_grid_product is not None
        or conv.use_so2_attn_res
        or conv.layer_scale
        or conv.radial_degree_mixer is None
        or conv.radial_hidden_proj is None
    ):
        return False
    if any(type(norm).__name__ != "Identity" for norm in conv.so2_inter_norms):
        return False
    if any(linear.bias0 is not None for linear in conv.so2_linears):
        return False
    non_linears = conv.non_linearities
    if any(
        type(non_linears[layer]).__name__ != "GatedActivation"
        or (
            getattr(non_linears[layer].scalar_act, "activation", None)
            or getattr(non_linears[layer], "activation_function", None)
        )
        != "silu"
        for layer in range(_SUPPORTED_LAYERS - 1)
    ):
        return False
    return type(non_linears[-1]).__name__ == "Identity"


def make_cute_value_path(conv: SO2Convolution) -> _CuteSO2ValuePath | None:
    """Build the fused CuTe value-path entry for a convolution block.

    Parameters
    ----------
    conv : SO2Convolution
        The convolution block to accelerate.

    Returns
    -------
    _CuteSO2ValuePath or None
        The entry callable when the CuTe backend is available and ``conv`` matches
        the validated configuration; otherwise ``None`` (the caller falls back to
        the reference path).
    """
    if not SEZM_CUTE_AVAILABLE or not _is_supported(conv):
        return None
    return _CuteSO2ValuePath(conv)
