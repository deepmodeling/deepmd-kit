# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: ANN001
"""Factory binding the cuTile operators into the SO(2) convolution value path.

The value path replaces the dense rotate-to-local, radial degree mixing and
multi-layer gated mixing of ``so2_message`` with two operators, returning the
same pre-rotate-back per-focus local features the aggregation consumes.

Supported configuration
-----------------------
The factory validates the block layout and returns ``None`` when it does not
match, in which case the convolution keeps the dense reference path. Support is
deliberately narrower than the Triton path in two respects, both forced by the
tile model rather than by effort:

- the focus width must be a power of two, because it is a tile extent, which
  excludes the non-power-of-two width the Triton kernels handle by masking;
- the radial degree mixer must be the rank-one ``degree_channel`` form, whose
  per-edge kernel is a scalar per degree pair and therefore elementwise.

Cross-focus competition is also excluded: it would make the stack output depend
on a softmax over focus streams, and no deployed configuration in the DPA4
family enables it together with more than one focus stream.
"""

from __future__ import (
    annotations,
)

from typing import (
    TYPE_CHECKING,
)

import torch
from torch import (
    Tensor,
)

from ..common import (
    CUTILE_AVAILABLE,
    next_pow2,
)
from .so2_mixing_stack import (
    so2_mixing_stack,
)
from .so2_rotate_mix import (
    so2_rotate_mix,
)

if TYPE_CHECKING:
    from deepmd.pt.model.descriptor.sezm_nn.edge_cache import (
        EdgeFeatureCache,
    )

__all__ = ["make_cutile_value_path"]

_MAX_LMAX = 6


class CuTileValuePath:
    """Run the SO(2) value path of one convolution through the cuTile operators.

    The call contract mirrors ``so2_message(..., return_local=True)``: it returns
    the per-focus local features ``(E, F, D_m, Cf)`` and the projected radial
    features whose ``l = 0`` slice feeds the attention aggregation.

    The stacked weights are assembled from the live parameters on every call and
    are never cached: the first call may run inside a ``make_fx`` fake-tensor
    trace, where a cache would capture fake weights, and the parameters
    themselves change whenever a checkpoint is loaded or a training run reaches
    its next validation. The assembly is a short chain of parameter-only
    operations that the compile pipeline folds out of the hot path, and the
    padding and fp16 split that follow it happen inside the operator, where a
    compiler cannot elide the narrowing.
    """

    def __init__(self, conv) -> None:
        self._conv = conv

    def _stack_weights(self) -> tuple[Tensor, Tensor, Tensor]:
        """Stack the per-layer SO(2) block weights and gate projections."""
        conv = self._conv
        split = (conv.lmax + 1) * conv.so2_focus_dim
        blocks_m0, blocks_m1, gates = [], [], []
        for layer, linear in enumerate(conv.so2_linears):
            weight = linear._build_so2_weight().detach().permute(1, 0, 2).contiguous()
            blocks_m0.append(weight[:, :split, :split])
            blocks_m1.append(weight[:, split:, split:])
            non_linear = conv.non_linearities[layer]
            if type(non_linear).__name__ == "GatedActivation":
                gates.append(
                    non_linear.gate_linear.weight.detach()
                    .view(
                        conv.so2_focus_dim,
                        conv.n_focus,
                        conv.lmax * conv.so2_focus_dim,
                    )
                    .permute(1, 0, 2)
                )
        return (
            torch.stack(blocks_m0).contiguous(),
            torch.stack(blocks_m1).contiguous(),
            torch.stack(gates).contiguous(),
        )

    def __call__(
        self, x: Tensor, edge_cache: EdgeFeatureCache, radial_feat: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return the local features and the projected radial features."""
        conv = self._conv
        n_edge = edge_cache.src.shape[0]
        w0, w1, gw = self._stack_weights()

        rad_feat = (
            conv.radial_hidden_proj(radial_feat)
            if conv.radial_hidden_proj is not None
            else radial_feat
        )
        mixer = conv.radial_degree_mixer
        compact = torch.matmul(rad_feat.reshape(n_edge, -1), mixer.weight)
        u0 = so2_rotate_mix(
            x.contiguous(),
            edge_cache.src,
            edge_cache.D_full,
            compact.contiguous(),
            mixer.channel_basis.reshape(-1).contiguous(),
            conv.lmax,
            conv.so2_focus_dim,
            conv.n_focus,
        )
        x_local = so2_mixing_stack(u0, w0, w1, gw, conv.lmax, conv.so2_focus_dim)
        return (
            x_local.view(n_edge, conv.n_focus, 3 * conv.lmax + 1, conv.so2_focus_dim),
            rad_feat,
        )


def _is_supported(conv) -> bool:
    """Return whether ``conv`` matches the configuration the cuTile path serves.

    The block layout is decided before any submodule is inspected. A convolution
    outside the layout may not have built the SO(2) stack at all -- a Cartesian
    edge frame skips it entirely -- and the contract of this predicate is to
    decline, never to raise: the caller falls back to the dense reference.
    """
    focus_dim = conv.so2_focus_dim
    if (
        conv.mmax != 1
        or not 1 <= conv.lmax <= _MAX_LMAX
        or conv.mixing_layers < 2
        or conv.edge_cartesian
        or not conv.needs_local_frame
        or focus_dim != next_pow2(focus_dim)
        or conv.node_wise_grid_product is not None
        or conv.use_so2_attn_res
        or conv.layer_scale
        or (conv.focus_compete and conv.n_focus > 1)
    ):
        return False
    mixer = conv.radial_degree_mixer
    if mixer is None or mixer.mode != "degree_channel" or mixer.rank != 1:
        return False
    linears = conv.so2_linears
    if linears[0].weight_m0.dtype is not torch.float32:
        return False
    if any(type(norm).__name__ != "Identity" for norm in conv.so2_inter_norms):
        return False
    if any(linear.bias0 is not None for linear in linears):
        return False
    if any(
        linear.in_channels != focus_dim or linear.out_channels != focus_dim
        for linear in linears
    ):
        return False
    non_linears = conv.non_linearities
    if any(
        type(non_linears[layer]).__name__ != "GatedActivation"
        or (
            getattr(non_linears[layer].scalar_act, "activation", None)
            or getattr(non_linears[layer], "activation_function", None)
        )
        != "silu"
        for layer in range(conv.mixing_layers - 1)
    ):
        return False
    return type(non_linears[conv.mixing_layers - 1]).__name__ == "Identity"


def make_cutile_value_path(conv) -> CuTileValuePath | None:
    """Build the cuTile value-path entry for a convolution block.

    Parameters
    ----------
    conv : SO2Convolution
        The convolution block to accelerate.

    Returns
    -------
    CuTileValuePath or None
        The entry callable when ``cuda.tile`` is available and ``conv`` matches
        the supported configuration; otherwise ``None``, and the caller keeps the
        dense reference path.
    """
    if not CUTILE_AVAILABLE or not _is_supported(conv):
        return None
    return CuTileValuePath(conv)
