# SPDX-License-Identifier: LGPL-3.0-or-later
"""pt_expt DPA4 activations with the optional fused gated-activation kernel.

The dpmodel activations are array-API only. This wrapper injects the fused
Triton gated activation around :class:`GatedActivation`, mirroring
``deepmd.pt.model.descriptor.sezm_nn.activation``: one kernel per focus stream
replaces the gate projection, the sigmoid expansion and the degree-wise
multiply of the self-gated focus-major layout. The gate is resolved at
construction so export records a static dispatch choice; unsupported layouts
retain the dpmodel reference path.
"""

from __future__ import (
    annotations,
)

from typing import (
    TYPE_CHECKING,
    Any,
)

from deepmd.dpmodel.descriptor.dpa4_nn.activation import (
    GatedActivation as GatedActivationDP,
)
from deepmd.pt_expt.common import (
    torch_module,
)
from deepmd.pt_expt.kernels.utils import (
    triton_infer_level,
    triton_train_level,
)

if TYPE_CHECKING:
    import torch


@torch_module
class GatedActivation(GatedActivationDP):
    """Gated SO(2) activation with an opt-in fused Triton kernel."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Fast path (``DP_TRITON_INFER >= 1`` or ``DP_TRITON_TRAIN >= 1``): one
        # kernel per focus stream folds the gate projection, its sigmoid, the
        # degree expansion and the multiply of the self-gated ``fndc`` layout,
        # keeping the gate logits and the expanded gates off device memory.
        # The operator carries a differentiable backward and a hand-derived
        # second order, so it serves force-loss training as well.
        #
        # The binding is bounded by the register footprint the kernel needs to
        # hold one focus stream's degrees on chip: all degrees at
        # ``Cf <= 32`` and ``lmax <= 3`` at ``Cf = 64``. The wider shapes are
        # numerically complete through the operator, but end to end they lose
        # to the compiler-fused dense expression, whose intermediates the
        # scheduler shares with the surrounding graph while the operator
        # boundary forces its saved tensors to materialize.
        self.triton_infer_level = triton_infer_level()
        self.triton_train_level = triton_train_level()
        self._fused_gated_act = None
        register_footprint_ok = self.channels <= 32 or (
            self.channels <= 64 and self.lmax <= 3
        )
        if (
            1 <= self.lmax <= 6
            and self.mmax == 1
            and self.layout == "fndc"
            and self.activation_function == "silu"
            and not self.mlp_bias
            and register_footprint_ok
            and max(self.triton_infer_level, self.triton_train_level) >= 1
        ):
            from deepmd.pt_expt.kernels.triton.sezm.so2_value_path import (
                fused_gated_activation,
            )

            self._fused_gated_act = fused_gated_activation

    def call(self, x: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        active_level = (
            self.triton_train_level if self.training else self.triton_infer_level
        )
        if (
            self._fused_gated_act is not None
            and gate is None
            and x.is_cuda
            and active_level >= 1
        ):
            n_focus, n_edge = x.shape[0], x.shape[1]
            weight = self.gate_linear.weight.view(
                self.channels, self.n_focus, self.lmax * self.channels
            )
            out = self._fused_gated_act(
                x.reshape(n_focus, n_edge, -1).contiguous(),
                weight.permute(1, 0, 2).contiguous(),
                weight.permute(1, 2, 0).contiguous(),
                self.lmax,
                self.channels,
            )
            return out.view_as(x)
        return super().call(x, gate)
