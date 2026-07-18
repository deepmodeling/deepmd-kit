# SPDX-License-Identifier: LGPL-3.0-or-later
"""pt_expt interaction block with eval-time activation checkpointing.

The dpmodel :class:`SeZMInteractionBlock` is array-API only and never
checkpoints (array-API has no ``torch.utils.checkpoint``).  This wrapper injects
the reference pt activation-checkpoint policy around the two recomputable units
of the block -- the SO(2) convolution and each FFN subblock -- mirroring
``deepmd.pt.model.descriptor.sezm_nn.block``.  Checkpointing trades compute for
memory on the eval-time autograd path (force from ``autograd.grad``) and is
opt-in through the ``DP_ACT_INFER`` environment variable.
"""

from __future__ import (
    annotations,
)

import dataclasses
import os
from typing import (
    TYPE_CHECKING,
    Any,
)

import torch
from torch.utils.checkpoint import (
    checkpoint,
)

from deepmd.dpmodel.descriptor.dpa4_nn.block import (
    SeZMInteractionBlock as SeZMInteractionBlockDP,
)
from deepmd.dpmodel.descriptor.dpa4_nn.block import (
    exchange_ghost_features,
)
from deepmd.pt_expt.common import (
    torch_module,
)

if TYPE_CHECKING:
    from deepmd.dpmodel.descriptor.dpa4_nn.edge_cache import (
        EdgeCache,
    )

# Environment values that enable an inference flag.
_TRUTHY = {"1", "true", "yes", "on"}


@torch_module
class SeZMInteractionBlock(SeZMInteractionBlockDP):
    """SeZM interaction block with eval-time activation checkpointing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Inference env policy, sampled once here (see
        # ``_use_infer_activation_checkpoint``).
        self._act_infer = os.environ.get("DP_ACT_INFER", "").strip().lower() in _TRUTHY
        self._compile_infer = (
            os.environ.get("DP_COMPILE_INFER", "").strip().lower() in _TRUTHY
        )

    def _use_infer_activation_checkpoint(self, *tensors: torch.Tensor) -> bool:
        """Return whether eval-time activation checkpointing should be used.

        Disabled on the compiled inference path (``DP_COMPILE_INFER``): Inductor
        already reuses activation buffers, so recomputation only adds latency for
        a negligible memory gain there.
        """
        return (
            not self.training
            and self._act_infer
            and not self._compile_infer
            and torch.is_grad_enabled()
            and any(tensor.requires_grad for tensor in tensors)
        )

    def _run_so2_unit(
        self,
        x: torch.Tensor,
        edge_cache: EdgeCache,
        radial_feat: torch.Tensor,
        comm_dict: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if comm_dict is not None:
            x = exchange_ghost_features(x, comm_dict)
        if self._use_infer_activation_checkpoint(x, radial_feat):
            edge_cache_no_proj = dataclasses.replace(
                edge_cache,
                D_to_m_cache=None,
                Dt_from_m_cache=None,
            )
            return checkpoint(
                lambda x_, radial_feat_: self._run_so2_unit_impl(
                    x_,
                    edge_cache_no_proj,
                    radial_feat_,
                ),
                x,
                radial_feat,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        return self._run_so2_unit_impl(x, edge_cache, radial_feat)

    def _run_ffn_unit(self, x: torch.Tensor, unit_idx: int) -> torch.Tensor:
        if self._use_infer_activation_checkpoint(x):
            return checkpoint(
                lambda x_: self._run_ffn_unit_impl(x_, unit_idx),
                x,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        return self._run_ffn_unit_impl(x, unit_idx)
