# SPDX-License-Identifier: LGPL-3.0-or-later
"""pt_expt DPA4 grid nets with the optional fused CUDA pair projection."""

from typing import (
    Any,
)

import torch

from deepmd.dpmodel.descriptor.dpa4_nn.grid_net import S2GridNet as S2GridNetDP
from deepmd.dpmodel.descriptor.dpa4_nn.grid_net import SO3GridNet as SO3GridNetDP
from deepmd.pt_expt.common import (
    torch_module,
)
from deepmd.pt_expt.kernels.utils import (
    cuda_infer_level,
)


def _bind_grid_pair(module: Any) -> None:
    """Bind the fused coefficient-grid pair operator when it serves the layout."""
    if (
        cuda_infer_level() < 1
        or module.projector.to_grid_mat.dtype is not torch.float32
    ):
        return
    from deepmd.pt_expt.kernels.cuda.dpa4.grid_pair import (
        SUPPORTED_SLOTS,
        grid_pair,
        op_available,
    )

    slots = int(module.projector.to_grid_mat.shape[1])
    if op_available() and slots in SUPPORTED_SLOTS:
        module._grid_pair_fn = grid_pair


@torch_module
class S2GridNet(S2GridNetDP):
    """S2 grid net with an opt-in fused CUDA pair projection."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        _bind_grid_pair(self)


@torch_module
class SO3GridNet(SO3GridNetDP):
    """SO(3) grid net with an opt-in fused CUDA pair projection."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        _bind_grid_pair(self)
