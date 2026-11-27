# SPDX-License-Identifier: LGPL-3.0-or-later
"""pt_expt DPA4 grid nets with the optional fused pair projections."""

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
    triton_train_level,
)


def _refresh_grid_projection(module: Any, incompatible_keys: Any) -> None:
    """Rebuild derived matrices after the saved projector buffers are loaded."""
    module._set_grid_projection()


def _bind_grid_pair(module: Any) -> None:
    """Bind the fused coefficient-grid pair operators that serve the layout.

    The inference operator (CUDA, register-resident walk) and the training
    operator (Triton tensor-core sandwich with analytic first and second
    order) are independent bindings; ``_pair_grid`` dispatches on the
    training mode. The training binding follows the measured crossover:
    unprojected grids below 75 slots retain the compiler composition.
    Inversion-projected products use disjoint parity tiles at every width,
    avoiding grid fields in the forward and both derivative orders.
    """
    if module.projector.to_grid_mat.dtype is not torch.float32:
        return
    slots = int(module.projector.to_grid_mat.shape[1])
    if cuda_infer_level() >= 1:
        from deepmd.pt_expt.kernels.cuda.dpa4.grid_pair import (
            SUPPORTED_PARITY_SLOTS,
            SUPPORTED_SLOTS,
            grid_pair,
            op_available,
        )

        if (
            op_available()
            and slots in SUPPORTED_SLOTS
            and (
                not module._project_inversion
                or (slots, module._grid_even_count) in SUPPORTED_PARITY_SLOTS
            )
        ):
            module._grid_pair_fn = grid_pair
    if triton_train_level() >= 1 and (module._project_inversion or slots >= 75):
        from deepmd.pt_expt.kernels.triton.sezm.grid_pair import (
            GRID_PAIR_TRITON_AVAILABLE,
            grid_pair_train,
        )

        if GRID_PAIR_TRITON_AVAILABLE:
            module._grid_pair_train_fn = grid_pair_train


@torch_module
class S2GridNet(S2GridNetDP):
    """S2 grid net with an opt-in fused CUDA pair projection."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.register_load_state_dict_post_hook(_refresh_grid_projection)
        _bind_grid_pair(self)


@torch_module
class SO3GridNet(SO3GridNetDP):
    """SO(3) grid net with an opt-in fused CUDA pair projection."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.register_load_state_dict_post_hook(_refresh_grid_projection)
        _bind_grid_pair(self)
