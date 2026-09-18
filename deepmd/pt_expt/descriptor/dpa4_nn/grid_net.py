# SPDX-License-Identifier: LGPL-3.0-or-later
"""pt_expt DPA4 grid nets with the optional fused pair projections."""

from typing import (
    Any,
)

import torch

from deepmd.dpmodel.common import (
    get_xp_precision,
)
from deepmd.dpmodel.descriptor.dpa4_nn.grid_net import S2GridNet as S2GridNetDP
from deepmd.dpmodel.descriptor.dpa4_nn.grid_net import SO3GridNet as SO3GridNetDP
from deepmd.pt_expt.common import (
    torch_module,
)
from deepmd.pt_expt.kernels.utils import (
    cuda_infer_level,
    triton_train_level,
)


def bind_grid_pair_operators(module: Any) -> None:
    """Bind the fused coefficient-grid pair operators that serve the layout.

    The inference operator (CUDA, register-resident walk) and the training
    operator (Triton tensor-core sandwich with analytic first and second
    order) are independent bindings; ``_pair_grid`` dispatches on the
    training mode. The training binding follows the measured crossover:
    below 75 slots the dense section is small and the operator's dispatch
    chain costs more than its kernels save on the host-bound
    configurations, so the narrow grids stay with the compiler.

    Parameters
    ----------
    module : Any
        PT-expt grid module receiving the eligible operator bindings.
    """
    module.dtype = get_xp_precision(torch, module.precision)
    if module.projector.to_grid_mat.dtype is not torch.float32:
        return
    slots = int(module.projector.to_grid_mat.shape[1])
    if cuda_infer_level() >= 1:
        from deepmd.pt_expt.kernels.cuda.dpa4.grid_pair import (
            SUPPORTED_SLOTS,
            grid_pair,
            op_available,
        )

        if op_available() and slots in SUPPORTED_SLOTS:
            module.cuda_infer_l_1_grid_pair = grid_pair
    if triton_train_level() >= 1 and slots >= 75:
        from deepmd.pt_expt.kernels.triton.sezm.grid_pair import (
            GRID_PAIR_TRITON_AVAILABLE,
            grid_pair_train,
        )

        if GRID_PAIR_TRITON_AVAILABLE:
            module.triton_train_l_1_grid_pair = grid_pair_train


def run_cute_infer_grid_pair(
    module: Any,
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor | None:
    """Run the CuTe grid product when its exact inference contract matches.

    Parameters
    ----------
    module : Any
        Grid module carrying the projector and frozen inference state.
    left : torch.Tensor
        Left coefficient operand with shape ``(N, D, F, n_frames * C)``.
    right : torch.Tensor
        Right coefficient operand with the same shape as ``left``.

    Returns
    -------
    torch.Tensor or None
        Coefficient result with the same shape as ``left``, or ``None`` when
        the exact CuTe contract is not satisfied.
    """
    from deepmd.pt_expt.kernels.cute.sezm import (
        runtime_policy,
    )

    if (
        not runtime_policy.is_cute_infer_enabled()
        or module.training
        or any(parameter.requires_grad for parameter in module.parameters())
    ):
        return None
    from deepmd.pt_expt.kernels.cute.sezm.output_grid.product import (
        maybe_run_cute_output_grid_product,
    )

    return maybe_run_cute_output_grid_product(
        left,
        right,
        module.projector.to_grid_mat,
        module.projector.from_grid_mat,
        n_frames=module.n_frames,
    )


@torch_module
class S2GridNet(S2GridNetDP):
    """S2 grid net with an opt-in fused CUDA pair projection."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        bind_grid_pair_operators(self)

    def run_cute_infer_grid_pair(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor | None:
        """Run the CuTe grid product when its exact inference contract matches.

        Parameters
        ----------
        left : torch.Tensor
            Left coefficient operand with shape ``(N, D, F, n_frames * C)``.
        right : torch.Tensor
            Right coefficient operand with the same shape as ``left``.

        Returns
        -------
        torch.Tensor or None
            Coefficient result with the same shape as ``left``, or ``None``
            when the exact CuTe contract is not satisfied.
        """
        return run_cute_infer_grid_pair(self, left, right)


@torch_module
class SO3GridNet(SO3GridNetDP):
    """SO(3) grid net with an opt-in fused CUDA pair projection."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        bind_grid_pair_operators(self)

    def run_cute_infer_grid_pair(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor | None:
        """Run the CuTe grid product when its exact inference contract matches.

        Parameters
        ----------
        left : torch.Tensor
            Left coefficient operand with shape ``(N, D, F, n_frames * C)``.
        right : torch.Tensor
            Right coefficient operand with the same shape as ``left``.

        Returns
        -------
        torch.Tensor or None
            Coefficient result with the same shape as ``left``, or ``None``
            when the exact CuTe contract is not satisfied.
        """
        return run_cute_infer_grid_pair(self, left, right)
