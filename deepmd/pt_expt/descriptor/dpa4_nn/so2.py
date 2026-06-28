# SPDX-License-Identifier: LGPL-3.0-or-later
"""pt_expt SO(2) convolution and radial mixer with opt-in fused Triton kernels.

The dpmodel SO(2) modules are array-API only.  These wrappers inject the
reference pt opt-in Triton inference path (``DP_TRITON_INFER``) around the two
rotation hot paths of the SO(2) convolution and the low-rank branch of the
dynamic radial degree mixer, mirroring
``deepmd.pt.model.descriptor.sezm_nn.so2``.  The kernels run only during
inference (``not self.training``); training and CPU / fp64 inference fall back to
the dpmodel dense path.
"""

from __future__ import (
    annotations,
)

import os
from typing import (
    TYPE_CHECKING,
    Any,
)

from deepmd.dpmodel.descriptor.dpa4_nn.so2 import (
    DynamicRadialDegreeMixer as DynamicRadialDegreeMixerDP,
)
from deepmd.dpmodel.descriptor.dpa4_nn.so2 import SO2Convolution as SO2ConvolutionDP
from deepmd.pt_expt.common import (
    torch_module,
)

if TYPE_CHECKING:
    import torch

    from deepmd.dpmodel.descriptor.dpa4_nn.edge_cache import (
        EdgeFeatureCache,
    )

_TRITON_INFER_TRUE = ("1", "true", "yes", "on")


def use_triton_infer() -> bool:
    """Return whether the opt-in Triton inference kernels are enabled.

    The flag is controlled by the ``DP_TRITON_INFER`` environment variable and
    is read at module construction time so that it becomes a compile-time
    constant in the traced (``make_fx``) graph. It only takes effect during
    inference; training always uses the dense reference path.

    Returns
    -------
    bool
        ``True`` when ``DP_TRITON_INFER`` is set to a truthy value.
    """
    return os.environ.get("DP_TRITON_INFER", "0").strip().lower() in _TRITON_INFER_TRUE


@torch_module
class DynamicRadialDegreeMixer(DynamicRadialDegreeMixerDP):
    """Dynamic radial degree mixer with an opt-in fused Triton low-rank branch."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Inference fast path (opt-in via ``DP_TRITON_INFER``): a fused Triton
        # kernel replaces the dense scatter and the tiny batched matmul of the
        # ``degree_channel`` low-rank branch in the ``mmax == 1`` layout.
        self.use_triton_infer = use_triton_infer()
        self._radial_mix_block = None
        if (
            self.use_triton_infer
            and self.mode == "degree_channel"
            and self.rank > 0
            and self.mmax == 1
        ):
            from .triton.radial_mix import (
                radial_mix_block,
            )

            self._radial_mix_block = radial_mix_block

    def _mix_rank_compact(
        self, compact: torch.Tensor, x_local: torch.Tensor
    ) -> torch.Tensor:
        if self._radial_mix_block is not None and not self.training:
            return self._radial_mix_block(
                compact, x_local, self.channel_basis, self.lmax
            )
        return super()._mix_rank_compact(compact, x_local)


@torch_module
class SO2Convolution(SO2ConvolutionDP):
    """SO(2) convolution with opt-in fused Triton rotation kernels."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # ``use_triton_infer`` is read once at construction so it is a
        # compile-time constant in the traced (``make_fx``) graph, and it only
        # takes effect during inference.
        self.use_triton_infer = use_triton_infer()

        # === Triton rotation kernels: block for mmax == 1, dense otherwise ===
        self._rotate_to_local_fn = None
        self._rotate_back_fn = None
        if self.use_triton_infer:
            from .triton.so2_rotation import (
                rotate_back_block_so2,
                rotate_back_dense,
                rotate_to_local_block,
                rotate_to_local_dense,
            )

            if self.mmax == 1:
                self._rotate_to_local_fn = lambda x, src, wigner: rotate_to_local_block(
                    x, src, wigner, self.lmax
                )
                # The block kernel reads the (E, F, D_m, Cf) focus layout directly,
                # so the rotate-back path passes ``x_local`` before the global
                # reshape and the transpose-back copy is skipped.
                self._rotate_back_fn = lambda x_local, wigner: rotate_back_block_so2(
                    x_local, wigner, self.lmax
                )
            else:
                self._rotate_to_local_fn = lambda x, src, wigner: rotate_to_local_dense(
                    x, src, wigner, self.coeff_index_m, self.ebed_dim_full
                )
                self._rotate_back_fn = lambda x_local, wigner: rotate_back_dense(
                    x_local, wigner, self.coeff_index_m, self.ebed_dim_full
                )

    def _rotate_to_local(
        self, x: torch.Tensor, edge_cache: EdgeFeatureCache
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.use_triton_infer and not self.training:
            # ``self._rotate_to_local_fn`` was bound in ``__init__`` (the block
            # kernel for the m-major ``mmax == 1`` layout, dense otherwise).
            D_full = edge_cache.D_full
            x_local = self._rotate_to_local_fn(x, edge_cache.src, D_full)
            x_dst_local: torch.Tensor | None = None
            if self.node_wise_grid_product is not None:
                x_dst_local = self._rotate_to_local_fn(x, edge_cache.dst, D_full)
            return x_local, x_dst_local
        return super()._rotate_to_local(x, edge_cache)

    def _rotate_back(
        self, x_local: torch.Tensor, edge_cache: EdgeFeatureCache, n_edge: int
    ) -> torch.Tensor:
        if self.use_triton_infer and not self.training:
            Dt_full = edge_cache.Dt_full
            if self.mmax == 1:
                # The block kernel consumes the (E, F, D_m, Cf) focus layout in
                # place, folding the inverse transpose into its channel addressing.
                return self._rotate_back_fn(x_local, Dt_full)
            # Restore reduced global layout (E, D_m, C_wide) for the dense kernel.
            x_std = (
                x_local.transpose(1, 2)
                .contiguous()
                .reshape(n_edge, self.reduced_dim, self.hidden_channels)
            )
            return self._rotate_back_fn(x_std, Dt_full)
        return super()._rotate_back(x_local, edge_cache, n_edge)
