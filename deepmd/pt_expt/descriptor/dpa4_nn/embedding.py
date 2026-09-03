# SPDX-License-Identifier: LGPL-3.0-or-later
"""PyTorch runtime bindings for DPA4 initial embeddings."""

from __future__ import (
    annotations,
)

from typing import (
    Any,
)

import torch

from deepmd.dpmodel.common import (
    get_xp_precision,
)
from deepmd.dpmodel.descriptor.dpa4_nn.embedding import (
    GeometricInitialEmbedding as GeometricInitialEmbeddingDP,
)
from deepmd.pt_expt.common import (
    torch_module,
)
from deepmd.pt_expt.kernels.cute.sezm.so2.wigner_layout import (
    ZONAL_PANEL_OFFSETS,
)
from deepmd.pt_expt.kernels.utils import (
    cuda_infer_level,
)

from .edge_cache import (
    cached_edge_csr,
)


@torch_module
class GeometricInitialEmbedding(GeometricInitialEmbeddingDP):
    """Geometric initial embedding with an optional fused CUDA scatter."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.register_buffer(
            "packed_zonal_offsets",
            torch.tensor(
                ZONAL_PANEL_OFFSETS if self.lmax == 3 else (),
                device=self.non_scalar_row_index.device,
                dtype=torch.long,
            ),
            persistent=False,
        )
        # === Fused message-and-scatter operator ===
        # The reference composition materializes the per-edge message, an
        # (E, D-1, C) tensor that dominates the cost of this module. The fused
        # operator keeps it in registers and reduces through the destination CSR.
        self.cuda_infer_l_1_scatter = False
        # ``None`` keeps the runtime ``zonal_coupling.is_cuda`` dispatch; the
        # freeze pins it to the AOTI target because tracing always runs on CPU.
        self.force_cuda_infer_l_1_scatter: bool | None = None
        if (
            cuda_infer_level() >= 1
            and get_xp_precision(torch, self.precision) is torch.float32
        ):
            from deepmd.pt_expt.kernels.cuda.dpa4.zonal_scatter import (
                op_available,
                supported,
            )

            self.cuda_infer_l_1_scatter = op_available() and supported(
                self.lmax, self.ebed_dim - 1, self.channels
            )

    def run_cute_infer_gie(
        self,
        *,
        n_nodes: int | torch.SymInt,
        edge_cache: Any,
        radial_feat: torch.Tensor,
        zonal_coupling: torch.Tensor,
        spin_l1_message: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Run the CuTe GIE path when its exact inference contract matches.

        Parameters
        ----------
        n_nodes : int or torch.SymInt
            Number of nodes addressed by the edge cache.
        edge_cache : EdgeCache
            Per-forward geometric edge cache.
        radial_feat : torch.Tensor
            Radial features with shape ``(E, lmax, C)``.
        zonal_coupling : torch.Tensor
            Zonal coupling with shape ``(E, D - 1)``.
        spin_l1_message : torch.Tensor, optional
            Native-spin message with shape ``(E, 3, C)``.

        Returns
        -------
        torch.Tensor or None
            Initial embedding with shape ``(N, D, C)``, or ``None`` when the
            exact CuTe contract is not satisfied.
        """
        if self.training or spin_l1_message is not None:
            return None
        from deepmd.pt_expt.kernels.cute.sezm.gie import (
            maybe_run_cute_gie,
        )

        return maybe_run_cute_gie(
            self,
            n_nodes=n_nodes,
            edge_cache=edge_cache,
            radial_feat=radial_feat,
            zonal_coupling=zonal_coupling,
        )

    def can_run_cuda_infer_l_1_scatter(self, zonal_coupling: torch.Tensor) -> bool:
        """Return whether the fused scatter serves the runtime or trace target.

        Parameters
        ----------
        zonal_coupling : torch.Tensor
            Zonal coupling with shape ``(E, D - 1)``.

        Returns
        -------
        bool
            Whether the CUDA inference level-one scatter is eligible.
        """
        target_is_cuda = (
            zonal_coupling.is_cuda
            if self.force_cuda_infer_l_1_scatter is None
            else self.force_cuda_infer_l_1_scatter
        )
        return self.cuda_infer_l_1_scatter and not self.training and target_is_cuda

    def run_cuda_infer_l_1_scatter(
        self,
        n_nodes: int | torch.SymInt,
        edge_cache: Any,
        radial_feat: torch.Tensor,
        zonal_coupling: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build and reduce the geometric message with the fused CUDA operator.

        Parameters
        ----------
        n_nodes : int or torch.SymInt
            Number of nodes (nf * nloc).
        edge_cache : EdgeCache
            Per-edge cache supplying the destination endpoint, its CSR view and
            the smooth degree normalization.
        radial_feat : torch.Tensor
            Per-edge radial features with shape (E, lmax, C) for degrees
            1 to lmax.
        zonal_coupling : torch.Tensor
            Zonal coupling with shape (E, D-1).

        Returns
        -------
        torch.Tensor
            Initial features to add with shape (N, D, C), with l=0 zero.
        """
        from deepmd.pt_expt.kernels.cuda.dpa4.zonal_scatter import (
            zonal_scatter,
        )

        # === Step 1. Destination CSR, shared with every other edge consumer ===
        order, row_ptr = cached_edge_csr(edge_cache, "dst", n_nodes)

        # === Step 2. Fused message build, reduction, padding and normalization ===
        # The operator emits the packed node layout already normalized, so the
        # scalar row and the degree scaling cost no extra pass. The scaling is
        # differentiated: the smooth degree is a sum over the cutoff envelope
        # and carries a gradient back to the geometry.
        return zonal_scatter(
            zonal_coupling.contiguous(),
            radial_feat.contiguous(),
            edge_cache.dst,
            order,
            row_ptr,
            edge_cache.inv_sqrt_deg.reshape(-1),
            n_nodes,
        )  # (N, D, C)
