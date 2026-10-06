# SPDX-License-Identifier: LGPL-3.0-or-later
"""Batch-aware selection of persistent torch neighbor-list strategies."""

from typing import (
    Any,
)

import torch

from deepmd.dpmodel.utils.exclude_mask import (
    PairExcludeMask,
)
from deepmd.dpmodel.utils.neighbor_list import (
    EdgeNeighborList,
    NeighborList,
)
from deepmd.pt.utils.nv_nlist import (
    NvNeighborList,
    is_nv_available,
)
from deepmd.pt_expt.utils.vesin_neighbor_list import (
    VesinNeighborList,
    is_vesin_torch_available,
)


class AutoNeighborList(NeighborList):
    """Prefer Vesin for one frame and Toolkit-Ops for CUDA batches."""

    def __init__(self, skin: float = 0.0) -> None:
        self._vesin = VesinNeighborList(skin=skin)
        self._nv = NvNeighborList()

    def build(
        self,
        coord: Any,
        atype: Any,
        box: Any,
        rcut: float,
        sel: list[int],
        return_mode: str = "extended",
        pair_excl: PairExcludeMask | None = None,
    ) -> tuple[Any, Any, Any, Any] | EdgeNeighborList:
        """Build the requested schema with the available inference strategy."""
        vesin_available = is_vesin_torch_available()
        if atype.shape[0] == 1 and vesin_available:
            builder = self._vesin
        elif coord.device.type == "cuda" and is_nv_available():
            builder = self._nv
        elif vesin_available:
            builder = self._vesin
        elif is_nv_available():
            builder = self._nv
        else:
            raise RuntimeError(
                "Neighbor-list construction requires vesin[torch] or "
                "nvalchemi-toolkit-ops; install deepmd-kit[torch]."
            )
        return builder.build(
            coord, atype, box, rcut, sel, return_mode=return_mode, pair_excl=pair_excl
        )


def make_neighbor_list(
    method: str, device: torch.device, skin: float = 0.0
) -> NeighborList | None:
    """Select an inference strategy, or native construction when auto has none."""
    if method == "native":
        return None
    if method == "auto":
        if is_vesin_torch_available() or (device.type == "cuda" and is_nv_available()):
            return AutoNeighborList(skin=skin)
        return None
    if method == "vesin":
        if not is_vesin_torch_available():
            raise ImportError(
                "Vesin neighbor search requires vesin[torch]>=0.6; "
                "install it or select the native neighbor builder."
            )
        return VesinNeighborList(skin=skin)
    if method == "nv":
        if device.type != "cuda":
            raise ValueError("NV neighbor search requires CUDA inference tensors")
        if not is_nv_available():
            raise ImportError("NV neighbor search requires nvalchemi-toolkit-ops")
        return NvNeighborList()
    raise ValueError(
        f"Unknown neighbor backend {method!r}; expected 'auto', 'native', 'vesin', or 'nv'."
    )
