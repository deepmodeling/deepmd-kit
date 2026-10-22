# SPDX-License-Identifier: LGPL-3.0-or-later
"""Batch-aware selection of persistent torch neighbor-list strategies."""

from typing import (
    Any,
    Literal,
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


class AdaptiveNeighborList(NeighborList):
    """Select a cell-list backend from the device and batch shape.

    ``auto`` favors persistent Vesin searches for single frames and CPU
    inference, and Toolkit-Ops for CUDA batches. ``native`` retains the
    unbuffered batched policy: Vesin for one CPU frame, Toolkit-Ops otherwise,
    with either backend serving as the fallback when the other is unavailable.
    The selection is independent of the requested neighbor representation.
    """

    def __init__(
        self, skin: float = 0.0, *, method: Literal["auto", "native"] = "auto"
    ) -> None:
        self._method = method
        self._vesin = VesinNeighborList(skin=0.0 if method == "native" else skin)
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
        single_vesin = atype.shape[0] == 1 and (
            self._method == "auto" or coord.device.type == "cpu"
        )
        batched_nv = self._method == "native" or coord.device.type == "cuda"
        if single_vesin and vesin_available:
            builder = self._vesin
        elif batched_nv and is_nv_available():
            builder = self._nv
        elif vesin_available:
            builder = self._vesin
        elif is_nv_available():
            builder = self._nv
        else:
            raise RuntimeError(
                "Neighbor-list construction requires a neighbor-list backend. "
                "Install one of:\n"
                "  pip install 'vesin[torch]'\n"
                "  pip install nvalchemi-toolkit-ops\n"
                "Vesin also ships with the PyTorch extra: "
                "pip install 'deepmd-kit[torch]'."
            )
        return builder.build(
            coord, atype, box, rcut, sel, return_mode=return_mode, pair_excl=pair_excl
        )


def make_neighbor_list(
    method: str,
    device: torch.device,
    skin: float = 0.0,
    *,
    return_mode: Literal["extended", "edges"] = "extended",
) -> NeighborList | None:
    """Select a search strategy compatible with the lower-input contract.

    ``None`` delegates extended-coordinate construction to the caller's native
    builder. Edge inputs require an explicit carry-all builder: falling back to
    a fixed-capacity neighbor list would discard neighbors before conversion.
    """
    if method == "native":
        if return_mode == "edges":
            return AdaptiveNeighborList(method="native")
        return None
    if method == "auto":
        if (
            return_mode == "edges"
            or is_vesin_torch_available()
            or (device.type == "cuda" and is_nv_available())
        ):
            return AdaptiveNeighborList(skin=skin)
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
