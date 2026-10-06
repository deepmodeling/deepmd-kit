# SPDX-License-Identifier: LGPL-3.0-or-later
"""Persistent geometric neighbor searches shared by inference input schemas."""

from __future__ import (
    annotations,
)

import math
from typing import (
    TYPE_CHECKING,
)

import torch

if TYPE_CHECKING:
    from vesin_torch import (
        NeighborList,
    )


class VesinNeighborSearch:
    """Reuse Vesin search storage and optional Verlet candidates.

    Each frame owns a calculator. Vesin refreshes displacements and filters
    candidates at the requested cutoff on every call, and rebuilds candidates
    when motion exceeds half the skin or the cell or point count changes.
    Search results are copied so later calls cannot overwrite indices retained
    by an earlier autograd graph.

    Parameters
    ----------
    skin : float, default: 0.0
        Extra candidate-search radius in angstrom. Zero disables Verlet caching.
    sorted : bool, default: False
        Whether pairs are returned in ascending center-index order.
    """

    def __init__(self, skin: float = 0.0, *, sorted: bool = False) -> None:
        if not math.isfinite(skin) or skin < 0:
            raise ValueError(
                f"neighbor_skin must be finite and non-negative, got {skin}"
            )
        self.skin = float(skin)
        self.sorted = sorted
        self._key: tuple[float, torch.device] | None = None
        self._calculators: list[NeighborList] = []

    def compute(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor | None,
        rcut: float,
        *,
        frame: int = 0,
        return_vectors: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return neighbor indices and image shifts or displacement vectors.

        Parameters
        ----------
        positions : torch.Tensor
            Coordinates in angstrom with shape (N, 3). The search detaches
            these coordinates and runs on their device.
        cell : torch.Tensor or None
            Cell vectors in angstrom with shape (3, 3), with the same dtype
            and device as ``positions``. None selects non-periodic boundaries.
        rcut : float
            Neighbor cutoff radius in angstrom.
        frame : int, optional
            Non-negative cache slot, by default 0. Each frame in a batch uses
            a distinct slot, reused for its successive configurations.
        return_vectors : bool, optional
            Return neighbor-minus-center vectors in angstrom instead of
            lattice shifts. These vectors are detached from the coordinates
            and are intended for inference with explicit force outputs.

        Returns
        -------
        ii : torch.Tensor
            Local center indices with shape (E,), int64.
        jj : torch.Tensor
            Local neighbor indices with shape (E,), int64.
        geometry : torch.Tensor
            Integer lattice shifts with shape (E, 3), int64, or displacement
            vectors with the dtype of ``positions`` when ``return_vectors``
            is True. All outputs reside on the device of ``positions`` and
            retain their values across subsequent searches.
        """
        from vesin_torch import (
            NeighborList,
        )

        device = positions.device
        key = (float(rcut), device)
        if self._key != key:
            self._calculators.clear()
            self._key = key
        while len(self._calculators) <= frame:
            self._calculators.append(
                NeighborList(
                    cutoff=float(rcut),
                    full_list=True,
                    sorted=self.sorted,
                    skin=self.skin,
                )
            )
        box = (
            cell
            if cell is not None
            else torch.zeros((3, 3), dtype=positions.dtype, device=device)
        )
        with torch.device(device):
            try:
                ii, jj, geometry = self._calculators[frame].compute(
                    points=positions.detach(),
                    box=box.detach(),
                    periodic=cell is not None,
                    quantities="ijD" if return_vectors else "ijS",
                )
            except RuntimeError as error:
                # An empty CUDA Verlet cache can expose a null pair buffer that
                # PyTorch cannot wrap. An uncached search supplies valid empty
                # tensors; nonempty results and unrelated errors still fail.
                if (
                    device.type != "cuda"
                    or self.skin == 0
                    or str(error)
                    != "The specified pointer resides on host memory and is not "
                    "registered with any CUDA device."
                ):
                    raise
                ii, jj, geometry = NeighborList(
                    cutoff=float(rcut), full_list=True
                ).compute(
                    points=positions.detach(),
                    box=box.detach(),
                    periodic=cell is not None,
                    quantities="ijD" if return_vectors else "ijS",
                )
                if ii.numel() != 0:
                    raise
        if not return_vectors:
            geometry = geometry.to(torch.int64)
        return ii.to(torch.int64), jj.to(torch.int64), geometry.reshape(-1, 3)
