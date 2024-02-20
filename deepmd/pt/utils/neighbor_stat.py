# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Iterator,
    Optional,
    Tuple,
)

import numpy as np
import torch

from deepmd.pt.utils.auto_batch_size import (
    AutoBatchSize,
)
from deepmd.pt.utils.env import (
    DEVICE,
)
from deepmd.pt.utils.nlist import (
    extend_coord_with_ghosts,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.utils.neighbor_stat import NeighborStat as BaseNeighborStat


class NeighborStatOP(torch.nn.Module):
    """Class for getting neighbor statics data information.

    Parameters
    ----------
    ntypes
        The num of atom types
    rcut
        The cut-off radius
    mixed_types : bool, optional
        If True, treat neighbors of all types as a single type.
    """

    def __init__(
        self,
        ntypes: int,
        rcut: float,
        mixed_types: bool,
    ) -> None:
        super().__init__()
        self.rcut = rcut
        self.ntypes = ntypes
        self.mixed_types = mixed_types

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        cell: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the neareest neighbor distance between atoms, maximum nbor size of
        atoms and the output data range of the environment matrix.

        Parameters
        ----------
        coord
            The coordinates of atoms.
        atype
            The atom types.
        cell
            The cell.

        Returns
        -------
        torch.Tensor
            The minimal squared distance between two atoms, in the shape of (nframes,)
        torch.Tensor
            The maximal number of neighbors
        """
        nframes = coord.shape[0]
        coord = coord.view(nframes, -1, 3)
        nloc = coord.shape[1]
        coord = coord.view(nframes, nloc * 3)
        extend_coord, extend_atype, _ = extend_coord_with_ghosts(
            coord, atype, cell, self.rcut
        )

        coord1 = extend_coord.reshape(nframes, -1)
        nall = coord1.shape[1] // 3
        coord0 = coord1[:, : nloc * 3]
        diff = (
            coord1.reshape([nframes, -1, 3])[:, None, :, :]
            - coord0.reshape([nframes, -1, 3])[:, :, None, :]
        )
        assert list(diff.shape) == [nframes, nloc, nall, 3]
        # remove the diagonal elements
        mask = torch.eye(nloc, nall, dtype=torch.bool, device=diff.device)
        diff[:, mask] = torch.inf
        rr2 = torch.sum(torch.square(diff), dim=-1)
        min_rr2, _ = torch.min(rr2, dim=-1)
        # count the number of neighbors
        if not self.mixed_types:
            mask = rr2 < self.rcut**2
            nnei = torch.zeros(
                (nframes, nloc, self.ntypes), dtype=torch.int32, device=mask.device
            )
            for ii in range(self.ntypes):
                nnei[:, :, ii] = torch.sum(
                    mask & extend_atype.eq(ii)[:, None, :], dim=-1
                )
        else:
            mask = rr2 < self.rcut**2
            # virtual types (<0) are not counted
            nnei = torch.sum(mask & extend_atype.ge(0)[:, None, :], dim=-1).view(
                nframes, nloc, 1
            )
        max_nnei, _ = torch.max(nnei, dim=1)
        return min_rr2, max_nnei


class NeighborStat(BaseNeighborStat):
    """Neighbor statistics using pure NumPy.

    Parameters
    ----------
    ntypes : int
        The num of atom types
    rcut : float
        The cut-off radius
    mixed_type : bool, optional, default=False
        Treat all types as a single type.
    """

    def __init__(
        self,
        ntypes: int,
        rcut: float,
        mixed_type: bool = False,
    ) -> None:
        super().__init__(ntypes, rcut, mixed_type)
        op = NeighborStatOP(ntypes, rcut, mixed_type)
        self.op = torch.jit.script(op)
        self.auto_batch_size = AutoBatchSize()

    def iterator(
        self, data: DeepmdDataSystem
    ) -> Iterator[Tuple[np.ndarray, float, str]]:
        """Abstract method for producing data.

        Yields
        ------
        np.ndarray
            The maximal number of neighbors
        float
            The squared minimal distance between two atoms
        str
            The directory of the data system
        """
        for ii in range(len(data.system_dirs)):
            for jj in data.data_systems[ii].dirs:
                data_set = data.data_systems[ii]
                data_set_data = data_set._load_set(jj)
                minrr2, max_nnei = self.auto_batch_size.execute_all(
                    self._execute,
                    data_set_data["coord"].shape[0],
                    data_set.get_natoms(),
                    data_set_data["coord"],
                    data_set_data["type"],
                    data_set_data["box"] if data_set.pbc else None,
                )
                yield np.max(max_nnei, axis=0), np.min(minrr2), jj

    def _execute(
        self,
        coord: np.ndarray,
        atype: np.ndarray,
        cell: Optional[np.ndarray],
    ):
        """Execute the operation.

        Parameters
        ----------
        coord
            The coordinates of atoms.
        atype
            The atom types.
        cell
            The cell.
        """
        minrr2, max_nnei = self.op(
            torch.from_numpy(coord).to(DEVICE),
            torch.from_numpy(atype).to(DEVICE),
            torch.from_numpy(cell).to(DEVICE) if cell is not None else None,
        )
        minrr2 = minrr2.detach().cpu().numpy()
        max_nnei = max_nnei.detach().cpu().numpy()
        return minrr2, max_nnei
