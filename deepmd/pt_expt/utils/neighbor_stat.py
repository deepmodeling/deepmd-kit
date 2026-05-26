# SPDX-License-Identifier: LGPL-3.0-or-later
from collections.abc import (
    Iterator,
)

import numpy as np
import torch

from deepmd.dpmodel.utils.neighbor_stat import NeighborStatOP as NeighborStatOPDP
from deepmd.pt_expt.common import (
    torch_module,
)
from deepmd.pt_expt.utils.env import (
    DEVICE,
    GLOBAL_PT_FLOAT_PRECISION,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.utils.neighbor_stat import NeighborStat as BaseNeighborStat


@torch_module
class NeighborStatOP(NeighborStatOPDP):
    pass


class NeighborStat(BaseNeighborStat):
    """Neighbor statistics using torch on DEVICE.

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
        self.op = NeighborStatOP(ntypes, rcut, mixed_type)

    def iterator(
        self, data: DeepmdDataSystem
    ) -> Iterator[tuple[np.ndarray, float, str]]:
        """Produce neighbor statistics for each data set.

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
                minrr2, max_nnei = self._execute(
                    data_set_data["coord"],
                    data_set_data["type"],
                    data_set_data["box"] if data_set.pbc else None,
                )
                yield np.max(max_nnei, axis=0), np.min(minrr2), jj

    def _execute(
        self,
        coord: np.ndarray,
        atype: np.ndarray,
        cell: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Execute the operation on DEVICE."""
        minrr2, max_nnei = self.op(
            torch.from_numpy(coord).to(device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION),
            torch.from_numpy(atype).to(device=DEVICE, dtype=torch.long),
            (
                torch.from_numpy(cell).to(
                    device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION
                )
                if cell is not None
                else None
            ),
        )
        minrr2 = minrr2.detach().cpu().numpy()
        max_nnei = max_nnei.detach().cpu().numpy()
        return minrr2, max_nnei
