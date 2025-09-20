# SPDX-License-Identifier: LGPL-3.0-or-later
from collections.abc import (
    Iterator,
)
from typing import (
    Optional,
)

import numpy as np

from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.utils.neighbor_stat import (
    NeighborStatOP,
)
from deepmd.jax.common import (
    to_jax_array,
)
from deepmd.jax.utils.auto_batch_size import (
    AutoBatchSize,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.utils.neighbor_stat import NeighborStat as BaseNeighborStat


class NeighborStat(BaseNeighborStat):
    """Neighbor statistics using JAX.

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
        self.auto_batch_size = AutoBatchSize()

    def iterator(
        self, data: DeepmdDataSystem
    ) -> Iterator[tuple[np.ndarray, float, str]]:
        """Iterator method for producing neighbor statistics data.

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
            to_jax_array(coord),
            to_jax_array(atype),
            to_jax_array(cell),
        )
        minrr2 = to_numpy_array(minrr2)
        max_nnei = to_numpy_array(max_nnei)
        return minrr2, max_nnei
