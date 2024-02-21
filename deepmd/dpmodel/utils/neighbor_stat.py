# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Iterator,
    Optional,
    Tuple,
)

import numpy as np

from deepmd.dpmodel.common import (
    NativeOP,
)
from deepmd.dpmodel.utils.nlist import (
    extend_coord_with_ghosts,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.utils.neighbor_stat import NeighborStat as BaseNeighborStat


class NeighborStatOP(NativeOP):
    """Class for getting neighbor statics data information.

    Parameters
    ----------
    ntypes
        The num of atom types
    rcut
        The cut-off radius
    mixed_types : bool, optional
        If True, treat all types as a single type.
    """

    def __init__(
        self,
        ntypes: int,
        rcut: float,
        mixed_types: bool,
    ) -> None:
        self.rcut = rcut
        self.ntypes = ntypes
        self.mixed_types = mixed_types

    def call(
        self,
        coord: np.ndarray,
        atype: np.ndarray,
        cell: Optional[np.ndarray],
    ) -> Tuple[float, np.ndarray]:
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
        float
            The minimal squared distance between two atoms
        np.ndarray
            The maximal number of neighbors
        """
        nframes = coord.shape[0]
        coord = coord.reshape(nframes, -1, 3)
        nloc = coord.shape[1]
        coord = coord.reshape(nframes, nloc * 3)
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
        mask = np.eye(nloc, nall, dtype=bool)
        diff[:, mask] = np.inf
        rr2 = np.sum(np.square(diff), axis=-1)
        min_rr2 = np.min(rr2, axis=-1)
        # count the number of neighbors
        if not self.mixed_types:
            mask = rr2 < self.rcut**2
            nnei = np.zeros((nframes, nloc, self.ntypes), dtype=int)
            for ii in range(self.ntypes):
                nnei[:, :, ii] = np.sum(
                    mask & (extend_atype == ii)[:, None, :], axis=-1
                )
        else:
            mask = rr2 < self.rcut**2
            # virtual type (<0) are not counted
            nnei = np.sum(mask & (extend_atype >= 0)[:, None, :], axis=-1).reshape(
                nframes, nloc, 1
            )
        max_nnei = np.max(nnei, axis=1)
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
        self.op = NeighborStatOP(ntypes, rcut, mixed_type)

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
                minrr2, max_nnei = self.op(
                    data_set_data["coord"],
                    data_set_data["type"],
                    data_set_data["box"] if data_set.pbc else None,
                )
                yield np.max(max_nnei, axis=0), np.min(minrr2), jj
