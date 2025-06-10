# SPDX-License-Identifier: LGPL-3.0-or-later
from collections.abc import (
    Iterator,
)
from typing import (
    Optional,
)

import numpy as np
import paddle

from deepmd.pd.utils.auto_batch_size import (
    AutoBatchSize,
)
from deepmd.pd.utils.env import (
    DEVICE,
)
from deepmd.pd.utils.nlist import (
    extend_coord_with_ghosts,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.utils.neighbor_stat import NeighborStat as BaseNeighborStat


class NeighborStatOP(paddle.nn.Layer):
    """Class for getting neighbor statistics data information.

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
        self.rcut = float(rcut)
        self.ntypes = ntypes
        self.mixed_types = mixed_types

    def forward(
        self,
        coord: paddle.Tensor,
        atype: paddle.Tensor,
        cell: Optional[paddle.Tensor],
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
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
        paddle.Tensor
            The minimal squared distance between two atoms, in the shape of (nframes,)
        paddle.Tensor
            The maximal number of neighbors
        """
        nframes = coord.shape[0]
        coord = coord.reshape([nframes, -1, 3])
        nloc = coord.shape[1]
        coord = coord.reshape([nframes, nloc * 3])
        extend_coord, extend_atype, _ = extend_coord_with_ghosts(
            coord, atype, cell, self.rcut
        )

        coord1 = extend_coord.reshape([nframes, -1])
        nall = coord1.shape[1] // 3
        coord0 = coord1[:, : nloc * 3]
        diff: paddle.Tensor = coord1.reshape([nframes, -1, 3]).unsqueeze(
            1
        ) - coord0.reshape([nframes, -1, 3]).unsqueeze(2)
        assert list(diff.shape) == [nframes, nloc, nall, 3]
        # remove the diagonal elements
        mask = paddle.eye(nloc, nall).to(dtype=paddle.bool, device=diff.place)
        # diff[:, mask] = float("inf")
        # diff.masked_fill_(
        #     paddle.broadcast_to(mask.unsqueeze([0, -1]), diff.shape),
        #     paddle.to_tensor(float("inf")),
        # )
        diff[paddle.broadcast_to(mask.unsqueeze([0, -1]), diff.shape)] = float("inf")
        rr2 = paddle.sum(paddle.square(diff), axis=-1)
        min_rr2 = paddle.min(rr2, axis=-1)
        # count the number of neighbors
        if not self.mixed_types:
            mask = rr2 < self.rcut**2
            nnei = paddle.zeros([nframes, nloc, self.ntypes], dtype=paddle.int64)
            for ii in range(self.ntypes):
                nnei[:, :, ii] = paddle.sum(
                    mask & ((extend_atype == ii)[:, None, :]), axis=-1
                )
        else:
            mask = rr2 < self.rcut**2
            # virtual types (<0) are not counted
            nnei = paddle.sum(
                mask & ((extend_atype >= 0).unsqueeze(1)), axis=-1
            ).reshape([nframes, nloc, 1])
        max_nnei = paddle.max(nnei, axis=1)
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
        # self.op = paddle.jit.to_static(op)
        self.op = op
        self.auto_batch_size = AutoBatchSize()

    def iterator(
        self, data: DeepmdDataSystem
    ) -> Iterator[tuple[np.ndarray, float, str]]:
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
        with paddle.no_grad():
            minrr2, max_nnei = self.op(
                paddle.to_tensor(coord, place=DEVICE),
                paddle.to_tensor(atype, place=DEVICE),
                paddle.to_tensor(cell, place=DEVICE) if cell is not None else None,
            )
        minrr2 = minrr2.numpy()
        max_nnei = max_nnei.numpy()
        return minrr2, max_nnei
