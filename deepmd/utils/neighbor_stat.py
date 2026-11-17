# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
import math
import os
from abc import (
    ABC,
    abstractmethod,
)
from collections.abc import (
    Iterator,
)

import numpy as np

from deepmd.utils.data_system import (
    DeepmdDataSystem,
)

log = logging.getLogger(__name__)


def iter_neighbor_stat_batches(
    data: DeepmdDataSystem,
    rcut: float,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray | None, str, np.ndarray | None]]:
    """Yield aligned frame batches for NumPy and PyTorch-exportable statistics.

    ``DP_NEIGHBOR_STAT_BATCH_SIZE`` limits the number of frames handed to a
    distance calculation (default: 8), independently of training/inference
    batch sizes. Validate it before loading any sets. This bounds the large
    pair-distance intermediates, not the input: ``_load_set`` still loads a
    complete set into host memory.

    Periodic-image bounds are computed once from all cells in each set.
    Reuse these bounds in every batch, including OOM retries, so that each
    frame has exactly the same image candidates as a full-set calculation.
    """
    value = os.environ.get("DP_NEIGHBOR_STAT_BATCH_SIZE", "8")
    try:
        batch_size = int(value)
    except ValueError as exc:
        raise ValueError(
            "DP_NEIGHBOR_STAT_BATCH_SIZE must be a positive integer"
        ) from exc
    if batch_size < 1:
        raise ValueError("DP_NEIGHBOR_STAT_BATCH_SIZE must be a positive integer")
    for data_set in data.data_systems:
        for set_path in data_set.dirs:
            arrays = data_set._load_set(set_path)
            coord, atype = arrays["coord"], arrays["type"]
            cell = arrays["box"] if data_set.pbc else None
            nbuff = None
            if cell is not None:
                # Keep the dpmodel dependency local to this NumPy/pt_expt path.
                from deepmd.dpmodel.utils.region import (
                    to_face_distance,
                )

                to_face = to_face_distance(cell.reshape(-1, 3, 3))
                nbuff = np.max(np.ceil(rcut / to_face).astype(np.int64), axis=0)
            for start in range(0, coord.shape[0], batch_size):
                frames = slice(start, start + batch_size)
                yield (
                    coord[frames],
                    atype[frames],
                    cell[frames] if cell is not None else None,
                    str(set_path),
                    nbuff,
                )


class NeighborStat(ABC):
    """Abstract base class for getting training data information.

    It loads data from DeepmdData object, and measures the data info, including
    neareest nbor distance between atoms, max nbor size of atoms and the output
    data range of the environment matrix.

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
        self.rcut = rcut
        self.ntypes = ntypes
        self.mixed_type = mixed_type

    def get_stat(self, data: DeepmdDataSystem) -> tuple[float, np.ndarray]:
        """Get the data statistics of the training data, including nearest nbor distance between atoms, max nbor size of atoms.

        Parameters
        ----------
        data
            Class for manipulating many data systems. It is implemented with the help of DeepmdData.

        Returns
        -------
        min_nbor_dist
            The nearest distance between neighbor atoms
        max_nbor_size
            An array with ntypes integers, denotes the actual achieved max sel
        """
        min_nbor_dist = 100.0
        max_nbor_size = np.zeros(1 if self.mixed_type else self.ntypes, dtype=int)

        for mn, dt, jj in self.iterator(data):
            if np.isinf(dt):
                log.warning(
                    f"Atoms with no neighbors found in {jj}. Please make sure it's what you expected."
                )
            if dt < min_nbor_dist:
                if math.isclose(dt, 0.0, rel_tol=1e-6):
                    # it's unexpected that the distance between two atoms is zero
                    # zero distance will cause nan (#874)
                    raise RuntimeError(
                        f"Some atoms are overlapping in {jj}. Please check your"
                        " training data to remove duplicated atoms."
                    )
                min_nbor_dist = dt
            max_nbor_size = np.maximum(mn, max_nbor_size)

        # do sqrt in the final
        min_nbor_dist = math.sqrt(min_nbor_dist)
        log.info(
            f"Neighbor statistics: training data with minimal neighbor distance: {min_nbor_dist:f}"
        )
        log.info(
            f"Neighbor statistics: training data with maximum neighbor size: {max_nbor_size!s} (cutoff radius: {self.rcut:f})"
        )
        return min_nbor_dist, max_nbor_size

    @abstractmethod
    def iterator(
        self, data: DeepmdDataSystem
    ) -> Iterator[tuple[np.ndarray, float, str]]:
        """Abstract method for producing data.

        Yields
        ------
        mn : np.ndarray
            The maximal number of neighbors
        dt : float
            The squared minimal distance between two atoms
        jj : str
            The directory of the data system
        """
