# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
import math
from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Iterator,
    Tuple,
)

import numpy as np

from deepmd.utils.data_system import (
    DeepmdDataSystem,
)

log = logging.getLogger(__name__)


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

    def get_stat(self, data: DeepmdDataSystem) -> Tuple[float, np.ndarray]:
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
                    "Atoms with no neighbors found in %s. Please make sure it's what you expected."
                    % jj
                )
            if dt < min_nbor_dist:
                if math.isclose(dt, 0.0, rel_tol=1e-6):
                    # it's unexpected that the distance between two atoms is zero
                    # zero distance will cause nan (#874)
                    raise RuntimeError(
                        "Some atoms are overlapping in %s. Please check your"
                        " training data to remove duplicated atoms." % jj
                    )
                min_nbor_dist = dt
            max_nbor_size = np.maximum(mn, max_nbor_size)

        # do sqrt in the final
        min_nbor_dist = math.sqrt(min_nbor_dist)
        log.info("training data with min nbor dist: " + str(min_nbor_dist))
        log.info("training data with max nbor size: " + str(max_nbor_size))
        return min_nbor_dist, max_nbor_size

    @abstractmethod
    def iterator(
        self, data: DeepmdDataSystem
    ) -> Iterator[Tuple[np.ndarray, float, str]]:
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
