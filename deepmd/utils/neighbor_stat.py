import logging
import math
from typing import (
    List,
    Tuple,
)

import numpy as np
import paddle

from deepmd.env import (
    op_module,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)

log = logging.getLogger(__name__)


class NeighborStat:
    """Class for getting training data information.

    It loads data from DeepmdData object, and measures the data info, including neareest nbor distance between atoms, max nbor size of atoms and the output data range of the environment matrix.

    Parameters
    ----------
    ntypes
            The num of atom types
    rcut
            The cut-off radius
    one_type : bool, optional, default=False
        Treat all types as a single type.
    """

    def __init__(
        self,
        ntypes: int,
        rcut: float,
        one_type: bool = False,
    ) -> None:
        """Constructor."""
        self.rcut = rcut
        self.ntypes = ntypes
        self.one_type = one_type

    def get_stat(self, data: DeepmdDataSystem) -> Tuple[float, List[int]]:
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
            A list with ntypes integers, denotes the actual achieved max sel
        """
        self.min_nbor_dist = 100.0
        self.max_nbor_size = [0]
        if not self.one_type:
            self.max_nbor_size *= self.ntypes

        for ii in range(len(data.system_dirs)):
            for jj in data.data_systems[ii].dirs:
                data_set = data.data_systems[ii]._load_set(jj)
                for kk in range(np.array(data_set["type"]).shape[0]):
                    coord = np.array(data_set["coord"])[kk].reshape(
                        [-1, data.natoms[ii] * 3]
                    )
                    coord = paddle.to_tensor(coord, dtype="float32", place="cpu")

                    _type = np.array(data_set["type"])[kk].reshape(
                        [-1, data.natoms[ii]]
                    )
                    _type = paddle.to_tensor(_type, dtype="int32", place="cpu")

                    natoms_vec = np.array(data.natoms_vec[ii])
                    natoms_vec = paddle.to_tensor(
                        natoms_vec, dtype="int64", place="cpu"
                    )

                    box = np.array(data_set["box"])[kk].reshape([-1, 9])
                    box = paddle.to_tensor(box, dtype="float32", place="cpu")

                    default_mesh = np.array(data.default_mesh[ii])
                    default_mesh = paddle.to_tensor(
                        default_mesh, dtype="int32", place="cpu"
                    )

                    rcut = self.rcut
                    mn, dt = op_module.neighbor_stat(
                        coord,
                        _type,
                        natoms_vec,
                        box,
                        default_mesh,
                        rcut,
                    )
                    if dt.size != 0:
                        dt = paddle.min(dt).item()
                    else:
                        dt = self.rcut
                        log.warning(
                            "Atoms with no neighbors found in %s. Please make sure it's what you expected."
                            % jj
                        )
                    if dt < self.min_nbor_dist:
                        if math.isclose(dt, 0.0, rel_tol=1e-6):
                            # it's unexpected that the distance between two atoms is zero
                            # zero distance will cause nan (#874)
                            raise RuntimeError(
                                "Some atoms are overlapping in %s. Please check your"
                                " training data to remove duplicated atoms." % jj
                            )
                        self.min_nbor_dist = dt
                    var = paddle.max(mn, axis=0).numpy()
                    self.max_nbor_size = np.maximum(var, self.max_nbor_size)

        log.info("training data with min nbor dist: " + str(self.min_nbor_dist))
        log.info("training data with max nbor size: " + str(self.max_nbor_size))
        return self.min_nbor_dist, self.max_nbor_size
