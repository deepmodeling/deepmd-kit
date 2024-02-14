# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    Iterator,
    Tuple,
)

import numpy as np

from deepmd.tf.env import (
    GLOBAL_NP_FLOAT_PRECISION,
    default_tf_session_config,
    op_module,
    tf,
)
from deepmd.tf.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.tf.utils.parallel_op import (
    ParallelOp,
)
from deepmd.utils.neighbor_stat import NeighborStat as BaseNeighborStat

log = logging.getLogger(__name__)


class NeighborStat(BaseNeighborStat):
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
        super().__init__(ntypes, rcut, one_type)
        sub_graph = tf.Graph()

        def builder():
            place_holders = {}
            for ii in ["coord", "box"]:
                place_holders[ii] = tf.placeholder(
                    GLOBAL_NP_FLOAT_PRECISION, [None, None], name="t_" + ii
                )
            place_holders["type"] = tf.placeholder(
                tf.int32, [None, None], name="t_type"
            )
            place_holders["natoms_vec"] = tf.placeholder(
                tf.int32, [self.ntypes + 2], name="t_natoms"
            )
            place_holders["default_mesh"] = tf.placeholder(
                tf.int32, [None], name="t_mesh"
            )
            t_type = place_holders["type"]
            t_natoms = place_holders["natoms_vec"]
            if self.one_type:
                # all types = 0, natoms_vec = [natoms, natoms, natoms]
                t_type = tf.clip_by_value(t_type, -1, 0)
                t_natoms = tf.tile(t_natoms[0:1], [3])

            _max_nbor_size, _min_nbor_dist = op_module.neighbor_stat(
                place_holders["coord"],
                t_type,
                t_natoms,
                place_holders["box"],
                place_holders["default_mesh"],
                rcut=self.rcut,
            )
            place_holders["dir"] = tf.placeholder(tf.string)
            _min_nbor_dist = tf.reduce_min(_min_nbor_dist)
            _max_nbor_size = tf.reduce_max(_max_nbor_size, axis=0)
            return place_holders, (_max_nbor_size, _min_nbor_dist, place_holders["dir"])

        with sub_graph.as_default():
            self.p = ParallelOp(builder, config=default_tf_session_config)

        self.sub_sess = tf.Session(graph=sub_graph, config=default_tf_session_config)

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

        def feed():
            for ii in range(len(data.system_dirs)):
                for jj in data.data_systems[ii].dirs:
                    data_set = data.data_systems[ii]._load_set(jj)
                    for kk in range(np.array(data_set["type"]).shape[0]):
                        yield {
                            "coord": np.array(data_set["coord"])[kk].reshape(
                                [-1, data.natoms[ii] * 3]
                            ),
                            "type": np.array(data_set["type"])[kk].reshape(
                                [-1, data.natoms[ii]]
                            ),
                            "natoms_vec": np.array(data.natoms_vec[ii]),
                            "box": np.array(data_set["box"])[kk].reshape([-1, 9]),
                            "default_mesh": np.array(data.default_mesh[ii]),
                            "dir": str(jj),
                        }

        return self.p.generate(self.sub_sess, feed())
