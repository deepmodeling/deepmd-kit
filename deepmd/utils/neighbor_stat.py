import math
import logging
import numpy as np
from tqdm import tqdm
from deepmd.env import tf
from typing import Tuple, List
from deepmd.env import op_module
from deepmd.env import default_tf_session_config
from deepmd.env import GLOBAL_NP_FLOAT_PRECISION
from deepmd.utils.data_system import DeepmdDataSystem

log = logging.getLogger(__name__)

class NeighborStat():
    """
    Class for getting training data information. 
    It loads data from DeepmdData object, and measures the data info, including neareest nbor distance between atoms, max nbor size of atoms and the output data range of the environment matrix.
    """
    def __init__(self,
                 ntypes : int,
                 rcut: float) -> None:
        """
        Constructor

        Parameters
        ----------
        ntypes
                The num of atom types
        rcut
                The cut-off radius
        """
        self.rcut = rcut
        self.ntypes = ntypes
        self.place_holders = {}
        sub_graph = tf.Graph()
        with sub_graph.as_default():
            for ii in ['coord', 'box']:
                self.place_holders[ii] = tf.placeholder(GLOBAL_NP_FLOAT_PRECISION, [None, None], name='t_'+ii)
            self.place_holders['type'] = tf.placeholder(tf.int32, [None, None], name='t_type')
            self.place_holders['natoms_vec'] = tf.placeholder(tf.int32, [self.ntypes+2], name='t_natoms')
            self.place_holders['default_mesh'] = tf.placeholder(tf.int32, [None], name='t_mesh')
            self._max_nbor_size, self._min_nbor_dist \
                = op_module.neighbor_stat(self.place_holders['coord'],
                                         self.place_holders['type'],
                                         self.place_holders['natoms_vec'],
                                         self.place_holders['box'],
                                         self.place_holders['default_mesh'],
                                         rcut = self.rcut)
        self.sub_sess = tf.Session(graph = sub_graph, config=default_tf_session_config)

    def get_stat(self,
                 data : DeepmdDataSystem) -> Tuple[float, List[int]]:
        """
        get the data statistics of the training data, including nearest nbor distance between atoms, max nbor size of atoms

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
        self.max_nbor_size = [0] * self.ntypes

        # for ii in tqdm(range(len(data.system_dirs)), desc = 'DEEPMD INFO    |-> deepmd.utils.neighbor_stat\t\t\tgetting neighbor status'):
        for ii in range(len(data.system_dirs)):
            for jj in data.data_systems[ii].dirs:
                data_set = data.data_systems[ii]._load_set(jj)
                for kk in range(np.array(data_set['type']).shape[0]):
                    mn, dt \
                        = self.sub_sess.run([self._max_nbor_size, self._min_nbor_dist], 
                                            feed_dict = {
                                                self.place_holders['coord']: np.array(data_set['coord'])[kk].reshape([-1, data.natoms[ii] * 3]),
                                                self.place_holders['type']: np.array(data_set['type'])[kk].reshape([-1, data.natoms[ii]]),
                                                self.place_holders['natoms_vec']: np.array(data.natoms_vec[ii]),
                                                self.place_holders['box']: np.array(data_set['box'])[kk].reshape([-1, 9]),
                                                self.place_holders['default_mesh']: np.array(data.default_mesh[ii]),
                                            })
                    dt = np.min(dt)
                    if dt < self.min_nbor_dist:
                        self.min_nbor_dist = dt
                    for ww in range(self.ntypes):
                        var = np.max(mn[:, ww])
                        if var > self.max_nbor_size[ww]:
                            self.max_nbor_size[ww] = var

        log.info('training data with min nbor dist: ' + str(self.min_nbor_dist))
        log.info('training data with max nbor size: ' + str(self.max_nbor_size))
        return self.min_nbor_dist, self.max_nbor_size
