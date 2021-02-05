import math
import numpy as np
from tqdm import tqdm
from deepmd.env import tf
from typing import Tuple, List
from deepmd.env import op_module
from deepmd.env import default_tf_session_config
from deepmd.RunOptions import global_np_float_precision

class EnvMatStat():
    """
    Class for getting training data information. 
    It loads data from DeepmdData object, and measures the data info, including neareest nbor distance between atoms, max nbor size of atoms and the output data range of the environment matrix.
    """
    def __init__(self,
                 descrpt_type : str,
                 ntypes : int,
                 rcut,
                 rcut_smth,
                 sel,
                 davg,
                 dstd) -> None:
        """
        Constructor

        Parameters
        ----------
        descrpt_type
                The descrpt type of the embedding net
        ntypes
                The num of atom types
        rcut
                The cut-off radius
        rcut_smth
                From where the environment matrix should be smoothed
        sel : list[str]
                sel[i] specifies the maxmum number of type i atoms in the cut-off radius
        davg
                Average of training data
        dstd
                Standard deviation of training data
        """
        self.davg = davg
        self.dstd = dstd
        self.ntypes = ntypes
        self.descrpt_type = descrpt_type
        assert self.descrpt_type == 'se_a', 'Model compression error: descriptor type must be se_a!'
        self.place_holders = {}
        sub_graph = tf.Graph()
        with sub_graph.as_default():
            for ii in ['coord', 'box', 'avg', 'std']:
                self.place_holders[ii] = tf.placeholder(global_np_float_precision, [None, None], name='t_'+ii)
            self.place_holders['type'] = tf.placeholder(tf.int32, [None, None], name='t_type')
            self.place_holders['natoms_vec'] = tf.placeholder(tf.int32, [self.ntypes+2], name='t_natoms')
            self.place_holders['default_mesh'] = tf.placeholder(tf.int32, [None], name='t_mesh')
            self.sel = sel
            self.rcut = rcut
            self.rcut_smth = rcut_smth
            self.distance, self.max_nbor_size, self.table_range \
                = op_module.env_mat_stat(self.place_holders['coord'],
                                         self.place_holders['type'],
                                         self.place_holders['natoms_vec'],
                                         self.place_holders['box'],
                                         self.place_holders['default_mesh'],
                                         self.place_holders['avg'],
                                         self.place_holders['std'],
                                         sel = self.sel,
                                         rcut = self.rcut,
                                         rcut_smth = self.rcut_smth)
        self.sub_sess = tf.Session(graph = sub_graph, config=default_tf_session_config)

    def env_mat_stat(self,
                     data) -> Tuple[float, int, List[float]]:
        """
        get the data info of the training data, including neareest nbor distance between atoms, max nbor size of atoms and the output data range of the environment matrix

        Parameters
        ----------
        data
                Class for manipulating many data systems. It is implemented with the help of DeepmdData.
        
        Returns
        -------
        distance
                The nearest nbor distance between atoms
        max_nbor_size
                The max nbor size of atoms
        env_mat_range
                The output data range of the environment matrix
        """
        self.lower = 0.0
        self.upper = 0.0
        self.dist  = 100.0
        self.max_nbor = 0

        davg = self.davg
        dstd = self.dstd
        if davg is None:
            davg = np.zeros([self.ntypes, self.ndescrpt])
        if dstd is None:
            dstd = np.ones ([self.ntypes, self.ndescrpt])

        for ii in tqdm(range(len(data.system_dirs)), desc = '# DEEPMD: getting data info'):
            for jj in data.data_systems[ii].dirs:
                data_set = data.data_systems[ii]._load_set(jj)
                for kk in range(np.array(data_set['type']).shape[0]):
                    dt, mn, tr \
                        = self.sub_sess.run([self.distance, self.max_nbor_size, self.table_range], 
                                            feed_dict = {
                                                self.place_holders['coord']: np.array(data_set['coord'])[kk].reshape([-1, data.natoms[ii] * 3]),
                                                self.place_holders['type']: np.array(data_set['type'])[kk].reshape([-1, data.natoms[ii]]),
                                                self.place_holders['natoms_vec']: np.array(data.natoms_vec[ii]),
                                                self.place_holders['box']: np.array(data_set['box'])[kk].reshape([-1, 9]),
                                                self.place_holders['default_mesh']: np.array(data.default_mesh[ii]),
                                                self.place_holders['avg']: davg,
                                                self.place_holders['std']: dstd,
                                            })
                    dr = np.array([np.min(tr), np.max(tr)]).astype(global_np_float_precision)
                    dt = np.min(dt)
                    mn = np.max(mn)
                    if (dr[0] < self.lower): 
                        self.lower = dr[0]
                    if (dr[1] > self.upper):
                        self.upper = dr[1]
                    if (dt < self.dist):
                        self.dist = dt
                    if (mn > self.max_nbor):
                        self.max_nbor = mn

        print('# DEEPMD: training data with lower boundary: ' + str(self.lower))
        print('# DEEPMD: training data with upper boundary: ' + str(self.upper))
        print('# DEEPMD: training data with min   distance: ' + str(self.dist))
        print('# DEEPMD: training data with max   nborsize: ' + str(self.max_nbor))
        env_mat_range = [self.lower, self.upper]
        return self.distance, self.max_nbor_size, env_mat_range
        