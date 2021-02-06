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
                 ndescrpt : int,
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
        ndescrpt
                The width of environment matrix
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
        self.init_stat = False
        self.davg = davg
        self.dstd = dstd
        if self.davg is None:
            self.davg = np.zeros([self.ntypes, self.ndescrpt])
        if self.dstd is None:
            self.dstd = np.ones ([self.ntypes, self.ndescrpt])
        self.ntypes = ntypes
        self.ndescrpt = ndescrpt
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
            self._min_nbor_dist, self._max_nbor_size \
                = op_module.env_mat_stat(self.place_holders['coord'],
                                         self.place_holders['type'],
                                         self.place_holders['natoms_vec'],
                                         self.place_holders['box'],
                                         self.place_holders['default_mesh'],
                                         sel = self.sel,
                                         rcut = self.rcut,
                                         rcut_smth = self.rcut_smth)
        self.sub_sess = tf.Session(graph = sub_graph, config=default_tf_session_config)

    def get_env_mat_stat(self,
                         data) -> Tuple[float, int]:
        """
        get the data statistics of the training data, including nearest nbor distance between atoms, max nbor size of atoms

        Parameters
        ----------
        data
                Class for manipulating many data systems. It is implemented with the help of DeepmdData.
        
        Returns
        -------
        min_nbor_dist
                The nearest nbor distance between atoms
        max_nbor_size
                The max nbor size of atoms
        """
        self.max_nbor_size = 0
        self.min_nbor_dist = 100.0

        for ii in tqdm(range(len(data.system_dirs)), desc = '# DEEPMD: getting data info'):
            for jj in data.data_systems[ii].dirs:
                data_set = data.data_systems[ii]._load_set(jj)
                for kk in range(np.array(data_set['type']).shape[0]):
                    dt, mn \
                        = self.sub_sess.run([self._min_nbor_dist, self._max_nbor_size], 
                                            feed_dict = {
                                                self.place_holders['coord']: np.array(data_set['coord'])[kk].reshape([-1, data.natoms[ii] * 3]),
                                                self.place_holders['type']: np.array(data_set['type'])[kk].reshape([-1, data.natoms[ii]]),
                                                self.place_holders['natoms_vec']: np.array(data.natoms_vec[ii]),
                                                self.place_holders['box']: np.array(data_set['box'])[kk].reshape([-1, 9]),
                                                self.place_holders['default_mesh']: np.array(data.default_mesh[ii]),
                                            })
                    dt = np.min(dt)
                    mn = np.max(mn)
                    if (dt < self.min_nbor_dist):
                        self.min_nbor_dist = dt
                    if (mn > self.max_nbor_size):
                        self.max_nbor_size = mn
        self.init_stat = True
        return self.min_nbor_dist, self.max_nbor_size

    def get_env_mat_range(self,
                     data) -> List[float]:
        """
        get the data statistics of the training data, including the output data range of the environment matrix

        Parameters
        ----------
        data
                Class for manipulating many data systems. It is implemented with the help of DeepmdData.
        
        Returns
        -------
        env_mat_range
                The output data range of the environment matrix
                env_mat_range[0] denotes the lower boundary of environment matrix
                env_mat_range[1] denotes the upper boundary of environment matrix
        """
        if self.init_stat:
            min_nbor_dist = self.min_nbor_dist
            max_nbor_size = self.max_nbor_size
        else:
            min_nbor_dist, max_nbor_size = self.get_env_mat_stat(data)
        self.env_mat_range = self._get_internal_env_mat_range(min_nbor_dist, max_nbor_size)
        print('# DEEPMD: training data with lower boundary: ' + str(self.env_mat_range[0]))
        print('# DEEPMD: training data with upper boundary: ' + str(self.env_mat_range[1]))
        print('# DEEPMD: training data with min   distance: ' + str(self.min_nbor_dist))
        print('# DEEPMD: training data with max   nborsize: ' + str(self.max_nbor_size))
        return self.env_mat_range

    def _get_internal_env_mat_range(self,
                                    min_nbor_dist, 
                                    max_nbor_size):
        """
        Warning: different descrpt_type may have different method to get the mat range
        """
        lower = 100.0
        upper = -10.0
        sw    = self._spline5_switch(self.min_nbor_dist, self.rcut_smth, self.rcut)
        for ii in range(self.ntypes):
            if lower > -self.davg[ii][0] / self.dstd[ii][0]:
                lower = -self.davg[ii][0] / self.dstd[ii][0]
            if upper < ((1 / self.min_nbor_dist) * sw - self.davg[ii][0]) / self.dstd[ii][0]:
                upper = ((1 / self.min_nbor_dist) * sw - self.davg[ii][0]) / self.dstd[ii][0]
        return [lower, upper]

    def _spline5_switch(self,
                        xx,
                        rmin,
                        rmax):
        if xx < rmin:
            vv = 1
        elif xx < rmax:
            uu = (xx - rmin) / (rmax - rmin)
            vv = uu*uu*uu * (-6 * uu*uu + 15 * uu - 10) + 1
        else:
            vv = 0
        return vv
