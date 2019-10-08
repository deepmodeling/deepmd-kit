import os,sys
import numpy as np
import unittest

from deepmd.env import tf
from tensorflow.python.framework import ops

# load grad of force module
import deepmd._prod_force_grad
import deepmd._prod_virial_grad
import deepmd._prod_force_se_r_grad
import deepmd._prod_virial_se_r_grad
import deepmd._soft_min_force_grad
import deepmd._soft_min_virial_grad

from common import force_test
from common import virial_test
from common import force_dw_test
from common import virial_dw_test
from common import Data

from deepmd.DescrptSeAR import DescrptSeAR

from deepmd.RunOptions import global_tf_float_precision
from deepmd.RunOptions import global_np_float_precision
from deepmd.RunOptions import global_ener_float_precision

class Inter():
    def __init__ (self, 
                  data) :
        self.sess = tf.Session()
        self.data = data
        self.natoms = self.data.get_natoms()
        self.ntypes = self.data.get_ntypes()
        param_a = {
            'sel' : [12,24],
            'rcut': 4,
            'rcut_smth' : 3.5,
            'neuron': [5, 10, 20],
            'seed': 1,
        }
        param_r = {
            'sel' : [20,40],
            'rcut': 6,
            'rcut_smth' : 6.5,
            'neuron': [10, 20, 40],
            'seed': 1,
        }
        param = {'a': param_a, 'r': param_r}
        self.descrpt = DescrptSeAR(param)
        self.ndescrpt = self.descrpt.get_dim_out()
        # davg = np.zeros ([self.ntypes, self.ndescrpt])
        # dstd = np.ones  ([self.ntypes, self.ndescrpt])
        # self.t_avg = tf.constant(davg.astype(np.float64))
        # self.t_std = tf.constant(dstd.astype(np.float64))
        avg_a = np.zeros([self.ntypes, self.descrpt.descrpt_a.ndescrpt])
        std_a = np.ones ([self.ntypes, self.descrpt.descrpt_a.ndescrpt])
        avg_r = np.zeros([self.ntypes, self.descrpt.descrpt_r.ndescrpt])
        std_r = np.ones ([self.ntypes, self.descrpt.descrpt_r.ndescrpt])
        self.avg = [avg_a, avg_r]
        self.std = [std_a, std_r]
        self.default_mesh = np.zeros (6, dtype = np.int32)
        self.default_mesh[3] = 2
        self.default_mesh[4] = 2
        self.default_mesh[5] = 2
        # make place holder
        self.coord      = tf.placeholder(global_tf_float_precision, [None, self.natoms[0] * 3], name='t_coord')
        self.box        = tf.placeholder(global_tf_float_precision, [None, 9], name='t_box')
        self.type       = tf.placeholder(tf.int32,   [None, self.natoms[0]], name = "t_type")
        self.tnatoms    = tf.placeholder(tf.int32,   [None], name = "t_natoms")
        
    def _net (self,
              inputs, 
              name,
              reuse = False) :
        with tf.variable_scope(name, reuse=reuse):
            net_w = tf.get_variable ('net_w', 
                                     [self.descrpt.get_dim_out()], 
                                     global_tf_float_precision,
                                     tf.constant_initializer (self.net_w_i))
        dot_v = tf.matmul (tf.reshape (inputs, [-1, self.descrpt.get_dim_out()]),
                           tf.reshape (net_w, [self.descrpt.get_dim_out(), 1]))
        return tf.reshape (dot_v, [-1])
        
    def comp_ef (self, 
                 dcoord, 
                 dbox, 
                 dtype,
                 tnatoms,
                 name,
                 reuse = None) :
        dout = self.descrpt.build(dcoord, dtype, tnatoms, dbox, self.default_mesh, self.avg, self.std, suffix=name, reuse=reuse)
        inputs_reshape = tf.reshape (dout, [-1, self.descrpt.get_dim_out()])
        atom_ener = self._net (inputs_reshape, name, reuse = reuse)
        atom_ener_reshape = tf.reshape(atom_ener, [-1, self.natoms[0]])       
        energy = tf.reduce_sum (atom_ener_reshape, axis = 1)        
        force, virial, av = self.descrpt.prod_force_virial(atom_ener_reshape, tnatoms)
        return energy, force, virial


class TestDescrptAR(Inter, unittest.TestCase):
    def __init__ (self, *args, **kwargs):
        data = Data()
        Inter.__init__(self, data)
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.controller = object()

    def test_force (self) :
        force_test(self, self, suffix = '_se_ar')

    def test_virial (self) :
        virial_test(self, self, suffix = '_se_ar')


if __name__ == '__main__':
    unittest.main()
