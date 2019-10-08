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

from deepmd.DescrptLocFrame import op_module

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
        self.sel = [12,24]
        self.sel_a = [0,0]
        self.rcut_smth = 2.45
        self.rcut = 10.0
        self.nnei = np.cumsum(self.sel)[-1]
        self.ndescrpt = self.nnei * 1
        davg = np.zeros ([self.ntypes, self.ndescrpt])
        dstd = np.ones  ([self.ntypes, self.ndescrpt])
        self.t_avg = tf.constant(davg.astype(global_np_float_precision))
        self.t_std = tf.constant(dstd.astype(global_np_float_precision))
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
                                     [self.ndescrpt], 
                                     global_tf_float_precision,
                                     tf.constant_initializer (self.net_w_i))
        dot_v = tf.matmul (tf.reshape (inputs, [-1, self.ndescrpt]),
                           tf.reshape (net_w, [self.ndescrpt, 1]))
        return tf.reshape (dot_v, [-1])
        
    def comp_ef (self, 
                 dcoord, 
                 dbox, 
                 dtype,
                 tnatoms,
                 name,
                 reuse = None) :
        descrpt, descrpt_deriv, rij, nlist \
            = op_module.descrpt_se_r (dcoord, 
                                      dtype,
                                      tnatoms,
                                      dbox, 
                                      tf.constant(self.default_mesh),
                                      self.t_avg,
                                      self.t_std,
                                      rcut = self.rcut, 
                                      rcut_smth = self.rcut_smth,
                                      sel = self.sel)
        inputs_reshape = tf.reshape (descrpt, [-1, self.ndescrpt])
        atom_ener = self._net (inputs_reshape, name, reuse = reuse)
        atom_ener_reshape = tf.reshape(atom_ener, [-1, self.natoms[0]])        
        energy = tf.reduce_sum (atom_ener_reshape, axis = 1)        
        net_deriv_ = tf.gradients (atom_ener, inputs_reshape)
        net_deriv = net_deriv_[0]
        net_deriv_reshape = tf.reshape (net_deriv, [-1, self.natoms[0] * self.ndescrpt]) 

        force = op_module.prod_force_se_r (net_deriv_reshape, 
                                            descrpt_deriv, 
                                            nlist, 
                                            tnatoms)
        virial, atom_vir = op_module.prod_virial_se_r (net_deriv_reshape, 
                                                        descrpt_deriv, 
                                                        rij,
                                                        nlist, 
                                                        tnatoms)
        return energy, force, virial


    def comp_f_dw (self, 
                   dcoord, 
                   dbox, 
                   dtype,                 
                   tnatoms,
                   name,
                   reuse = None) :
        energy, force, virial = self.comp_ef (dcoord, dbox, dtype, tnatoms, name, reuse)
        with tf.variable_scope(name, reuse=True):
            net_w = tf.get_variable ('net_w', [self.ndescrpt], global_tf_float_precision, tf.constant_initializer (self.net_w_i))
        f_mag = tf.reduce_sum (tf.nn.tanh(force))
        f_mag_dw = tf.gradients (f_mag, net_w)
        assert (len(f_mag_dw) == 1), "length of dw is wrong"        
        return f_mag, f_mag_dw[0]


    def comp_v_dw (self, 
                   dcoord, 
                   dbox, 
                   dtype,                 
                   tnatoms,
                   name,
                   reuse = None) :
        energy, force, virial = self.comp_ef (dcoord, dbox, dtype, tnatoms, name, reuse)
        with tf.variable_scope(name, reuse=True):
            net_w = tf.get_variable ('net_w', [self.ndescrpt], global_tf_float_precision, tf.constant_initializer (self.net_w_i))
        v_mag = tf.reduce_sum (virial)
        v_mag_dw = tf.gradients (v_mag, net_w)
        assert (len(v_mag_dw) == 1), "length of dw is wrong"        
        return v_mag, v_mag_dw[0]



class TestSmooth(Inter, unittest.TestCase):
    def __init__ (self, *args, **kwargs):
        data = Data()
        Inter.__init__(self, data)
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.controller = object()

    def test_force (self) :
        force_test(self, self, suffix = '_se_r')

    def test_virial (self) :
        virial_test(self, self, suffix = '_se_r')

    def test_force_dw (self) :
        force_dw_test(self, self, suffix = '_se_r')

    def test_virial_dw (self) :
        virial_dw_test(self, self, suffix = '_se_r')


if __name__ == '__main__':
    unittest.main()
