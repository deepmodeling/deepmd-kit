import paddle
from paddle_ops import prod_env_mat_a, prod_force_se_a, prod_virial_se_a

import os,sys
import numpy as np
import unittest
import time

from deepmd.env import op_module

from deepmd.env import GLOBAL_TF_FLOAT_PRECISION
from deepmd.env import GLOBAL_NP_FLOAT_PRECISION
from deepmd.env import GLOBAL_ENER_FLOAT_PRECISION

import unittest

from deepmd.env import tf
from tensorflow.python.framework import ops

from common import Data

if GLOBAL_NP_FLOAT_PRECISION == np.float32 :
    global_default_fv_hh = 1e-2
    global_default_dw_hh = 1e-2
    global_default_places = 3
else :
    global_default_fv_hh = 1e-5
    global_default_dw_hh = 1e-4
    global_default_places = 5

class Inter():
    def setUp (self, 
               data, 
               pbc = True) :
        self.sess = tf.Session()
        self.data = data
        self.natoms = self.data.get_natoms()
        self.ntypes = self.data.get_ntypes()
        self.sel_a = [12,24]
        self.sel_r = [0,0]
        self.rcut_a = -1
        self.rcut_r_smth = 2.45
        self.rcut_r = 10.0
        self.nnei_a = np.cumsum(self.sel_a)[-1]
        self.nnei_r = np.cumsum(self.sel_r)[-1]
        self.nnei = self.nnei_a + self.nnei_r
        self.ndescrpt_a = self.nnei_a * 4
        self.ndescrpt_r = self.nnei_r * 1
        self.ndescrpt = self.ndescrpt_a + self.ndescrpt_r
        davg = np.zeros ([self.ntypes, self.ndescrpt])
        dstd = np.ones  ([self.ntypes, self.ndescrpt])
        self.t_avg = tf.constant(davg.astype(GLOBAL_NP_FLOAT_PRECISION))
        self.t_std = tf.constant(dstd.astype(GLOBAL_NP_FLOAT_PRECISION))
        if pbc:
            self.default_mesh = np.zeros (6, dtype = np.int32)
            self.default_mesh[3] = 2
            self.default_mesh[4] = 2
            self.default_mesh[5] = 2
        else:
            self.default_mesh = np.array([], dtype = np.int32)
        # make place holder
        self.coord      = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None, self.natoms[0] * 3], name='t_coord')
        self.box        = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None, 9], name='t_box')
        self.type       = tf.placeholder(tf.int32,   [None, self.natoms[0]], name = "t_type")
        self.tnatoms    = tf.placeholder(tf.int32,   [None], name = "t_natoms")
        self.efield     = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None, self.natoms[0] * 3], name='t_efield')
        
    def _net (self,
             inputs, 
             name,
              reuse = False) :
        with tf.variable_scope(name, reuse=reuse):
            net_w = tf.get_variable ('net_w', 
                                     [self.ndescrpt], 
                                     GLOBAL_TF_FLOAT_PRECISION,
                                     tf.constant_initializer (self.net_w_i))
        dot_v = tf.matmul (tf.reshape (inputs, [-1, self.ndescrpt]),
                           tf.reshape (net_w, [self.ndescrpt, 1]))
        return tf.reshape (dot_v, [-1])
        
    def comp_ef_before (self, 
                 dcoord, 
                 dbox, 
                 dtype,
                 tnatoms,
                 name,
                 reuse = None) :
        descrpt, descrpt_deriv, rij, nlist \
            = op_module.prod_env_mat_a (dcoord, 
                                       dtype,
                                       tnatoms,
                                       dbox, 
                                       tf.constant(self.default_mesh),
                                       self.t_avg,
                                       self.t_std,
                                       rcut_a = self.rcut_a, 
                                       rcut_r = self.rcut_r, 
                                       rcut_r_smth = self.rcut_r_smth,
                                       sel_a = self.sel_a, 
                                       sel_r = self.sel_r)
        inputs_reshape = tf.reshape (descrpt, [-1, self.ndescrpt])
        atom_ener = self._net (inputs_reshape, name, reuse = reuse)
        atom_ener_reshape = tf.reshape(atom_ener, [-1, self.natoms[0]])        
        energy = tf.reduce_sum (atom_ener_reshape, axis = 1)        
        net_deriv_ = tf.gradients (atom_ener, inputs_reshape)
        net_deriv = net_deriv_[0]
        net_deriv_reshape = tf.reshape (net_deriv, [-1, self.natoms[0] * self.ndescrpt]) 

        return net_deriv_reshape, descrpt_deriv, nlist, tnatoms, rij, energy


    def comp_f_dw (self, 
                   dcoord, 
                   dbox, 
                   dtype,                 
                   tnatoms,
                   name,
                   reuse = None) :
        net_deriv_reshape, descrpt_deriv, nlist, tnatoms, rij, energy = self.comp_ef_before(dcoord, dbox, dtype, tnatoms, name, reuse)
        
        return net_deriv_reshape, descrpt_deriv, nlist, tnatoms, rij, energy


def force_test (inter, 
                testCase, 
                places = global_default_places, 
                hh = global_default_fv_hh, 
                suffix = '') :
    # set weights
    w0 = np.ones (inter.ndescrpt)
    inter.net_w_i = np.copy(w0)
    # make network
    net_deriv_reshape, descrpt_deriv, nlist, tnatoms, rij, energy \
        = inter.comp_ef_before (inter.coord, inter.box, inter.type, inter.tnatoms, name = "test_f" + suffix)
    inter.sess.run (tf.global_variables_initializer())
    # get data
    dcoord, dbox, dtype = inter.data.get_data ()
    defield = inter.data.efield
    # cmp e0, f0
    [dnet_deriv_reshape, ddescrpt_deriv, dnlist, dtnatoms, drij, denergy] = \
        inter.sess.run ([net_deriv_reshape, descrpt_deriv, nlist, tnatoms, rij, energy], 
                                        feed_dict = {
                                            inter.coord:     dcoord,
                                            inter.box:       dbox,
                                            inter.type:      dtype,
                                            inter.efield:    defield,
                                            inter.tnatoms:   inter.natoms}
    )
    
    force = prod_force_se_a (paddle.to_tensor(dnet_deriv_reshape.reshape(dnet_deriv_reshape.shape), dtype="float32"), 
                                        paddle.to_tensor(ddescrpt_deriv.reshape(ddescrpt_deriv.shape), dtype="float32"), 
                                        paddle.to_tensor(dnlist.reshape(dnlist.shape), dtype="int32"), 
                                        paddle.to_tensor (dtnatoms.reshape(dtnatoms.shape), dtype="int32"),
                                        n_a_sel = inter.nnei_a, 
                                        n_r_sel = inter.nnei_r)
    dforce = force.numpy()
    # dim force
    sel_idx = np.arange(inter.natoms[0])    
    for idx in sel_idx:
        for dd in range(3):
            dcoordp = np.copy(dcoord)
            dcoordm = np.copy(dcoord)
            dcoordp[0,idx*3+dd] = dcoord[0,idx*3+dd] + hh
            dcoordm[0,idx*3+dd] = dcoord[0,idx*3+dd] - hh
            [enerp] = inter.sess.run ([energy], 
                                     feed_dict = {
                                         inter.coord:     dcoordp,
                                         inter.box:       dbox,
                                         inter.type:      dtype,
                                         inter.efield:    defield,
                                         inter.tnatoms:   inter.natoms}
            )
            [enerm] = inter.sess.run ([energy], 
                                     feed_dict = {
                                         inter.coord:     dcoordm,
                                         inter.box:       dbox,
                                         inter.type:      dtype,
                                         inter.efield:    defield,
                                         inter.tnatoms:   inter.natoms}
            )
            c_force = -(enerp[0] - enerm[0]) / (2*hh)
            testCase.assertAlmostEqual(c_force, dforce[0,idx*3+dd], 
                                       places = places,
                                       msg = "force component [%d,%d] failed" % (idx, dd))

def force_dw_test (inter, 
                   testCase,
                   places = global_default_places,
                   hh = global_default_dw_hh, 
                   suffix = '') :
    dcoord, dbox, dtype = inter.data.get_data()
    defield = inter.data.efield
    feed_dict_test0 = {
        inter.coord:     dcoord,
        inter.box:       dbox,
        inter.type:      dtype,
        inter.efield:    defield,
        inter.tnatoms:   inter.natoms}

    w0 = np.ones (inter.ndescrpt)
    inter.net_w_i = np.copy(w0)

    net_deriv_reshape, descrpt_deriv, nlist, tnatoms, rij, energy \
            = inter.comp_f_dw(inter.coord, inter.box, inter.type, inter.tnatoms, name = "f_dw_test_0" + suffix)
    inter.sess.run (tf.global_variables_initializer())
    [dnet_deriv_reshape, ddescrpt_deriv, dnlist, dtnatoms, drij, denergy] = \
            inter.sess.run ([net_deriv_reshape, descrpt_deriv, nlist, tnatoms, rij, energy], feed_dict = feed_dict_test0)

    paddle.set_device("cpu")
    net_deriv_pd_tensor = paddle.to_tensor(dnet_deriv_reshape.reshape(dnet_deriv_reshape.shape), dtype="float32")
    net_deriv_pd_tensor.stop_gradient = False
    force = prod_force_se_a (net_deriv_pd_tensor, 
                                    paddle.to_tensor(ddescrpt_deriv.reshape(ddescrpt_deriv.shape), dtype="float32"), 
                                    paddle.to_tensor(dnlist.reshape(dnlist.shape), dtype="int32"), 
                                    paddle.to_tensor (dtnatoms.reshape(dtnatoms.shape), dtype="int32"),
                                    n_a_sel = inter.nnei_a, 
                                    n_r_sel = inter.nnei_r)
    
    rlt = paddle.fluid.layers.reduce_sum(paddle.fluid.layers.tanh(force))
    rlt.backward()
    dw = net_deriv_pd_tensor.grad

def virial_test (inter, 
                 testCase, 
                 places = global_default_places, 
                 hh = global_default_fv_hh, 
                 suffix = '') :
    # set weights
    w0 = np.ones (inter.ndescrpt)
    inter.net_w_i = np.copy(w0)
    # make network
    net_deriv_reshape, descrpt_deriv, nlist, tnatoms, rij, energy \
        = inter.comp_ef_before (inter.coord, inter.box, inter.type, inter.tnatoms, name = "test_v" + suffix)
    inter.sess.run (tf.global_variables_initializer())
    # get data
    dcoord, dbox, dtype, defield = inter.data.get_test_box_data(hh)
    # cmp e, f, v
    [dnet_deriv_reshape, ddescrpt_deriv, dnlist, dtnatoms, drij, denergy] = \
        inter.sess.run ([net_deriv_reshape, descrpt_deriv, nlist, tnatoms, rij, energy], 
                                        feed_dict = {
                                            inter.coord:     dcoord,
                                            inter.box:       dbox,
                                            inter.type:      dtype,
                                            inter.efield:    defield,
                                            inter.tnatoms:   inter.natoms}
    )
    virial, atom_vir = prod_virial_se_a (paddle.to_tensor(dnet_deriv_reshape.reshape(dnet_deriv_reshape.shape), dtype="float32"), 
                                        paddle.to_tensor(ddescrpt_deriv.reshape(ddescrpt_deriv.shape), dtype="float32"), 
                                        paddle.to_tensor(drij.reshape(drij.shape), dtype="float32"),
                                        paddle.to_tensor(dnlist.reshape(dnlist.shape), dtype="int32"), 
                                        paddle.to_tensor (dtnatoms.reshape(dtnatoms.shape), dtype="int32"),
                                        n_a_sel = inter.nnei_a, 
                                        n_r_sel = inter.nnei_r)
    dvirial = virial.numpy()
    dvirial.shape
    print("dvirial shape is {} \n value is : {}".format(dvirial.shape, dvirial))                                  
    # ana_vir = dvirial[0].reshape([3,3])
    # num_vir = np.zeros([3,3])
    # for ii in range(3):
    #     for jj in range(3):
    #         ep = energy[1+(ii*3+jj)*2+0]
    #         em = energy[1+(ii*3+jj)*2+1]
    #         num_vir[ii][jj] = -(ep - em) / (2.*hh)
    # num_vir = np.transpose(num_vir, [1,0])    
    # box3 = dbox[0].reshape([3,3])
    # num_vir = np.matmul(num_vir, box3)
    # for ii in range(3):
    #     for jj in range(3):
    #         testCase.assertAlmostEqual(ana_vir[ii][jj], num_vir[ii][jj],
    #                                    places=places, 
    #                                    msg = 'virial component %d %d ' % (ii,jj))
    


class TestPdSmooth(Inter, unittest.TestCase):
    def setUp(self):
        self.places = 5
        data = Data()
        Inter.setUp(self, data)

    def test_force (self) :
        force_test(self, self, suffix = '_smth')
    
    # def test_force_dw (self) :
    #     force_dw_test(self, self, suffix = '_smth')

    def test_virial (self) :
        virial_test(self, self, suffix = '_smth')

if __name__ == '__main__':
    unittest.main()
