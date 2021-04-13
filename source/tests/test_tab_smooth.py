import os,sys
import numpy as np
from deepmd.env import tf
import unittest

from tensorflow.python.framework import ops

# load grad of force module
import deepmd.op
from deepmd.utils.pair_tab import PairTab

from common import force_test
from common import virial_test
from common import force_dw_test
from common import virial_dw_test
from common import Data
from test_descrpt_smooth import Inter 

from deepmd.env import op_module

def _make_tab(ntype) :
    xx = np.arange(0,9,0.001)
    yy = 1000/(xx+.5)**6
    prt = xx
    ninter = ntype * (ntype + 1) // 2
    for ii in range(ninter) :
        prt = np.append(prt, yy)
    prt = np.reshape(prt, [ninter+1, -1])
    np.savetxt('tab.xvg', prt.T)


class IntplInter(Inter):
    def setUp (self, 
               data) :
        # tabulated
        Inter.setUp(self, data)
        _make_tab(data.get_ntypes())
        self.srtab = PairTab('tab.xvg')
        self.smin_alpha = 0.3
        self.sw_rmin = 1
        self.sw_rmax = 3.45
        tab_info, tab_data = self.srtab.get()
        with tf.variable_scope('tab', reuse=tf.AUTO_REUSE):
            self.tab_info = tf.get_variable('t_tab_info',
                                            tab_info.shape,
                                            dtype = tf.float64,
                                            trainable = False,
                                            initializer = tf.constant_initializer(tab_info))
            self.tab_data = tf.get_variable('t_tab_data',
                                            tab_data.shape,
                                            dtype = tf.float64,
                                            trainable = False,
                                            initializer = tf.constant_initializer(tab_data))

    def tearDown(self):
        os.remove('tab.xvg')
        
    def comp_ef (self, 
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

        sw_lambda, sw_deriv \
            = op_module.soft_min_switch(dtype, 
                                        rij, 
                                        nlist,
                                        tnatoms,
                                        sel_a = self.sel_a,
                                        sel_r = self.sel_r,
                                        alpha = self.smin_alpha,
                                        rmin = self.sw_rmin,
                                        rmax = self.sw_rmax)
        inv_sw_lambda = 1.0 - sw_lambda
        tab_atom_ener, tab_force, tab_atom_virial \
            = op_module.pair_tab(
                self.tab_info,
                self.tab_data,
                dtype,
                rij,
                nlist,
                tnatoms,
                sw_lambda,
                sel_a = self.sel_a,
                sel_r = self.sel_r)
        energy_diff = tab_atom_ener - tf.reshape(atom_ener, [-1, self.natoms[0]])
        tab_atom_ener = tf.reshape(sw_lambda, [-1]) * tf.reshape(tab_atom_ener, [-1])
        atom_ener = tf.reshape(inv_sw_lambda, [-1]) * atom_ener
        energy_raw = tab_atom_ener + atom_ener

        energy_raw = tf.reshape(energy_raw, [-1, self.natoms[0]])
        energy = tf.reduce_sum (energy_raw, axis = 1)

        net_deriv_ = tf.gradients (atom_ener, inputs_reshape)
        net_deriv = net_deriv_[0]
        net_deriv_reshape = tf.reshape (net_deriv, [-1, self.natoms[0] * self.ndescrpt]) 

        force = op_module.prod_force_se_a (net_deriv_reshape, 
                                      descrpt_deriv, 
                                      nlist, 
                                      tnatoms,
                                      n_a_sel = self.nnei_a, 
                                      n_r_sel = self.nnei_r)
        sw_force \
            = op_module.soft_min_force(energy_diff, 
                                       sw_deriv,
                                       nlist, 
                                       tnatoms,
                                       n_a_sel = self.nnei_a,
                                       n_r_sel = self.nnei_r)
        force = force + sw_force + tab_force
        virial, atom_vir = op_module.prod_virial_se_a (net_deriv_reshape, 
                                                  descrpt_deriv, 
                                                  rij,
                                                  nlist, 
                                                  tnatoms,
                                                  n_a_sel = self.nnei_a, 
                                                  n_r_sel = self.nnei_r)
        sw_virial, sw_atom_virial \
            = op_module.soft_min_virial (energy_diff,
                                         sw_deriv,
                                         rij,
                                         nlist,
                                         tnatoms,
                                         n_a_sel = self.nnei_a,
                                         n_r_sel = self.nnei_r)
        # atom_virial = atom_virial + sw_atom_virial + tab_atom_virial
        virial = virial + sw_virial \
                 + tf.reduce_sum(tf.reshape(tab_atom_virial, [-1, self.natoms[1], 9]), axis = 1)

        return energy, force, virial

    

class TestTabSmooth(IntplInter, unittest.TestCase):
    # def __init__ (self, *args, **kwargs):
    #     self.places = 5
    #     data = Data()
    #     IntplInter.__init__(self, data)
    #     unittest.TestCase.__init__(self, *args, **kwargs)
    #     self.controller = object()

    def setUp(self):
        self.places = 5
        data = Data()
        IntplInter.setUp(self, data)

    def test_force (self) :
        force_test(self, self, places=5, suffix = '_tab_smth')

    def test_virial (self) :
        virial_test(self, self, places=5, suffix = '_tab_smth')

    def test_force_dw (self) :
        force_dw_test(self, self, places=8, suffix = '_tab_smth')

    def test_virial_dw (self) :
        virial_dw_test(self, self, places=8, suffix = '_tab_smth')


if __name__ == '__main__':
    unittest.main()
