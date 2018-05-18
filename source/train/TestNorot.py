#!/usr/bin/env python3

import os
import sys
import time
import numpy as np
import glob
import tensorflow as tf

from tensorflow.python.framework import ops

# load force module
module_path = os.path.dirname(os.path.realpath(__file__)) + "/"
assert (os.path.isfile (module_path  + "libop_abi.so" )), "op module does not exist"
op_module = tf.load_op_library(module_path + "libop_abi.so")

# load grad of force module
sys.path.append (module_path )
import _prod_force_norot_grad
import _prod_virial_norot_grad

class DataSets (object):
    def __init__ (self, 
                  set_prefix = "set",
                  hh = 1e-6,
                  seed = None) :
        self.dirs = glob.glob (set_prefix + ".*")
        self.dirs.sort()        
        self.test_dir = self.dirs[-1]
        self.set_count = 0
        self.hh = hh
        self.load_test_set (self.test_dir)

    def get_numb_set (self) :
        return len (self.train_dirs)
    
    def stats_energy (self) :
        eners = []
        for ii in self.dirs:
            ei = np.load (ii + "/energy.npy")
            eners.append (np.average(ei))
        return np.average (eners)    

    def load_test_set (self,
                       set_name) :
        start_time = time.time()
        coord_test = np.load (set_name + "/coord.npy")
        box_test = np.load (set_name + "/box.npy")
        # dirty workaround, type in type.raw should be sorted
        type_test = np.loadtxt (set_name + "/../type.raw")

        self.coord_test0        = np.array([coord_test[0]])
        self.box_test0          = np.array([box_test[0]])
        self.type_test0         = np.array([type_test])

        self.coord_test         = [coord_test[0]]
        self.box_test           = [box_test[0]]
        self.type_test          = np.array([type_test])

        coord0 = np.copy (self.coord_test[0])

        self.natoms = self.type_test[0].shape[0]
        for ii in range(self.natoms * 3) :
            p_coord = np.copy (coord0)
            n_coord = np.copy (coord0)
            p_coord[ii] += self.hh
            n_coord[ii] -= self.hh
            self.coord_test.append (p_coord)
            self.coord_test.append (n_coord)
            self.box_test.append (box_test[0])
            self.box_test.append (box_test[0])

        self.coord_test = np.array(self.coord_test)
        self.box_test = np.array(self.box_test)
        self.type_test = np.tile (self.type_test, (2 * self.natoms * 3 + 1, 1))

        end_time = time.time()
        
    def get_test (self) :
        return self.coord_test, self.box_test, self.type_test        

    def get_test0 (self) :
        return self.coord_test0, self.box_test0, self.type_test0        

    def get_test_box (self,
                      hh) :
        coord0_, box0_, type0_ = self.get_test0()
        coord0 = coord0_[0]
        box0 = box0_[0]
        type0 = type0_[0]
        nc = np.array( [coord0, coord0*(1+hh), coord0*(1-hh)] )
        nb = np.array( [box0, box0*(1+hh), box0*(1-hh)] )
        nt = np.array( [type0, type0, type0] )
        for dd in range(3) :
            tmpc = np.copy (coord0)
            tmpb = np.copy (box0)
            tmpc = np.reshape(tmpc, [-1, 3])
            tmpc [:,dd] *= (1+hh)
            tmpc = np.reshape(tmpc, [-1])
            tmpb = np.reshape(tmpb, [-1, 3])
            tmpb [dd,:] *= (1+hh)
            tmpb = np.reshape(tmpb, [-1])
            nc = np.append (nc, [tmpc], axis = 0)
            nb = np.append (nb, [tmpb], axis = 0)
            nt = np.append (nt, [type0], axis = 0)
            tmpc = np.copy (coord0)
            tmpb = np.copy (box0)
            tmpc = np.reshape(tmpc, [-1, 3])
            tmpc [:,dd] *= (1-hh)
            tmpc = np.reshape(tmpc, [-1])
            tmpb = np.reshape(tmpb, [-1, 3])
            tmpb [dd,:] *= (1-hh)
            tmpb = np.reshape(tmpb, [-1])
            nc = np.append (nc, [tmpc], axis = 0)
            nb = np.append (nb, [tmpb], axis = 0)
            nt = np.append (nt, [type0], axis = 0)
        return nc, nb, nt

    def get_natoms (self) :
        ntype1 = np.sum (self.type_test0)
        tmp = np.array([self.natoms, self.natoms, self.natoms - ntype1, ntype1])
        return tmp.astype(np.int32)

    def get_h (self) :
        return self.hh

class Model (object) :
    def __init__ (self, 
                  sess, 
                  data, 
                  comp = 0) :
        self.sess = sess
        self.natoms = data.get_natoms()
        self.comp = comp
        self.sel_a = [12, 24]
        self.sel_r = [0,0]
        self.rcut_a = -1
        self.rcut_r_smth = 2.45
        self.rcut_r = 3.45     
        self.axis_rule = [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0]
        self.nnei_a = np.cumsum(self.sel_a)[-1]
        self.nnei_r = np.cumsum(self.sel_r)[-1]
        self.nnei = self.nnei_a + self.nnei_r
        self.ndescrpt_a = self.nnei_a * 4
        self.ndescrpt_r = self.nnei_r * 1
        self.ndescrpt = self.ndescrpt_a + self.ndescrpt_r
        davg = np.zeros (self.ndescrpt)
        dstd = np.ones  (self.ndescrpt)
        self.t_avg = tf.constant(davg.astype(np.float64))
        self.t_std = tf.constant(dstd.astype(np.float64))
        self.default_mesh = np.zeros (6, dtype = np.int32)
        self.default_mesh[3] = 2
        self.default_mesh[4] = 2
        self.default_mesh[5] = 2

    def net (self,
             inputs, 
             name,
             reuse = False) :
        with tf.variable_scope(name, reuse=reuse):
            net_w = tf.get_variable ('net_w', 
                                     [self.ndescrpt], 
                                     tf.float64,
                                     tf.constant_initializer (self.net_w_i))
        dot_v = tf.matmul (tf.reshape (inputs, [-1, self.ndescrpt]),
                           tf.reshape (net_w, [self.ndescrpt, 1]))
        return tf.reshape (dot_v, [-1])

    def comp_descrpt (self, 
                      dcoord, 
                      dbox, 
                      dtype,
                      tnatoms,
                      name) :
        descrpt, descrpt_deriv, rij, nlist, \
            = op_module.descrpt_norot (dcoord, 
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
        return descrpt, descrpt_deriv

    def comp_ef (self, 
                 dcoord, 
                 dbox, 
                 dtype,
                 tnatoms,
                 name,
                 reuse = None) :
        descrpt, descrpt_deriv, rij, nlist, \
            = op_module.descrpt_norot (dcoord, 
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
        atom_ener = self.net (inputs_reshape, name, reuse = reuse)
        atom_ener_reshape = tf.reshape(atom_ener, [-1, self.natoms[0]])        
        energy = tf.reduce_sum (atom_ener_reshape, axis = 1)        
        net_deriv_ = tf.gradients (atom_ener, inputs_reshape)
        net_deriv = net_deriv_[0]
        net_deriv_reshape = tf.reshape (net_deriv, [-1, self.natoms[0] * self.ndescrpt]) 

        force = op_module.prod_force_norot (net_deriv_reshape, 
                                            descrpt_deriv, 
                                            nlist, 
                                            tnatoms,
                                            n_a_sel = self.nnei_a, 
                                            n_r_sel = self.nnei_r)
        virial, atom_vir = op_module.prod_virial_norot (net_deriv_reshape, 
                                                        descrpt_deriv, 
                                                        rij,
                                                        nlist, 
                                                        tnatoms,
                                                        n_a_sel = self.nnei_a, 
                                                        n_r_sel = self.nnei_r)
        return energy, force, virial

    def comp_fl (self, 
                 dcoord, 
                 dbox, 
                 dtype,                 
                 tnatoms,
                 name,
                 reuse = None) :
        energy, force, virial = self.comp_ef (dcoord, dbox, dtype, tnatoms, name, reuse)
        with tf.variable_scope(name, reuse=True):
            net_w = tf.get_variable ('net_w', [self.ndescrpt], tf.float64, tf.constant_initializer (self.net_w_i))
        f_mag = tf.reduce_sum (tf.nn.tanh(force))
        f_mag_dw = tf.gradients (f_mag, net_w)
        assert (len(f_mag_dw) == 1), "length of dw is wrong"        
        return f_mag, f_mag_dw[0]

    def comp_vl (self, 
                 dcoord, 
                 dbox, 
                 dtype,                 
                 tnatoms,
                 name,
                 reuse = None) :
        energy, force, virial = self.comp_ef (dcoord, dbox, dtype, tnatoms, name, reuse)
        with tf.variable_scope(name, reuse=True):
            net_w = tf.get_variable ('net_w', [self.ndescrpt], tf.float64, tf.constant_initializer (self.net_w_i))
        v_mag = tf.reduce_sum (virial)
        v_mag_dw = tf.gradients (v_mag, net_w)
        assert (len(v_mag_dw) == 1), "length of dw is wrong"        
        return v_mag, v_mag_dw[0]

    def make_place (self) :
        self.coord      = tf.placeholder(tf.float64, [None, self.natoms[0] * 3], name='t_coord')
        self.box        = tf.placeholder(tf.float64, [None, 9], name='t_box')
        self.type       = tf.placeholder(tf.int32,   [None, self.natoms[0]], name = "t_type")
        self.tnatoms    = tf.placeholder(tf.int32,   [None], name = "t_natoms")
        
    def make_feed_dict (self, 
                        data ) :
        dcoord, dbox, dtype = data.get_test ()
        return {self.coord:     dcoord,
                self.box:       dbox,
                self.type:      dtype, 
                self.tnatoms:   self.natoms,
        }

    def make_feed_dict0 (self, 
                         data ) :
        dcoord, dbox, dtype = data.get_test0 ()
        return {self.coord:     dcoord,
                self.box:       dbox,
                self.type:      dtype,
                self.tnatoms:   self.natoms,
        }

    def test_descrpt (self, 
                      data) :
        self.make_place ()
        feed_dict_test = self.make_feed_dict (data)
        feed_dict_test0 = self.make_feed_dict0 (data)

        self.net_w_i = 1 * np.ones (self.ndescrpt)
        t_descrpt, t_descrpt_deriv = self.comp_descrpt (self.coord, self.box, self.type, self.tnatoms,  name = "test_0")

        self.sess.run (tf.global_variables_initializer())
        [mydescrpt, mydescrpt_deriv] = self.sess.run ([t_descrpt, t_descrpt_deriv], feed_dict = feed_dict_test)
        
        print (mydescrpt)
        print (mydescrpt.shape)
        print (mydescrpt_deriv)
        print (mydescrpt_deriv.shape)
        

    def test_force (self, 
                    data) :
        self.make_place ()
        feed_dict_test = self.make_feed_dict (data)
        feed_dict_test0 = self.make_feed_dict0 (data)

        self.net_w_i = 1 * np.ones (self.ndescrpt)
        t_energy, t_force, t_virial = self.comp_ef (self.coord, self.box, self.type, self.tnatoms,  name = "test_0")                
        self.sess.run (tf.global_variables_initializer())
        energy = self.sess.run (t_energy, feed_dict = feed_dict_test)
        force  = self.sess.run (t_force , feed_dict = feed_dict_test)
        # virial = self.sess.run (t_virial , feed_dict = feed_dict_test)        
        
        hh2 = data.get_h() * 2.
        ndof = (len(energy) - 1) // 2
        absolut_e = []
        relativ_e = []
        for ii in range (ndof) :
            idx0 = ii * 2 + 1
            idx1 = ii * 2 + 2
            #              +hh            -hh
            num_force = - (energy[idx0] - energy[idx1]) / hh2
            ana_force = force[0][ii]
            diff = np.abs(num_force - ana_force)
            absolut_e.append (diff)
            relativ_e.append (diff / np.abs(ana_force))
            print ("component  %6u \t value %12.5e numerical %12.5e \t diff: %10.2e \t relat: %10.2e" % (ii, ana_force, num_force, diff, np.abs(diff/ana_force)))

        print ("max absolute %e" % np.max(absolut_e))
        print ("max relative %e" % np.max(relativ_e))        

    def comp_vol (self, 
                  box) : 
        return np.linalg.det (np.reshape(box, (3,3)))

    def test_virial (self, 
                     data ) :
        hh = 1e-6

        self.make_place ()
        dcoord, dbox, dtype = data.get_test_box (hh)
        feed_dict_box =  {self.coord:     dcoord,
                          self.box:       dbox,
                          self.type:      dtype, 
                          self.tnatoms:   self.natoms, 
        }

        self.net_w_i = 1 * np.ones (self.ndescrpt)

        t_energy, t_force, t_virial = self.comp_ef (self.coord, self.box, self.type, self.tnatoms,  name = "test_0")
        self.sess.run (tf.global_variables_initializer())
        virial = self.sess.run (t_virial , feed_dict = feed_dict_box)
        energy = self.sess.run (t_energy , feed_dict = feed_dict_box)

        print ("printing virial")
        ana_vir3 = (virial[0][0] + virial[0][4] + virial[0][8])/3. / self.comp_vol(dbox[0])
        num_vir3 = -(energy[1] - energy[2]) / (self.comp_vol(dbox[1]) - self.comp_vol(dbox[2]))
        print ( "all-dir:  ana %14.5e  num %14.5e  diff %.2e" % (ana_vir3, num_vir3, np.abs(ana_vir3 - num_vir3)) )
        vir_idx = [0, 4, 8]
        ana_v = []
        num_v = []
        for dd in range (3) :
            ana_v.append (virial[0][vir_idx[dd]] / self.comp_vol(dbox[0]))
            idx = 2 * (dd+1) + 1
            num_v.append ( -(energy[idx] - energy[idx+1]) / (self.comp_vol(dbox[idx]) - self.comp_vol(dbox[idx+1])) )
        for dd in range (3) :
            print ( "dir   %d:  ana %14.5e  num %14.5e  diff %.2e" % (dd, ana_v[dd], num_v[dd], np.abs(ana_v[dd] - num_v[dd])) )

    def test_dw (self, 
                 data) :
        self.make_place ()
        feed_dict_test0 = self.make_feed_dict0 (data)

        w0 = np.ones (self.ndescrpt)
        self.net_w_i = np.copy(w0)
        
        t_ll, t_dw = self.comp_fl (self.coord, self.box, self.type, self.tnatoms, name = "test_0")
        self.sess.run (tf.global_variables_initializer())
        ll_0 = self.sess.run (t_ll, feed_dict = feed_dict_test0)
        dw_0 = self.sess.run (t_dw, feed_dict = feed_dict_test0)
        
        hh = 1e-4
        absolut_e = []
        relativ_e = []
        for ii in range (self.ndescrpt) :
            self.net_w_i = np.copy (w0)
            self.net_w_i[ii] += hh
            t_ll, t_dw = self.comp_fl (self.coord, self.box, self.type, self.tnatoms, name = "test_" + str(ii*2+1))
            self.sess.run (tf.global_variables_initializer())
            ll_1 = self.sess.run (t_ll, feed_dict = feed_dict_test0)
            self.net_w_i[ii] -= 2. * hh
            t_ll, t_dw = self.comp_fl (self.coord, self.box, self.type, self.tnatoms, name = "test_" + str(ii*2+2))
            self.sess.run (tf.global_variables_initializer())
            ll_2 = self.sess.run (t_ll, feed_dict = feed_dict_test0)
            num_v = (ll_1 - ll_2) / (2. * hh)
            ana_v = dw_0[ii]
            diff = np.abs (num_v - ana_v)
            if (np.abs(ana_v) < 1e-10) :
                diff_r = diff
            else :
                diff_r = diff / np.abs(ana_v)
            print ("component  %6u \t value %12.5e n_v %.12e \t diff: %10.2e \t relat: %10.2e" % (ii, ana_v, num_v, diff, diff_r))
            absolut_e.append (diff)
            relativ_e.append (diff_r)
        
        print ("max absolute %e" % np.max(absolut_e))
        print ("max relative %e" % np.max(relativ_e))

    def test_virial_dw (self, 
                        data) :
        self.make_place ()
        feed_dict_test0 = self.make_feed_dict0 (data)

        w0 = np.ones (self.ndescrpt)
        self.net_w_i = np.copy(w0)
        
        t_ll, t_dw = self.comp_vl (self.coord, self.box, self.type, self.tnatoms, name = "test_0")
        self.sess.run (tf.global_variables_initializer())
        ll_0 = self.sess.run (t_ll, feed_dict = feed_dict_test0)
        dw_0 = self.sess.run (t_dw, feed_dict = feed_dict_test0)
        
        hh = 1e-4
        absolut_e = []
        relativ_e = []
        for ii in range (self.ndescrpt) :
            self.net_w_i = np.copy (w0)
            self.net_w_i[ii] += hh
            t_ll, t_dw = self.comp_vl (self.coord, self.box, self.type, self.tnatoms, name = "test_" + str(ii*2+1))
            self.sess.run (tf.global_variables_initializer())
            ll_1 = self.sess.run (t_ll, feed_dict = feed_dict_test0)
            self.net_w_i[ii] -= 2. * hh
            t_ll, t_dw = self.comp_vl (self.coord, self.box, self.type, self.tnatoms, name = "test_" + str(ii*2+2))
            self.sess.run (tf.global_variables_initializer())
            ll_2 = self.sess.run (t_ll, feed_dict = feed_dict_test0)
            num_v = (ll_1 - ll_2) / (2. * hh)
            ana_v = dw_0[ii]
            diff = np.abs (num_v - ana_v)
            if (np.abs(ana_v) < 1e-10) :
                diff_r = diff
            else :
                diff_r = diff / np.abs(ana_v)
            print ("component  %6u \t value %12.5e  n_v %12.5e \t diff: %10.2e \t relat: %10.2e" % (ii, ana_v, num_v, diff, diff_r))
            absolut_e.append (diff)
            relativ_e.append (diff_r)
        
        print ("max absolute %e" % np.max(absolut_e))
        print ("max relative %e" % np.max(relativ_e))

def _main () :
    data = DataSets (set_prefix = "set")
    tf.reset_default_graph()

    with tf.Session() as sess:
        md = Model (sess, data)
        md.test_force (data)
        # md.test_virial (data)
        # md.test_dw (data)
        # md.test_virial_dw (data)

if __name__ == '__main__':
    _main()

