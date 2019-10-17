import os,sys,platform
import numpy as np
import unittest
from deepmd.env import tf

from deepmd.RunOptions import global_tf_float_precision
from deepmd.RunOptions import global_np_float_precision
from deepmd.RunOptions import global_ener_float_precision
from deepmd.EwaldRecp import op_module

if global_np_float_precision == np.float32 :
    global_default_fv_hh = 1e-2
    global_default_dw_hh = 1e-2
    global_default_places = 3
else :
    global_default_fv_hh = 1e-6
    global_default_dw_hh = 1e-4
    global_default_places = 5


class TestEwaldRecp (unittest.TestCase) :
    def setUp(self):
        boxl = 6
        box_pert = 1
        self.natoms = 16
        self.nframes = 2
        self.ewald_h = 1
        self.ewald_beta = 1
        self.dbox = []
        self.dcoord = []
        self.dcharge = []
        for ii in range(self.nframes):
            # box
            box = np.ones([3,3]) * boxl
            box += np.random.random([3,3]) * box_pert
            self.dbox.append(0.5 * (box + box.T))
            # scaled 
            self.coord = np.random.random([self.natoms, 3])
            # real coords
            self.dcoord.append(np.matmul(self.coord, box))
            # charge
            dcharge = np.random.random([self.natoms])
            dcharge -= np.average(dcharge)
            assert(np.abs(np.sum(self.dcharge) - 0) < 1e-12)
            self.dcharge.append(dcharge)
        self.dbox = np.array(self.dbox).reshape([self.nframes, 9])
        self.dcoord = np.array(self.dcoord).reshape([self.nframes, 3*self.natoms])
        self.dcharge = np.array(self.dcharge).reshape([self.nframes, self.natoms])
        # place holders
        self.coord      = tf.placeholder(global_tf_float_precision, [None, self.natoms * 3], name='t_coord')
        self.charge     = tf.placeholder(global_tf_float_precision, [None, self.natoms], name='t_charge')
        self.box        = tf.placeholder(global_tf_float_precision, [None, 9], name='t_box')
        self.nloc    = tf.placeholder(tf.int32, [1], name = "t_nloc")        

    def test_force(self):
        hh = 1e-4
        places = 5
        sess = tf.Session()
        t_energy, t_force, t_virial \
            = op_module.ewald_recp(self.coord, self.charge, self.nloc, self.box, 
                                   ewald_h = self.ewald_h,
                                   ewald_beta = self.ewald_beta)
        [force] = sess.run([t_force], 
                           feed_dict = {
                               self.coord:  self.dcoord,
                               self.charge: self.dcharge,
                               self.box:    self.dbox,
                               self.nloc:   [self.natoms],
                           })
        for idx in range(self.natoms):
            for dd in range(3):
                dcoordp = np.copy(self.dcoord)
                dcoordm = np.copy(self.dcoord)
                dcoordp[:,idx*3+dd] = self.dcoord[:,idx*3+dd] + hh
                dcoordm[:,idx*3+dd] = self.dcoord[:,idx*3+dd] - hh
                energyp = sess.run([t_energy], 
                                   feed_dict = {
                                       self.coord:  dcoordp,
                                       self.charge: self.dcharge,
                                       self.box:    self.dbox,
                                       self.nloc:   [self.natoms],
                                   })                                
                energym = sess.run([t_energy], 
                                   feed_dict = {
                                       self.coord:  dcoordm,
                                       self.charge: self.dcharge,
                                       self.box:    self.dbox,
                                       self.nloc:   [self.natoms],
                                   })
                c_force = -(energyp[0] - energym[0]) / (2*hh)
                for ff in range(self.nframes):
                    self.assertAlmostEqual(c_force[ff], force[ff,idx*3+dd], 
                                           places = places,
                                           msg = "frame %d force component [%d,%d] failed" % (ff, idx, dd))


                



