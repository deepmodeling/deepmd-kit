import os,sys,platform
import numpy as np
import unittest
from deepmd.env import tf

from deepmd.env import GLOBAL_TF_FLOAT_PRECISION
from deepmd.env import GLOBAL_NP_FLOAT_PRECISION
from deepmd.env import GLOBAL_ENER_FLOAT_PRECISION
from deepmd.infer.ewald_recp import op_module
from deepmd.infer.ewald_recp import EwaldRecp

if GLOBAL_NP_FLOAT_PRECISION == np.float32 :
    global_default_fv_hh = 1e-2
    global_default_dw_hh = 1e-2
    global_default_places = 3
else :
    global_default_fv_hh = 1e-6
    global_default_dw_hh = 1e-4
    global_default_places = 5


class TestEwaldRecp (unittest.TestCase) :
    def setUp(self):
        boxl = 4.5 # NOTICE grid should not change before and after box pert...
        box_pert = 0.2
        self.natoms = 16
        self.nframes = 2
        self.ewald_h = 1
        self.ewald_beta = 1
        self.dbox = []
        self.dcoord = []
        self.rcoord = []
        self.dcharge = []
        for ii in range(self.nframes):
            # box
            box = np.eye(3) * boxl
            box[1][1] += 1
            box[2][2] += 2
            box += np.random.random([3,3]) * box_pert
            box = 0.5 * (box + box.T)
            self.dbox.append(box)
            # scaled 
            coord = np.random.random([self.natoms, 3])
            self.rcoord.append(coord)
            # real coords
            self.dcoord.append(np.matmul(coord, box))
            # charge
            dcharge = np.random.random([self.natoms])
            dcharge -= np.average(dcharge)
            assert(np.abs(np.sum(self.dcharge) - 0) < 1e-12)
            self.dcharge.append(dcharge)
        self.dbox = np.array(self.dbox).reshape([self.nframes, 9])
        self.rcoord = np.array(self.rcoord).reshape([self.nframes, 3*self.natoms])
        self.dcoord = np.array(self.dcoord).reshape([self.nframes, 3*self.natoms])
        self.dcharge = np.array(self.dcharge).reshape([self.nframes, self.natoms])
        # place holders
        self.coord      = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name='t_coord')
        self.charge     = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name='t_charge')
        self.box        = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name='t_box')
        self.nloc    = tf.placeholder(tf.int32, [1], name = "t_nloc")        

    def test_py_interface(self):
        hh = 1e-4
        places = 4
        sess = tf.Session()
        t_energy, t_force, t_virial \
            = op_module.ewald_recp(self.coord, self.charge, self.nloc, self.box, 
                                   ewald_h = self.ewald_h,
                                   ewald_beta = self.ewald_beta)
        [e, f, v] = sess.run([t_energy, t_force, t_virial], 
                           feed_dict = {
                               self.coord:  self.dcoord.reshape([-1]),
                               self.charge: self.dcharge.reshape([-1]),
                               self.box:    self.dbox.reshape([-1]),
                               self.nloc:   [self.natoms],
                           })
        er = EwaldRecp(self.ewald_h, self.ewald_beta)
        e1, f1, v1 = er.eval(self.dcoord, self.dcharge, self.dbox)        
        for ff in range(self.nframes):
            self.assertAlmostEqual(e[ff], e1[ff], 
                                   places = places,
                                   msg = "frame %d energy failed" % (ff))
            for idx in range(self.natoms):
                for dd in range(3):
                    self.assertAlmostEqual(f[ff, idx*3+dd], f1[ff,idx*3+dd], 
                                           places = places,
                                           msg = "frame %d force component [%d,%d] failed" % (ff, idx, dd))
            for d0 in range(3):
                for d1 in range(3):
                    self.assertAlmostEqual(v[ff, d0*3+d1], v[ff,d0*3+d1], 
                                           places = places,
                                           msg = "frame %d virial component [%d,%d] failed" % (ff, d0, d1))



    def test_force(self):
        hh = 1e-4
        places = 6
        sess = tf.Session()
        t_energy, t_force, t_virial \
            = op_module.ewald_recp(self.coord, self.charge, self.nloc, self.box, 
                                   ewald_h = self.ewald_h,
                                   ewald_beta = self.ewald_beta)
        [force] = sess.run([t_force], 
                           feed_dict = {
                               self.coord:  self.dcoord.reshape([-1]),
                               self.charge: self.dcharge.reshape([-1]),
                               self.box:    self.dbox.reshape([-1]),
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
                                       self.coord:  dcoordp.reshape([-1]),
                                       self.charge: self.dcharge.reshape([-1]),
                                       self.box:    self.dbox.reshape([-1]),
                                       self.nloc:   [self.natoms],
                                   })                                
                energym = sess.run([t_energy], 
                                   feed_dict = {
                                       self.coord:  dcoordm.reshape([-1]),
                                       self.charge: self.dcharge.reshape([-1]),
                                       self.box:    self.dbox.reshape([-1]),
                                       self.nloc:   [self.natoms],
                                   })
                c_force = -(energyp[0] - energym[0]) / (2*hh)
                for ff in range(self.nframes):
                    self.assertAlmostEqual(c_force[ff], force[ff,idx*3+dd], 
                                           places = places,
                                           msg = "frame %d force component [%d,%d] failed" % (ff, idx, dd))


    def test_virial(self):
        hh = 1e-4
        places = 6
        sess = tf.Session()
        t_energy, t_force, t_virial \
            = op_module.ewald_recp(self.coord, self.charge, self.nloc, self.box, 
                                   ewald_h = self.ewald_h,
                                   ewald_beta = self.ewald_beta)
        [virial] = sess.run([t_virial], 
                           feed_dict = {
                               self.coord:  self.dcoord.reshape([-1]),
                               self.charge: self.dcharge.reshape([-1]),
                               self.box:    self.dbox.reshape([-1]),
                               self.nloc:   [self.natoms],
                           })

        from scipy.stats import ortho_group

        

        self.dbox3 = np.reshape(self.dbox, [self.nframes, 3,3])
        self.drbox3 = np.linalg.inv(self.dbox3)
        # print(np.matmul(self.dbox3, self.drbox3))
        # print(np.matmul(self.drbox3, self.dbox3))
        self.dcoord3 = np.reshape(self.dcoord, [self.nframes, self.natoms, 3])
        self.rcoord3 = np.matmul(self.dcoord3, self.drbox3)
        # print(np.linalg.norm(self.dcoord - np.matmul(self.rcoord3, self.dbox3).reshape([self.nframes,-1])))
        # print(np.matmul(self.dcoord3, self.drbox3))
        # print('check rcoord ', np.linalg.norm(self.rcoord3 - self.rcoord.reshape([self.nframes, self.natoms, 3])))

        num_deriv = np.zeros([self.nframes,3,3])
        for ii in range(3):
            for jj in range(3):
                dbox3p = np.copy(self.dbox3)
                dbox3m = np.copy(self.dbox3)
                dbox3p[:,ii,jj] = self.dbox3[:,ii,jj] + hh
                dbox3m[:,ii,jj] = self.dbox3[:,ii,jj] - hh
                dboxp = np.reshape(dbox3p, [-1,9])
                dboxm = np.reshape(dbox3m, [-1,9])
                dcoord = self.dcoord
                dcoord3p = np.matmul(self.rcoord3, dbox3p)
                dcoord3m = np.matmul(self.rcoord3, dbox3m)
                dcoordp = np.reshape(dcoord3p, [self.nframes,-1])
                dcoordm = np.reshape(dcoord3m, [self.nframes,-1])
                energyp = sess.run([t_energy],
                                   feed_dict = {
                                       self.coord:  dcoordp.reshape([-1]),
                                       self.charge: self.dcharge.reshape([-1]),
                                       self.box:    dboxp.reshape([-1]),
                                       self.nloc:   [self.natoms],
                                   })
                energym = sess.run([t_energy], 
                                   feed_dict = {
                                       self.coord:  dcoordm.reshape([-1]),
                                       self.charge: self.dcharge.reshape([-1]),
                                       self.box:    dboxm.reshape([-1]),
                                       self.nloc:   [self.natoms],
                                   })
                num_deriv[:,ii,jj] = -(energyp[0] - energym[0]) / (2.*hh)
        num_deriv_t = np.transpose(num_deriv, [0,2,1])
        t_esti = np.matmul(num_deriv_t, self.dbox3)
        # # t_esti = np.matmul(num_deriv, self.dbox3)
        # print(num_deriv[0])
        # print(t_esti[0])
        # # print(0.5 * (t_esti[0] + t_esti[0].T))
        # print(virial[0].reshape([3,3]))
        # # print(0.5 * (t_esti[0] + t_esti[0].T) - virial[0].reshape([3,3]))
        # print(0.5 * (t_esti[0] + t_esti[0]) - virial[0].reshape([3,3]))
        # print(0.5 * (t_esti[0] + t_esti[0].T) - virial[0].reshape([3,3]))        
        for ff in range(self.nframes):
            for ii in range(3):
                for jj in range(3):                
                    self.assertAlmostEqual(t_esti[ff][ii][jj], virial[ff,ii*3+jj], 
                                           places = places,
                                           msg = "frame %d virial component [%d,%d] failed" % (ff, ii, jj))
            
                



