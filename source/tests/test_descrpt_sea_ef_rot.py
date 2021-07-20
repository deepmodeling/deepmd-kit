import os,sys
import numpy as np
import unittest

from deepmd.env import tf
from tensorflow.python.framework import ops
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION
from deepmd.env import GLOBAL_NP_FLOAT_PRECISION
from deepmd.env import GLOBAL_ENER_FLOAT_PRECISION

from deepmd.env import op_module
from deepmd.descriptor import DescrptSeA
from deepmd.descriptor import DescrptSeAEfLower

class TestEfRot(unittest.TestCase):
    def setUp(self):
        self.sess = tf.Session()
        self.natoms = [5, 5, 2, 3]
        self.ntypes = 2
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
        # no pbc
        self.default_mesh = np.array([], dtype = np.int32)
        # make place holder
        self.coord      = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None, self.natoms[0] * 3], name='t_coord')
        self.efield     = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None, self.natoms[0] * 3], name='t_efield')
        self.box        = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None, 9], name='t_box')
        self.type       = tf.placeholder(tf.int32,   [None, self.natoms[0]], name = "t_type")
        self.tnatoms    = tf.placeholder(tf.int32,   [None], name = "t_natoms")


    def _normalize_3d(self, a):
        na = tf.norm(a, axis = 1)
        na = tf.tile(tf.reshape(na, [-1,1]), tf.constant([1, 3]))
        b = tf.divide(a, na)
        return b

    def build_efv(self,
                  dcoord,
                  dbox,
                  dtype,
                  tnatoms,
                  name, 
                  op,
                  reuse = None):
        efield = tf.reshape(self.efield, [-1, 3])
        efield = self._normalize_3d(efield)
        efield = tf.reshape(efield, [-1, tnatoms[0] * 3])
        if op != op_module.prod_env_mat_a :            
            descrpt = DescrptSeAEfLower(op, **{'sel':self.sel_a, 'rcut': 6, 'rcut_smth' : 5.5})
        else:
            descrpt = DescrptSeA(**{'sel':self.sel_a, 'rcut': 6, 'rcut_smth' : 0.5})
        dout = descrpt.build(dcoord,
                             dtype,
                             tnatoms,
                             dbox,
                             tf.constant(self.default_mesh),
                             {'efield': efield},
                             suffix = name,
                             reuse = reuse)
        dout = tf.reshape(dout, [-1, descrpt.get_dim_out()])
        atom_ener = tf.reduce_sum(dout, axis = 1)
        atom_ener_reshape = tf.reshape(atom_ener, [-1, self.natoms[0]])        
        energy = tf.reduce_sum (atom_ener_reshape, axis = 1)        
        force, virial, atom_vir \
            = descrpt.prod_force_virial (atom_ener, tnatoms)
        return energy, force, virial, atom_ener, atom_vir        

    def make_test_data(self, nframes):
        dcoord = np.random.random([nframes, self.natoms[0], 3])
        for ii in range(nframes):
            dcoord[ii, :, :] = dcoord[ii, :, :] - np.tile(dcoord[ii, 0, :], [self.natoms[0], 1])
        dcoord = dcoord.reshape([nframes, -1])
        one_box = np.eye(3).reshape([1, 9])
        dbox = np.tile(one_box, [nframes, 1])
        one_type = []
        for ii in range(2, 2+self.ntypes):
            one_type = one_type + [ii-2 for jj in range(self.natoms[ii])]
        np.random.shuffle(one_type)
        one_type = np.array(one_type, dtype = int).reshape([1,-1])
        dtype = np.tile(one_type, [nframes, 1])
        defield = np.random.random(dcoord.shape)
        return dcoord, dbox, dtype, defield

    def rotate_mat(self, axis_, theta):
        axis = axis_ / np.linalg.norm(axis_)
        onemcc = (1. - np.cos(theta))
        cc = np.cos(theta)
        ss = np.sin(theta)
        r = [
            cc + axis[0]**2 * onemcc,
            axis[0] * axis[1] * onemcc - axis[2] * ss,
            axis[0] * axis[2] * onemcc + axis[1] * ss,
            axis[0] * axis[1] * onemcc + axis[2] * ss,
            cc + axis[1]**2 * onemcc,
            axis[1] * axis[2] * onemcc - axis[0] * ss,
            axis[0] * axis[2] * onemcc - axis[1] * ss,
            axis[1] * axis[2] * onemcc + axis[0] * ss,
            cc + axis[2]**2 * onemcc]
        return np.array(r).reshape(3, 3)
            
    def test_rot_axis(self, suffix=''):
        suffix = '_axis'
        t_p_e, t_p_f, t_p_f, t_p_ae, t_p_av \
            = self.build_efv (self.coord, self.box, self.type, self.tnatoms, name = "test_rot" + suffix, op = op_module.descrpt_se_a_ef_para)
        t_v_e, t_v_f, t_v_f, t_v_ae, t_v_av \
            = self.build_efv (self.coord, self.box, self.type, self.tnatoms, name = "test_rot" + suffix, op = op_module.descrpt_se_a_ef_vert, reuse = True)
        t_e, t_f, t_f, t_ae, t_av \
            = self.build_efv (self.coord, self.box, self.type, self.tnatoms, name = "test_rot" + suffix, op = op_module.prod_env_mat_a, reuse = True)
        self.sess.run (tf.global_variables_initializer())

        np.random.seed(0)        
        # make test data
        nframes = 2      
        dcoord, dbox, dtype, defield = self.make_test_data(nframes)
        [p_ae0] = self.sess.run ([t_p_ae], 
                                feed_dict = {
                                    self.coord:     dcoord,
                                    self.box:       dbox,
                                    self.type:      dtype,
                                    self.efield:    defield,
                                    self.tnatoms:   self.natoms})
        [v_ae0] = self.sess.run ([t_v_ae], 
                                feed_dict = {
                                    self.coord:     dcoord,
                                    self.box:       dbox,
                                    self.type:      dtype,
                                    self.efield:    defield,
                                    self.tnatoms:   self.natoms})
        [ae0] = self.sess.run ([t_ae], 
                                feed_dict = {
                                    self.coord:     dcoord,
                                    self.box:       dbox,
                                    self.type:      dtype,
                                    self.efield:    defield,
                                    self.tnatoms:   self.natoms})
        # print(p_ae0)
        # print(v_ae0)
        # print(ae0)

        for kk in range(0, self.natoms[0]):
            # print(f0)
            theta = 45. / 180. * np.pi
            rr0 = self.rotate_mat(defield[0][kk*3:kk*3+3], theta)
            # rr0 = self.rotate_mat([0, 0, 1], theta)
            rr1 = self.rotate_mat(defield[1][kk*3:kk*3+3], theta)
            # print(rr0, np.matmul(rr0, rr0.T), np.matmul(rr0.T, rr0))
            # print(rr1)
            dcoord_ = np.copy(dcoord).reshape([nframes, -1, 3])
            dcoord0 = np.matmul(dcoord_[0], rr0)
            dcoord1 = np.matmul(dcoord_[1], rr1)
            new_dcoord = np.concatenate([dcoord0, dcoord1], axis = 0).reshape([nframes, -1])
            defield_ = np.copy(defield).reshape([nframes, -1, 3])
            defield0 = np.matmul(defield_[0], rr0)
            defield1 = np.matmul(defield_[1], rr1)
            new_defield = np.concatenate([defield0, defield1], axis = 0).reshape([nframes, -1])

            [p_ae1] = self.sess.run ([t_p_ae], 
                                    feed_dict = {
                                        self.coord:     new_dcoord,
                                        self.box:       dbox,
                                        self.type:      dtype,
                                        self.efield:    defield,
                                        self.tnatoms:   self.natoms})
            [v_ae1] = self.sess.run ([t_v_ae], 
                                    feed_dict = {
                                        self.coord:     new_dcoord,
                                        self.box:       dbox,
                                        self.type:      dtype,
                                        self.efield:    defield,
                                        self.tnatoms:   self.natoms})
            [ae1] = self.sess.run ([t_ae], 
                                    feed_dict = {
                                        self.coord:     new_dcoord,
                                        self.box:       dbox,
                                        self.type:      dtype,
                                        self.efield:    defield,
                                        self.tnatoms:   self.natoms})
            for ii in range(0, self.natoms[0]):
                for jj in range(0, self.natoms[0]):
                    diff = dcoord[0][3*jj:3*jj+3] - dcoord[0][3*ii:3*ii+3]
                    dot = np.dot(dcoord[0][3*jj:3*jj+3] , dcoord[0][3*ii:3*ii+3])
                    diff1 = new_dcoord[0][3*jj:3*jj+3] - new_dcoord[0][3*ii:3*ii+3]
                    dot1 = np.dot(new_dcoord[0][3*jj:3*jj+3] , new_dcoord[0][3*ii:3*ii+3])
                    assert(np.abs(np.linalg.norm(diff) - np.linalg.norm(diff1)) < 1e-15)
                    assert(np.abs(dot - dot1) < 1e-15)

            for ii in range(1, self.natoms[0]):
                if ii == kk:
                    self.assertAlmostEqual(p_ae0[ii], p_ae1[ii])
                    self.assertAlmostEqual(v_ae0[ii], v_ae1[ii])
                else:
                    self.assertNotAlmostEqual(p_ae0[ii], p_ae1[ii])
                    self.assertNotAlmostEqual(v_ae0[ii], v_ae1[ii])


    def test_rot_diff_axis(self, suffix=''):
        suffix = '_diff_axis'
        t_p_e, t_p_f, t_p_f, t_p_ae, t_p_av \
            = self.build_efv (self.coord, self.box, self.type, self.tnatoms, name = "test_rot" + suffix, op = op_module.descrpt_se_a_ef_para)
        t_v_e, t_v_f, t_v_f, t_v_ae, t_v_av \
            = self.build_efv (self.coord, self.box, self.type, self.tnatoms, name = "test_rot" + suffix, op = op_module.descrpt_se_a_ef_vert, reuse = True)
        t_e, t_f, t_f, t_ae, t_av \
            = self.build_efv (self.coord, self.box, self.type, self.tnatoms, name = "test_rot" + suffix, op = op_module.prod_env_mat_a, reuse = True)
        self.sess.run (tf.global_variables_initializer())

        np.random.seed(0)        
        # make test data
        nframes = 2      
        dcoord, dbox, dtype, defield = self.make_test_data(nframes)
        [p_ae0] = self.sess.run ([t_p_ae], 
                                feed_dict = {
                                    self.coord:     dcoord,
                                    self.box:       dbox,
                                    self.type:      dtype,
                                    self.efield:    defield,
                                    self.tnatoms:   self.natoms})
        [v_ae0] = self.sess.run ([t_v_ae], 
                                feed_dict = {
                                    self.coord:     dcoord,
                                    self.box:       dbox,
                                    self.type:      dtype,
                                    self.efield:    defield,
                                    self.tnatoms:   self.natoms})
        [ae0] = self.sess.run ([t_ae], 
                                feed_dict = {
                                    self.coord:     dcoord,
                                    self.box:       dbox,
                                    self.type:      dtype,
                                    self.efield:    defield,
                                    self.tnatoms:   self.natoms})

        # print(f0)
        theta = 45. / 180. * np.pi
        rr0 = self.rotate_mat([0, 0, 1], theta)
        rr1 = self.rotate_mat([1, 0, 0], theta)
        dcoord_ = np.copy(dcoord).reshape([nframes, -1, 3])
        dcoord0 = np.matmul(dcoord_[0], rr0)
        dcoord1 = np.matmul(dcoord_[1], rr1)
        new_dcoord = np.concatenate([dcoord0, dcoord1], axis = 0).reshape([nframes, -1])
        defield_ = np.copy(defield).reshape([nframes, -1, 3])
        defield0 = np.matmul(defield_[0], rr0)
        defield1 = np.matmul(defield_[1], rr1)
        new_defield = np.concatenate([defield0, defield1], axis = 0).reshape([nframes, -1])

        [p_ae1] = self.sess.run ([t_p_ae], 
                                feed_dict = {
                                    self.coord:     new_dcoord,
                                    self.box:       dbox,
                                    self.type:      dtype,
                                    self.efield:    defield,
                                    self.tnatoms:   self.natoms})
        [v_ae1] = self.sess.run ([t_v_ae], 
                                feed_dict = {
                                    self.coord:     new_dcoord,
                                    self.box:       dbox,
                                    self.type:      dtype,
                                    self.efield:    defield,
                                    self.tnatoms:   self.natoms})
        [ae1] = self.sess.run ([t_ae], 
                                feed_dict = {
                                    self.coord:     new_dcoord,
                                    self.box:       dbox,
                                    self.type:      dtype,
                                    self.efield:    defield,
                                    self.tnatoms:   self.natoms})

        for ii in range(0, self.natoms[0]):
            self.assertNotAlmostEqual(p_ae0[ii], p_ae1[ii])
            self.assertNotAlmostEqual(v_ae0[ii], v_ae1[ii])

    def test_rot_field_corot(self, suffix=''):
        suffix = '_field_corot'
        t_p_e, t_p_f, t_p_f, t_p_ae, t_p_av \
            = self.build_efv (self.coord, self.box, self.type, self.tnatoms, name = "test_rot" + suffix, op = op_module.descrpt_se_a_ef_para)
        t_v_e, t_v_f, t_v_f, t_v_ae, t_v_av \
            = self.build_efv (self.coord, self.box, self.type, self.tnatoms, name = "test_rot" + suffix, op = op_module.descrpt_se_a_ef_vert, reuse = True)
        t_e, t_f, t_f, t_ae, t_av \
            = self.build_efv (self.coord, self.box, self.type, self.tnatoms, name = "test_rot" + suffix, op = op_module.prod_env_mat_a, reuse = True)
        self.sess.run (tf.global_variables_initializer())

        np.random.seed(0)        
        # make test data
        nframes = 2      
        dcoord, dbox, dtype, defield = self.make_test_data(nframes)
        [p_ae0] = self.sess.run ([t_p_ae], 
                                feed_dict = {
                                    self.coord:     dcoord,
                                    self.box:       dbox,
                                    self.type:      dtype,
                                    self.efield:    defield,
                                    self.tnatoms:   self.natoms})
        [v_ae0] = self.sess.run ([t_v_ae], 
                                feed_dict = {
                                    self.coord:     dcoord,
                                    self.box:       dbox,
                                    self.type:      dtype,
                                    self.efield:    defield,
                                    self.tnatoms:   self.natoms})
        [ae0] = self.sess.run ([t_ae], 
                                feed_dict = {
                                    self.coord:     dcoord,
                                    self.box:       dbox,
                                    self.type:      dtype,
                                    self.efield:    defield,
                                    self.tnatoms:   self.natoms})

        # print(f0)
        theta = 45. / 180. * np.pi
        rr0 = self.rotate_mat(defield[0][0:3], theta)
        rr1 = self.rotate_mat(defield[1][0:3], theta)
        dcoord_ = np.copy(dcoord).reshape([nframes, -1, 3])
        dcoord0 = np.matmul(dcoord_[0], rr0)
        dcoord1 = np.matmul(dcoord_[1], rr1)
        new_dcoord = np.concatenate([dcoord0, dcoord1], axis = 0).reshape([nframes, -1])
        defield_ = np.copy(defield).reshape([nframes, -1, 3])
        defield0 = np.matmul(defield_[0], rr0)
        defield1 = np.matmul(defield_[1], rr1)
        new_defield = np.concatenate([defield0, defield1], axis = 0).reshape([nframes, -1])

        [p_ae1] = self.sess.run ([t_p_ae], 
                                feed_dict = {
                                    self.coord:     new_dcoord,
                                    self.box:       dbox,
                                    self.type:      dtype,
                                    self.efield:    new_defield,
                                    self.tnatoms:   self.natoms})
        [v_ae1] = self.sess.run ([t_v_ae], 
                                feed_dict = {
                                    self.coord:     new_dcoord,
                                    self.box:       dbox,
                                    self.type:      dtype,
                                    self.efield:    new_defield,
                                    self.tnatoms:   self.natoms})
        [ae1] = self.sess.run ([t_ae], 
                                feed_dict = {
                                    self.coord:     new_dcoord,
                                    self.box:       dbox,
                                    self.type:      dtype,
                                    self.efield:    new_defield,
                                    self.tnatoms:   self.natoms})

        for ii in range(0, self.natoms[0]):
            self.assertAlmostEqual(p_ae0[ii], p_ae1[ii])
            self.assertAlmostEqual(v_ae0[ii], v_ae1[ii])
            self.assertAlmostEqual(ae0[ii], ae1[ii])
        


if __name__ == '__main__':
    unittest.main()
