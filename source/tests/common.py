import os, sys, dpdata
import numpy as np
import pathlib

from deepmd.env import tf
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION
from deepmd.env import GLOBAL_NP_FLOAT_PRECISION
from deepmd.env import GLOBAL_ENER_FLOAT_PRECISION
from deepmd.common import j_loader as dp_j_loader

if GLOBAL_NP_FLOAT_PRECISION == np.float32 :
    global_default_fv_hh = 1e-2
    global_default_dw_hh = 1e-2
    global_default_places = 3
else :
    global_default_fv_hh = 1e-5
    global_default_dw_hh = 1e-4
    global_default_places = 5

tests_path = pathlib.Path(__file__).parent.absolute()

def j_loader(filename):
    return dp_j_loader(tests_path/filename)

def gen_data() :
    tmpdata = Data(rand_pert = 0.1, seed = 1)
    sys = dpdata.LabeledSystem()
    sys.data['atom_names'] = ['foo', 'bar']
    sys.data['coords'] = tmpdata.coord
    sys.data['atom_types'] = tmpdata.atype
    sys.data['cells'] = tmpdata.cell
    nframes = tmpdata.nframes
    natoms = tmpdata.natoms
    sys.data['coords'] = sys.data['coords'].reshape([nframes,natoms,3])
    sys.data['cells'] = sys.data['cells'].reshape([nframes,3,3])
    sys.data['energies'] = np.zeros([nframes,1])
    sys.data['forces'] = np.zeros([nframes,natoms,3])
    sys.to_deepmd_npy('system', prec=np.float64)    
    np.save('system/set.000/fparam.npy', tmpdata.fparam)
    np.save('system/set.000/aparam.npy', tmpdata.aparam.reshape([nframes, natoms, 2]))

class Data():
    def __init__ (self, 
                  rand_pert = 0.1, 
                  seed = 1, 
                  box_scale = 20) :
        coord = [[0.0, 0.0, 0.1], [1.1, 0.0, 0.1], [0.0, 1.1, 0.1], 
                 [4.0, 0.0, 0.0], [5.1, 0.0, 0.0], [4.0, 1.1, 0.0]]
        self.nframes = 1
        self.coord = np.array(coord)
        self.coord = self._copy_nframes(self.coord)
        np.random.seed(seed)
        self.coord += rand_pert * np.random.random(self.coord.shape)
        self.fparam = np.array([[0.1, 0.2]])
        self.aparam = np.tile(self.fparam, [1, 6])
        self.fparam = self._copy_nframes(self.fparam)
        self.aparam = self._copy_nframes(self.aparam)
        self.atype = np.array([0, 1, 1, 0, 1, 1], dtype = int)
        self.cell = box_scale * np.eye(3)
        self.cell = self._copy_nframes(self.cell)
        self.coord = self.coord.reshape([self.nframes, -1])
        self.cell = self.cell.reshape([self.nframes, -1])
        self.natoms = len(self.atype)        
        self.idx_map = np.lexsort ((np.arange(self.natoms), self.atype))
        self.coord = self.coord.reshape([self.nframes, -1, 3])
        self.coord = self.coord[:,self.idx_map,:]
        self.coord = self.coord.reshape([self.nframes, -1])        
        self.efield = np.random.random(self.coord.shape)
        self.atype = self.atype[self.idx_map]
        self.datype = self._copy_nframes(self.atype)

    def _copy_nframes(self, xx):
        return np.tile(xx, [self.nframes, 1])
        
    def get_data(self) :
        return self.coord, self.cell, self.datype

    def get_natoms (self) :
        ret = [self.natoms, self.natoms]
        for ii in range(max(self.atype) + 1) :
            ret.append(np.sum(self.atype == ii))        
        return np.array(ret, dtype = np.int32)
    
    def get_ntypes(self) :
        return max(self.atype) + 1

    # def get_test_box_data (self,
    #                        hh) :
    #     coord0_, box0_, type0_ = self.get_data()
    #     coord0 = coord0_[0]
    #     box0 = box0_[0]
    #     type0 = type0_[0]
    #     nc = np.array( [coord0, coord0*(1+hh), coord0*(1-hh)] )
    #     nb = np.array( [box0, box0*(1+hh), box0*(1-hh)] )
    #     nt = np.array( [type0, type0, type0] )
    #     for dd in range(3) :
    #         tmpc = np.copy (coord0)
    #         tmpb = np.copy (box0)
    #         tmpc = np.reshape(tmpc, [-1, 3])
    #         tmpc [:,dd] *= (1+hh)
    #         tmpc = np.reshape(tmpc, [-1])
    #         tmpb = np.reshape(tmpb, [-1, 3])
    #         tmpb [dd,:] *= (1+hh)
    #         tmpb = np.reshape(tmpb, [-1])
    #         nc = np.append (nc, [tmpc], axis = 0)
    #         nb = np.append (nb, [tmpb], axis = 0)
    #         nt = np.append (nt, [type0], axis = 0)
    #         tmpc = np.copy (coord0)
    #         tmpb = np.copy (box0)
    #         tmpc = np.reshape(tmpc, [-1, 3])
    #         tmpc [:,dd] *= (1-hh)
    #         tmpc = np.reshape(tmpc, [-1])
    #         tmpb = np.reshape(tmpb, [-1, 3])
    #         tmpb [dd,:] *= (1-hh)
    #         tmpb = np.reshape(tmpb, [-1])
    #         nc = np.append (nc, [tmpc], axis = 0)
    #         nb = np.append (nb, [tmpb], axis = 0)
    #         nt = np.append (nt, [type0], axis = 0)
    #     return nc, nb, nt

    def get_test_box_data (self,
                           hh, 
                           rand_pert = 0.1) :
        coord0_, box0_, type0_ = self.get_data()
        coord = coord0_[0]
        box = box0_[0]
        box += rand_pert * np.random.random(box.shape)
        atype = type0_[0]
        nframes = 1
        natoms = coord.size // 3
        box3 = np.reshape(box, [nframes, 3,3])
        rbox3 = np.linalg.inv(box3)
        coord3 = np.reshape(coord, [nframes, natoms, 3])
        rcoord3 = np.matmul(coord3, rbox3)
        
        all_coord = [coord.reshape([nframes, natoms*3])]
        all_box = [box.reshape([nframes,9])]
        all_atype = [atype]
        all_efield = [self.efield]
        for ii in range(3):
            for jj in range(3):
                box3p = np.copy(box3)
                box3m = np.copy(box3)
                box3p[:,ii,jj] = box3[:,ii,jj] + hh
                box3m[:,ii,jj] = box3[:,ii,jj] - hh
                boxp = np.reshape(box3p, [-1,9])
                boxm = np.reshape(box3m, [-1,9])
                coord3p = np.matmul(rcoord3, box3p)
                coord3m = np.matmul(rcoord3, box3m)
                coordp = np.reshape(coord3p, [nframes,-1])
                coordm = np.reshape(coord3m, [nframes,-1])
                all_coord.append(coordp)
                all_coord.append(coordm)
                all_box.append(boxp)
                all_box.append(boxm)
                all_atype.append(atype)
                all_atype.append(atype)
                all_efield.append(self.efield)
                all_efield.append(self.efield)
        all_coord = np.reshape(all_coord, [-1, natoms * 3])
        all_box = np.reshape(all_box, [-1, 9])
        all_atype = np.reshape(all_atype, [-1, natoms])        
        all_efield = np.reshape(all_efield, [-1, natoms * 3])        
        return all_coord, all_box, all_atype, all_efield


def force_test (inter, 
                testCase, 
                places = global_default_places, 
                hh = global_default_fv_hh, 
                suffix = '') :
    # set weights
    w0 = np.ones (inter.ndescrpt)
    inter.net_w_i = np.copy(w0)
    # make network
    t_energy, t_force, t_virial \
        = inter.comp_ef (inter.coord, inter.box, inter.type, inter.tnatoms, name = "test_f" + suffix)
    inter.sess.run (tf.global_variables_initializer())
    # get data
    dcoord, dbox, dtype = inter.data.get_data ()
    defield = inter.data.efield
    # cmp e0, f0
    [energy, force] = inter.sess.run ([t_energy, t_force], 
                                     feed_dict = {
                                         inter.coord:     dcoord,
                                         inter.box:       dbox,
                                         inter.type:      dtype,
                                         inter.efield:    defield,
                                         inter.tnatoms:   inter.natoms}
    )
    # dim force
    sel_idx = np.arange(inter.natoms[0])    
    for idx in sel_idx:
        for dd in range(3):
            dcoordp = np.copy(dcoord)
            dcoordm = np.copy(dcoord)
            dcoordp[0,idx*3+dd] = dcoord[0,idx*3+dd] + hh
            dcoordm[0,idx*3+dd] = dcoord[0,idx*3+dd] - hh
            [enerp] = inter.sess.run ([t_energy], 
                                     feed_dict = {
                                         inter.coord:     dcoordp,
                                         inter.box:       dbox,
                                         inter.type:      dtype,
                                         inter.efield:    defield,
                                         inter.tnatoms:   inter.natoms}
            )
            [enerm] = inter.sess.run ([t_energy], 
                                     feed_dict = {
                                         inter.coord:     dcoordm,
                                         inter.box:       dbox,
                                         inter.type:      dtype,
                                         inter.efield:    defield,
                                         inter.tnatoms:   inter.natoms}
            )
            c_force = -(enerp[0] - enerm[0]) / (2*hh)
            testCase.assertAlmostEqual(c_force, force[0,idx*3+dd], 
                                       places = places,
                                       msg = "force component [%d,%d] failed" % (idx, dd))

def comp_vol (box) : 
    return np.linalg.det (np.reshape(box, (3,3)))

def virial_test (inter, 
                 testCase, 
                 places = global_default_places, 
                 hh = global_default_fv_hh, 
                 suffix = '') :
    # set weights
    w0 = np.ones (inter.ndescrpt)
    inter.net_w_i = np.copy(w0)
    # make network
    t_energy, t_force, t_virial \
        = inter.comp_ef (inter.coord, inter.box, inter.type, inter.tnatoms, name = "test_v" + suffix)
    inter.sess.run (tf.global_variables_initializer())
    # get data
    dcoord, dbox, dtype, defield = inter.data.get_test_box_data(hh)
    # cmp e, f, v
    [energy, force, virial] \
        = inter.sess.run ([t_energy, t_force, t_virial], 
                          feed_dict = {
                              inter.coord:     dcoord,
                              inter.box:       dbox,
                              inter.type:      dtype,
                              inter.efield:    defield,
                              inter.tnatoms:   inter.natoms}
        )
    ana_vir = virial[0].reshape([3,3])
    num_vir = np.zeros([3,3])
    for ii in range(3):
        for jj in range(3):
            ep = energy[1+(ii*3+jj)*2+0]
            em = energy[1+(ii*3+jj)*2+1]
            num_vir[ii][jj] = -(ep - em) / (2.*hh)
    num_vir = np.transpose(num_vir, [1,0])    
    box3 = dbox[0].reshape([3,3])
    num_vir = np.matmul(num_vir, box3)
    for ii in range(3):
        for jj in range(3):
            testCase.assertAlmostEqual(ana_vir[ii][jj], num_vir[ii][jj],
                                       places=places, 
                                       msg = 'virial component %d %d ' % (ii,jj))
    


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
        
    t_ll, t_dw = inter.comp_f_dw (inter.coord, inter.box, inter.type, inter.tnatoms, name = "f_dw_test_0" + suffix)
    inter.sess.run (tf.global_variables_initializer())
    ll_0 = inter.sess.run (t_ll, feed_dict = feed_dict_test0)
    dw_0 = inter.sess.run (t_dw, feed_dict = feed_dict_test0)
        
    absolut_e = []
    relativ_e = []
    test_list = range (inter.ndescrpt) 
    ntest = 3
    if inter.sel_a[0] != 0:
        test_list = np.concatenate((np.arange(0,ntest), np.arange(inter.sel_a[0]*4, inter.sel_a[0]*4+ntest)))
    else :
        test_list = np.arange(0,ntest)

    for ii in test_list:
        inter.net_w_i = np.copy (w0)
        inter.net_w_i[ii] += hh
        t_ll, t_dw = inter.comp_f_dw (inter.coord, inter.box, inter.type, inter.tnatoms, name = "f_dw_test_" + str(ii*2+1) + suffix)
        inter.sess.run (tf.global_variables_initializer())
        ll_1 = inter.sess.run (t_ll, feed_dict = feed_dict_test0)
        inter.net_w_i[ii] -= 2. * hh
        t_ll, t_dw = inter.comp_f_dw (inter.coord, inter.box, inter.type, inter.tnatoms, name = "f_dw_test_" + str(ii*2+2) + suffix)
        inter.sess.run (tf.global_variables_initializer())
        ll_2 = inter.sess.run (t_ll, feed_dict = feed_dict_test0)
        num_v = (ll_1 - ll_2) / (2. * hh)
        ana_v = dw_0[ii]
        diff = np.abs (num_v - ana_v)
        # print(ii, num_v, ana_v)
        testCase.assertAlmostEqual(num_v, ana_v, places = places)


def virial_dw_test (inter, 
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

    t_ll, t_dw = inter.comp_v_dw (inter.coord, inter.box, inter.type, inter.tnatoms, name = "v_dw_test_0" + suffix)
    inter.sess.run (tf.global_variables_initializer())
    ll_0 = inter.sess.run (t_ll, feed_dict = feed_dict_test0)
    dw_0 = inter.sess.run (t_dw, feed_dict = feed_dict_test0)
        
    absolut_e = []
    relativ_e = []
    test_list = range (inter.ndescrpt) 
    ntest = 3
    if inter.sel_a[0] != 0 :
        test_list = np.concatenate((np.arange(0,ntest), np.arange(inter.sel_a[0]*4, inter.sel_a[0]*4+ntest)))
    else :
        test_list = np.arange(0,ntest)
        
    for ii in test_list:
        inter.net_w_i = np.copy (w0)
        inter.net_w_i[ii] += hh
        t_ll, t_dw = inter.comp_v_dw (inter.coord, inter.box, inter.type, inter.tnatoms, name = "v_dw_test_" + str(ii*2+1) + suffix)
        inter.sess.run (tf.global_variables_initializer())
        ll_1 = inter.sess.run (t_ll, feed_dict = feed_dict_test0)
        inter.net_w_i[ii] -= 2. * hh
        t_ll, t_dw = inter.comp_v_dw (inter.coord, inter.box, inter.type, inter.tnatoms, name = "v_dw_test_" + str(ii*2+2) + suffix)
        inter.sess.run (tf.global_variables_initializer())
        ll_2 = inter.sess.run (t_ll, feed_dict = feed_dict_test0)
        num_v = (ll_1 - ll_2) / (2. * hh)
        ana_v = dw_0[ii]
        testCase.assertAlmostEqual(num_v, ana_v, places = places)

