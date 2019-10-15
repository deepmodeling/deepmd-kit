import os, sys, dpdata
import numpy as np

from deepmd.env import tf
from deepmd.RunOptions import global_tf_float_precision
from deepmd.RunOptions import global_np_float_precision
from deepmd.RunOptions import global_ener_float_precision

if global_np_float_precision == np.float32 :
    global_default_fv_hh = 1e-2
    global_default_dw_hh = 1e-2
    global_default_places = 3
else :
    global_default_fv_hh = 1e-6
    global_default_dw_hh = 1e-4
    global_default_places = 5

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
                  seed = 1) :
        coord = [[0.0, 0.0, 0.1], [1.1, 0.0, 0.1], [0.0, 1.1, 0.1], 
                 [4.0, 0.0, 0.0], [5.1, 0.0, 0.0], [4.0, 1.1, 0.0]]
        self.coord = np.array(coord)
        np.random.seed(seed)
        self.coord += rand_pert * np.random.random(self.coord.shape)
        self.fparam = np.array([[0.1, 0.2]])
        self.aparam = np.tile(self.fparam, [1, 6])
        self.atype = np.array([0, 1, 1, 0, 1, 1], dtype = int)
        self.cell = 20 * np.eye(3)
        self.nframes = 1
        self.coord = self.coord.reshape([self.nframes, -1])
        self.cell = self.cell.reshape([self.nframes, -1])
        self.natoms = len(self.atype)        
        self.idx_map = np.lexsort ((np.arange(self.natoms), self.atype))
        self.coord = self.coord.reshape([1, -1, 3])
        self.coord = self.coord[:,self.idx_map,:]
        self.coord = self.coord.reshape([1, -1])        
        self.atype = self.atype[self.idx_map]
        self.datype = np.tile(self.atype, [self.nframes,1])
        
    def get_data(self) :
        return self.coord, self.cell, self.datype

    def get_natoms (self) :
        ret = [self.natoms, self.natoms]
        for ii in range(max(self.atype) + 1) :
            ret.append(np.sum(self.atype == ii))        
        return np.array(ret, dtype = np.int32)
    
    def get_ntypes(self) :
        return max(self.atype) + 1

    def get_test_box_data (self,
                           hh) :
        coord0_, box0_, type0_ = self.get_data()
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
    # cmp e0, f0
    [energy, force] = inter.sess.run ([t_energy, t_force], 
                                     feed_dict = {
                                         inter.coord:     dcoord,
                                         inter.box:       dbox,
                                         inter.type:      dtype,
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
                                         inter.tnatoms:   inter.natoms}
            )
            [enerm] = inter.sess.run ([t_energy], 
                                     feed_dict = {
                                         inter.coord:     dcoordm,
                                         inter.box:       dbox,
                                         inter.type:      dtype,
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
    dcoord, dbox, dtype = inter.data.get_test_box_data(hh)
    # cmp e, f, v
    [energy, force, virial] \
        = inter.sess.run ([t_energy, t_force, t_virial], 
                          feed_dict = {
                              inter.coord:     dcoord,
                              inter.box:       dbox,
                              inter.type:      dtype,
                              inter.tnatoms:   inter.natoms}
        )
    # check
    ana_vir3 = (virial[0][0] + virial[0][4] + virial[0][8])/3. / comp_vol(dbox[0])
    num_vir3 = -(energy[1] - energy[2]) / (comp_vol(dbox[1]) - comp_vol(dbox[2]))
    testCase.assertAlmostEqual(ana_vir3, num_vir3, places=places)
    vir_idx = [0, 4, 8]
    for dd in range (3) :
        ana_v = (virial[0][vir_idx[dd]] / comp_vol(dbox[0]))
        idx = 2 * (dd+1) + 1
        num_v = ( -(energy[idx] - energy[idx+1]) / (comp_vol(dbox[idx]) - comp_vol(dbox[idx+1])) )
        testCase.assertAlmostEqual(ana_v, num_v, places=places)


def force_dw_test (inter, 
                   testCase,
                   places = global_default_places,
                   hh = global_default_dw_hh, 
                   suffix = '') :
    dcoord, dbox, dtype = inter.data.get_data()
    feed_dict_test0 = {
        inter.coord:     dcoord,
        inter.box:       dbox,
        inter.type:      dtype,
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
    feed_dict_test0 = {
        inter.coord:     dcoord,
        inter.box:       dbox,
        inter.type:      dtype,
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
