import os, sys, dpdata, shutil
import numpy as np
import pathlib
from glob import glob

from deepmd.env import tf
from deepmd.env import GLOBAL_NP_FLOAT_PRECISION
from deepmd.common import j_loader as dp_j_loader
from deepmd.utils import random as dp_random
from deepmd.entrypoints.main import main

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

def del_data():
    if os.path.isdir('system'):
        shutil.rmtree('system')
    if os.path.isdir('system_mixed_type'):
        shutil.rmtree('system_mixed_type')

def gen_data_type_specific(nframes = 1):
    tmpdata = Data(rand_pert=0.1, seed=1, nframes=nframes)
    sys = dpdata.LabeledSystem()
    sys.data['atom_names'] = ['foo', 'bar']
    sys.data['coords'] = tmpdata.coord
    sys.data['atom_types'] = tmpdata.atype
    sys.data['cells'] = tmpdata.cell
    nframes = tmpdata.nframes
    natoms = tmpdata.natoms
    sys.data['coords'] = sys.data['coords'].reshape([nframes, natoms, 3])
    sys.data['cells'] = sys.data['cells'].reshape([nframes, 3, 3])
    sys.data['energies'] = np.zeros([nframes, 1])
    sys.data['forces'] = np.zeros([nframes, natoms, 3])
    sys.to_deepmd_npy('system', prec=np.float64)
    np.save('system/set.000/fparam.npy', tmpdata.fparam)
    np.save('system/set.000/aparam.npy', tmpdata.aparam.reshape([nframes, natoms, 2]))

def gen_data_mixed_type(nframes = 1):
    tmpdata = Data(rand_pert=0.1, seed=1, nframes=nframes)
    sys = dpdata.LabeledSystem()
    real_type_map = ['foo', 'bar']
    sys.data['atom_names'] = ['X']
    sys.data['coords'] = tmpdata.coord
    sys.data['atom_types'] = np.zeros_like(tmpdata.atype)
    sys.data['cells'] = tmpdata.cell
    nframes = tmpdata.nframes
    natoms = tmpdata.natoms
    sys.data['coords'] = sys.data['coords'].reshape([nframes, natoms, 3])
    sys.data['cells'] = sys.data['cells'].reshape([nframes, 3, 3])
    sys.data['energies'] = np.zeros([nframes, 1])
    sys.data['forces'] = np.zeros([nframes, natoms, 3])
    sys.to_deepmd_npy('system_mixed_type', prec=np.float64)
    np.savetxt('system_mixed_type/type_map.raw', real_type_map, fmt='%s')
    np.save('system_mixed_type/set.000/real_atom_types.npy', tmpdata.atype.reshape(1, -1).repeat(nframes, 0))
    np.save('system_mixed_type/set.000/fparam.npy', tmpdata.fparam)
    np.save('system_mixed_type/set.000/aparam.npy', tmpdata.aparam.reshape([nframes, natoms, 2]))

def gen_data(nframes = 1, mixed_type=False) :
    if not mixed_type:
        gen_data_type_specific(nframes)
    else:
        gen_data_mixed_type(nframes)


class Data():
    def __init__ (self, 
                  rand_pert = 0.1, 
                  seed = 1, 
                  box_scale = 20,
                  nframes = 1):
        coord = [[0.0, 0.0, 0.1], [1.1, 0.0, 0.1], [0.0, 1.1, 0.1], 
                 [4.0, 0.0, 0.0], [5.1, 0.0, 0.0], [4.0, 1.1, 0.0]]
        self.nframes = nframes
        self.coord = np.array(coord)
        self.coord = self._copy_nframes(self.coord)
        dp_random.seed(seed)
        self.coord += rand_pert * dp_random.random(self.coord.shape)
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
        self.efield = dp_random.random(self.coord.shape)
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
        box += rand_pert * dp_random.random(box.shape)
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
    np.testing.assert_almost_equal(ana_vir, num_vir,
                                   places, 
                                   err_msg = 'virial component')
    


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


def finite_difference(f, x, delta=1e-6):
    in_shape = x.shape
    y0 = f(x)
    out_shape = y0.shape
    res = np.empty(out_shape+in_shape)
    for idx in np.ndindex(*in_shape):
        diff = np.zeros(in_shape)
        diff[idx] += delta
        y1p = f(x+diff)
        y1n = f(x-diff)
        res[(Ellipsis, *idx)] = (y1p - y1n) / (2 * delta)
    return res


def strerch_box(old_coord, old_box, new_box):
    ocoord = old_coord.reshape(-1,3)
    obox = old_box.reshape(3,3)
    nbox = new_box.reshape(3,3)
    ncoord = ocoord @ np.linalg.inv(obox) @ nbox
    return ncoord.reshape(old_coord.shape)


def run_dp(cmd: str) -> int:
    """Run DP directly from the entry point instead of the subprocess.
    
    It is quite slow to start DeePMD-kit with subprocess.

    Parameters
    ----------
    cmd : str
        The command to run.

    Returns
    -------
    int
        Always returns 0.
    """
    cmds = cmd.split()
    if cmds[0] == 'dp':
        cmds = cmds[1:]
    else:
        raise RuntimeError('The command is not dp')

    main(cmds)
    return 0


# some tests still need this class
class DataSets (object):
    """
    Outdated class for one data system.
    .. deprecated:: 2.0.0
        This class is not maintained any more.
    """
    def __init__ (self, 
                  sys_path,
                  set_prefix,
                  seed = None, 
                  shuffle_test = True) :
        self.dirs = glob.glob (os.path.join(sys_path, set_prefix + ".*"))
        self.dirs.sort()
        # load atom type
        self.atom_type, self.idx_map, self.idx3_map = self.load_type (sys_path)
        # load atom type map
        self.type_map = self.load_type_map(sys_path)
        if self.type_map is not None:
            assert(len(self.type_map) >= max(self.atom_type)+1)
        # train dirs
        self.test_dir   = self.dirs[-1]
        if len(self.dirs) == 1 :
            self.train_dirs = self.dirs
        else :
            self.train_dirs = self.dirs[:-1]
        # check fparam
        has_fparam = [ os.path.isfile(os.path.join(ii, 'fparam.npy')) for ii in self.dirs ]
        if any(has_fparam) and (not all(has_fparam)) :
            raise RuntimeError("system %s: if any set has frame parameter, then all sets should have frame parameter" % sys_path)
        if all(has_fparam) :
            self.has_fparam = 0
        else :
            self.has_fparam = -1
        # check aparam
        has_aparam = [ os.path.isfile(os.path.join(ii, 'aparam.npy')) for ii in self.dirs ]
        if any(has_aparam) and (not all(has_aparam)) :
            raise RuntimeError("system %s: if any set has frame parameter, then all sets should have frame parameter" % sys_path)
        if all(has_aparam) :
            self.has_aparam = 0
        else :
            self.has_aparam = -1
        # energy norm
        self.eavg = self.stats_energy()
        # load sets
        self.set_count = 0
        self.load_batch_set (self.train_dirs[self.set_count % self.get_numb_set()])
        self.load_test_set (self.test_dir, shuffle_test)

    def check_batch_size (self, batch_size) :
        for ii in self.train_dirs :
            tmpe = np.load(os.path.join(ii, "coord.npy"))
            if tmpe.shape[0] < batch_size :
                return ii, tmpe.shape[0]
        return None

    def check_test_size (self, test_size) :
        tmpe = np.load(os.path.join(self.test_dir, "coord.npy"))
        if tmpe.shape[0] < test_size :
            return self.test_dir, tmpe.shape[0]
        else :
            return None

    def load_type (self, sys_path) :
        atom_type = np.loadtxt (os.path.join(sys_path, "type.raw"), dtype=np.int32, ndmin=1)
        natoms = atom_type.shape[0]
        idx = np.arange (natoms)
        idx_map = np.lexsort ((idx, atom_type))
        atom_type3 = np.repeat(atom_type, 3)
        idx3 = np.arange (natoms * 3)
        idx3_map = np.lexsort ((idx3, atom_type3))
        return atom_type, idx_map, idx3_map

    def load_type_map(self, sys_path) :
        fname = os.path.join(sys_path, 'type_map.raw')
        if os.path.isfile(fname) :            
            with open(os.path.join(sys_path, 'type_map.raw')) as fp:
                return fp.read().split()                
        else :
            return None

    def get_type_map(self) :
        return self.type_map

    def get_numb_set (self) :
        return len (self.train_dirs)

    def stats_energy (self) :
        eners = np.array([])
        for ii in self.train_dirs:
            ener_file = os.path.join(ii, "energy.npy")
            if os.path.isfile(ener_file) :
                ei = np.load(ener_file)
                eners = np.append(eners, ei)
        if eners.size == 0 :
            return 0
        else :
            return np.average(eners)

    def load_energy(self, 
                    set_name,
                    nframes,
                    nvalues,
                    energy_file, 
                    atom_energy_file) :
        """
        return : coeff_ener, ener, coeff_atom_ener, atom_ener
        """
        # load atom_energy
        coeff_atom_ener, atom_ener = self.load_data(set_name, atom_energy_file, [nframes, nvalues], False)
        # ignore energy_file
        if coeff_atom_ener == 1:
            ener = np.sum(atom_ener, axis = 1)
            coeff_ener = 1
        # load energy_file
        else:
            coeff_ener, ener = self.load_data(set_name, energy_file, [nframes], False)
        return coeff_ener, ener, coeff_atom_ener, atom_ener

    def load_data(self, set_name, data_name, shape, is_necessary = True):
        path = os.path.join(set_name, data_name+".npy")
        if os.path.isfile (path) :
            data = np.load(path)
            data = np.reshape(data, shape)
            if is_necessary:
                return data
            return 1, data
        elif is_necessary:
            raise OSError("%s not found!" % path)
        else:
            data = np.zeros(shape)
        return 0, data

    def load_set(self, set_name, shuffle = True):
        data = {}
        data["box"] = self.load_data(set_name, "box", [-1, 9])
        nframe = data["box"].shape[0]
        data["coord"] = self.load_data(set_name, "coord", [nframe, -1])
        ncoord = data["coord"].shape[1]
        if self.has_fparam >= 0:
            data["fparam"] = self.load_data(set_name, "fparam", [nframe, -1])
            if self.has_fparam == 0 :
                self.has_fparam = data["fparam"].shape[1]
            else :
                assert self.has_fparam == data["fparam"].shape[1]
        if self.has_aparam >= 0:
            data["aparam"] = self.load_data(set_name, "aparam", [nframe, -1])
            if self.has_aparam == 0 :
                self.has_aparam = data["aparam"].shape[1] // (ncoord//3)
            else :
                assert self.has_aparam == data["aparam"].shape[1] // (ncoord//3)
        data["prop_c"] = np.zeros(5)
        data["prop_c"][0], data["energy"], data["prop_c"][3], data["atom_ener"] \
            = self.load_energy (set_name, nframe, ncoord // 3, "energy", "atom_ener")
        data["prop_c"][1], data["force"] = self.load_data(set_name, "force", [nframe, ncoord], False)
        data["prop_c"][2], data["virial"] = self.load_data(set_name, "virial", [nframe, 9], False)
        data["prop_c"][4], data["atom_pref"] = self.load_data(set_name, "atom_pref", [nframe, ncoord//3], False)
        data["atom_pref"] = np.repeat(data["atom_pref"], 3, axis=1)
        # shuffle data
        if shuffle:
            idx = np.arange (nframe)
            dp_random.shuffle(idx)
            for ii in data:
                if ii != "prop_c":
                    data[ii] = data[ii][idx]
        data["type"] = np.tile (self.atom_type, (nframe, 1))
        # sort according to type
        for ii in ["type", "atom_ener"]:
            data[ii] = data[ii][:, self.idx_map]
        for ii in ["coord", "force", "atom_pref"]:
            data[ii] = data[ii][:, self.idx3_map]
        return data

    def load_batch_set (self,
                        set_name) :
        self.batch_set = self.load_set(set_name, True)
        self.reset_iter ()

    def load_test_set (self,
                       set_name, 
                       shuffle_test) :
        self.test_set = self.load_set(set_name, shuffle_test)

    def reset_iter (self) :
        self.iterator = 0              
        self.set_count += 1

    def get_set(self, data, idx = None) :
        new_data = {}
        for ii in data:
            dd = data[ii]
            if ii == "prop_c":
                new_data[ii] = dd.astype(np.float32)
            else:
                if idx is not None:
                    dd = dd[idx]
                if ii == "type":
                    new_data[ii] = dd
                else:
                    new_data[ii] = dd.astype(GLOBAL_NP_FLOAT_PRECISION)
        return new_data

    def get_test (self) :
        """
        returned property prefector [4] in order: 
        energy, force, virial, atom_ener
        """
        return self.get_set(self.test_set)

    def get_batch (self,
                   batch_size) :
        """
        returned property prefector [4] in order: 
        energy, force, virial, atom_ener
        """
        set_size = self.batch_set["energy"].shape[0]
        # assert (batch_size <= set_size), "batch size should be no more than set size"
        if self.iterator + batch_size > set_size :
            self.load_batch_set (self.train_dirs[self.set_count % self.get_numb_set()])
            set_size = self.batch_set["energy"].shape[0]
        # print ("%d %d %d" % (self.iterator, self.iterator + batch_size, set_size))
        iterator_1 = self.iterator + batch_size
        if iterator_1 >= set_size :
            iterator_1 = set_size
        idx = np.arange (self.iterator, iterator_1)
        self.iterator += batch_size
        return self.get_set(self.batch_set, idx)

    def get_natoms (self) :
        sample_type = self.batch_set["type"][0]
        natoms = len(sample_type)
        return natoms

    def get_natoms_2 (self, ntypes) :
        sample_type = self.batch_set["type"][0]
        natoms = len(sample_type)
        natoms_vec = np.zeros (ntypes).astype(int)
        for ii in range (ntypes) :
            natoms_vec[ii] = np.count_nonzero(sample_type == ii)
        return natoms, natoms_vec

    def get_natoms_vec (self, ntypes) :
        natoms, natoms_vec = self.get_natoms_2 (ntypes)
        tmp = [natoms, natoms]
        tmp = np.append (tmp, natoms_vec)
        return tmp.astype(np.int32)

    def set_numb_batch (self, 
                        batch_size) :
        return self.batch_set["energy"].shape[0] // batch_size

    def get_sys_numb_batch (self, batch_size) :
        return self.set_numb_batch(batch_size) * self.get_numb_set()

    def get_ener (self) :
        return self.eavg

    def numb_fparam(self) :
        return self.has_fparam

    def numb_aparam(self) :
        return self.has_aparam
