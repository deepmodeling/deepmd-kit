#!/usr/bin/env python3

import glob,os,json
import numpy as np


def valid_dir(name) :
    if not os.path.isfile(os.path.join(name, 'type.raw')) :
        raise OSError
    sets = glob.glob(os.path.join(name, 'set.*'))
    for ii in sets :
        if not os.path.isfile(os.path.join(ii, 'box.npy')) :
            raise OSError
        if not os.path.isfile(os.path.join(ii, 'coord.npy')) :
            raise OSError
        

def load_systems(dirs) :
    all_type = []
    all_box = []
    for ii in dirs :
        sys_type = np.loadtxt(os.path.join(ii, 'type.raw'), dtype = int)
        sys_box = None
        sets = glob.glob(os.path.join(ii, 'set.*'))
        for ii in sets :
            if type(sys_box) is not np.ndarray :
                sys_box = np.load(os.path.join(ii, 'box.npy'))
            else :
                sys_box = np.concatenate((sys_box, np.load(os.path.join(ii, 'box.npy'))), axis = 0)
        all_type.append(sys_type)
        all_box.append(sys_box)
    return all_type, all_box


def get_system_names() :
    dirs = input("Enter system path(s) (seperated by space, wide card supported): \n") 
    dirs = dirs.split()
    real_dirs = []
    for ii in dirs :
        real_dirs += glob.glob(ii)
    for ii in real_dirs :
        valid_dir(ii)
    return real_dirs

def get_rcut() :
    dv = 6
    rcut = input("Enter rcut (default %f A): \n" % dv) 
    try:
        rcut = float(rcut)
    except ValueError:
        rcut = dv
    if rcut <= 0:
        raise ValueError('rcut should be > 0')
    return rcut


def get_batch_size_rule() :
    dv = 32
    matom = input("Enter the minimal number of atoms in a batch (default %d): \n" % dv)
    try:
        matom = int(matom)
    except ValueError:
        matom = dv
    if matom <= 0:
        raise ValueError('the number should be > 0')
    return matom


def get_stop_batch():
    dv = 1000000
    sb = input("Enter the stop batch (default %d): \n" % dv)
    try:
        sb = int(sb)
    except ValueError:
        sb = dv
    if sb <= 0:
        raise ValueError('the number should be > 0')
    return sb


def get_ntypes (all_type) :
    coll = []
    for ii in all_type:
        coll += list(ii)
    list_coll = set(coll)
    return len(list_coll)


def get_max_density(all_type, all_box) :
    ntypes = get_ntypes(all_type)
    all_max = []
    for tt, bb in zip(all_type, all_box) :
        vv = np.reshape(bb, [-1,3,3]) 
        vv = np.linalg.det(vv)
        min_v = np.min(vv)
        type_count = []
        for ii in range(ntypes) :
            type_count.append(sum(tt == ii))
        max_den = type_count / min_v
        all_max.append(max_den)
    all_max = np.max(all_max, axis = 0)
    return all_max




def suggest_sel(all_type, all_box, rcut, ratio = 1.5) :
    max_den = get_max_density(all_type, all_box)
    return [int(ii) for ii in max_den * 4./3. * np.pi * rcut**3 * ratio]


def suggest_batch_size(all_type, min_atom) :
    bs = []
    for ii in all_type :
        natoms = len(ii)
        tbs = min_atom // natoms
        if (min_atom // natoms) * natoms != min_atom :
            tbs += 1
        bs.append(tbs)
    return bs


def suggest_decay(sb):
    decay_steps = int(sb // 200)
    decay_rate = 0.95
    return decay_steps, decay_rate


def default_data() :
    data = {}
    data['use_smooth'] = True
    data['sel_a'] = []
    data['rcut_smth'] = -1
    data['rcut'] = -1
    data['filter_neuron'] = [20, 40, 80]
    data['filter_resnet_dt'] = False
    data['axis_neuron'] = 8
    data['fitting_neuron'] = [240, 240, 240]
    data['fitting_resnet_dt'] = True
    data['coord_norm'] = True
    data['type_fitting_net'] = False
    data['systems'] = []
    data['set_prefix'] = 'set'
    data['stop_batch'] = -1
    data['batch_size'] = -1
    data['start_lr'] = 0.001
    data['decay_steps'] = -1
    data['decay_rate'] = 0.95
    data['start_pref_e'] = 0.02
    data['limit_pref_e'] = 1
    data['start_pref_f'] = 1000
    data['limit_pref_f'] = 1
    data['start_pref_v'] = 0
    data['limit_pref_v'] = 0
    data['seed'] = 1
    data['disp_file'] = 'lcurve.out'
    data['disp_freq'] = 1000
    data['numb_test'] = 10
    data['save_freq'] = 10000
    data["save_ckpt"] = "model.ckpt"
    data["disp_training"] = True
    data["time_training"] = True
    return data


def config(args) :
    all_sys = get_system_names()
    if len(all_sys) == 0 :
        raise RuntimeError('no system specified')
    rcut = get_rcut()
    matom = get_batch_size_rule()
    stop_batch = get_stop_batch()

    all_type, all_box = load_systems(all_sys)
    sel = suggest_sel(all_type, all_box, rcut, ratio = 1.5)
    bs = suggest_batch_size(all_type, matom)
    decay_steps, decay_rate = suggest_decay(stop_batch)
    
    jdata = default_data()
    jdata['systems'] = all_sys
    jdata['sel_a'] = sel
    jdata['rcut'] = rcut
    jdata['rcut_smth'] = rcut - 0.2
    jdata['stop_batch'] = stop_batch
    jdata['batch_size'] = bs
    jdata['decay_steps'] = decay_steps
    jdata['decay_rate'] = decay_rate    

    with open(args.output, 'w') as fp:
        json.dump(jdata, fp, indent=4)

