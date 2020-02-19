#!/usr/bin/env python3

import re
import os
import sys
import argparse
import numpy as np

from deepmd.Data import DeepmdData
from deepmd.common import expand_sys_str
from deepmd import DeepEval
from deepmd import DeepPot
from deepmd import DeepDipole
from deepmd import DeepPolar
from deepmd import DeepWFC
from tensorflow.python.framework import ops

def test (args):
    de = DeepEval(args.model)
    all_sys = expand_sys_str(args.system)
    err_coll = []
    siz_coll = []
    for ii in all_sys:
        args.system = ii
        print ("# ---------------output of dp test--------------- ")
        print ("# testing system : " + ii)
        if de.model_type == 'ener':
            err, siz = test_ener(args)
        elif de.model_type == 'dipole':
            err, siz = test_dipole(args)
        elif de.model_type == 'polar':
            err, siz = test_polar(args)
        elif de.model_type == 'wfc':
            err, siz = test_wfc(args)
        else :
            raise RuntimeError('unknow model type '+de.model_type)
        print ("# ----------------------------------------------- ")
        err_coll.append(err)
        siz_coll.append(siz)
    avg_err = weighted_average(err_coll, siz_coll)
    if len(all_sys) > 1:
        print ("# ----------weighted average of errors----------- ")
        print ("# number of systems : %d" % len(all_sys))
        if de.model_type == 'ener':
            print_ener_sys_avg(avg_err)
        elif de.model_type == 'dipole':
            print_dipole_sys_avg(avg_err)
        elif de.model_type == 'polar':
            print_polar_sys_avg(avg_err)
        elif de.model_type == 'wfc':
            print_wfc_sys_avg(avg_err)
        else :
            raise RuntimeError('unknow model type '+de.model_type)
        print ("# ----------------------------------------------- ")


def l2err (diff) :    
    return np.sqrt(np.average (diff*diff))


def weighted_average(err_coll, siz_coll):
    nsys = len(err_coll)
    nitems = len(err_coll[0])
    assert(len(err_coll) == len(siz_coll))
    sum_err = np.zeros(nitems)
    sum_siz = np.zeros(nitems)
    for sys_error, sys_size in zip(err_coll, siz_coll):
        for ii in range(nitems):
            ee = sys_error[ii]
            ss = sys_size [ii]
            sum_err[ii] += ee * ee * ss
            sum_siz[ii] += ss
    for ii in range(nitems):
        sum_err[ii] = np.sqrt(sum_err[ii] / sum_siz[ii])
    return sum_err


def test_ener (args) :
    if args.rand_seed is not None :
        np.random.seed(args.rand_seed % (2**32))

    dp = DeepPot(args.model)
    data = DeepmdData(args.system, args.set_prefix, shuffle_test = args.shuffle_test, type_map = dp.get_type_map())
    data.add('energy', 1, atomic=False, must=False, high_prec=True)
    data.add('force',  3, atomic=True,  must=False, high_prec=False)
    data.add('virial', 9, atomic=False, must=False, high_prec=False)
    if dp.get_dim_fparam() > 0:
        data.add('fparam', dp.get_dim_fparam(), atomic=False, must=True, high_prec=False)
    if dp.get_dim_aparam() > 0:
        data.add('aparam', dp.get_dim_aparam(), atomic=True,  must=True, high_prec=False)

    test_data = data.get_test ()
    natoms = len(test_data["type"][0])
    nframes = test_data["box"].shape[0]
    numb_test = args.numb_test
    numb_test = min(nframes, numb_test)

    coord = test_data["coord"][:numb_test].reshape([numb_test, -1])
    box = test_data["box"][:numb_test]
    if not data.pbc:
        box = None
    atype = test_data["type"][0]
    if dp.get_dim_fparam() > 0:
        fparam = test_data["fparam"][:numb_test] 
    else :
        fparam = None
    if dp.get_dim_aparam() > 0:
        aparam = test_data["aparam"][:numb_test] 
    else :
        aparam = None
    detail_file = args.detail_file
    if detail_file is not None:
        atomic = True
    else:
        atomic = False

    ret = dp.eval(coord, box, atype, fparam = fparam, aparam = aparam, atomic = atomic)
    energy = ret[0]
    force  = ret[1]
    virial = ret[2]
    energy = energy.reshape([numb_test,1])
    force = force.reshape([numb_test,-1])
    virial = virial.reshape([numb_test,9])
    if atomic:
        ae = ret[3]
        av = ret[4]
        ae = ae.reshape([numb_test,-1])
        av = av.reshape([numb_test,-1])

    l2e = (l2err (energy - test_data["energy"][:numb_test].reshape([-1,1])))
    l2f = (l2err (force  - test_data["force"] [:numb_test]))
    l2v = (l2err (virial - test_data["virial"][:numb_test]))
    l2ea= l2e/natoms
    l2va= l2v/natoms

    # print ("# energies: %s" % energy)
    print ("# number of test data : %d " % numb_test)
    print ("Energy L2err        : %e eV" % l2e)
    print ("Energy L2err/Natoms : %e eV" % l2ea)
    print ("Force  L2err        : %e eV/A" % l2f)
    print ("Virial L2err        : %e eV" % l2v)
    print ("Virial L2err/Natoms : %e eV" % l2va)

    if detail_file is not None :
        pe = np.concatenate((np.reshape(test_data["energy"][:numb_test], [-1,1]),
                             np.reshape(energy, [-1,1])), 
                            axis = 1)
        np.savetxt(detail_file+".e.out", pe, 
                   header = 'data_e pred_e')
        pf = np.concatenate((np.reshape(test_data["force"] [:numb_test], [-1,3]), 
                             np.reshape(force,  [-1,3])), 
                            axis = 1)
        np.savetxt(detail_file+".f.out", pf,
                   header = 'data_fx data_fy data_fz pred_fx pred_fy pred_fz')
        pv = np.concatenate((np.reshape(test_data["virial"][:numb_test], [-1,9]), 
                             np.reshape(virial, [-1,9])), 
                            axis = 1)
        np.savetxt(detail_file+".v.out", pv,
                   header = 'data_vxx data_vxy data_vxz data_vyx data_vyy data_vyz data_vzx data_vzy data_vzz pred_vxx pred_vxy pred_vxz pred_vyx pred_vyy pred_vyz pred_vzx pred_vzy pred_vzz')        
    return [l2ea, l2f, l2va], [energy.size, force.size, virial.size]


def print_ener_sys_avg(avg):
    print ("Energy L2err/Natoms : %e eV" % avg[0])
    print ("Force  L2err        : %e eV/A" % avg[1])
    print ("Virial L2err/Natoms : %e eV" % avg[2])


def test_wfc (args) :
    if args.rand_seed is not None :
        np.random.seed(args.rand_seed % (2**32))

    dp = DeepWFC(args.model)    
    data = DeepmdData(args.system, args.set_prefix, shuffle_test = args.shuffle_test)
    data.add('wfc', 12, atomic=True, must=True, high_prec=False, type_sel = dp.get_sel_type())
    test_data = data.get_test ()
    numb_test = args.numb_test
    natoms = len(test_data["type"][0])
    nframes = test_data["box"].shape[0]
    numb_test = min(nframes, numb_test)
                      
    coord = test_data["coord"][:numb_test].reshape([numb_test, -1])
    box = test_data["box"][:numb_test]
    atype = test_data["type"][0]
    wfc = dp.eval(coord, box, atype)

    wfc = wfc.reshape([numb_test,-1])
    l2f = (l2err (wfc  - test_data["wfc"] [:numb_test]))

    print ("# number of test data : %d " % numb_test)
    print ("WFC  L2err : %e eV/A" % l2f)

    detail_file = args.detail_file
    if detail_file is not None :
        pe = np.concatenate((np.reshape(test_data["wfc"][:numb_test], [-1,12]),
                             np.reshape(wfc, [-1,12])), 
                            axis = 1)
        np.savetxt(detail_file+".out", pe, 
                   header = 'ref_wfc(12 dofs)   predicted_wfc(12 dofs)')
    return [l2f], [wfc.size]


def print_wfc_sys_avg(avg):
    print ("WFC  L2err : %e eV/A" % avg[0])


def test_polar (args) :
    if args.rand_seed is not None :
        np.random.seed(args.rand_seed % (2**32))

    dp = DeepPolar(args.model)    
    data = DeepmdData(args.system, args.set_prefix, shuffle_test = args.shuffle_test)
    data.add('polarizability', 9, atomic=True, must=True, high_prec=False, type_sel = dp.get_sel_type())
    test_data = data.get_test ()
    numb_test = args.numb_test
    natoms = len(test_data["type"][0])
    nframes = test_data["box"].shape[0]
    numb_test = min(nframes, numb_test)
                      
    coord = test_data["coord"][:numb_test].reshape([numb_test, -1])
    box = test_data["box"][:numb_test]
    atype = test_data["type"][0]
    polar = dp.eval(coord, box, atype)

    polar = polar.reshape([numb_test,-1])
    l2f = (l2err (polar  - test_data["polarizability"] [:numb_test]))

    print ("# number of test data : %d " % numb_test)
    print ("Polarizability  L2err : %e eV/A" % l2f)

    detail_file = args.detail_file
    if detail_file is not None :
        pe = np.concatenate((np.reshape(test_data["polarizability"][:numb_test], [-1,9]),
                             np.reshape(polar, [-1,9])), 
                            axis = 1)
        np.savetxt(detail_file+".out", pe, 
                   header = 'data_pxx data_pxy data_pxz data_pyx data_pyy data_pyz data_pzx data_pzy data_pzz pred_pxx pred_pxy pred_pxz pred_pyx pred_pyy pred_pyz pred_pzx pred_pzy pred_pzz')
    return [l2f], [polar.size]


def print_polar_sys_avg(avg):
    print ("Polarizability  L2err : %e eV/A" % avg[0])


def test_dipole (args) :
    if args.rand_seed is not None :
        np.random.seed(args.rand_seed % (2**32))

    dp = DeepDipole(args.model)    
    data = DeepmdData(args.system, args.set_prefix, shuffle_test = args.shuffle_test)
    data.add('dipole', 3, atomic=True, must=True, high_prec=False, type_sel = dp.get_sel_type())
    test_data = data.get_test ()
    numb_test = args.numb_test
    natoms = len(test_data["type"][0])
    nframes = test_data["box"].shape[0]
    numb_test = min(nframes, numb_test)
                      
    coord = test_data["coord"][:numb_test].reshape([numb_test, -1])
    box = test_data["box"][:numb_test]
    atype = test_data["type"][0]
    dipole = dp.eval(coord, box, atype)

    dipole = dipole.reshape([numb_test,-1])
    l2f = (l2err (dipole  - test_data["dipole"] [:numb_test]))

    print ("# number of test data : %d " % numb_test)
    print ("Dipole  L2err         : %e eV/A" % l2f)

    detail_file = args.detail_file
    if detail_file is not None :
        pe = np.concatenate((np.reshape(test_data["dipole"][:numb_test], [-1,3]),
                             np.reshape(dipole, [-1,3])), 
                            axis = 1)
        np.savetxt(detail_file+".out", pe, 
                   header = 'data_x data_y data_z pred_x pred_y pred_z')
    return [l2f], [dipole.size]


def print_dipole_sys_avg(avg):
    print ("Dipole  L2err         : %e eV/A" % avg[0])
