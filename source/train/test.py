#!/usr/bin/env python3

import re
import os
import sys
import argparse
import numpy as np

from deepmd.Data import DeepmdData
from deepmd import DeepEval
from deepmd import DeepPot
from deepmd import DeepDipole
from deepmd import DeepPolar
from deepmd import DeepWFC
from tensorflow.python.framework import ops

def test (args):
    de = DeepEval(args.model)
    if de.model_type == 'ener':
        test_ener(args)
    elif de.model_type == 'dipole':
        test_dipole(args)
    elif de.model_type == 'polar':
        test_polar(args)
    elif de.model_type == 'wfc':
        test_wfc(args)
    else :
        raise RuntimeError('unknow model type '+de.model_type)

def l2err (diff) :    
    return np.sqrt(np.average (diff*diff))

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
    atype = test_data["type"][0]
    if dp.get_dim_fparam() > 0:
        fparam = test_data["fparam"][:numb_test] 
    else :
        fparam = None
    if dp.get_dim_aparam() > 0:
        aparam = test_data["aparam"][:numb_test] 
    else :
        aparam = None

    energy, force, virial, ae, av = dp.eval(coord, box, atype, fparam = fparam, aparam = aparam, atomic = True)
    energy = energy.reshape([numb_test,1])
    force = force.reshape([numb_test,-1])
    virial = virial.reshape([numb_test,9])
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

    detail_file = args.detail_file
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
