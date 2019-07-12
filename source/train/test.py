#!/usr/bin/env python3

import re
import os
import sys
import argparse
import numpy as np
import tensorflow as tf

from Data import DataSets
from DeepPot import DeepPot
from tensorflow.python.framework import ops

def l2err (diff) :
    return np.sqrt(np.average (diff*diff))

def test (args) :
    if args.rand_seed is not None :
        np.random.seed(args.rand_seed % (2**32))

    data = DataSets (args.system, args.set_prefix, shuffle_test = args.shuffle_test)
    test_prop_c, test_energy, test_force, test_virial, test_ae, test_coord, test_box, test_type, test_fparam = data.get_test ()
    numb_test = args.numb_test
    natoms = len(test_type[0])
    nframes = test_box.shape[0]
    dp = DeepPot(args.model)
    coord = test_coord[:numb_test].reshape([numb_test, -1])
    box = test_box[:numb_test]
    atype = test_type[0]
    energy, force, virial, ae, av = dp.eval(coord, box, atype, fparam = test_fparam, atomic = True)
    energy = energy.reshape([nframes,1])
    force = force.reshape([nframes,-1])
    virial = virial.reshape([nframes,9])
    ae = ae.reshape([nframes,-1])
    av = av.reshape([nframes,-1])

    l2e = (l2err (energy - test_energy[:numb_test]))
    l2f = (l2err (force  - test_force [:numb_test]))
    l2v = (l2err (virial - test_virial[:numb_test]))
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
        pe = np.concatenate((np.reshape(test_energy[:numb_test], [-1,1]),
                             np.reshape(energy, [-1,1])), 
                            axis = 1)
        np.savetxt(detail_file+".e.out", pe, 
                   header = 'data_e pred_e')
        pf = np.concatenate((np.reshape(test_force [:numb_test], [-1,3]), 
                             np.reshape(force,  [-1,3])), 
                            axis = 1)
        np.savetxt(detail_file+".f.out", pf,
                   header = 'data_fx data_fy data_fz pred_fx pred_fy pred_fz')
        pv = np.concatenate((np.reshape(test_virial[:numb_test], [-1,9]), 
                             np.reshape(virial, [-1,9])), 
                            axis = 1)
        np.savetxt(detail_file+".v.out", pv,
                   header = 'data_vxx data_vxy data_vxz data_vyx data_vyy data_vyz data_vzx data_vzy data_vzz pred_vxx pred_vxy pred_vxz pred_vyx pred_vyy pred_vyz pred_vzx pred_vzy pred_vzz')        

