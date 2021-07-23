import os,sys
import numpy as np
import unittest

from collections import defaultdict
from deepmd.descriptor import DescrptSeA
from deepmd.fit import EnerFitting
from common import j_loader

input_json = 'water_se_a_afparam.json'

def _make_fake_data(sys_natoms, sys_nframes, avgs, stds):
    all_stat = defaultdict(list)
    nsys = len(sys_natoms)
    ndof = len(avgs)
    for ii in range(nsys):
        tmp_data_f = []
        tmp_data_a = []
        for jj in range(ndof) :
            tmp_data_f.append(np.random.normal(loc = avgs[jj], 
                                               scale = stds[jj],
                                               size = (sys_nframes[ii],1)))
            tmp_data_a.append(np.random.normal(loc = avgs[jj], 
                                               scale = stds[jj],
                                               size = (sys_nframes[ii], sys_natoms[ii])))
        tmp_data_f = np.transpose(tmp_data_f, (1,2,0))
        tmp_data_a = np.transpose(tmp_data_a, (1,2,0))
        all_stat['fparam'].append(tmp_data_f)
        all_stat['aparam'].append(tmp_data_a)
    return all_stat

def _brute_fparam(data, ndim):
    adata = data['fparam']
    all_data = []
    for ii in adata:
        tmp = np.reshape(ii, [-1, ndim])
        if len(all_data) == 0:
            all_data = np.array(tmp)
        else:
            all_data = np.concatenate((all_data, tmp), axis = 0)
    avg = np.average(all_data, axis = 0)
    std = np.std(all_data, axis = 0)
    return avg, std

def _brute_aparam(data, ndim):
    adata = data['aparam']
    all_data = []
    for ii in adata:
        tmp = np.reshape(ii, [-1, ndim])
        if len(all_data) == 0:
            all_data = np.array(tmp)
        else:
            all_data = np.concatenate((all_data, tmp), axis = 0)
    avg = np.average(all_data, axis = 0)
    std = np.std(all_data, axis = 0)
    return avg, std


class TestEnerFittingStat (unittest.TestCase) :
    def test (self) :
        jdata = j_loader(input_json)
        jdata = jdata['model']
        # descrpt = DescrptSeA(jdata['descriptor'])
        # fitting = EnerFitting(jdata['fitting_net'], descrpt)
        descrpt = DescrptSeA(6.0, 
                             5.8,
                             [46, 92],
                             neuron = [25, 50, 100], 
                             axis_neuron = 16)
        fitting = EnerFitting(descrpt,
                              neuron = [240, 240, 240],
                              resnet_dt = True,
                              numb_fparam = 2,
                              numb_aparam = 2)
        avgs = [0, 10]
        stds = [2, 0.4]
        sys_natoms = [10, 100]
        sys_nframes = [5, 2]
        all_data = _make_fake_data(sys_natoms, sys_nframes, avgs, stds)
        frefa, frefs = _brute_fparam(all_data, len(avgs))
        arefa, arefs = _brute_aparam(all_data, len(avgs))
        fitting.compute_input_stats(all_data, protection = 1e-2)
        # print(frefa, frefs)
        for ii in range(len(avgs)):
            self.assertAlmostEqual(frefa[ii], fitting.fparam_avg[ii])
            self.assertAlmostEqual(frefs[ii], fitting.fparam_std[ii])
            self.assertAlmostEqual(arefa[ii], fitting.aparam_avg[ii])
            self.assertAlmostEqual(arefs[ii], fitting.aparam_std[ii])
