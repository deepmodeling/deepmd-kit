# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.pt.model.descriptor import (
    DescrptSeA,
)
from deepmd.pt.model.task import (
    EnergyFittingNet,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
    to_torch_tensor,
)


def _make_fake_data_pt(sys_natoms, sys_nframes, avgs, stds):
    merged_output_stat = []
    nsys = len(sys_natoms)
    ndof = len(avgs)
    for ii in range(nsys):
        sys_dict = {}
        tmp_data_f = []
        tmp_data_a = []
        for jj in range(ndof):
            rng = np.random.default_rng(2025 * ii + 220 * jj)
            tmp_data_f.append(
                rng.normal(loc=avgs[jj], scale=stds[jj], size=(sys_nframes[ii], 1))
            )
            rng = np.random.default_rng(220 * ii + 1636 * jj)
            tmp_data_a.append(
                rng.normal(
                    loc=avgs[jj], scale=stds[jj], size=(sys_nframes[ii], sys_natoms[ii])
                )
            )
        tmp_data_f = np.transpose(tmp_data_f, (1, 2, 0))
        tmp_data_a = np.transpose(tmp_data_a, (1, 2, 0))
        sys_dict["fparam"] = to_torch_tensor(tmp_data_f)
        sys_dict["aparam"] = to_torch_tensor(tmp_data_a)
        merged_output_stat.append(sys_dict)
    return merged_output_stat


def _brute_fparam_pt(data, ndim):
    adata = [to_numpy_array(ii["fparam"]) for ii in data]
    all_data = []
    for ii in adata:
        tmp = np.reshape(ii, [-1, ndim])
        if len(all_data) == 0:
            all_data = np.array(tmp)
        else:
            all_data = np.concatenate((all_data, tmp), axis=0)
    avg = np.average(all_data, axis=0)
    std = np.std(all_data, axis=0)
    return avg, std


def _brute_aparam_pt(data, ndim):
    adata = [to_numpy_array(ii["aparam"]) for ii in data]
    all_data = []
    for ii in adata:
        tmp = np.reshape(ii, [-1, ndim])
        if len(all_data) == 0:
            all_data = np.array(tmp)
        else:
            all_data = np.concatenate((all_data, tmp), axis=0)
    avg = np.average(all_data, axis=0)
    std = np.std(all_data, axis=0)
    return avg, std


class TestEnerFittingStat(unittest.TestCase):
    def test(self) -> None:
        descrpt = DescrptSeA(6.0, 5.8, [46, 92], neuron=[25, 50, 100], axis_neuron=16)
        fitting = EnergyFittingNet(
            descrpt.get_ntypes(),
            descrpt.get_dim_out(),
            neuron=[240, 240, 240],
            resnet_dt=True,
            numb_fparam=3,
            numb_aparam=3,
        )
        avgs = [0, 10, 100]
        stds = [2, 0.4, 0.00001]
        sys_natoms = [10, 100]
        sys_nframes = [5, 2]
        all_data = _make_fake_data_pt(sys_natoms, sys_nframes, avgs, stds)
        frefa, frefs = _brute_fparam_pt(all_data, len(avgs))
        arefa, arefs = _brute_aparam_pt(all_data, len(avgs))
        fitting.compute_input_stats(all_data, protection=1e-2)
        frefs_inv = 1.0 / frefs
        arefs_inv = 1.0 / arefs
        frefs_inv[frefs_inv > 100] = 100
        arefs_inv[arefs_inv > 100] = 100
        np.testing.assert_almost_equal(frefa, to_numpy_array(fitting.fparam_avg))
        np.testing.assert_almost_equal(
            frefs_inv, to_numpy_array(fitting.fparam_inv_std)
        )
        np.testing.assert_almost_equal(arefa, to_numpy_array(fitting.aparam_avg))
        np.testing.assert_almost_equal(
            arefs_inv, to_numpy_array(fitting.aparam_inv_std)
        )
