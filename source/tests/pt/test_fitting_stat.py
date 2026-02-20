# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import os
import shutil
import tempfile
import unittest
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)
from typing import (
    NoReturn,
)

import h5py
import numpy as np
import torch

from deepmd.pt.entrypoints.main import (
    get_trainer,
)
from deepmd.pt.model.descriptor import (
    DescrptSeA,
)
from deepmd.pt.model.task import (
    EnergyFittingNet,
)
from deepmd.pt.utils.multi_task import (
    preprocess_shared_params,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
    to_torch_tensor,
)
from deepmd.utils.argcheck import (
    normalize,
)
from deepmd.utils.compat import (
    update_deepmd_input,
)
from deepmd.utils.path import (
    DPPath,
)

from .model.test_permutation import (
    model_se_e2_a,
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
    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        h5file = str((Path(self.tempdir.name) / "testcase.h5").resolve())
        with h5py.File(h5file, "w") as f:
            pass
        self.stat_file_path = DPPath(h5file, "a")

    def test(self) -> None:
        descrpt = DescrptSeA(6.0, 5.8, [46, 92], neuron=[25, 50, 100], axis_neuron=16)
        avgs = [0, 10, 100]
        stds = [2, 0.4, 0.00001]
        sys_natoms = [10, 100]
        sys_nframes = [5, 2]
        all_data = _make_fake_data_pt(sys_natoms, sys_nframes, avgs, stds)
        frefa, frefs = _brute_fparam_pt(all_data, len(avgs))
        arefa, arefs = _brute_aparam_pt(all_data, len(avgs))
        frefs_inv = 1.0 / frefs
        arefs_inv = 1.0 / arefs
        frefs_inv[frefs_inv > 100] = 100
        arefs_inv[arefs_inv > 100] = 100

        # 1. test fitting stat is applied
        fitting = EnergyFittingNet(
            descrpt.get_ntypes(),
            descrpt.get_dim_out(),
            neuron=[240, 240, 240],
            resnet_dt=True,
            numb_fparam=3,
            numb_aparam=3,
        )
        fitting.compute_input_stats(
            all_data, protection=1e-2, stat_file_path=self.stat_file_path
        )
        np.testing.assert_almost_equal(frefa, to_numpy_array(fitting.fparam_avg))
        np.testing.assert_almost_equal(
            frefs_inv, to_numpy_array(fitting.fparam_inv_std)
        )
        np.testing.assert_almost_equal(arefa, to_numpy_array(fitting.aparam_avg))
        np.testing.assert_almost_equal(
            arefs_inv, to_numpy_array(fitting.aparam_inv_std)
        )
        del fitting

        # 2. test fitting stat writing to file is correct
        concat_fparam = np.concatenate(
            [
                to_numpy_array(all_data[ii]["fparam"].reshape(-1, 3))
                for ii in range(len(sys_nframes))
            ]
        )
        concat_aparam = np.concatenate(
            [
                to_numpy_array(all_data[ii]["aparam"].reshape(-1, 3))
                for ii in range(len(sys_nframes))
            ]
        )
        fparam_stat = (self.stat_file_path / "fparam").load_numpy()
        aparam_stat = (self.stat_file_path / "aparam").load_numpy()
        np.testing.assert_almost_equal(
            fparam_stat[:, 0], np.array([concat_fparam.shape[0]] * 3)
        )
        np.testing.assert_almost_equal(fparam_stat[:, 1], np.sum(concat_fparam, axis=0))
        np.testing.assert_almost_equal(
            fparam_stat[:, 2], np.sum(concat_fparam**2, axis=0)
        )
        np.testing.assert_almost_equal(
            aparam_stat[:, 0], np.array([concat_aparam.shape[0]] * 3)
        )
        np.testing.assert_almost_equal(aparam_stat[:, 1], np.sum(concat_aparam, axis=0))
        np.testing.assert_almost_equal(
            aparam_stat[:, 2], np.sum(concat_aparam**2, axis=0)
        )

        # 3. test fitting stat load from file
        def raise_error() -> NoReturn:
            raise RuntimeError

        fitting = EnergyFittingNet(
            descrpt.get_ntypes(),
            descrpt.get_dim_out(),
            neuron=[240, 240, 240],
            resnet_dt=True,
            numb_fparam=3,
            numb_aparam=3,
        )
        fitting.compute_input_stats(
            raise_error, protection=1e-2, stat_file_path=self.stat_file_path
        )
        np.testing.assert_almost_equal(frefa, to_numpy_array(fitting.fparam_avg))
        np.testing.assert_almost_equal(
            frefs_inv, to_numpy_array(fitting.fparam_inv_std)
        )
        np.testing.assert_almost_equal(arefa, to_numpy_array(fitting.aparam_avg))
        np.testing.assert_almost_equal(
            arefs_inv, to_numpy_array(fitting.aparam_inv_std)
        )


def get_weighted_fitting_stat(model_prob: list, *stat_arrays, protection: float):
    n_arrays = len(stat_arrays)
    assert len(model_prob) == n_arrays

    nframes = [stat.shape[0] for stat in stat_arrays]
    sums = [stat.sum(axis=0) for stat in stat_arrays]
    squared_sums = [(stat**2).sum(axis=0) for stat in stat_arrays]

    weighted_sum = sum(model_prob[i] * sums[i] for i in range(n_arrays))
    total_weighted_frames = sum(model_prob[i] * nframes[i] for i in range(n_arrays))
    weighted_avg = weighted_sum / total_weighted_frames

    weighted_square_sum = sum(model_prob[i] * squared_sums[i] for i in range(n_arrays))
    weighted_square_avg = weighted_square_sum / total_weighted_frames
    weighted_std = np.sqrt(weighted_square_avg - weighted_avg**2)
    weighted_std = np.where(weighted_std < protection, protection, weighted_std)

    return weighted_avg, weighted_std


class TestMultiTaskFittingStat(unittest.TestCase):
    def setUp(self) -> None:
        multitask_sharefit_template_json = str(
            Path(__file__).parent / "water/multitask_sharefit.json"
        )
        with open(multitask_sharefit_template_json) as f:
            multitask_se_e2_a = json.load(f)
        multitask_se_e2_a["model"]["shared_dict"]["my_descriptor"] = model_se_e2_a[
            "descriptor"
        ]
        self.data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.data_file_without_fparam = [
            str(Path(__file__).parent / "water/data/data_1")
        ]
        self.data_file_single = [str(Path(__file__).parent / "water/data/single")]
        self.stat_files = "se_e2_a_share_fit"
        os.makedirs(self.stat_files, exist_ok=True)

        self.config = multitask_se_e2_a
        self.config["training"]["data_dict"]["model_1"]["stat_file"] = (
            f"{self.stat_files}/model_1"
        )
        self.config["training"]["data_dict"]["model_2"]["stat_file"] = (
            f"{self.stat_files}/model_2"
        )
        self.config["model"]["shared_dict"]["my_fitting"]["numb_fparam"] = 2
        self.default_fparam = [1.0, 0.0]
        self.config["model"]["shared_dict"]["my_fitting"]["default_fparam"] = (
            self.default_fparam
        )
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1

        self.origin_config = deepcopy(self.config)

    def test_sharefitting_with_fparam(self):
        # test multitask training with fparam
        self.config = deepcopy(self.origin_config)
        model_prob = [0.3, 0.7]
        self.config["training"]["model_prob"]["model_1"] = model_prob[0]
        self.config["training"]["model_prob"]["model_2"] = model_prob[1]

        self.config["training"]["data_dict"]["model_1"]["training_data"]["systems"] = (
            self.data_file
        )
        self.config["training"]["data_dict"]["model_1"]["validation_data"][
            "systems"
        ] = self.data_file
        self.config["training"]["data_dict"]["model_2"]["training_data"]["systems"] = (
            self.data_file_single
        )
        self.config["training"]["data_dict"]["model_2"]["validation_data"][
            "systems"
        ] = self.data_file_single
        self.config["model"]["model_dict"]["model_1"]["data_stat_nbatch"] = 100

        self.config["model"], self.shared_links = preprocess_shared_params(
            self.config["model"]
        )
        self.config = update_deepmd_input(self.config, warning=True)
        self.config = normalize(self.config, multi_task=True)
        trainer = get_trainer(deepcopy(self.config), shared_links=self.shared_links)
        trainer.run()

        # check fparam shared
        multi_state_dict = trainer.wrapper.model.state_dict()
        torch.testing.assert_close(
            multi_state_dict["model_1.atomic_model.fitting_net.fparam_avg"],
            multi_state_dict["model_2.atomic_model.fitting_net.fparam_avg"],
        )
        torch.testing.assert_close(
            multi_state_dict["model_1.atomic_model.fitting_net.fparam_inv_std"],
            multi_state_dict["model_2.atomic_model.fitting_net.fparam_inv_std"],
        )

        # check fitting stat in stat_file is correct
        fparam_stat_model1 = np.load(f"{self.stat_files}/model_1/O H B/fparam")
        fparam_stat_model2 = np.load(f"{self.stat_files}/model_2/O H B/fparam")
        fparam_data1 = np.load(f"{self.data_file[0]}/set.000/fparam.npy")
        fparam_data2 = np.load(f"{self.data_file_single[0]}/set.000/fparam.npy")
        np.testing.assert_almost_equal(
            fparam_stat_model1[:, 0], [fparam_data1.shape[0]] * 2
        )
        np.testing.assert_almost_equal(
            fparam_stat_model1[:, 1], fparam_data1.sum(axis=0)
        )
        np.testing.assert_almost_equal(
            fparam_stat_model1[:, 2], (fparam_data1**2).sum(axis=0)
        )
        np.testing.assert_almost_equal(
            fparam_stat_model2[:, 0], [fparam_data2.shape[0]] * 2
        )
        np.testing.assert_almost_equal(
            fparam_stat_model2[:, 1], fparam_data2.sum(axis=0)
        )
        np.testing.assert_almost_equal(
            fparam_stat_model2[:, 2], (fparam_data2**2).sum(axis=0)
        )

        # check shared fitting stat is computed correctly
        weighted_avg, weighted_std = get_weighted_fitting_stat(
            model_prob, fparam_data1, fparam_data2, protection=1e-2
        )
        np.testing.assert_almost_equal(
            weighted_avg,
            to_numpy_array(
                multi_state_dict["model_1.atomic_model.fitting_net.fparam_avg"]
            ),
        )
        np.testing.assert_almost_equal(
            1 / weighted_std,
            to_numpy_array(
                multi_state_dict["model_1.atomic_model.fitting_net.fparam_inv_std"]
            ),
        )

    def test_sharefitting_using_default_fparam(self):
        # test multitask training with fparam
        self.config = deepcopy(self.origin_config)
        # add model3
        self.config["model"]["model_dict"]["model_3"] = deepcopy(
            self.config["model"]["model_dict"]["model_2"]
        )
        self.config["loss_dict"]["model_3"] = deepcopy(
            self.config["loss_dict"]["model_2"]
        )
        self.config["training"]["model_prob"]["model_3"] = deepcopy(
            self.config["training"]["model_prob"]["model_2"]
        )
        self.config["training"]["data_dict"]["model_3"] = deepcopy(
            self.config["training"]["data_dict"]["model_2"]
        )
        self.config["training"]["data_dict"]["model_3"]["stat_file"] = self.config[
            "training"
        ]["data_dict"]["model_3"]["stat_file"].replace("model_2", "model_3")
        self.config["model"]["shared_dict"]["my_fitting"]["dim_case_embd"] = 3

        model_prob = [0.1, 0.3, 0.6]
        self.config["training"]["model_prob"]["model_1"] = model_prob[0]
        self.config["training"]["model_prob"]["model_2"] = model_prob[1]
        self.config["training"]["model_prob"]["model_3"] = model_prob[2]

        self.config["training"]["data_dict"]["model_1"]["training_data"]["systems"] = (
            self.data_file_without_fparam
        )
        self.config["training"]["data_dict"]["model_1"]["validation_data"][
            "systems"
        ] = self.data_file_without_fparam
        self.config["training"]["data_dict"]["model_2"]["training_data"]["systems"] = (
            self.data_file_single
        )
        self.config["training"]["data_dict"]["model_2"]["validation_data"][
            "systems"
        ] = self.data_file_single
        self.config["training"]["data_dict"]["model_3"]["stat_file"] = (
            f"{self.stat_files}/model_3"
        )
        self.config["training"]["data_dict"]["model_3"]["training_data"]["systems"] = (
            self.data_file
        )
        self.config["training"]["data_dict"]["model_3"]["validation_data"][
            "systems"
        ] = self.data_file
        data_stat_protect = 5e-3
        self.config["model"]["model_dict"]["model_1"]["data_stat_nbatch"] = 3
        self.config["model"]["model_dict"]["model_3"]["data_stat_nbatch"] = 100
        self.config["model"]["model_dict"]["model_1"]["data_stat_protect"] = (
            data_stat_protect
        )
        self.config["model"]["model_dict"]["model_2"]["data_stat_protect"] = (
            data_stat_protect
        )
        self.config["model"]["model_dict"]["model_3"]["data_stat_protect"] = (
            data_stat_protect
        )

        self.config["model"], self.shared_links = preprocess_shared_params(
            self.config["model"]
        )
        self.config = update_deepmd_input(self.config, warning=True)
        self.config = normalize(self.config, multi_task=True)
        trainer = get_trainer(deepcopy(self.config), shared_links=self.shared_links)
        trainer.run()

        # check fparam shared
        multi_state_dict = trainer.wrapper.model.state_dict()
        torch.testing.assert_close(
            multi_state_dict["model_1.atomic_model.fitting_net.fparam_avg"],
            multi_state_dict["model_2.atomic_model.fitting_net.fparam_avg"],
        )
        torch.testing.assert_close(
            multi_state_dict["model_1.atomic_model.fitting_net.fparam_avg"],
            multi_state_dict["model_3.atomic_model.fitting_net.fparam_avg"],
        )
        torch.testing.assert_close(
            multi_state_dict["model_1.atomic_model.fitting_net.fparam_inv_std"],
            multi_state_dict["model_2.atomic_model.fitting_net.fparam_inv_std"],
        )
        torch.testing.assert_close(
            multi_state_dict["model_1.atomic_model.fitting_net.fparam_inv_std"],
            multi_state_dict["model_3.atomic_model.fitting_net.fparam_inv_std"],
        )

        # check fitting stat in stat_file is correct
        fparam_stat_model1 = np.load(f"{self.stat_files}/model_1/O H B/fparam")
        fparam_stat_model2 = np.load(f"{self.stat_files}/model_2/O H B/fparam")
        fparam_stat_model3 = np.load(f"{self.stat_files}/model_3/O H B/fparam")
        fparam_data1 = np.array([self.default_fparam]).repeat(3, axis=0)
        fparam_data2 = np.load(f"{self.data_file_single[0]}/set.000/fparam.npy")
        fparam_data3 = np.load(f"{self.data_file[0]}/set.000/fparam.npy")
        np.testing.assert_almost_equal(
            fparam_stat_model1[:, 0], [fparam_data1.shape[0]] * 2
        )
        np.testing.assert_almost_equal(
            fparam_stat_model1[:, 1], fparam_data1.sum(axis=0)
        )
        np.testing.assert_almost_equal(
            fparam_stat_model1[:, 2], (fparam_data1**2).sum(axis=0)
        )
        np.testing.assert_almost_equal(
            fparam_stat_model2[:, 0], [fparam_data2.shape[0]] * 2
        )
        np.testing.assert_almost_equal(
            fparam_stat_model2[:, 1], fparam_data2.sum(axis=0)
        )
        np.testing.assert_almost_equal(
            fparam_stat_model2[:, 2], (fparam_data2**2).sum(axis=0)
        )
        np.testing.assert_almost_equal(
            fparam_stat_model3[:, 0], [fparam_data3.shape[0]] * 2
        )
        np.testing.assert_almost_equal(
            fparam_stat_model3[:, 1], fparam_data3.sum(axis=0)
        )
        np.testing.assert_almost_equal(
            fparam_stat_model3[:, 2], (fparam_data3**2).sum(axis=0)
        )

        # check shared fitting stat is computed correctly
        weighted_avg, weighted_std = get_weighted_fitting_stat(
            model_prob,
            fparam_data1,
            fparam_data2,
            fparam_data3,
            protection=data_stat_protect,
        )
        np.testing.assert_almost_equal(
            weighted_avg,
            to_numpy_array(
                multi_state_dict["model_1.atomic_model.fitting_net.fparam_avg"]
            ),
        )
        np.testing.assert_almost_equal(
            1 / weighted_std,
            to_numpy_array(
                multi_state_dict["model_1.atomic_model.fitting_net.fparam_inv_std"]
            ),
        )

    def tearDown(self) -> None:
        for f in os.listdir("."):
            if f.startswith("model") and f.endswith(".pt"):
                os.remove(f)
            if f in ["lcurve.out", "checkpoint"]:
                os.remove(f)
            if f in [self.stat_files]:
                shutil.rmtree(f)
