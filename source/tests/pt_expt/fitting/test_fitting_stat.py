# SPDX-License-Identifier: LGPL-3.0-or-later
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

import numpy as np
import torch

from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.descriptor import (
    DescrptSeA,
)
from deepmd.pt_expt.entrypoints.main import (
    get_trainer,
)
from deepmd.pt_expt.fitting import (
    EnergyFittingNet,
)
from deepmd.pt_expt.utils import (
    env,
)
from deepmd.pt_expt.utils.multi_task import (
    preprocess_shared_params,
)
from deepmd.utils.argcheck import (
    normalize,
)
from deepmd.utils.compat import (
    update_deepmd_input,
)


def _make_fake_data_pt(sys_natoms, sys_nframes, avgs, stds):
    """Make fake data as numpy arrays for dpmodel compute_input_stats."""
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
        # dpmodel's compute_input_stats expects numpy arrays
        sys_dict["fparam"] = tmp_data_f
        sys_dict["aparam"] = tmp_data_a
        sys_dict["find_fparam"] = True
        sys_dict["find_aparam"] = True
        merged_output_stat.append(sys_dict)
    return merged_output_stat


def _brute_fparam_pt(data, ndim):
    adata = [ii["fparam"] for ii in data]
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
    adata = [ii["aparam"] for ii in data]
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


def _get_weighted_fitting_stat(
    model_prob: list, *stat_arrays, protection: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute probability-weighted fparam avg and std (matching PT)."""
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


# Paths to the water data used by PT tests.
# ``source/tests/pt/water`` is a symlink to ``model/water``; use the real
# path so CI checkouts that materialise symlinks as text files still work.
_PT_DATA = str(
    Path(__file__).parent.parent.parent / "pt" / "model" / "water" / "data" / "data_0"
)
_PT_DATA_NO_FPARAM = str(
    Path(__file__).parent.parent.parent / "pt" / "model" / "water" / "data" / "data_1"
)
_PT_DATA_SINGLE = str(
    Path(__file__).parent.parent.parent / "pt" / "model" / "water" / "data" / "single"
)

_descriptor_se_e2_a = {
    "type": "se_e2_a",
    "sel": [6, 12],
    "rcut_smth": 0.50,
    "rcut": 3.00,
    "neuron": [8, 16],
    "resnet_dt": False,
    "axis_neuron": 4,
    "type_one_side": True,
    "seed": 1,
}

_fitting_net = {
    "neuron": [16, 16],
    "resnet_dt": True,
    "seed": 1,
}


def _skip_if_no_data() -> None:
    if not os.path.isdir(_PT_DATA):
        raise unittest.SkipTest(f"Test data not found: {_PT_DATA}")


class TestEnerFittingStat(unittest.TestCase):
    def setUp(self) -> None:
        self.device = env.DEVICE

    def test(self) -> None:
        descrpt = DescrptSeA(6.0, 5.8, [46, 92], neuron=[25, 50, 100], axis_neuron=16)
        fitting = EnergyFittingNet(
            descrpt.get_ntypes(),
            descrpt.get_dim_out(),
            neuron=[240, 240, 240],
            resnet_dt=True,
            numb_fparam=3,
            numb_aparam=3,
        ).to(self.device)
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
        # fparam_avg and fparam_inv_std are torch tensors on device
        fparam_avg_np = (
            fitting.fparam_avg.detach().cpu().numpy()
            if torch.is_tensor(fitting.fparam_avg)
            else fitting.fparam_avg
        )
        fparam_inv_std_np = (
            fitting.fparam_inv_std.detach().cpu().numpy()
            if torch.is_tensor(fitting.fparam_inv_std)
            else fitting.fparam_inv_std
        )
        aparam_avg_np = (
            fitting.aparam_avg.detach().cpu().numpy()
            if torch.is_tensor(fitting.aparam_avg)
            else fitting.aparam_avg
        )
        aparam_inv_std_np = (
            fitting.aparam_inv_std.detach().cpu().numpy()
            if torch.is_tensor(fitting.aparam_inv_std)
            else fitting.aparam_inv_std
        )
        np.testing.assert_almost_equal(frefa, fparam_avg_np)
        np.testing.assert_almost_equal(frefs_inv, fparam_inv_std_np)
        np.testing.assert_almost_equal(arefa, aparam_avg_np)
        np.testing.assert_almost_equal(arefs_inv, aparam_inv_std_np)


class TestMultiTaskFittingStat(unittest.TestCase):
    """Test shared fitting stat (fparam_avg/fparam_inv_std) in multi-task.

    Corresponds to PT's TestMultiTaskFittingStat in test_fitting_stat.py.
    Verifies:
    1. fparam stats are shared between models (same tensor values)
    2. stat file contents match raw data (number, sum, squared_sum)
    3. weighted stat computation matches model values
    4. case_embd with default_fparam works correctly
    """

    @classmethod
    def setUpClass(cls) -> None:
        _skip_if_no_data()
        if not os.path.isdir(_PT_DATA_SINGLE):
            raise unittest.SkipTest(f"Test data not found: {_PT_DATA_SINGLE}")

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="pt_expt_fitstat_")
        self._old_cwd = os.getcwd()
        os.chdir(self.tmpdir)
        self.stat_files = "se_e2_a_share_fit"
        os.makedirs(self.stat_files, exist_ok=True)

    def tearDown(self) -> None:
        os.chdir(self._old_cwd)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_sharefit_config(
        self,
        *,
        numb_fparam: int = 2,
        numb_aparam: int = 0,
        default_fparam: list | None = None,
        dim_case_embd: int = 2,
        model_keys: list[str] | None = None,
        data_dirs: dict[str, str] | None = None,
        model_probs: dict[str, float] | None = None,
    ) -> dict:
        """Build a multi-task config with shared fitting + fparam."""
        if model_keys is None:
            model_keys = ["model_1", "model_2"]
        if data_dirs is None:
            data_dirs = dict.fromkeys(model_keys, _PT_DATA)
        if model_probs is None:
            model_probs = {mk: 1.0 / len(model_keys) for mk in model_keys}

        shared_fitting: dict = deepcopy(_fitting_net)
        shared_fitting["numb_fparam"] = numb_fparam
        if numb_aparam > 0:
            shared_fitting["numb_aparam"] = numb_aparam
        shared_fitting["dim_case_embd"] = dim_case_embd
        if default_fparam is not None:
            shared_fitting["default_fparam"] = default_fparam

        shared_dict: dict = {
            "my_type_map": ["O", "H"],
            "my_descriptor": deepcopy(_descriptor_se_e2_a),
            "my_fitting": shared_fitting,
        }

        model_dict = {}
        loss_dict = {}
        data_dict = {}
        for mk in model_keys:
            model_dict[mk] = {
                "type_map": "my_type_map",
                "descriptor": "my_descriptor",
                "fitting_net": "my_fitting",
                "data_stat_nbatch": 1,
            }
            loss_dict[mk] = {
                "type": "ener",
                "start_pref_e": 0.02,
                "limit_pref_e": 1,
                "start_pref_f": 1000,
                "limit_pref_f": 1,
                "start_pref_v": 0,
                "limit_pref_v": 0,
            }
            data_dict[mk] = {
                "stat_file": f"{self.stat_files}/{mk}",
                "training_data": {
                    "systems": [data_dirs[mk]],
                    "batch_size": 1,
                },
                "validation_data": {
                    "systems": [data_dirs[mk]],
                    "batch_size": 1,
                    "numb_btch": 1,
                },
            }

        config = {
            "model": {
                "shared_dict": shared_dict,
                "model_dict": model_dict,
            },
            "learning_rate": {
                "type": "exp",
                "decay_steps": 500,
                "start_lr": 0.001,
                "stop_lr": 3.51e-8,
            },
            "loss_dict": loss_dict,
            "training": {
                "model_prob": model_probs,
                "data_dict": data_dict,
                "numb_steps": 1,
                "seed": 10,
                "disp_file": "lcurve.out",
                "disp_freq": 1,
                "save_freq": 1,
            },
        }
        return config

    def test_sharefitting_with_fparam(self) -> None:
        """Shared fitting with fparam data: weighted fparam stat merging."""
        model_prob = [0.3, 0.7]
        config = self._make_sharefit_config(
            numb_fparam=2,
            default_fparam=[1.0, 0.0],
            data_dirs={"model_1": _PT_DATA, "model_2": _PT_DATA_SINGLE},
            model_probs={"model_1": model_prob[0], "model_2": model_prob[1]},
        )
        # data_0 has 80 frames; use data_stat_nbatch=100 to cover all frames
        config["model"]["model_dict"]["model_1"]["data_stat_nbatch"] = 80
        config["model"], shared_links = preprocess_shared_params(config["model"])
        config = update_deepmd_input(config, warning=False)
        config = normalize(config, multi_task=True)
        trainer = get_trainer(deepcopy(config), shared_links=shared_links)
        trainer.run()

        # fparam_avg and fparam_inv_std should be shared between models
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
        fparam_stat_model1 = np.load(f"{self.stat_files}/model_1/O H/fparam")
        fparam_stat_model2 = np.load(f"{self.stat_files}/model_2/O H/fparam")
        fparam_data1 = np.load(os.path.join(_PT_DATA, "set.000", "fparam.npy"))
        fparam_data2 = np.load(os.path.join(_PT_DATA_SINGLE, "set.000", "fparam.npy"))
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
        weighted_avg, weighted_std = _get_weighted_fitting_stat(
            model_prob, fparam_data1, fparam_data2, protection=1e-2
        )
        np.testing.assert_almost_equal(
            weighted_avg,
            to_numpy_array(
                multi_state_dict["model_1.atomic_model.fitting_net.fparam_avg"]
            ),
        )
        np.testing.assert_almost_equal(
            1.0 / weighted_std,
            to_numpy_array(
                multi_state_dict["model_1.atomic_model.fitting_net.fparam_inv_std"]
            ),
        )

    def test_sharefitting_using_default_fparam(self) -> None:
        """3 models with dim_case_embd=3, default fparam, no fparam in data."""
        default_fparam = [1.0, 0.0]
        model_prob = [0.1, 0.3, 0.6]
        data_stat_protect = 5e-3
        config = self._make_sharefit_config(
            numb_fparam=2,
            default_fparam=default_fparam,
            dim_case_embd=3,
            model_keys=["model_1", "model_2", "model_3"],
            data_dirs={
                "model_1": _PT_DATA_NO_FPARAM,
                "model_2": _PT_DATA_SINGLE,
                "model_3": _PT_DATA,
            },
            model_probs={
                "model_1": model_prob[0],
                "model_2": model_prob[1],
                "model_3": model_prob[2],
            },
        )
        # model_1 uses data without fparam → default_fparam is used
        config["model"]["model_dict"]["model_1"]["data_stat_nbatch"] = 3
        config["model"]["model_dict"]["model_3"]["data_stat_nbatch"] = 80
        config["model"]["model_dict"]["model_1"]["data_stat_protect"] = (
            data_stat_protect
        )
        config["model"]["model_dict"]["model_2"]["data_stat_protect"] = (
            data_stat_protect
        )
        config["model"]["model_dict"]["model_3"]["data_stat_protect"] = (
            data_stat_protect
        )
        config["model"], shared_links = preprocess_shared_params(config["model"])
        config = update_deepmd_input(config, warning=False)
        config = normalize(config, multi_task=True)
        trainer = get_trainer(deepcopy(config), shared_links=shared_links)
        trainer.run()

        # fparam_avg shared across all 3 models
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
        fparam_stat_model1 = np.load(f"{self.stat_files}/model_1/O H/fparam")
        fparam_stat_model2 = np.load(f"{self.stat_files}/model_2/O H/fparam")
        fparam_stat_model3 = np.load(f"{self.stat_files}/model_3/O H/fparam")
        fparam_data1 = np.array([default_fparam]).repeat(3, axis=0)
        fparam_data2 = np.load(os.path.join(_PT_DATA_SINGLE, "set.000", "fparam.npy"))
        fparam_data3 = np.load(os.path.join(_PT_DATA, "set.000", "fparam.npy"))
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
        weighted_avg, weighted_std = _get_weighted_fitting_stat(
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
            1.0 / weighted_std,
            to_numpy_array(
                multi_state_dict["model_1.atomic_model.fitting_net.fparam_inv_std"]
            ),
        )

        # case_embd should be set on all 3 models
        ce1 = trainer.wrapper.model["model_1"].atomic_model.fitting_net.case_embd
        ce2 = trainer.wrapper.model["model_2"].atomic_model.fitting_net.case_embd
        ce3 = trainer.wrapper.model["model_3"].atomic_model.fitting_net.case_embd
        self.assertIsNotNone(ce1)
        self.assertIsNotNone(ce2)
        self.assertIsNotNone(ce3)

        # dim_case_embd=3 → each is a 3-element one-hot vector
        self.assertEqual(ce1.shape[-1], 3)
        self.assertEqual(ce2.shape[-1], 3)
        self.assertEqual(ce3.shape[-1], 3)

        # Each should be one-hot
        self.assertEqual(ce1.sum().item(), 1.0)
        self.assertEqual(ce2.sum().item(), 1.0)
        self.assertEqual(ce3.sum().item(), 1.0)

        # All three should be different
        self.assertFalse(torch.equal(ce1, ce2))
        self.assertFalse(torch.equal(ce1, ce3))
        self.assertFalse(torch.equal(ce2, ce3))

        # case_embd should NOT be shared in state_dict
        for state_key in multi_state_dict:
            if "case_embd" in state_key and "model_1" in state_key:
                k2 = state_key.replace("model_1", "model_2")
                k3 = state_key.replace("model_1", "model_3")
                self.assertFalse(
                    torch.equal(multi_state_dict[state_key], multi_state_dict[k2]),
                )
                self.assertFalse(
                    torch.equal(multi_state_dict[state_key], multi_state_dict[k3]),
                )

    def test_sharefitting_with_aparam(self) -> None:
        """Weighted aparam stat merging in share_params (unit test).

        Directly tests the aparam branch in InvarFitting.share_params by
        creating two fittings with different aparam stats and verifying that
        share_params produces the correct probability-weighted merged result.
        """
        descrpt = DescrptSeA(6.0, 5.8, [46, 92], neuron=[25, 50, 100], axis_neuron=16)
        ntypes = descrpt.get_ntypes()
        dim_out = descrpt.get_dim_out()

        fit_base = EnergyFittingNet(
            ntypes, dim_out, neuron=[16, 16], numb_aparam=3, seed=1
        ).to(env.DEVICE)
        fit_link = EnergyFittingNet(
            ntypes, dim_out, neuron=[16, 16], numb_aparam=3, seed=2
        ).to(env.DEVICE)

        # give both fittings different aparam stats
        data_base = _make_fake_data_pt(
            [10, 100], [5, 2], [0, 10, 100], [2, 0.4, 0.00001]
        )
        data_link = _make_fake_data_pt([50], [8], [5, 20, 50], [1, 0.5, 0.01])
        fit_base.compute_input_stats(data_base, protection=1e-2)
        fit_link.compute_input_stats(data_link, protection=1e-2)

        # record base's aparam_avg before share_params
        orig_base_avg = fit_base.aparam_avg.clone()

        # share_params with model_prob=0.6 — should do weighted merging
        model_prob = 0.6
        fit_link.share_params(
            fit_base, shared_level=0, model_prob=model_prob, protection=1e-2
        )

        # base's aparam_avg was UPDATED (weighted merging happened)
        self.assertFalse(
            torch.equal(fit_base.aparam_avg, orig_base_avg),
            "aparam_avg should have changed after weighted merging",
        )

        # buffers are shared (same data_ptr)
        self.assertEqual(fit_link.aparam_avg.data_ptr(), fit_base.aparam_avg.data_ptr())
        self.assertEqual(
            fit_link.aparam_inv_std.data_ptr(), fit_base.aparam_inv_std.data_ptr()
        )

        # verify the merged stats match manual computation
        # reconstruct raw aparam data from each fitting's stats
        base_aparam_stats = fit_base.get_param_stats().get("aparam", [])
        # the merged stats should have 3 StatItem objects
        self.assertEqual(len(base_aparam_stats), 3)

        # manually compute the weighted average from raw data
        # data_base has two systems: [10 natoms, 5 frames] + [100 natoms, 2 frames]
        # data_link has one system: [50 natoms, 8 frames]
        # aparam per system: reshape to (nframes * natoms, numb_aparam)
        all_base = np.concatenate(
            [d["aparam"].reshape(-1, 3) for d in data_base], axis=0
        )
        all_link = np.concatenate(
            [d["aparam"].reshape(-1, 3) for d in data_link], axis=0
        )
        # weighted stat: base contributes with weight 1.0, link with model_prob
        total_n = all_base.shape[0] + model_prob * all_link.shape[0]
        weighted_sum = all_base.sum(axis=0) + model_prob * all_link.sum(axis=0)
        weighted_avg = weighted_sum / total_n
        weighted_sq_sum = (all_base**2).sum(axis=0) + model_prob * (all_link**2).sum(
            axis=0
        )
        weighted_sq_avg = weighted_sq_sum / total_n
        weighted_std = np.sqrt(weighted_sq_avg - weighted_avg**2)
        weighted_std = np.where(weighted_std < 1e-2, 1e-2, weighted_std)

        aparam_avg_np = to_numpy_array(fit_base.aparam_avg)
        aparam_inv_std_np = to_numpy_array(fit_base.aparam_inv_std)
        np.testing.assert_almost_equal(aparam_avg_np, weighted_avg)
        np.testing.assert_almost_equal(aparam_inv_std_np, 1.0 / weighted_std)

    def test_sharefitting_resume_preserves_stats(self) -> None:
        """resume=True in share_params skips stat merging, preserves buffers."""
        descrpt = DescrptSeA(6.0, 5.8, [46, 92], neuron=[25, 50, 100], axis_neuron=16)
        ntypes = descrpt.get_ntypes()
        dim_out = descrpt.get_dim_out()

        fit_base = EnergyFittingNet(
            ntypes, dim_out, neuron=[16, 16], numb_fparam=2, seed=1
        ).to(env.DEVICE)
        fit_link = EnergyFittingNet(
            ntypes, dim_out, neuron=[16, 16], numb_fparam=2, seed=2
        ).to(env.DEVICE)

        # give both fittings different stats
        data_base = _make_fake_data_pt([10], [5], [0, 10], [2, 0.4])
        data_link = _make_fake_data_pt([100], [2], [100, 0], [0.001, 3])
        fit_base.compute_input_stats(data_base, protection=1e-2)
        fit_link.compute_input_stats(data_link, protection=1e-2)

        # record base's fparam_avg BEFORE sharing
        orig_avg = fit_base.fparam_avg.clone()
        orig_inv_std = fit_base.fparam_inv_std.clone()

        # share_params with resume=True: should NOT re-merge stats
        fit_link.share_params(fit_base, shared_level=0, resume=True)

        # base's fparam_avg unchanged (no weighted merging happened)
        torch.testing.assert_close(fit_base.fparam_avg, orig_avg)
        torch.testing.assert_close(fit_base.fparam_inv_std, orig_inv_std)

        # buffers are shared (same data_ptr)
        self.assertEqual(fit_link.fparam_avg.data_ptr(), fit_base.fparam_avg.data_ptr())
        self.assertEqual(
            fit_link.fparam_inv_std.data_ptr(), fit_base.fparam_inv_std.data_ptr()
        )

    def test_case_embd_mismatched_dim_raises(self) -> None:
        """dim_case_embd must be the same across all models."""
        config = self._make_sharefit_config(dim_case_embd=2)
        # Override model_2 to have a different dim_case_embd
        config["model"]["model_dict"]["model_2"]["fitting_net"] = deepcopy(_fitting_net)
        config["model"]["model_dict"]["model_2"]["fitting_net"]["dim_case_embd"] = 3
        config["model"], shared_links = preprocess_shared_params(config["model"])
        config = update_deepmd_input(config, warning=False)
        config = normalize(config, multi_task=True)
        with self.assertRaises(
            ValueError, msg="Should reject mismatched dim_case_embd"
        ):
            get_trainer(config, shared_links=shared_links)
