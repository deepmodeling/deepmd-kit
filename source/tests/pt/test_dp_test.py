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
from unittest import (
    mock,
)

import numpy as np
import torch

from deepmd.entrypoints.test import test as dp_test
from deepmd.infer.deep_eval import (
    DeepEval,
)
from deepmd.infer.model_test import (
    build_tester,
)
from deepmd.pt.entrypoints.main import (
    get_trainer,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
)
from deepmd.utils.data import (
    DeepmdData,
)

from .model.test_permutation import (
    model_property,
    model_se_e2_a,
    model_spin,
)


class DPTest:
    def _run_dp_test(
        self, use_input_json: bool, numb_test: int = 1, use_train: bool = False
    ) -> None:
        trainer = get_trainer(deepcopy(self.config))
        with torch.device("cpu"):
            input_dict, label_dict, _ = trainer.get_data(is_train=False)
        has_spin = getattr(trainer.model, "has_spin", False)
        if callable(has_spin):
            has_spin = has_spin()
        if not has_spin:
            input_dict.pop("spin", None)
        input_dict["do_atomic_virial"] = True
        result = trainer.model(**input_dict)
        model = torch.jit.script(trainer.model)
        tmp_fd, tmp_model_path = tempfile.mkstemp(suffix=".pth")
        os.close(tmp_fd)
        torch.jit.save(model, tmp_model_path)
        val_sys = self.config["training"]["validation_data"]["systems"]
        if isinstance(val_sys, list):
            val_sys = val_sys[0]
        dp_test(
            model=tmp_model_path,
            system=None if use_input_json else val_sys,
            datafile=None,
            train_json=self.input_json if use_input_json and use_train else None,
            valid_json=self.input_json if use_input_json and not use_train else None,
            set_prefix="set",
            numb_test=numb_test,
            rand_seed=None,
            shuffle_test=False,
            detail_file=self.detail_file,
            atomic=False,
        )
        os.unlink(tmp_model_path)
        natom = input_dict["atype"].shape[1]
        pred_e = np.loadtxt(self.detail_file + ".e.out", ndmin=2)[0, 1]
        np.testing.assert_almost_equal(
            pred_e,
            to_numpy_array(result["energy"])[0][0],
        )
        pred_e_peratom = np.loadtxt(self.detail_file + ".e_peratom.out", ndmin=2)[0, 1]
        np.testing.assert_almost_equal(pred_e_peratom, pred_e / natom)
        if not has_spin:
            pred_f = np.loadtxt(self.detail_file + ".f.out", ndmin=2)[:, 3:6]
            np.testing.assert_almost_equal(
                pred_f,
                to_numpy_array(result["force"]).reshape(-1, 3),
            )
            if os.path.exists(self.detail_file + ".v.out"):
                pred_v = np.loadtxt(self.detail_file + ".v.out", ndmin=2)[:, 9:18]
                np.testing.assert_almost_equal(
                    pred_v,
                    to_numpy_array(result["virial"]),
                )
                pred_v_peratom = np.loadtxt(
                    self.detail_file + ".v_peratom.out", ndmin=2
                )[:, 9:18]
                np.testing.assert_almost_equal(pred_v_peratom, pred_v / natom)
            else:
                self.assertFalse(os.path.exists(self.detail_file + ".v_peratom.out"))
        else:
            pred_fr = np.loadtxt(self.detail_file + ".fr.out", ndmin=2)[:, 3:6]
            np.testing.assert_almost_equal(
                pred_fr,
                to_numpy_array(result["force"]).reshape(-1, 3),
            )
            if os.path.exists(self.detail_file + ".fm.out"):
                pred_fm = np.loadtxt(self.detail_file + ".fm.out", ndmin=2)[:, 3:6]
                np.testing.assert_almost_equal(
                    pred_fm,
                    to_numpy_array(
                        result["force_mag"][result["mask_mag"].bool().squeeze(-1)]
                    ).reshape(-1, 3),
                )

    def test_dp_test_1_frame(self) -> None:
        self._run_dp_test(False)

    def test_dp_test_input_json(self) -> None:
        self._run_dp_test(True)

    def test_dp_test_input_json_train(self) -> None:
        with open(self.input_json) as f:
            cfg = json.load(f)
        cfg["training"]["validation_data"]["systems"] = ["non-existent"]
        with open(self.input_json, "w") as f:
            json.dump(cfg, f, indent=4)
        self._run_dp_test(True, use_train=True)

    def tearDown(self) -> None:
        for f in os.listdir("."):
            if f.startswith("model") and f.endswith(".pt"):
                os.remove(f)
            if f.startswith(self.detail_file):
                os.remove(f)
            if f in ["lcurve.out", self.input_json]:
                os.remove(f)
            if f in ["stat_files"]:
                shutil.rmtree(f)


class TestDPTestSeA(DPTest, unittest.TestCase):
    def setUp(self) -> None:
        self.detail_file = "test_dp_test_ener_detail"
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        data_file = [str(Path(__file__).parent / "water/data/single")]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file
        self.config["model"] = deepcopy(model_se_e2_a)
        self.input_json = "test_dp_test.json"
        with open(self.input_json, "w") as fp:
            json.dump(self.config, fp, indent=4)


class TestDPTestSeASpin(DPTest, unittest.TestCase):
    def setUp(self) -> None:
        self.detail_file = "test_dp_test_ener_spin_detail"
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        data_file = [str(Path(__file__).parent / "NiO/data/single")]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file
        self.config["model"] = deepcopy(model_spin)
        self.config["model"]["type_map"] = ["Ni", "O", "B"]
        self.input_json = "test_dp_test.json"
        with open(self.input_json, "w") as fp:
            json.dump(self.config, fp, indent=4)


class TestDPTestSeARglob(unittest.TestCase):
    def setUp(self) -> None:
        self.detail_file = "test_dp_test_ener_rglob_detail"
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        data_file = [str(Path(__file__).parent / "water/data/single")]
        self.config["training"]["training_data"]["systems"] = data_file
        root_dir = str(Path(__file__).parent)
        self.config["training"]["validation_data"]["systems"] = root_dir
        self.config["training"]["validation_data"]["rglob_patterns"] = [
            "water/data/single"
        ]
        self.config["model"] = deepcopy(model_se_e2_a)
        self.input_json = "test_dp_test_rglob.json"
        with open(self.input_json, "w") as fp:
            json.dump(self.config, fp, indent=4)

    def test_dp_test_input_json_rglob(self) -> None:
        trainer = get_trainer(deepcopy(self.config))
        with torch.device("cpu"):
            input_dict, _, _ = trainer.get_data(is_train=False)
        input_dict.pop("spin", None)
        model = torch.jit.script(trainer.model)
        tmp_fd, tmp_model_path = tempfile.mkstemp(suffix=".pth")
        os.close(tmp_fd)
        torch.jit.save(model, tmp_model_path)
        dp_test(
            model=tmp_model_path,
            system=None,
            datafile=None,
            valid_json=self.input_json,
            set_prefix="set",
            numb_test=1,
            rand_seed=None,
            shuffle_test=False,
            detail_file=self.detail_file,
            atomic=False,
        )
        os.unlink(tmp_model_path)
        self.assertTrue(os.path.exists(self.detail_file + ".e.out"))

    def tearDown(self) -> None:
        for f in os.listdir("."):
            if f.startswith("model") and f.endswith(".pt"):
                os.remove(f)
            if f.startswith(self.detail_file):
                os.remove(f)
            if f in ["lcurve.out", self.input_json]:
                os.remove(f)
            if f in ["stat_files"]:
                shutil.rmtree(f)


class TestDPTestSeARglobTrain(unittest.TestCase):
    def setUp(self) -> None:
        self.detail_file = "test_dp_test_ener_rglob_train_detail"
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        root_dir = str(Path(__file__).parent)
        self.config["training"]["training_data"]["systems"] = root_dir
        self.config["training"]["training_data"]["rglob_patterns"] = [
            "water/data/single"
        ]
        data_file = [str(Path(__file__).parent / "water/data/single")]
        self.config["training"]["validation_data"]["systems"] = data_file
        self.config["model"] = deepcopy(model_se_e2_a)
        self.input_json = "test_dp_test_rglob_train.json"
        with open(self.input_json, "w") as fp:
            json.dump(self.config, fp, indent=4)

    def test_dp_test_input_json_rglob_train(self) -> None:
        trainer = get_trainer(deepcopy(self.config))
        with torch.device("cpu"):
            input_dict, _, _ = trainer.get_data(is_train=False)
        input_dict.pop("spin", None)
        model = torch.jit.script(trainer.model)
        tmp_fd, tmp_model_path = tempfile.mkstemp(suffix=".pth")
        os.close(tmp_fd)
        torch.jit.save(model, tmp_model_path)
        dp_test(
            model=tmp_model_path,
            system=None,
            datafile=None,
            train_json=self.input_json,
            set_prefix="set",
            numb_test=1,
            rand_seed=None,
            shuffle_test=False,
            detail_file=self.detail_file,
            atomic=False,
        )
        os.unlink(tmp_model_path)
        self.assertTrue(os.path.exists(self.detail_file + ".e.out"))

    def tearDown(self) -> None:
        for f in os.listdir("."):
            if f.startswith("model") and f.endswith(".pt"):
                os.remove(f)
            if f.startswith(self.detail_file):
                os.remove(f)
            if f in ["lcurve.out", self.input_json]:
                os.remove(f)
            if f in ["stat_files"]:
                shutil.rmtree(f)


class TestDPTestForceWeight(DPTest, unittest.TestCase):
    def setUp(self) -> None:
        self.detail_file = "test_dp_test_force_weight_detail"
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        system_dir = self._prepare_weighted_system()
        data_file = [system_dir]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file
        self.config["model"] = deepcopy(model_se_e2_a)
        self.system_dir = system_dir
        self.input_json = "test_dp_test_force_weight.json"
        with open(self.input_json, "w") as fp:
            json.dump(self.config, fp, indent=4)

    def _prepare_weighted_system(self) -> str:
        src = Path(__file__).parent / "water/data/single"
        tmp_dir = tempfile.mkdtemp()
        shutil.copytree(src, tmp_dir, dirs_exist_ok=True)
        set_dir = Path(tmp_dir) / "set.000"
        forces = np.load(set_dir / "force.npy")
        forces[0, :3] += 1.0
        forces[0, -3:] += 10.0
        np.save(set_dir / "force.npy", forces)
        natoms = forces.shape[1] // 3
        atom_pref = np.ones((forces.shape[0], natoms), dtype=forces.dtype)
        atom_pref[:, 0] = 2.0
        atom_pref[:, -1] = 0.0
        np.save(set_dir / "atom_pref.npy", atom_pref)
        return tmp_dir

    def test_force_weight(self) -> None:
        trainer = get_trainer(deepcopy(self.config))
        with torch.device("cpu"):
            trainer.get_data(is_train=False)
        model = torch.jit.script(trainer.model)
        tmp_fd, tmp_model_path = tempfile.mkstemp(suffix=".pth")
        os.close(tmp_fd)
        torch.jit.save(model, tmp_model_path)
        dp = DeepEval(tmp_model_path)
        data = DeepmdData(
            self.system_dir,
            set_prefix="set",
            shuffle_test=False,
            type_map=dp.get_type_map(),
            sort_atoms=False,
        )
        err = build_tester(dp, atomic=False).run(
            data,
            self.system_dir,
            numb_test=1,
            detail_file=None,
        )
        test_data = data.get_test()
        coord = test_data["coord"].reshape([1, -1])
        box = test_data["box"][:1]
        atype = test_data["type"][0]
        ret = dp.eval(
            coord,
            box,
            atype,
            fparam=None,
            aparam=None,
            atomic=False,
            efield=None,
            mixed_type=False,
            spin=None,
        )
        force_pred = ret[1].reshape([1, -1])
        force_true = test_data["force"][:1]
        weight = test_data["atom_pref"][:1]
        diff = force_pred - force_true
        mae_unweighted = np.sum(np.abs(diff)) / diff.size
        rmse_unweighted = np.sqrt(np.sum(diff * diff) / diff.size)
        denom = weight.sum()
        mae_weighted = np.sum(np.abs(diff) * weight) / denom
        rmse_weighted = np.sqrt(np.sum(diff * diff * weight) / denom)
        np.testing.assert_allclose(err["mae_f"][0], mae_unweighted)
        np.testing.assert_allclose(err["rmse_f"][0], rmse_unweighted)
        np.testing.assert_allclose(err["mae_fw"][0], mae_weighted)
        np.testing.assert_allclose(err["rmse_fw"][0], rmse_weighted)
        os.unlink(tmp_model_path)

    def tearDown(self) -> None:
        super().tearDown()
        shutil.rmtree(self.system_dir)


class TestDPTestStress(DPTest, unittest.TestCase):
    """Verify the stress output of ``dp test`` (sigma = virial / volume, eV/Å^3)."""

    def setUp(self) -> None:
        self.detail_file = "test_dp_test_stress_detail"
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        self.system_dir = self._prepare_virial_system()
        data_file = [self.system_dir]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file
        self.config["model"] = deepcopy(model_se_e2_a)
        self.input_json = "test_dp_test_stress.json"
        with open(self.input_json, "w") as fp:
            json.dump(self.config, fp, indent=4)

    def _prepare_virial_system(self) -> str:
        src = Path(__file__).parent / "water/data/single"
        tmp_dir = tempfile.mkdtemp()
        shutil.copytree(src, tmp_dir, dirs_exist_ok=True)
        set_dir = Path(tmp_dir) / "set.000"
        for data_file in set_dir.glob("*.npy"):
            values = np.load(data_file)
            np.save(data_file, np.repeat(values, 3, axis=0))
        nframes = np.load(set_dir / "box.npy").shape[0]
        rng = np.random.default_rng(0)
        np.save(set_dir / "virial.npy", rng.standard_normal((nframes, 9)))
        return tmp_dir

    def test_stress(self) -> None:
        trainer = get_trainer(deepcopy(self.config))
        with torch.device("cpu"):
            trainer.get_data(is_train=False)
        model = torch.jit.script(trainer.model)
        tmp_fd, tmp_model_path = tempfile.mkstemp(suffix=".pth")
        os.close(tmp_fd)
        torch.jit.save(model, tmp_model_path)
        dp = DeepEval(tmp_model_path)

        def run_test(detail_file: str) -> tuple[dict, DeepmdData]:
            data = DeepmdData(
                self.system_dir,
                set_prefix="set",
                shuffle_test=False,
                type_map=dp.get_type_map(),
                sort_atoms=False,
            )
            errors = build_tester(dp, atomic=False).run(
                data,
                self.system_dir,
                numb_test=3,
                detail_file=detail_file,
            )
            return errors, data

        single_detail = f"{self.detail_file}_single_chunk"
        with mock.patch.dict(os.environ, clear=False):
            os.environ.pop("DP_TEST_CHUNK_ATOMS", None)
            single_err, _ = run_test(single_detail)
        with mock.patch.dict(
            os.environ,
            {"DP_TEST_CHUNK_ATOMS": "192"},
            clear=False,
        ):
            err, data = run_test(self.detail_file)
        os.unlink(tmp_model_path)

        self.assertEqual(single_err.keys(), err.keys())
        for key in err:
            np.testing.assert_allclose(single_err[key][0], err[key][0])
            self.assertEqual(single_err[key][1], err[key][1])
        for suffix in ("e", "f", "v", "s"):
            single_path = Path(f"{single_detail}.{suffix}.out")
            chunked_path = Path(f"{self.detail_file}.{suffix}.out")
            np.testing.assert_allclose(
                np.loadtxt(single_path, ndmin=2),
                np.loadtxt(chunked_path, ndmin=2),
            )
            self.assertEqual(
                sum(
                    line.startswith("#")
                    for line in chunked_path.read_text().splitlines()
                ),
                1,
            )

        test_data = data.get_test()
        box = test_data["box"][:3].reshape(-1, 3, 3)
        volume = np.abs(np.linalg.det(box)).reshape(-1, 1)

        stress_out = np.loadtxt(self.detail_file + ".s.out", ndmin=2)
        ref_s, pred_s = stress_out[:, 0:9], stress_out[:, 9:18]
        virial_out = np.loadtxt(self.detail_file + ".v.out", ndmin=2)
        ref_v, pred_v = virial_out[:, 0:9], virial_out[:, 9:18]

        # stress detail is the negative virial detail divided by the cell volume (eV/Å^3)
        np.testing.assert_almost_equal(ref_s, -ref_v / volume)
        np.testing.assert_almost_equal(pred_s, -pred_v / volume)

        # reported MAE/RMSE match the stress arrays (in eV/Å^3)
        diff = pred_s - ref_s
        np.testing.assert_allclose(err["mae_s"][0], np.mean(np.abs(diff)))
        np.testing.assert_allclose(err["rmse_s"][0], np.sqrt(np.mean(diff * diff)))

    def tearDown(self) -> None:
        super().tearDown()
        shutil.rmtree(self.system_dir)


class TestDPTestPropertySeA(unittest.TestCase):
    def setUp(self) -> None:
        self.detail_file = "test_dp_test_property_detail"
        input_json = str(Path(__file__).parent / "property/input.json")
        with open(input_json) as f:
            self.config = json.load(f)
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        self.system_tmpdir = tempfile.TemporaryDirectory()
        shutil.copytree(
            Path(__file__).parent / "property/single",
            self.system_tmpdir.name,
            dirs_exist_ok=True,
        )
        set_dir = Path(self.system_tmpdir.name) / "set.000000"
        global_property = np.load(set_dir / "band_property.npy")
        atom_types = np.load(set_dir / "real_atom_types.npy")
        np.save(
            set_dir / "atom_band_property.npy",
            np.zeros(
                (
                    global_property.shape[0],
                    atom_types.shape[1],
                    global_property.shape[1],
                ),
                dtype=global_property.dtype,
            ),
        )
        data_file = [self.system_tmpdir.name]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file
        self.config["model"] = deepcopy(model_property)
        self.config["model"]["type_map"] = [
            self.config["model"]["type_map"][i] for i in [1, 0, 3, 2]
        ]
        self.datafile = "test_dp_test_property_systems.txt"
        Path(self.datafile).write_text(
            "\n".join(data_file * 2) + "\n",
            encoding="utf-8",
        )
        self.input_json = "test_dp_test_property.json"
        with open(self.input_json, "w") as fp:
            json.dump(self.config, fp, indent=4)

    def test_dp_test_1_frame(self) -> None:
        trainer = get_trainer(deepcopy(self.config))
        with torch.device("cpu"):
            input_dict, label_dict, _ = trainer.get_data(is_train=False)
        input_dict.pop("spin", None)
        result = trainer.model(**input_dict)
        model = torch.jit.script(trainer.model)
        tmp_fd, tmp_model_path = tempfile.mkstemp(suffix=".pth")
        os.close(tmp_fd)
        torch.jit.save(model, tmp_model_path)
        dp_test(
            model=tmp_model_path,
            system=None,
            datafile=self.datafile,
            set_prefix="set",
            numb_test=0,
            rand_seed=None,
            shuffle_test=False,
            detail_file=self.detail_file,
            atomic=True,
        )
        os.unlink(tmp_model_path)
        pred_property = np.loadtxt(self.detail_file + ".property.out.0")[:, 1]
        self.assertTrue(os.path.exists(self.detail_file + ".property.out.1.0"))
        np.testing.assert_almost_equal(
            pred_property,
            to_numpy_array(result[model.get_var_name()])[0],
        )

    def tearDown(self) -> None:
        for f in os.listdir("."):
            if f.startswith("model") and f.endswith(".pt"):
                os.remove(f)
            if f.startswith(self.detail_file):
                os.remove(f)
            if f in ["lcurve.out", self.input_json, self.datafile]:
                os.remove(f)
            if f in ["stat_files"]:
                shutil.rmtree(f)
        self.system_tmpdir.cleanup()


if __name__ == "__main__":
    unittest.main()
