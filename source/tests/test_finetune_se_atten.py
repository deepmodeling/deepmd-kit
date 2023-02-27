import json
import os
import subprocess as sp
import unittest

import numpy as np
from common import (
    j_loader,
    run_dp,
    tests_path,
)
from packaging.version import parse as parse_version

from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
    tf,
)
from deepmd.infer import (
    DeepPotential,
)
from deepmd.utils.argcheck import (
    normalize,
)
from deepmd.utils.compat import (
    update_deepmd_input,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.utils.graph import (
    get_tensor_by_name,
)

if GLOBAL_NP_FLOAT_PRECISION == np.float32:
    default_places = 4
else:
    default_places = 10


def _file_delete(file):
    if os.path.isdir(file):
        os.rmdir(file)
    elif os.path.isfile(file):
        os.remove(file)


def _subprocess_run(command):
    popen = sp.Popen(command.split(), shell=False, stdout=sp.PIPE, stderr=sp.STDOUT)
    for line in iter(popen.stdout.readline, b""):
        if hasattr(line, "decode"):
            line = line.decode("utf-8")
        line = line.rstrip()
        print(line)
    popen.wait()
    return popen.returncode


def _init_models():
    data_file = str(tests_path / os.path.join("finetune", "data"))
    data_file_mixed_type = str(tests_path / os.path.join("finetune", "data_mixed_type"))
    pretrained_model = str(tests_path / "pretrained_model_se_atten.pb")
    finetuned_model = str(tests_path / "finetuned_model_se_atten.pb")
    finetuned_model_mixed_type = str(
        tests_path / "finetuned_model_se_atten_mixed_type.pb"
    )
    INPUT_PRE = str(tests_path / "input_pretrain_se_atten.json")
    INPUT_FINETUNE = str(tests_path / "input_finetune_se_atten.json")
    INPUT_FINETUNE_MIX = str(tests_path / "input_finetune_se_atten_mixed_type.json")
    jdata_pre = j_loader(
        str(tests_path / os.path.join("finetune", "input_pretrain.json"))
    )
    jdata_finetune = j_loader(
        str(tests_path / os.path.join("finetune", "input_finetune.json"))
    )
    jdata_pre["training"]["training_data"]["systems"] = data_file
    jdata_pre["training"]["validation_data"]["systems"] = data_file
    jdata_finetune["training"]["training_data"]["systems"] = data_file
    jdata_finetune["training"]["validation_data"]["systems"] = data_file
    type_map_pre = jdata_pre["model"]["type_map"]
    type_map_finetune = jdata_finetune["model"]["type_map"]
    with open(INPUT_PRE, "w") as fp:
        json.dump(jdata_pre, fp, indent=4)
    with open(INPUT_FINETUNE, "w") as fp:
        json.dump(jdata_finetune, fp, indent=4)
    jdata_finetune["training"]["training_data"]["systems"] = data_file_mixed_type
    jdata_finetune["training"]["validation_data"]["systems"] = data_file_mixed_type
    with open(INPUT_FINETUNE_MIX, "w") as fp:
        json.dump(jdata_finetune, fp, indent=4)

    ret = run_dp("dp train " + INPUT_PRE)
    np.testing.assert_equal(ret, 0, "DP train failed!")
    ret = run_dp("dp freeze -o " + pretrained_model)
    np.testing.assert_equal(ret, 0, "DP freeze failed!")
    ret = run_dp("dp train " + INPUT_FINETUNE + " -t " + pretrained_model)
    np.testing.assert_equal(ret, 0, "DP finetune failed!")
    ret = run_dp("dp freeze -o " + finetuned_model)
    np.testing.assert_equal(ret, 0, "DP freeze failed!")
    ret = run_dp("dp train " + INPUT_FINETUNE_MIX + " -t " + pretrained_model)
    np.testing.assert_equal(ret, 0, "DP finetune failed!")
    ret = run_dp("dp freeze -o " + finetuned_model_mixed_type)
    np.testing.assert_equal(ret, 0, "DP freeze failed!")

    jdata_pre = update_deepmd_input(
        jdata_pre, warning=True, dump="input_v2_compat.json"
    )
    jdata_pre = normalize(jdata_pre)
    rcut = jdata_pre["model"]["descriptor"]["rcut"]
    type_map = jdata_pre["model"]["type_map"]
    data = DeepmdDataSystem(
        systems=[data_file],
        batch_size=1,
        test_size=1,
        rcut=rcut,
        type_map=type_map,
        trn_all_set=True,
    )
    data_requirement = {
        "energy": {
            "ndof": 1,
            "atomic": False,
            "must": False,
            "high_prec": True,
            "type_sel": None,
            "repeat": 1,
            "default": 0.0,
        }
    }
    data.add_dict(data_requirement)
    return (
        INPUT_PRE,
        INPUT_FINETUNE,
        INPUT_FINETUNE_MIX,
        pretrained_model,
        finetuned_model,
        finetuned_model_mixed_type,
        type_map_pre,
        type_map_finetune,
        data,
    )


if not parse_version(tf.__version__) < parse_version("1.15"):
    (
        INPUT_PRE,
        INPUT_FINETUNE,
        INPUT_FINETUNE_MIX,
        PRE_MODEL,
        FINETUNED_MODEL,
        FINETUNED_MODEL_MIX,
        PRE_MAP,
        FINETUNED_MAP,
        VALID_DATA,
    ) = _init_models()


@unittest.skipIf(
    parse_version(tf.__version__) < parse_version("1.15"),
    f"The current tf version {tf.__version__} is too low to run the new testing model.",
)
class TestFinetuneSeAtten(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.valid_data = VALID_DATA

    @classmethod
    def tearDownClass(self):
        _file_delete(INPUT_PRE)
        _file_delete(INPUT_FINETUNE)
        _file_delete(INPUT_FINETUNE_MIX)
        _file_delete(PRE_MODEL)
        _file_delete(FINETUNED_MODEL)
        _file_delete(FINETUNED_MODEL_MIX)
        _file_delete("out.json")
        _file_delete(str(tests_path / "checkpoint"))
        _file_delete("model.ckpt.meta")
        _file_delete("model.ckpt.index")
        _file_delete("model.ckpt.data-00000-of-00001")
        _file_delete("model.ckpt-0.meta")
        _file_delete("model.ckpt-0.index")
        _file_delete("model.ckpt-0.data-00000-of-00001")
        _file_delete("model.ckpt-1.meta")
        _file_delete("model.ckpt-1.index")
        _file_delete("model.ckpt-1.data-00000-of-00001")
        _file_delete("input_v2_compat.json")
        _file_delete("lcurve.out")

    def test_finetune_standard(self):
        pretrained_bias = get_tensor_by_name(PRE_MODEL, "fitting_attr/t_bias_atom_e")
        finetuned_bias = get_tensor_by_name(
            FINETUNED_MODEL, "fitting_attr/t_bias_atom_e"
        )
        sorter = np.argsort(PRE_MAP)
        idx_type_map = sorter[np.searchsorted(PRE_MAP, FINETUNED_MAP, sorter=sorter)]
        test_data = self.valid_data.get_test()
        atom_nums = np.tile(np.bincount(test_data["type"][0])[idx_type_map], (4, 1))

        dp = DeepPotential(PRE_MODEL)
        energy = dp.eval(test_data["coord"], test_data["box"], test_data["type"][0])[0]
        energy_diff = test_data["energy"] - energy
        finetune_shift = finetuned_bias[idx_type_map] - pretrained_bias[idx_type_map]
        ground_truth_shift = np.linalg.lstsq(atom_nums, energy_diff, rcond=None)[
            0
        ].reshape(-1)

        dp_finetuned = DeepPotential(FINETUNED_MODEL)
        energy_finetuned = dp_finetuned.eval(
            test_data["coord"], test_data["box"], test_data["type"][0]
        )[0]
        energy_diff_finetuned = test_data["energy"] - energy_finetuned
        finetune_results = np.linalg.lstsq(
            atom_nums, energy_diff_finetuned, rcond=None
        )[0].reshape(-1)

        # check values
        np.testing.assert_almost_equal(
            finetune_shift, ground_truth_shift, default_places
        )
        np.testing.assert_almost_equal(finetune_results, 0.0, default_places)

    def test_finetune_mixed_type(self):
        pretrained_bias = get_tensor_by_name(PRE_MODEL, "fitting_attr/t_bias_atom_e")
        finetuned_bias_mixed_type = get_tensor_by_name(
            FINETUNED_MODEL_MIX, "fitting_attr/t_bias_atom_e"
        )
        sorter = np.argsort(PRE_MAP)
        idx_type_map = sorter[np.searchsorted(PRE_MAP, FINETUNED_MAP, sorter=sorter)]
        test_data = self.valid_data.get_test()
        atom_nums = np.tile(np.bincount(test_data["type"][0])[idx_type_map], (4, 1))

        dp = DeepPotential(PRE_MODEL)
        energy = dp.eval(test_data["coord"], test_data["box"], test_data["type"][0])[0]
        energy_diff = test_data["energy"] - energy
        finetune_shift = (
            finetuned_bias_mixed_type[idx_type_map] - pretrained_bias[idx_type_map]
        )
        ground_truth_shift = np.linalg.lstsq(atom_nums, energy_diff, rcond=None)[
            0
        ].reshape(-1)

        dp_finetuned_mixed_type = DeepPotential(FINETUNED_MODEL_MIX)
        energy_finetuned = dp_finetuned_mixed_type.eval(
            test_data["coord"], test_data["box"], test_data["type"][0]
        )[0]
        energy_diff_finetuned = test_data["energy"] - energy_finetuned
        finetune_results = np.linalg.lstsq(
            atom_nums, energy_diff_finetuned, rcond=None
        )[0].reshape(-1)

        # check values
        np.testing.assert_almost_equal(
            finetune_shift, ground_truth_shift, default_places
        )
        np.testing.assert_almost_equal(finetune_results, 0.0, default_places)
