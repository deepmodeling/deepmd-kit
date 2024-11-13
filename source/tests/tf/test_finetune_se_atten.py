# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import os
import unittest

import numpy as np
from packaging.version import parse as parse_version

from deepmd.tf.env import (
    GLOBAL_NP_FLOAT_PRECISION,
    tf,
)
from deepmd.tf.infer import (
    DeepPotential,
)
from deepmd.tf.utils.argcheck import (
    normalize,
)
from deepmd.tf.utils.compat import (
    update_deepmd_input,
)
from deepmd.tf.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.tf.utils.graph import (
    get_tensor_by_name,
)

from .common import (
    j_loader,
    run_dp,
    tests_path,
)

if GLOBAL_NP_FLOAT_PRECISION == np.float32:
    default_places = 4
else:
    default_places = 10


def _file_delete(file) -> None:
    if os.path.isdir(file):
        os.rmdir(file)
    elif os.path.isfile(file):
        os.remove(file)


def _init_models(setup_model, i):
    data_file = str(tests_path / os.path.join("finetune", "data"))
    data_file_mixed_type = str(tests_path / os.path.join("finetune", "data_mixed_type"))
    pretrained_model = str(tests_path / "pretrained_model_se_atten.pb")
    finetuned_model = str(tests_path / "finetuned_model_se_atten.pb")
    # ckpt_pretrain = str(tests_path / f"pretrain{i}.ckpt")
    # ckpt_finetune = str(tests_path / f"finetune{i}.ckpt")
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
    # jdata_pre["training"]["save_ckpt"] = ckpt_pretrain
    setup_model(jdata_pre)

    jdata_finetune["training"]["training_data"]["systems"] = data_file
    jdata_finetune["training"]["validation_data"]["systems"] = data_file
    # jdata_finetune["training"]["save_ckpt"] = ckpt_finetune
    setup_model(jdata_finetune)
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


@unittest.skipIf(
    parse_version(tf.__version__) < parse_version("1.15"),
    f"The current tf version {tf.__version__} is too low to run the new testing model.",
)
class TestFinetuneSeAtten(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not parse_version(tf.__version__) < parse_version("1.15"):

            def previous_se_atten(jdata) -> None:
                jdata["model"]["descriptor"]["tebd_input_mode"] = "concat"
                jdata["model"]["descriptor"]["attn_layer"] = 2

            def stripped_model(jdata) -> None:
                jdata["model"]["descriptor"]["tebd_input_mode"] = "strip"
                jdata["model"]["descriptor"]["attn_layer"] = 2

            def compressible_model(jdata) -> None:
                jdata["model"]["descriptor"]["tebd_input_mode"] = "strip"
                jdata["model"]["descriptor"]["attn_layer"] = 0

            models = [previous_se_atten, stripped_model, compressible_model]
            INPUT_PRES = []
            INPUT_FINETUNES = []
            INPUT_FINETUNE_MIXS = []
            PRE_MODELS = []
            FINETUNED_MODELS = []
            FINETUNED_MODEL_MIXS = []
            PRE_MAPS = []
            FINETUNED_MAPS = []
            VALID_DATAS = []
            for i, model in enumerate(models):
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
                ) = _init_models(model, i)
                INPUT_PRES.append(INPUT_PRE)
                INPUT_FINETUNES.append(INPUT_FINETUNE)
                INPUT_FINETUNE_MIXS.append(INPUT_FINETUNE_MIX)
                PRE_MODELS.append(PRE_MODEL)
                FINETUNED_MODELS.append(FINETUNED_MODEL)
                FINETUNED_MODEL_MIXS.append(FINETUNED_MODEL_MIX)
                PRE_MAPS.append(PRE_MAP)
                FINETUNED_MAPS.append(FINETUNED_MAP)
                VALID_DATAS.append(VALID_DATA)
        cls.INPUT_PRES = INPUT_PRES
        cls.INPUT_FINETUNES = INPUT_FINETUNES
        cls.INPUT_FINETUNE_MIXS = INPUT_FINETUNE_MIXS
        cls.PRE_MODELS = PRE_MODELS
        cls.FINETUNED_MODELS = FINETUNED_MODELS
        cls.FINETUNED_MODEL_MIXS = FINETUNED_MODEL_MIXS
        cls.PRE_MAPS = PRE_MAPS
        cls.FINETUNED_MAPS = FINETUNED_MAPS
        cls.VALID_DATAS = VALID_DATAS

    @classmethod
    def tearDownClass(cls) -> None:
        for i in range(len(cls.INPUT_PRES)):
            _file_delete(cls.INPUT_PRES[i])
            _file_delete(cls.INPUT_FINETUNES[i])
            _file_delete(cls.INPUT_FINETUNE_MIXS[i])
            _file_delete(cls.PRE_MODELS[i])
            _file_delete(cls.FINETUNED_MODELS[i])
            _file_delete(cls.FINETUNED_MODEL_MIXS[i])
            _file_delete("out.json")
            _file_delete("model.ckpt.meta")
            _file_delete("model.ckpt.index")
            _file_delete("model.ckpt.data-00000-of-00001")
            _file_delete("model.ckpt-0.meta")
            _file_delete("model.ckpt-0.index")
            _file_delete("model.ckpt-0.data-00000-of-00001")
            _file_delete("model.ckpt-1.meta")
            _file_delete("model.ckpt-1.index")
            _file_delete("model.ckpt-1.data-00000-of-00001")
            _file_delete(str(tests_path / "checkpoint"))
            _file_delete("input_v2_compat.json")
            _file_delete("lcurve.out")

    def test_finetune_standard(self) -> None:
        for i in range(len(self.INPUT_PRES)):
            self.valid_data = self.VALID_DATAS[i]
            pretrained_bias = get_tensor_by_name(
                self.PRE_MODELS[i], "fitting_attr/t_bias_atom_e"
            )
            finetuned_bias = get_tensor_by_name(
                self.FINETUNED_MODELS[i], "fitting_attr/t_bias_atom_e"
            )
            sorter = np.argsort(self.PRE_MAPS[i])
            idx_type_map = sorter[
                np.searchsorted(self.PRE_MAPS[i], self.FINETUNED_MAPS[i], sorter=sorter)
            ]
            test_data = self.valid_data.get_test()
            atom_nums = np.tile(np.bincount(test_data["type"][0])[idx_type_map], (4, 1))

            dp = DeepPotential(self.PRE_MODELS[i])
            energy = dp.eval(
                test_data["coord"], test_data["box"], test_data["type"][0]
            )[0]
            energy_diff = test_data["energy"] - energy
            finetune_shift = (
                finetuned_bias[idx_type_map] - pretrained_bias[idx_type_map]
            )
            ground_truth_shift = np.linalg.lstsq(atom_nums, energy_diff, rcond=None)[
                0
            ].reshape(-1)

            dp_finetuned = DeepPotential(self.FINETUNED_MODELS[i])
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

    def test_finetune_mixed_type(self) -> None:
        for i in range(len(self.INPUT_PRES)):
            self.valid_data = self.VALID_DATAS[i]
            pretrained_bias = get_tensor_by_name(
                self.PRE_MODELS[i], "fitting_attr/t_bias_atom_e"
            )
            finetuned_bias_mixed_type = get_tensor_by_name(
                self.FINETUNED_MODEL_MIXS[i], "fitting_attr/t_bias_atom_e"
            )
            sorter = np.argsort(self.PRE_MAPS[i])
            idx_type_map = sorter[
                np.searchsorted(self.PRE_MAPS[i], self.FINETUNED_MAPS[i], sorter=sorter)
            ]
            test_data = self.valid_data.get_test()
            atom_nums = np.tile(np.bincount(test_data["type"][0])[idx_type_map], (4, 1))

            dp = DeepPotential(self.PRE_MODELS[i])
            energy = dp.eval(
                test_data["coord"], test_data["box"], test_data["type"][0]
            )[0]
            energy_diff = test_data["energy"] - energy
            finetune_shift = (
                finetuned_bias_mixed_type[idx_type_map] - pretrained_bias[idx_type_map]
            )
            ground_truth_shift = np.linalg.lstsq(atom_nums, energy_diff, rcond=None)[
                0
            ].reshape(-1)

            dp_finetuned_mixed_type = DeepPotential(self.FINETUNED_MODEL_MIXS[i])
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
