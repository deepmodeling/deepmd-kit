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
from deepmd.tf.train.run_options import (
    RunOptions,
)
from deepmd.tf.train.trainer import (
    DPTrainer,
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


def _init_models(model_setup, i):
    data_file = str(tests_path / os.path.join("init_frz_model", "data"))
    frozen_model = str(tests_path / f"init_frz_se_atten{i}.pb")
    ckpt = str(tests_path / f"init_frz_se_atten{i}.ckpt")
    run_opt_ckpt = RunOptions(init_model=ckpt, log_level=20)
    run_opt_frz = RunOptions(init_frz_model=frozen_model, log_level=20)
    INPUT = str(tests_path / "input.json")
    jdata = j_loader(str(tests_path / os.path.join("init_frz_model", "input.json")))
    jdata["training"]["training_data"]["systems"] = data_file
    jdata["training"]["validation_data"]["systems"] = data_file
    jdata["training"]["save_ckpt"] = ckpt
    jdata["model"]["descriptor"]["type"] = "se_atten"
    jdata["model"]["descriptor"]["sel"] = 120
    model_setup(jdata)
    with open(INPUT, "w") as fp:
        json.dump(jdata, fp, indent=4)
    ret = run_dp("dp train " + INPUT)
    np.testing.assert_equal(ret, 0, "DP train failed!")
    ret = run_dp("dp freeze -c " + str(tests_path) + " -o " + frozen_model)
    np.testing.assert_equal(ret, 0, "DP freeze failed!")

    jdata = update_deepmd_input(jdata, warning=True, dump=f"input_v2_compat{i}.json")
    jdata = normalize(jdata)
    model_ckpt = DPTrainer(jdata, run_opt=run_opt_ckpt)
    model_frz = DPTrainer(jdata, run_opt=run_opt_frz)
    rcut = model_ckpt.model.get_rcut()
    type_map = model_ckpt.model.get_type_map()
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
        },
        "force": {
            "ndof": 3,
            "atomic": True,
            "must": False,
            "high_prec": False,
            "type_sel": None,
            "repeat": 1,
            "default": 0.0,
        },
        "virial": {
            "ndof": 9,
            "atomic": False,
            "must": False,
            "high_prec": False,
            "type_sel": None,
            "repeat": 1,
            "default": 0.0,
        },
        "atom_ener": {
            "ndof": 1,
            "atomic": True,
            "must": False,
            "high_prec": False,
            "type_sel": None,
            "repeat": 1,
            "default": 0.0,
        },
        "atom_pref": {
            "ndof": 1,
            "atomic": True,
            "must": False,
            "high_prec": False,
            "type_sel": None,
            "repeat": 3,
            "default": 0.0,
        },
    }
    data.add_dict(data_requirement)
    stop_batch = jdata["training"]["numb_steps"]

    return INPUT, ckpt, frozen_model, model_ckpt, model_frz, data, stop_batch


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


@unittest.skipIf(
    parse_version(tf.__version__) < parse_version("1.15"),
    f"The current tf version {tf.__version__} is too low to run the new testing model.",
)
class TestInitFrzModelAtten(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        models = [previous_se_atten, stripped_model, compressible_model]
        INPUTS = []
        CKPTS = []
        FROZEN_MODELS = []
        CKPT_TRAINERS = []
        FRZ_TRAINERS = []
        VALID_DATAS = []
        STOP_BATCHS = []
        for i, model in enumerate(models):
            (
                INPUT,
                CKPT,
                FROZEN_MODEL,
                CKPT_TRAINER,
                FRZ_TRAINER,
                VALID_DATA,
                STOP_BATCH,
            ) = _init_models(model, i)
            INPUTS.append(INPUT)
            CKPTS.append(CKPT)
            FROZEN_MODELS.append(FROZEN_MODEL)
            CKPT_TRAINERS.append(CKPT_TRAINER)
            FRZ_TRAINERS.append(FRZ_TRAINER)
            VALID_DATAS.append(VALID_DATA)
            STOP_BATCHS.append(STOP_BATCH)
        cls.INPUTS = INPUTS
        cls.CKPTS = CKPTS
        cls.FROZEN_MODELS = FROZEN_MODELS
        cls.CKPT_TRAINERS = CKPT_TRAINERS
        cls.FRZ_TRAINERS = FRZ_TRAINERS
        cls.VALID_DATAS = VALID_DATAS
        cls.STOP_BATCHS = STOP_BATCHS
        cls.dp_ckpts = CKPT_TRAINERS
        cls.dp_frzs = FRZ_TRAINERS
        cls.valid_datas = VALID_DATAS
        cls.stop_batchs = STOP_BATCHS

    @classmethod
    def tearDownClass(cls) -> None:
        for i in range(len(cls.dp_ckpts)):
            _file_delete(cls.INPUTS[i])
            _file_delete(cls.FROZEN_MODELS[i])
            _file_delete("out.json")
            _file_delete(str(tests_path / "checkpoint"))
            _file_delete(cls.CKPTS[i] + ".meta")
            _file_delete(cls.CKPTS[i] + ".index")
            _file_delete(cls.CKPTS[i] + ".data-00000-of-00001")
            _file_delete(cls.CKPTS[i] + "-0.meta")
            _file_delete(cls.CKPTS[i] + "-0.index")
            _file_delete(cls.CKPTS[i] + "-0.data-00000-of-00001")
            _file_delete(cls.CKPTS[i] + "-1.meta")
            _file_delete(cls.CKPTS[i] + "-1.index")
            _file_delete(cls.CKPTS[i] + "-1.data-00000-of-00001")
            _file_delete(f"input_v2_compat{i}.json")
            _file_delete("lcurve.out")

    def test_single_frame(self) -> None:
        for i in range(len(self.dp_ckpts)):
            self.dp_ckpt = self.CKPT_TRAINERS[i]
            self.dp_frz = self.FRZ_TRAINERS[i]
            self.valid_data = self.VALID_DATAS[i]
            self.stop_batch = self.STOP_BATCHS[i]

            valid_batch = self.valid_data.get_batch()
            natoms = valid_batch["natoms_vec"]
            tf.reset_default_graph()
            self.dp_ckpt.build(self.valid_data, self.stop_batch)
            self.dp_ckpt._init_session()
            feed_dict_ckpt = self.dp_ckpt.get_feed_dict(valid_batch, is_training=False)
            ckpt_rmse_ckpt = self.dp_ckpt.loss.eval(
                self.dp_ckpt.sess, feed_dict_ckpt, natoms
            )
            tf.reset_default_graph()

            self.dp_frz.build(self.valid_data, self.stop_batch)
            self.dp_frz._init_session()
            feed_dict_frz = self.dp_frz.get_feed_dict(valid_batch, is_training=False)
            ckpt_rmse_frz = self.dp_frz.loss.eval(
                self.dp_frz.sess, feed_dict_frz, natoms
            )
            tf.reset_default_graph()

            # check values
            np.testing.assert_almost_equal(
                ckpt_rmse_ckpt["rmse_e"], ckpt_rmse_frz["rmse_e"], default_places
            )
            np.testing.assert_almost_equal(
                ckpt_rmse_ckpt["rmse_f"], ckpt_rmse_frz["rmse_f"], default_places
            )
            np.testing.assert_almost_equal(
                ckpt_rmse_ckpt["rmse_v"], ckpt_rmse_frz["rmse_v"], default_places
            )
