# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import os
import unittest

import numpy as np
from packaging.version import (
    Version,
)

from deepmd.tf.common import (
    clear_session,
)
from deepmd.tf.env import (
    TF_VERSION,
)

# from deepmd.tf.entrypoints.compress import compress
from .common import (
    j_loader,
    run_dp,
    tests_path,
)


def _file_delete(file) -> None:
    if os.path.isdir(file):
        os.rmdir(file)
    elif os.path.isfile(file):
        os.remove(file)


class TestMixedPrecTraining(unittest.TestCase):
    def setUp(self) -> None:
        data_file = str(tests_path / os.path.join("model_compression", "data"))
        self.INPUT = str(tests_path / "input.json")
        jdata = j_loader(
            str(tests_path / os.path.join("model_compression", "input.json"))
        )
        jdata["training"]["training_data"]["systems"] = data_file
        jdata["training"]["validation_data"]["systems"] = data_file
        jdata["training"]["mixed_precision"] = {}
        jdata["training"]["mixed_precision"]["compute_prec"] = "float16"
        jdata["training"]["mixed_precision"]["output_prec"] = "float32"
        with open(self.INPUT, "w") as fp:
            json.dump(jdata, fp, indent=4)

    def test_training(self) -> None:
        _TF_VERSION = Version(TF_VERSION)
        # check the TF_VERSION, when TF < 1.12, mixed precision is not allowed
        if _TF_VERSION >= Version("1.14.0"):
            ret = run_dp("dp train " + self.INPUT)
            np.testing.assert_equal(ret, 0, "DP train failed!")

    def tearDown(self) -> None:
        _file_delete(self.INPUT)
        _file_delete("out.json")
        _file_delete("checkpoint")
        _file_delete("model.ckpt.meta")
        _file_delete("model.ckpt.index")
        _file_delete("model.ckpt.data-00000-of-00001")
        _file_delete("model.ckpt-100.meta")
        _file_delete("model.ckpt-100.index")
        _file_delete("model.ckpt-100.data-00000-of-00001")
        _file_delete("input_v2_compat.json")
        _file_delete("lcurve.out")
        clear_session()
