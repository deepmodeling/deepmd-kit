# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import os
import unittest

from packaging.version import parse as parse_version

from deepmd.tf.env import (
    tf,
)

# from deepmd.tf.entrypoints.compress import compress
from .common import (
    j_loader,
    run_dp,
    tests_path,
)


@unittest.skipIf(
    parse_version(tf.__version__) < parse_version("2"),
    f"The current tf version {tf.__version__} is too low to run the new testing model.",
)
class TestCompressedTrainingSeAtten(unittest.TestCase):
    def setUp(self) -> None:
        data_file = str(tests_path / os.path.join("model_compression", "data"))
        self.input_file = str(tests_path / "input.json")
        self.frozen_model = str(tests_path / "dp-compress-training-original.pb")
        self.compressed_model = str(tests_path / "dp-compress-training-compressed.pb")
        self.frozen_compress_training_model = str(
            tests_path / "dp-compress-training-compress-training.pb"
        )
        self.ckpt_file = str(tests_path / "dp-compress-training.ckpt")
        self.checkpoint_dir = str(tests_path)
        jdata = j_loader(
            str(tests_path / os.path.join("model_compression", "input.json"))
        )
        jdata["model"]["descriptor"] = {}
        jdata["model"]["descriptor"]["type"] = "se_atten_v2"
        jdata["model"]["descriptor"]["sel"] = 20
        jdata["model"]["descriptor"]["attn_layer"] = 0
        jdata["training"]["training_data"]["systems"] = data_file
        jdata["training"]["validation_data"]["systems"] = data_file
        jdata["training"]["save_ckpt"] = self.ckpt_file
        with open(self.input_file, "w") as fp:
            json.dump(jdata, fp, indent=4)

    def test_compressed_training(self) -> None:
        run_dp(f"dp train {self.input_file}")
        run_dp(f"dp freeze -c {self.checkpoint_dir} -o {self.frozen_model}")
        run_dp(f"dp compress -i {self.frozen_model} -o {self.compressed_model}")
        # compress training
        run_dp(f"dp train {self.input_file} -f {self.compressed_model}")
        # restart compress training
        run_dp(f"dp train {self.input_file} -r {self.ckpt_file}")
        # freeze compress training
        run_dp(
            f"dp freeze -c {self.checkpoint_dir} -o {self.frozen_compress_training_model}"
        )
        # it should not be able to compress again
        with self.assertRaises(RuntimeError):
            run_dp(
                f"dp compress -i {self.frozen_compress_training_model} -o {self.compressed_model}"
            )
