import os,json
import numpy as np
import unittest
import subprocess as sp
from packaging.version import Version

from deepmd.infer import DeepPot
# from deepmd.entrypoints.compress import compress
from common import j_loader, tests_path
from deepmd.env import TF_VERSION


def _file_delete(file) :
    if os.path.isdir(file):
        os.rmdir(file)
    elif os.path.isfile(file):
        os.remove(file)

def _subprocess_run(command):
    popen = sp.Popen(command.split(), shell=False, stdout=sp.PIPE, stderr=sp.STDOUT)
    for line in iter(popen.stdout.readline, b''):
        if hasattr(line, 'decode'):
            line = line.decode('utf-8')
        line = line.rstrip()
        print(line)
    popen.wait()
    return popen.returncode

class TestMixedPrecTraining(unittest.TestCase):
    def setUp(self):
        data_file  = str(tests_path / os.path.join("model_compression", "data"))
        self.INPUT = str(tests_path / "input.json")
        jdata = j_loader(str(tests_path / os.path.join("model_compression", "input.json")))
        jdata["training"]["training_data"]["systems"] = data_file
        jdata["training"]["validation_data"]["systems"] = data_file
        jdata["training"]["mixed_precision"] = {}
        jdata["training"]["mixed_precision"]["compute_prec"] = "float16"
        jdata["training"]["mixed_precision"]["output_prec"] = "float32"
        with open(self.INPUT, "w") as fp:
            json.dump(jdata, fp, indent=4)

    def test_training(self):
        _TF_VERSION = Version(TF_VERSION)
        # check the TF_VERSION, when TF < 1.12, mixed precision is not allowed 
        if _TF_VERSION >= Version('1.14.0'):
            ret = _subprocess_run("dp train " + self.INPUT)
            np.testing.assert_equal(ret, 0, 'DP train failed!')

    def tearDown(self):
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
