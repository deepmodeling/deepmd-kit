import os
import shutil
import numpy as np
import unittest
import subprocess as sp

from deepmd.env import tf
from deepmd.infer import DeepPot
from common import tests_path
from infer.convert2pb import convert_pbtxt_to_pb
from deepmd.utils.graph import get_tensor_by_name

from deepmd.env import GLOBAL_NP_FLOAT_PRECISION

def _file_delete(file) :
    if os.path.exists(file):
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

class TestTransform(unittest.TestCase) :
    @classmethod
    def setUpClass(self):
        self.env = os.environ.get("DP_INTERFACE_PREC")
        os.environ["DP_INTERFACE_PREC"] = "ascend_mix"
        self.old_model = str(tests_path / "dp-old.pb")
        self.new_model = str(tests_path / "dp-ascend.pb")
        convert_pbtxt_to_pb(str(tests_path / os.path.join("infer","deeppot-2.pbtxt")), self.old_model)
        ret = _subprocess_run("dp transfer-to-ascend mix_precision -i " + self.old_model + " -o " + self.new_model)
        np.testing.assert_equal(ret, 0, 'DP transfer failed!')

        self.dp = DeepPot(self.new_model)
        self.dp_org = DeepPot(self.old_model)
        self.coords = np.array([12.83, 2.56, 2.18,
                                12.09, 2.87, 2.74,
                                00.25, 3.32, 1.68,
                                3.36, 3.00, 1.81,
                                3.51, 2.51, 2.60,
                                4.27, 3.22, 1.56])
        self.atype = [0, 1, 1, 0, 1, 1]
        self.box = np.array([13., 0., 0., 0., 13., 0., 0., 0., 13.])

    @classmethod
    def tearDownClass(self):
        _file_delete(self.old_model)
        _file_delete(self.new_model)
        _file_delete("ascend-transfer.json")
        _file_delete("input_v2_compat.json")
        _file_delete("lcurve.out")
        shutil.rmtree("model-transfer")
        if self.env:
            os.environ["DP_INTERFACE_PREC"] = self.env
        else:
            del os.environ['DP_INTERFACE_PREC']


    def test_attrs(self):
        self.assertEqual(self.dp.get_ntypes(), 2)
        self.assertAlmostEqual(self.dp.get_rcut(), 6.0, places = 4)
        self.assertEqual(self.dp.get_type_map(), ['O', 'H'])
        self.assertEqual(self.dp.get_dim_fparam(), 0)
        self.assertEqual(self.dp.get_dim_aparam(), 0)
        t_model_type = bytes.decode(get_tensor_by_name(self.new_model, 'model_type'))
        self.assertEqual(t_model_type, 'ascend_transfer_model')

    def test_1frame_atm(self):
        ee, ff, vv, ae, av = self.dp.eval(self.coords, self.box, self.atype, atomic = True)
        ee_org, ff_org, vv_org, ae_org, av_org = self.dp_org.eval(self.coords, self.box, self.atype, atomic = True)
        # check shape of the returns
        nframes = 1
        natoms = len(self.atype)
        self.assertEqual(ee.shape, (nframes,1))
        self.assertEqual(ff.shape, (nframes,natoms,3))
        self.assertEqual(vv.shape, (nframes,9))
        self.assertEqual(ae.shape, (nframes,natoms,1))
        self.assertEqual(av.shape, (nframes,natoms,9))
        # check values
        np.testing.assert_almost_equal(ee.ravel(), ee_org.ravel(), 1)
        np.testing.assert_almost_equal(ff.ravel(), ff_org.ravel(), 1)
        np.testing.assert_almost_equal(vv.ravel(), vv.ravel(), 1)
        np.testing.assert_almost_equal(ae.ravel(), ae_org.ravel(), 1)
        np.testing.assert_almost_equal(av.ravel(), av_org.ravel(), 1)

