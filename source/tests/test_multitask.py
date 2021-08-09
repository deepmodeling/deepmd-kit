
import dpdata,os,sys,unittest
import numpy as np
from deepmd.env import tf
from common import Data, gen_data, del_data, j_loader,tests_path

from deepmd.utils.data_system import DeepmdDataDocker
from deepmd.descriptor import DescrptSeA
from deepmd.fit import EnerFitting
from deepmd.model import EnerModel
from deepmd.common import j_must_have

GLOBAL_ENER_FLOAT_PRECISION = tf.float64
GLOBAL_TF_FLOAT_PRECISION = tf.float64
GLOBAL_NP_FLOAT_PRECISION = np.float64

def _file_delete(file) :
    if os.path.exists(file):
        os.remove(file)

class TestModel(tf.test.TestCase):
    def setUp(self) :
        self.INPUT = str(tests_path / os.path.join("multi-task", "input.json"))
        self.data_file  = str(tests_path / os.path.join("multi-task", "data"))
        jdata = j_loader(self.INPUT)
        for sub_sys in jdata['training']['training_data']['systems']:
            for i in range(len(sub_sys['data'])):
                sub_sys['data'][i] = str(tests_path / sub_sys['data'][i])
        for sub_sys in jdata['training']['validation_data']['systems']:
            for i in range(len(sub_sys['data'])):
                sub_sys['data'][i] = str(tests_path / sub_sys['data'][i])
        

    def tearDown(self):
        _file_delete("out.json")
        _file_delete("checkpoint")
        _file_delete("lcurve.out")
        _file_delete("model.ckpt.meta")
        _file_delete("model.ckpt.index")
        _file_delete("model.ckpt.data-00000-of-00001")

    def test_model_atom_ener(self):  
        ret = os.system("dp train -mt True " + self.INPUT)
        assert(ret == 0), "DP train error!"
        dd = np.loadtxt("lcurve.out",skiprows=1)[:,:9]
        dd = dd.reshape([-1])
        ref_loss = [0.00000000,1.00000000,307.00000000,307.00000000,156.00000000,156.00000000,0.79700000,0.91400000,0.00100000,100.00000000,
        0.00000000,11.70000000,15.50000000,1.08000000,1.44000000,0.99500000,1.53000000,0.00003500,200.00000000,
        1.00000000,2160.00000000,2160.00000000,156.00000000,156.00000000,0.82100000,0.84500000,0.00000004]

        for ii in range(dd.size):
            self.assertAlmostEqual(dd[ii], ref_loss[ii], places = 8)
