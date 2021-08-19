
import dpdata,os,sys,unittest,json
import numpy as np
from deepmd.env import tf
from common import Data, gen_data, del_data, j_loader,tests_path
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
        with open(self.INPUT, "w") as fp:
            json.dump(jdata, fp, indent=4)

    def tearDown(self):
        _file_delete("out.json")
        _file_delete("checkpoint")
        _file_delete("lcurve.out")
        _file_delete("model.ckpt.meta")
        _file_delete("model.ckpt.index")
        _file_delete("model.ckpt.data-00000-of-00001")

    def test_model_atom_ener(self):  
        ret = os.system("dp train -mt " + self.INPUT)
        assert(ret == 0), "DP train error!"
        dd = np.loadtxt("lcurve.out",skiprows=1)[:,:9]
        dd = dd.reshape([-1])
        ref_loss = [0.0,24.8,19.6,10.2,10.1,0.608,0.434,0.002,0.0,
                    100.0,3.79,1.81,0.0575,0.183,0.842,0.0866,3.5e-05,
                    0.0,200.0,2110.0,2110.0,152.0,152.0,1.9,1.98,3.9e-08,1.0]

        for ii in range(dd.size):
            self.assertAlmostEqual(dd[ii], ref_loss[ii], places = 8)
