
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
        self.INPUT = str(tests_path / 'input_mt.json')
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
        dd = dd.reshape([3,-1])
                
        ref_loss = [0.0,307.0,307.0,156.0,156.0,0.765,0.809,0.001,1.0,1.0,2150.0,2150.0,156.0,156.0,0.792,0.813,5.9e-06,1.0,
        2.0,2160.0,2160.0,156.0,156.0,0.829,0.787,3.5e-08,1.0]

        for ii in range(3):
            for jj in range(9):
                self.assertAlmostEqual(dd[ii][jj], ref_loss[ii*9+jj], places = 8)
