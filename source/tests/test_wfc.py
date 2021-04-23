import dpdata,os,sys,unittest
import numpy as np
from deepmd.env import tf
from common import Data,gen_data, j_loader

from deepmd.utils.data_system import DataSystem
from deepmd.descriptor import DescrptLocFrame
from deepmd.fit import WFCFitting
from deepmd.model import WFCModel
from deepmd.common import j_must_have

GLOBAL_ENER_FLOAT_PRECISION = tf.float64
GLOBAL_TF_FLOAT_PRECISION = tf.float64
GLOBAL_NP_FLOAT_PRECISION = np.float64

class TestModel(unittest.TestCase):
    def setUp(self) :
        gen_data()

    def test_model(self):
        jfile = 'wfc.json'
        jdata = j_loader(jfile)

        systems = j_must_have(jdata, 'systems')
        set_pfx = j_must_have(jdata, 'set_prefix')
        batch_size = j_must_have(jdata, 'batch_size')
        test_size = j_must_have(jdata, 'numb_test')
        batch_size = 1
        test_size = 1
        stop_batch = j_must_have(jdata, 'stop_batch')
        rcut = j_must_have (jdata['model']['descriptor'], 'rcut')
        
        data = DataSystem(systems, set_pfx, batch_size, test_size, rcut, run_opt = None)
        
        test_data = data.get_test ()
        numb_test = 1
        
        jdata['model']['descriptor'].pop('type', None)        
        jdata['model']['descriptor'].pop('_comment', None)        
        descrpt = DescrptLocFrame(**jdata['model']['descriptor'])
        fitting = WFCFitting(jdata['model']['fitting_net'], descrpt)
        model = WFCModel(descrpt, fitting)

        input_data = {'coord' : [test_data['coord']], 
                      'box': [test_data['box']], 
                      'type': [test_data['type']],
                      'natoms_vec' : [test_data['natoms_vec']],
                      'default_mesh' : [test_data['default_mesh']],
                      'fparam': [test_data['fparam']],
        }
        model._compute_input_stat(input_data)

        t_prop_c           = tf.placeholder(tf.float32, [5],    name='t_prop_c')
        t_energy           = tf.placeholder(GLOBAL_ENER_FLOAT_PRECISION, [None], name='t_energy')
        t_force            = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name='t_force')
        t_virial           = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name='t_virial')
        t_atom_ener        = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name='t_atom_ener')
        t_coord            = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name='i_coord')
        t_type             = tf.placeholder(tf.int32,   [None], name='i_type')
        t_natoms           = tf.placeholder(tf.int32,   [model.ntypes+2], name='i_natoms')
        t_box              = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None, 9], name='i_box')
        t_mesh             = tf.placeholder(tf.int32,   [None], name='i_mesh')
        is_training        = tf.placeholder(tf.bool)
        t_fparam = None

        model_pred \
            = model.build (t_coord, 
                           t_type, 
                           t_natoms, 
                           t_box, 
                           t_mesh,
                           t_fparam,
                           suffix = "wfc", 
                           reuse = False)
        wfc = model_pred['wfc']

        feed_dict_test = {t_prop_c:        test_data['prop_c'],
                          t_coord:         np.reshape(test_data['coord']    [:numb_test, :], [-1]),
                          t_box:           test_data['box']                 [:numb_test, :],
                          t_type:          np.reshape(test_data['type']     [:numb_test, :], [-1]),
                          t_natoms:        test_data['natoms_vec'],
                          t_mesh:          test_data['default_mesh'],
                          is_training:     False}

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        [p] = sess.run([wfc], feed_dict = feed_dict_test)

        p = p.reshape([-1])
        refp = [-9.105016838228578990e-01,7.196284362034099935e-01,-9.548516928185298014e-02,2.764615027095288724e+00,2.661319598995644520e-01,7.579512949131941846e-02,-2.107409067376114997e+00,-1.299080016614967414e-01,-5.962778584850070285e-01,2.913899917663253514e-01,-1.226917174638697094e+00,1.829523069930876655e+00,1.015704024959750873e+00,-1.792333611099589386e-01,5.032898080485321834e-01,1.808561721292949453e-01,2.468863482075112081e+00,-2.566442546384765100e-01,-1.467453783795173994e-01,-1.822963931552128658e+00,5.843600156865462747e-01,-1.493875280832117403e+00,1.693322352814763398e-01,-1.877325443995481624e+00]

        places = 6
        for ii in range(p.size) :
            self.assertAlmostEqual(p[ii], refp[ii], places = places)


        
