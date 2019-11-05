import dpdata,os,sys,json,unittest
import numpy as np
from deepmd.env import tf
from common import Data,gen_data

from deepmd.RunOptions import RunOptions
from deepmd.DataSystem import DataSystem
from deepmd.DescrptSeA import DescrptSeA
from deepmd.Fitting import PolarFittingSeA
from deepmd.Model import PolarModel
from deepmd.common import j_must_have, j_must_have_d, j_have

global_ener_float_precision = tf.float64
global_tf_float_precision = tf.float64
global_np_float_precision = np.float64

class TestModel(unittest.TestCase):
    def setUp(self) :
        gen_data()

    def test_model(self):
        jfile = 'polar_se_a.json'
        with open(jfile) as fp:
            jdata = json.load (fp)
        run_opt = RunOptions(None) 
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
        
        descrpt = DescrptSeA(jdata['model']['descriptor'])
        fitting = PolarFittingSeA(jdata['model']['fitting_net'], descrpt)
        model = PolarModel(jdata['model'], descrpt, fitting)

        # model._compute_dstats([test_data['coord']], [test_data['box']], [test_data['type']], [test_data['natoms_vec']], [test_data['default_mesh']])
        input_data = {'coord' : [test_data['coord']], 
                      'box': [test_data['box']], 
                      'type': [test_data['type']],
                      'natoms_vec' : [test_data['natoms_vec']],
                      'default_mesh' : [test_data['default_mesh']],
                      'fparam': [test_data['fparam']],
        }
        model._compute_dstats(input_data)

        t_prop_c           = tf.placeholder(tf.float32, [5],    name='t_prop_c')
        t_energy           = tf.placeholder(global_ener_float_precision, [None], name='t_energy')
        t_force            = tf.placeholder(global_tf_float_precision, [None], name='t_force')
        t_virial           = tf.placeholder(global_tf_float_precision, [None], name='t_virial')
        t_atom_ener        = tf.placeholder(global_tf_float_precision, [None], name='t_atom_ener')
        t_coord            = tf.placeholder(global_tf_float_precision, [None], name='i_coord')
        t_type             = tf.placeholder(tf.int32,   [None], name='i_type')
        t_natoms           = tf.placeholder(tf.int32,   [model.ntypes+2], name='i_natoms')
        t_box              = tf.placeholder(global_tf_float_precision, [None, 9], name='i_box')
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
                           suffix = "polar_se_a", 
                           reuse = False)
        polar = model_pred['polar']

        feed_dict_test = {t_prop_c:        test_data['prop_c'],
                          t_coord:         np.reshape(test_data['coord']    [:numb_test, :], [-1]),
                          t_box:           test_data['box']                 [:numb_test, :],
                          t_type:          np.reshape(test_data['type']     [:numb_test, :], [-1]),
                          t_natoms:        test_data['natoms_vec'],
                          t_mesh:          test_data['default_mesh'],
                          is_training:     False}

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        [p] = sess.run([polar], feed_dict = feed_dict_test)

        p = p.reshape([-1])
        refp = [3.39695248e+01,  2.16564043e+01,  8.18501479e-01,  2.16564043e+01,  1.38211789e+01,  5.22775159e-01,  8.18501479e-01,  5.22775159e-01, 1.97847218e-02, 8.08467431e-01,  3.42081126e+00, -2.01072261e-01,  3.42081126e+00, 1.54924596e+01, -9.06153697e-01, -2.01072261e-01, -9.06153697e-01,  5.30193262e-02]

        places = 6
        for ii in range(p.size) :
            self.assertAlmostEqual(p[ii], refp[ii], places = places)


        
