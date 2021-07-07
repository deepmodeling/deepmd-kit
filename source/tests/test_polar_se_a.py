import dpdata,os,sys,unittest
import numpy as np
from deepmd.env import tf
from common import Data,gen_data, j_loader
from common import finite_difference, strerch_box

from deepmd.utils.data_system import DataSystem
from deepmd.descriptor import DescrptSeA
from deepmd.fit import PolarFittingSeA
from deepmd.model import PolarModel
from deepmd.common import j_must_have

GLOBAL_ENER_FLOAT_PRECISION = tf.float64
GLOBAL_TF_FLOAT_PRECISION = tf.float64
GLOBAL_NP_FLOAT_PRECISION = np.float64

class TestModel(tf.test.TestCase):
    def setUp(self) :
        gen_data()

    def test_model(self):
        jfile = 'polar_se_a.json'
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
        jdata['model']['fitting_net'].pop('type', None)
        descrpt = DescrptSeA(**jdata['model']['descriptor'], uniform_seed = True)
        jdata['model']['fitting_net']['descrpt'] = descrpt
        fitting = PolarFittingSeA(**jdata['model']['fitting_net'], uniform_seed = True)
        model = PolarModel(descrpt, fitting)

        # model._compute_dstats([test_data['coord']], [test_data['box']], [test_data['type']], [test_data['natoms_vec']], [test_data['default_mesh']])
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
                           suffix = "polar_se_a", 
                           reuse = False)
        polar = model_pred['polar']
        gpolar = model_pred['global_polar']
        force = model_pred['force']
        virial = model_pred['virial']
        atom_virial = model_pred['atom_virial']

        feed_dict_test = {t_prop_c:        test_data['prop_c'],
                          t_coord:         np.reshape(test_data['coord']    [:numb_test, :], [-1]),
                          t_box:           test_data['box']                 [:numb_test, :],
                          t_type:          np.reshape(test_data['type']     [:numb_test, :], [-1]),
                          t_natoms:        test_data['natoms_vec'],
                          t_mesh:          test_data['default_mesh'],
                          is_training:     False}

        sess = self.test_session().__enter__()
        sess.run(tf.global_variables_initializer())
        [p, gp] = sess.run([polar, gpolar], feed_dict = feed_dict_test)

        p = p.reshape([-1])
        refp = [3.39695248e+01,  2.16564043e+01,  8.18501479e-01,  2.16564043e+01,  1.38211789e+01,  5.22775159e-01,  8.18501479e-01,  5.22775159e-01, 1.97847218e-02, 8.08467431e-01,  3.42081126e+00, -2.01072261e-01,  3.42081126e+00, 1.54924596e+01, -9.06153697e-01, -2.01072261e-01, -9.06153697e-01,  5.30193262e-02]

        places = 6
        for ii in range(p.size) :
            self.assertAlmostEqual(p[ii], refp[ii], places = places)

        gp = gp.reshape([-1])
        refgp = np.array(refp).reshape(-1, 9).sum(0)
        
        places = 5
        for ii in range(gp.size) :
            self.assertAlmostEqual(gp[ii], refgp[ii], places = places)

        # make sure only one frame is used
        feed_dict_single = {t_prop_c:        test_data['prop_c'],
                            t_coord:         np.reshape(test_data['coord']    [:1, :], [-1]),
                            t_box:           test_data['box']                 [:1, :],
                            t_type:          np.reshape(test_data['type']     [:1, :], [-1]),
                            t_natoms:        test_data['natoms_vec'],
                            t_mesh:          test_data['default_mesh'],
                            is_training:     False}

        [pf, pv, pav] = sess.run([force, virial, atom_virial], feed_dict = feed_dict_single)
        pf, pv = pf.reshape(-1), pv.reshape(-1)
        spv = pav.reshape(1, 9, -1, 9).sum(2).reshape(-1)

        base_dict = feed_dict_single.copy()
        coord0 = base_dict.pop(t_coord)
        box0 = base_dict.pop(t_box)

        fdf = - finite_difference(
                    lambda coord: sess.run(gpolar, 
                        feed_dict={**base_dict, 
                                t_coord:coord, 
                                t_box:box0}).reshape(-1),
                    test_data['coord'][:numb_test, :].reshape([-1])).reshape(-1)
        fdv = - (finite_difference(
                    lambda box: sess.run(gpolar, 
                        feed_dict={**base_dict, 
                                t_coord:strerch_box(coord0, box0, box), 
                                t_box:box}).reshape(-1),
                    test_data['box'][:numb_test, :]).reshape([-1,3,3]).transpose(0,2,1)
                @ box0.reshape(3,3)).reshape(-1)

        delta = 1e-4
        for ii in range(pf.size) :
            self.assertAlmostEqual(pf[ii], fdf[ii], delta = delta)
        for ii in range(pv.size) :
            self.assertAlmostEqual(pv[ii], fdv[ii], delta = delta)
        # make sure atomic virial sum to virial        
        places = 10
        for ii in range(pv.size) :
            self.assertAlmostEqual(pv[ii], spv[ii], places = places)
