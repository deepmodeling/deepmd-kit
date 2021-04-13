import dpdata,os,sys,unittest
import numpy as np
from deepmd.env import tf
from common import Data,gen_data, j_loader

from deepmd.utils.data_system import DataSystem
from deepmd.descriptor import DescrptLocFrame
from deepmd.fit import EnerFitting
from deepmd.model import EnerModel
from deepmd.common import j_must_have

GLOBAL_ENER_FLOAT_PRECISION = tf.float64
GLOBAL_TF_FLOAT_PRECISION = tf.float64
GLOBAL_NP_FLOAT_PRECISION = np.float64


class TestModel(unittest.TestCase):
    def setUp(self) :
        gen_data()

    def test_model(self):
        jfile = 'water.json'
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
        fitting = EnerFitting(descrpt, 
                              neuron = [240, 120, 60, 30, 10], 
                              seed = 1)
        model = EnerModel(
            descrpt, 
            fitting, 
        )

        # model._compute_dstats([test_data['coord']], [test_data['box']], [test_data['type']], [test_data['natoms_vec']], [test_data['default_mesh']])
        input_data = {'coord' : [test_data['coord']], 
                      'box': [test_data['box']], 
                      'type': [test_data['type']],
                      'natoms_vec' : [test_data['natoms_vec']],
                      'default_mesh' : [test_data['default_mesh']]
        }
        model._compute_input_stat(input_data)
        model.fitting.bias_atom_e = data.compute_energy_shift()

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
                           suffix = "loc_frame", 
                           reuse = False)
        energy = model_pred['energy']
        force  = model_pred['force']
        virial = model_pred['virial']
        atom_ener =  model_pred['atom_ener']

        feed_dict_test = {t_prop_c:        test_data['prop_c'],
                          t_energy:        test_data['energy']              [:numb_test],
                          t_force:         np.reshape(test_data['force']    [:numb_test, :], [-1]),
                          t_virial:        np.reshape(test_data['virial']   [:numb_test, :], [-1]),
                          t_atom_ener:     np.reshape(test_data['atom_ener'][:numb_test, :], [-1]),
                          t_coord:         np.reshape(test_data['coord']    [:numb_test, :], [-1]),
                          t_box:           test_data['box']                 [:numb_test, :],
                          t_type:          np.reshape(test_data['type']     [:numb_test, :], [-1]),
                          t_natoms:        test_data['natoms_vec'],
                          t_mesh:          test_data['default_mesh'],
                          is_training:     False}

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        [e, f, v] = sess.run([energy, force, virial], 
                             feed_dict = feed_dict_test)

        e = e.reshape([-1])
        f = f.reshape([-1])
        v = v.reshape([-1])
        refe = [1.165945032784766511e+01]
        reff = [2.356319331246305437e-01,1.772322096063349284e-01,1.455439548950788684e-02,1.968599426000810226e-01,2.648214484898352983e-01,7.595232354012236564e-02,-2.121321856338151401e-01,-2.463886119018566037e-03,-2.075636300914874069e-02,-9.360310077571798101e-03,-1.751965198776750943e-01,-2.046405309983102827e-02,-1.990194093283037535e-01,-1.828347741191920298e-02,-6.916374506995154325e-02,-1.197997068502068031e-02,-2.461097746875573200e-01,1.987744214930105627e-02]
        refv = [-4.998509978510510265e-01,-1.966169437179327711e-02,1.136130543869883977e-02,-1.966169437179334650e-02,-4.575353297894450555e-01,-2.668666556859019493e-03,1.136130543869887100e-02,-2.668666556859039876e-03,2.455466940358383508e-03]
        refe = np.reshape(refe, [-1])
        reff = np.reshape(reff, [-1])
        refv = np.reshape(refv, [-1])

        places = 10
        for ii in range(e.size) :
            self.assertAlmostEqual(e[ii], refe[ii], places = places)
        for ii in range(f.size) :
            self.assertAlmostEqual(f[ii], reff[ii], places = places)
        for ii in range(v.size) :
            self.assertAlmostEqual(v[ii], refv[ii], places = places)

