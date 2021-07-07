import dpdata,os,sys,unittest
import numpy as np
from deepmd.env import tf
from common import Data,gen_data, j_loader

from deepmd.utils.data_system import DataSystem
from deepmd.descriptor import DescrptSeT
from deepmd.fit import EnerFitting
from deepmd.model import EnerModel
from deepmd.common import j_must_have

GLOBAL_ENER_FLOAT_PRECISION = tf.float64
GLOBAL_TF_FLOAT_PRECISION = tf.float64
GLOBAL_NP_FLOAT_PRECISION = np.float64

class TestModel(tf.test.TestCase):
    def setUp(self) :
        gen_data()

    def test_model(self):
        jfile = 'water_se_t.json'
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
        descrpt = DescrptSeT(**jdata['model']['descriptor'], uniform_seed = True)
        jdata['model']['fitting_net']['descrpt'] = descrpt
        fitting = EnerFitting(**jdata['model']['fitting_net'], uniform_seed = True)
        model = EnerModel(descrpt, fitting)

        input_data = {'coord' : [test_data['coord']], 
                      'box': [test_data['box']], 
                      'type': [test_data['type']],
                      'natoms_vec' : [test_data['natoms_vec']],
                      'default_mesh' : [test_data['default_mesh']]
        }
        model._compute_input_stat(input_data)
        model.descrpt.bias_atom_e = data.compute_energy_shift()

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

        model_pred\
            = model.build (t_coord, 
                           t_type, 
                           t_natoms, 
                           t_box, 
                           t_mesh,
                           t_fparam,
                           suffix = "se_t", 
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

        sess = self.test_session().__enter__()
        sess.run(tf.global_variables_initializer())
        [e, f, v] = sess.run([energy, force, virial], 
                             feed_dict = feed_dict_test)

        e = e.reshape([-1])
        f = f.reshape([-1])
        v = v.reshape([-1])
        np.savetxt('e.out', e.reshape([1, -1]))
        np.savetxt('f.out', f.reshape([1, -1]), delimiter = ',')
        np.savetxt('v.out', v.reshape([1, -1]), delimiter = ',')
        refe = [4.826771866004193612e+01]
        reff = [5.355088169393570574e+00,5.606772412401632266e+00,2.703270748296462966e-01,5.381408138049708967e+00,5.261355614357515975e+00,-4.079549918988090162e-01,-5.182324474551911919e+00,3.695481388907447262e-01,-5.238474288082559799e-02,1.665564584447352670e-01,-5.955401876564963892e+00,-2.217626865156164251e-01,-5.967343479332643419e+00,9.073821102416884665e-02,3.703103995504785639e-01,2.466151879965444438e-01,-5.373012500109097367e+00,4.146494691512622732e-02]
        refv = [-1.336768232407933077e+01,4.818050125305787801e-01,3.589284283410607568e-01,4.818050125305786691e-01,-1.225345559839458964e+01,-1.701405121682751653e-01,3.589284283410607568e-01,-1.701405121682752486e-01,-3.428455515842296353e-02]
        refe = np.reshape(refe, [-1])
        reff = np.reshape(reff, [-1])
        refv = np.reshape(refv, [-1])

        places = 6
        for ii in range(e.size) :
            self.assertAlmostEqual(e[ii], refe[ii], places = places)
        for ii in range(f.size) :
            self.assertAlmostEqual(f[ii], reff[ii], places = places)
        for ii in range(v.size) :
            self.assertAlmostEqual(v[ii], refv[ii], places = places)
