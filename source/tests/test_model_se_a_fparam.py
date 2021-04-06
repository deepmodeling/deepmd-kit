import dpdata,os,sys,unittest
import numpy as np
from deepmd.env import tf
from common import Data,gen_data, j_loader

from deepmd.utils.data_system import DataSystem
from deepmd.descriptor import DescrptSeA
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
        jfile = 'water_se_a_fparam.json'
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
        descrpt = DescrptSeA(**jdata['model']['descriptor'])
        jdata['model']['fitting_net']['descrpt'] = descrpt
        fitting = EnerFitting(**jdata['model']['fitting_net'])
        # descrpt = DescrptSeA(jdata['model']['descriptor'])
        # fitting = EnerFitting(jdata['model']['fitting_net'], descrpt)
        model = EnerModel(descrpt, fitting)

        # model._compute_dstats([test_data['coord']], [test_data['box']], [test_data['type']], [test_data['natoms_vec']], [test_data['default_mesh']])
        input_data = {'coord' : [test_data['coord']], 
                      'box': [test_data['box']], 
                      'type': [test_data['type']],
                      'natoms_vec' : [test_data['natoms_vec']],
                      'default_mesh' : [test_data['default_mesh']],
                      'fparam': [test_data['fparam']],
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
        t_fparam           = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name='i_fparam')
        is_training        = tf.placeholder(tf.bool)
        input_dict = {}
        input_dict['fparam'] = t_fparam

        model_pred\
            = model.build (t_coord, 
                           t_type, 
                           t_natoms, 
                           t_box, 
                           t_mesh,
                           input_dict,
                           suffix = "se_a_fparam", 
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
                          t_fparam:        np.reshape(test_data['fparam']   [:numb_test, :], [-1]),
                          is_training:     False}

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        [e, f, v] = sess.run([energy, force, virial], 
                             feed_dict = feed_dict_test)

        e = e.reshape([-1])
        f = f.reshape([-1])
        v = v.reshape([-1])
        refe = [61.35473702079649]
        reff = [7.789591210641927388e-02,9.411176646369459609e-02,3.785806413688173194e-03,1.430830954178063386e-01,1.146964190520970150e-01,-1.320340288927138173e-02,-7.308720494747594776e-02,6.508269338140809657e-02,5.398739145542804643e-04,5.863268336973800898e-02,-1.603409523950408699e-01,-5.083084610994957619e-03,-2.551569799443983988e-01,3.087934885732580501e-02,1.508590526622844222e-02,4.863249399791078065e-02,-1.444292753594846324e-01,-1.125098094204559241e-03]
        refv = [-6.069498397488943819e-01,1.101778888191114192e-01,1.981907430646132409e-02,1.101778888191114608e-01,-3.315612988100872793e-01,-5.999739184898976799e-03,1.981907430646132756e-02,-5.999739184898974197e-03,-1.198656608172396325e-03]
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
