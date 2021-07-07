
import dpdata,os,sys,unittest
import numpy as np
from deepmd.env import tf
import pickle
from common import Data,gen_data, j_loader

from deepmd.utils.data_system import DataSystem
from deepmd.descriptor import DescrptSeA
from deepmd.fit import EnerFitting
from deepmd.model import EnerModel
from deepmd.common import j_must_have

GLOBAL_ENER_FLOAT_PRECISION = tf.float64
GLOBAL_TF_FLOAT_PRECISION = tf.float64
GLOBAL_NP_FLOAT_PRECISION = np.float64

class TestModel(tf.test.TestCase):
    def setUp(self) :
        gen_data()

    def test_fitting(self):
        jfile = 'water_se_a_type.json'
        jdata = j_loader(jfile)

        systems = j_must_have(jdata, 'systems')
        set_pfx = j_must_have(jdata, 'set_prefix')
        batch_size = j_must_have(jdata, 'batch_size')
        test_size = j_must_have(jdata, 'numb_test')
        batch_size = 1
        test_size = 1
        stop_batch = j_must_have(jdata, 'stop_batch')
        rcut = j_must_have (jdata['model']['descriptor'], 'rcut')
        sel = j_must_have (jdata['model']['descriptor'], 'sel')
        ntypes=len(sel)
        
        data = DataSystem(systems, set_pfx, batch_size, test_size, rcut, run_opt = None)
        
        test_data = data.get_test ()
        numb_test = 1

        jdata['model']['descriptor']['neuron'] = [5, 5, 5]
        jdata['model']['descriptor']['axis_neuron'] = 2
        
        jdata['model']['descriptor'].pop('type', None)        
        descrpt = DescrptSeA(**jdata['model']['descriptor'], uniform_seed = True)
        jdata['model']['fitting_net']['descrpt'] = descrpt
        fitting = EnerFitting(**jdata['model']['fitting_net'], uniform_seed = True)
        
        # model._compute_dstats([test_data['coord']], [test_data['box']], [test_data['type']], [test_data['natoms_vec']], [test_data['default_mesh']])
        input_data = {'coord' : [test_data['coord']], 
                      'box': [test_data['box']], 
                      'type': [test_data['type']],
                      'natoms_vec' : [test_data['natoms_vec']],
                      'default_mesh' : [test_data['default_mesh']]
        }
        

        t_prop_c           = tf.placeholder(tf.float32, [5],    name='t_prop_c')
        t_energy           = tf.placeholder(GLOBAL_ENER_FLOAT_PRECISION, [None], name='t_energy')
        t_force            = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name='t_force')
        t_virial           = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name='t_virial')
        t_atom_ener        = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name='t_atom_ener')
        t_coord            = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name='i_coord')
        t_type             = tf.placeholder(tf.int32,   [None], name='i_type')
        t_natoms           = tf.placeholder(tf.int32,   [ntypes+2], name='i_natoms')
        t_box              = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None, 9], name='i_box')
        t_mesh             = tf.placeholder(tf.int32,   [None], name='i_mesh')
        is_training        = tf.placeholder(tf.bool)
        t_fparam = None
        
        dout = np.array([0.0005722682145569174,-0.00020202686217742682,-0.00020202686217742682,7.13250554992363e-05,-0.0014770058171250015,0.000521468690207748,-0.001143865186937176,0.0004038453384193948,0.0005617335409639567,-0.00019831394075147532,
                    0.00048086740718842236,-0.0001693584775806112,-0.0001693584775806112,5.966987137476082e-05,-0.0012342029581315136,0.00043492340851472783,-0.0009566016612537016,0.00033706767041080107,0.00047065988464132244,-0.0001657950398095401,
                    0.0003647849239740657,-0.00013744939018250384,-0.00013744939018250384,5.1825826955234744e-05,-0.00096004206555711,0.00036185565262332876,-0.0007267433909643961,0.0002738914365542745,0.00038019365906978136,-0.00014322754331896057,
                    0.0004675256930823109,-0.00017634410399626168,-0.00017634410399626168,6.652672908755666e-05,-0.0012328062885292486,0.00046500213384094614,-0.0009328887521346069,0.0003518668613172834,0.0004877847509912577,-0.00018396318824508986,
                    0.0005154794374703516,-0.00019422534512034776,-0.00019422534512034776,7.318151797939947e-05,-0.0013576642997136488,0.0005115548790018505,-0.0010275333676074971,0.00038716440070070385,0.0005376426714609369,-0.00020257810468163985,
                    0.0004482204892297628,-0.00016887749501640607,-0.00016887749501640607,6.364643102775375e-05,-0.001181345877677835,0.0004452029242063362,-0.0008941636427724908,0.0003369586197174627,0.0004677878512312651,-0.00017625260641095753])
        type_embedding = np.array([1.4916816460764615,0.2720153234707013,-2.4385153754181985,-1.8454294510880027,2.874575701113528,1.1225116575801295,0.4204818970813372,-2.3784087249787587,-1.5053748251050598,2.769329403073084])

        dout= dout.reshape([-1,10])
        type_embedding = type_embedding.reshape([ntypes,-1])
        atom_ener = fitting.build(tf.convert_to_tensor(dout),
                                  t_natoms, 
                                  {'type_embedding':tf.convert_to_tensor(type_embedding)},
                                  reuse = False,
                                  suffix = "se_a_type_fit_")

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
        [pred_atom_ener] = sess.run([atom_ener], 
                             feed_dict = feed_dict_test)

        pred_atom_ener=pred_atom_ener.reshape([-1])

        ref_atom_ener =[10.121733946849165,10.121712105320634,10.143275419743475,10.143299785396826,10.143311150431957,10.143295201182019]

        places = 10
                
        
        for ii in range(pred_atom_ener.size) :
            self.assertAlmostEqual(pred_atom_ener[ii], ref_atom_ener[ii], places = places)
        
        
