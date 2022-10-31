import dpdata, os, sys, unittest
import numpy as np
from deepmd.env import tf
import pickle
from common import Data, gen_data, j_loader

from deepmd.utils.data_system import DeepmdDataSystem
from deepmd.descriptor import DescrptSeAtten
from deepmd.common import data_requirement
from deepmd.fit import EnerFitting
from deepmd.model import EnerModel
from deepmd.utils.type_embed import TypeEmbedNet
from deepmd.common import j_must_have
from common import tf
from packaging.version import parse as parse_version

GLOBAL_ENER_FLOAT_PRECISION = tf.float64
GLOBAL_TF_FLOAT_PRECISION = tf.float64
GLOBAL_NP_FLOAT_PRECISION = np.float64


@unittest.skipIf(parse_version(tf.__version__) < parse_version("1.15"),
    f"The current tf version {tf.__version__} is too low to run the new testing model.")
class TestDataLargeBatch(tf.test.TestCase):
    def setUp(self):
        gen_data(mixed_type=True)

    def test_data_mixed_type(self):
        jfile = 'water_se_atten_mixed_type.json'
        jdata = j_loader(jfile)

        systems = j_must_have(jdata, 'systems')
        batch_size = 1
        test_size = 1
        rcut = j_must_have(jdata['model']['descriptor'], 'rcut')
        type_map = j_must_have(jdata['model'], 'type_map')

        data = DeepmdDataSystem(systems, batch_size, test_size, rcut, type_map=type_map)
        data_requirement = {'energy': {'ndof': 1,
                    'atomic': False,
                    'must': False,
                    'high_prec': True,
                    'type_sel': None,
                    'repeat': 1,
                    'default': 0.0},
         'force': {'ndof': 3,
                   'atomic': True,
                   'must': False,
                   'high_prec': False,
                   'type_sel': None,
                   'repeat': 1,
                   'default': 0.0},
         'virial': {'ndof': 9,
                    'atomic': False,
                    'must': False,
                    'high_prec': False,
                    'type_sel': None,
                    'repeat': 1,
                    'default': 0.0},
         'atom_ener': {'ndof': 1,
                       'atomic': True,
                       'must': False,
                       'high_prec': False,
                       'type_sel': None,
                       'repeat': 1,
                       'default': 0.0},
         'atom_pref': {'ndof': 1,
                       'atomic': True,
                       'must': False,
                       'high_prec': False,
                       'type_sel': None,
                       'repeat': 3,
                       'default': 0.0}}
        data.add_dict(data_requirement)

        test_data = data.get_test()
        numb_test = 1

        jdata['model']['descriptor'].pop('type', None)
        jdata['model']['descriptor']['ntypes'] = 2
        descrpt = DescrptSeAtten(**jdata['model']['descriptor'], uniform_seed=True)
        jdata['model']['fitting_net']['descrpt'] = descrpt
        fitting = EnerFitting(**jdata['model']['fitting_net'], uniform_seed=True)
        typeebd_param = jdata['model']['type_embedding']
        typeebd = TypeEmbedNet(
            neuron=typeebd_param['neuron'],
            resnet_dt=typeebd_param['resnet_dt'],
            activation_function=None,
            seed=typeebd_param['seed'],
            uniform_seed=True,
            padding=True)
        model = EnerModel(descrpt, fitting, typeebd)

        # model._compute_dstats([test_data['coord']], [test_data['box']], [test_data['type']], [test_data['natoms_vec']], [test_data['default_mesh']])
        input_data = {'coord': [test_data['coord']],
                      'box': [test_data['box']],
                      'type': [test_data['type']],
                      'natoms_vec': [test_data['natoms_vec']],
                      'default_mesh': [test_data['default_mesh']],
                      'real_natoms_vec': [test_data['real_natoms_vec']]
                      }
        model._compute_input_stat(input_data, mixed_type=True)
        model.descrpt.bias_atom_e = np.array([0.,  0.])

        t_energy = tf.placeholder(GLOBAL_ENER_FLOAT_PRECISION, [None], name='t_energy')
        t_force = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name='t_force')
        t_virial = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name='t_virial')
        t_atom_ener = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name='t_atom_ener')
        t_coord = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name='i_coord')
        t_type = tf.placeholder(tf.int32, [None], name='i_type')
        t_natoms = tf.placeholder(tf.int32, [model.ntypes + 2], name='i_natoms')
        t_box = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None, 9], name='i_box')
        t_mesh = tf.placeholder(tf.int32, [None], name='i_mesh')
        is_training = tf.placeholder(tf.bool)
        t_fparam = None
        inputs_dict = {}

        model_pred \
            = model.build(t_coord,
                          t_type,
                          t_natoms,
                          t_box,
                          t_mesh,
                          inputs_dict,
                          suffix="se_atten",
                          reuse=False)

        energy = model_pred['energy']
        force = model_pred['force']
        virial = model_pred['virial']
        atom_ener = model_pred['atom_ener']

        feed_dict_test = {t_energy: np.reshape(test_data['energy'][:numb_test], [-1]),
                          t_force: np.reshape(test_data['force'][:numb_test, :], [-1]),
                          t_virial: np.reshape(test_data['virial'][:numb_test, :], [-1]),
                          t_atom_ener: np.reshape(test_data['atom_ener'][:numb_test, :], [-1]),
                          t_coord: np.reshape(test_data['coord'][:numb_test, :], [-1]),
                          t_box: test_data['box'][:numb_test, :],
                          t_type: np.reshape(test_data['type'][:numb_test, :], [-1]),
                          t_natoms: test_data['natoms_vec'],
                          t_mesh: test_data['default_mesh'],
                          is_training: False}


        sess = self.test_session().__enter__()
        sess.run(tf.global_variables_initializer())
        [e, f, v] = sess.run([energy, force, virial],
                             feed_dict=feed_dict_test)
        # print(sess.run(model.type_embedding))
        # np.savetxt('tmp.out', sess.run(descrpt.dout, feed_dict = feed_dict_test), fmt='%.10e')
        # # print(sess.run(model.atype_embed, feed_dict = feed_dict_test))
        # print(sess.run(fitting.inputs, feed_dict = feed_dict_test))
        # print(sess.run(fitting.outs, feed_dict = feed_dict_test))
        # print(sess.run(fitting.atype_embed, feed_dict = feed_dict_test))

        e = e.reshape([-1])
        f = f.reshape([-1])
        v = v.reshape([-1])
        np.savetxt('e.out', e.reshape([1, -1]), delimiter=',')
        np.savetxt('f.out', f.reshape([1, -1]), delimiter=',')
        np.savetxt('v.out', v.reshape([1, -1]), delimiter=',')

        refe = [6.12188445792698e+01]
        reff = [-2.7590100298321299e-03, -2.7392865283639755e-03, 8.5672424478673337e-05,  7.3154109032780492e-03, 7.6754109031673332e-04, -1.0882393042639207e-03, 9.8633073531477645e-03,  3.6631966083397029e-03, -2.2379079261940034e-04, -4.2393697523149913e-03, 4.9491210390296492e-04,  1.6970049039709007e-04, -8.9021867696626039e-03, -4.7967452269658322e-03, 9.2569990351204447e-04, -1.2781517046160920e-03, 2.6103819527704053e-03,  1.3095727849551296e-04]
        refv = [-1.0171833662757776e-02, -6.7981543912862021e-03, 6.1480942994810296e-04, -6.7981543912861942e-03, 3.0092645628232335e-03,  3.8060849919518031e-04, 6.1480942994810383e-04,  3.8060849919518036e-04, -5.6890657188056002e-05]

        refe = np.reshape(refe, [-1])
        reff = np.reshape(reff, [-1])
        refv = np.reshape(refv, [-1])

        places = 10
        np.testing.assert_almost_equal(e, refe, places)
        np.testing.assert_almost_equal(f, reff, places)
        np.testing.assert_almost_equal(v, refv, places)
