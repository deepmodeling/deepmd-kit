import dpdata,os,sys,unittest
import numpy as np
from deepmd.env import tf
from common import Data, gen_data, del_data, j_loader

from deepmd.utils.data_system import DataSystem
from deepmd.descriptor import DescrptSeA
from deepmd.fit import EnerFitting
from deepmd.model import EnerModel
from deepmd.common import j_must_have

GLOBAL_ENER_FLOAT_PRECISION = tf.float64
GLOBAL_TF_FLOAT_PRECISION = tf.float64
GLOBAL_NP_FLOAT_PRECISION = np.float64


class TestModelSpin(tf.test.TestCase):
    def setUp(self) :
        gen_data()

    def tearDown(self):
        del_data()

    def test_model_spin(self):        
        jfile = 'test_model_spin.json'
        jdata = j_loader(jfile)

        # set system information
        systems = j_must_have(jdata['training']['training_data'], 'systems')
        set_pfx = j_must_have(jdata['training'], 'set_prefix')
        batch_size = j_must_have(jdata['training']['training_data'], 'batch_size')
        test_size = j_must_have(jdata['training']['validation_data'], 'numb_btch')
        stop_batch = j_must_have(jdata['training'], 'numb_steps')
        rcut = j_must_have(jdata['model']['descriptor'], 'rcut')
        data = DataSystem(systems, set_pfx, batch_size, test_size, rcut, run_opt = None)        
        test_data = data.get_test()

        # initialize model
        descrpt_param = jdata['model']['descriptor']
        spin_param = jdata['model']['spin']
        fitting_param = jdata['model']['fitting_net']
        descrpt = DescrptSeA(**descrpt_param, **spin_param, uniform_seed=True)
        fitting_param.pop('type', None)
        fitting_param['descrpt'] = descrpt
        fitting = EnerFitting(**fitting_param, uniform_seed=True)
        model = EnerModel(descrpt, fitting, spin=spin_param)
        
        input_data = {'coord' : [test_data['coord']], 
                      'box': [test_data['box']], 
                      'type': [test_data['type']],
                      'natoms_vec' : [test_data['natoms_vec']],
                      'default_mesh' : [test_data['default_mesh']]
        }

        model._compute_input_stat(input_data)
        model.descrpt.bias_atom_e = data.compute_energy_shift()

        t_prop_c           = tf.placeholder(tf.float32,                  [5],        name='t_prop_c')
        t_energy           = tf.placeholder(GLOBAL_ENER_FLOAT_PRECISION, [None],     name='t_energy')
        t_coord            = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION,   [None],     name='i_coord')
        t_type             = tf.placeholder(tf.int32,                    [None],     name='i_type')
        t_natoms           = tf.placeholder(tf.int32,                    [None], name='i_natoms')
        t_box              = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION,   [None, 9],  name='i_box')
        t_mesh             = tf.placeholder(tf.int32,                    [None],     name='i_mesh')
        is_training        = tf.placeholder(tf.bool)
        t_fparam = None

        model_pred = model.build(t_coord, 
                                 t_type, 
                                 t_natoms, 
                                 t_box, 
                                 t_mesh,
                                 t_fparam,
                                 suffix = "model_spin", 
                                 reuse = False)
        energy = model_pred['energy']
        force  = model_pred['force']
        virial = model_pred['virial']

        # feed data and get results
        feed_dict_test = {t_prop_c:        test_data['prop_c'],
                          t_energy:        test_data['energy'],
                          t_coord:         np.reshape(test_data['coord'],  [-1]),
                          t_box:           np.reshape(test_data['box'],  [-1, 9]),
                          t_type:          np.reshape(test_data['type'],     [-1]),
                          t_natoms:        np.array([48, 48, 16, 16, 16]),
                          t_mesh:          test_data['default_mesh'],
                          is_training:     False
        }

        sess = self.test_session().__enter__()
        sess.run(tf.global_variables_initializer())
        [out_ener, out_force, out_virial] = sess.run([energy, force, virial], 
                                                      feed_dict = feed_dict_test)

        out_ener = np.reshape(out_ener, [-1])
        natoms_real = np.sum(test_data['natoms_vec'][2 : 2 + len(spin_param['use_spin'])])
        force_real = np.reshape(out_force[:, :natoms_real * 3], [-1])
        force_mag = np.reshape(out_force[:, natoms_real * 3:], [-1])

        refe = [328.2811048784]
        refr = [0.0008516080533514307, 0.0006341264741627538, -0.0006826707158890475, 0.00036037380852110834, 0.0002413891612357195, -0.0008370306267048554, 0.0011761514569021388, 0.0005475190113725589, -0.0022065933925052832, 0.0008337938582969307, 0.00012083396790948074, -0.0006206008269615398, 0.0008666527097862479, 0.0006705112848659949, -0.0016910510912273337, 0.0011875504578171393, 0.0006380393040718666, -0.0016885031448699885, 0.0006742215128637818, 0.0003938270919001483, -0.0007297211610271456, 0.0025206257583097152, 0.0005923747151772179, -0.0023177199151020074, -0.0008805860272345909, -0.0002811361473567531, 0.000757961435012491, -0.001673219851590321, -0.0007123550573573719, 0.00102142453417251, -0.001435414539960985, -0.00011489325375279615, 0.0016482539298038213, -0.001304574041043087, -0.0010599956002250903, 0.001808574499619204, -0.00046647589823583645, -1.3558676109988194e-05, 0.0019708099168209087, -9.021786205723383e-05, -0.0007709267567045357, 0.001442821333105358, -0.0012651649689524652, -0.0003256496864118691, 0.0011885630330921004, -0.0011189656976622183, -0.0008628007115321853, 0.0011813179522318917, 1.968851345324627e-05, -6.061002896786453e-05, 1.5191720873038236e-05, 7.008451439526318e-05, 8.787867973969613e-05, 1.753907327747081e-05, 1.3081426486775175e-05, -4.906924779743798e-05, -0.00010932133634353335, -3.535930585288616e-05, 4.5439790658194545e-05, 5.8832875223726135e-06, 2.0867590438682957e-05, 4.8369132241785716e-05, -2.6824212900169102e-05, -0.00019193355054008147, 4.087202088657816e-05, -2.9537775974054627e-06, -3.235281936601786e-05, 0.00010184046789129578, 2.1921452321442734e-05, -2.4880061667433532e-05, -2.5794048666680265e-05, 6.168939386804888e-05, -5.908980232683879e-05, -6.693477763739536e-05, -9.693429508797068e-05, 8.586634204106929e-05, 5.444096578436826e-05, 6.076047202532367e-06, 4.481824903699526e-05, -2.6684206158444287e-05, 5.1214076417644446e-06, -9.920553350098885e-05, -5.94992175870584e-06, -0.00011154609674185282, 9.014621546593938e-05, 5.495586424987237e-05, -4.175075832923205e-05, -0.00016157341327951657, -1.2363868671551805e-05, -1.625404531550157e-05, -7.090401009234104e-05, 9.59153322917869e-05, -2.1855894973052353e-05, 9.438691619635563e-05, 2.0388724669350896e-05, 4.818227501091653e-05]
        refm = [0.001472162000757307, 0.0008404385450225105, -0.004139859290911352, 0.002881978062234717, 0.001075270596039984, -0.0037715014015646626, 0.0017191476932703104, 0.001956227224023315, -0.0035669189751811915, 0.0011072712451062083, 0.0021868198853462442, -0.004278693985884618, 0.0008441078333230077, 0.002383996787396433, -0.0038984195097267137, 0.0008044509985172528, 0.001775878628426965, -0.004074018392971612, 0.0007149002059685153, 0.002833982337255616, -0.0036368882840563813, 0.001417926343624617, 0.000912536291335237, -0.004376268534306017, -0.0012623069097200026, -0.0016607332840985205, 0.004644764904497132, -0.0018852429487173155, -0.0005769365975336776, 0.004562145122355945, -0.0009626060028496647, -0.0026553702208655643, 0.0031598735723267686, -0.0008036324198509105, -0.0021449126702813583, 0.00428426550457741, -0.0024163042702998465, -0.0011995831638908756, 0.0029399032922180347, -0.0009563056408783487, -0.0027956711929794844, 0.00374468385328309, -0.0018083894040460118, -0.0008482089684915652, 0.004793766867247354, -0.001898919222722678, -0.0007013661944374079, 0.004456898706672258]
        refe = np.reshape(refe, [-1])
        refr = np.reshape(refr, [-1])
        refm = np.reshape(refm, [-1])

        places = 10
        np.testing.assert_almost_equal(out_ener, refe, places)
        np.testing.assert_almost_equal(force_real, refr, places)
        np.testing.assert_almost_equal(force_mag, refm, places)



if __name__ == '__main__':
    unittest.main()
