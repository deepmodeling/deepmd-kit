import dpdata,os,sys,json
import numpy as np
import tensorflow as tf
from common import Data

lib_path = os.path.dirname(os.path.realpath(__file__)) + ".."
sys.path.append (lib_path)

from deepmd.RunOptions import RunOptions
from deepmd.DataSystem import DataSystem
from deepmd.Model import NNPModel
from deepmd.Model import LearingRate
from deepmd.common import j_must_have, j_must_have_d, j_have

def gen_data() :
    tmpdata = Data(rand_pert = 0.1, seed = 1)
    sys = dpdata.LabeledSystem()
    sys.data['coords'] = tmpdata.coord
    sys.data['atom_types'] = tmpdata.atype
    sys.data['cells'] = tmpdata.cell
    nframes = tmpdata.nframes
    natoms = tmpdata.natoms
    print(sys.data['coords'])
    sys.data['coords'] = sys.data['coords'].reshape([nframes,natoms,3])
    sys.data['cells'] = sys.data['cells'].reshape([nframes,3,3])
    sys.data['energies'] = np.zeros([nframes,1])
    sys.data['forces'] = np.zeros([nframes,natoms,3])
    sys.data['virials'] = []
    sys.to_deepmd_npy('system', prec=np.float64)    

def compute_efv(jfile):
    fp = open (jfile, 'r')
    jdata = json.load (fp)
    run_opt = RunOptions(None) 
    systems = j_must_have(jdata, 'systems')
    set_pfx = j_must_have(jdata, 'set_prefix')
    batch_size = j_must_have(jdata, 'batch_size')
    test_size = j_must_have(jdata, 'numb_test')
    batch_size = 1
    test_size = 1
    stop_batch = j_must_have(jdata, 'stop_batch')
    rcut = j_must_have (jdata, 'rcut')

    data = DataSystem(systems, set_pfx, batch_size, test_size, rcut, run_opt)

    tot_numb_batches = sum(data.get_nbatches())
    lr = LearingRate (jdata, tot_numb_batches)

    model = NNPModel (jdata, run_opt = run_opt)
    model.build (data, lr)

    test_prop_c, \
        test_energy, test_force, test_virial, test_atom_ener, \
        test_coord, test_box, test_type, test_fparam, \
        natoms_vec, \
        default_mesh \
        = data.get_test ()
    feed_dict_test = {model.t_prop_c:        test_prop_c,
                      model.t_energy:        test_energy              [:model.numb_test],
                      model.t_force:         np.reshape(test_force    [:model.numb_test, :], [-1]),
                      model.t_virial:        np.reshape(test_virial   [:model.numb_test, :], [-1]),
                      model.t_atom_ener:     np.reshape(test_atom_ener[:model.numb_test, :], [-1]),
                      model.t_coord:         np.reshape(test_coord    [:model.numb_test, :], [-1]),
                      model.t_box:           test_box                 [:model.numb_test, :],
                      model.t_type:          np.reshape(test_type     [:model.numb_test, :], [-1]),
                      model.t_natoms:        natoms_vec,
                      model.t_mesh:          default_mesh,
                      model.is_training:     False}

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    [e, f, v] = sess.run([model.energy, model.force, model.virial], 
                         feed_dict = feed_dict_test)
    return e,f,v

def _main() :
    gen_data()
    e,f,v = compute_efv('water_smth.json')
    np.savetxt('e.out', e, delimiter=',')
    np.savetxt('f.out', f, delimiter=',')
    np.savetxt('v.out', v, delimiter=',')
    

_main()
