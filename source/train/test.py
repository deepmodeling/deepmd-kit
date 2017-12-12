#!/usr/bin/env python3

import re
import os
import sys
import argparse
import numpy as np
import tensorflow as tf

lib_path = os.path.dirname(os.path.realpath(__file__)) + "/../lib/"
sys.path.append (lib_path)

from deepmd.Data import DataSets

from tensorflow.python.framework import ops

# load force module
module_path = os.path.dirname(os.path.realpath(__file__)) + "/../lib/"
assert (os.path.isfile (module_path  + "deepmd/libop_abi.so" )), "force module does not exist"
op_module = tf.load_op_library(module_path + "deepmd/libop_abi.so")

# load grad of force module
sys.path.append (module_path )
import deepmd._prod_force_grad
import deepmd._prod_virial_grad

def load_graph(frozen_graph_filename, 
               prefix = 'load'):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the 
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def, 
            input_map=None, 
            return_elements=None, 
            name=prefix, 
            op_dict=None, 
            producer_op_list=None
        )
    return graph

def rep_int (s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def analyze_ntype (graph) :
    names = []
    for op in graph.get_operations():
        f1 = op.name.split('/')[1]
        if ('layer' in f1) and (not 'gradients'in f1) and (not 'final' in f1) :
            f1_fs = f1.split ('_')
            assert len(f1_fs) == 4 and rep_int (f1_fs[-1]), "unexpected field of " + f1_fs
            names.append (int(f1_fs[-1]))
    s_name = sorted(set(names))
    assert len(s_name)-1 == s_name[-1], "the type is not an seq, unexpected"
    return len(s_name)

def l2err (diff) :
    return np.sqrt(np.average (diff*diff))

def test (sess, data, numb_test = None) :
    graph = sess.graph
    ntypes = analyze_ntype (graph)

    natoms_vec = data.get_natoms_vec (ntypes)
    natoms_vec = natoms_vec.astype(np.int32)
    
    test_prop_c, test_energy, test_force, test_virial, test_coord, test_box, test_type = data.get_test ()

    ncell = np.ones (3, dtype=np.int32)
    avg_box = np.average (test_box, axis = 0)
    cell_size = 3
    avg_box = np.reshape (avg_box, [3,3])
    for ii in range (3) :
        ncell[ii] = int ( np.linalg.norm(avg_box[ii]) / cell_size )
        if (ncell[ii] < 2) : ncell[ii] = 2
    default_mesh = np.zeros (6, dtype = np.int32)
    default_mesh[3] = ncell[0]
    default_mesh[4] = ncell[1]
    default_mesh[5] = ncell[2]

    t_coord  = graph.get_tensor_by_name ('load/t_coord:0')
    t_type   = graph.get_tensor_by_name ('load/t_type:0')
    t_natoms = graph.get_tensor_by_name ('load/t_natoms:0')
    t_box    = graph.get_tensor_by_name ('load/t_box:0')
    t_mesh   = graph.get_tensor_by_name ('load/t_mesh:0')

    t_energy = graph.get_tensor_by_name ('load/energy_test:0')
    t_force  = graph.get_tensor_by_name ('load/force_test:0')
    t_virial = graph.get_tensor_by_name ('load/virial_test:0')

    feed_dict_test = {t_coord:         np.reshape(test_coord   [:numb_test, :], [-1]),
                      t_box:           test_box                [:numb_test, :],
                      t_type:          np.reshape(test_type    [:numb_test, :], [-1]),
                      t_natoms:        natoms_vec,
                      t_mesh:          default_mesh}

    energy = sess.run (t_energy, feed_dict = feed_dict_test)
    force  = sess.run (t_force,  feed_dict = feed_dict_test)
    virial = sess.run (t_virial, feed_dict = feed_dict_test)

    l2e = (l2err (energy - test_energy[:numb_test]))
    l2f = (l2err (force  - test_force [:numb_test]))
    l2v = (l2err (virial - test_virial[:numb_test]))

    # print ("# energies: %s" % energy)
    print ("# number of test data : %d " % numb_test)
    print ("E : %e" % l2e)
    print ("F : %e" % l2f)
    print ("V : %e" % l2v)

def _main () :
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="frozen_model.pb", type=str, 
                        help="Frozen model file to import")
    parser.add_argument("-s", "--system", default=".", type=str, 
                        help="The system dir")
    parser.add_argument("-S", "--set-prefix", default="set", type=str, 
                        help="The set prefix")
    parser.add_argument("-n", "--numb-test", default=100, type=int, 
                        help="The number of data for test")
    args = parser.parse_args()

    graph = load_graph(args.model)
    data = DataSets (args.system, args.set_prefix)

    with tf.Session(graph = graph) as sess:        
        test (sess, data, args.numb_test)

    # for op in graph.get_operations():
    #     print (op.name)

if __name__ == '__main__':
    _main()
