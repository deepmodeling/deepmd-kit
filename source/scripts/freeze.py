#!/usr/bin/env python3

import os, argparse
import sys

import tensorflow as tf
from tensorflow.python.framework import graph_util

dir = os.path.dirname(os.path.realpath(__file__))

from tensorflow.python.framework import ops

# load force module
module_path = os.path.dirname(os.path.realpath(__file__)) + "/../lib/"
assert (os.path.isfile (module_path  + "deepmd/libop_abi.so" )), "force module does not exist"
op_module = tf.load_op_library(module_path + "deepmd/libop_abi.so")

# load grad of force module
sys.path.append (module_path )
import deepmd._prod_force_grad
import deepmd._prod_virial_grad
import deepmd._prod_force_norot_grad
import deepmd._prod_virial_norot_grad

def freeze_graph(model_folder, 
                 output, 
                 output_node_names):
    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path
    
    # We precise the file fullname of our freezed graph
    absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_folder + "/" + output

    # Before exporting our graph, we need to precise what is our output node
    # This is how TF decides what part of the Graph he has to keep and what part it can dump
    # NOTE: this variable is plural, because you can have multiple output nodes
    # output_node_names = "energy_test,force_test,virial_test,t_rcut"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True
    
    # We import the meta graph and retrieve a Saver
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    # We start a session and restore the graph weights
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            input_graph_def, # The graph_def is used to retrieve the nodes 
            output_node_names.split(",") # The output node names are used to select the usefull nodes
        ) 

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))


if __name__ == '__main__':

    default_frozen_nodes = "energy_test,force_test,virial_test,atom_energy_test,atom_virial_test,t_rcut,t_ntypes"

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--folder", type=str, default = ".", 
                        help="path to checkpoint folder")
    parser.add_argument("-o", "--output", type=str, default = "frozen_model.pb", 
                        help="name of graph, will output to the checkpoint folder")
    parser.add_argument("-n", "--nodes", type=str, default = default_frozen_nodes,
                        help="the frozen nodes, defaults is " + default_frozen_nodes)
    args = parser.parse_args()

    freeze_graph(args.folder, args.output, args.nodes)
