#!/usr/bin/env python3

import os,sys
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops
module_path = os.path.dirname(os.path.realpath(__file__))
assert (os.path.isfile (os.path.join(module_path, "libop_abi.so"))), "op module does not exist"
op_module = tf.load_op_library(os.path.join(module_path, "libop_abi.so"))

class DeepEval():
    """
    common method for DeepPot, DeepWFC, DeepPolar, ...
    """
    def __init__(self, 
                 model_file) :
        model_file = model_file
        graph = self.load_graph (model_file)
        t_mt = graph.get_tensor_by_name('load/model_attr/model_type:0')
        sess = tf.Session (graph = graph)        
        [mt] = sess.run([t_mt], feed_dict = {})
        self.model_type = mt.decode('utf-8')

    def load_graph(self, 
                   frozen_graph_filename, 
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
                producer_op_list=None
            )
        return graph

    def make_default_mesh(self, test_box) :
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
        return default_mesh

    def sort_input(self, coord, atom_type, sel_atoms = None) :
        if sel_atoms is not None:
            selection = [False] * np.size(atom_type)
            for ii in sel_atoms:
                selection += (atom_type == ii)
            sel_atom_type = atom_type[selection]
        natoms = atom_type.size
        idx = np.arange (natoms)
        idx_map = np.lexsort ((idx, atom_type))
        nframes = coord.shape[0]
        coord = coord.reshape([nframes, -1, 3])
        coord = np.reshape(coord[:,idx_map,:], [nframes, -1])
        atom_type = atom_type[idx_map]
        if sel_atoms is not None:
            sel_natoms = np.size(sel_atom_type)
            sel_idx = np.arange(sel_natoms)
            sel_idx_map = np.lexsort((sel_idx, sel_atom_type))
            sel_atom_type = sel_atom_type[sel_idx_map]
            return coord, atom_type, idx_map, sel_atom_type, sel_idx_map
        else:
            return coord, atom_type, idx_map

    def reverse_map(self, vec, imap):
        ret = np.zeros(vec.shape)        
        for idx,ii in enumerate(imap) :
            ret[:,ii,:] = vec[:,idx,:]
        return ret
        
    def make_natoms_vec(self, atom_types) :
        natoms_vec = np.zeros (self.ntypes+2).astype(int)
        natoms = atom_types.size
        natoms_vec[0] = natoms
        natoms_vec[1] = natoms
        for ii in range (self.ntypes) :
            natoms_vec[ii+2] = np.count_nonzero(atom_types == ii)
        return natoms_vec

    
