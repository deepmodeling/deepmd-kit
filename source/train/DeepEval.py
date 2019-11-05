#!/usr/bin/env python3
import platform
import os,sys
import numpy as np

from deepmd.env import tf

from tensorflow.python.framework import ops

if platform.system() == "Windows":
    ext = "dll"
elif platform.system() == "Darwin":
    ext = "dylib"
else:
    ext = "so"
module_path = os.path.dirname(os.path.realpath(__file__))
assert (os.path.isfile (os.path.join(module_path, "libop_abi.{}".format(ext)))), "op module does not exist"
op_module = tf.load_op_library(os.path.join(module_path, "libop_abi.{}".format(ext)))

class DeepEval():
    """
    common methods for DeepPot, DeepWFC, DeepPolar, ...
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


class DeepTensor(DeepEval) :
    """
    Evaluates a tensor model
    """
    def __init__(self, 
                 model_file, 
                 variable_name,                  
                 variable_dof) :
        DeepEval.__init__(self, model_file)
        self.model_file = model_file
        self.graph = self.load_graph (self.model_file)
        self.variable_name = variable_name
        self.variable_dof = variable_dof
        # checkout input/output tensors from graph
        self.t_ntypes = self.graph.get_tensor_by_name ('load/descrpt_attr/ntypes:0')
        self.t_rcut   = self.graph.get_tensor_by_name ('load/descrpt_attr/rcut:0')
        self.t_tmap   = self.graph.get_tensor_by_name ('load/model_attr/tmap:0')
        self.t_sel_type= self.graph.get_tensor_by_name ('load/model_attr/sel_type:0')
        # inputs
        self.t_coord  = self.graph.get_tensor_by_name ('load/t_coord:0')
        self.t_type   = self.graph.get_tensor_by_name ('load/t_type:0')
        self.t_natoms = self.graph.get_tensor_by_name ('load/t_natoms:0')
        self.t_box    = self.graph.get_tensor_by_name ('load/t_box:0')
        self.t_mesh   = self.graph.get_tensor_by_name ('load/t_mesh:0')
        # outputs
        self.t_tensor = self.graph.get_tensor_by_name ('load/o_%s:0' % self.variable_name)
        # start a tf session associated to the graph
        self.sess = tf.Session (graph = self.graph)        
        [self.ntypes, self.rcut, self.tmap, self.tselt] = self.sess.run([self.t_ntypes, self.t_rcut, self.t_tmap, self.t_sel_type])
        self.tmap = self.tmap.decode('UTF-8').split()

    def get_ntypes(self) :
        return self.ntypes

    def get_rcut(self) :
        return self.rcut

    def get_type_map(self):
        return self.tmap

    def get_sel_type(self):
        return self.tselt

    def eval(self,
             coords, 
             cells, 
             atom_types) :
        # standarize the shape of inputs
        coords = np.array(coords)
        cells = np.array(cells)
        atom_types = np.array(atom_types, dtype = int)

        # reshape the inputs 
        cells = np.reshape(cells, [-1, 9])
        nframes = cells.shape[0]
        coords = np.reshape(coords, [nframes, -1])
        natoms = coords.shape[1] // 3

        # sort inputs
        coords, atom_types, imap, sel_at, sel_imap = self.sort_input(coords, atom_types, sel_atoms = self.get_sel_type())

        # make natoms_vec and default_mesh
        natoms_vec = self.make_natoms_vec(atom_types)
        assert(natoms_vec[0] == natoms)
        default_mesh = self.make_default_mesh(cells)

        # evaluate
        tensor = []
        feed_dict_test = {}
        feed_dict_test[self.t_natoms] = natoms_vec
        feed_dict_test[self.t_mesh  ] = default_mesh
        feed_dict_test[self.t_type  ] = atom_types
        t_out = [self.t_tensor]
        for ii in range(nframes) :
            feed_dict_test[self.t_coord] = np.reshape(coords[ii:ii+1, :], [-1])
            feed_dict_test[self.t_box  ] = np.reshape(cells [ii:ii+1, :], [-1])
            v_out = self.sess.run (t_out, feed_dict = feed_dict_test)
            tensor.append(v_out[0])

        # reverse map of the outputs
        tensor = np.array(tensor)
        tensor = self.reverse_map(np.reshape(tensor, [nframes,-1,self.variable_dof]), sel_imap)

        tensor = np.reshape(tensor, [nframes, len(sel_at), self.variable_dof])
        return tensor

    
