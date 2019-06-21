#!/usr/bin/env python3

import os,sys
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops
module_path = os.path.dirname(os.path.realpath(__file__))
assert (os.path.isfile (os.path.join(module_path, "libop_abi.so"))), "op module does not exist"
op_module = tf.load_op_library(os.path.join(module_path, "libop_abi.so"))

def _load_graph(frozen_graph_filename, 
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


def _rep_int (s):
    try: 
        int(s)
        return True
    except ValueError:
        return False


def _make_default_mesh(test_box) :
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


class DeepPot () :
    def __init__(self, 
                 model_file) :
        self.model_file = model_file
        self.graph = _load_graph (self.model_file)
        # checkout input/output tensors from graph
        self.t_ntypes = self.graph.get_tensor_by_name ('load/model_attr/ntypes:0')
        self.t_rcut   = self.graph.get_tensor_by_name ('load/model_attr/rcut:0')
        self.t_dfparam= self.graph.get_tensor_by_name ('load/model_attr/dfparam:0')
        self.t_tmap   = self.graph.get_tensor_by_name ('load/model_attr/tmap:0')
        # inputs
        self.t_coord  = self.graph.get_tensor_by_name ('load/i_coord:0')
        self.t_type   = self.graph.get_tensor_by_name ('load/i_type:0')
        self.t_natoms = self.graph.get_tensor_by_name ('load/i_natoms:0')
        self.t_box    = self.graph.get_tensor_by_name ('load/i_box:0')
        self.t_mesh   = self.graph.get_tensor_by_name ('load/i_mesh:0')
        # outputs
        self.t_energy = self.graph.get_tensor_by_name ('load/o_energy:0')
        self.t_force  = self.graph.get_tensor_by_name ('load/o_force:0')
        self.t_virial = self.graph.get_tensor_by_name ('load/o_virial:0')
        self.t_ae     = self.graph.get_tensor_by_name ('load/o_atom_energy:0')
        self.t_av     = self.graph.get_tensor_by_name ('load/o_atom_virial:0')
        self.t_fparam = None
        # check if the graph has fparam
        for op in self.graph.get_operations():
            if op.name == 'load/i_fparam' :
                self.t_fparam = self.graph.get_tensor_by_name ('load/i_fparam:0')
        self.has_fparam = self.t_fparam is not None
        # start a tf session associated to the graph
        self.sess = tf.Session (graph = self.graph)        
        [self.ntypes, self.rcut, self.dfparam, self.tmap] = self.sess.run([self.t_ntypes, self.t_rcut, self.t_dfparam, self.t_tmap])
        self.tmap = self.tmap.decode('UTF-8').split()


    def get_ntypes(self) :
        return self.ntypes

    def get_rcut(self) :
        return self.rcut

    def get_dim_fparam(self) :
        return self.dfparam

    def get_type_map(self):
        return self.tmap


    def eval(self,
             coords, 
             cells, 
             atom_types, 
             fparam = None, 
             atomic = False) :
        # standarize the shape of inputs
        coords = np.array(coords)
        cells = np.array(cells)
        atom_types = np.array(atom_types, dtype = int)
        if self.has_fparam :
            assert(fparam is not None)
            fparam = np.array(fparam)

        # reshape the inputs 
        cells = np.reshape(cells, [-1, 9])
        nframes = cells.shape[0]
        coords = np.reshape(coords, [nframes, -1])
        natoms = coords.shape[1] // 3
        if self.has_fparam :
            fdim = self.get_dim_fparam()
            if fparam.size == nframes * fdim :
                fparam = np.reshape(fparam, [nframes, fdim])
            elif fparam.size == fdim :
                fparam = np.tile(fparam.reshape([-1]), [nframes, 1])
            else :
                raise RuntimeError('got wrong size of frame param, should be either %d x %d or %d' % (nframes, fdim, fdim))

        # sort inputs
        coords, atom_types, imap = self._sort_input(coords, atom_types)

        # make natoms_vec and default_mesh
        natoms_vec = self._make_natoms_vec(atom_types)
        assert(natoms_vec[0] == natoms)
        default_mesh = _make_default_mesh(cells)

        # evaluate
        energy = []
        force = []
        virial = []
        ae = []
        av = []
        feed_dict_test = {}
        feed_dict_test[self.t_natoms] = natoms_vec
        feed_dict_test[self.t_mesh  ] = default_mesh
        feed_dict_test[self.t_type  ] = atom_types
        t_out = [self.t_energy, 
                 self.t_force, 
                 self.t_virial]
        if atomic :
            t_out += [self.t_ae, 
                      self.t_av]
        for ii in range(nframes) :
            feed_dict_test[self.t_coord] = np.reshape(coords[ii:ii+1, :], [-1])
            feed_dict_test[self.t_box  ] = cells[ii:ii+1, :]
            if self.has_fparam:
                feed_dict_test[self.t_fparam] = np.reshape(fparam[ii:ii+1, :], [-1])
            v_out = self.sess.run (t_out, feed_dict = feed_dict_test)
            energy.append(v_out[0])
            force .append(v_out[1])
            virial.append(v_out[2])
            if atomic:
                ae.append(v_out[3])
                av.append(v_out[4])

        # reverse map of the outputs
        force  = self._reverse_map(np.reshape(force, [nframes,-1,3]), imap)
        if atomic :
            ae  = self._reverse_map(np.reshape(ae, [nframes,-1,1]), imap)
            av  = self._reverse_map(np.reshape(av, [nframes,-1,9]), imap)

        energy = np.reshape(energy, [nframes, 1])
        force = np.reshape(force, [nframes, natoms, 3])
        virial = np.reshape(virial, [nframes, 9])
        if atomic:
            ae = np.reshape(ae, [nframes, natoms, 1])
            av = np.reshape(av, [nframes, natoms, 9])
            return energy, force, virial, ae, av
        else :
            return energy, force, virial


    def _sort_input(self, coord, atom_type) :
        natoms = atom_type.size
        idx = np.arange (natoms)
        idx_map = np.lexsort ((idx, atom_type))
        nframes = coord.shape[0]
        coord = coord.reshape([nframes, -1, 3])
        coord = np.reshape(coord[:,idx_map,:], [nframes, -1])
        atom_type = atom_type[idx_map]
        return coord, atom_type, idx_map


    def _reverse_map(self, vec, imap):
        ret = np.zeros(vec.shape)
        for idx,ii in enumerate(imap) :
            ret[:,ii,:] = vec[:,idx,:]
        return ret

        
    def _make_natoms_vec(self, atom_types) :
        natoms_vec = np.zeros (self.ntypes+2).astype(int)
        natoms = atom_types.size
        natoms_vec[0] = natoms
        natoms_vec[1] = natoms
        for ii in range (self.ntypes) :
            natoms_vec[ii+2] = np.count_nonzero(atom_types == ii)
        return natoms_vec

