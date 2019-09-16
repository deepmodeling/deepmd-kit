#!/usr/bin/env python3

import os,sys
import numpy as np
import tensorflow as tf
from deepmd.DeepEval import DeepEval

class DeepPolar (DeepEval) :
    def __init__(self, 
                 model_file) :
        DeepEval.__init__(self, model_file)
        self.model_file = model_file
        self.graph = self.load_graph (self.model_file)
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
        self.t_polar  = self.graph.get_tensor_by_name ('load/o_polar:0')
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
        polar = []
        feed_dict_test = {}
        feed_dict_test[self.t_natoms] = natoms_vec
        feed_dict_test[self.t_mesh  ] = default_mesh
        feed_dict_test[self.t_type  ] = atom_types
        t_out = [self.t_polar]
        for ii in range(nframes) :
            feed_dict_test[self.t_coord] = np.reshape(coords[ii:ii+1, :], [-1])
            feed_dict_test[self.t_box  ] = np.reshape(cells [ii:ii+1, :], [-1])
            v_out = self.sess.run (t_out, feed_dict = feed_dict_test)
            polar.append(v_out[0])

        # reverse map of the outputs
        polar = np.array(polar)
        polar = self.reverse_map(np.reshape(polar, [nframes,-1,9]), sel_imap)

        polar = np.reshape(polar, [nframes, len(sel_at), 9])
        return polar

