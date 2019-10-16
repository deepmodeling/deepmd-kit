#!/usr/bin/env python3

import os,sys
import numpy as np
from deepmd.env import tf
from deepmd.DeepEval import DeepEval

class DeepPot (DeepEval) :
    def __init__(self, 
                 model_file) :
        self.model_file = model_file
        self.graph = self.load_graph (self.model_file)
        # checkout input/output tensors from graph
        self.t_ntypes = self.graph.get_tensor_by_name ('load/descrpt_attr/ntypes:0')
        self.t_rcut   = self.graph.get_tensor_by_name ('load/descrpt_attr/rcut:0')
        self.t_dfparam= self.graph.get_tensor_by_name ('load/fitting_attr/dfparam:0')
        self.t_daparam= self.graph.get_tensor_by_name ('load/fitting_attr/daparam:0')
        self.t_tmap   = self.graph.get_tensor_by_name ('load/model_attr/tmap:0')
        # inputs
        self.t_coord  = self.graph.get_tensor_by_name ('load/t_coord:0')
        self.t_type   = self.graph.get_tensor_by_name ('load/t_type:0')
        self.t_natoms = self.graph.get_tensor_by_name ('load/t_natoms:0')
        self.t_box    = self.graph.get_tensor_by_name ('load/t_box:0')
        self.t_mesh   = self.graph.get_tensor_by_name ('load/t_mesh:0')
        # outputs
        self.t_energy = self.graph.get_tensor_by_name ('load/o_energy:0')
        self.t_force  = self.graph.get_tensor_by_name ('load/o_force:0')
        self.t_virial = self.graph.get_tensor_by_name ('load/o_virial:0')
        self.t_ae     = self.graph.get_tensor_by_name ('load/o_atom_energy:0')
        self.t_av     = self.graph.get_tensor_by_name ('load/o_atom_virial:0')
        self.t_fparam = None
        self.t_aparam = None
        # check if the graph has fparam
        for op in self.graph.get_operations():
            if op.name == 'load/t_fparam' :
                self.t_fparam = self.graph.get_tensor_by_name ('load/t_fparam:0')
        self.has_fparam = self.t_fparam is not None
        # check if the graph has aparam
        for op in self.graph.get_operations():
            if op.name == 'load/t_aparam' :
                self.t_aparam = self.graph.get_tensor_by_name ('load/t_aparam:0')
        self.has_aparam = self.t_aparam is not None
        # start a tf session associated to the graph
        self.sess = tf.Session (graph = self.graph)        
        [self.ntypes, self.rcut, self.dfparam, self.daparam, self.tmap] = self.sess.run([self.t_ntypes, self.t_rcut, self.t_dfparam, self.t_daparam, self.t_tmap])
        self.tmap = self.tmap.decode('UTF-8').split()


    def get_ntypes(self) :
        return self.ntypes

    def get_rcut(self) :
        return self.rcut

    def get_dim_fparam(self) :
        return self.dfparam

    def get_dim_aparam(self) :
        return self.daparam

    def get_type_map(self):
        return self.tmap


    def eval(self,
             coords, 
             cells, 
             atom_types, 
             fparam = None, 
             aparam = None, 
             atomic = False) :
        # standarize the shape of inputs
        coords = np.array(coords)
        cells = np.array(cells)
        atom_types = np.array(atom_types, dtype = int)
        if self.has_fparam :
            assert(fparam is not None)
            fparam = np.array(fparam)
        if self.has_aparam :
            assert(aparam is not None)
            aparam = np.array(aparam)

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
        if self.has_aparam :
            fdim = self.get_dim_aparam()
            if aparam.size == nframes * natoms * fdim:
                aparam = np.reshape(aparam, [nframes, natoms * fdim])
            elif aparam.size == natoms * fdim :
                aparam = np.tile(aparam.reshape([-1]), [nframes, 1])
            elif aparam.size == fdim :
                aparam = np.tile(aparam.reshape([-1]), [nframes, natoms])
            else :
                raise RuntimeError('got wrong size of frame param, should be either %d x %d x %d or %d x %d or %d' % (nframes, natoms, fdim, natoms, fdim, fdim))

        # sort inputs
        coords, atom_types, imap = self.sort_input(coords, atom_types)

        # make natoms_vec and default_mesh
        natoms_vec = self.make_natoms_vec(atom_types)
        assert(natoms_vec[0] == natoms)
        default_mesh = self.make_default_mesh(cells)

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
            feed_dict_test[self.t_box  ] = np.reshape(cells [ii:ii+1, :], [-1])
            if self.has_fparam:
                feed_dict_test[self.t_fparam] = np.reshape(fparam[ii:ii+1, :], [-1])
            if self.has_aparam:
                feed_dict_test[self.t_aparam] = np.reshape(aparam[ii:ii+1, :], [-1])
            v_out = self.sess.run (t_out, feed_dict = feed_dict_test)
            energy.append(v_out[0])
            force .append(v_out[1])
            virial.append(v_out[2])
            if atomic:
                ae.append(v_out[3])
                av.append(v_out[4])

        # reverse map of the outputs
        force  = self.reverse_map(np.reshape(force, [nframes,-1,3]), imap)
        if atomic :
            ae  = self.reverse_map(np.reshape(ae, [nframes,-1,1]), imap)
            av  = self.reverse_map(np.reshape(av, [nframes,-1,9]), imap)

        energy = np.reshape(energy, [nframes, 1])
        force = np.reshape(force, [nframes, natoms, 3])
        virial = np.reshape(virial, [nframes, 9])
        if atomic:
            ae = np.reshape(ae, [nframes, natoms, 1])
            av = np.reshape(av, [nframes, natoms, 9])
            return energy, force, virial, ae, av
        else :
            return energy, force, virial

