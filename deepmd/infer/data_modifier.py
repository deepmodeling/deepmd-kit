import os
import numpy as np
from typing import Tuple, List

from deepmd.infer.deep_dipole import DeepDipole
from deepmd.infer.ewald_recp import EwaldRecp
from deepmd.env import tf
from deepmd.common import select_idx_map, make_default_mesh
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION
from deepmd.env import GLOBAL_NP_FLOAT_PRECISION
from deepmd.env import GLOBAL_ENER_FLOAT_PRECISION
from deepmd.env import global_cvt_2_tf_float
from deepmd.env import global_cvt_2_ener_float
from deepmd.env import op_module


class DipoleChargeModifier(DeepDipole):
    def __init__(self, 
                 model_name : str, 
                 model_charge_map : List[float],
                 sys_charge_map : List[float], 
                 ewald_h : float = 1, 
                 ewald_beta : float = 1
    ) -> None:
        """
        Constructor 

        Parameters
        ----------
        model_name
                The model file for the DeepDipole model
        model_charge_map
                Gives the amount of charge for the wfcc
        sys_charge_map
                Gives the amount of charge for the real atoms
        ewald_h
                Grid spacing of the reciprocal part of Ewald sum. Unit: A
        ewald_beta
                Splitting parameter of the Ewald sum. Unit: A^{-1}
        """
        # the dipole model is loaded with prefix 'dipole_charge'
        self.modifier_prefix = 'dipole_charge'
        # init dipole model
        DeepDipole.__init__(self, 
                            model_name, 
                            load_prefix = self.modifier_prefix, 
                            default_tf_graph = True)
        self.model_name = model_name
        self.model_charge_map = model_charge_map
        self.sys_charge_map = sys_charge_map
        self.sel_type = list(self.get_sel_type())
        # init ewald recp
        self.ewald_h = ewald_h
        self.ewald_beta = ewald_beta
        self.er = EwaldRecp(self.ewald_h, self.ewald_beta)
        # dimension of dipole
        self.ext_dim = 3
        self.t_ndesc  = self.graph.get_tensor_by_name(os.path.join(self.modifier_prefix, 'descrpt_attr/ndescrpt:0'))
        self.t_sela  = self.graph.get_tensor_by_name(os.path.join(self.modifier_prefix, 'descrpt_attr/sel:0'))
        [self.ndescrpt, self.sel_a] = self.sess.run([self.t_ndesc, self.t_sela])
        self.sel_r = [ 0 for ii in range(len(self.sel_a)) ]
        self.nnei_a = np.cumsum(self.sel_a)[-1]
        self.nnei_r = np.cumsum(self.sel_r)[-1]
        self.nnei = self.nnei_a + self.nnei_r
        self.ndescrpt_a = self.nnei_a * 4
        self.ndescrpt_r = self.nnei_r * 1
        assert(self.ndescrpt == self.ndescrpt_a + self.ndescrpt_r)
        self.force = None
        self.ntypes = len(self.sel_a)


    def build_fv_graph(self) -> tf.Tensor:
        """
        Build the computational graph for the force and virial inference.
        """
        with tf.variable_scope('modifier_attr') :
            t_mdl_name = tf.constant(self.model_name, 
                                     name = 'mdl_name', 
                                     dtype = tf.string)
            t_modi_type = tf.constant(self.modifier_prefix, 
                                      name = 'type', 
                                      dtype = tf.string)
            t_mdl_charge_map = tf.constant(' '.join([str(ii) for ii in self.model_charge_map]),
                                            name = 'mdl_charge_map', 
                                            dtype = tf.string)
            t_sys_charge_map = tf.constant(' '.join([str(ii) for ii in self.sys_charge_map]),
                                            name = 'sys_charge_map', 
                                            dtype = tf.string)
            t_ewald_h = tf.constant(self.ewald_h,
                                    name = 'ewald_h', 
                                    dtype = tf.float64)
            t_ewald_b = tf.constant(self.ewald_beta,
                                    name = 'ewald_beta',
                                    dtype = tf.float64)
        with self.graph.as_default():
            return self._build_fv_graph_inner()        


    def _build_fv_graph_inner(self):
        self.t_ef = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name = 't_ef')
        nf = 10
        nfxnas = 64*nf
        nfxna = 192*nf
        nf = -1
        nfxnas = -1
        nfxna = -1
        self.t_box_reshape = tf.reshape(self.t_box, [-1, 9])
        t_nframes = tf.shape(self.t_box_reshape)[0]
        # (nframes x natoms_sel) x 1 x 3
        self.t_ef_reshape = tf.reshape(self.t_ef, [nfxnas, 1, 3])
        # (nframes x natoms) x ndescrpt
        self.descrpt = self.graph.get_tensor_by_name(os.path.join(self.modifier_prefix, 'o_rmat:0'))
        self.descrpt_deriv = self.graph.get_tensor_by_name(os.path.join(self.modifier_prefix, 'o_rmat_deriv:0'))
        self.nlist = self.graph.get_tensor_by_name(os.path.join(self.modifier_prefix, 'o_nlist:0'))
        self.rij = self.graph.get_tensor_by_name(os.path.join(self.modifier_prefix, 'o_rij:0'))
        # self.descrpt_reshape = tf.reshape(self.descrpt, [nf, 192 * self.ndescrpt])
        # self.descrpt_deriv = tf.reshape(self.descrpt_deriv, [nf, 192 * self.ndescrpt * 3])

        # nframes x (natoms_sel x 3)
        self.t_tensor_reshpe = tf.reshape(self.t_tensor, [t_nframes, -1])
        # nframes x (natoms x 3)
        self.t_tensor_reshpe = self._enrich(self.t_tensor_reshpe, dof = 3)
        # (nframes x natoms) x 3
        self.t_tensor_reshpe = tf.reshape(self.t_tensor_reshpe, [nfxna, 3])
        # (nframes x natoms) x 1
        self.t_dipole_x = tf.slice(self.t_tensor_reshpe, [0, 0], [nfxna, 1])
        self.t_dipole_y = tf.slice(self.t_tensor_reshpe, [0, 1], [nfxna, 1])
        self.t_dipole_z = tf.slice(self.t_tensor_reshpe, [0, 2], [nfxna, 1])
        self.t_dipole_z = tf.reshape(self.t_dipole_z, [nfxna, 1])
        # (nframes x natoms) x ndescrpt
        [self.t_dipole_x_d] = tf.gradients(self.t_dipole_x, self.descrpt)
        [self.t_dipole_y_d] = tf.gradients(self.t_dipole_y, self.descrpt)
        [self.t_dipole_z_d] = tf.gradients(self.t_dipole_z, self.descrpt)
        # nframes x (natoms x ndescrpt)
        self.t_dipole_x_d = tf.reshape(self.t_dipole_x_d, [-1, self.t_natoms[0] * self.ndescrpt])
        self.t_dipole_y_d = tf.reshape(self.t_dipole_y_d, [-1, self.t_natoms[0] * self.ndescrpt])
        self.t_dipole_z_d = tf.reshape(self.t_dipole_z_d, [-1, self.t_natoms[0] * self.ndescrpt])
        # nframes x (natoms_sel x ndescrpt)
        self.t_dipole_x_d = self._slice_descrpt_deriv(self.t_dipole_x_d)
        self.t_dipole_y_d = self._slice_descrpt_deriv(self.t_dipole_y_d)
        self.t_dipole_z_d = self._slice_descrpt_deriv(self.t_dipole_z_d)
        # (nframes x natoms_sel) x ndescrpt
        self.t_dipole_x_d = tf.reshape(self.t_dipole_x_d, [nfxnas, self.ndescrpt])
        self.t_dipole_y_d = tf.reshape(self.t_dipole_y_d, [nfxnas, self.ndescrpt])
        self.t_dipole_z_d = tf.reshape(self.t_dipole_z_d, [nfxnas, self.ndescrpt])
        # (nframes x natoms_sel) x 3 x ndescrpt
        self.t_dipole_d = tf.concat([self.t_dipole_x_d, self.t_dipole_y_d, self.t_dipole_z_d], axis = 1)
        self.t_dipole_d = tf.reshape(self.t_dipole_d, [nfxnas, 3*self.ndescrpt])
        # (nframes x natoms_sel) x 3 x ndescrpt
        self.t_dipole_d = tf.reshape(self.t_dipole_d, [-1, 3, self.ndescrpt])
        # (nframes x natoms_sel) x 1 x ndescrpt
        self.t_ef_d = tf.matmul(self.t_ef_reshape, self.t_dipole_d)
        # nframes x (natoms_sel x ndescrpt)
        self.t_ef_d = tf.reshape(self.t_ef_d, [t_nframes, -1])
        # nframes x (natoms x ndescrpt)
        self.t_ef_d = self._enrich(self.t_ef_d, dof = self.ndescrpt)
        self.t_ef_d = tf.reshape(self.t_ef_d, [nf, self.t_natoms[0] * self.ndescrpt])
        # t_ef_d is force (with -1), prod_forc takes deriv, so we need the opposite
        self.t_ef_d_oppo = -self.t_ef_d
        
        force = op_module.prod_force_se_a(self.t_ef_d_oppo,
                                          self.descrpt_deriv, 
                                          self.nlist, 
                                          self.t_natoms,
                                          n_a_sel = self.nnei_a,
                                          n_r_sel = self.nnei_r)
        virial, atom_virial \
            = op_module.prod_virial_se_a (self.t_ef_d_oppo,
                                          self.descrpt_deriv,
                                          self.rij,
                                          self.nlist,
                                          self.t_natoms,
                                          n_a_sel = self.nnei_a,
                                          n_r_sel = self.nnei_r)
        force = tf.identity(force, name='o_dm_force')
        virial = tf.identity(virial, name='o_dm_virial')
        atom_virial = tf.identity(atom_virial, name='o_dm_av')
        return force, virial, atom_virial


    def _enrich(self, dipole, dof = 3):
        coll = []                
        sel_start_idx = 0
        for type_i in range(self.ntypes):
            if type_i in self.sel_type:
                di = tf.slice(dipole, 
                              [ 0, sel_start_idx           * dof],
                              [-1, self.t_natoms[2+type_i] * dof])
                sel_start_idx += self.t_natoms[2+type_i]
            else:
                di = tf.zeros([tf.shape(dipole)[0], self.t_natoms[2+type_i] * dof],
                              dtype = GLOBAL_TF_FLOAT_PRECISION)
            coll.append(di)
        return tf.concat(coll, axis = 1)

    def _slice_descrpt_deriv(self, deriv):
        coll = []
        start_idx = 0
        for type_i in range(self.ntypes):
            if type_i in self.sel_type:
                di = tf.slice(deriv, 
                              [ 0, start_idx               * self.ndescrpt],
                              [-1, self.t_natoms[2+type_i] * self.ndescrpt])
                coll.append(di)
            start_idx += self.t_natoms[2+type_i]
        return tf.concat(coll, axis = 1)        


    def eval(self, 
             coord : np.array, 
             box : np.array, 
             atype : np.array, 
             eval_fv : bool = True
    ) -> Tuple[np.array, np.array, np.array]:
        """
        Evaluate the modification
        
        Parameters
        ----------
        coord
                The coordinates of atoms
        box
                The simulation region. PBC is assumed
        atype
                The atom types
        eval_fv
                Evaluate force and virial

        Returns
        -------
        tot_e
                The energy modification
        tot_f
                The force modification
        tot_v
                The virial modification
        """
        atype = np.array(atype, dtype=int)
        coord, atype, imap = self.sort_input(coord, atype)
        # natoms = coord.shape[1] // 3
        natoms = atype.size
        nframes = coord.shape[0]
        box = np.reshape(box, [nframes, 9])
        atype = np.reshape(atype, [natoms])
        sel_idx_map = select_idx_map(atype, self.sel_type)
        nsel = len(sel_idx_map)
        # setup charge
        charge = np.zeros([natoms])
        for ii in range(natoms):
            charge[ii] = self.sys_charge_map[atype[ii]]
        charge = np.tile(charge, [nframes, 1])

        # add wfcc
        all_coord, all_charge, dipole = self._extend_system(coord, box, atype, charge)
        
        # print('compute er')
        batch_size = 5
        tot_e = []
        all_f = []
        all_v = []
        for ii in range(0,nframes,batch_size):
            e,f,v = self.er.eval(all_coord[ii:ii+batch_size], all_charge[ii:ii+batch_size], box[ii:ii+batch_size])
            tot_e.append(e)
            all_f.append(f)
            all_v.append(v)
        tot_e = np.concatenate(tot_e, axis = 0)
        all_f = np.concatenate(all_f, axis = 0)
        all_v = np.concatenate(all_v, axis = 0)
        # print('finish  er')
        # reshape
        tot_e.reshape([nframes,1])

        tot_f = None
        tot_v = None
        if self.force is None:
            self.force, self.virial, self.av = self.build_fv_graph()
        if eval_fv:
            # compute f
            ext_f = all_f[:,natoms*3:]
            corr_f = []
            corr_v = []
            corr_av = []
            for ii in range(0,nframes,batch_size):
                f, v, av = self._eval_fv(coord[ii:ii+batch_size], box[ii:ii+batch_size], atype, ext_f[ii:ii+batch_size])
                corr_f.append(f)
                corr_v.append(v)
                corr_av.append(av)
            corr_f = np.concatenate(corr_f, axis = 0)
            corr_v = np.concatenate(corr_v, axis = 0)
            corr_av = np.concatenate(corr_av, axis = 0)
            tot_f = all_f[:,:natoms*3] + corr_f
            for ii in range(nsel):            
                orig_idx = sel_idx_map[ii]            
                tot_f[:,orig_idx*3:orig_idx*3+3] += ext_f[:,ii*3:ii*3+3]                
            tot_f = self.reverse_map(np.reshape(tot_f, [nframes,-1,3]), imap)
            # reshape
            tot_f = tot_f.reshape([nframes,natoms,3])
            # compute v
            dipole3 = np.reshape(dipole, [nframes, nsel, 3])
            ext_f3 = np.reshape(ext_f, [nframes, nsel, 3])
            ext_f3 = np.transpose(ext_f3, [0, 2, 1])
            # fd_corr_v = -np.matmul(ext_f3, dipole3).T.reshape([nframes, 9])
            # fd_corr_v = -np.matmul(ext_f3, dipole3)
            # fd_corr_v = np.transpose(fd_corr_v, [0, 2, 1]).reshape([nframes, 9])
            fd_corr_v = -np.matmul(ext_f3, dipole3).reshape([nframes, 9])
            # print(all_v, '\n', corr_v, '\n', fd_corr_v)
            tot_v = all_v + corr_v + fd_corr_v
            # reshape
            tot_v = tot_v.reshape([nframes,9])

        return tot_e, tot_f, tot_v


    def _eval_fv(self, coords, cells, atom_types, ext_f) :
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
        default_mesh = make_default_mesh(cells)

        # evaluate
        tensor = []
        feed_dict_test = {}
        feed_dict_test[self.t_natoms] = natoms_vec
        feed_dict_test[self.t_type  ] = np.tile(atom_types, [nframes, 1]).reshape([-1])
        feed_dict_test[self.t_coord ] = coords.reshape([-1])
        feed_dict_test[self.t_box   ] = cells.reshape([-1])
        feed_dict_test[self.t_mesh  ] = default_mesh.reshape([-1])
        feed_dict_test[self.t_ef    ] = ext_f.reshape([-1])
        # print(self.sess.run(tf.shape(self.t_tensor), feed_dict = feed_dict_test))
        fout, vout, avout \
            = self.sess.run([self.force, self.virial, self.av],
                            feed_dict = feed_dict_test)
        # print('fout: ', fout.shape, fout)
        fout = self.reverse_map(np.reshape(fout, [nframes,-1,3]), imap)
        fout = np.reshape(fout, [nframes, -1])
        return fout, vout, avout


    def _extend_system(self, coord, box, atype, charge):
        natoms = coord.shape[1] // 3
        nframes = coord.shape[0]
        # sel atoms and setup ref coord
        sel_idx_map = select_idx_map(atype, self.sel_type)
        nsel = len(sel_idx_map)
        coord3 = coord.reshape([nframes, natoms, 3])
        ref_coord = coord3[:,sel_idx_map,:]
        ref_coord = np.reshape(ref_coord, [nframes, nsel * 3])
        
        dipole = DeepDipole.eval(self, coord, box, atype)
        dipole = np.reshape(dipole, [nframes, nsel * 3])
        
        wfcc_coord = ref_coord + dipole
        # wfcc_coord = dipole
        wfcc_charge = np.zeros([nsel])
        for ii in range(nsel):
            orig_idx = self.sel_type.index(atype[sel_idx_map[ii]])
            wfcc_charge[ii] = self.model_charge_map[orig_idx]
        wfcc_charge = np.tile(wfcc_charge, [nframes, 1])

        wfcc_coord = np.reshape(wfcc_coord, [nframes, nsel * 3])
        wfcc_charge = np.reshape(wfcc_charge, [nframes, nsel])

        all_coord = np.concatenate((coord, wfcc_coord), axis = 1)
        all_charge = np.concatenate((charge, wfcc_charge), axis = 1)

        return all_coord, all_charge, dipole


    def modify_data(self, 
                    data : dict) -> None:        
        """
        Modify data.

        Parameters
        ----------
        data
                Internal data of DeepmdData.
                Be a dict, has the following keys
                - coord         coordinates
                - box           simulation box
                - type          atom types
                - find_energy   tells if data has energy
                - find_force    tells if data has force
                - find_virial   tells if data has virial
                - energy        energy
                - force         force
                - virial        virial
        """
        if 'find_energy' not in data and 'find_force' not in data and 'find_virial' not in data:
            return

        get_nframes=None
        coord = data['coord'][:get_nframes,:]
        box = data['box'][:get_nframes,:]
        atype = data['type'][:get_nframes,:]
        atype = atype[0]
        nframes = coord.shape[0]

        tot_e, tot_f, tot_v = self.eval(coord, box, atype)

        # print(tot_f[:,0])
        
        if 'find_energy' in data and data['find_energy'] == 1.0 :
            data['energy'] -= tot_e.reshape(data['energy'].shape)
        if 'find_force' in data and data['find_force'] == 1.0 :
            data['force'] -= tot_f.reshape(data['force'].shape)
        if 'find_virial' in data and data['find_virial'] == 1.0 :
            data['virial'] -= tot_v.reshape(data['virial'].shape)


                           
