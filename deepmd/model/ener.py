import numpy as np
from typing import Tuple, List

from deepmd.env import paddle
from deepmd.utils.pair_tab import PairTab
from deepmd.common import ClassArg
from deepmd.env import global_cvt_2_ener_float, MODEL_VERSION, GLOBAL_ENER_FLOAT_PRECISION
from deepmd.env import op_module, paddle_ops
from .model_stat import make_stat_input, merge_sys_stat

import sys

class EnerModel(paddle.nn.Layer) :
    model_type = 'ener'

    def __init__ (
            self, 
            descrpt, 
            fitting, 
            type_map : List[str] = None,
            data_stat_nbatch : int = 10,
            data_stat_protect : float = 1e-2,
            use_srtab : str = None,
            smin_alpha : float = None,
            sw_rmin : float = None,
            sw_rmax : float = None
    ) -> None:
        """
        Constructor

        Parameters
        ----------
        descrpt
                Descriptor
        fitting
                Fitting net
        type_map
                Mapping atom type to the name (str) of the type.
                For example `type_map[1]` gives the name of the type 1.
        data_stat_nbatch
                Number of frames used for data statistic
        data_stat_protect
                Protect parameter for atomic energy regression
        use_srtab
                The table for the short-range pairwise interaction added on top of DP. The table is a text data file with (N_t + 1) * N_t / 2 + 1 columes. The first colume is the distance between atoms. The second to the last columes are energies for pairs of certain types. For example we have two atom types, 0 and 1. The columes from 2nd to 4th are for 0-0, 0-1 and 1-1 correspondingly.
        smin_alpha
                The short-range tabulated interaction will be swithed according to the distance of the nearest neighbor. This distance is calculated by softmin. This parameter is the decaying parameter in the softmin. It is only required when `use_srtab` is provided.
        sw_rmin
                The lower boundary of the interpolation between short-range tabulated interaction and DP. It is only required when `use_srtab` is provided.
        sw_rmin
                The upper boundary of the interpolation between short-range tabulated interaction and DP. It is only required when `use_srtab` is provided.
        """
        # descriptor
        super(EnerModel, self).__init__(name_scope="EnerModel")
        self.descrpt = descrpt
        self.rcut = self.descrpt.get_rcut()
        self.ntypes = self.descrpt.get_ntypes()
        # fitting
        self.fitting = fitting
        self.numb_fparam = self.fitting.get_numb_fparam()
        # other inputs
        if type_map is None:
            self.type_map = []
        else:
            self.type_map = type_map
        self.data_stat_nbatch = data_stat_nbatch
        self.data_stat_protect = data_stat_protect
        self.srtab_name = use_srtab
        if self.srtab_name is not None :
            self.srtab = PairTab(self.srtab_name)
            self.smin_alpha = smin_alpha
            self.sw_rmin = sw_rmin
            self.sw_rmax = sw_rmax
        else :
            self.srtab = None
        
        self.t_tmap = ' '.join(self.type_map)
        self.t_mt = self.model_type
        self.t_ver = MODEL_VERSION


    def get_rcut (self) :
        return self.rcut

    def get_ntypes (self) :
        return self.ntypes

    def get_type_map (self) :
        return self.type_map

    def data_stat(self, data):
        all_stat = make_stat_input(data, self.data_stat_nbatch, merge_sys = False)
        m_all_stat = merge_sys_stat(all_stat)
        self._compute_input_stat(m_all_stat, protection = self.data_stat_protect)
        self._compute_output_stat(all_stat)
        # self.bias_atom_e = data.compute_energy_shift(self.rcond)

    def _compute_input_stat (self, all_stat, protection = 1e-2) :
        self.descrpt.compute_input_stats(all_stat['coord'],
                                         all_stat['box'],
                                         all_stat['type'],
                                         all_stat['natoms_vec'],
                                         all_stat['default_mesh'], 
                                         all_stat)
        self.fitting.compute_input_stats(all_stat, protection = protection)

    def _compute_output_stat (self, all_stat) :
        self.fitting.compute_output_stats(all_stat)

    #@paddle.jit.to_static
    def forward (self, 
               coord_, 
               atype_,
               natoms,
               box, 
               mesh,
               input_dict,
               suffix = '', 
               reuse = None):
        coord = paddle.reshape(coord_, [-1, natoms[1] * 3])
        atype = paddle.reshape(atype_, [-1, natoms[1]])

        dout = self.descrpt(coord_,
                            atype_,
                            natoms,
                            box,
                            mesh,
                            input_dict,
                            suffix = suffix,
                            reuse = reuse)

        self.dout = dout
        
        atom_ener = self.fitting (dout, 
                                  natoms, 
                                  input_dict, 
                                  reuse = reuse, 
                                  suffix = suffix)

        self.atom_ener = atom_ener
        energy_raw = atom_ener

        energy_raw = paddle.reshape(energy_raw, [-1, natoms[0]], name = 'o_atom_energy'+suffix)
        energy = paddle.sum(paddle.cast(energy_raw, GLOBAL_ENER_FLOAT_PRECISION), axis=1, name='o_energy'+suffix)

        force, virial, atom_virial = self.descrpt.prod_force_virial (atom_ener, natoms)

        force = paddle.reshape (force, [-1, 3 * natoms[1]], name = "o_force"+suffix)
        virial = paddle.reshape (virial, [-1, 9], name = "o_virial"+suffix)
        atom_virial = paddle.reshape (atom_virial, [-1, 9 * natoms[1]], name = "o_atom_virial"+suffix)

        model_dict = {}
        model_dict['energy'] = energy
        model_dict['force'] = force
        model_dict['virial'] = virial
        model_dict['atom_ener'] = energy_raw
        model_dict['atom_virial'] = atom_virial
        model_dict['coord'] = coord
        model_dict['atype'] = atype

        return model_dict

