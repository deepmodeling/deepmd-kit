import math
import numpy as np
from typing import Tuple, List

from deepmd.env import paddle
from paddle import to_tensor
from deepmd.common import get_activation_func, get_precision, ACTIVATION_FN_DICT, PRECISION_DICT, docstring_parameter, get_np_precision
from deepmd.utils.argcheck import list_to_doc
from deepmd.env import GLOBAL_PD_FLOAT_PRECISION, GLOBAL_NP_FLOAT_PRECISION, paddle_ops
from deepmd.utils.network import EmbeddingNet
from deepmd.utils.tabulate import DeepTabulate

from collections import defaultdict
import sys


class DescrptSeA(paddle.nn.Layer):
    @docstring_parameter(list_to_doc(ACTIVATION_FN_DICT.keys()), list_to_doc(PRECISION_DICT.keys()))
    def __init__ (self, 
                  rcut: float,
                  rcut_smth: float,
                  sel: List[str],
                  neuron: List[int] = [24,48,96],
                  axis_neuron: int = 8,
                  resnet_dt: bool = False,
                  trainable: bool = True,
                  seed: int = 1,
                  type_one_side: bool = True,
                  exclude_types: List[int] = [],
                  set_davg_zero: bool = False,
                  activation_function: str = 'tanh',
                  precision: str = 'default'
    ) -> None:
        """
        Constructor

        Parameters
        ----------
        rcut
                The cut-off radius
        rcut_smth
                From where the environment matrix should be smoothed
        sel : list[str]
                sel[i] specifies the maxmum number of type i atoms in the cut-off radius
        neuron : list[int]
                Number of neurons in each hidden layers of the embedding net
        axis_neuron
                Number of the axis neuron (number of columns of the sub-matrix of the embedding matrix)
        resnet_dt
                Time-step `dt` in the resnet construction:
                y = x + dt * \phi (Wx + b)
        trainable
                If the weights of embedding net are trainable.
        seed
                Random seed for initializing the network parameters.
        type_one_side
                Try to build N_types embedding nets. Otherwise, building N_types^2 embedding nets
        exclude_types : list[int]
                The Excluded types
        set_davg_zero
                Set the shift of embedding net input to zero.
        activation_function
                The activation function in the embedding net. Supported options are {0}
        precision
                The precision of the embedding net parameters. Supported options are {1}
        """
        super(DescrptSeA, self).__init__(name_scope="DescrptSeA")
        self.sel_a = sel
        self.rcut_r = rcut
        self.rcut_r_smth = rcut_smth
        self.filter_neuron = neuron
        self.n_axis_neuron = axis_neuron
        self.filter_resnet_dt = resnet_dt
        self.seed = seed
        self.trainable = trainable
        self.filter_activation_fn = get_activation_func(activation_function)
        self.filter_precision = get_precision(precision)
        self.filter_np_precision = get_np_precision(precision)
        self.exclude_types = set()
        for tt in exclude_types:
            assert(len(tt) == 2)
            self.exclude_types.add((tt[0], tt[1]))
            self.exclude_types.add((tt[1], tt[0]))
        self.set_davg_zero = set_davg_zero
        self.type_one_side = type_one_side
        if self.type_one_side and len(exclude_types) != 0:
            raise RuntimeError('"type_one_side" is not compatible with "exclude_types"')

        # descrpt config
        self.sel_r = [ 0 for ii in range(len(self.sel_a)) ]
        self.ntypes = len(self.sel_a)
        assert(self.ntypes == len(self.sel_r))
        self.rcut_a = -1
        # numb of neighbors and numb of descrptors
        self.nnei_a = np.cumsum(self.sel_a)[-1]
        self.nnei_r = np.cumsum(self.sel_r)[-1]
        self.nnei = self.nnei_a + self.nnei_r
        self.ndescrpt_a = self.nnei_a * 4
        self.ndescrpt_r = self.nnei_r * 1
        self.ndescrpt = self.ndescrpt_a + self.ndescrpt_r
        self.useBN = False
        self.dstd = None
        self.davg = None
        self.compress = False
        self.avg_zero = paddle.zeros([self.ntypes, self.ndescrpt], dtype=GLOBAL_PD_FLOAT_PRECISION)
        self.std_ones = paddle.ones ([self.ntypes, self.ndescrpt], dtype=GLOBAL_PD_FLOAT_PRECISION)

        nets = []
        for type_input in range(self.ntypes) :
            layer = []
            for type_i in range(self.ntypes) :
                layer.append(EmbeddingNet(self.filter_neuron, self.filter_precision, self.filter_activation_fn, self.filter_resnet_dt, self.seed, self.trainable, name='filter_type_'+str(type_input)+str(type_i)))
            nets.append(paddle.nn.LayerList(layer))

        self.embedding_nets = paddle.nn.LayerList(nets)

        self.t_rcut = paddle.to_tensor(np.max([self.rcut_r, self.rcut_a]), dtype = GLOBAL_PD_FLOAT_PRECISION)
        self.t_ntypes = paddle.to_tensor(self.ntypes, dtype = "int32")
        self.t_ndescrpt = paddle.to_tensor(self.ndescrpt, dtype = "int32")
        self.t_sel = paddle.to_tensor(self.sel_a, dtype = "int32")
        self.t_avg = paddle.to_tensor(np.zeros([self.ntypes, self.ndescrpt]), dtype = GLOBAL_PD_FLOAT_PRECISION)
        self.t_std = paddle.to_tensor(np.ones([self.ntypes, self.ndescrpt]), dtype = GLOBAL_PD_FLOAT_PRECISION)

    def get_rcut (self) -> float:
        """
        Returns the cut-off radius
        """
        return self.rcut_r

    def get_ntypes (self) -> int:
        """
        Returns the number of atom types
        """
        return self.ntypes

    def get_dim_out (self) -> int:
        """
        Returns the output dimension of this descriptor
        """
        return self.filter_neuron[-1] * self.n_axis_neuron

    def get_dim_rot_mat_1 (self) -> int:
        """
        Returns the first dimension of the rotation matrix. The rotation is of shape dim_1 x 3
        """
        return self.filter_neuron[-1]

    def get_nlist (self) -> Tuple[paddle.Tensor, paddle.Tensor, List[int], List[int]]:
        """
        Returns
        -------
        nlist
                Neighbor list
        rij
                The relative distance between the neighbor and the center atom.
        sel_a
                The number of neighbors with full information
        sel_r
                The number of neighbors with only radial information
        """
        return self.nlist, self.rij, self.sel_a, self.sel_r

    def compute_input_stats (self,
                             data_coord : list, 
                             data_box : list, 
                             data_atype : list, 
                             natoms_vec : list,
                             mesh : list, 
                             input_dict : dict
    ) -> None :
        """
        Compute the statisitcs (avg and std) of the training data. The input will be normalized by the statistics.
        
        Parameters
        ----------
        data_coord
                The coordinates. Can be generated by deepmd.model.make_stat_input
        data_box
                The box. Can be generated by deepmd.model.make_stat_input
        data_atype
                The atom types. Can be generated by deepmd.model.make_stat_input
        natoms_vec
                The vector for the number of atoms of the system and different types of atoms. Can be generated by deepmd.model.make_stat_input
        mesh
                The mesh for neighbor searching. Can be generated by deepmd.model.make_stat_input
        input_dict
                Dictionary for additional input
        """
        all_davg = []
        all_dstd = []
        if True:
            sumr = []
            suma = []
            sumn = []
            sumr2 = []
            suma2 = []
            for cc,bb,tt,nn,mm in zip(data_coord,data_box,data_atype,natoms_vec,mesh) :
                sysr,sysr2,sysa,sysa2,sysn \
                    = self._compute_dstats_sys_smth(cc,bb,tt,nn,mm)
                sumr.append(sysr)
                suma.append(sysa)
                sumn.append(sysn)
                sumr2.append(sysr2)
                suma2.append(sysa2)
            sumr = np.sum(sumr, axis = 0)
            suma = np.sum(suma, axis = 0)
            sumn = np.sum(sumn, axis = 0)
            sumr2 = np.sum(sumr2, axis = 0)
            suma2 = np.sum(suma2, axis = 0)
            for type_i in range(self.ntypes) :
                davgunit = [sumr[type_i]/sumn[type_i], 0, 0, 0]
                dstdunit = [self._compute_std(sumr2[type_i], sumr[type_i], sumn[type_i]), 
                            self._compute_std(suma2[type_i], suma[type_i], sumn[type_i]), 
                            self._compute_std(suma2[type_i], suma[type_i], sumn[type_i]), 
                            self._compute_std(suma2[type_i], suma[type_i], sumn[type_i])
                            ]
                davg = np.tile(davgunit, self.ndescrpt // 4)
                dstd = np.tile(dstdunit, self.ndescrpt // 4)
                all_davg.append(davg)
                all_dstd.append(dstd)

        if not self.set_davg_zero:
            self.davg = np.array(all_davg)
        self.dstd = np.array(all_dstd)

        np.save("tf", self.davg)

        self.t_avg = paddle.to_tensor(self.davg, dtype = GLOBAL_NP_FLOAT_PRECISION)
        self.t_std = paddle.to_tensor(self.dstd, dtype = GLOBAL_NP_FLOAT_PRECISION)

    def enable_compression(self,
                           min_nbor_dist : float,
                           model_file : str = 'frozon_model.pb',
                           table_extrapolate : float = 5,
                           table_stride_1 : float = 0.01,
                           table_stride_2 : float = 0.1,
                           check_frequency : int = -1
    ) -> None:
        """
        Reveive the statisitcs (distance, max_nbor_size and env_mat_range) of the training data.
        
        Parameters
        ----------
        min_nbor_dist
                The nearest distance between atoms
        model_file
                The original frozen model, which will be compressed by the program
        table_extrapolate
                The scale of model extrapolation
        table_stride_1
                The uniform stride of the first table
        table_stride_2
                The uniform stride of the second table
        check_frequency
                The overflow check frequency
        """
        self.compress = True
        self.model_file = model_file
        self.table_config = [table_extrapolate, table_stride_1, table_stride_2, check_frequency]
        self.table = DeepTabulate(self.model_file, self.type_one_side)
        self.lower, self.upper \
            = self.table.build(min_nbor_dist, 
                               table_extrapolate, 
                               table_stride_1, 
                               table_stride_2)

    def forward (self, 
               coord_, 
               atype_ ,
               natoms ,
               box_ , 
               mesh,
               input_dict, 
               reuse = None,
               suffix = ''):
        """
        Build the computational graph for the descriptor

        Parameters
        ----------
        coord_
                The coordinate of atoms
        atype_
                The type of atoms
        natoms
                The number of atoms. This tensor has the length of Ntypes + 2
                natoms[0]: number of local atoms
                natoms[1]: total number of atoms held by this processor
                natoms[i]: 2 <= i < Ntypes+2, number of type i atoms
        mesh
                For historical reasons, only the length of the Tensor matters.
                if size of mesh == 6, pbc is assumed. 
                if size of mesh == 0, no-pbc is assumed. 
        input_dict
                Dictionary for additional inputs
        reuse
                The weights in the networks should be reused when get the variable.
        suffix
                Name suffix to identify this descriptor

        Returns
        -------
        descriptor
                The output descriptor
        """

        coord = paddle.reshape(coord_, [-1, natoms[1] * 3])
        box   = paddle.reshape(box_, [-1, 9])
        atype = paddle.reshape(atype_, [-1, natoms[1]])

        self.descrpt, self.descrpt_deriv, self.rij, self.nlist \
            = paddle_ops.prod_env_mat_a(coord,
                                       atype,
                                       natoms,
                                       box,
                                       mesh,
                                       self.t_avg,
                                       self.t_std,
                                       rcut_a = self.rcut_a,
                                       rcut_r = self.rcut_r,
                                       rcut_r_smth = self.rcut_r_smth,
                                       sel_a = self.sel_a,
                                       sel_r = self.sel_r)

        self.descrpt_reshape = paddle.reshape(self.descrpt, [-1, self.ndescrpt])
        self.descrpt_reshape.stop_gradient = False

        self.dout, self.qmat = self._pass_filter(self.descrpt_reshape, 
                                                 atype,
                                                 natoms, 
                                                 input_dict,
                                                 suffix = suffix, 
                                                 reuse = reuse, 
                                                 trainable = self.trainable)
        
        return self.dout
    
    def get_rot_mat(self) -> paddle.Tensor:
        """
        Get rotational matrix
        """
        return self.qmat


    def prod_force_virial(self, 
                          atom_ener, 
                          natoms):
        """
        Compute force and virial

        Parameters
        ----------
        atom_ener
                The atomic energy
        natoms
                The number of atoms. This tensor has the length of Ntypes + 2
                natoms[0]: number of local atoms
                natoms[1]: total number of atoms held by this processor
                natoms[i]: 2 <= i < Ntypes+2, number of type i atoms
        Return
        ------
        force
                The force on atoms
        virial
                The total virial
        atom_virial
                The atomic virial
        """

        net_deriv = paddle.grad(atom_ener, self.descrpt_reshape, create_graph=True)[0]
        net_deriv_reshape = paddle.reshape (net_deriv, [-1, natoms[0] * self.ndescrpt])

        force \
            = paddle_ops.prod_force_se_a (net_deriv_reshape,
                                          self.descrpt_deriv,
                                          self.nlist,
                                          natoms,
                                          n_a_sel = self.nnei_a,
                                          n_r_sel = self.nnei_r)

        virial, atom_virial \
            = paddle_ops.prod_virial_se_a (net_deriv_reshape,
                                           self.descrpt_deriv,
                                           self.rij,
                                           self.nlist,
                                           natoms,
                                           n_a_sel = self.nnei_a,
                                           n_r_sel = self.nnei_r)

        return force, virial, atom_virial
        

    def _pass_filter(self, 
                     inputs,
                     atype,
                     natoms,
                     input_dict,
                     reuse = None,
                     suffix = '', 
                     trainable = True) :
        start_index = 0
        inputs = paddle.reshape(inputs, [-1, self.ndescrpt * natoms[0]])
        output = []
        output_qmat = []
        for type_i in range(self.ntypes):
            inputs_i = paddle.slice (inputs, axes=[0, 1], 
                                 starts = [0, start_index * self.ndescrpt],
                                 ends = [inputs.shape[0], (start_index + natoms[2+type_i]) * self.ndescrpt])
            inputs_i = paddle.reshape(inputs_i, [-1, self.ndescrpt])
            layer, qmat = self._filter(paddle.cast(inputs_i, self.filter_precision), type_i, natoms=natoms, reuse=reuse, seed = self.seed, trainable = trainable, activation_fn = self.filter_activation_fn)
            layer = paddle.reshape(layer, [inputs.shape[0], natoms[2+type_i] * self.get_dim_out()])
            qmat  = paddle.reshape(qmat,  [inputs.shape[0], natoms[2+type_i] * self.get_dim_rot_mat_1() * 3])
            output.append(layer)
            output_qmat.append(qmat)
            start_index += natoms[2+type_i]

        output = paddle.concat(output, axis = 1)
        output_qmat = paddle.concat(output_qmat, axis = 1)
        return output, output_qmat


    def _compute_dstats_sys_smth (self,
                                 data_coord, 
                                 data_box, 
                                 data_atype,                             
                                 natoms_vec,
                                 mesh) :
        input_dict = {}
        input_dict['coord'] = paddle.to_tensor(data_coord, dtype=GLOBAL_NP_FLOAT_PRECISION)
        input_dict['box'] = paddle.to_tensor(data_box, dtype=GLOBAL_PD_FLOAT_PRECISION)
        input_dict['type'] = paddle.to_tensor(data_atype, dtype="int32")
        input_dict['natoms_vec'] = paddle.to_tensor(natoms_vec, dtype="int32")
        input_dict['default_mesh'] = paddle.to_tensor(mesh, dtype="int32")
        
        self.stat_descrpt, descrpt_deriv, rij, nlist = paddle_ops.prod_env_mat_a(input_dict['coord'],
                                         input_dict['type'],
                                         input_dict['natoms_vec'],
                                         input_dict['box'],
                                         input_dict['default_mesh'],
                                         self.avg_zero,
                                         self.std_ones,
                                         rcut_a = self.rcut_a,
                                         rcut_r = self.rcut_r,
                                         rcut_r_smth = self.rcut_r_smth,
                                         sel_a = self.sel_a,
                                         sel_r = self.sel_r)

        dd_all = self.stat_descrpt.numpy()
        natoms = natoms_vec
        dd_all = np.reshape(dd_all, [-1, self.ndescrpt * natoms[0]])
        start_index = 0
        sysr = []
        sysa = []
        sysn = []
        sysr2 = []
        sysa2 = []
        for type_i in range(self.ntypes):
            end_index = start_index + self.ndescrpt * natoms[2+type_i]
            dd = dd_all[:, start_index:end_index]
            dd = np.reshape(dd, [-1, self.ndescrpt])
            start_index = end_index        
            # compute
            dd = np.reshape (dd, [-1, 4])
            ddr = dd[:,:1]
            dda = dd[:,1:]
            sumr = np.sum(ddr)
            suma = np.sum(dda) / 3.
            sumn = dd.shape[0]
            sumr2 = np.sum(np.multiply(ddr, ddr))
            suma2 = np.sum(np.multiply(dda, dda)) / 3.
            sysr.append(sumr)
            sysa.append(suma)
            sysn.append(sumn)
            sysr2.append(sumr2)
            sysa2.append(suma2)
        return sysr, sysr2, sysa, sysa2, sysn


    def _compute_std (self,sumv2, sumv, sumn) :
        val = np.sqrt(sumv2/sumn - np.multiply(sumv/sumn, sumv/sumn))
        if np.abs(val) < 1e-2:
            val = 1e-2
        return val

    def _filter(self, 
                inputs, 
                type_input,
                natoms,
                activation_fn=paddle.nn.functional.tanh, 
                stddev=1.0,
                bavg=0.0,
                reuse=None,
                seed=None, 
                trainable = True):
        # natom x (nei x 4)
        shape = inputs.shape
        outputs_size = [1] + self.filter_neuron
        outputs_size_2 = self.n_axis_neuron

        start_index = 0
        xyz_scatter_total = []
        for type_i in range(self.ntypes):
          # cut-out inputs
          # with natom x (nei_type_i x 4)  
          inputs_i = paddle.slice (inputs, axes=[0, 1],
                               starts = [ 0, start_index*4],
                               ends = [inputs.shape[0], (start_index + self.sel_a[type_i])* 4] )
          start_index += self.sel_a[type_i]
          shape_i = inputs_i.shape
          # with (natom x nei_type_i) x 4
          inputs_reshape = paddle.reshape(inputs_i, [-1, 4])
          # with (natom x nei_type_i) x 1
          xyz_scatter = paddle.reshape(paddle.slice(inputs_reshape, [0, 1],[0,0],[inputs_reshape.shape[0],1]), [-1,1])
          # with (natom x nei_type_i) x out_size
          
          xyz_scatter = self.embedding_nets[type_input][type_i](xyz_scatter)
         
          # natom x nei_type_i x out_size
          xyz_scatter = paddle.reshape(xyz_scatter, (-1, shape_i[1]//4, outputs_size[-1]))

          # xyz_scatter_total.append(xyz_scatter)
          if type_i == 0 :
              xyz_scatter_1 = paddle.fluid.layers.matmul(paddle.reshape(inputs_i, [-1, shape_i[1]//4, 4]), xyz_scatter, transpose_x = True)
          else :
              xyz_scatter_1 += paddle.fluid.layers.matmul(paddle.reshape(inputs_i, [-1, shape_i[1]//4, 4]), xyz_scatter, transpose_x = True)

        # natom x nei x outputs_size
        # xyz_scatter = paddle.concat(xyz_scatter_total, axis=1)
        # natom x nei x 4
        # inputs_reshape = paddle.reshape(inputs, [-1, shape[1]//4, 4])
        # natom x 4 x outputs_size
        # xyz_scatter_1 = paddle.matmul(inputs_reshape, xyz_scatter, transpose_a = True)
        xyz_scatter_1 = xyz_scatter_1 * (4.0 / shape[1])

        # natom x 4 x outputs_size_2
        xyz_scatter_2 = paddle.slice(xyz_scatter_1, [0,1,2], [0,0,0],[xyz_scatter_1.shape[0],xyz_scatter_1.shape[1],outputs_size_2])

        # # natom x 3 x outputs_size_2
        # qmat = paddle.slice(xyz_scatter_2, [0,1,0], [-1, 3, -1])
        # natom x 3 x outputs_size_1
        qmat = paddle.slice(xyz_scatter_1, [0,1,2], [0,1,0], [xyz_scatter_1.shape[0], 4, xyz_scatter_1.shape[2]])
        # natom x outputs_size_1 x 3
        qmat = paddle.transpose(qmat, perm = [0, 2, 1])
        # natom x outputs_size x outputs_size_2
        result = paddle.fluid.layers.matmul(xyz_scatter_1, xyz_scatter_2, transpose_x = True)
        # natom x (outputs_size x outputs_size_2)
        result = paddle.reshape(result, [-1, outputs_size_2 * outputs_size[-1]])

        return result, qmat

