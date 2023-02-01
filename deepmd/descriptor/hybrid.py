import numpy as np
from typing import Tuple, List

from deepmd.env import tf
from deepmd.env import op_module
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION
from deepmd.env import GLOBAL_NP_FLOAT_PRECISION
# from deepmd.descriptor import DescrptLocFrame
# from deepmd.descriptor import DescrptSeA
# from deepmd.descriptor import DescrptSeT
# from deepmd.descriptor import DescrptSeAEbd
# from deepmd.descriptor import DescrptSeAEf
# from deepmd.descriptor import DescrptSeR
from .descriptor import Descriptor
from .se_a import DescrptSeA
from .se_r import DescrptSeR
from .se_t import DescrptSeT
from .se_a_ebd import DescrptSeAEbd
from .se_a_ef import DescrptSeAEf
from .loc_frame import DescrptLocFrame

@Descriptor.register("hybrid")
class DescrptHybrid (Descriptor):
    """Concate a list of descriptors to form a new descriptor.

    Parameters
    ----------
    list : list
            Build a descriptor from the concatenation of the list of descriptors.
    """
    def __init__ (self, 
                  list : list,
                  multi_task: bool = False
    ) -> None :
        """
        Constructor
        """
        # warning: list is conflict with built-in list
        descrpt_list = list
        if descrpt_list == [] or descrpt_list is None:
            raise RuntimeError('cannot build descriptor from an empty list of descriptors.')
        formatted_descript_list = []
        self.multi_task = multi_task
        for ii in descrpt_list:
            if isinstance(ii, Descriptor):
                formatted_descript_list.append(ii)
            elif isinstance(ii, dict):
                if multi_task:
                    ii['multi_task'] = True
                formatted_descript_list.append(Descriptor(**ii))
            else:
                raise NotImplementedError
        self.descrpt_list = formatted_descript_list
        self.numb_descrpt = len(self.descrpt_list)
        for ii in range(1, self.numb_descrpt):
            assert(self.descrpt_list[ii].get_ntypes() == 
                   self.descrpt_list[ 0].get_ntypes()), \
                   f'number of atom types in {ii}th descrptor does not match others'


    def get_rcut (self) -> float:
        """
        Returns the cut-off radius
        """
        all_rcut = [ii.get_rcut() for ii in self.descrpt_list]
        return np.max(all_rcut)


    def get_ntypes (self) -> int:
        """
        Returns the number of atom types
        """
        return self.descrpt_list[0].get_ntypes()


    def get_dim_out (self) -> int:
        """
        Returns the output dimension of this descriptor
        """
        all_dim_out = [ii.get_dim_out() for ii in self.descrpt_list]
        return sum(all_dim_out)


    def get_nlist(
            self,
    ) -> Tuple[tf.Tensor, tf.Tensor, List[int], List[int]]:
        """Get the neighbor information of the descriptor, returns the
        nlist of the descriptor with the largest cut-off radius.

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
        maxr_idx = np.argmax([ii.get_rcut() for ii in self.descrpt_list])
        return self.get_nlist_i(maxr_idx)


    def get_nlist_i(self, 
                    ii : int
    ) -> Tuple[tf.Tensor, tf.Tensor, List[int], List[int]]:
        """Get the neighbor information of the ii-th descriptor

        Parameters
        ----------
        ii : int
                The index of the descriptor

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
        return self.descrpt_list[ii].nlist, self.descrpt_list[ii].rij, self.descrpt_list[ii].sel_a, self.descrpt_list[ii].sel_r
    

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
        for ii in self.descrpt_list:
            ii.compute_input_stats(data_coord, data_box, data_atype, natoms_vec, mesh, input_dict)

    def merge_input_stats(self, stat_dict):
        """
        Merge the statisitcs computed from compute_input_stats to obtain the self.davg and self.dstd.

        Parameters
        ----------
        stat_dict
                The dict of statisitcs computed from compute_input_stats, including:
            sumr
                    The sum of radial statisitcs.
            suma
                    The sum of relative coord statisitcs.
            sumn
                    The sum of neighbor numbers.
            sumr2
                    The sum of square of radial statisitcs.
            suma2
                    The sum of square of relative coord statisitcs.
        """
        for ii in self.descrpt_list:
            ii.merge_input_stats(stat_dict)

    def build (self, 
               coord_ : tf.Tensor, 
               atype_ : tf.Tensor,
               natoms : tf.Tensor,
               box_ : tf.Tensor, 
               mesh : tf.Tensor,
               input_dict : dict, 
               reuse : bool = None,
               suffix : str = ''
    ) -> tf.Tensor:
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
        with tf.variable_scope('descrpt_attr' + suffix, reuse = reuse) :
            t_rcut = tf.constant(self.get_rcut(), 
                                 name = 'rcut', 
                                 dtype = GLOBAL_TF_FLOAT_PRECISION)
            t_ntypes = tf.constant(self.get_ntypes(), 
                                   name = 'ntypes', 
                                   dtype = tf.int32)
        all_dout = []
        for idx,ii in enumerate(self.descrpt_list):
            dout = ii.build(coord_, atype_, natoms, box_, mesh, input_dict, suffix=suffix+f'_{idx}', reuse=reuse)
            dout = tf.reshape(dout, [-1, ii.get_dim_out()])
            all_dout.append(dout)
        dout = tf.concat(all_dout, axis = 1)
        dout = tf.reshape(dout, [-1, natoms[0], self.get_dim_out()])
        return dout
        

    def prod_force_virial(self, 
                          atom_ener : tf.Tensor, 
                          natoms : tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
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

        Returns
        -------
        force
                The force on atoms
        virial
                The total virial
        atom_virial
                The atomic virial
        """
        for idx,ii in enumerate(self.descrpt_list):
            ff, vv, av = ii.prod_force_virial(atom_ener, natoms)
            if idx == 0:
                force = ff
                virial = vv
                atom_virial = av
            else:
                force += ff
                virial += vv
                atom_virial += av
        return force, virial, atom_virial

    def enable_compression(self,
                           min_nbor_dist: float,
                           graph: tf.Graph,
                           graph_def: tf.GraphDef,
                           table_extrapolate: float = 5.,
                           table_stride_1: float = 0.01,
                           table_stride_2: float = 0.1,
                           check_frequency: int = -1,
                           suffix: str = ""
                           ) -> None:
        """
        Reveive the statisitcs (distance, max_nbor_size and env_mat_range) of the
        training data.

        Parameters
        ----------
        min_nbor_dist : float
                The nearest distance between atoms
        graph : tf.Graph
                The graph of the model
        graph_def : tf.GraphDef
                The graph_def of the model
        table_extrapolate : float, default: 5.
                The scale of model extrapolation
        table_stride_1 : float, default: 0.01
                The uniform stride of the first table
        table_stride_2 : float, default: 0.1
                The uniform stride of the second table
        check_frequency : int, default: -1
                The overflow check frequency
        suffix : str, optional
                The suffix of the scope
        """
        for idx, ii in enumerate(self.descrpt_list):
            ii.enable_compression(min_nbor_dist, graph, graph_def, table_extrapolate, table_stride_1, table_stride_2, check_frequency, suffix=f"{suffix}_{idx}")


    def enable_mixed_precision(self, mixed_prec : dict = None) -> None:
        """
        Reveive the mixed precision setting.

        Parameters
        ----------
        mixed_prec
                The mixed precision setting used in the embedding net
        """
        for idx, ii in enumerate(self.descrpt_list):
            ii.enable_mixed_precision(mixed_prec)


    def init_variables(self,
                       graph: tf.Graph,
                       graph_def: tf.GraphDef,
                       suffix : str = "",
    ) -> None:
        """
        Init the embedding net variables with the given dict

        Parameters
        ----------
        graph : tf.Graph
            The input frozen model graph
        graph_def : tf.GraphDef
            The input frozen model graph_def
        suffix : str, optional
            The suffix of the scope
        """
        for idx, ii in enumerate(self.descrpt_list):
            ii.init_variables(graph, graph_def, suffix=f"{suffix}_{idx}")

    def get_tensor_names(self, suffix : str = "") -> Tuple[str]:
        """Get names of tensors.
        
        Parameters
        ----------
        suffix : str
            The suffix of the scope

        Returns
        -------
        Tuple[str]
            Names of tensors
        """
        tensor_names = []
        for idx, ii in enumerate(self.descrpt_list):
            tensor_names.extend(ii.get_tensor_names(suffix=f"{suffix}_{idx}"))
        return tuple(tensor_names)

    def pass_tensors_from_frz_model(self,
                                    *tensors : tf.Tensor,
    ) -> None:
        """
        Pass the descrpt_reshape tensor as well as descrpt_deriv tensor from the frz graph_def

        Parameters
        ----------
        *tensors : tf.Tensor
            passed tensors
        """
        jj = 0
        for ii in self.descrpt_list:
            n_tensors = len(ii.get_tensor_names())
            ii.pass_tensors_from_frz_model(*tensors[jj:jj+n_tensors])
            jj += n_tensors
