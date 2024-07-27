from typing import (
    List,
    Optional,
    Tuple,
)

import numpy as np
import torch
import torch.nn.functional as F

from deepmd.pt.utils.tabulate import (
    DPTabulate,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
)
from deepmd.pt.utils.type_embed import (
    embed_atom_type,
)

class DescrptSeA():
    def enable_compression(
        self,
        min_nbor_dist: float, 
        table_extrapolate: float = 5,
        table_stride_1: float = 0.01,
        table_stride_2: float = 0.1,
        check_frequency: int = -1,
    ) -> None:
        """Reveive the statisitcs (distance, max_nbor_size and env_mat_range) of the training data.

        Parameters
        ----------
        min_nbor_dist
            The nearest distance between atoms
        table_extrapolate
            The scale of model extrapolation
        table_stride_1
            The uniform stride of the first table
        table_stride_2
            The uniform stride of the second table
        check_frequency
            The overflow check frequency
        suffix : str, optional
            The suffix of the scope
        """
        self.compress = True
        self.table = DPTabulate(

        )
        self.table_config = [
            table_extrapolate,
            table_stride_1,
            table_stride_2,
            check_frequency,
        ]
        self.lower, self.upper = self.table.build(
            min_nbor_dist, table_extrapolate, table_stride_1, table_stride_2
        )
    
    def build(
        self,
        coord_: torch.Tensor,
        atype_: torch.Tensor,
        natoms: torch.Tensor,
        box_: torch.Tensor,
        mesh: torch.Tensor,
        input_dict: dict,
        reuse: Optional[bool] = None,
        suffix: str = "",
    ) -> torch.Tensor:
        """Build the computational graph for the descriptor.

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
        box_ : torch.Tensor
            The box of the system
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
        davg = self.davg
        dstd = self.dstd

        if davg is None:
            davg = np.zeros([self.ntypes, self.ndescrpt])
        if dstd is None:
            dstd = np.ones([self.ntypes, self.ndescrpt])

        self.t_avg = torch.nn.Parameter(
            torch.tensor(davg, dtype=PRECISION_DICT["default"]),
            requires_grad=False,
        )
        self.t_std = torch.nn.Parameter(
            torch.tensor(dstd, dtype=PRECISION_DICT["default"]),
            requires_grad=False,
        )

        with torch.no_grad():
            coord = coord_.view(-1, natoms[1] * 3)
            box = box_.view(-1, 9)
            atype = atype_.view(-1, natoms[1])
        self.atype = atype

        # Assuming build_op_descriptor and other methods are defined elsewhere
        op_descriptor = op_module.prod_env_mat_a
        self.descrpt, self.descrpt_deriv, self.rij, self.nlist = op_descriptor(
            coord,
            atype,
            natoms,
            box,
            mesh,
            self.t_avg,
            self.t_std,
            rcut_a=self.rcut_a,
            rcut_r=self.rcut_r,
            rcut_r_smth=self.rcut_r_smth,
            sel_a=self.sel_a,
            sel_r=self.sel_r,
        )

        nlist_t = self.nlist.view(-1) + 1
        atype_t = torch.cat([torch.tensor([self.ntypes]), self.atype.view(-1)], dim=0)
        self.nei_type_vec = torch.nn.functional.embedding(nlist_t, atype_t)

        # Assuming _identity_tensors and _pass_filter are defined elsewhere
        self.descrpt_reshape = self.descrpt.view(-1, self.ndescrpt)

        self.dout, self.qmat = self._pass_filter(
            self.descrpt_reshape,
            atype,
            natoms,
            input_dict,
            suffix=suffix,
            reuse=reuse,
            trainable=self.trainable,
        )

        return self.dout
    
    def _pass_filter(
        self, inputs, atype, natoms, input_dict, reuse=None, suffix="", trainable=True
    ):
        if input_dict is not None:
            type_embedding = input_dict.get("type_embedding", None)
            if type_embedding is not None:
                self.use_tebd = True
        else:
            type_embedding = None
        if self.stripped_type_embedding and type_embedding is None:
            raise RuntimeError("type_embedding is required for se_a_tebd_v2 model.")
        
        start_index = 0
        inputs = inputs.view(-1, natoms[0], self.ndescrpt)
        output = []
        output_qmat = []
        
        if not self.type_one_side and type_embedding is None:
            for type_i in range(self.ntypes):
                inputs_i = inputs[:, start_index:start_index + natoms[2 + type_i], :]
                inputs_i = inputs_i.view(-1, self.ndescrpt)
                filter_name = "filtseler_type_" + str(type_i) + suffix
                layer, qmat = self._filter(
                    inputs_i,
                    type_i,
                    name=filter_name,
                    natoms=natoms,
                    reuse=reuse,
                    trainable=trainable,
                    activation_fn=self.filter_activation_fn,
                )
                layer = layer.view(inputs.size(0), natoms[2 + type_i], self.get_dim_out())
                qmat = qmat.view(inputs.size(0), natoms[2 + type_i], self.get_dim_rot_mat_1() * 3)
                output.append(layer)
                output_qmat.append(qmat)
                start_index += natoms[2 + type_i]
        else:
            inputs_i = inputs.view(-1, self.ndescrpt)
            type_i = -1
            # if nvnmd_cfg.enable and nvnmd_cfg.quantize_descriptor:
            #     inputs_i = descrpt2r4(inputs_i, natoms)
            
            self.atype_nloc = atype[:, :natoms[0]].view(-1)
            
            if len(self.exclude_types):
                mask = self.build_type_exclude_mask(
                    self.exclude_types,
                    self.ntypes,
                    self.sel_a,
                    self.ndescrpt,
                    self.atype_nloc,
                    inputs_i.size(0),
                )
                inputs_i *= mask

            layer, qmat = self._filter(
                inputs_i,
                type_i,
                name="filter_type_all" + suffix,
                natoms=natoms,
                reuse=reuse,
                trainable=trainable,
                activation_fn=self.filter_activation_fn,
                type_embedding=type_embedding,
            )
            layer = layer.view(inputs.size(0), natoms[0], self.get_dim_out())
            qmat = qmat.view(inputs.size(0), natoms[0], self.get_dim_rot_mat_1() * 3)
            output.append(layer)
            output_qmat.append(qmat)
        
        output = torch.cat(output, dim=1)
        output_qmat = torch.cat(output_qmat, dim=1)
        
        return output, output_qmat
    
    def _filter(
            self,
            inputs,
            type_input,
            natoms,
            type_embedding=None,
        ):
        nframes = inputs.view(-1, natoms[0], self.ndescrpt).shape[0]
        # natom x (nei x 4)
        shape = list(inputs.shape)
        outputs_size = [1, *self.filter_neuron]
        outputs_size_2 = self.n_axis_neuron
        all_excluded = all(
            (type_input, type_i) in self.exclude_types for type_i in range(self.ntypes)
        )
        if all_excluded:
            # all types are excluded so result and qmat should be zeros
            # we can safaly return a zero matrix...
            # See also https://stackoverflow.com/a/34725458/9567349
            # result: natom x outputs_size x outputs_size_2
            # qmat: natom x outputs_size x 3
            natom = inputs.shape[0]
            result = torch.zeros((natom, outputs_size_2, outputs_size[-1]), dtype=PRECISION_DICT["default"])
            qmat = torch.zeros((natom, outputs_size[-1], 3), dtype=PRECISION_DICT["default"])
            return result, qmat

        start_index = 0
        type_i = 0
        # natom x 4 x outputs_size
        if type_embedding is None:
            rets = []
            for type_i in range(self.ntypes):
                ret = self._filter_lower(
                    type_i,
                    type_input,
                    start_index,
                    self.sel_a[type_i],
                    inputs,
                    nframes,
                    natoms,
                    type_embedding=type_embedding,
                    is_exclude=(type_input, type_i) in self.exclude_types,
                )
                if (type_input, type_i) not in self.exclude_types:
                    # add zero is meaningless; skip
                    rets.append(ret)
                start_index += self.sel_a[type_i]
            # faster to use add_n than multiple add
            xyz_scatter_1 = torch.stack(rets).sum(0)
        else:
            xyz_scatter_1 = self._filter_lower(
                type_i,
                type_input,
                start_index,
                np.cumsum(self.sel_a)[-1],
                inputs,
                nframes,
                natoms,
                type_embedding=type_embedding,
                is_exclude=False,
            )

        if self.original_sel is None:
            # shape[1] = nnei * 4
            nnei = shape[1] // 4
        else:
            nnei = torch.tensor(sum(self.original_sel), dtype=torch.int32)

        xyz_scatter_1 = xyz_scatter_1 / nnei
        # natom x 4 x outputs_size_2
        xyz_scatter_2 = xyz_scatter_1[:, :, :outputs_size_2]
        # # natom x 3 x outputs_size_2
        # qmat = tf.slice(xyz_scatter_2, [0,1,0], [-1, 3, -1])
        # natom x 3 x outputs_size_1
        qmat = xyz_scatter_1[:, 1:4, :]
        # natom x outputs_size_1 x 3
        qmat = qmat.permute(0, 2, 1)
        # natom x outputs_size x outputs_size_2
        result = torch.matmul(xyz_scatter_1.permute(0, 2, 1), xyz_scatter_2)
        # natom x (outputs_size x outputs_size_2)
        result = result.view(-1, outputs_size_2 * outputs_size[-1])

        return result, qmat
    
    def _filter_lower(
            self,
            type_i,
            type_input,
            start_index,
            incrs_index,
            inputs,
            nframes,
            natoms,
            type_embedding=None,
            is_exclude=False,
        ):
        """Input env matrix, returns R.G."""
        outputs_size = [1, *self.filter_neuron]
        # cut-out inputs
        # with natom x (nei_type_i x 4)
        inputs_i = inputs[:, start_index * 4:start_index * 4 + incrs_index * 4]
        shape_i = inputs_i.shape
        natom = inputs_i.shape[0]
        # reshape inputs
        # with (natom x nei_type_i) x 4
        inputs_reshape = inputs_i.view(-1, 4)
        # with (natom x nei_type_i) x 1
        xyz_scatter = inputs_reshape[:, :1]

        if type_embedding is not None:
            if self.stripped_type_embedding:
                if self.type_one_side:
                    extra_embedding_index = self.nei_type_vec
                else:
                    padding_ntypes = type_embedding.shape[0]
                    atype_expand = self.atype_nloc.view(-1, 1)
                    idx_i = atype_expand * padding_ntypes
                    idx_j = self.nei_type_vec.view(-1, self.nnei)
                    idx = idx_i + idx_j
                    index_of_two_side = idx.view(-1)
                    extra_embedding_index = index_of_two_side
            else:
                xyz_scatter = self._concat_type_embedding(xyz_scatter, nframes, natoms, type_embedding)
                if self.compress:
                    raise RuntimeError(
                        "compression of type embedded descriptor is not supported when tebd_input_mode is not set to 'strip'"
                    )
        
        if self.compress and (not is_exclude):
            if self.stripped_type_embedding:
                net_output = F.embedding(extra_embedding_index, self.extra_embedding)
                net = "filter_net"
                info = [
                    self.lower[net],
                    self.upper[net],
                    self.upper[net] * self.table_config[0],
                    self.table_config[1],
                    self.table_config[2],
                    self.table_config[3],
                ]
                return torch.ops.deepmd.tabulate_fusion_se_atten(
                    self.table.data[net].to(dtype=self.filter_precision),
                    info,
                    xyz_scatter,
                    inputs_i.view(natom, shape_i[1] // 4, 4),
                    net_output,
                    last_layer_size=outputs_size[-1],
                    is_sorted=False,
                )
            else:
                net = f"filter_{'-1' if self.type_one_side else str(type_input)}_net_{type_i}"
                info = [
                    self.lower[net],
                    self.upper[net],
                    self.upper[net] * self.table_config[0],
                    self.table_config[1],
                    self.table_config[2],
                    self.table_config[3],
                ]
                return torch.ops.deepmd.tabulate_fusion_se_a(
                    self.table.data[net].to(dtype=self.filter_precision),
                    info,
                    xyz_scatter,
                    inputs_i.view(natom, shape_i[1] // 4, 4),
                    last_layer_size=outputs_size[-1],
                )
    
    def _concat_type_embedding(
        self,
        xyz_scatter,
        nframes,
        natoms,
        type_embedding,
    ):
        """Concatenate `type_embedding` of neighbors and `xyz_scatter`.
        If not self.type_one_side, concatenate `type_embedding` of center atoms as well.

        Parameters
        ----------
        xyz_scatter:
            shape is [nframes*natoms[0]*self.nnei, 1]
        nframes:
            shape is []
        natoms:
            shape is [1+1+self.ntypes]
        type_embedding:
            shape is [self.ntypes, Y] where Y=jdata['type_embedding']['neuron'][-1]

        Returns
        -------
        embedding:
            environment of each atom represented by embedding.
        """
        te_out_dim = type_embedding.size(-1)
        self.t_nei_type = torch.tensor(self.nei_type, dtype=torch.int32)
        nei_embed = type_embedding[self.t_nei_type.long()]  # shape is [self.nnei, 1+te_out_dim]
        nei_embed = nei_embed.repeat(nframes * natoms[0], 1)  # shape is [nframes*natoms[0]*self.nnei, te_out_dim]
        nei_embed = nei_embed.view(-1, te_out_dim)
        embedding_input = torch.cat([xyz_scatter, nei_embed], dim=1)  # shape is [nframes*natoms[0]*self.nnei, 1+te_out_dim]
        
        if not self.type_one_side:
            atm_embed = embed_atom_type(self.ntypes, natoms, type_embedding)  # shape is [natoms[0], te_out_dim]
            atm_embed = atm_embed.repeat(nframes, self.nnei)  # shape is [nframes*natoms[0], self.nnei*te_out_dim]
            atm_embed = atm_embed.view(-1, te_out_dim)  # shape is [nframes*natoms[0]*self.nnei, te_out_dim]
            embedding_input = torch.cat([embedding_input, atm_embed], dim=1)  # shape is [nframes*natoms[0]*self.nnei, 1+te_out_dim+te_out_dim]
        
        return embedding_input