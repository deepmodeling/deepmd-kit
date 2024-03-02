# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Callable,
    List,
    Optional,
    Union,
)

import torch

from deepmd.pt.model.network.network import (
    Identity,
    Linear,
    TypeEmbedNet,
)
from deepmd.pt.utils.nlist import (
    build_multiple_neighbor_list,
    get_multiple_nlist_key,
)
from deepmd.pt.utils.update_sel import (
    UpdateSel,
)
from deepmd.utils.path import (
    DPPath,
)

from .base_descriptor import (
    BaseDescriptor,
)
from .repformers import (
    DescrptBlockRepformers,
)
from .se_atten import (
    DescrptBlockSeAtten,
)


@BaseDescriptor.register("dpa2")
class DescrptDPA2(torch.nn.Module, BaseDescriptor):
    def __init__(
        self,
        ntypes: int,
        repinit_rcut: float,
        repinit_rcut_smth: float,
        repinit_nsel: int,
        repformer_rcut: float,
        repformer_rcut_smth: float,
        repformer_nsel: int,
        # kwargs
        tebd_dim: int = 8,
        concat_output_tebd: bool = True,
        repinit_neuron: List[int] = [25, 50, 100],
        repinit_axis_neuron: int = 16,
        repinit_set_davg_zero: bool = True,  # TODO
        repinit_activation="tanh",
        # repinit still unclear:
        # ffn, ffn_embed_dim, scaling_factor, normalize,
        repformer_nlayers: int = 3,
        repformer_g1_dim: int = 128,
        repformer_g2_dim: int = 16,
        repformer_axis_dim: int = 4,
        repformer_do_bn_mode: str = "no",
        repformer_bn_momentum: float = 0.1,
        repformer_update_g1_has_conv: bool = True,
        repformer_update_g1_has_drrd: bool = True,
        repformer_update_g1_has_grrg: bool = True,
        repformer_update_g1_has_attn: bool = True,
        repformer_update_g2_has_g1g1: bool = True,
        repformer_update_g2_has_attn: bool = True,
        repformer_update_h2: bool = False,
        repformer_attn1_hidden: int = 64,
        repformer_attn1_nhead: int = 4,
        repformer_attn2_hidden: int = 16,
        repformer_attn2_nhead: int = 4,
        repformer_attn2_has_gate: bool = False,
        repformer_activation: str = "tanh",
        repformer_update_style: str = "res_avg",
        repformer_set_davg_zero: bool = True,  # TODO
        repformer_add_type_ebd_to_seq: bool = False,
        trainable: bool = True,
        type: Optional[
            str
        ] = None,  # work around the bad design in get_trainer and DpLoaderSet!
        rcut: Optional[
            float
        ] = None,  # work around the bad design in get_trainer and DpLoaderSet!
        rcut_smth: Optional[
            float
        ] = None,  # work around the bad design in get_trainer and DpLoaderSet!
        sel: Optional[
            int
        ] = None,  # work around the bad design in get_trainer and DpLoaderSet!
    ):
        r"""The DPA-2 descriptor. see https://arxiv.org/abs/2312.15492.

        Parameters
        ----------
        ntypes : int
            Number of atom types
        repinit_rcut : float
            The cut-off radius of the repinit block
        repinit_rcut_smth : float
            From this position the inverse distance smoothly decays
            to 0 at the cut-off. Use in the repinit block.
        repinit_nsel : int
            Maximally possible number of neighbors for repinit block.
        repformer_rcut : float
            The cut-off radius of the repformer block
        repformer_rcut_smth : float
            From this position the inverse distance smoothly decays
            to 0 at the cut-off. Use in the repformer block.
        repformer_nsel : int
            Maximally possible number of neighbors for repformer block.
        tebd_dim : int
            The dimension of atom type embedding
        concat_output_tebd : bool
            Whether to concat type embedding at the output of the descriptor.
        repinit_neuron : List[int]
            repinit block: the number of neurons in the embedding net.
        repinit_axis_neuron : int
            repinit block: the number of dimension of split  in the
            symmetrization op.
        repinit_activation : str
            repinit block: the activation function in the embedding net
        repformer_nlayers : int
            repformers block: the number of repformer layers
        repformer_g1_dim : int
            repformers block: the dimension of single-atom rep
        repformer_g2_dim : int
            repformers block: the dimension of invariant pair-atom rep
        repformer_axis_dim : int
            repformers block: the number of dimension of split  in the
            symmetrization ops.
        repformer_do_bn_mode : bool
            repformers block: do batch norm in the repformer layers
        repformer_bn_momentum : float
            repformers block: moment in the batch normalization
        repformer_update_g1_has_conv : bool
            repformers block: update the g1 rep with convolution term
        repformer_update_g1_has_drrd : bool
            repformers block: update the g1 rep with the drrd term
        repformer_update_g1_has_grrg : bool
            repformers block: update the g1 rep with the grrg term
        repformer_update_g1_has_attn : bool
            repformers block: update the g1 rep with the localized
            self-attention
        repformer_update_g2_has_g1g1 : bool
            repformers block: update the g2 rep with the g1xg1 term
        repformer_update_g2_has_attn : bool
            repformers block: update the g2 rep with the gated self-attention
        repformer_update_h2 : bool
            repformers block: update the h2 rep
        repformer_attn1_hidden : int
            repformers block: the hidden dimension of localized self-attention
        repformer_attn1_nhead : int
            repformers block: the number of heads in localized self-attention
        repformer_attn2_hidden : int
            repformers block: the hidden dimension of gated self-attention
        repformer_attn2_nhead : int
            repformers block: the number of heads in gated self-attention
        repformer_attn2_has_gate : bool
            repformers block: has gate in the gated self-attention
        repformer_activation : str
            repformers block: the activation function in the MLPs.
        repformer_update_style : str
            repformers block: style of update a rep.
            can be res_avg or res_incr.
            res_avg updates a rep `u` with:
                    u = 1/\sqrt{n+1} (u + u_1 + u_2 + ... + u_n)
            res_incr updates a rep `u` with:
                    u = u + 1/\sqrt{n} (u_1 + u_2 + ... + u_n)
        repformer_set_davg_zero : bool
            repformers block: set the avg to zero in statistics
        repformer_add_type_ebd_to_seq : bool
            repformers block: concatenate the type embedding at the output.
        trainable : bool
            If the parameters in the descriptor are trainable.

        Returns
        -------
        descriptor:         torch.Tensor
            the descriptor of shape nb x nloc x g1_dim.
            invariant single-atom representation.
        g2:                 torch.Tensor
            invariant pair-atom representation.
        h2:                 torch.Tensor
            equivariant pair-atom representation.
        rot_mat:            torch.Tensor
            rotation matrix for equivariant fittings
        sw:                 torch.Tensor
            The switch function for decaying inverse distance.

        """
        super().__init__()
        del type, rcut, rcut_smth, sel
        self.repinit = DescrptBlockSeAtten(
            repinit_rcut,
            repinit_rcut_smth,
            repinit_nsel,
            ntypes,
            attn_layer=0,
            neuron=repinit_neuron,
            axis_neuron=repinit_axis_neuron,
            tebd_dim=tebd_dim,
            tebd_input_mode="concat",
            # tebd_input_mode='dot_residual_s',
            set_davg_zero=repinit_set_davg_zero,
            activation_function=repinit_activation,
        )
        self.repformers = DescrptBlockRepformers(
            repformer_rcut,
            repformer_rcut_smth,
            repformer_nsel,
            ntypes,
            nlayers=repformer_nlayers,
            g1_dim=repformer_g1_dim,
            g2_dim=repformer_g2_dim,
            axis_dim=repformer_axis_dim,
            direct_dist=False,
            do_bn_mode=repformer_do_bn_mode,
            bn_momentum=repformer_bn_momentum,
            update_g1_has_conv=repformer_update_g1_has_conv,
            update_g1_has_drrd=repformer_update_g1_has_drrd,
            update_g1_has_grrg=repformer_update_g1_has_grrg,
            update_g1_has_attn=repformer_update_g1_has_attn,
            update_g2_has_g1g1=repformer_update_g2_has_g1g1,
            update_g2_has_attn=repformer_update_g2_has_attn,
            update_h2=repformer_update_h2,
            attn1_hidden=repformer_attn1_hidden,
            attn1_nhead=repformer_attn1_nhead,
            attn2_hidden=repformer_attn2_hidden,
            attn2_nhead=repformer_attn2_nhead,
            attn2_has_gate=repformer_attn2_has_gate,
            activation_function=repformer_activation,
            update_style=repformer_update_style,
            set_davg_zero=repformer_set_davg_zero,
            smooth=True,
            add_type_ebd_to_seq=repformer_add_type_ebd_to_seq,
        )
        self.type_embedding = TypeEmbedNet(ntypes, tebd_dim)
        if self.repinit.dim_out == self.repformers.dim_in:
            self.g1_shape_tranform = Identity()
        else:
            self.g1_shape_tranform = Linear(
                self.repinit.dim_out,
                self.repformers.dim_in,
                bias=False,
                init="glorot",
            )
        assert self.repinit.rcut > self.repformers.rcut
        assert self.repinit.sel[0] > self.repformers.sel[0]
        self.concat_output_tebd = concat_output_tebd
        self.tebd_dim = tebd_dim
        self.rcut = self.repinit.get_rcut()
        self.ntypes = ntypes
        self.sel = self.repinit.sel
        # set trainable
        for param in self.parameters():
            param.requires_grad = trainable

    def get_rcut(self) -> float:
        """Returns the cut-off radius."""
        return self.rcut

    def get_nsel(self) -> int:
        """Returns the number of selected atoms in the cut-off radius."""
        return sum(self.sel)

    def get_sel(self) -> List[int]:
        """Returns the number of selected atoms for each type."""
        return self.sel

    def get_ntypes(self) -> int:
        """Returns the number of element types."""
        return self.ntypes

    def get_dim_out(self) -> int:
        """Returns the output dimension of this descriptor."""
        ret = self.repformers.dim_out
        if self.concat_output_tebd:
            ret += self.tebd_dim
        return ret

    def get_dim_emb(self) -> int:
        """Returns the embedding dimension of this descriptor."""
        return self.repformers.dim_emb

    def mixed_types(self) -> bool:
        """If true, the discriptor
        1. assumes total number of atoms aligned across frames;
        2. requires a neighbor list that does not distinguish different atomic types.

        If false, the discriptor
        1. assumes total number of atoms of each atom type aligned across frames;
        2. requires a neighbor list that distinguishes different atomic types.

        """
        return True

    def share_params(self, base_class, shared_level, resume=False):
        """
        Share the parameters of self to the base_class with shared_level during multitask training.
        If not start from checkpoint (resume is False),
        some seperated parameters (e.g. mean and stddev) will be re-calculated across different classes.
        """
        assert (
            self.__class__ == base_class.__class__
        ), "Only descriptors of the same type can share params!"
        # For DPA2 descriptors, the user-defined share-level
        # shared_level: 0
        # share all parameters in type_embedding, repinit and repformers
        if shared_level == 0:
            self._modules["type_embedding"] = base_class._modules["type_embedding"]
            self.repinit.share_params(base_class.repinit, 0, resume=resume)
            self._modules["g1_shape_tranform"] = base_class._modules[
                "g1_shape_tranform"
            ]
            self.repformers.share_params(base_class.repformers, 0, resume=resume)
        # shared_level: 1
        # share all parameters in type_embedding and repinit
        elif shared_level == 1:
            self._modules["type_embedding"] = base_class._modules["type_embedding"]
            self.repinit.share_params(base_class.repinit, 0, resume=resume)
        # shared_level: 2
        # share all parameters in type_embedding and repformers
        elif shared_level == 2:
            self._modules["type_embedding"] = base_class._modules["type_embedding"]
            self._modules["g1_shape_tranform"] = base_class._modules[
                "g1_shape_tranform"
            ]
            self.repformers.share_params(base_class.repformers, 0, resume=resume)
        # shared_level: 3
        # share all parameters in type_embedding
        elif shared_level == 3:
            self._modules["type_embedding"] = base_class._modules["type_embedding"]
        # Other shared levels
        else:
            raise NotImplementedError

    @property
    def dim_out(self):
        return self.get_dim_out()

    @property
    def dim_emb(self):
        """Returns the embedding dimension g2."""
        return self.get_dim_emb()

    def compute_input_stats(
        self,
        merged: Union[Callable[[], List[dict]], List[dict]],
        path: Optional[DPPath] = None,
    ):
        """
        Compute the input statistics (e.g. mean and stddev) for the descriptors from packed data.

        Parameters
        ----------
        merged : Union[Callable[[], List[dict]], List[dict]]
            - List[dict]: A list of data samples from various data systems.
                Each element, `merged[i]`, is a data dictionary containing `keys`: `torch.Tensor`
                originating from the `i`-th data system.
            - Callable[[], List[dict]]: A lazy function that returns data samples in the above format
                only when needed. Since the sampling process can be slow and memory-intensive,
                the lazy function helps by only sampling once.
        path : Optional[DPPath]
            The path to the stat file.

        """
        for ii, descrpt in enumerate([self.repinit, self.repformers]):
            descrpt.compute_input_stats(merged, path)

    def serialize(self) -> dict:
        """Serialize the obj to dict."""
        raise NotImplementedError

    @classmethod
    def deserialize(cls) -> "DescrptDPA2":
        """Deserialize from a dict."""
        raise NotImplementedError

    def forward(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
    ):
        """Compute the descriptor.

        Parameters
        ----------
        coord_ext
            The extended coordinates of atoms. shape: nf x (nallx3)
        atype_ext
            The extended aotm types. shape: nf x nall
        nlist
            The neighbor list. shape: nf x nloc x nnei
        mapping
            The index mapping, mapps extended region index to local region.

        Returns
        -------
        descriptor
            The descriptor. shape: nf x nloc x (ng x axis_neuron)
        gr
            The rotationally equivariant and permutationally invariant single particle
            representation. shape: nf x nloc x ng x 3
        g2
            The rotationally invariant pair-partical representation.
            shape: nf x nloc x nnei x ng
        h2
            The rotationally equivariant pair-partical representation.
            shape: nf x nloc x nnei x 3
        sw
            The smooth switch function. shape: nf x nloc x nnei

        """
        nframes, nloc, nnei = nlist.shape
        nall = extended_coord.view(nframes, -1).shape[1] // 3
        # nlists
        nlist_dict = build_multiple_neighbor_list(
            extended_coord,
            nlist,
            [self.repformers.get_rcut(), self.repinit.get_rcut()],
            [self.repformers.get_nsel(), self.repinit.get_nsel()],
        )
        # repinit
        g1_ext = self.type_embedding(extended_atype)
        g1_inp = g1_ext[:, :nloc, :]
        g1, _, _, _, _ = self.repinit(
            nlist_dict[
                get_multiple_nlist_key(self.repinit.get_rcut(), self.repinit.get_nsel())
            ],
            extended_coord,
            extended_atype,
            g1_ext,
            mapping,
        )
        # linear to change shape
        g1 = self.g1_shape_tranform(g1)
        # mapping g1
        assert mapping is not None
        mapping_ext = (
            mapping.view(nframes, nall).unsqueeze(-1).expand(-1, -1, g1.shape[-1])
        )
        g1_ext = torch.gather(g1, 1, mapping_ext)
        # repformer
        g1, g2, h2, rot_mat, sw = self.repformers(
            nlist_dict[
                get_multiple_nlist_key(
                    self.repformers.get_rcut(), self.repformers.get_nsel()
                )
            ],
            extended_coord,
            extended_atype,
            g1_ext,
            mapping,
        )
        if self.concat_output_tebd:
            g1 = torch.cat([g1, g1_inp], dim=-1)
        return g1, rot_mat, g2, h2, sw

    @classmethod
    def update_sel(cls, global_jdata: dict, local_jdata: dict):
        """Update the selection and perform neighbor statistics.

        Parameters
        ----------
        global_jdata : dict
            The global data, containing the training section
        local_jdata : dict
            The local data refer to the current class
        """
        local_jdata_cpy = local_jdata.copy()
        update_sel = UpdateSel()
        local_jdata_cpy = update_sel.update_one_sel(
            global_jdata,
            local_jdata_cpy,
            True,
            rcut_key="repinit_rcut",
            sel_key="repinit_nsel",
        )
        local_jdata_cpy = update_sel.update_one_sel(
            global_jdata,
            local_jdata_cpy,
            True,
            rcut_key="repformer_rcut",
            sel_key="repformer_nsel",
        )
        return local_jdata_cpy
