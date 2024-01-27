# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    List,
    Optional,
)

import torch

from deepmd.pt.model.descriptor import (
    Descriptor,
)
from deepmd.pt.model.network.network import (
    TypeEmbedNet,
)

from .se_atten import (
    DescrptBlockSeAtten,
)


@Descriptor.register("dpa1")
@Descriptor.register("se_atten")
class DescrptDPA1(Descriptor):
    def __init__(
        self,
        rcut,
        rcut_smth,
        sel,
        ntypes: int,
        neuron: list = [25, 50, 100],
        axis_neuron: int = 16,
        tebd_dim: int = 8,
        tebd_input_mode: str = "concat",
        # set_davg_zero: bool = False,
        set_davg_zero: bool = True,  # TODO
        attn: int = 128,
        attn_layer: int = 2,
        attn_dotr: bool = True,
        attn_mask: bool = False,
        post_ln=True,
        ffn=False,
        ffn_embed_dim=1024,
        activation="tanh",
        scaling_factor=1.0,
        head_num=1,
        normalize=True,
        temperature=None,
        return_rot=False,
        concat_output_tebd: bool = True,
        type: Optional[str] = None,
    ):
        super().__init__()
        del type
        self.se_atten = DescrptBlockSeAtten(
            rcut,
            rcut_smth,
            sel,
            ntypes,
            neuron=neuron,
            axis_neuron=axis_neuron,
            tebd_dim=tebd_dim,
            tebd_input_mode=tebd_input_mode,
            set_davg_zero=set_davg_zero,
            attn=attn,
            attn_layer=attn_layer,
            attn_dotr=attn_dotr,
            attn_mask=attn_mask,
            post_ln=post_ln,
            ffn=ffn,
            ffn_embed_dim=ffn_embed_dim,
            activation=activation,
            scaling_factor=scaling_factor,
            head_num=head_num,
            normalize=normalize,
            temperature=temperature,
            return_rot=return_rot,
        )
        self.type_embedding = TypeEmbedNet(ntypes, tebd_dim)
        self.tebd_dim = tebd_dim
        self.concat_output_tebd = concat_output_tebd

    def get_rcut(self) -> float:
        """Returns the cut-off radius."""
        return self.se_atten.get_rcut()

    def get_nsel(self) -> int:
        """Returns the number of selected atoms in the cut-off radius."""
        return self.se_atten.get_nsel()

    def get_sel(self) -> List[int]:
        """Returns the number of selected atoms for each type."""
        return self.se_atten.get_sel()

    def get_ntype(self) -> int:
        """Returns the number of element types."""
        return self.se_atten.get_ntype()

    def get_dim_out(self) -> int:
        """Returns the output dimension."""
        ret = self.se_atten.get_dim_out()
        if self.concat_output_tebd:
            ret += self.tebd_dim
        return ret

    @property
    def dim_out(self):
        return self.get_dim_out()

    @property
    def dim_emb(self):
        return self.se_atten.dim_emb

    def compute_input_stats(self, merged):
        return self.se_atten.compute_input_stats(merged)

    def init_desc_stat(self, sumr, suma, sumn, sumr2, suma2):
        self.se_atten.init_desc_stat(sumr, suma, sumn, sumr2, suma2)

    @classmethod
    def get_stat_name(cls, config):
        descrpt_type = config["type"]
        assert descrpt_type in ["dpa1", "se_atten"]
        return f'stat_file_dpa1_rcut{config["rcut"]:.2f}_smth{config["rcut_smth"]:.2f}_sel{config["sel"]}.npz'

    @classmethod
    def get_data_process_key(cls, config):
        descrpt_type = config["type"]
        assert descrpt_type in ["dpa1", "se_atten"]
        return {"sel": config["sel"], "rcut": config["rcut"]}

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
            The index mapping, not required by this descriptor.

        Returns
        -------
        descriptor
            The descriptor. shape: nf x nloc x (ng x axis_neuron)
        gr
            The rotationally equivariant and permutationally invariant single particle
            representation. shape: nf x nloc x ng x 3
        g2
            The rotationally invariant pair-partical channel.
            shape: nf x nloc x nnei x ng
        h2
            The rotationally equivariant pair-partical channel.
            shape: nf x nloc x nnei x 3
        sw
            The smooth switch function. shape: nf x nloc x nnei

        """
        del mapping
        nframes, nloc, nnei = nlist.shape
        nall = extended_coord.view(nframes, -1).shape[1] // 3
        g1_ext = self.type_embedding(extended_atype)
        g1_inp = g1_ext[:, :nloc, :]
        g1, g2, h2, rot_mat, sw = self.se_atten(
            nlist,
            extended_coord,
            extended_atype,
            g1_ext,
            mapping=None,
        )
        if self.concat_output_tebd:
            g1 = torch.cat([g1, g1_inp], dim=-1)

        return g1, rot_mat, g2, h2, sw
