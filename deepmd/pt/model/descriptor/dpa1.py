# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Callable,
    List,
    Optional,
    Union,
)

import torch

from deepmd.pt.model.network.network import (
    TypeEmbedNet,
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
from .se_atten import (
    DescrptBlockSeAtten,
)


@BaseDescriptor.register("dpa1")
@BaseDescriptor.register("se_atten")
class DescrptDPA1(BaseDescriptor, torch.nn.Module):
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
        activation_function="tanh",
        scaling_factor=1.0,
        head_num=1,
        normalize=True,
        temperature=None,
        return_rot=False,
        concat_output_tebd: bool = True,
        type: Optional[str] = None,
        # not implemented
        resnet_dt: bool = False,
        type_one_side: bool = True,
        precision: str = "default",
        trainable: bool = True,
        exclude_types: Optional[List[List[int]]] = None,
        stripped_type_embedding: bool = False,
        smooth_type_embdding: bool = False,
    ):
        super().__init__()
        if resnet_dt:
            raise NotImplementedError("resnet_dt is not supported.")
        if not type_one_side:
            raise NotImplementedError("type_one_side is not supported.")
        if precision != "default" and precision != "float64":
            raise NotImplementedError("precison is not supported.")
        if exclude_types is not None and exclude_types != []:
            raise NotImplementedError("exclude_types is not supported.")
        if stripped_type_embedding:
            raise NotImplementedError("stripped_type_embedding is not supported.")
        if smooth_type_embdding:
            raise NotImplementedError("smooth_type_embdding is not supported.")
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
            activation_function=activation_function,
            scaling_factor=scaling_factor,
            head_num=head_num,
            normalize=normalize,
            temperature=temperature,
            return_rot=return_rot,
        )
        self.type_embedding = TypeEmbedNet(ntypes, tebd_dim)
        self.tebd_dim = tebd_dim
        self.concat_output_tebd = concat_output_tebd
        # set trainable
        for param in self.parameters():
            param.requires_grad = trainable

    def get_rcut(self) -> float:
        """Returns the cut-off radius."""
        return self.se_atten.get_rcut()

    def get_nsel(self) -> int:
        """Returns the number of selected atoms in the cut-off radius."""
        return self.se_atten.get_nsel()

    def get_sel(self) -> List[int]:
        """Returns the number of selected atoms for each type."""
        return self.se_atten.get_sel()

    def get_ntypes(self) -> int:
        """Returns the number of element types."""
        return self.se_atten.get_ntypes()

    def get_dim_out(self) -> int:
        """Returns the output dimension."""
        ret = self.se_atten.get_dim_out()
        if self.concat_output_tebd:
            ret += self.tebd_dim
        return ret

    def get_dim_emb(self) -> int:
        return self.se_atten.dim_emb

    def mixed_types(self) -> bool:
        """If true, the discriptor
        1. assumes total number of atoms aligned across frames;
        2. requires a neighbor list that does not distinguish different atomic types.

        If false, the discriptor
        1. assumes total number of atoms of each atom type aligned across frames;
        2. requires a neighbor list that distinguishes different atomic types.

        """
        return self.se_atten.mixed_types()

    def share_params(self, base_class, shared_level, resume=False):
        """
        Share the parameters of self to the base_class with shared_level during multitask training.
        If not start from checkpoint (resume is False),
        some seperated parameters (e.g. mean and stddev) will be re-calculated across different classes.
        """
        assert (
            self.__class__ == base_class.__class__
        ), "Only descriptors of the same type can share params!"
        # For DPA1 descriptors, the user-defined share-level
        # shared_level: 0
        # share all parameters in both type_embedding and se_atten
        if shared_level == 0:
            self._modules["type_embedding"] = base_class._modules["type_embedding"]
            self.se_atten.share_params(base_class.se_atten, 0, resume=resume)
        # shared_level: 1
        # share all parameters in type_embedding
        elif shared_level == 1:
            self._modules["type_embedding"] = base_class._modules["type_embedding"]
        # Other shared levels
        else:
            raise NotImplementedError

    @property
    def dim_out(self):
        return self.get_dim_out()

    @property
    def dim_emb(self):
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
        return self.se_atten.compute_input_stats(merged, path)

    def serialize(self) -> dict:
        """Serialize the obj to dict."""
        raise NotImplementedError

    @classmethod
    def deserialize(cls) -> "DescrptDPA1":
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
            The index mapping, not required by this descriptor.

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
        return UpdateSel().update_one_sel(global_jdata, local_jdata_cpy, True)
