# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    List,
    Optional,
)

import torch

from deepmd.model_format import EnvMat as DPEnvMat
from deepmd.pt.model.descriptor import (
    Descriptor,
)
from deepmd.pt.model.network.mlp import (
    EmbdLayer,
    NetworkCollection,
)
from deepmd.pt.model.network.network import (
    TypeEmbedNet,
)
from deepmd.pt.utils import (
    env,
)

from .se_atten import (
    DescrptBlockSeAtten,
    NeighborGatedAttention,
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
        activation_function="tanh",
        precision: str = "float64",
        resnet_dt: bool = False,
        scaling_factor=1.0,
        normalize=True,
        temperature=None,
        concat_output_tebd: bool = True,
        type: Optional[str] = None,
        old_impl: bool = False,
        **kwargs,
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
            activation_function=activation_function,
            precision=precision,
            resnet_dt=resnet_dt,
            scaling_factor=scaling_factor,
            normalize=normalize,
            temperature=temperature,
            old_impl=old_impl,
            **kwargs,
        )
        self.type_embedding_old = None
        self.type_embedding = None
        self.old_impl = old_impl
        if self.old_impl:
            self.type_embedding_old = TypeEmbedNet(ntypes, tebd_dim)
        else:
            self.type_embedding = EmbdLayer(
                ntypes, tebd_dim, padding=True, precision=precision
            )
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
        if self.old_impl:
            assert self.type_embedding_old is not None
            g1_ext = self.type_embedding_old(extended_atype)
        else:
            assert self.type_embedding is not None
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

    def set_stat_mean_and_stddev(
        self,
        mean: torch.Tensor,
        stddev: torch.Tensor,
    ) -> None:
        self.se_atten.mean = mean
        self.se_atten.stddev = stddev

    def serialize(self) -> dict:
        obj = self.se_atten
        return {
            "rcut": obj.rcut,
            "rcut_smth": obj.rcut_smth,
            "sel": obj.sel,
            "ntypes": obj.ntypes,
            "neuron": obj.neuron,
            "axis_neuron": obj.axis_neuron,
            "tebd_dim": obj.tebd_dim,
            "tebd_input_mode": obj.tebd_input_mode,
            "set_davg_zero": obj.set_davg_zero,
            "attn": obj.attn_dim,
            "attn_layer": obj.attn_layer,
            "attn_dotr": obj.attn_dotr,
            "attn_mask": obj.attn_mask,
            "activation_function": obj.activation_function,
            "precision": obj.precision,
            "resnet_dt": obj.resnet_dt,
            "scaling_factor": obj.scaling_factor,
            "normalize": obj.normalize,
            "temperature": obj.temperature,
            "concat_output_tebd": self.concat_output_tebd,
            "embeddings": obj.filter_layers.serialize(),
            "attention_layers": obj.dpa1_attention.serialize(),
            "env_mat": DPEnvMat(obj.rcut, obj.rcut_smth).serialize(),
            "type_embedding": self.type_embedding.serialize(),
            "@variables": {
                "davg": obj["davg"].detach().cpu().numpy(),
                "dstd": obj["dstd"].detach().cpu().numpy(),
            },
            ## to be updated when the options are supported.
            "trainable": True,
            "type_one_side": True,
            "exclude_types": [],
            "spin": None,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptDPA1":
        variables = data.pop("@variables")
        embeddings = data.pop("embeddings")
        type_embedding = data.pop("type_embedding")
        attention_layers = data.pop("attention_layers")
        env_mat = data.pop("env_mat")
        obj = cls(**data)

        def t_cvt(xx):
            return torch.tensor(xx, dtype=obj.se_atten.prec, device=env.DEVICE)

        obj.type_embedding = EmbdLayer.deserialize(type_embedding)
        obj.se_atten["davg"] = t_cvt(variables["davg"])
        obj.se_atten["dstd"] = t_cvt(variables["dstd"])
        obj.se_atten.filter_layers = NetworkCollection.deserialize(embeddings)
        obj.se_atten.dpa1_attention = NeighborGatedAttention.deserialize(
            attention_layers
        )
        return obj
