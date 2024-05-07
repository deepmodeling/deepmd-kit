# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np

from deepmd.dpmodel.utils.network import (
    Identity,
    NativeLayer,
)
from deepmd.dpmodel.utils.nlist import (
    build_multiple_neighbor_list,
    get_multiple_nlist_key,
)
from deepmd.dpmodel.utils.type_embed import (
    TypeEmbedNet,
)
from deepmd.dpmodel.utils.update_sel import (
    UpdateSel,
)
from deepmd.utils.path import (
    DPPath,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

try:
    from deepmd._version import version as __version__
except ImportError:
    __version__ = "unknown"

from typing import (
    List,
    Optional,
    Tuple,
)

from deepmd.dpmodel import (
    NativeOP,
)
from deepmd.dpmodel.utils import (
    EnvMat,
    NetworkCollection,
)

from .base_descriptor import (
    BaseDescriptor,
)
from .dpa1 import (
    DescrptBlockSeAtten,
)
from .repformers import (
    DescrptBlockRepformers,
    RepformerLayer,
)


@BaseDescriptor.register("dpa2")
class DescrptDPA2(NativeOP, BaseDescriptor):
    def __init__(
        self,
        # args for repinit
        ntypes: int,
        repinit_rcut: float,
        repinit_rcut_smth: float,
        repinit_nsel: int,
        repformer_rcut: float,
        repformer_rcut_smth: float,
        repformer_nsel: int,
        # kwargs for repinit
        repinit_neuron: List[int] = [25, 50, 100],
        repinit_axis_neuron: int = 16,
        repinit_tebd_dim: int = 8,
        repinit_tebd_input_mode: str = "concat",
        repinit_set_davg_zero: bool = True,
        repinit_activation_function="tanh",
        # kwargs for repformer
        repformer_nlayers: int = 3,
        repformer_g1_dim: int = 128,
        repformer_g2_dim: int = 16,
        repformer_axis_neuron: int = 4,
        repformer_direct_dist: bool = False,
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
        repformer_activation_function: str = "tanh",
        repformer_update_style: str = "res_avg",
        repformer_update_residual: float = 0.001,
        repformer_update_residual_init: str = "norm",
        repformer_set_davg_zero: bool = True,
        # kwargs for descriptor
        concat_output_tebd: bool = True,
        precision: str = "float64",
        smooth: bool = True,
        exclude_types: List[Tuple[int, int]] = [],
        env_protection: float = 0.0,
        trainable: bool = True,
        seed: Optional[int] = None,
        resnet_dt: bool = False,
        trainable_ln: bool = True,
        ln_eps: Optional[float] = 1e-5,
        type_one_side: bool = False,
        add_tebd_to_repinit_out: bool = False,
    ):
        r"""The DPA-2 descriptor. see https://arxiv.org/abs/2312.15492.

        Parameters
        ----------
        repinit_rcut : float
            (Used in the repinit block.)
            The cut-off radius.
        repinit_rcut_smth : float
            (Used in the repinit block.)
            Where to start smoothing. For example the 1/r term is smoothed from rcut to rcut_smth.
        repinit_nsel : int
            (Used in the repinit block.)
            Maximally possible number of selected neighbors.
        repinit_neuron : list, optional
            (Used in the repinit block.)
            Number of neurons in each hidden layers of the embedding net.
            When two layers are of the same size or one layer is twice as large as the previous layer,
            a skip connection is built.
        repinit_axis_neuron : int, optional
            (Used in the repinit block.)
            Size of the submatrix of G (embedding matrix).
        repinit_tebd_dim : int, optional
            (Used in the repinit block.)
            The dimension of atom type embedding.
        repinit_tebd_input_mode : str, optional
            (Used in the repinit block.)
            The input mode of the type embedding. Supported modes are ['concat', 'strip'].
        repinit_set_davg_zero : bool, optional
            (Used in the repinit block.)
            Set the normalization average to zero.
        repinit_activation_function : str, optional
            (Used in the repinit block.)
            The activation function in the embedding net.
        repformer_rcut : float
            (Used in the repformer block.)
            The cut-off radius.
        repformer_rcut_smth : float
            (Used in the repformer block.)
            Where to start smoothing. For example the 1/r term is smoothed from rcut to rcut_smth.
        repformer_nsel : int
            (Used in the repformer block.)
            Maximally possible number of selected neighbors.
        repformer_nlayers : int, optional
            (Used in the repformer block.)
            Number of repformer layers.
        repformer_g1_dim : int, optional
            (Used in the repformer block.)
            Dimension of the first graph convolution layer.
        repformer_g2_dim : int, optional
            (Used in the repformer block.)
            Dimension of the second graph convolution layer.
        repformer_axis_neuron : int, optional
            (Used in the repformer block.)
            Size of the submatrix of G (embedding matrix).
        repformer_direct_dist : bool, optional
            (Used in the repformer block.)
            Whether to use direct distance information (1/r term) in the repformer block.
        repformer_update_g1_has_conv : bool, optional
            (Used in the repformer block.)
            Whether to update the g1 rep with convolution term.
        repformer_update_g1_has_drrd : bool, optional
            (Used in the repformer block.)
            Whether to update the g1 rep with the drrd term.
        repformer_update_g1_has_grrg : bool, optional
            (Used in the repformer block.)
            Whether to update the g1 rep with the grrg term.
        repformer_update_g1_has_attn : bool, optional
            (Used in the repformer block.)
            Whether to update the g1 rep with the localized self-attention.
        repformer_update_g2_has_g1g1 : bool, optional
            (Used in the repformer block.)
            Whether to update the g2 rep with the g1xg1 term.
        repformer_update_g2_has_attn : bool, optional
            (Used in the repformer block.)
            Whether to update the g2 rep with the gated self-attention.
        repformer_update_h2 : bool, optional
            (Used in the repformer block.)
            Whether to update the h2 rep.
        repformer_attn1_hidden : int, optional
            (Used in the repformer block.)
            The hidden dimension of localized self-attention to update the g1 rep.
        repformer_attn1_nhead : int, optional
            (Used in the repformer block.)
            The number of heads in localized self-attention to update the g1 rep.
        repformer_attn2_hidden : int, optional
            (Used in the repformer block.)
            The hidden dimension of gated self-attention to update the g2 rep.
        repformer_attn2_nhead : int, optional
            (Used in the repformer block.)
            The number of heads in gated self-attention to update the g2 rep.
        repformer_attn2_has_gate : bool, optional
            (Used in the repformer block.)
            Whether to use gate in the gated self-attention to update the g2 rep.
        repformer_activation_function : str, optional
            (Used in the repformer block.)
            The activation function in the embedding net.
        repformer_update_style : str, optional
            (Used in the repformer block.)
            Style to update a representation.
            Supported options are:
            -'res_avg': Updates a rep `u` with: u = 1/\\sqrt{n+1} (u + u_1 + u_2 + ... + u_n)
            -'res_incr': Updates a rep `u` with: u = u + 1/\\sqrt{n} (u_1 + u_2 + ... + u_n)
            -'res_residual': Updates a rep `u` with: u = u + (r1*u_1 + r2*u_2 + ... + r3*u_n)
            where `r1`, `r2` ... `r3` are residual weights defined by `repformer_update_residual`
            and `repformer_update_residual_init`.
        repformer_update_residual : float, optional
            (Used in the repformer block.)
            When update using residual mode, the initial std of residual vector weights.
        repformer_update_residual_init : str, optional
            (Used in the repformer block.)
            When update using residual mode, the initialization mode of residual vector weights.
        repformer_set_davg_zero : bool, optional
            (Used in the repformer block.)
            Set the normalization average to zero.
        concat_output_tebd : bool, optional
            Whether to concat type embedding at the output of the descriptor.
        precision : str, optional
            The precision of the embedding net parameters.
        smooth : bool, optional
            Whether to use smoothness in processes such as attention weights calculation.
        exclude_types : List[List[int]], optional
            The excluded pairs of types which have no interaction with each other.
            For example, `[[0, 1]]` means no interaction between type 0 and type 1.
        env_protection : float, optional
            Protection parameter to prevent division by zero errors during environment matrix calculations.
            For example, when using paddings, there may be zero distances of neighbors, which may make division by zero error during environment matrix calculations without protection.
        trainable : bool, optional
            If the parameters are trainable.
        seed : int, optional
            (Unused yet) Random seed for parameter initialization.
        resnet_dt : bool, optional
            Whether to use a "Timestep" in the skip connection.
        trainable_ln : bool, optional
            Whether to use trainable shift and scale weights in layer normalization.
        ln_eps : float, optional
            The epsilon value for layer normalization.
        type_one_side : bool, optional
            Whether to use one-side type embedding.
        add_tebd_to_repinit_out : bool, optional
            Whether to add type embedding to the output representation from repinit before inputting it into repformer.

        Returns
        -------
        descriptor:         torch.Tensor
            the descriptor of shape nf x nloc x g1_dim.
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
        #  to keep consistent with default value in this backends
        if ln_eps is None:
            ln_eps = 1e-5
        self.repinit = DescrptBlockSeAtten(
            repinit_rcut,
            repinit_rcut_smth,
            repinit_nsel,
            ntypes,
            attn_layer=0,
            neuron=repinit_neuron,
            axis_neuron=repinit_axis_neuron,
            tebd_dim=repinit_tebd_dim,
            tebd_input_mode=repinit_tebd_input_mode,
            set_davg_zero=repinit_set_davg_zero,
            exclude_types=exclude_types,
            env_protection=env_protection,
            activation_function=repinit_activation_function,
            precision=precision,
            resnet_dt=resnet_dt,
            trainable_ln=trainable_ln,
            ln_eps=ln_eps,
            smooth=smooth,
            type_one_side=type_one_side,
        )
        self.repformers = DescrptBlockRepformers(
            repformer_rcut,
            repformer_rcut_smth,
            repformer_nsel,
            ntypes,
            nlayers=repformer_nlayers,
            g1_dim=repformer_g1_dim,
            g2_dim=repformer_g2_dim,
            axis_neuron=repformer_axis_neuron,
            direct_dist=repformer_direct_dist,
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
            activation_function=repformer_activation_function,
            update_style=repformer_update_style,
            update_residual=repformer_update_residual,
            update_residual_init=repformer_update_residual_init,
            set_davg_zero=repformer_set_davg_zero,
            smooth=smooth,
            exclude_types=exclude_types,
            env_protection=env_protection,
            precision=precision,
            resnet_dt=resnet_dt,
            trainable_ln=trainable_ln,
            ln_eps=ln_eps,
        )
        self.type_embedding = TypeEmbedNet(
            ntypes=ntypes,
            neuron=[repinit_tebd_dim],
            padding=True,
            activation_function="Linear",
            precision=precision,
        )
        self.concat_output_tebd = concat_output_tebd
        self.precision = precision
        self.smooth = smooth
        self.exclude_types = exclude_types
        self.env_protection = env_protection
        self.trainable = trainable
        self.resnet_dt = resnet_dt
        self.trainable_ln = trainable_ln
        self.ln_eps = ln_eps
        self.type_one_side = type_one_side
        self.add_tebd_to_repinit_out = add_tebd_to_repinit_out

        if self.repinit.dim_out == self.repformers.dim_in:
            self.g1_shape_tranform = Identity()
        else:
            self.g1_shape_tranform = NativeLayer(
                self.repinit.dim_out,
                self.repformers.dim_in,
                bias=False,
                precision=precision,
            )
        self.tebd_transform = None
        if self.add_tebd_to_repinit_out:
            self.tebd_transform = NativeLayer(
                repinit_tebd_dim,
                self.repformers.dim_in,
                bias=False,
                precision=precision,
            )
        assert self.repinit.rcut > self.repformers.rcut
        assert self.repinit.sel[0] > self.repformers.sel[0]

        self.tebd_dim = repinit_tebd_dim
        self.rcut = self.repinit.get_rcut()
        self.ntypes = ntypes
        self.sel = self.repinit.sel

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
        raise NotImplementedError

    @property
    def dim_out(self):
        return self.get_dim_out()

    @property
    def dim_emb(self):
        """Returns the embedding dimension g2."""
        return self.get_dim_emb()

    def compute_input_stats(self, merged: List[dict], path: Optional[DPPath] = None):
        """Update mean and stddev for descriptor elements."""
        raise NotImplementedError

    def call(
        self,
        coord_ext: np.ndarray,
        atype_ext: np.ndarray,
        nlist: np.ndarray,
        mapping: Optional[np.ndarray] = None,
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
            The index mapping, maps extended region index to local region.

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
        nall = coord_ext.reshape(nframes, -1).shape[1] // 3
        # nlists
        nlist_dict = build_multiple_neighbor_list(
            coord_ext,
            nlist,
            [self.repformers.get_rcut(), self.repinit.get_rcut()],
            [self.repformers.get_nsel(), self.repinit.get_nsel()],
        )
        # repinit
        g1_ext = self.type_embedding.call()[atype_ext]
        g1_inp = g1_ext[:, :nloc, :]
        g1, _, _, _, _ = self.repinit(
            nlist_dict[
                get_multiple_nlist_key(self.repinit.get_rcut(), self.repinit.get_nsel())
            ],
            coord_ext,
            atype_ext,
            g1_ext,
            mapping,
        )
        # linear to change shape
        g1 = self.g1_shape_tranform(g1)
        if self.add_tebd_to_repinit_out:
            assert self.tebd_transform is not None
            g1 = g1 + self.tebd_transform(g1_inp)
        # mapping g1
        assert mapping is not None
        mapping_ext = np.tile(mapping.reshape(nframes, nall, 1), (1, 1, g1.shape[-1]))
        g1_ext = np.take_along_axis(g1, mapping_ext, axis=1)
        # repformer
        g1, g2, h2, rot_mat, sw = self.repformers(
            nlist_dict[
                get_multiple_nlist_key(
                    self.repformers.get_rcut(), self.repformers.get_nsel()
                )
            ],
            coord_ext,
            atype_ext,
            g1_ext,
            mapping,
        )
        if self.concat_output_tebd:
            g1 = np.concatenate([g1, g1_inp], axis=-1)
        return g1, rot_mat, g2, h2, sw

    def serialize(self) -> dict:
        repinit = self.repinit
        repformers = self.repformers
        data = {
            "@class": "Descriptor",
            "type": "dpa2",
            "@version": 1,
            "ntypes": self.ntypes,
            "repinit_rcut": repinit.rcut,
            "repinit_rcut_smth": repinit.rcut_smth,
            "repinit_nsel": repinit.sel,
            "repformer_rcut": repformers.rcut,
            "repformer_rcut_smth": repformers.rcut_smth,
            "repformer_nsel": repformers.sel,
            "repinit_neuron": repinit.neuron,
            "repinit_axis_neuron": repinit.axis_neuron,
            "repinit_tebd_dim": repinit.tebd_dim,
            "repinit_tebd_input_mode": repinit.tebd_input_mode,
            "repinit_set_davg_zero": repinit.set_davg_zero,
            "repinit_activation_function": repinit.activation_function,
            "repformer_nlayers": repformers.nlayers,
            "repformer_g1_dim": repformers.g1_dim,
            "repformer_g2_dim": repformers.g2_dim,
            "repformer_axis_neuron": repformers.axis_neuron,
            "repformer_direct_dist": repformers.direct_dist,
            "repformer_update_g1_has_conv": repformers.update_g1_has_conv,
            "repformer_update_g1_has_drrd": repformers.update_g1_has_drrd,
            "repformer_update_g1_has_grrg": repformers.update_g1_has_grrg,
            "repformer_update_g1_has_attn": repformers.update_g1_has_attn,
            "repformer_update_g2_has_g1g1": repformers.update_g2_has_g1g1,
            "repformer_update_g2_has_attn": repformers.update_g2_has_attn,
            "repformer_update_h2": repformers.update_h2,
            "repformer_attn1_hidden": repformers.attn1_hidden,
            "repformer_attn1_nhead": repformers.attn1_nhead,
            "repformer_attn2_hidden": repformers.attn2_hidden,
            "repformer_attn2_nhead": repformers.attn2_nhead,
            "repformer_attn2_has_gate": repformers.attn2_has_gate,
            "repformer_activation_function": repformers.activation_function,
            "repformer_update_style": repformers.update_style,
            "repformer_set_davg_zero": repformers.set_davg_zero,
            "concat_output_tebd": self.concat_output_tebd,
            "precision": self.precision,
            "smooth": self.smooth,
            "exclude_types": self.exclude_types,
            "env_protection": self.env_protection,
            "trainable": self.trainable,
            "resnet_dt": self.resnet_dt,
            "trainable_ln": self.trainable_ln,
            "ln_eps": self.ln_eps,
            "type_one_side": self.type_one_side,
            "add_tebd_to_repinit_out": self.add_tebd_to_repinit_out,
            "type_embedding": self.type_embedding.serialize(),
            "g1_shape_tranform": self.g1_shape_tranform.serialize(),
        }
        if self.add_tebd_to_repinit_out:
            data.update(
                {
                    "tebd_transform": self.tebd_transform.serialize(),
                }
            )
        repinit_variable = {
            "embeddings": repinit.embeddings.serialize(),
            "env_mat": EnvMat(repinit.rcut, repinit.rcut_smth).serialize(),
            "@variables": {
                "davg": repinit["davg"],
                "dstd": repinit["dstd"],
            },
        }
        if repinit.tebd_input_mode in ["strip"]:
            repinit_variable.update(
                {"embeddings_strip": repinit.embeddings_strip.serialize()}
            )
        repformers_variable = {
            "g2_embd": repformers.g2_embd.serialize(),
            "repformer_layers": [layer.serialize() for layer in repformers.layers],
            "env_mat": EnvMat(repformers.rcut, repformers.rcut_smth).serialize(),
            "@variables": {
                "davg": repformers["davg"],
                "dstd": repformers["dstd"],
            },
        }
        data.update(
            {
                "repinit": repinit_variable,
                "repformers": repformers_variable,
            }
        )
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptDPA2":
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
        data.pop("type")
        repinit_variable = data.pop("repinit").copy()
        repformers_variable = data.pop("repformers").copy()
        type_embedding = data.pop("type_embedding")
        g1_shape_tranform = data.pop("g1_shape_tranform")
        tebd_transform = data.pop("tebd_transform", None)
        add_tebd_to_repinit_out = data["add_tebd_to_repinit_out"]
        obj = cls(**data)
        obj.type_embedding = TypeEmbedNet.deserialize(type_embedding)
        if add_tebd_to_repinit_out:
            assert isinstance(tebd_transform, dict)
            obj.tebd_transform = NativeLayer.deserialize(tebd_transform)
        if obj.repinit.dim_out != obj.repformers.dim_in:
            obj.g1_shape_tranform = NativeLayer.deserialize(g1_shape_tranform)

        # deserialize repinit
        statistic_repinit = repinit_variable.pop("@variables")
        env_mat = repinit_variable.pop("env_mat")
        tebd_input_mode = data["repinit_tebd_input_mode"]
        obj.repinit.embeddings = NetworkCollection.deserialize(
            repinit_variable.pop("embeddings")
        )
        if tebd_input_mode in ["strip"]:
            obj.repinit.embeddings_strip = NetworkCollection.deserialize(
                repinit_variable.pop("embeddings_strip")
            )
        obj.repinit["davg"] = statistic_repinit["davg"]
        obj.repinit["dstd"] = statistic_repinit["dstd"]

        # deserialize repformers
        statistic_repformers = repformers_variable.pop("@variables")
        env_mat = repformers_variable.pop("env_mat")
        repformer_layers = repformers_variable.pop("repformer_layers")
        obj.repformers.g2_embd = NativeLayer.deserialize(
            repformers_variable.pop("g2_embd")
        )
        obj.repformers["davg"] = statistic_repformers["davg"]
        obj.repformers["dstd"] = statistic_repformers["dstd"]
        obj.repformers.layers = [
            RepformerLayer.deserialize(layer) for layer in repformer_layers
        ]
        return obj

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
