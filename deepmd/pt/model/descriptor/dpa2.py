# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import torch

from deepmd.dpmodel.utils import EnvMat as DPEnvMat
from deepmd.pt.model.network.mlp import (
    Identity,
    MLPLayer,
    NetworkCollection,
)
from deepmd.pt.model.network.network import (
    TypeEmbedNet,
    TypeEmbedNetConsistent,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.nlist import (
    build_multiple_neighbor_list,
    get_multiple_nlist_key,
)
from deepmd.pt.utils.update_sel import (
    UpdateSel,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
)
from deepmd.utils.path import (
    DPPath,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .base_descriptor import (
    BaseDescriptor,
)
from .repformer_layer import (
    RepformerLayer,
)
from .repformers import (
    DescrptBlockRepformers,
)
from .se_atten import (
    DescrptBlockSeAtten,
)


@BaseDescriptor.register("dpa2")
class DescrptDPA2(BaseDescriptor, torch.nn.Module):
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
        repinit_resnet_dt: bool = False,
        repinit_type_one_side: bool = False,
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
        repformer_trainable_ln: bool = True,
        repformer_ln_eps: Optional[float] = 1e-5,
        # kwargs for descriptor
        concat_output_tebd: bool = True,
        precision: str = "float64",
        smooth: bool = True,
        exclude_types: List[Tuple[int, int]] = [],
        env_protection: float = 0.0,
        trainable: bool = True,
        seed: Optional[int] = None,
        add_tebd_to_repinit_out: bool = False,
        old_impl: bool = False,
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
        repinit_resnet_dt : bool, optional
            (Used in the repinit block.)
            Whether to use a "Timestep" in the skip connection.
        repinit_type_one_side : bool, optional
            (Used in the repinit block.)
            Whether to use one-side type embedding.
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
        repformer_trainable_ln : bool, optional
            (Used in the repformer block.)
            Whether to use trainable shift and scale weights in layer normalization.
        repformer_ln_eps : float, optional
            (Used in the repformer block.)
            The epsilon value for layer normalization.
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
        add_tebd_to_repinit_out : bool, optional
            Whether to add type embedding to the output representation from repinit before inputting it into repformer.

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
        #  to keep consistent with default value in this backends
        if repformer_ln_eps is None:
            repformer_ln_eps = 1e-5
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
            resnet_dt=repinit_resnet_dt,
            smooth=smooth,
            type_one_side=repinit_type_one_side,
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
            trainable_ln=repformer_trainable_ln,
            ln_eps=repformer_ln_eps,
            old_impl=old_impl,
        )
        self.type_embedding = TypeEmbedNet(
            ntypes, repinit_tebd_dim, precision=precision
        )
        self.concat_output_tebd = concat_output_tebd
        self.precision = precision
        self.smooth = smooth
        self.exclude_types = exclude_types
        self.env_protection = env_protection
        self.trainable = trainable
        self.add_tebd_to_repinit_out = add_tebd_to_repinit_out

        if self.repinit.dim_out == self.repformers.dim_in:
            self.g1_shape_tranform = Identity()
        else:
            self.g1_shape_tranform = MLPLayer(
                self.repinit.dim_out,
                self.repformers.dim_in,
                bias=False,
                precision=precision,
                init="glorot",
            )
        self.tebd_transform = None
        if self.add_tebd_to_repinit_out:
            self.tebd_transform = MLPLayer(
                repinit_tebd_dim,
                self.repformers.dim_in,
                bias=False,
                precision=precision,
            )
        assert self.repinit.rcut > self.repformers.rcut
        assert self.repinit.sel[0] > self.repformers.sel[0]

        self.tebd_dim = repinit_tebd_dim
        self.rcut = self.repinit.get_rcut()
        self.rcut_smth = self.repinit.get_rcut_smth()
        self.ntypes = ntypes
        self.sel = self.repinit.sel
        # set trainable
        for param in self.parameters():
            param.requires_grad = trainable

    def get_rcut(self) -> float:
        """Returns the cut-off radius."""
        return self.rcut

    def get_rcut_smth(self) -> float:
        """Returns the radius where the neighbor information starts to smoothly decay to 0."""
        return self.rcut_smth

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

    def get_env_protection(self) -> float:
        """Returns the protection of building environment matrix."""
        # the env_protection of repinit is the same as that of the repformer
        return self.repinit.get_env_protection()

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
            "repinit_resnet_dt": repinit.resnet_dt,
            "repinit_type_one_side": repinit.type_one_side,
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
            "repformer_trainable_ln": repformers.trainable_ln,
            "repformer_ln_eps": repformers.ln_eps,
            "concat_output_tebd": self.concat_output_tebd,
            "precision": self.precision,
            "smooth": self.smooth,
            "exclude_types": self.exclude_types,
            "env_protection": self.env_protection,
            "trainable": self.trainable,
            "add_tebd_to_repinit_out": self.add_tebd_to_repinit_out,
            "type_embedding": self.type_embedding.embedding.serialize(),
            "g1_shape_tranform": self.g1_shape_tranform.serialize(),
        }
        if self.add_tebd_to_repinit_out:
            data.update(
                {
                    "tebd_transform": self.tebd_transform.serialize(),
                }
            )
        repinit_variable = {
            "embeddings": repinit.filter_layers.serialize(),
            "env_mat": DPEnvMat(repinit.rcut, repinit.rcut_smth).serialize(),
            "@variables": {
                "davg": to_numpy_array(repinit["davg"]),
                "dstd": to_numpy_array(repinit["dstd"]),
            },
        }
        if repinit.tebd_input_mode in ["strip"]:
            repinit_variable.update(
                {"embeddings_strip": repinit.filter_layers_strip.serialize()}
            )
        repformers_variable = {
            "g2_embd": repformers.g2_embd.serialize(),
            "repformer_layers": [layer.serialize() for layer in repformers.layers],
            "env_mat": DPEnvMat(repformers.rcut, repformers.rcut_smth).serialize(),
            "@variables": {
                "davg": to_numpy_array(repformers["davg"]),
                "dstd": to_numpy_array(repformers["dstd"]),
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
        obj.type_embedding.embedding = TypeEmbedNetConsistent.deserialize(
            type_embedding
        )
        if add_tebd_to_repinit_out:
            assert isinstance(tebd_transform, dict)
            obj.tebd_transform = MLPLayer.deserialize(tebd_transform)
        if obj.repinit.dim_out != obj.repformers.dim_in:
            obj.g1_shape_tranform = MLPLayer.deserialize(g1_shape_tranform)

        def t_cvt(xx):
            return torch.tensor(xx, dtype=obj.repinit.prec, device=env.DEVICE)

        # deserialize repinit
        statistic_repinit = repinit_variable.pop("@variables")
        env_mat = repinit_variable.pop("env_mat")
        tebd_input_mode = data["repinit_tebd_input_mode"]
        obj.repinit.filter_layers = NetworkCollection.deserialize(
            repinit_variable.pop("embeddings")
        )
        if tebd_input_mode in ["strip"]:
            obj.repinit.filter_layers_strip = NetworkCollection.deserialize(
                repinit_variable.pop("embeddings_strip")
            )
        obj.repinit["davg"] = t_cvt(statistic_repinit["davg"])
        obj.repinit["dstd"] = t_cvt(statistic_repinit["dstd"])

        # deserialize repformers
        statistic_repformers = repformers_variable.pop("@variables")
        env_mat = repformers_variable.pop("env_mat")
        repformer_layers = repformers_variable.pop("repformer_layers")
        obj.repformers.g2_embd = MLPLayer.deserialize(
            repformers_variable.pop("g2_embd")
        )
        obj.repformers["davg"] = t_cvt(statistic_repformers["davg"])
        obj.repformers["dstd"] = t_cvt(statistic_repformers["dstd"])
        obj.repformers.layers = torch.nn.ModuleList(
            [RepformerLayer.deserialize(layer) for layer in repformer_layers]
        )
        return obj

    def forward(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
        comm_dict: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Compute the descriptor.

        Parameters
        ----------
        extended_coord
            The extended coordinates of atoms. shape: nf x (nallx3)
        extended_atype
            The extended aotm types. shape: nf x nall
        nlist
            The neighbor list. shape: nf x nloc x nnei
        mapping
            The index mapping, mapps extended region index to local region.
        comm_dict
            The data needed for communication for parallel inference.

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
        if self.add_tebd_to_repinit_out:
            assert self.tebd_transform is not None
            g1 = g1 + self.tebd_transform(g1_inp)
        # mapping g1
        if comm_dict is None:
            assert mapping is not None
            mapping_ext = (
                mapping.view(nframes, nall).unsqueeze(-1).expand(-1, -1, g1.shape[-1])
            )
            g1_ext = torch.gather(g1, 1, mapping_ext)
            g1 = g1_ext
        # repformer
        g1, g2, h2, rot_mat, sw = self.repformers(
            nlist_dict[
                get_multiple_nlist_key(
                    self.repformers.get_rcut(), self.repformers.get_nsel()
                )
            ],
            extended_coord,
            extended_atype,
            g1,
            mapping,
            comm_dict,
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
