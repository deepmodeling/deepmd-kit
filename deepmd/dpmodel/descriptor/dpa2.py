# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np

from deepmd.dpmodel import (
    NativeOP,
)
from deepmd.dpmodel.utils import (
    EnvMat,
    NetworkCollection,
)
from deepmd.dpmodel.utils.network import (
    Identity,
    NativeLayer,
)
from deepmd.dpmodel.utils.nlist import (
    build_multiple_neighbor_list,
    get_multiple_nlist_key,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.dpmodel.utils.type_embed import (
    TypeEmbedNet,
)
from deepmd.dpmodel.utils.update_sel import (
    UpdateSel,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.utils.finetune import (
    get_index_between_two_maps,
    map_pair_exclude_types,
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
from .descriptor import (
    extend_descrpt_stat,
)
from .dpa1 import (
    DescrptBlockSeAtten,
)
from .repformers import (
    DescrptBlockRepformers,
    RepformerLayer,
)


class RepinitArgs:
    def __init__(
        self,
        rcut: float,
        rcut_smth: float,
        nsel: int,
        neuron: List[int] = [25, 50, 100],
        axis_neuron: int = 16,
        tebd_dim: int = 8,
        tebd_input_mode: str = "concat",
        set_davg_zero: bool = True,
        activation_function="tanh",
        resnet_dt: bool = False,
        type_one_side: bool = False,
    ):
        r"""The constructor for the RepinitArgs class which defines the parameters of the repinit block in DPA2 descriptor.

        Parameters
        ----------
        rcut : float
            The cut-off radius.
        rcut_smth : float
            Where to start smoothing. For example the 1/r term is smoothed from rcut to rcut_smth.
        nsel : int
            Maximally possible number of selected neighbors.
        neuron : list, optional
            Number of neurons in each hidden layers of the embedding net.
            When two layers are of the same size or one layer is twice as large as the previous layer,
            a skip connection is built.
        axis_neuron : int, optional
            Size of the submatrix of G (embedding matrix).
        tebd_dim : int, optional
            The dimension of atom type embedding.
        tebd_input_mode : str, optional
            The input mode of the type embedding. Supported modes are ['concat', 'strip'].
        set_davg_zero : bool, optional
            Set the normalization average to zero.
        activation_function : str, optional
            The activation function in the embedding net.
        resnet_dt : bool, optional
            Whether to use a "Timestep" in the skip connection.
        type_one_side : bool, optional
            Whether to use one-side type embedding.
        """
        self.rcut = rcut
        self.rcut_smth = rcut_smth
        self.nsel = nsel
        self.neuron = neuron
        self.axis_neuron = axis_neuron
        self.tebd_dim = tebd_dim
        self.tebd_input_mode = tebd_input_mode
        self.set_davg_zero = set_davg_zero
        self.activation_function = activation_function
        self.resnet_dt = resnet_dt
        self.type_one_side = type_one_side

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(key)

    def serialize(self) -> dict:
        return {
            "rcut": self.rcut,
            "rcut_smth": self.rcut_smth,
            "nsel": self.nsel,
            "neuron": self.neuron,
            "axis_neuron": self.axis_neuron,
            "tebd_dim": self.tebd_dim,
            "tebd_input_mode": self.tebd_input_mode,
            "set_davg_zero": self.set_davg_zero,
            "activation_function": self.activation_function,
            "resnet_dt": self.resnet_dt,
            "type_one_side": self.type_one_side,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "RepinitArgs":
        return cls(**data)


class RepformerArgs:
    def __init__(
        self,
        rcut: float,
        rcut_smth: float,
        nsel: int,
        nlayers: int = 3,
        g1_dim: int = 128,
        g2_dim: int = 16,
        axis_neuron: int = 4,
        direct_dist: bool = False,
        update_g1_has_conv: bool = True,
        update_g1_has_drrd: bool = True,
        update_g1_has_grrg: bool = True,
        update_g1_has_attn: bool = True,
        update_g2_has_g1g1: bool = True,
        update_g2_has_attn: bool = True,
        update_h2: bool = False,
        attn1_hidden: int = 64,
        attn1_nhead: int = 4,
        attn2_hidden: int = 16,
        attn2_nhead: int = 4,
        attn2_has_gate: bool = False,
        activation_function: str = "tanh",
        update_style: str = "res_avg",
        update_residual: float = 0.001,
        update_residual_init: str = "norm",
        set_davg_zero: bool = True,
        trainable_ln: bool = True,
        ln_eps: Optional[float] = 1e-5,
    ):
        r"""The constructor for the RepformerArgs class which defines the parameters of the repformer block in DPA2 descriptor.

        Parameters
        ----------
        rcut : float
            The cut-off radius.
        rcut_smth : float
            Where to start smoothing. For example the 1/r term is smoothed from rcut to rcut_smth.
        nsel : int
            Maximally possible number of selected neighbors.
        nlayers : int, optional
            Number of repformer layers.
        g1_dim : int, optional
            Dimension of the first graph convolution layer.
        g2_dim : int, optional
            Dimension of the second graph convolution layer.
        axis_neuron : int, optional
            Size of the submatrix of G (embedding matrix).
        direct_dist : bool, optional
            Whether to use direct distance information (1/r term) in the repformer block.
        update_g1_has_conv : bool, optional
            Whether to update the g1 rep with convolution term.
        update_g1_has_drrd : bool, optional
            Whether to update the g1 rep with the drrd term.
        update_g1_has_grrg : bool, optional
            Whether to update the g1 rep with the grrg term.
        update_g1_has_attn : bool, optional
            Whether to update the g1 rep with the localized self-attention.
        update_g2_has_g1g1 : bool, optional
            Whether to update the g2 rep with the g1xg1 term.
        update_g2_has_attn : bool, optional
            Whether to update the g2 rep with the gated self-attention.
        update_h2 : bool, optional
            Whether to update the h2 rep.
        attn1_hidden : int, optional
            The hidden dimension of localized self-attention to update the g1 rep.
        attn1_nhead : int, optional
            The number of heads in localized self-attention to update the g1 rep.
        attn2_hidden : int, optional
            The hidden dimension of gated self-attention to update the g2 rep.
        attn2_nhead : int, optional
            The number of heads in gated self-attention to update the g2 rep.
        attn2_has_gate : bool, optional
            Whether to use gate in the gated self-attention to update the g2 rep.
        activation_function : str, optional
            The activation function in the embedding net.
        update_style : str, optional
            Style to update a representation.
            Supported options are:
            -'res_avg': Updates a rep `u` with: u = 1/\\sqrt{n+1} (u + u_1 + u_2 + ... + u_n)
            -'res_incr': Updates a rep `u` with: u = u + 1/\\sqrt{n} (u_1 + u_2 + ... + u_n)
            -'res_residual': Updates a rep `u` with: u = u + (r1*u_1 + r2*u_2 + ... + r3*u_n)
            where `r1`, `r2` ... `r3` are residual weights defined by `update_residual`
            and `update_residual_init`.
        update_residual : float, optional
            When update using residual mode, the initial std of residual vector weights.
        update_residual_init : str, optional
            When update using residual mode, the initialization mode of residual vector weights.
        set_davg_zero : bool, optional
            Set the normalization average to zero.
        trainable_ln : bool, optional
            Whether to use trainable shift and scale weights in layer normalization.
        ln_eps : float, optional
            The epsilon value for layer normalization.
        """
        self.rcut = rcut
        self.rcut_smth = rcut_smth
        self.nsel = nsel
        self.nlayers = nlayers
        self.g1_dim = g1_dim
        self.g2_dim = g2_dim
        self.axis_neuron = axis_neuron
        self.direct_dist = direct_dist
        self.update_g1_has_conv = update_g1_has_conv
        self.update_g1_has_drrd = update_g1_has_drrd
        self.update_g1_has_grrg = update_g1_has_grrg
        self.update_g1_has_attn = update_g1_has_attn
        self.update_g2_has_g1g1 = update_g2_has_g1g1
        self.update_g2_has_attn = update_g2_has_attn
        self.update_h2 = update_h2
        self.attn1_hidden = attn1_hidden
        self.attn1_nhead = attn1_nhead
        self.attn2_hidden = attn2_hidden
        self.attn2_nhead = attn2_nhead
        self.attn2_has_gate = attn2_has_gate
        self.activation_function = activation_function
        self.update_style = update_style
        self.update_residual = update_residual
        self.update_residual_init = update_residual_init
        self.set_davg_zero = set_davg_zero
        self.trainable_ln = trainable_ln
        #  to keep consistent with default value in this backends
        if ln_eps is None:
            ln_eps = 1e-5
        self.ln_eps = ln_eps

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(key)

    def serialize(self) -> dict:
        return {
            "rcut": self.rcut,
            "rcut_smth": self.rcut_smth,
            "nsel": self.nsel,
            "nlayers": self.nlayers,
            "g1_dim": self.g1_dim,
            "g2_dim": self.g2_dim,
            "axis_neuron": self.axis_neuron,
            "direct_dist": self.direct_dist,
            "update_g1_has_conv": self.update_g1_has_conv,
            "update_g1_has_drrd": self.update_g1_has_drrd,
            "update_g1_has_grrg": self.update_g1_has_grrg,
            "update_g1_has_attn": self.update_g1_has_attn,
            "update_g2_has_g1g1": self.update_g2_has_g1g1,
            "update_g2_has_attn": self.update_g2_has_attn,
            "update_h2": self.update_h2,
            "attn1_hidden": self.attn1_hidden,
            "attn1_nhead": self.attn1_nhead,
            "attn2_hidden": self.attn2_hidden,
            "attn2_nhead": self.attn2_nhead,
            "attn2_has_gate": self.attn2_has_gate,
            "activation_function": self.activation_function,
            "update_style": self.update_style,
            "update_residual": self.update_residual,
            "update_residual_init": self.update_residual_init,
            "set_davg_zero": self.set_davg_zero,
            "trainable_ln": self.trainable_ln,
            "ln_eps": self.ln_eps,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "RepformerArgs":
        return cls(**data)


@BaseDescriptor.register("dpa2")
class DescrptDPA2(NativeOP, BaseDescriptor):
    def __init__(
        self,
        ntypes: int,
        # args for repinit
        repinit: Union[RepinitArgs, dict],
        # args for repformer
        repformer: Union[RepformerArgs, dict],
        # kwargs for descriptor
        concat_output_tebd: bool = True,
        precision: str = "float64",
        smooth: bool = True,
        exclude_types: List[Tuple[int, int]] = [],
        env_protection: float = 0.0,
        trainable: bool = True,
        seed: Optional[Union[int, List[int]]] = None,
        add_tebd_to_repinit_out: bool = False,
        use_econf_tebd: bool = False,
        type_map: Optional[List[str]] = None,
    ):
        r"""The DPA-2 descriptor. see https://arxiv.org/abs/2312.15492.

        Parameters
        ----------
        repinit : Union[RepinitArgs, dict]
            The arguments used to initialize the repinit block, see docstr in `RepinitArgs` for details information.
        repformer : Union[RepformerArgs, dict]
            The arguments used to initialize the repformer block, see docstr in `RepformerArgs` for details information.
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
        use_econf_tebd : bool, Optional
            Whether to use electronic configuration type embedding.
        type_map : List[str], Optional
            A list of strings. Give the name to each type of atoms.

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

        def init_subclass_params(sub_data, sub_class):
            if isinstance(sub_data, dict):
                return sub_class(**sub_data)
            elif isinstance(sub_data, sub_class):
                return sub_data
            else:
                raise ValueError(
                    f"Input args must be a {sub_class.__name__} class or a dict!"
                )

        self.repinit_args = init_subclass_params(repinit, RepinitArgs)
        self.repformer_args = init_subclass_params(repformer, RepformerArgs)

        self.repinit = DescrptBlockSeAtten(
            self.repinit_args.rcut,
            self.repinit_args.rcut_smth,
            self.repinit_args.nsel,
            ntypes,
            attn_layer=0,
            neuron=self.repinit_args.neuron,
            axis_neuron=self.repinit_args.axis_neuron,
            tebd_dim=self.repinit_args.tebd_dim,
            tebd_input_mode=self.repinit_args.tebd_input_mode,
            set_davg_zero=self.repinit_args.set_davg_zero,
            exclude_types=exclude_types,
            env_protection=env_protection,
            activation_function=self.repinit_args.activation_function,
            precision=precision,
            resnet_dt=self.repinit_args.resnet_dt,
            smooth=smooth,
            type_one_side=self.repinit_args.type_one_side,
            seed=child_seed(seed, 0),
        )
        self.repformers = DescrptBlockRepformers(
            self.repformer_args.rcut,
            self.repformer_args.rcut_smth,
            self.repformer_args.nsel,
            ntypes,
            nlayers=self.repformer_args.nlayers,
            g1_dim=self.repformer_args.g1_dim,
            g2_dim=self.repformer_args.g2_dim,
            axis_neuron=self.repformer_args.axis_neuron,
            direct_dist=self.repformer_args.direct_dist,
            update_g1_has_conv=self.repformer_args.update_g1_has_conv,
            update_g1_has_drrd=self.repformer_args.update_g1_has_drrd,
            update_g1_has_grrg=self.repformer_args.update_g1_has_grrg,
            update_g1_has_attn=self.repformer_args.update_g1_has_attn,
            update_g2_has_g1g1=self.repformer_args.update_g2_has_g1g1,
            update_g2_has_attn=self.repformer_args.update_g2_has_attn,
            update_h2=self.repformer_args.update_h2,
            attn1_hidden=self.repformer_args.attn1_hidden,
            attn1_nhead=self.repformer_args.attn1_nhead,
            attn2_hidden=self.repformer_args.attn2_hidden,
            attn2_nhead=self.repformer_args.attn2_nhead,
            attn2_has_gate=self.repformer_args.attn2_has_gate,
            activation_function=self.repformer_args.activation_function,
            update_style=self.repformer_args.update_style,
            update_residual=self.repformer_args.update_residual,
            update_residual_init=self.repformer_args.update_residual_init,
            set_davg_zero=self.repformer_args.set_davg_zero,
            smooth=smooth,
            exclude_types=exclude_types,
            env_protection=env_protection,
            precision=precision,
            trainable_ln=self.repformer_args.trainable_ln,
            ln_eps=self.repformer_args.ln_eps,
            seed=child_seed(seed, 1),
        )
        self.use_econf_tebd = use_econf_tebd
        self.type_map = type_map
        self.type_embedding = TypeEmbedNet(
            ntypes=ntypes,
            neuron=[self.repinit_args.tebd_dim],
            padding=True,
            activation_function="Linear",
            precision=precision,
            use_econf_tebd=use_econf_tebd,
            type_map=type_map,
            seed=child_seed(seed, 2),
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
            self.g1_shape_tranform = NativeLayer(
                self.repinit.dim_out,
                self.repformers.dim_in,
                bias=False,
                precision=precision,
            )
        self.tebd_transform = None
        if self.add_tebd_to_repinit_out:
            self.tebd_transform = NativeLayer(
                self.repinit_args.tebd_dim,
                self.repformers.dim_in,
                bias=False,
                precision=precision,
            )
        assert self.repinit.rcut > self.repformers.rcut
        assert self.repinit.sel[0] > self.repformers.sel[0]

        self.tebd_dim = self.repinit_args.tebd_dim
        self.rcut = self.repinit.get_rcut()
        self.ntypes = ntypes
        self.sel = self.repinit.sel

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

    def get_type_map(self) -> List[str]:
        """Get the name to each type of atoms."""
        return self.type_map

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

    def has_message_passing(self) -> bool:
        """Returns whether the descriptor has message passing."""
        return any(
            [self.repinit.has_message_passing(), self.repformers.has_message_passing()]
        )

    def get_env_protection(self) -> float:
        """Returns the protection of building environment matrix."""
        return self.env_protection

    def share_params(self, base_class, shared_level, resume=False):
        """
        Share the parameters of self to the base_class with shared_level during multitask training.
        If not start from checkpoint (resume is False),
        some seperated parameters (e.g. mean and stddev) will be re-calculated across different classes.
        """
        raise NotImplementedError

    def change_type_map(
        self, type_map: List[str], model_with_new_type_stat=None
    ) -> None:
        """Change the type related params to new ones, according to `type_map` and the original one in the model.
        If there are new types in `type_map`, statistics will be updated accordingly to `model_with_new_type_stat` for these new types.
        """
        assert (
            self.type_map is not None
        ), "'type_map' must be defined when performing type changing!"
        remap_index, has_new_type = get_index_between_two_maps(self.type_map, type_map)
        self.type_map = type_map
        self.type_embedding.change_type_map(type_map=type_map)
        self.exclude_types = map_pair_exclude_types(self.exclude_types, remap_index)
        self.ntypes = len(type_map)
        repinit = self.repinit
        repformers = self.repformers
        if has_new_type:
            # the avg and std of new types need to be updated
            extend_descrpt_stat(
                repinit,
                type_map,
                des_with_stat=model_with_new_type_stat.repinit
                if model_with_new_type_stat is not None
                else None,
            )
            extend_descrpt_stat(
                repformers,
                type_map,
                des_with_stat=model_with_new_type_stat.repformers
                if model_with_new_type_stat is not None
                else None,
            )
        repinit.ntypes = self.ntypes
        repformers.ntypes = self.ntypes
        repinit.reinit_exclude(self.exclude_types)
        repformers.reinit_exclude(self.exclude_types)
        repinit["davg"] = repinit["davg"][remap_index]
        repinit["dstd"] = repinit["dstd"][remap_index]
        repformers["davg"] = repformers["davg"][remap_index]
        repformers["dstd"] = repformers["dstd"][remap_index]

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

    def set_stat_mean_and_stddev(
        self,
        mean: List[np.ndarray],
        stddev: List[np.ndarray],
    ) -> None:
        """Update mean and stddev for descriptor."""
        for ii, descrpt in enumerate([self.repinit, self.repformers]):
            descrpt.mean = mean[ii]
            descrpt.stddev = stddev[ii]

    def get_stat_mean_and_stddev(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Get mean and stddev for descriptor."""
        return [self.repinit.mean, self.repformers.mean], [
            self.repinit.stddev,
            self.repformers.stddev,
        ]

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
            "repinit_args": self.repinit_args.serialize(),
            "repformer_args": self.repformer_args.serialize(),
            "concat_output_tebd": self.concat_output_tebd,
            "precision": self.precision,
            "smooth": self.smooth,
            "exclude_types": self.exclude_types,
            "env_protection": self.env_protection,
            "trainable": self.trainable,
            "add_tebd_to_repinit_out": self.add_tebd_to_repinit_out,
            "use_econf_tebd": self.use_econf_tebd,
            "type_map": self.type_map,
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
                "repinit_variable": repinit_variable,
                "repformers_variable": repformers_variable,
            }
        )
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptDPA2":
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
        data.pop("type")
        repinit_variable = data.pop("repinit_variable").copy()
        repformers_variable = data.pop("repformers_variable").copy()
        type_embedding = data.pop("type_embedding")
        g1_shape_tranform = data.pop("g1_shape_tranform")
        tebd_transform = data.pop("tebd_transform", None)
        add_tebd_to_repinit_out = data["add_tebd_to_repinit_out"]
        data["repinit"] = RepinitArgs(**data.pop("repinit_args"))
        data["repformer"] = RepformerArgs(**data.pop("repformer_args"))
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
        tebd_input_mode = data["repinit"].tebd_input_mode
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
    def update_sel(
        cls,
        train_data: DeepmdDataSystem,
        type_map: Optional[List[str]],
        local_jdata: dict,
    ) -> Tuple[dict, Optional[float]]:
        """Update the selection and perform neighbor statistics.

        Parameters
        ----------
        train_data : DeepmdDataSystem
            data used to do neighbor statictics
        type_map : list[str], optional
            The name of each type of atoms
        local_jdata : dict
            The local data refer to the current class

        Returns
        -------
        dict
            The updated local data
        float
            The minimum distance between two atoms
        """
        local_jdata_cpy = local_jdata.copy()
        update_sel = UpdateSel()
        min_nbor_dist, repinit_sel = update_sel.update_one_sel(
            train_data,
            type_map,
            local_jdata_cpy["repinit"]["rcut"],
            local_jdata_cpy["repinit"]["nsel"],
            True,
        )
        local_jdata_cpy["repinit"]["nsel"] = repinit_sel[0]
        min_nbor_dist, repformer_sel = update_sel.update_one_sel(
            train_data,
            type_map,
            local_jdata_cpy["repformer"]["rcut"],
            local_jdata_cpy["repformer"]["nsel"],
            True,
        )
        local_jdata_cpy["repformer"]["nsel"] = repformer_sel[0]
        return local_jdata_cpy, min_nbor_dist
