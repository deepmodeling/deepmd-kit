# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Callable,
    NoReturn,
    Optional,
    Union,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel import (
    NativeOP,
)
from deepmd.dpmodel.array_api import (
    xp_take_along_axis,
)
from deepmd.dpmodel.common import (
    cast_precision,
    to_numpy_array,
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
from .se_t_tebd import (
    DescrptBlockSeTTebd,
)


class RepinitArgs:
    def __init__(
        self,
        rcut: float,
        rcut_smth: float,
        nsel: int,
        neuron: list[int] = [25, 50, 100],
        axis_neuron: int = 16,
        tebd_dim: int = 8,
        tebd_input_mode: str = "concat",
        set_davg_zero: bool = True,
        activation_function="tanh",
        resnet_dt: bool = False,
        type_one_side: bool = False,
        use_three_body: bool = False,
        three_body_neuron: list[int] = [2, 4, 8],
        three_body_sel: int = 40,
        three_body_rcut: float = 4.0,
        three_body_rcut_smth: float = 0.5,
    ) -> None:
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
        use_three_body : bool, optional
            Whether to concatenate three-body representation in the output descriptor.
        three_body_neuron : list, optional
            Number of neurons in each hidden layers of the three-body embedding net.
            When two layers are of the same size or one layer is twice as large as the previous layer,
            a skip connection is built.
        three_body_sel : int, optional
            Maximally possible number of selected neighbors in the three-body representation.
        three_body_rcut : float, optional
            The cut-off radius in the three-body representation.
        three_body_rcut_smth : float, optional
            Where to start smoothing in the three-body representation.
            For example the 1/r term is smoothed from three_body_rcut to three_body_rcut_smth.
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
        self.use_three_body = use_three_body
        self.three_body_neuron = three_body_neuron
        self.three_body_sel = three_body_sel
        self.three_body_rcut = three_body_rcut
        self.three_body_rcut_smth = three_body_rcut_smth

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
            "use_three_body": self.use_three_body,
            "three_body_neuron": self.three_body_neuron,
            "three_body_sel": self.three_body_sel,
            "three_body_rcut": self.three_body_rcut,
            "three_body_rcut_smth": self.three_body_rcut_smth,
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
        use_sqrt_nnei: bool = True,
        g1_out_conv: bool = True,
        g1_out_mlp: bool = True,
        ln_eps: Optional[float] = 1e-5,
    ) -> None:
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
        use_sqrt_nnei : bool, optional
            Whether to use the square root of the number of neighbors for symmetrization_op normalization instead of using the number of neighbors directly.
        g1_out_conv : bool, optional
            Whether to put the convolutional update of g1 separately outside the concatenated MLP update.
        g1_out_mlp : bool, optional
            Whether to put the self MLP update of g1 separately outside the concatenated MLP update.
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
        self.use_sqrt_nnei = use_sqrt_nnei
        self.g1_out_conv = g1_out_conv
        self.g1_out_mlp = g1_out_mlp
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
            "use_sqrt_nnei": self.use_sqrt_nnei,
            "g1_out_conv": self.g1_out_conv,
            "g1_out_mlp": self.g1_out_mlp,
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
        exclude_types: list[tuple[int, int]] = [],
        env_protection: float = 0.0,
        trainable: bool = True,
        seed: Optional[Union[int, list[int]]] = None,
        add_tebd_to_repinit_out: bool = False,
        use_econf_tebd: bool = False,
        use_tebd_bias: bool = False,
        type_map: Optional[list[str]] = None,
    ) -> None:
        r"""The DPA-2 descriptor[1]_.

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
        exclude_types : list[list[int]], optional
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
        use_tebd_bias : bool, Optional
            Whether to use bias in the type embedding layer.
        type_map : list[str], Optional
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

        References
        ----------
        .. [1] Zhang, D., Liu, X., Zhang, X. et al. DPA-2: a
           large atomic model as a multi-task learner. npj
           Comput Mater 10, 293 (2024). https://doi.org/10.1038/s41524-024-01493-2
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
        self.use_three_body = self.repinit_args.use_three_body
        if self.use_three_body:
            self.repinit_three_body = DescrptBlockSeTTebd(
                self.repinit_args.three_body_rcut,
                self.repinit_args.three_body_rcut_smth,
                self.repinit_args.three_body_sel,
                ntypes,
                neuron=self.repinit_args.three_body_neuron,
                tebd_dim=self.repinit_args.tebd_dim,
                tebd_input_mode=self.repinit_args.tebd_input_mode,
                set_davg_zero=self.repinit_args.set_davg_zero,
                exclude_types=exclude_types,
                env_protection=env_protection,
                activation_function=self.repinit_args.activation_function,
                precision=precision,
                resnet_dt=self.repinit_args.resnet_dt,
                smooth=smooth,
                seed=child_seed(seed, 5),
            )
        else:
            self.repinit_three_body = None
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
            use_sqrt_nnei=self.repformer_args.use_sqrt_nnei,
            g1_out_conv=self.repformer_args.g1_out_conv,
            g1_out_mlp=self.repformer_args.g1_out_mlp,
            ln_eps=self.repformer_args.ln_eps,
            seed=child_seed(seed, 1),
        )
        self.rcsl_list = [
            (self.repformers.get_rcut(), self.repformers.get_nsel()),
            (self.repinit.get_rcut(), self.repinit.get_nsel()),
        ]
        if self.use_three_body:
            self.rcsl_list.append(
                (self.repinit_three_body.get_rcut(), self.repinit_three_body.get_nsel())
            )
        self.rcsl_list.sort()
        for ii in range(1, len(self.rcsl_list)):
            assert self.rcsl_list[ii - 1][1] <= self.rcsl_list[ii][1], (
                "rcut and sel are not in the same order"
            )
        self.rcut_list = [ii[0] for ii in self.rcsl_list]
        self.nsel_list = [ii[1] for ii in self.rcsl_list]
        self.use_econf_tebd = use_econf_tebd
        self.use_tebd_bias = use_tebd_bias
        self.type_map = type_map
        self.type_embedding = TypeEmbedNet(
            ntypes=ntypes,
            neuron=[self.repinit_args.tebd_dim],
            padding=True,
            activation_function="Linear",
            precision=precision,
            use_econf_tebd=use_econf_tebd,
            use_tebd_bias=use_tebd_bias,
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

        self.repinit_out_dim = self.repinit.dim_out
        if self.repinit_args.use_three_body:
            assert self.repinit_three_body is not None
            self.repinit_out_dim += self.repinit_three_body.dim_out

        if self.repinit_out_dim == self.repformers.dim_in:
            self.g1_shape_tranform = Identity()
        else:
            self.g1_shape_tranform = NativeLayer(
                self.repinit_out_dim,
                self.repformers.dim_in,
                bias=False,
                precision=precision,
                seed=child_seed(seed, 3),
            )
        self.tebd_transform = None
        if self.add_tebd_to_repinit_out:
            self.tebd_transform = NativeLayer(
                self.repinit_args.tebd_dim,
                self.repformers.dim_in,
                bias=False,
                precision=precision,
                seed=child_seed(seed, 4),
            )
        assert self.repinit.rcut > self.repformers.rcut
        assert self.repinit.sel[0] > self.repformers.sel[0]

        self.tebd_dim = self.repinit_args.tebd_dim
        self.rcut = self.repinit.get_rcut()
        self.ntypes = ntypes
        self.sel = self.repinit.sel
        self.precision = precision

    def get_rcut(self) -> float:
        """Returns the cut-off radius."""
        return self.rcut

    def get_rcut_smth(self) -> float:
        """Returns the radius where the neighbor information starts to smoothly decay to 0."""
        return self.rcut_smth

    def get_nsel(self) -> int:
        """Returns the number of selected atoms in the cut-off radius."""
        return sum(self.sel)

    def get_sel(self) -> list[int]:
        """Returns the number of selected atoms for each type."""
        return self.sel

    def get_ntypes(self) -> int:
        """Returns the number of element types."""
        return self.ntypes

    def get_type_map(self) -> list[str]:
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
        """If true, the descriptor
        1. assumes total number of atoms aligned across frames;
        2. requires a neighbor list that does not distinguish different atomic types.

        If false, the descriptor
        1. assumes total number of atoms of each atom type aligned across frames;
        2. requires a neighbor list that distinguishes different atomic types.

        """
        return True

    def has_message_passing(self) -> bool:
        """Returns whether the descriptor has message passing."""
        return any(
            [self.repinit.has_message_passing(), self.repformers.has_message_passing()]
        )

    def need_sorted_nlist_for_lower(self) -> bool:
        """Returns whether the descriptor needs sorted nlist when using `forward_lower`."""
        return True

    def get_env_protection(self) -> float:
        """Returns the protection of building environment matrix."""
        return self.env_protection

    def share_params(self, base_class, shared_level, resume=False) -> NoReturn:
        """
        Share the parameters of self to the base_class with shared_level during multitask training.
        If not start from checkpoint (resume is False),
        some separated parameters (e.g. mean and stddev) will be re-calculated across different classes.
        """
        raise NotImplementedError

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat=None
    ) -> None:
        """Change the type related params to new ones, according to `type_map` and the original one in the model.
        If there are new types in `type_map`, statistics will be updated accordingly to `model_with_new_type_stat` for these new types.
        """
        assert self.type_map is not None, (
            "'type_map' must be defined when performing type changing!"
        )
        remap_index, has_new_type = get_index_between_two_maps(self.type_map, type_map)
        self.type_map = type_map
        self.type_embedding.change_type_map(type_map=type_map)
        self.exclude_types = map_pair_exclude_types(self.exclude_types, remap_index)
        self.ntypes = len(type_map)
        repinit = self.repinit
        repformers = self.repformers
        repinit_three_body = self.repinit_three_body
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
            if self.use_three_body:
                extend_descrpt_stat(
                    repinit_three_body,
                    type_map,
                    des_with_stat=model_with_new_type_stat.repinit_three_body
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
        if self.use_three_body:
            repinit_three_body.ntypes = self.ntypes
            repinit_three_body.reinit_exclude(self.exclude_types)
            repinit_three_body["davg"] = repinit_three_body["davg"][remap_index]
            repinit_three_body["dstd"] = repinit_three_body["dstd"][remap_index]

    @property
    def dim_out(self):
        return self.get_dim_out()

    @property
    def dim_emb(self):
        """Returns the embedding dimension g2."""
        return self.get_dim_emb()

    def compute_input_stats(
        self,
        merged: Union[Callable[[], list[dict]], list[dict]],
        path: Optional[DPPath] = None,
    ):
        """
        Compute the input statistics (e.g. mean and stddev) for the descriptors from packed data.

        Parameters
        ----------
        merged : Union[Callable[[], list[dict]], list[dict]]
            - list[dict]: A list of data samples from various data systems.
                Each element, `merged[i]`, is a data dictionary containing `keys`: `torch.Tensor`
                originating from the `i`-th data system.
            - Callable[[], list[dict]]: A lazy function that returns data samples in the above format
                only when needed. Since the sampling process can be slow and memory-intensive,
                the lazy function helps by only sampling once.
        path : Optional[DPPath]
            The path to the stat file.

        """
        descrpt_list = [self.repinit, self.repformers]
        if self.use_three_body:
            descrpt_list.append(self.repinit_three_body)
        for ii, descrpt in enumerate(descrpt_list):
            descrpt.compute_input_stats(merged, path)

    def set_stat_mean_and_stddev(
        self,
        mean: list[np.ndarray],
        stddev: list[np.ndarray],
    ) -> None:
        """Update mean and stddev for descriptor."""
        descrpt_list = [self.repinit, self.repformers]
        if self.use_three_body:
            descrpt_list.append(self.repinit_three_body)
        for ii, descrpt in enumerate(descrpt_list):
            descrpt.mean = mean[ii]
            descrpt.stddev = stddev[ii]

    def get_stat_mean_and_stddev(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Get mean and stddev for descriptor."""
        mean_list = [self.repinit.mean, self.repformers.mean]
        stddev_list = [
            self.repinit.stddev,
            self.repformers.stddev,
        ]
        if self.use_three_body:
            mean_list.append(self.repinit_three_body.mean)
            stddev_list.append(self.repinit_three_body.stddev)
        return mean_list, stddev_list

    @cast_precision
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
        xp = array_api_compat.array_namespace(coord_ext, atype_ext, nlist)
        use_three_body = self.use_three_body
        nframes, nloc, nnei = nlist.shape
        nall = xp.reshape(coord_ext, (nframes, -1)).shape[1] // 3
        # nlists
        nlist_dict = build_multiple_neighbor_list(
            coord_ext,
            nlist,
            self.rcut_list,
            self.nsel_list,
        )
        type_embedding = self.type_embedding.call()
        # repinit
        g1_ext = xp.reshape(
            xp.take(type_embedding, xp.reshape(atype_ext, [-1]), axis=0),
            (nframes, nall, self.tebd_dim),
        )
        g1_inp = g1_ext[:, :nloc, :]
        g1, _, _, _, _ = self.repinit(
            nlist_dict[
                get_multiple_nlist_key(self.repinit.get_rcut(), self.repinit.get_nsel())
            ],
            coord_ext,
            atype_ext,
            g1_ext,
            mapping,
            type_embedding=type_embedding,
        )
        if use_three_body:
            assert self.repinit_three_body is not None
            g1_three_body, __, __, __, __ = self.repinit_three_body(
                nlist_dict[
                    get_multiple_nlist_key(
                        self.repinit_three_body.get_rcut(),
                        self.repinit_three_body.get_nsel(),
                    )
                ],
                coord_ext,
                atype_ext,
                g1_ext,
                mapping,
                type_embedding=type_embedding,
            )
            g1 = xp.concat([g1, g1_three_body], axis=-1)
        # linear to change shape
        g1 = self.g1_shape_tranform(g1)
        if self.add_tebd_to_repinit_out:
            assert self.tebd_transform is not None
            g1 = g1 + self.tebd_transform(g1_inp)
        # mapping g1
        assert mapping is not None
        mapping_ext = xp.tile(
            xp.reshape(mapping, (nframes, nall, 1)), (1, 1, g1.shape[-1])
        )
        g1_ext = xp_take_along_axis(g1, mapping_ext, axis=1)
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
            g1 = xp.concat([g1, g1_inp], axis=-1)
        return g1, rot_mat, g2, h2, sw

    def serialize(self) -> dict:
        repinit = self.repinit
        repformers = self.repformers
        repinit_three_body = self.repinit_three_body
        data = {
            "@class": "Descriptor",
            "type": "dpa2",
            "@version": 3,
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
            "use_tebd_bias": self.use_tebd_bias,
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
                "davg": to_numpy_array(repinit["davg"]),
                "dstd": to_numpy_array(repinit["dstd"]),
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
                "davg": to_numpy_array(repformers["davg"]),
                "dstd": to_numpy_array(repformers["dstd"]),
            },
        }
        data.update(
            {
                "repinit_variable": repinit_variable,
                "repformers_variable": repformers_variable,
            }
        )
        if self.use_three_body:
            repinit_three_body_variable = {
                "embeddings": repinit_three_body.embeddings.serialize(),
                "env_mat": EnvMat(
                    repinit_three_body.rcut, repinit_three_body.rcut_smth
                ).serialize(),
                "@variables": {
                    "davg": to_numpy_array(repinit_three_body["davg"]),
                    "dstd": to_numpy_array(repinit_three_body["dstd"]),
                },
            }
            if repinit_three_body.tebd_input_mode in ["strip"]:
                repinit_three_body_variable.update(
                    {
                        "embeddings_strip": repinit_three_body.embeddings_strip.serialize()
                    }
                )
            data.update(
                {
                    "repinit_three_body_variable": repinit_three_body_variable,
                }
            )
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptDPA2":
        data = data.copy()
        version = data.pop("@version")
        check_version_compatibility(version, 3, 1)
        data.pop("@class")
        data.pop("type")
        repinit_variable = data.pop("repinit_variable").copy()
        repformers_variable = data.pop("repformers_variable").copy()
        repinit_three_body_variable = (
            data.pop("repinit_three_body_variable").copy()
            if "repinit_three_body_variable" in data
            else None
        )
        type_embedding = data.pop("type_embedding")
        g1_shape_tranform = data.pop("g1_shape_tranform")
        tebd_transform = data.pop("tebd_transform", None)
        add_tebd_to_repinit_out = data["add_tebd_to_repinit_out"]
        if version < 3:
            # compat with old version
            data["repformer_args"]["use_sqrt_nnei"] = False
            data["repformer_args"]["g1_out_conv"] = False
            data["repformer_args"]["g1_out_mlp"] = False
        data["repinit"] = RepinitArgs(**data.pop("repinit_args"))
        data["repformer"] = RepformerArgs(**data.pop("repformer_args"))
        # compat with version 1
        if "use_tebd_bias" not in data:
            data["use_tebd_bias"] = True
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

        if data["repinit"].use_three_body:
            # deserialize repinit_three_body
            statistic_repinit_three_body = repinit_three_body_variable.pop("@variables")
            env_mat = repinit_three_body_variable.pop("env_mat")
            tebd_input_mode = data["repinit"].tebd_input_mode
            obj.repinit_three_body.embeddings = NetworkCollection.deserialize(
                repinit_three_body_variable.pop("embeddings")
            )
            if tebd_input_mode in ["strip"]:
                obj.repinit_three_body.embeddings_strip = NetworkCollection.deserialize(
                    repinit_three_body_variable.pop("embeddings_strip")
                )
            obj.repinit_three_body["davg"] = statistic_repinit_three_body["davg"]
            obj.repinit_three_body["dstd"] = statistic_repinit_three_body["dstd"]

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
        type_map: Optional[list[str]],
        local_jdata: dict,
    ) -> tuple[dict, Optional[float]]:
        """Update the selection and perform neighbor statistics.

        Parameters
        ----------
        train_data : DeepmdDataSystem
            data used to do neighbor statistics
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
        min_nbor_dist, repinit_three_body_sel = update_sel.update_one_sel(
            train_data,
            type_map,
            local_jdata_cpy["repinit"]["three_body_rcut"],
            local_jdata_cpy["repinit"]["three_body_sel"],
            True,
        )
        local_jdata_cpy["repinit"]["three_body_sel"] = repinit_three_body_sel[0]
        min_nbor_dist, repformer_sel = update_sel.update_one_sel(
            train_data,
            type_map,
            local_jdata_cpy["repformer"]["rcut"],
            local_jdata_cpy["repformer"]["nsel"],
            True,
        )
        local_jdata_cpy["repformer"]["nsel"] = repformer_sel[0]
        return local_jdata_cpy, min_nbor_dist
