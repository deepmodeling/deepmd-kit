# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
    Union,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel import (
    NativeOP,
)
from deepmd.dpmodel.common import (
    cast_precision,
    to_numpy_array,
)
from deepmd.dpmodel.utils import (
    EnvMat,
)
from deepmd.dpmodel.utils.network import (
    NativeLayer,
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
from .repflows import (
    DescrptBlockRepflows,
    RepFlowLayer,
)


class RepFlowArgs:
    r"""The constructor for the RepFlowArgs class which defines the parameters of the repflow block in DPA3 descriptor.

    Parameters
    ----------
    n_dim : int, optional
        The dimension of node representation.
    e_dim : int, optional
        The dimension of edge representation.
    a_dim : int, optional
        The dimension of angle representation.
    nlayers : int, optional
        Number of repflow layers.
    e_rcut : float, optional
        The edge cut-off radius.
    e_rcut_smth : float, optional
        Where to start smoothing for edge. For example the 1/r term is smoothed from rcut to rcut_smth.
    e_sel : int, optional
        Maximally possible number of selected edge neighbors.
    a_rcut : float, optional
        The angle cut-off radius.
    a_rcut_smth : float, optional
        Where to start smoothing for angle. For example the 1/r term is smoothed from rcut to rcut_smth.
    a_sel : int, optional
        Maximally possible number of selected angle neighbors.
    a_compress_rate : int, optional
        The compression rate for angular messages. The default value is 0, indicating no compression.
        If a non-zero integer c is provided, the node and edge dimensions will be compressed
        to a_dim/c and a_dim/2c, respectively, within the angular message.
    a_compress_e_rate : int, optional
        The extra compression rate for edge in angular message compression. The default value is 1.
        When using angular message compression with a_compress_rate c and a_compress_e_rate c_e,
        the edge dimension will be compressed to (c_e * a_dim / 2c) within the angular message.
    a_compress_use_split : bool, optional
        Whether to split first sub-vectors instead of linear mapping during angular message compression.
        The default value is False.
    n_multi_edge_message : int, optional
        The head number of multiple edge messages to update node feature.
        Default is 1, indicating one head edge message.
    axis_neuron : int, optional
        The number of dimension of submatrix in the symmetrization ops.
    update_angle : bool, optional
        Where to update the angle rep. If not, only node and edge rep will be used.
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
    fix_stat_std : float, optional
        If non-zero (default is 0.3), use this constant as the normalization standard deviation
        instead of computing it from data statistics.
    skip_stat : bool, optional
        (Deprecated, kept only for compatibility.) This parameter is obsolete and will be removed.
        If set to True, it forces fix_stat_std=0.3 for backward compatibility.
        Transition to fix_stat_std parameter immediately.
    optim_update : bool, optional
        Whether to enable the optimized update method.
        Uses a more efficient process when enabled. Defaults to True
    smooth_edge_update : bool, optional
        Whether to make edge update smooth.
        If True, the edge update from angle message will not use self as padding.
    edge_init_use_dist : bool, optional
        Whether to use direct distance r to initialize the edge features instead of 1/r.
        Note that when using this option, the activation function will not be used when initializing edge features.
    use_exp_switch : bool, optional
        Whether to use an exponential switch function instead of a polynomial one in the neighbor update.
        The exponential switch function ensures neighbor contributions smoothly diminish as the interatomic distance
        `r` approaches the cutoff radius `rcut`. Specifically, the function is defined as:
        s(r) = \\exp(-\\exp(20 * (r - rcut_smth) / rcut_smth)) for 0 < r \\leq rcut, and s(r) = 0 for r > rcut.
        Here, `rcut_smth` is an adjustable smoothing factor and `rcut_smth` should be chosen carefully
        according to `rcut`, ensuring s(r) approaches zero smoothly at the cutoff.
        Typical recommended values are `rcut_smth` = 5.3 for `rcut` = 6.0, and 3.5 for `rcut` = 4.0.
    use_dynamic_sel : bool, optional
        Whether to dynamically select neighbors within the cutoff radius.
        If True, the exact number of neighbors within the cutoff radius is used
        without padding to a fixed selection numbers.
        When enabled, users can safely set larger values for `e_sel` or `a_sel` (e.g., 1200 or 300, respectively)
        to guarantee capturing all neighbors within the cutoff radius.
        Note that when using dynamic selection, the `smooth_edge_update` must be True.
    sel_reduce_factor : float, optional
        Reduction factor applied to neighbor-scale normalization when `use_dynamic_sel` is True.
        In the dynamic selection case, neighbor-scale normalization will use `e_sel / sel_reduce_factor`
        or `a_sel / sel_reduce_factor` instead of the raw `e_sel` or `a_sel` values,
        accommodating larger selection numbers.
    """

    def __init__(
        self,
        n_dim: int = 128,
        e_dim: int = 64,
        a_dim: int = 64,
        nlayers: int = 6,
        e_rcut: float = 6.0,
        e_rcut_smth: float = 5.0,
        e_sel: int = 120,
        a_rcut: float = 4.0,
        a_rcut_smth: float = 3.5,
        a_sel: int = 20,
        a_compress_rate: int = 0,
        a_compress_e_rate: int = 1,
        a_compress_use_split: bool = False,
        n_multi_edge_message: int = 1,
        axis_neuron: int = 4,
        update_angle: bool = True,
        update_style: str = "res_residual",
        update_residual: float = 0.1,
        update_residual_init: str = "const",
        fix_stat_std: float = 0.3,
        skip_stat: bool = False,
        optim_update: bool = True,
        smooth_edge_update: bool = False,
        edge_init_use_dist: bool = False,
        use_exp_switch: bool = False,
        use_dynamic_sel: bool = False,
        sel_reduce_factor: float = 10.0,
    ) -> None:
        self.n_dim = n_dim
        self.e_dim = e_dim
        self.a_dim = a_dim
        self.nlayers = nlayers
        self.e_rcut = e_rcut
        self.e_rcut_smth = e_rcut_smth
        self.e_sel = e_sel
        self.a_rcut = a_rcut
        self.a_rcut_smth = a_rcut_smth
        self.a_sel = a_sel
        self.a_compress_rate = a_compress_rate
        self.n_multi_edge_message = n_multi_edge_message
        self.axis_neuron = axis_neuron
        self.update_angle = update_angle
        self.update_style = update_style
        self.update_residual = update_residual
        self.update_residual_init = update_residual_init
        self.fix_stat_std = (
            fix_stat_std if not skip_stat else 0.3
        )  # backward compatibility
        self.skip_stat = skip_stat
        self.a_compress_e_rate = a_compress_e_rate
        self.a_compress_use_split = a_compress_use_split
        self.optim_update = optim_update
        self.smooth_edge_update = smooth_edge_update
        self.edge_init_use_dist = edge_init_use_dist
        self.use_exp_switch = use_exp_switch
        self.use_dynamic_sel = use_dynamic_sel
        self.sel_reduce_factor = sel_reduce_factor

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(key)

    def serialize(self) -> dict:
        return {
            "n_dim": self.n_dim,
            "e_dim": self.e_dim,
            "a_dim": self.a_dim,
            "nlayers": self.nlayers,
            "e_rcut": self.e_rcut,
            "e_rcut_smth": self.e_rcut_smth,
            "e_sel": self.e_sel,
            "a_rcut": self.a_rcut,
            "a_rcut_smth": self.a_rcut_smth,
            "a_sel": self.a_sel,
            "a_compress_rate": self.a_compress_rate,
            "a_compress_e_rate": self.a_compress_e_rate,
            "a_compress_use_split": self.a_compress_use_split,
            "n_multi_edge_message": self.n_multi_edge_message,
            "axis_neuron": self.axis_neuron,
            "update_angle": self.update_angle,
            "update_style": self.update_style,
            "update_residual": self.update_residual,
            "update_residual_init": self.update_residual_init,
            "fix_stat_std": self.fix_stat_std,
            "optim_update": self.optim_update,
            "smooth_edge_update": self.smooth_edge_update,
            "edge_init_use_dist": self.edge_init_use_dist,
            "use_exp_switch": self.use_exp_switch,
            "use_dynamic_sel": self.use_dynamic_sel,
            "sel_reduce_factor": self.sel_reduce_factor,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "RepFlowArgs":
        return cls(**data)


@BaseDescriptor.register("dpa3")
class DescrptDPA3(NativeOP, BaseDescriptor):
    r"""The DPA3 descriptor[1]_.

    Parameters
    ----------
    repflow : Union[RepFlowArgs, dict]
        The arguments used to initialize the repflow block, see docstr in `RepFlowArgs` for details information.
    concat_output_tebd : bool, optional
        Whether to concat type embedding at the output of the descriptor.
    activation_function : str, optional
        The activation function in the embedding net.
    precision : str, optional
        The precision of the embedding net parameters.
    exclude_types : list[list[int]], optional
        The excluded pairs of types which have no interaction with each other.
        For example, `[[0, 1]]` means no interaction between type 0 and type 1.
    env_protection : float, optional
        Protection parameter to prevent division by zero errors during environment matrix calculations.
        For example, when using paddings, there may be zero distances of neighbors, which may make division by zero error during environment matrix calculations without protection.
    trainable : bool, optional
        If the parameters are trainable.
    seed : int, optional
        Random seed for parameter initialization.
    use_econf_tebd : bool, Optional
        Whether to use electronic configuration type embedding.
    use_tebd_bias : bool, Optional
        Whether to use bias in the type embedding layer.
    use_loc_mapping : bool, Optional
        Whether to use local atom index mapping in training or non-parallel inference.
        When True, local indexing and mapping are applied to neighbor lists and embeddings during descriptor computation.
    type_map : list[str], Optional
        A list of strings. Give the name to each type of atoms.

    References
    ----------
    .. [1] Zhang, D., Peng, A., Cai, C. et al. Graph neural
       network model for the era of large atomistic models.
       arXiv preprint arXiv:2506.01686 (2025).
    """

    def __init__(
        self,
        ntypes: int,
        # args for repflow
        repflow: Union[RepFlowArgs, dict],
        # kwargs for descriptor
        concat_output_tebd: bool = False,
        activation_function: str = "silu",
        precision: str = "float64",
        exclude_types: list[tuple[int, int]] = [],
        env_protection: float = 0.0,
        trainable: bool = True,
        seed: Optional[Union[int, list[int]]] = None,
        use_econf_tebd: bool = False,
        use_tebd_bias: bool = False,
        use_loc_mapping: bool = True,
        type_map: Optional[list[str]] = None,
    ) -> None:
        super().__init__()

        def init_subclass_params(sub_data, sub_class):
            if isinstance(sub_data, dict):
                return sub_class(**sub_data)
            elif isinstance(sub_data, sub_class):
                return sub_data
            else:
                raise ValueError(
                    f"Input args must be a {sub_class.__name__} class or a dict!"
                )

        self.repflow_args = init_subclass_params(repflow, RepFlowArgs)
        self.activation_function = activation_function

        self.repflows = DescrptBlockRepflows(
            self.repflow_args.e_rcut,
            self.repflow_args.e_rcut_smth,
            self.repflow_args.e_sel,
            self.repflow_args.a_rcut,
            self.repflow_args.a_rcut_smth,
            self.repflow_args.a_sel,
            ntypes,
            nlayers=self.repflow_args.nlayers,
            n_dim=self.repflow_args.n_dim,
            e_dim=self.repflow_args.e_dim,
            a_dim=self.repflow_args.a_dim,
            a_compress_rate=self.repflow_args.a_compress_rate,
            a_compress_e_rate=self.repflow_args.a_compress_e_rate,
            a_compress_use_split=self.repflow_args.a_compress_use_split,
            n_multi_edge_message=self.repflow_args.n_multi_edge_message,
            axis_neuron=self.repflow_args.axis_neuron,
            update_angle=self.repflow_args.update_angle,
            activation_function=self.activation_function,
            update_style=self.repflow_args.update_style,
            update_residual=self.repflow_args.update_residual,
            update_residual_init=self.repflow_args.update_residual_init,
            fix_stat_std=self.repflow_args.fix_stat_std,
            optim_update=self.repflow_args.optim_update,
            smooth_edge_update=self.repflow_args.smooth_edge_update,
            edge_init_use_dist=self.repflow_args.edge_init_use_dist,
            use_exp_switch=self.repflow_args.use_exp_switch,
            use_dynamic_sel=self.repflow_args.use_dynamic_sel,
            sel_reduce_factor=self.repflow_args.sel_reduce_factor,
            use_loc_mapping=use_loc_mapping,
            exclude_types=exclude_types,
            env_protection=env_protection,
            precision=precision,
            seed=child_seed(seed, 1),
        )

        self.use_econf_tebd = use_econf_tebd
        self.use_tebd_bias = use_tebd_bias
        self.use_loc_mapping = use_loc_mapping
        self.type_map = type_map
        self.tebd_dim = self.repflow_args.n_dim
        self.type_embedding = TypeEmbedNet(
            ntypes=ntypes,
            neuron=[self.tebd_dim],
            padding=True,
            activation_function="Linear",
            precision=precision,
            use_econf_tebd=self.use_econf_tebd,
            use_tebd_bias=use_tebd_bias,
            type_map=type_map,
            seed=child_seed(seed, 2),
        )
        self.concat_output_tebd = concat_output_tebd
        self.precision = precision
        self.exclude_types = exclude_types
        self.env_protection = env_protection
        self.trainable = trainable

        assert self.repflows.e_rcut >= self.repflows.a_rcut
        assert self.repflows.e_sel >= self.repflows.a_sel

        self.rcut = self.repflows.get_rcut()
        self.rcut_smth = self.repflows.get_rcut_smth()
        self.sel = self.repflows.get_sel()
        self.ntypes = ntypes

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
        ret = self.repflows.dim_out
        if self.concat_output_tebd:
            ret += self.tebd_dim
        return ret

    def get_dim_emb(self) -> int:
        """Returns the embedding dimension of this descriptor."""
        return self.repflows.dim_emb

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
        return self.repflows.has_message_passing()

    def need_sorted_nlist_for_lower(self) -> bool:
        """Returns whether the descriptor needs sorted nlist when using `forward_lower`."""
        return True

    def get_env_protection(self) -> float:
        """Returns the protection of building environment matrix."""
        return self.repflows.get_env_protection()

    def share_params(self, base_class, shared_level, resume=False) -> None:
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
        repflow = self.repflows
        if has_new_type:
            # the avg and std of new types need to be updated
            extend_descrpt_stat(
                repflow,
                type_map,
                des_with_stat=model_with_new_type_stat.repflows
                if model_with_new_type_stat is not None
                else None,
            )
        repflow.ntypes = self.ntypes
        repflow.reinit_exclude(self.exclude_types)
        repflow["davg"] = repflow["davg"][remap_index]
        repflow["dstd"] = repflow["dstd"][remap_index]

    @property
    def dim_out(self):
        return self.get_dim_out()

    @property
    def dim_emb(self):
        """Returns the embedding dimension g2."""
        return self.get_dim_emb()

    def compute_input_stats(self, merged: list[dict], path: Optional[DPPath] = None):
        """Update mean and stddev for descriptor elements."""
        descrpt_list = [self.repflows]
        for ii, descrpt in enumerate(descrpt_list):
            descrpt.compute_input_stats(merged, path)

    def set_stat_mean_and_stddev(
        self,
        mean: list[np.ndarray],
        stddev: list[np.ndarray],
    ) -> None:
        """Update mean and stddev for descriptor."""
        descrpt_list = [self.repflows]
        for ii, descrpt in enumerate(descrpt_list):
            descrpt.mean = mean[ii]
            descrpt.stddev = stddev[ii]

    def get_stat_mean_and_stddev(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Get mean and stddev for descriptor."""
        mean_list = [self.repflows.mean]
        stddev_list = [self.repflows.stddev]
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
            The index mapping, mapps extended region index to local region.

        Returns
        -------
        node_ebd
            The output descriptor. shape: nf x nloc x n_dim (or n_dim + tebd_dim)
        rot_mat
            The rotationally equivariant and permutationally invariant single particle
            representation. shape: nf x nloc x e_dim x 3
        edge_ebd
            The edge embedding.
            shape: nf x nloc x nnei x e_dim
        h2
            The rotationally equivariant pair-partical representation.
            shape: nf x nloc x nnei x 3
        sw
            The smooth switch function. shape: nf x nloc x nnei

        """
        xp = array_api_compat.array_namespace(coord_ext, atype_ext, nlist)
        nframes, nloc, nnei = nlist.shape
        nall = xp.reshape(coord_ext, (nframes, -1)).shape[1] // 3

        type_embedding = self.type_embedding.call()
        if self.use_loc_mapping:
            node_ebd_ext = xp.reshape(
                xp.take(type_embedding, xp.reshape(atype_ext[:, :nloc], [-1]), axis=0),
                (nframes, nloc, self.tebd_dim),
            )
        else:
            node_ebd_ext = xp.reshape(
                xp.take(type_embedding, xp.reshape(atype_ext, [-1]), axis=0),
                (nframes, nall, self.tebd_dim),
            )
        node_ebd_inp = node_ebd_ext[:, :nloc, :]
        # repflows
        node_ebd, edge_ebd, h2, rot_mat, sw = self.repflows(
            nlist,
            coord_ext,
            atype_ext,
            node_ebd_ext,
            mapping,
        )
        if self.concat_output_tebd:
            node_ebd = xp.concat([node_ebd, node_ebd_inp], axis=-1)
        return node_ebd, rot_mat, edge_ebd, h2, sw

    def serialize(self) -> dict:
        repflows = self.repflows
        data = {
            "@class": "Descriptor",
            "type": "dpa3",
            "@version": 2,
            "ntypes": self.ntypes,
            "repflow_args": self.repflow_args.serialize(),
            "concat_output_tebd": self.concat_output_tebd,
            "activation_function": self.activation_function,
            "precision": self.precision,
            "exclude_types": self.exclude_types,
            "env_protection": self.env_protection,
            "trainable": self.trainable,
            "use_econf_tebd": self.use_econf_tebd,
            "use_tebd_bias": self.use_tebd_bias,
            "use_loc_mapping": self.use_loc_mapping,
            "type_map": self.type_map,
            "type_embedding": self.type_embedding.serialize(),
        }
        repflow_variable = {
            "edge_embd": repflows.edge_embd.serialize(),
            "angle_embd": repflows.angle_embd.serialize(),
            "repflow_layers": [layer.serialize() for layer in repflows.layers],
            "env_mat": EnvMat(repflows.rcut, repflows.rcut_smth).serialize(),
            "@variables": {
                "davg": to_numpy_array(repflows["davg"]),
                "dstd": to_numpy_array(repflows["dstd"]),
            },
        }
        data.update(
            {
                "repflow_variable": repflow_variable,
            }
        )
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptDPA3":
        data = data.copy()
        version = data.pop("@version")
        check_version_compatibility(version, 2, 1)
        data.pop("@class")
        data.pop("type")
        repflow_variable = data.pop("repflow_variable").copy()
        type_embedding = data.pop("type_embedding")
        data["repflow"] = RepFlowArgs(**data.pop("repflow_args"))
        obj = cls(**data)
        obj.type_embedding = TypeEmbedNet.deserialize(type_embedding)

        # deserialize repflow
        statistic_repflows = repflow_variable.pop("@variables")
        env_mat = repflow_variable.pop("env_mat")
        repflow_layers = repflow_variable.pop("repflow_layers")
        obj.repflows.edge_embd = NativeLayer.deserialize(
            repflow_variable.pop("edge_embd")
        )
        obj.repflows.angle_embd = NativeLayer.deserialize(
            repflow_variable.pop("angle_embd")
        )
        obj.repflows["davg"] = statistic_repflows["davg"]
        obj.repflows["dstd"] = statistic_repflows["dstd"]
        obj.repflows.layers = [
            RepFlowLayer.deserialize(layer) for layer in repflow_layers
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
        min_nbor_dist, repflow_e_sel = update_sel.update_one_sel(
            train_data,
            type_map,
            local_jdata_cpy["repflow"]["e_rcut"],
            local_jdata_cpy["repflow"]["e_sel"],
            True,
        )
        local_jdata_cpy["repflow"]["e_sel"] = repflow_e_sel[0]

        min_nbor_dist, repflow_a_sel = update_sel.update_one_sel(
            train_data,
            type_map,
            local_jdata_cpy["repflow"]["a_rcut"],
            local_jdata_cpy["repflow"]["a_sel"],
            True,
        )
        local_jdata_cpy["repflow"]["a_sel"] = repflow_a_sel[0]

        return local_jdata_cpy, min_nbor_dist
