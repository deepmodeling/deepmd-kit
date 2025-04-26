# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Callable,
    Optional,
    Union,
)

import paddle

from deepmd.dpmodel.descriptor.dpa2 import (
    RepformerArgs,
    RepinitArgs,
)
from deepmd.dpmodel.utils import EnvMat as DPEnvMat
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pd.model.network.mlp import (
    Identity,
    MLPLayer,
    NetworkCollection,
)
from deepmd.pd.model.network.network import (
    TypeEmbedNet,
    TypeEmbedNetConsistent,
)
from deepmd.pd.utils import (
    env,
)
from deepmd.pd.utils.env import (
    PRECISION_DICT,
)
from deepmd.pd.utils.nlist import (
    build_multiple_neighbor_list,
    get_multiple_nlist_key,
)
from deepmd.pd.utils.update_sel import (
    UpdateSel,
)
from deepmd.pd.utils.utils import (
    to_numpy_array,
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
from .repformer_layer import (
    RepformerLayer,
)
from .repformers import (
    DescrptBlockRepformers,
)
from .se_atten import (
    DescrptBlockSeAtten,
)
from .se_t_tebd import (
    DescrptBlockSeTTebd,
)


@BaseDescriptor.register("dpa2")
class DescrptDPA2(BaseDescriptor, paddle.nn.Layer):
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
            Random seed for parameter initialization.
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
        descriptor:         paddle.Tensor
            the descriptor of shape nb x nloc x g1_dim.
            invariant single-atom representation.
        g2:                 paddle.Tensor
            invariant pair-atom representation.
        h2:                 paddle.Tensor
            equivariant pair-atom representation.
        rot_mat:            paddle.Tensor
            rotation matrix for equivariant fittings
        sw:                 paddle.Tensor
            The switch function for decaying inverse distance.

        References
        ----------
        .. [1] Zhang, D., Liu, X., Zhang, X. et al. DPA-2: a
           large atomic model as a multi-task learner. npj
           Comput Mater 10, 293 (2024). https://doi.org/10.1038/s41524-024-01493-2
        """
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

        self.repinit_args = init_subclass_params(repinit, RepinitArgs)
        self.repformer_args = init_subclass_params(repformer, RepformerArgs)
        self.tebd_input_mode = self.repinit_args.tebd_input_mode

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
            ln_eps=self.repformer_args.ln_eps,
            use_sqrt_nnei=self.repformer_args.use_sqrt_nnei,
            g1_out_conv=self.repformer_args.g1_out_conv,
            g1_out_mlp=self.repformer_args.g1_out_mlp,
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
            ntypes,
            self.repinit_args.tebd_dim,
            precision=precision,
            seed=child_seed(seed, 2),
            use_econf_tebd=self.use_econf_tebd,
            use_tebd_bias=use_tebd_bias,
            type_map=type_map,
        )
        self.concat_output_tebd = concat_output_tebd
        self.precision = precision
        self.prec = PRECISION_DICT[self.precision]
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
            self.g1_shape_tranform = MLPLayer(
                self.repinit_out_dim,
                self.repformers.dim_in,
                bias=False,
                precision=precision,
                init="glorot",
                seed=child_seed(seed, 3),
            )
        self.tebd_transform = None
        if self.add_tebd_to_repinit_out:
            self.tebd_transform = MLPLayer(
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
        self.rcut_smth = self.repinit.get_rcut_smth()
        self.ntypes = ntypes
        self.sel = self.repinit.sel
        # set trainable
        for param in self.parameters():
            param.stop_gradient = not trainable
        self.compress = False

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
        # the env_protection of repinit is the same as that of the repformer
        return self.repinit.get_env_protection()

    def share_params(self, base_class, shared_level, resume=False) -> None:
        """
        Share the parameters of self to the base_class with shared_level during multitask training.
        If not start from checkpoint (resume is False),
        some separated parameters (e.g. mean and stddev) will be re-calculated across different classes.
        """
        assert self.__class__ == base_class.__class__, (
            "Only descriptors of the same type can share params!"
        )
        # For DPA2 descriptors, the user-defined share-level
        # shared_level: 0
        # share all parameters in type_embedding, repinit and repformers
        if shared_level == 0:
            self._sub_layers["type_embedding"] = base_class._sub_layers[
                "type_embedding"
            ]
            self.repinit.share_params(base_class.repinit, 0, resume=resume)
            if self.use_three_body:
                self.repinit_three_body.share_params(
                    base_class.repinit_three_body, 0, resume=resume
                )
            self._sub_layers["g1_shape_tranform"] = base_class._sub_layers[
                "g1_shape_tranform"
            ]
            self.repformers.share_params(base_class.repformers, 0, resume=resume)
        # shared_level: 1
        # share all parameters in type_embedding
        elif shared_level == 1:
            self._sub_layers["type_embedding"] = base_class._sub_layers[
                "type_embedding"
            ]
        # Other shared levels
        else:
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
    ) -> None:
        """
        Compute the input statistics (e.g. mean and stddev) for the descriptors from packed data.

        Parameters
        ----------
        merged : Union[Callable[[], list[dict]], list[dict]]
            - list[dict]: A list of data samples from various data systems.
                Each element, `merged[i]`, is a data dictionary containing `keys`: `paddle.Tensor`
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
        mean: list[paddle.Tensor],
        stddev: list[paddle.Tensor],
    ) -> None:
        """Update mean and stddev for descriptor."""
        descrpt_list = [self.repinit, self.repformers]
        if self.use_three_body:
            descrpt_list.append(self.repinit_three_body)
        for ii, descrpt in enumerate(descrpt_list):
            descrpt.mean = mean[ii]
            descrpt.stddev = stddev[ii]

    def get_stat_mean_and_stddev(
        self,
    ) -> tuple[list[paddle.Tensor], list[paddle.Tensor]]:
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
                "repinit_variable": repinit_variable,
                "repformers_variable": repformers_variable,
            }
        )
        if self.use_three_body:
            repinit_three_body_variable = {
                "embeddings": repinit_three_body.filter_layers.serialize(),
                "env_mat": DPEnvMat(
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
                        "embeddings_strip": repinit_three_body.filter_layers_strip.serialize()
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
        obj.type_embedding.embedding = TypeEmbedNetConsistent.deserialize(
            type_embedding
        )
        if add_tebd_to_repinit_out:
            assert isinstance(tebd_transform, dict)
            obj.tebd_transform = MLPLayer.deserialize(tebd_transform)
        if obj.repinit.dim_out != obj.repformers.dim_in:
            obj.g1_shape_tranform = MLPLayer.deserialize(g1_shape_tranform)

        def t_cvt(xx):
            return paddle.to_tensor(xx, dtype=obj.repinit.prec, place=env.DEVICE)

        # deserialize repinit
        statistic_repinit = repinit_variable.pop("@variables")
        env_mat = repinit_variable.pop("env_mat")
        tebd_input_mode = data["repinit"].tebd_input_mode
        obj.repinit.filter_layers = NetworkCollection.deserialize(
            repinit_variable.pop("embeddings")
        )
        if tebd_input_mode in ["strip"]:
            obj.repinit.filter_layers_strip = NetworkCollection.deserialize(
                repinit_variable.pop("embeddings_strip")
            )
        obj.repinit["davg"] = t_cvt(statistic_repinit["davg"])
        obj.repinit["dstd"] = t_cvt(statistic_repinit["dstd"])

        if data["repinit"].use_three_body:
            # deserialize repinit_three_body
            statistic_repinit_three_body = repinit_three_body_variable.pop("@variables")
            env_mat = repinit_three_body_variable.pop("env_mat")
            tebd_input_mode = data["repinit"].tebd_input_mode
            obj.repinit_three_body.filter_layers = NetworkCollection.deserialize(
                repinit_three_body_variable.pop("embeddings")
            )
            if tebd_input_mode in ["strip"]:
                obj.repinit_three_body.filter_layers_strip = (
                    NetworkCollection.deserialize(
                        repinit_three_body_variable.pop("embeddings_strip")
                    )
                )
            obj.repinit_three_body["davg"] = t_cvt(statistic_repinit_three_body["davg"])
            obj.repinit_three_body["dstd"] = t_cvt(statistic_repinit_three_body["dstd"])

        # deserialize repformers
        statistic_repformers = repformers_variable.pop("@variables")
        env_mat = repformers_variable.pop("env_mat")
        repformer_layers = repformers_variable.pop("repformer_layers")
        obj.repformers.g2_embd = MLPLayer.deserialize(
            repformers_variable.pop("g2_embd")
        )
        obj.repformers["davg"] = t_cvt(statistic_repformers["davg"])
        obj.repformers["dstd"] = t_cvt(statistic_repformers["dstd"])
        obj.repformers.layers = paddle.nn.LayerList(
            [RepformerLayer.deserialize(layer) for layer in repformer_layers]
        )
        return obj

    def forward(
        self,
        extended_coord: paddle.Tensor,
        extended_atype: paddle.Tensor,
        nlist: paddle.Tensor,
        mapping: Optional[paddle.Tensor] = None,
        comm_dict: Optional[dict[str, paddle.Tensor]] = None,
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
        # cast the input to internal precsion
        extended_coord = extended_coord.to(dtype=self.prec)

        use_three_body = self.use_three_body
        nframes, nloc, nnei = nlist.shape
        nall = extended_coord.reshape([nframes, -1]).shape[1] // 3
        # nlists
        nlist_dict = build_multiple_neighbor_list(
            extended_coord.detach(),
            nlist,
            self.rcut_list,
            self.nsel_list,
        )
        # repinit
        g1_ext = self.type_embedding(extended_atype)
        g1_inp = g1_ext[:, :nloc, :]
        if self.tebd_input_mode in ["strip"]:
            type_embedding = self.type_embedding.get_full_embedding(g1_ext.place)
        else:
            type_embedding = None
        g1, _, _, _, _ = self.repinit(
            nlist_dict[
                get_multiple_nlist_key(self.repinit.get_rcut(), self.repinit.get_nsel())
            ],
            extended_coord,
            extended_atype,
            g1_ext,
            mapping,
            type_embedding,
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
                extended_coord,
                extended_atype,
                g1_ext,
                mapping,
                type_embedding,
            )
            g1 = paddle.concat([g1, g1_three_body], axis=-1)
        # linear to change shape
        g1 = self.g1_shape_tranform(g1)
        if self.add_tebd_to_repinit_out:
            assert self.tebd_transform is not None
            g1 = g1 + self.tebd_transform(g1_inp)
        # mapping g1
        if comm_dict is None:
            assert mapping is not None
            mapping_ext = (
                mapping.reshape([nframes, nall])
                .unsqueeze(-1)
                .expand([-1, -1, g1.shape[-1]])
            )
            g1_ext = paddle.take_along_axis(g1, mapping_ext, 1)
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
            comm_dict=comm_dict,
        )
        if self.concat_output_tebd:
            g1 = paddle.concat([g1, g1_inp], axis=-1)
        return (
            g1.to(dtype=env.GLOBAL_PD_FLOAT_PRECISION),
            rot_mat.to(dtype=env.GLOBAL_PD_FLOAT_PRECISION),
            g2.to(dtype=env.GLOBAL_PD_FLOAT_PRECISION),
            h2.to(dtype=env.GLOBAL_PD_FLOAT_PRECISION),
            sw.to(dtype=env.GLOBAL_PD_FLOAT_PRECISION),
        )

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

    def enable_compression(
        self,
        min_nbor_dist: float,
        table_extrapolate: float = 5,
        table_stride_1: float = 0.01,
        table_stride_2: float = 0.1,
        check_frequency: int = -1,
    ) -> None:
        """Receive the statistics (distance, max_nbor_size and env_mat_range) of the training data.

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
        """
        # do some checks before the mocel compression process
        raise ValueError("Compression is already enabled.")
