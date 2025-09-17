# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Callable,
    Optional,
    Union,
)

import torch

from deepmd.dpmodel.descriptor.dpa3 import (
    RepFlowArgs,
)
from deepmd.dpmodel.utils import EnvMat as DPEnvMat
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pt.model.network.mlp import (
    MLPLayer,
)
from deepmd.pt.model.network.network import (
    TypeEmbedNet,
    TypeEmbedNetConsistent,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
)
from deepmd.pt.utils.update_sel import (
    UpdateSel,
)
from deepmd.pt.utils.utils import (
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
from .repflow_layer import (
    RepFlowLayer,
)
from .repflows import (
    DescrptBlockRepflows,
)


@BaseDescriptor.register("dpa3")
class DescrptDPA3(BaseDescriptor, torch.nn.Module):
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
    enable_mad : bool, Optional
        Whether to enable MAD (Mean Average Distance) computation. Set to True to compute MAD values for regularization use.
    mad_cutoff_ratio : float, Optional
        The ratio to distinguish neighbor and remote nodes for MAD calculation. (Reserved for future extensions)

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
        enable_mad: bool = False, # new added
        mad_cutoff_ratio: float = 0.5, # new added (保留以便后续扩展)
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
# here defined the repflows
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
        self.use_loc_mapping = use_loc_mapping
        self.use_tebd_bias = use_tebd_bias
        self.type_map = type_map
        self.tebd_dim = self.repflow_args.n_dim
        self.type_embedding = TypeEmbedNet(
            ntypes,
            self.tebd_dim,
            precision=precision,
            seed=child_seed(seed, 2),
            use_econf_tebd=self.use_econf_tebd,
            use_tebd_bias=use_tebd_bias,
            type_map=type_map,
        )
        self.concat_output_tebd = concat_output_tebd
        self.precision = precision
        self.prec = PRECISION_DICT[self.precision]
        self.exclude_types = exclude_types
        self.env_protection = env_protection
        self.trainable = trainable

        assert self.repflows.e_rcut >= self.repflows.a_rcut, (
            f"Edge radial cutoff (e_rcut: {self.repflows.e_rcut}) "
            f"must be greater than or equal to angular cutoff (a_rcut: {self.repflows.a_rcut})!"
        )
        assert self.repflows.e_sel >= self.repflows.a_sel, (
            f"Edge sel number (e_sel: {self.repflows.e_sel}) "
            f"must be greater than or equal to angular sel (a_sel: {self.repflows.a_sel})!"
        )

        self.rcut = self.repflows.get_rcut()
        self.rcut_smth = self.repflows.get_rcut_smth()
        self.sel = self.repflows.get_sel()
        self.ntypes = ntypes

        # set trainable
        for param in self.parameters():
            param.requires_grad = trainable
        self.compress = False

        # MAD相关参数存储
        self.enable_mad = enable_mad
        self.mad_cutoff_ratio = mad_cutoff_ratio
        # 存储MAD值供损失函数使用 (变量名保持last_mad_gap以兼容损失函数)
        self.last_mad_gap = None

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
        assert self.__class__ == base_class.__class__, (
            "Only descriptors of the same type can share params!"
        )
        # For DPA3 descriptors, the user-defined share-level
        # shared_level: 0
        # share all parameters in type_embedding, repflow
        if shared_level == 0:
            self._modules["type_embedding"] = base_class._modules["type_embedding"]
            self.repflows.share_params(base_class.repflows, 0, resume=resume)
        # shared_level: 1
        # share all parameters in type_embedding
        elif shared_level == 1:
            self._modules["type_embedding"] = base_class._modules["type_embedding"]
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
                Each element, `merged[i]`, is a data dictionary containing `keys`: `torch.Tensor`
                originating from the `i`-th data system.
            - Callable[[], list[dict]]: A lazy function that returns data samples in the above format
                only when needed. Since the sampling process can be slow and memory-intensive,
                the lazy function helps by only sampling once.
        path : Optional[DPPath]
            The path to the stat file.

        """
        descrpt_list = [self.repflows]
        for ii, descrpt in enumerate(descrpt_list):
            descrpt.compute_input_stats(merged, path)

    def set_stat_mean_and_stddev(
        self,
        mean: list[torch.Tensor],
        stddev: list[torch.Tensor],
    ) -> None:
        """Update mean and stddev for descriptor."""
        descrpt_list = [self.repflows]
        for ii, descrpt in enumerate(descrpt_list):
            descrpt.mean = mean[ii]
            descrpt.stddev = stddev[ii]

    def get_stat_mean_and_stddev(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Get mean and stddev for descriptor."""
        mean_list = [self.repflows.mean]
        stddev_list = [self.repflows.stddev]
        return mean_list, stddev_list

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
            "type_embedding": self.type_embedding.embedding.serialize(),
            "enable_mad": self.enable_mad, # new added
            "mad_cutoff_ratio": self.mad_cutoff_ratio,
        }
        repflow_variable = {
            "edge_embd": repflows.edge_embd.serialize(),
            "angle_embd": repflows.angle_embd.serialize(),
            "repflow_layers": [layer.serialize() for layer in repflows.layers],
            "env_mat": DPEnvMat(repflows.rcut, repflows.rcut_smth).serialize(),
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
        obj.type_embedding.embedding = TypeEmbedNetConsistent.deserialize(
            type_embedding
        )

        def t_cvt(xx):
            return torch.tensor(xx, dtype=obj.repflows.prec, device=env.DEVICE)

        # deserialize repflow
        statistic_repflows = repflow_variable.pop("@variables")
        env_mat = repflow_variable.pop("env_mat")
        repflow_layers = repflow_variable.pop("repflow_layers")
        obj.repflows.edge_embd = MLPLayer.deserialize(repflow_variable.pop("edge_embd"))
        obj.repflows.angle_embd = MLPLayer.deserialize(
            repflow_variable.pop("angle_embd")
        )
        obj.repflows["davg"] = t_cvt(statistic_repflows["davg"])
        obj.repflows["dstd"] = t_cvt(statistic_repflows["dstd"])
        obj.repflows.layers = torch.nn.ModuleList(
            [RepFlowLayer.deserialize(layer) for layer in repflow_layers]
        )
        return obj

    def forward(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
        comm_dict: Optional[dict[str, torch.Tensor]] = None,
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
        parallel_mode = comm_dict is not None
        # cast the input to internal precsion
        extended_coord = extended_coord.to(dtype=self.prec)
        nframes, nloc, nnei = nlist.shape
        nall = extended_coord.view(nframes, -1).shape[1] // 3

        if not parallel_mode and self.use_loc_mapping:
            node_ebd_ext = self.type_embedding(extended_atype[:, :nloc])
        else:
            node_ebd_ext = self.type_embedding(extended_atype) # 节点嵌入表征   [nf, nall, tebd_dim] 
        node_ebd_inp = node_ebd_ext[:, :nloc, :] # 初始类型嵌入 [nframes, nloc, n_dim] (n_dim=128)
        # repflows
        node_ebd, edge_ebd, h2, rot_mat, sw = self.repflows(
            nlist,
            extended_coord,
            extended_atype,
            node_ebd_ext,
            mapping,
            comm_dict=comm_dict,
        )
        if self.concat_output_tebd: # 控制是否在输出时拼接初始类型嵌入
            node_ebd = torch.cat([node_ebd, node_ebd_inp], dim=-1) # 保留原始信息：确保初始的原子类型信息不会在多层RepFlow处理中完全丢失
        # 同时提供原始类型特征和经过环境学习的特征，这是一种残差连接的思想，类似于ResNet中跳跃连接，防止深层网络丢失重要的基础信息。

        # MAD计算（在启用时总是计算，不仅仅是训练时）
        if self.enable_mad:
            #print("Computing MAD for node_ebd shape:", node_ebd.shape)
            self.last_mad_gap = self._compute_mad(node_ebd)
            #print("MAD value:", self.last_mad_gap.item())
        else:
            self.last_mad_gap = None
        return (
            node_ebd.to(dtype=env.GLOBAL_PT_FLOAT_PRECISION), # # 1. 节点嵌入表征 [nframes, nloc, n_dim] (n_dim=128)
            rot_mat.to(dtype=env.GLOBAL_PT_FLOAT_PRECISION), # 2. 旋转等变矩阵 [nframes, nloc, e_dim, 3] (e_dim=128)
            edge_ebd.to(dtype=env.GLOBAL_PT_FLOAT_PRECISION),  # 3. 边嵌入表征 [nframes, nloc, nnei, e_dim] (e_dim=128)
            h2.to(dtype=env.GLOBAL_PT_FLOAT_PRECISION),  # 4. 方向向量 [nframes, nloc, nnei, 3] (3=xyz)
            sw.to(dtype=env.GLOBAL_PT_FLOAT_PRECISION),  # 5. 平滑开关函数 [nframes, nloc, nnei]
        )

    # 2. 添加简化的 _compute_mad 方法
    def _compute_mad(self, node_ebd: torch.Tensor) -> torch.Tensor:
        """计算基础MAD (Mean Average Distance) 用于正则化
        
        MAD使用余弦距离衡量节点嵌入表征之间的平均距离:
        余弦距离 = 1 - 余弦相似度 = 1 - (Hi · Hj) / (|Hi| · |Hj|)
        
        Parameters
        ---------- 
        node_ebd : torch.Tensor
            节点嵌入表征，形状 [nframes, nloc, embed_dim]
            
        Returns
        -------
        torch.Tensor
            所有节点对之间的平均余弦距离
        """
        import torch.nn.functional as F
        nframes, nloc, embed_dim = node_ebd.shape
        device = node_ebd.device
        if nloc <= 1:
            return torch.tensor(0.0, device=node_ebd.device)
        node_ebd_norm = F.normalize(node_ebd, p=2, dim=-1)  # [nf, nloc, embed_dim]
        
        # 计算余弦相似度矩阵
        cosine_sim = torch.bmm(node_ebd_norm, node_ebd_norm.transpose(-1, -2))
        
        # 余弦距离 = 1 - 余弦相似度
        cosine_dist = 1.0 - cosine_sim
        # 不相似 --> cosin_sim --> 0
        # Global MAD
        #global_mad = cosine_dist.mean()        
        # 排除对角线（自己与自己的距离为0）
        #eye_mask = torch.eye(nloc, dtype=torch.bool, device=device).unsqueeze(0).expand(nframes, -1, -1)
        #valid_mask = ~eye_mask
        
        # 计算所有有效节点对的平均距离
        #valid_distances = cosine_dist[valid_mask]
        mad_global = cosine_dist.sum() / (nframes * nloc * (nloc - 1))
        
        return mad_global

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
        raise NotImplementedError("Compression is unsupported for DPA3.")
