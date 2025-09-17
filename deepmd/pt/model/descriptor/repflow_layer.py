# SPDX-License-Identifier: LGPL-3.0-or-later
"""
RepFlow层实现模块

RepFlow (Representation Flow) 是DPA3模型的核心消息传递层，实现了：
1. 节点、边、角度的表征更新
2. 旋转等变和置换不变的消息传递
3. 多体相互作用的建模
4. 物理约束的保持

本模块包含RepFlowLayer类，是DPA3描述符的基础构建块。

残差连接策略说明：
RepFlow层支持三种残差连接策略，通过update_style参数控制：

1. "res_avg" (残差平均):
   u = (u₀ + u₁ + u₂ + ... + uₙ) / √(n+1)
   - 所有更新项等权重相加
   - 简单稳定，适合更新项重要性相近的情况

2. "res_incr" (残差增量):
   u = u₀ + (u₁ + u₂ + ... + uₙ) / √n  
   - 原始表征保持完整权重，更新项作为增量
   - 与ResNet思想最接近，适合更新项作为修正的情况

3. "res_residual" (残差权重，默认):
   u = u₀ + r₁*u₁ + r₂*u₂ + ... + rₙ*uₙ
   - 每个更新项有独立的可学习权重
   - 提供最大灵活性，模型自动学习权重分配
   - 需要更多参数，适合复杂任务
"""
from typing import (
    Optional,
    Union,
)

import torch
import torch.nn as nn

from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pt.model.descriptor.repformer_layer import (
    _apply_nlist_mask,  # 应用邻居列表掩码
    _apply_switch,      # 应用开关函数
    _make_nei_g1,       # 构建邻居节点特征
    get_residual,       # 获取残差连接权重
)
from deepmd.pt.model.network.mlp import (
    MLPLayer,  # 基础MLP层
)
from deepmd.pt.model.network.utils import (
    aggregate,  # 聚合函数
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,  # 精度字典
)
from deepmd.pt.utils.utils import (
    ActivationFn,      # 激活函数
    to_numpy_array,    # 转换为numpy数组
    to_torch_tensor,   # 转换为torch张量
)
from deepmd.utils.version import (
    check_version_compatibility,  # 版本兼容性检查
)


class RepFlowLayer(torch.nn.Module):
    """RepFlow层：DPA3模型的核心消息传递层
    
    RepFlow层实现了图神经网络中的消息传递机制，包括：
    1. 节点表征更新：通过自更新、对称化、边消息传递
    2. 边表征更新：通过自更新、角度消息传递
    3. 角度表征更新：通过自更新（如果启用）
    
    该层保证旋转等变性和置换不变性，是DPA3描述符的基础构建块。
    """
    def __init__(
        self,
        e_rcut: float,                    # 边截断半径
        e_rcut_smth: float,               # 边平滑截断半径
        e_sel: int,                       # 边邻居选择数量
        a_rcut: float,                    # 角度截断半径
        a_rcut_smth: float,               # 角度平滑截断半径
        a_sel: int,                       # 角度邻居选择数量
        ntypes: int,                      # 原子类型数量
        n_dim: int = 128,                 # 节点表征维度
        e_dim: int = 16,                  # 边表征维度
        a_dim: int = 64,                  # 角度表征维度
        a_compress_rate: int = 0,         # 角度压缩率
        a_compress_use_split: bool = False, # 是否使用分割压缩
        a_compress_e_rate: int = 1,       # 角度边压缩率
        n_multi_edge_message: int = 1,    # 多头边消息数量
        axis_neuron: int = 4,             # 轴神经元数量
        update_angle: bool = True,        # 是否更新角度表征
        optim_update: bool = True,        # 是否使用优化更新
        use_dynamic_sel: bool = False,    # 是否使用动态选择
        sel_reduce_factor: float = 10.0,  # 选择减少因子
        smooth_edge_update: bool = False, # 是否平滑边更新
        activation_function: str = "silu", # 激活函数
        update_style: str = "res_residual", # 更新风格
        update_residual: float = 0.1,     # 残差更新参数
        update_residual_init: str = "const", # 残差初始化方式
        precision: str = "float64",       # 数值精度
        seed: Optional[Union[int, list[int]]] = None, # 随机种子
    ) -> None:
        super().__init__()
        self.epsilon = 1e-4  # protection of 1./nnei
        self.e_rcut = float(e_rcut)
        self.e_rcut_smth = float(e_rcut_smth)
        self.ntypes = ntypes
        e_sel = [e_sel] if isinstance(e_sel, int) else e_sel
        self.nnei = sum(e_sel)
        assert len(e_sel) == 1
        self.e_sel = e_sel
        self.sec = self.e_sel
        self.a_rcut = a_rcut
        self.a_rcut_smth = a_rcut_smth
        self.a_sel = a_sel
        self.n_dim = n_dim
        self.e_dim = e_dim
        self.a_dim = a_dim
        self.a_compress_rate = a_compress_rate
        if a_compress_rate != 0:
            assert (a_dim * a_compress_e_rate) % (2 * a_compress_rate) == 0, (
                f"For a_compress_rate of {a_compress_rate}, a_dim*a_compress_e_rate must be divisible by {2 * a_compress_rate}. "
                f"Currently, a_dim={a_dim} and a_compress_e_rate={a_compress_e_rate} is not valid."
            )
        self.n_multi_edge_message = n_multi_edge_message
        assert self.n_multi_edge_message >= 1, "n_multi_edge_message must >= 1!"
        self.axis_neuron = axis_neuron
        self.update_angle = update_angle
        self.activation_function = activation_function
        self.act = ActivationFn(activation_function)
        self.update_style = update_style
        self.update_residual = update_residual
        self.update_residual_init = update_residual_init
        self.a_compress_e_rate = a_compress_e_rate
        self.a_compress_use_split = a_compress_use_split
        self.precision = precision
        self.seed = seed
        self.prec = PRECISION_DICT[precision]
        self.optim_update = optim_update
        self.smooth_edge_update = smooth_edge_update
        self.use_dynamic_sel = use_dynamic_sel
        self.sel_reduce_factor = sel_reduce_factor
        self.dynamic_e_sel = self.nnei / self.sel_reduce_factor
        self.dynamic_a_sel = self.a_sel / self.sel_reduce_factor

        assert update_residual_init in [
            "norm",
            "const",
        ], "'update_residual_init' only support 'norm' or 'const'!"

        self.update_residual = update_residual
        self.update_residual_init = update_residual_init
         # 残差连接权重列表
        self.n_residual = []
        self.e_residual = []
        self.a_residual = []
        self.edge_info_dim = self.n_dim * 2 + self.e_dim

        # node self mlp
        self.node_self_mlp = MLPLayer(
            n_dim,
            n_dim,
            precision=precision,
            seed=child_seed(seed, 0),
        )
         # 如果使用残差连接，添加残差权重
        if self.update_style == "res_residual":
            self.n_residual.append(
                get_residual(
                    n_dim,
                    self.update_residual,
                    self.update_residual_init,
                    precision=precision,
                    seed=child_seed(seed, 1),
                )
            )

        # node sym (grrg + drrd) # 节点对称化MLP：处理GRRG不变量
        # 输入维度：节点×axis + 边×axis = 128×4 + 64×4 = 768
        self.n_sym_dim = n_dim * self.axis_neuron + e_dim * self.axis_neuron
        self.node_sym_linear = MLPLayer(
            self.n_sym_dim,
            n_dim,
            precision=precision,
            seed=child_seed(seed, 2),
        )
        if self.update_style == "res_residual":
            self.n_residual.append(
                get_residual(
                    n_dim,
                    self.update_residual,
                    self.update_residual_init,
                    precision=precision,
                    seed=child_seed(seed, 3),
                )
            )
    # 节点-边消息传递MLP
    # 输入：边信息（320维） → 输出：多头节点更新（1×128或多头）
        # node edge message
        self.node_edge_linear = MLPLayer(
            self.edge_info_dim,
            self.n_multi_edge_message * n_dim,
            precision=precision,
            seed=child_seed(seed, 4),
        ) 
        # 如果使用残差连接，添加残差权重
        if self.update_style == "res_residual":
            for head_index in range(self.n_multi_edge_message):
                self.n_residual.append(
                    get_residual(
                        n_dim,
                        self.update_residual,
                        self.update_residual_init,
                        precision=precision,
                        seed=child_seed(child_seed(seed, 5), head_index),
                    )
                )
    # =============================================================================
    # 3. 边相关的神经网络层
    # =============================================================================
    
    # 边自更新MLP：边信息（320维） → 边表征（64维）
        # edge self message
        self.edge_self_linear = MLPLayer(
            self.edge_info_dim,
            e_dim,
            precision=precision,
            seed=child_seed(seed, 6),
        )
        if self.update_style == "res_residual":
            self.e_residual.append(
                get_residual(
                    e_dim,
                    self.update_residual,
                    self.update_residual_init,
                    precision=precision,
                    seed=child_seed(seed, 7),
                )
            )
    # =============================================================================
    # 4. 角度相关的神经网络层（如果启用角度更新）
    # =============================================================================
    # 如果启用角度更新，则构建角度相关的神经网络层
        if self.update_angle:
            self.angle_dim = self.a_dim # 角度维度 
            if self.a_compress_rate == 0:  # 无压缩模式
                # angle + node + edge * 2
                self.angle_dim += self.n_dim + 2 * self.e_dim # 角度维度 = 角度维度 + 节点维度 + 边维度 × 2
                self.a_compress_n_linear = None
                self.a_compress_e_linear = None
                self.e_a_compress_dim = e_dim
                self.n_a_compress_dim = n_dim
            else: # 压缩模式
                # angle + a_dim/c + a_dim/2c * 2 * e_rate
                self.angle_dim += (1 + self.a_compress_e_rate) * (
                    self.a_dim // self.a_compress_rate
                ) # 角度维度 = 角度维度 + 角度维度 / 压缩率 × 压缩率
                self.e_a_compress_dim = (
                    self.a_dim // (2 * self.a_compress_rate) * self.a_compress_e_rate
                ) # 边维度 = 角度维度 / 压缩率 × 压缩率
                self.n_a_compress_dim = self.a_dim // self.a_compress_rate # 节点维度 = 角度维度 / 压缩率
                if not self.a_compress_use_split: # 不使用分割模式
                    self.a_compress_n_linear = MLPLayer(
                        self.n_dim,
                        self.n_a_compress_dim,
                        precision=precision,
                        bias=False,
                        seed=child_seed(seed, 8),
                    ) # 
                    self.a_compress_e_linear = MLPLayer(
                        self.e_dim,
                        self.e_a_compress_dim,
                        precision=precision,
                        bias=False,
                        seed=child_seed(seed, 9),
                    )
                else:
                    self.a_compress_n_linear = None
                    self.a_compress_e_linear = None

            # edge angle message
                    # 边-角度消息传递：两层MLP设计
        # 第一层：角度信息 → 边表征
            self.edge_angle_linear1 = MLPLayer(
                self.angle_dim,
                self.e_dim,
                precision=precision,
                seed=child_seed(seed, 10),
            )
            self.edge_angle_linear2 = MLPLayer( # 第二层：边表征 → 边表征（进一步处理）
                self.e_dim,
                self.e_dim,
                precision=precision,
                seed=child_seed(seed, 11),
            )
            if self.update_style == "res_residual":
                self.e_residual.append(
                    get_residual(
                        self.e_dim,
                        self.update_residual,
                        self.update_residual_init,
                        precision=precision,
                        seed=child_seed(seed, 12),
                    )
                )

            # angle self message  # 角度自更新MLP
            self.angle_self_linear = MLPLayer(
                self.angle_dim,
                self.a_dim,
                precision=precision,
                seed=child_seed(seed, 13),
            )
            if self.update_style == "res_residual":
                self.a_residual.append(
                    get_residual(
                        self.a_dim,
                        self.update_residual,
                        self.update_residual_init,
                        precision=precision,
                        seed=child_seed(seed, 14),
                    )
                )
        else:
            self.angle_self_linear = None
            self.edge_angle_linear1 = None
            self.edge_angle_linear2 = None
            self.a_compress_n_linear = None
            self.a_compress_e_linear = None
            self.angle_dim = 0
        # 将残差权重转换为PyTorch参数列表
        # 这些权重用于"res_residual"策略，每个更新项都有独立的可学习权重
        self.n_residual = nn.ParameterList(self.n_residual)  # 节点残差权重
        self.e_residual = nn.ParameterList(self.e_residual)  # 边残差权重
        self.a_residual = nn.ParameterList(self.a_residual)  # 角度残差权重

    @staticmethod
    def _cal_hg(
        edge_ebd: torch.Tensor,
        h2: torch.Tensor,
        nlist_mask: torch.Tensor,
        sw: torch.Tensor,
    ) -> torch.Tensor:
        """计算转置旋转矩阵
        
        这个函数计算用于对称化操作的转置旋转矩阵，是GRRG不变量计算的关键步骤。
        通过边嵌入和旋转等变张量的矩阵乘法，生成旋转等变的几何信息。

        Parameters
        ----------
        edge_ebd : torch.Tensor
            邻居/原子对边嵌入，形状为 nb x nloc x nnei x e_dim
        h2 : torch.Tensor
            邻居/原子对等变表征张量，形状为 nb x nloc x nnei x 3
        nlist_mask : torch.Tensor
            邻居列表掩码，0表示无邻居，形状为 nb x nloc x nnei
        sw : torch.Tensor
            开关函数，在rcut_smth范围内为1，在rcut_smth到rcut之间平滑衰减到0，
            在rcut之外为0，形状为 nb x nloc x nnei

        Returns
        -------
        hg : torch.Tensor
            转置旋转矩阵，形状为 nb x nloc x 3 x e_dim
        """
        # 获取张量形状信息
        # edge_ebd:  nb x nloc x nnei x e_dim
        # h2:  nb x nloc x nnei x 3
        # msk: nb x nloc x nnei
        nb, nloc, nnei, _ = edge_ebd.shape
        e_dim = edge_ebd.shape[-1]
        
        # 应用邻居列表掩码和开关函数
        # 形状: nb x nloc x nnei x e_dim
        edge_ebd = _apply_nlist_mask(edge_ebd, nlist_mask)  # 将无效邻居的边嵌入置零
        edge_ebd = _apply_switch(edge_ebd, sw)  # 应用开关函数进行平滑截断
        
        # 计算邻居数量的逆平方根，用于归一化
        invnnei = torch.rsqrt(
            float(nnei)
            * torch.ones((nb, nloc, 1, 1), dtype=edge_ebd.dtype, device=edge_ebd.device)
        )
        
        # 计算转置旋转矩阵：h2^T * edge_ebd
        # h2转置后形状: nb x nloc x 3 x nnei
        # edge_ebd形状: nb x nloc x nnei x e_dim
        # 结果形状: nb x nloc x 3 x e_dim
        h2g2 = torch.matmul(torch.transpose(h2, -1, -2), edge_ebd) * invnnei
        return h2g2

    @staticmethod
    def _cal_hg_dynamic(
        flat_edge_ebd: torch.Tensor,
        flat_h2: torch.Tensor,
        flat_sw: torch.Tensor,
        owner: torch.Tensor,
        num_owner: int,
        nb: int,
        nloc: int,
        scale_factor: float,
    ) -> torch.Tensor:
        """计算转置旋转矩阵（动态选择版本）
        
        这是_cal_hg函数的动态选择版本，用于处理变长邻居列表的情况。
        通过聚合函数将扁平化的边信息聚合到节点，生成旋转等变的几何信息。

        Parameters
        ----------
        flat_edge_ebd : torch.Tensor
            扁平化的邻居/原子对不变表征张量，形状为 n_edge x e_dim
        flat_h2 : torch.Tensor
            扁平化的邻居/原子对等变表征张量，形状为 n_edge x 3
        flat_sw : torch.Tensor
            扁平化的开关函数，在rcut_smth范围内为1，在rcut_smth到rcut之间平滑衰减到0，
            在rcut之外为0，形状为 n_edge
        owner : torch.Tensor
            邻居归约的所有者索引
        num_owner : int
            所有者的总数
        nb : int
            批次数
        nloc : int
            局部原子数
        scale_factor : float
            归约后应用的缩放因子

        Returns
        -------
        hg : torch.Tensor
            转置旋转矩阵，形状为 nf x nloc x 3 x e_dim
        """
        n_edge, e_dim = flat_edge_ebd.shape
        
        # 应用开关函数到边嵌入
        # 形状: n_edge x e_dim
        flat_edge_ebd = flat_edge_ebd * flat_sw.unsqueeze(-1)
        
        # 计算外积：h2[:, None, :] * edge_ebd[:, :, None]
        # 形状: n_edge x 3 x e_dim
        flat_h2g2 = (flat_h2[..., None] * flat_edge_ebd[:, None, :]).reshape(
            -1, 3 * e_dim
        )
        
        # 使用聚合函数将边信息聚合到节点
        # 形状: nf x nloc x 3 x e_dim
        h2g2 = (
            aggregate(flat_h2g2, owner, average=False, num_owner=num_owner).reshape(
                nb, nloc, 3, e_dim
            )
            * scale_factor  # 应用缩放因子
        )
        return h2g2

    @staticmethod
    def _cal_grrg(h2g2: torch.Tensor, axis_neuron: int) -> torch.Tensor:
        """
        Calculate the atomic invariant rep.

        Parameters
        ----------
        h2g2
            The transposed rotation matrix, with shape nb x nloc x 3 x e_dim.
        axis_neuron
            Size of the submatrix.

        Returns
        -------
        grrg
            Atomic invariant rep, with shape nb x nloc x (axis_neuron x e_dim)
        """
        # nb x nloc x 3 x e_dim
        nb, nloc, _, e_dim = h2g2.shape
        # nb x nloc x 3 x axis
        h2g2m = h2g2[..., :axis_neuron]
        # nb x nloc x axis x e_dim
        g1_13 = torch.matmul(torch.transpose(h2g2m, -1, -2), h2g2) / (3.0**1)
        # nb x nloc x (axisxng2)
        g1_13 = g1_13.view(nb, nloc, axis_neuron * e_dim)
        return g1_13

    def symmetrization_op(
        self,
        edge_ebd: torch.Tensor,
        h2: torch.Tensor,
        nlist_mask: torch.Tensor,
        sw: torch.Tensor,
        axis_neuron: int,
    ) -> torch.Tensor:
        """
        Symmetrization operator to obtain atomic invariant rep.

        Parameters
        ----------
        edge_ebd
            Neighbor-wise/Pair-wise invariant rep tensors, with shape nb x nloc x nnei x e_dim.
        h2
            Neighbor-wise/Pair-wise equivariant rep tensors, with shape nb x nloc x nnei x 3.
        nlist_mask
            Neighbor list mask, where zero means no neighbor, with shape nb x nloc x nnei.
        sw
            The switch function, which equals 1 within the rcut_smth range, smoothly decays from 1 to 0 between rcut_smth and rcut,
            and remains 0 beyond rcut, with shape nb x nloc x nnei.
        axis_neuron
            Size of the submatrix.

        Returns
        -------
        grrg
            Atomic invariant rep, with shape nb x nloc x (axis_neuron x e_dim)
        """
        # edge_ebd:  nb x nloc x nnei x e_dim
        # h2:  nb x nloc x nnei x 3
        # msk: nb x nloc x nnei
        nb, nloc, nnei, _ = edge_ebd.shape
        # nb x nloc x 3 x e_dim
        h2g2 = self._cal_hg(
            edge_ebd,
            h2,
            nlist_mask,
            sw,
        )
        # nb x nloc x (axisxng2)
        g1_13 = self._cal_grrg(h2g2, axis_neuron)
        return g1_13

    def symmetrization_op_dynamic(
        self,
        flat_edge_ebd: torch.Tensor,
        flat_h2: torch.Tensor,
        flat_sw: torch.Tensor,
        owner: torch.Tensor,
        num_owner: int,
        nb: int,
        nloc: int,
        scale_factor: float,
        axis_neuron: int,
    ) -> torch.Tensor:
        """
        Symmetrization operator to obtain atomic invariant rep.

        Parameters
        ----------
        flat_edge_ebd
            Flatted neighbor-wise/pair-wise invariant rep tensors, with shape n_edge x e_dim.
        flat_h2
            Flatted neighbor-wise/pair-wise equivariant rep tensors, with shape n_edge x 3.
        flat_sw
            Flatted switch function, which equals 1 within the rcut_smth range, smoothly decays from 1 to 0 between rcut_smth and rcut,
            and remains 0 beyond rcut, with shape n_edge.
        owner
            The owner index of the neighbor to reduce on.
        num_owner : int
            The total number of the owner.
        nb : int
            The number of batches.
        nloc : int
            The number of local atoms.
        scale_factor : float
            The scale factor to apply after reduce.
        axis_neuron
            Size of the submatrix.

        Returns
        -------
        grrg
            Atomic invariant rep, with shape nb x nloc x (axis_neuron x e_dim)
        """
        # nb x nloc x 3 x e_dim
        h2g2 = self._cal_hg_dynamic(
            flat_edge_ebd,
            flat_h2,
            flat_sw,
            owner,
            num_owner,
            nb,
            nloc,
            scale_factor,
        )
        # nb x nloc x (axis x e_dim)
        grrg = self._cal_grrg(h2g2, axis_neuron)
        return grrg

    def optim_angle_update(
        self,
        angle_ebd: torch.Tensor,
        node_ebd: torch.Tensor,
        edge_ebd: torch.Tensor,
        feat: str = "edge",
    ) -> torch.Tensor:
        if feat == "edge":
            assert self.edge_angle_linear1 is not None
            matrix, bias = self.edge_angle_linear1.matrix, self.edge_angle_linear1.bias
        elif feat == "angle":
            assert self.angle_self_linear is not None
            matrix, bias = self.angle_self_linear.matrix, self.angle_self_linear.bias
        else:
            raise NotImplementedError
        assert bias is not None

        angle_dim = angle_ebd.shape[-1]
        node_dim = node_ebd.shape[-1]
        edge_dim = edge_ebd.shape[-1]
        # angle_dim, node_dim, edge_dim, edge_dim
        sub_angle, sub_node, sub_edge_ik, sub_edge_ij = torch.split(
            matrix, [angle_dim, node_dim, edge_dim, edge_dim]
        )

        # nf * nloc * a_sel * a_sel * angle_dim
        sub_angle_update = torch.matmul(angle_ebd, sub_angle)
        # nf * nloc * angle_dim
        sub_node_update = torch.matmul(node_ebd, sub_node)
        # nf * nloc * a_nnei * angle_dim
        sub_edge_update_ik = torch.matmul(edge_ebd, sub_edge_ik)
        sub_edge_update_ij = torch.matmul(edge_ebd, sub_edge_ij)

        result_update = (
            bias
            + sub_node_update.unsqueeze(2).unsqueeze(3)
            + sub_edge_update_ik.unsqueeze(2)
            + sub_edge_update_ij.unsqueeze(3)
            + sub_angle_update
        )
        return result_update

    def optim_angle_update_dynamic(
        self,
        flat_angle_ebd: torch.Tensor,
        node_ebd: torch.Tensor,
        flat_edge_ebd: torch.Tensor,
        n2a_index: torch.Tensor,
        eij2a_index: torch.Tensor,
        eik2a_index: torch.Tensor,
        feat: str = "edge",
    ) -> torch.Tensor:
        if feat == "edge":
            matrix, bias = self.edge_angle_linear1.matrix, self.edge_angle_linear1.bias
        elif feat == "angle":
            matrix, bias = self.angle_self_linear.matrix, self.angle_self_linear.bias
        else:
            raise NotImplementedError
        nf, nloc, node_dim = node_ebd.shape
        edge_dim = flat_edge_ebd.shape[-1]
        angle_dim = flat_angle_ebd.shape[-1]
        # angle_dim, node_dim, edge_dim, edge_dim
        sub_angle, sub_node, sub_edge_ik, sub_edge_ij = torch.split(
            matrix, [angle_dim, node_dim, edge_dim, edge_dim]
        )

        # n_angle * angle_dim
        sub_angle_update = torch.matmul(flat_angle_ebd, sub_angle)

        # nf * nloc * angle_dim
        sub_node_update = torch.matmul(node_ebd, sub_node)
        # n_angle * angle_dim
        sub_node_update = torch.index_select(
            sub_node_update.reshape(nf * nloc, sub_node_update.shape[-1]), 0, n2a_index
        )

        # n_edge * angle_dim
        sub_edge_update_ik = torch.matmul(flat_edge_ebd, sub_edge_ik)
        sub_edge_update_ij = torch.matmul(flat_edge_ebd, sub_edge_ij)
        # n_angle * angle_dim
        sub_edge_update_ik = torch.index_select(sub_edge_update_ik, 0, eik2a_index)
        sub_edge_update_ij = torch.index_select(sub_edge_update_ij, 0, eij2a_index)

        result_update = (
            bias
            + sub_node_update
            + sub_edge_update_ik
            + sub_edge_update_ij
            + sub_angle_update
        )
        return result_update

    def optim_edge_update(
        self,
        node_ebd: torch.Tensor,
        node_ebd_ext: torch.Tensor,
        edge_ebd: torch.Tensor,
        nlist: torch.Tensor,
        feat: str = "node",
    ) -> torch.Tensor:
        if feat == "node":
            matrix, bias = self.node_edge_linear.matrix, self.node_edge_linear.bias
        elif feat == "edge":
            matrix, bias = self.edge_self_linear.matrix, self.edge_self_linear.bias
        else:
            raise NotImplementedError
        assert bias is not None

        node_dim = node_ebd.shape[-1]
        edge_dim = edge_ebd.shape[-1]
        # node_dim, node_dim, edge_dim
        node, node_ext, edge = torch.split(matrix, [node_dim, node_dim, edge_dim])

        # nf * nloc * node/edge_dim
        sub_node_update = torch.matmul(node_ebd, node)
        # nf * nall * node/edge_dim
        sub_node_ext_update = torch.matmul(node_ebd_ext, node_ext)
        # nf * nloc * nnei * node/edge_dim
        sub_node_ext_update = _make_nei_g1(sub_node_ext_update, nlist)
        # nf * nloc * nnei * node/edge_dim
        sub_edge_update = torch.matmul(edge_ebd, edge)

        result_update = (
            bias + sub_node_update.unsqueeze(2) + sub_edge_update + sub_node_ext_update
        )
        return result_update

    def optim_edge_update_dynamic(
        self,
        node_ebd: torch.Tensor,
        node_ebd_ext: torch.Tensor,
        flat_edge_ebd: torch.Tensor,
        n2e_index: torch.Tensor,
        n_ext2e_index: torch.Tensor,
        feat: str = "node",
    ) -> torch.Tensor:
        if feat == "node":
            matrix, bias = self.node_edge_linear.matrix, self.node_edge_linear.bias
        elif feat == "edge":
            matrix, bias = self.edge_self_linear.matrix, self.edge_self_linear.bias
        else:
            raise NotImplementedError
        assert bias is not None
        nf, nall, node_dim = node_ebd_ext.shape
        _, nloc, _ = node_ebd.shape
        edge_dim = flat_edge_ebd.shape[-1]
        # node_dim, node_dim, edge_dim
        node, node_ext, edge = torch.split(matrix, [node_dim, node_dim, edge_dim])

        # nf * nloc * node/edge_dim
        sub_node_update = torch.matmul(node_ebd, node)
        # n_edge * node/edge_dim
        sub_node_update = torch.index_select(
            sub_node_update.reshape(nf * nloc, sub_node_update.shape[-1]), 0, n2e_index
        )

        # nf * nall * node/edge_dim
        sub_node_ext_update = torch.matmul(node_ebd_ext, node_ext)
        # n_edge * node/edge_dim
        sub_node_ext_update = torch.index_select(
            sub_node_ext_update.reshape(nf * nall, sub_node_update.shape[-1]),
            0,
            n_ext2e_index,
        )

        # n_edge * node/edge_dim
        sub_edge_update = torch.matmul(flat_edge_ebd, edge)

        result_update = bias + sub_node_update + sub_edge_update + sub_node_ext_update
        return result_update

    def forward(
        self,
        node_ebd_ext: torch.Tensor,  # nf x nall x n_dim [OR] nf x nloc x n_dim when not parallel_mode
        edge_ebd: torch.Tensor,  # nf x nloc x nnei x e_dim
        h2: torch.Tensor,  # nf x nloc x nnei x 3
        angle_ebd: torch.Tensor,  # nf x nloc x a_nnei x a_nnei x a_dim
        nlist: torch.Tensor,  # nf x nloc x nnei
        nlist_mask: torch.Tensor,  # nf x nloc x nnei
        sw: torch.Tensor,  # switch func, nf x nloc x nnei
        a_nlist: torch.Tensor,  # nf x nloc x a_nnei
        a_nlist_mask: torch.Tensor,  # nf x nloc x a_nnei
        a_sw: torch.Tensor,  # switch func, nf x nloc x a_nnei
        edge_index: torch.Tensor,  # n_edge x 2
        angle_index: torch.Tensor,  # n_angle x 3
    ):
        """RepFlow层的前向传播函数
        
        这是RepFlow层的核心函数，实现了完整的消息传递机制：
        1. 节点表征更新：自更新 + 对称化 + 边消息传递
        2. 边表征更新：自更新 + 角度消息传递
        3. 角度表征更新：自更新（如果启用）
        
        整个过程保证旋转等变性和置换不变性。

        Parameters
        ----------
        node_ebd_ext : nf x nall x n_dim
            扩展节点嵌入，包含所有原子的表征
        edge_ebd : nf x nloc x nnei x e_dim
            边嵌入，表示原子对之间的相互作用
        h2 : nf x nloc x nnei x 3
            旋转等变的原子对通道，用于保持旋转等变性
        angle_ebd : nf x nloc x a_nnei x a_nnei x a_dim
            角度嵌入，表示三体相互作用
        nlist : nf x nloc x nnei
            邻居列表，填充的邻居设为0
        nlist_mask : nf x nloc x nnei
            邻居列表掩码，真实邻居为1，否则为0
        sw : nf x nloc x nnei
            开关函数，用于平滑截断
        a_nlist : nf x nloc x a_nnei
            角度邻居列表，填充的邻居设为0
        a_nlist_mask : nf x nloc x a_nnei
            角度邻居列表掩码，真实邻居为1，否则为0
        a_sw : nf x nloc x a_nnei
            角度开关函数，用于平滑截断
        edge_index : n_edge x 2 (动态选择时使用)
            n2e_index : n_edge - 从节点(i)到边(ij)的广播索引
            n_ext2e_index : n_edge - 从扩展节点(j)到边(ij)的广播索引
        angle_index : n_angle x 3 (动态选择时使用)
            n2a_index : n_angle - 从扩展节点(j)到角度(ijk)的广播索引
            eij2a_index : n_angle - 从边(ij)到角度(ijk)的广播索引
            eik2a_index : n_angle - 从边(ik)到角度(ijk)的广播索引

        Returns
        -------
        n_updated: nf x nloc x n_dim
            更新后的节点表征
        e_updated: nf x nloc x nnei x e_dim
            更新后的边表征
        a_updated: nf x nloc x a_nnei x a_nnei x a_dim
            更新后的角度表征
        """
        # =============================================================================
        # 1. 输入预处理和形状检查
        # =============================================================================
        nb, nloc, nnei = nlist.shape  # 批次数、局部原子数、近邻数
        nall = node_ebd_ext.shape[1]  # 扩展节点数
        node_ebd = node_ebd_ext[:, :nloc, :]  # 局部节点表征
        n_edge = int(nlist_mask.sum().item())  # 实际边数量
        assert (nb, nloc) == node_ebd.shape[:2]
        
        # 检查h2张量的形状（根据是否使用动态选择）
        if not self.use_dynamic_sel:  # 不使用动态选择
            assert (nb, nloc, nnei, 3) == h2.shape
        else:  # 使用动态选择
            assert (n_edge, 3) == h2.shape
        del a_nlist  # 可能在未来使用

        # 提取索引信息
        n2e_index, n_ext2e_index = edge_index[:, 0], edge_index[:, 1]  # 节点到边的索引、扩展节点到边的索引
        n2a_index, eij2a_index, eik2a_index = (
            angle_index[:, 0],  # 节点到角度的索引
            angle_index[:, 1],  # 边ij到角度的索引
            angle_index[:, 2],  # 边ik到角度的索引
        )

        # =============================================================================
        # 2. 构建邻居节点嵌入
        # =============================================================================
        # 构建近邻节点嵌入：每个边对应的邻居节点特征
        # 形状: nb x nloc x nnei x n_dim [OR] n_edge x n_dim
        nei_node_ebd = (
            _make_nei_g1(node_ebd_ext, nlist)  # 标准模式：通过邻居列表构建
            if not self.use_dynamic_sel
            else torch.index_select(  # 动态模式：通过索引选择
                node_ebd_ext.reshape(-1, self.n_dim), 0, n_ext2e_index
            )
        )

        # =============================================================================
        # 3. 初始化更新列表（用于残差连接）
        # =============================================================================
        n_update_list: list[torch.Tensor] = [node_ebd]   # 节点更新列表，包含原始节点
        e_update_list: list[torch.Tensor] = [edge_ebd]   # 边更新列表，包含原始边  
        a_update_list: list[torch.Tensor] = [angle_ebd]  # 角度更新列表，包含原始角度

        # =============================================================================
        # 4. 节点表征更新
        # =============================================================================
        
        # 4.1 节点自更新：通过MLP处理节点特征
        node_self_mlp = self.act(self.node_self_mlp(node_ebd))
        n_update_list.append(node_self_mlp)
        
        # 4.2 节点对称化更新：基于GRRG不变量的几何信息
        # （从边信息聚合到节点）
        # 这部分实现了旋转不变性，通过对称化操作处理几何信息
        node_sym_list: list[torch.Tensor] = []
        
        # 计算边嵌入的GRRG不变量
        # symmetrization_op: 边嵌入 → 对称化不变量 [nf, nloc, axis*e_dim]
        node_sym_list.append(
            self.symmetrization_op(
                edge_ebd,
                h2,
                nlist_mask,
                sw,
                self.axis_neuron,
            )
            if not self.use_dynamic_sel
            else self.symmetrization_op_dynamic(  # 动态选择模式
                edge_ebd,
                h2,
                sw,
                owner=n2e_index,
                num_owner=nb * nloc,
                nb=nb,
                nloc=nloc,
                scale_factor=self.dynamic_e_sel ** (-0.5),
                axis_neuron=self.axis_neuron,
            )
        )
        
        # 计算邻居节点的GRRG不变量
        node_sym_list.append(
            self.symmetrization_op(
                nei_node_ebd,
                h2,
                nlist_mask,
                sw,
                self.axis_neuron,
            )
            if not self.use_dynamic_sel
            else self.symmetrization_op_dynamic(  # 动态选择模式
                nei_node_ebd,
                h2,
                sw,
                owner=n2e_index,
                num_owner=nb * nloc,
                nb=nb,
                nloc=nloc,
                scale_factor=self.dynamic_e_sel ** (-0.5),
                axis_neuron=self.axis_neuron,
            )
        )
        
        # 将两个GRRG不变量拼接并通过MLP处理
        node_sym = self.act(self.node_sym_linear(torch.cat(node_sym_list, dim=-1)))
        n_update_list.append(node_sym)

        # =============================================================================
        # 5. 节点-边消息传递
        # =============================================================================
        
        # 5.1 构建边信息（用于消息传递）
        if not self.optim_update:
            if not self.use_dynamic_sel:
                # 标准模式：拼接节点、邻居节点、边信息
                # 形状: nb x nloc x nnei x (n_dim * 2 + e_dim)
                edge_info = torch.cat(
                    [
                        torch.tile(node_ebd.unsqueeze(-2), [1, 1, self.nnei, 1]),  # 中心节点特征
                        nei_node_ebd,  # 邻居节点特征
                        edge_ebd,      # 边特征
                    ],
                    dim=-1,
                )
            else:
                # 动态模式：通过索引选择
                # 形状: n_edge x (n_dim * 2 + e_dim)
                edge_info = torch.cat(
                    [
                        torch.index_select(
                            node_ebd.reshape(-1, self.n_dim), 0, n2e_index
                        ),  # 中心节点特征
                        nei_node_ebd,  # 邻居节点特征
                        edge_ebd,      # 边特征
                    ],
                    dim=-1,
                )
        else:
            edge_info = None  # 优化模式不需要预构建边信息

        # 5.2 节点-边消息传递
        # 通过边信息更新节点表征，实现消息传递
        if not self.optim_update:
            assert edge_info is not None
            node_edge_update = self.act(
                self.node_edge_linear(edge_info)
            ) * sw.unsqueeze(-1)  # 应用开关函数
        else:
            # 优化模式：直接计算，避免构建大型中间张量
            node_edge_update = self.act(
                self.optim_edge_update(
                    node_ebd,
                    node_ebd_ext,
                    edge_ebd,
                    nlist,
                    "node",
                )
                if not self.use_dynamic_sel
                else self.optim_edge_update_dynamic(
                    node_ebd,
                    node_ebd_ext,
                    edge_ebd,
                    n2e_index,
                    n_ext2e_index,
                    "node",
                )
            ) * sw.unsqueeze(-1)  # 应用开关函数
            
        # 5.3 聚合边消息到节点
        # 将来自所有邻居的消息聚合到中心节点
        node_edge_update = (
            (torch.sum(node_edge_update, dim=-2) / self.nnei)  # 标准模式：平均聚合
            if not self.use_dynamic_sel
            else (  # 动态模式：使用聚合函数
                aggregate(
                    node_edge_update,
                    n2e_index,
                    average=False,
                    num_owner=nb * nloc,
                ).reshape(nb, nloc, node_edge_update.shape[-1])
                / self.dynamic_e_sel
            )
        )

        # 5.4 处理多头边消息（如果启用）
        if self.n_multi_edge_message > 1:
            # 将边消息重塑为多头格式
            # 形状: nb x nloc x h x n_dim
            node_edge_update_mul_head = node_edge_update.view(
                nb, nloc, self.n_multi_edge_message, self.n_dim
            )
            # 将每个头作为独立的更新项
            for head_index in range(self.n_multi_edge_message):
                n_update_list.append(node_edge_update_mul_head[..., head_index, :])
        else:
            n_update_list.append(node_edge_update)
            
        # 5.5 更新节点表征（使用残差连接）
        n_updated = self.list_update(n_update_list, "node")

        # =============================================================================
        # 6. 边表征更新 G2
        # =============================================================================
        
        # 6.1 边自更新：通过边信息更新边表征
        if not self.optim_update:
            assert edge_info is not None
            edge_self_update = self.act(self.edge_self_linear(edge_info))
        else:
            # 优化模式：直接计算边更新
            edge_self_update = self.act(
                self.optim_edge_update(
                    node_ebd,
                    node_ebd_ext,
                    edge_ebd,
                    nlist,
                    "edge",
                )
                if not self.use_dynamic_sel
                else self.optim_edge_update_dynamic(
                    node_ebd,
                    node_ebd_ext,
                    edge_ebd,
                    n2e_index,
                    n_ext2e_index,
                    "edge",
                )
            )
        e_update_list.append(edge_self_update)

        # =============================================================================
        # 7. 角度表征更新（如果启用）
        # =============================================================================
        
        if self.update_angle:
            assert self.angle_self_linear is not None
            assert self.edge_angle_linear1 is not None
            assert self.edge_angle_linear2 is not None
            
            # 7.1 角度信息压缩（如果启用）
            # 为了减少计算量，可以对节点和边特征进行压缩
            if self.a_compress_rate != 0:
                if not self.a_compress_use_split:
                    # 使用MLP进行压缩
                    assert self.a_compress_n_linear is not None
                    assert self.a_compress_e_linear is not None
                    node_ebd_for_angle = self.a_compress_n_linear(node_ebd)
                    edge_ebd_for_angle = self.a_compress_e_linear(edge_ebd)
                else:
                    # 使用分割方式：取前几个维度
                    node_ebd_for_angle = node_ebd[..., : self.n_a_compress_dim]
                    edge_ebd_for_angle = edge_ebd[..., : self.e_a_compress_dim]
            else:
                # 不压缩：直接使用原始特征
                node_ebd_for_angle = node_ebd
                edge_ebd_for_angle = edge_ebd

            # 7.2 处理角度边特征
            if not self.use_dynamic_sel:
                # 标准模式：截取角度邻居数量并应用掩码
                # 形状: nb x nloc x a_nnei x e_dim
                edge_ebd_for_angle = edge_ebd_for_angle[..., : self.a_sel, :]
                edge_ebd_for_angle = torch.where(
                    a_nlist_mask.unsqueeze(-1), edge_ebd_for_angle, 0.0
                )
                
            # 7.3 构建角度信息（用于角度消息传递）
            if not self.optim_update:
                # 标准模式：构建角度信息张量
                
                # 节点信息：为每个角度对复制节点特征
                # 形状: nb x nloc x a_nnei x a_nnei x n_dim [OR] n_angle x n_dim
                node_for_angle_info = (
                    torch.tile(
                        node_ebd_for_angle.unsqueeze(2).unsqueeze(2),
                        (1, 1, self.a_sel, self.a_sel, 1),
                    )
                    if not self.use_dynamic_sel
                    else torch.index_select(
                        node_ebd_for_angle.reshape(-1, self.n_a_compress_dim),
                        0,
                        n2a_index,
                    )
                )

                # 边信息k：为每个角度对复制边ik特征
                # 形状: nb x nloc x (a_nnei) x a_nnei x e_dim [OR] n_angle x e_dim
                edge_for_angle_k = (
                    torch.tile(
                        edge_ebd_for_angle.unsqueeze(2), (1, 1, self.a_sel, 1, 1)
                    )
                    if not self.use_dynamic_sel
                    else torch.index_select(edge_ebd_for_angle, 0, eik2a_index)
                )
                
                # 边信息j：为每个角度对复制边ij特征
                # 形状: nb x nloc x a_nnei x (a_nnei) x e_dim [OR] n_angle x e_dim
                edge_for_angle_j = (
                    torch.tile(
                        edge_ebd_for_angle.unsqueeze(3), (1, 1, 1, self.a_sel, 1)
                    )
                    if not self.use_dynamic_sel
                    else torch.index_select(edge_ebd_for_angle, 0, eij2a_index)
                )
                
                # 拼接边信息：边ik和边ij的特征
                # 形状: nb x nloc x a_nnei x a_nnei x (e_dim + e_dim) [OR] n_angle x (e_dim + e_dim)
                edge_for_angle_info = torch.cat(
                    [edge_for_angle_k, edge_for_angle_j], dim=-1
                )
                
                # 构建完整的角度信息列表
                angle_info_list = [angle_ebd]
                angle_info_list.append(node_for_angle_info)
                angle_info_list.append(edge_for_angle_info)
                
                # 拼接所有角度信息
                # 形状: nb x nloc x a_nnei x a_nnei x (a + n_dim + e_dim*2) or (a + a/c + a/c)
                # [OR] n_angle x (a + n_dim + e_dim*2) or (a + a/c + a/c)
                angle_info = torch.cat(angle_info_list, dim=-1)
            else:
                angle_info = None  # 优化模式不需要预构建角度信息

            # 7.4 边-角度消息传递 G2 MP
            # 通过角度信息更新边表征，实现三体相互作用建模
            if not self.optim_update:
                assert angle_info is not None
                edge_angle_update = self.act(self.edge_angle_linear1(angle_info))
            else:
                # 优化模式：直接计算角度消息
                edge_angle_update = self.act(
                    self.optim_angle_update(
                        angle_ebd,
                        node_ebd_for_angle,
                        edge_ebd_for_angle,
                        "edge",
                    )
                    if not self.use_dynamic_sel
                    else self.optim_angle_update_dynamic(
                        angle_ebd,
                        node_ebd_for_angle,
                        edge_ebd_for_angle,
                        n2a_index,
                        eij2a_index,
                        eik2a_index,
                        "edge",
                    )
                )

            # 7.5 处理角度消息的权重和聚合
            if not self.use_dynamic_sel:
                # 标准模式：应用角度开关函数并聚合
                # 形状: nb x nloc x a_nnei x a_nnei x e_dim
                weighted_edge_angle_update = (
                    a_sw[..., None, None] * a_sw[..., None, :, None] * edge_angle_update
                )
                
                # 沿角度维度聚合：从角度对聚合到边
                # 形状: nb x nloc x a_nnei x e_dim
                reduced_edge_angle_update = torch.sum(
                    weighted_edge_angle_update, dim=-2
                ) / (self.a_sel**0.5)
                
                # 填充到完整的边维度：将角度边扩展到所有边
                # 形状: nb x nloc x nnei x e_dim
                padding_edge_angle_update = torch.concat(
                    [
                        reduced_edge_angle_update,
                        torch.zeros(
                            [nb, nloc, self.nnei - self.a_sel, self.e_dim],
                            dtype=edge_ebd.dtype,
                            device=edge_ebd.device,
                        ),
                    ],
                    dim=2,
                )
            else:
                # 动态模式：使用聚合函数
                # 形状: n_angle x e_dim
                weighted_edge_angle_update = edge_angle_update * a_sw.unsqueeze(-1)
                
                # 聚合角度消息到边
                # 形状: n_edge x e_dim
                padding_edge_angle_update = aggregate(
                    weighted_edge_angle_update,
                    eij2a_index,
                    average=False,
                    num_owner=n_edge,
                ) / (self.dynamic_a_sel**0.5)

            # 7.6 平滑边更新处理（向后兼容）
            if not self.smooth_edge_update:
                # 注意：此功能将在未来版本中弃用
                # 不支持动态索引，但会通过检查
                if self.use_dynamic_sel:
                    raise NotImplementedError(
                        "smooth_edge_update must be True when use_dynamic_sel is True!"
                    )
                # 构建完整的掩码
                full_mask = torch.concat(
                    [
                        a_nlist_mask,
                        torch.zeros(
                            [nb, nloc, self.nnei - self.a_sel],
                            dtype=a_nlist_mask.dtype,
                            device=a_nlist_mask.device,
                        ),
                    ],
                    dim=-1,
                )
                # 应用掩码：在非角度边位置使用原始边特征
                padding_edge_angle_update = torch.where(
                    full_mask.unsqueeze(-1), padding_edge_angle_update, edge_ebd
                )
                
            # 7.7 边角度消息的进一步处理
            e_update_list.append(
                self.act(self.edge_angle_linear2(padding_edge_angle_update))
            )
            
            # 7.8 更新边表征（使用残差连接）
            e_updated = self.list_update(e_update_list, "edge")

            # 7.9 角度自更新消息
            # 通过角度信息更新角度表征
            if not self.optim_update:
                assert angle_info is not None
                angle_self_update = self.act(self.angle_self_linear(angle_info))
            else:
                # 优化模式：直接计算角度自更新
                angle_self_update = self.act(
                    self.optim_angle_update(
                        angle_ebd,
                        node_ebd_for_angle,
                        edge_ebd_for_angle,
                        "angle",
                    )
                    if not self.use_dynamic_sel
                    else self.optim_angle_update_dynamic(
                        angle_ebd,
                        node_ebd_for_angle,
                        edge_ebd_for_angle,
                        n2a_index,
                        eij2a_index,
                        eik2a_index,
                        "angle",
                    )
                )
            a_update_list.append(angle_self_update)
        else:
            # 如果未启用角度更新，只更新边表征
            e_updated = self.list_update(e_update_list, "edge")

        # =============================================================================
        # 8. 最终更新和返回
        # =============================================================================
        
        # 更新角度表征（使用残差连接）
        a_updated = self.list_update(a_update_list, "angle")
        
        # 返回更新后的所有表征
        return n_updated, e_updated, a_updated

    @torch.jit.export
    def list_update_res_avg(
        self,
        update_list: list[torch.Tensor],
    ) -> torch.Tensor:
        """残差平均更新策略
        
        这是三种残差连接策略之一，实现方式为：
        u = (u₀ + u₁ + u₂ + ... + uₙ) / √(n+1)
        
        其中：
        - u₀: 原始表征（第0个元素）
        - u₁, u₂, ..., uₙ: 各种更新项
        - n+1: 总项数（包括原始项）
        
        这种策略的特点：
        1. 所有更新项等权重相加
        2. 使用√(n+1)进行归一化，保持数值稳定性
        3. 适合更新项重要性相近的情况
        
        Parameters
        ----------
        update_list : list[torch.Tensor]
            更新列表，第一个元素是原始表征，后续是各种更新项
            例如：[node_ebd, node_self_update, node_sym_update, node_edge_update]
        
        Returns
        -------
        torch.Tensor
            更新后的表征，形状与原始表征相同
        """
        nitem = len(update_list)
        uu = update_list[0]  # 从原始表征开始
        for ii in range(1, nitem):
            uu = uu + update_list[ii]  # 累加所有更新项
        return uu / (float(nitem) ** 0.5)  # 归一化：除以√(n+1)

    @torch.jit.export
    def list_update_res_incr(self, update_list: list[torch.Tensor]) -> torch.Tensor:
        """残差增量更新策略
        
        这是三种残差连接策略之一，实现方式为：
        u = u₀ + (u₁ + u₂ + ... + uₙ) / √n
        
        其中：
        - u₀: 原始表征（第0个元素）
        - u₁, u₂, ..., uₙ: 各种更新项
        - n: 更新项数量（不包括原始项）
        
        这种策略的特点：
        1. 原始表征保持完整权重
        2. 更新项作为增量，按√n归一化
        3. 适合更新项作为对原始表征的修正的情况
        4. 与ResNet的残差连接思想最接近
        
        Parameters
        ----------
        update_list : list[torch.Tensor]
            更新列表，第一个元素是原始表征，后续是各种更新项
            例如：[node_ebd, node_self_update, node_sym_update, node_edge_update]
        
        Returns
        -------
        torch.Tensor
            更新后的表征，形状与原始表征相同
        """
        nitem = len(update_list)
        uu = update_list[0]  # 从原始表征开始
        # 计算更新项的归一化因子：1/√(n-1)，其中n-1是更新项数量
        scale = 1.0 / (float(nitem - 1) ** 0.5) if nitem > 1 else 0.0
        for ii in range(1, nitem):
            uu = uu + scale * update_list[ii]  # 添加归一化的更新项
        return uu

    @torch.jit.export
    def list_update_res_residual(
        self, update_list: list[torch.Tensor], update_name: str = "node"
    ) -> torch.Tensor:
        """残差权重更新策略（最灵活的策略）
        
        这是三种残差连接策略中最灵活的一种，实现方式为：
        u = u₀ + r₁*u₁ + r₂*u₂ + ... + rₙ*uₙ
        
        其中：
        - u₀: 原始表征（第0个元素）
        - u₁, u₂, ..., uₙ: 各种更新项
        - r₁, r₂, ..., rₙ: 可学习的残差权重参数
        
        这种策略的特点：
        1. 每个更新项都有独立的可学习权重
        2. 模型可以自动学习哪些更新项更重要
        3. 权重参数在训练过程中自适应调整
        4. 提供最大的表达能力和灵活性
        5. 需要更多的参数和计算资源
        
        权重初始化：
        - 初始标准差由update_residual参数控制（默认0.1）
        - 初始化方式由update_residual_init控制（"const"或"norm"）
        
        Parameters
        ----------
        update_list : list[torch.Tensor]
            更新列表，第一个元素是原始表征，后续是各种更新项
            例如：[node_ebd, node_self_update, node_sym_update, node_edge_update]
        update_name : str
            更新类型，决定使用哪组残差权重：
            - "node": 使用self.n_residual权重
            - "edge": 使用self.e_residual权重  
            - "angle": 使用self.a_residual权重
        
        Returns
        -------
        torch.Tensor
            更新后的表征，形状与原始表征相同
        """
        nitem = len(update_list)
        uu = update_list[0]  # 从原始表征开始
        
        # 根据更新类型选择对应的残差权重
        # 注意：这里使用"make jit happy"的写法，避免动态属性访问
        if update_name == "node":
            # 使用节点残差权重：n_residual = [r₁, r₂, r₃, ...]
            for ii, vv in enumerate(self.n_residual):
                uu = uu + vv * update_list[ii + 1]  # u₀ + r₁*u₁ + r₂*u₂ + ...
        elif update_name == "edge":
            # 使用边残差权重：e_residual = [r₁, r₂, r₃, ...]
            for ii, vv in enumerate(self.e_residual):
                uu = uu + vv * update_list[ii + 1]  # u₀ + r₁*u₁ + r₂*u₂ + ...
        elif update_name == "angle":
            # 使用角度残差权重：a_residual = [r₁, r₂, r₃, ...]
            for ii, vv in enumerate(self.a_residual):
                uu = uu + vv * update_list[ii + 1]  # u₀ + r₁*u₁ + r₂*u₂ + ...
        else:
            raise NotImplementedError(f"Unknown update_name: {update_name}")
        return uu

    @torch.jit.export
    def list_update(
        self, update_list: list[torch.Tensor], update_name: str = "node"
    ) -> torch.Tensor:
        """残差更新策略的统一入口函数
        
        根据配置的update_style参数，选择相应的残差连接策略：
        
        1. "res_avg": 残差平均策略
           - 公式: u = (u₀ + u₁ + u₂ + ... + uₙ) / √(n+1)
           - 特点: 所有项等权重，简单稳定
           - 适用: 更新项重要性相近的情况
        
        2. "res_incr": 残差增量策略  
           - 公式: u = u₀ + (u₁ + u₂ + ... + uₙ) / √n
           - 特点: 原始项保持完整权重，更新项作为增量
           - 适用: 更新项作为对原始表征的修正
        
        3. "res_residual": 残差权重策略（默认）
           - 公式: u = u₀ + r₁*u₁ + r₂*u₂ + ... + rₙ*uₙ
           - 特点: 每个更新项有独立可学习权重
           - 适用: 需要最大灵活性的情况
        
        策略选择建议：
        - 简单任务: 使用"res_avg"或"res_incr"
        - 复杂任务: 使用"res_residual"（默认）
        - 计算资源受限: 避免"res_residual"
        
        Parameters
        ----------
        update_list : list[torch.Tensor]
            更新列表，第一个元素是原始表征，后续是各种更新项
        update_name : str
            更新类型，用于"res_residual"策略选择对应的权重组
            - "node": 节点更新
            - "edge": 边更新
            - "angle": 角度更新
        
        Returns
        -------
        torch.Tensor
            更新后的表征，形状与原始表征相同
        """
        if self.update_style == "res_avg":
            return self.list_update_res_avg(update_list)
        elif self.update_style == "res_incr":
            return self.list_update_res_incr(update_list)
        elif self.update_style == "res_residual":
            return self.list_update_res_residual(update_list, update_name=update_name)
        else:
            raise RuntimeError(f"unknown update style {self.update_style}")

    def serialize(self) -> dict:
        """Serialize the networks to a dict.

        Returns
        -------
        dict
            The serialized networks.
        """
        data = {
            "@class": "RepFlowLayer",
            "@version": 2,
            "e_rcut": self.e_rcut,
            "e_rcut_smth": self.e_rcut_smth,
            "e_sel": self.e_sel,
            "a_rcut": self.a_rcut,
            "a_rcut_smth": self.a_rcut_smth,
            "a_sel": self.a_sel,
            "ntypes": self.ntypes,
            "n_dim": self.n_dim,
            "e_dim": self.e_dim,
            "a_dim": self.a_dim,
            "a_compress_rate": self.a_compress_rate,
            "a_compress_e_rate": self.a_compress_e_rate,
            "a_compress_use_split": self.a_compress_use_split,
            "n_multi_edge_message": self.n_multi_edge_message,
            "axis_neuron": self.axis_neuron,
            "activation_function": self.activation_function,
            "update_angle": self.update_angle,
            "update_style": self.update_style,
            "update_residual": self.update_residual,
            "update_residual_init": self.update_residual_init,
            "precision": self.precision,
            "optim_update": self.optim_update,
            "smooth_edge_update": self.smooth_edge_update,
            "use_dynamic_sel": self.use_dynamic_sel,
            "sel_reduce_factor": self.sel_reduce_factor,
            "node_self_mlp": self.node_self_mlp.serialize(),
            "node_sym_linear": self.node_sym_linear.serialize(),
            "node_edge_linear": self.node_edge_linear.serialize(),
            "edge_self_linear": self.edge_self_linear.serialize(),
        }
        if self.update_angle:
            data.update(
                {
                    "edge_angle_linear1": self.edge_angle_linear1.serialize(),
                    "edge_angle_linear2": self.edge_angle_linear2.serialize(),
                    "angle_self_linear": self.angle_self_linear.serialize(),
                }
            )
            if self.a_compress_rate != 0 and not self.a_compress_use_split:
                data.update(
                    {
                        "a_compress_n_linear": self.a_compress_n_linear.serialize(),
                        "a_compress_e_linear": self.a_compress_e_linear.serialize(),
                    }
                )
        if self.update_style == "res_residual":
            data.update(
                {
                    "@variables": {
                        "n_residual": [to_numpy_array(t) for t in self.n_residual],
                        "e_residual": [to_numpy_array(t) for t in self.e_residual],
                        "a_residual": [to_numpy_array(t) for t in self.a_residual],
                    }
                }
            )
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "RepFlowLayer":
        """Deserialize the networks from a dict.

        Parameters
        ----------
        data : dict
            The dict to deserialize from.
        """
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 2, 1)
        data.pop("@class")
        update_angle = data["update_angle"]
        a_compress_rate = data["a_compress_rate"]
        a_compress_use_split = data["a_compress_use_split"]
        node_self_mlp = data.pop("node_self_mlp")
        node_sym_linear = data.pop("node_sym_linear")
        node_edge_linear = data.pop("node_edge_linear")
        edge_self_linear = data.pop("edge_self_linear")
        edge_angle_linear1 = data.pop("edge_angle_linear1", None)
        edge_angle_linear2 = data.pop("edge_angle_linear2", None)
        angle_self_linear = data.pop("angle_self_linear", None)
        a_compress_n_linear = data.pop("a_compress_n_linear", None)
        a_compress_e_linear = data.pop("a_compress_e_linear", None)
        update_style = data["update_style"]
        variables = data.pop("@variables", {})
        n_residual = variables.get("n_residual", data.pop("n_residual", []))
        e_residual = variables.get("e_residual", data.pop("e_residual", []))
        a_residual = variables.get("a_residual", data.pop("a_residual", []))

        obj = cls(**data)
        obj.node_self_mlp = MLPLayer.deserialize(node_self_mlp)
        obj.node_sym_linear = MLPLayer.deserialize(node_sym_linear)
        obj.node_edge_linear = MLPLayer.deserialize(node_edge_linear)
        obj.edge_self_linear = MLPLayer.deserialize(edge_self_linear)

        if update_angle:
            assert isinstance(edge_angle_linear1, dict)
            assert isinstance(edge_angle_linear2, dict)
            assert isinstance(angle_self_linear, dict)
            obj.edge_angle_linear1 = MLPLayer.deserialize(edge_angle_linear1)
            obj.edge_angle_linear2 = MLPLayer.deserialize(edge_angle_linear2)
            obj.angle_self_linear = MLPLayer.deserialize(angle_self_linear)
            if a_compress_rate != 0 and not a_compress_use_split:
                assert isinstance(a_compress_n_linear, dict)
                assert isinstance(a_compress_e_linear, dict)
                obj.a_compress_n_linear = MLPLayer.deserialize(a_compress_n_linear)
                obj.a_compress_e_linear = MLPLayer.deserialize(a_compress_e_linear)

        if update_style == "res_residual":
            for ii, t in enumerate(obj.n_residual):
                t.data = to_torch_tensor(n_residual[ii])
            for ii, t in enumerate(obj.e_residual):
                t.data = to_torch_tensor(e_residual[ii])
            for ii, t in enumerate(obj.a_residual):
                t.data = to_torch_tensor(a_residual[ii])
        return obj
