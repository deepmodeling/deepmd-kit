# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
)

import torch


@torch.jit.script
def aggregate(
    data: torch.Tensor,
    owners: torch.Tensor,
    average: bool = True,
    num_owner: Optional[int] = None,
) -> torch.Tensor:
    """
    根据所有者索引聚合数据行
    
    在DPA3的RepFlow层中，这个函数用于将边或角度的特征聚合到对应的节点上。
    例如：将多条边的特征聚合到中心原子节点上。

    Parameters
    ----------
    data : torch.Tensor
        要聚合的数据张量 [n_row, feature_dim]
        例如：边特征 [n_edge, e_dim] 或角度特征 [n_angle, a_dim]
    owners : torch.Tensor
        指定每行数据的所有者索引 [n_row]
        例如：边特征中每个边对应的中心原子索引
    average : bool, optional
        如果为True，对行进行平均；如果为False，对行进行求和
        默认 = True
    num_owner : Optional[int], optional
        所有者的数量，当owners张量中不包含最大索引时需要指定
        默认 = None

    Returns
    -------
    torch.Tensor
        聚合后的输出 [num_owner, feature_dim]
        例如：聚合后的节点特征 [n_atoms, feature_dim]
    """
    # 计算每个所有者的数据行数（用于平均化）
    if num_owner is None or average:
        # 使用bincount统计每个所有者拥有的数据行数
        bin_count = torch.bincount(owners)
        # 避免除零错误：将0替换为1
        bin_count = bin_count.where(bin_count != 0, bin_count.new_ones(1))
        # 如果指定的num_owner与bin_count长度不匹配，进行填充
        if (num_owner is not None) and (bin_count.shape[0] != num_owner):
            difference = num_owner - bin_count.shape[0]
            bin_count = torch.cat([bin_count, bin_count.new_ones(difference)])
        else:
            num_owner = bin_count.shape[0]
    else:
        bin_count = None

    # 初始化输出张量
    output = data.new_zeros([num_owner, data.shape[1]])
    # 使用index_add_将数据按所有者索引累加
    output = output.index_add_(0, owners, data)
    # 如果需要平均化，除以每个所有者的数据行数
    if average:
        assert bin_count is not None
        output = (output.T / bin_count).T
    return output


@torch.jit.script
def get_graph_index(
    nlist: torch.Tensor,
    nlist_mask: torch.Tensor,
    a_nlist_mask: torch.Tensor,
    nall: int,
    use_loc_mapping: bool = True,
):
    """
    获取边图和角度图的索引映射，用于`aggregate`或`index_select`操作
    
    在DPA3的动态选择模式下，这个函数构建了图神经网络所需的索引映射：
    1. 边图索引：连接中心原子和邻居原子的边
    2. 角度图索引：连接中心原子和两个邻居原子形成的角度
    
    这些索引用于高效的消息传递和特征聚合操作。

    Parameters
    ----------
    nlist : torch.Tensor
        邻居列表 [nf, nloc, nnei]
        填充的邻居设置为0
    nlist_mask : torch.Tensor
        邻居列表的掩码 [nf, nloc, nnei]
        真实邻居为1，否则为0
    a_nlist_mask : torch.Tensor
        角度邻居列表的掩码 [nf, nloc, a_nnei]
        用于角度计算的真实邻居为1，否则为0
    nall : int
        扩展原子的总数
    use_loc_mapping : bool, optional
        是否使用局部索引映射，默认 = True

    Returns
    -------
    edge_index : torch.Tensor
        边图索引 [n_edge, 2]
        - n2e_index: 从节点(i)到边(ij)的广播索引，或从边(ij)到节点(i)的归约索引
        - n_ext2e_index: 从扩展节点(j)到边(ij)的广播索引
    angle_index : torch.Tensor
        角度图索引 [n_angle, 3]
        - n2a_index: 从扩展节点(j)到角度(ijk)的广播索引
        - eij2a_index: 从扩展边(ij)到角度(ijk)的广播索引，或从角度(ijk)到边(ij)的归约索引
        - eik2a_index: 从扩展边(ik)到角度(ijk)的广播索引
    """
    # 获取张量维度信息
    nf, nloc, nnei = nlist.shape
    _, _, a_nnei = a_nlist_mask.shape
    
    # 构建角度掩码：a_nnei x a_nnei 的3D掩码，用于角度计算
    # 只有两个邻居都存在时，才形成有效的角度
    a_nlist_mask_3d = a_nlist_mask[:, :, :, None] & a_nlist_mask[:, :, None, :]
    
    # 计算有效边和角度的数量
    n_edge = nlist_mask.sum().item()
    # n_angle = a_nlist_mask_3d.sum().item()  # 注释掉，因为后面会重新计算

    # =============================================================================
    # 1. 构建边图索引 (atom graph)
    # =============================================================================
    
    # 1.1 节点(i)到边(ij)的索引映射
    # 创建局部原子索引：每个帧的每个局部原子都有唯一索引
    nlist_loc_index = torch.arange(0, nf * nloc, dtype=nlist.dtype, device=nlist.device)
    # 扩展为 [nf, nloc, nnei] 形状，每个邻居都对应同一个中心原子
    n2e_index = nlist_loc_index.reshape(nf, nloc, 1).expand(-1, -1, nnei)
    # 只保留真实邻居对应的索引
    n2e_index = n2e_index[nlist_mask]  # 形状: [n_edge]

    # 1.2 扩展节点(j)到边(ij)的索引映射
    # 计算帧偏移量：每帧的原子索引需要加上帧偏移
    frame_shift = torch.arange(0, nf, dtype=nlist.dtype, device=nlist.device) * (
        nall if not use_loc_mapping else nloc
    )
    # 将邻居列表转换为全局索引
    shifted_nlist = nlist + frame_shift[:, None, None]
    # 只保留真实邻居对应的索引
    n_ext2e_index = shifted_nlist[nlist_mask]  # 形状: [n_edge]

    # =============================================================================
    # 2. 构建角度图索引 (angle graph)
    # =============================================================================
    
    # 2.1 节点(i)到角度(ijk)的索引映射
    # 扩展为 [nf, nloc, a_nnei, a_nnei] 形状
    n2a_index = nlist_loc_index.reshape(nf, nloc, 1, 1).expand(-1, -1, a_nnei, a_nnei)
    # 只保留有效角度对应的索引
    n2a_index = n2a_index[a_nlist_mask_3d]  # 形状: [n_angle]

    # 2.2 边(ij)到角度(ijk)的索引映射
    # 为每条边分配唯一ID
    edge_id = torch.arange(0, n_edge, dtype=nlist.dtype, device=nlist.device)
    # 创建边索引张量，形状与nlist相同
    edge_index = torch.zeros([nf, nloc, nnei], dtype=nlist.dtype, device=nlist.device)
    edge_index[nlist_mask] = edge_id
    # 只取前a_nnei个邻居，避免nnei x nnei的复杂度
    edge_index = edge_index[:, :, :a_nnei]
    
    # 2.3 边(ij)到角度(ijk)的索引：j边
    edge_index_ij = edge_index.unsqueeze(-1).expand(-1, -1, -1, a_nnei)
    eij2a_index = edge_index_ij[a_nlist_mask_3d]  # 形状: [n_angle]
    
    # 2.4 边(ik)到角度(ijk)的索引：k边
    edge_index_ik = edge_index.unsqueeze(-2).expand(-1, -1, a_nnei, -1)
    eik2a_index = edge_index_ik[a_nlist_mask_3d]  # 形状: [n_angle]

    # =============================================================================
    # 3. 返回索引张量
    # =============================================================================
    
    # 边图索引：[n_edge, 2] - [n2e_index, n_ext2e_index]
    edge_index = torch.cat(
        [n2e_index.unsqueeze(-1), n_ext2e_index.unsqueeze(-1)], dim=-1
    )
    
    # 角度图索引：[n_angle, 3] - [n2a_index, eij2a_index, eik2a_index]
    angle_index = torch.cat(
        [n2a_index.unsqueeze(-1), eij2a_index.unsqueeze(-1), eik2a_index.unsqueeze(-1)],
        dim=-1,
    )
    
    return edge_index, angle_index
