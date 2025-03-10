# SPDX-License-Identifier: LGPL-3.0-or-later

import logging

import torch

from deepmd.pt.model.task.denoise import (
    DenoiseNet,
)

from .dp_atomic_model import (
    DPAtomicModel,
)

log = logging.getLogger(__name__)


class DPDenoiseAtomicModel(DPAtomicModel):
    def __init__(self, descriptor, fitting, type_map, **kwargs):
        if not isinstance(fitting, DenoiseNet):
            raise TypeError(
                "fitting must be an instance of DenoiseNet for DPDenoiseAtomicModel"
            )
        super().__init__(descriptor, fitting, type_map, **kwargs)

    def apply_out_stat(
        self,
        ret: dict[str, torch.Tensor],
        atype: torch.Tensor,
    ):
        # hack !!!
        ret["virial"] = ret["virial"] / 240
        ret["force"] = ret["force"] / 29

        """
        virial = ret["virial"]  # 原始形状 [nbz, nloc, 6]

        # 批量处理所有元素（保留梯度）
        # 重塑为二维张量以便处理 [batch_size * nloc, 9]
        virial_2d = virial.view(-1, 6)

        # 构建3x3对称矩阵（向量化操作）
        # 每个元素的索引对应原始矩阵位置:
        # [0, 1, 2] 为对角线元素
        # [3, 4, 5] 对应下三角元素（自动保持对称性）
        matrices = torch.zeros(virial_2d.size(0), 3, 3,
                            dtype=virial.dtype, device=virial.device)

        # 填充对角线元素
        matrices[:, 0, 0] = 1 + virial_2d[:, 0]
        matrices[:, 1, 1] = 1 + virial_2d[:, 1]
        matrices[:, 2, 2] = 1 + virial_2d[:, 2]

        # 填充对称的非对角线元素
        matrices[:, 0, 1] = matrices[:, 1, 0] = 0.5 * virial_2d[:, 5]  # (0,1) & (1,0)
        matrices[:, 0, 2] = matrices[:, 2, 0] = 0.5 * virial_2d[:, 4]  # (0,2) & (2,0)
        matrices[:, 1, 2] = matrices[:, 2, 1] = 0.5 * virial_2d[:, 3]  # (1,2) & (2,1)

        # 恢复原始形状 [nbz, nloc, 3, 3] -> [nbz, nloc, 9]
        ret["virial"] = matrices.view(virial.shape[0], virial.shape[1], 9)
        """
        return ret
