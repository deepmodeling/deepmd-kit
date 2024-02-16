# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
)

import torch

from deepmd.utils.env_mat_stat import EnvMatStat as BaseEnvMatStat
from deepmd.utils.env_mat_stat import (
    StatItem,
)


class EnvMatStat(BaseEnvMatStat):
    def compute_stat(self, env_mat: Dict[str, torch.Tensor]) -> Dict[str, StatItem]:
        """Compute the statistics of the environment matrix for a single system.

        Parameters
        ----------
        env_mat : torch.Tensor
            The environment matrix.

        Returns
        -------
        Dict[str, StatItem]
            The statistics of the environment matrix.
        """
        stats = {}
        for kk, vv in env_mat.items():
            stats[kk] = StatItem(
                number=vv.numel(),
                mean=vv.mean().item(),
                squared_mean=torch.square(vv).mean().item(),
            )
        return stats
