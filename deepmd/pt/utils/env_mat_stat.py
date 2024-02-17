# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterator,
    List,
)

import numpy as np
import torch

from deepmd.common import (
    get_hash,
)
from deepmd.pt.model.descriptor.env_mat import (
    prod_env_mat_se_a,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
)
from deepmd.utils.env_mat_stat import EnvMatStat as BaseEnvMatStat
from deepmd.utils.env_mat_stat import (
    StatItem,
)

if TYPE_CHECKING:
    from deepmd.pt.model.descriptor import (
        DescriptorBlock,
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
                sum=vv.sum().item(),
                squared_sum=torch.square(vv).sum().item(),
            )
        return stats


class EnvMatStatSeA(EnvMatStat):
    """Environmental matrix statistics for the se_a environemntal matrix.

    Parameters
    ----------
    descriptor : DescriptorBlock
        The descriptor of the model.
    """

    def __init__(self, descriptor: "DescriptorBlock"):
        super().__init__()
        self.descriptor = descriptor

    def iter(
        self, data: List[Dict[str, torch.Tensor]]
    ) -> Iterator[Dict[str, StatItem]]:
        """Get the iterator of the environment matrix.

        Parameters
        ----------
        data : List[Dict[str, torch.Tensor]]
            The environment matrix.

        Yields
        ------
        Dict[str, StatItem]
            The statistics of the environment matrix.
        """
        zero_mean = torch.zeros(
            self.descriptor.get_ntypes(),
            self.descriptor.get_nsel(),
            4,
            dtype=env.GLOBAL_PT_FLOAT_PRECISION,
            device=env.DEVICE,
        )
        one_stddev = torch.ones(
            self.descriptor.get_ntypes(),
            self.descriptor.get_nsel(),
            4,
            dtype=env.GLOBAL_PT_FLOAT_PRECISION,
            device=env.DEVICE,
        )
        for system in data:
            coord, atype, box, natoms = (
                system["coord"],
                system["atype"],
                system["box"],
                system["natoms"],
            )
            (
                extended_coord,
                extended_atype,
                mapping,
                nlist,
            ) = extend_input_and_build_neighbor_list(
                coord,
                atype,
                self.descriptor.get_rcut(),
                self.descriptor.get_sel(),
                distinguish_types=self.descriptor.distinguish_types(),
                box=box,
            )
            env_mat, _, _ = prod_env_mat_se_a(
                extended_coord,
                nlist,
                atype,
                zero_mean,
                one_stddev,
                self.descriptor.get_rcut(),
                # TODO: export rcut_smth from DescriptorBlock
                self.descriptor.rcut_smth,
            )
            env_mat = env_mat.view(
                coord.shape[0], coord.shape[1], self.descriptor.get_nsel(), 4
            )
            env_mats = {}

            if "real_natoms_vec" not in system:
                end_indexes = torch.cumsum(natoms[0, 2:], 0)
                start_indexes = torch.cat(
                    [
                        torch.zeros(1, dtype=torch.int32, device=env.DEVICE),
                        end_indexes[:-1],
                    ]
                )
                for type_i in range(self.descriptor.get_ntypes()):
                    dd = env_mat[
                        :, start_indexes[type_i] : end_indexes[type_i], :, :
                    ]  # all descriptors for this element
                    env_mats[f"r_{type_i}"] = dd[:, :, :, :1]
                    env_mats[f"a_{type_i}"] = dd[:, :, :, 1:]
                    yield self.compute_stat(env_mats)
            else:
                for frame_item in range(env_mat.shape[0]):
                    dd_ff = env_mat[frame_item]
                    atype_frame = atype[frame_item]
                    for type_i in range(self.descriptor.get_ntypes()):
                        type_idx = atype_frame == type_i
                        dd = dd_ff[type_idx]
                        dd = dd.reshape([-1, 4])  # typen_atoms * nnei, 4
                        env_mats[f"r_{type_i}"] = dd[:, :1]
                        env_mats[f"a_{type_i}"] = dd[:, 1:]
                        yield self.compute_stat(env_mats)

    def get_hash(self) -> str:
        """Get the hash of the environment matrix.

        Returns
        -------
        str
            The hash of the environment matrix.
        """
        return get_hash(
            {
                "type": "se_a",
                "ntypes": self.descriptor.get_ntypes(),
                "rcut": round(self.descriptor.get_rcut(), 2),
                "rcut_smth": round(self.descriptor.rcut_smth, 2),
                "nsel": self.descriptor.get_nsel(),
                "sel": self.descriptor.get_sel(),
                "distinguish_types": self.descriptor.distinguish_types(),
            }
        )

    def __call__(self):
        avgs = self.get_avg()
        stds = self.get_std()

        all_davg = []
        all_dstd = []
        for type_i in range(self.descriptor.get_ntypes()):
            davgunit = [[avgs[f"r_{type_i}"], 0, 0, 0]]
            dstdunit = [
                [
                    stds[f"r_{type_i}"],
                    stds[f"a_{type_i}"],
                    stds[f"a_{type_i}"],
                    stds[f"a_{type_i}"],
                ]
            ]
            davg = np.tile(davgunit, [self.descriptor.get_nsel(), 1])
            dstd = np.tile(dstdunit, [self.descriptor.get_nsel(), 1])
            all_davg.append(davg)
            all_dstd.append(dstd)
        mean = np.stack(all_davg)
        stddev = np.stack(all_dstd)
        return mean, stddev
