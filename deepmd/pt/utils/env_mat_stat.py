# SPDX-License-Identifier: LGPL-3.0-or-later
from collections.abc import (
    Iterator,
)
from typing import (
    TYPE_CHECKING,
    Union,
)

import numpy as np
import torch

from deepmd.common import (
    get_hash,
)
from deepmd.pt.model.descriptor.env_mat import (
    prod_env_mat,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.exclude_mask import (
    PairExcludeMask,
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
    def compute_stat(self, env_mat: dict[str, torch.Tensor]) -> dict[str, StatItem]:
        """Compute the statistics of the environment matrix for a single system.

        Parameters
        ----------
        env_mat : torch.Tensor
            The environment matrix.

        Returns
        -------
        dict[str, StatItem]
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


class EnvMatStatSe(EnvMatStat):
    """Environmental matrix statistics for the se_a/se_r environmental matrix.

    Parameters
    ----------
    descriptor : DescriptorBlock
        The descriptor of the model.
    """

    def __init__(self, descriptor: "DescriptorBlock") -> None:
        super().__init__()
        self.descriptor = descriptor
        self.last_dim = (
            self.descriptor.ndescrpt // self.descriptor.nnei
        )  # se_r=1, se_a=4

    def iter(
        self, data: list[dict[str, Union[torch.Tensor, list[tuple[int, int]]]]]
    ) -> Iterator[dict[str, StatItem]]:
        """Get the iterator of the environment matrix.

        Parameters
        ----------
        data : list[dict[str, Union[torch.Tensor, list[tuple[int, int]]]]]
            The data.

        Yields
        ------
        dict[str, StatItem]
            The statistics of the environment matrix.
        """
        zero_mean = torch.zeros(
            self.descriptor.get_ntypes(),
            self.descriptor.get_nsel(),
            self.last_dim,
            dtype=env.GLOBAL_PT_FLOAT_PRECISION,
            device=env.DEVICE,
        )
        one_stddev = torch.ones(
            self.descriptor.get_ntypes(),
            self.descriptor.get_nsel(),
            self.last_dim,
            dtype=env.GLOBAL_PT_FLOAT_PRECISION,
            device=env.DEVICE,
        )
        if self.last_dim == 4:
            radial_only = False
        elif self.last_dim == 1:
            radial_only = True
        else:
            raise ValueError(
                "last_dim should be 1 for raial-only or 4 for full descriptor."
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
                mixed_types=self.descriptor.mixed_types(),
                box=box,
            )
            env_mat, _, _ = prod_env_mat(
                extended_coord,
                nlist,
                atype,
                zero_mean,
                one_stddev,
                self.descriptor.get_rcut(),
                self.descriptor.get_rcut_smth(),
                radial_only,
                protection=self.descriptor.get_env_protection(),
            )
            # apply excluded_types
            exclude_mask = self.descriptor.emask(nlist, extended_atype)
            env_mat *= exclude_mask.unsqueeze(-1)
            # reshape to nframes * nloc at the atom level,
            # so nframes/mixed_type do not matter
            env_mat = env_mat.view(
                coord.shape[0] * coord.shape[1],
                self.descriptor.get_nsel(),
                self.last_dim,
            )
            atype = atype.view(coord.shape[0] * coord.shape[1])
            # (1, nloc) eq (ntypes, 1), so broadcast is possible
            # shape: (ntypes, nloc)
            type_idx = torch.eq(
                atype.view(1, -1),
                torch.arange(
                    self.descriptor.get_ntypes(), device=env.DEVICE, dtype=torch.int32
                ).view(-1, 1),
            )
            if "pair_exclude_types" in system:
                # shape: (1, nloc, nnei)
                exclude_mask = PairExcludeMask(
                    self.descriptor.get_ntypes(), system["pair_exclude_types"]
                )(nlist, extended_atype).view(1, coord.shape[0] * coord.shape[1], -1)
                # shape: (ntypes, nloc, nnei)
                type_idx = torch.logical_and(type_idx.unsqueeze(-1), exclude_mask)
            for type_i in range(self.descriptor.get_ntypes()):
                dd = env_mat[type_idx[type_i]]
                dd = dd.reshape([-1, self.last_dim])  # typen_atoms * unmasked_nnei, 4
                env_mats = {}
                env_mats[f"r_{type_i}"] = dd[:, :1]
                if self.last_dim == 4:
                    env_mats[f"a_{type_i}"] = dd[:, 1:]
                yield self.compute_stat(env_mats)

    def get_hash(self) -> str:
        """Get the hash of the environment matrix.

        Returns
        -------
        str
            The hash of the environment matrix.
        """
        dscpt_type = "se_a" if self.last_dim == 4 else "se_r"
        return get_hash(
            {
                "type": dscpt_type,
                "ntypes": self.descriptor.get_ntypes(),
                "rcut": round(self.descriptor.get_rcut(), 2),
                "rcut_smth": round(self.descriptor.rcut_smth, 2),
                "nsel": self.descriptor.get_nsel(),
                "sel": self.descriptor.get_sel(),
                "mixed_types": self.descriptor.mixed_types(),
            }
        )

    def __call__(self):
        avgs = self.get_avg()
        stds = self.get_std()

        all_davg = []
        all_dstd = []

        for type_i in range(self.descriptor.get_ntypes()):
            if self.last_dim == 4:
                davgunit = [[avgs[f"r_{type_i}"], 0, 0, 0]]
                dstdunit = [
                    [
                        stds[f"r_{type_i}"],
                        stds[f"a_{type_i}"],
                        stds[f"a_{type_i}"],
                        stds[f"a_{type_i}"],
                    ]
                ]
            elif self.last_dim == 1:
                davgunit = [[avgs[f"r_{type_i}"]]]
                dstdunit = [
                    [
                        stds[f"r_{type_i}"],
                    ]
                ]
            davg = np.tile(davgunit, [self.descriptor.get_nsel(), 1])
            dstd = np.tile(dstdunit, [self.descriptor.get_nsel(), 1])
            all_davg.append(davg)
            all_dstd.append(dstd)

        mean = np.stack(all_davg)
        stddev = np.stack(all_dstd)
        return mean, stddev
