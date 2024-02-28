# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    List,
    Set,
    Tuple,
)

import numpy as np
import torch

from deepmd.pt.utils.utils import (
    to_torch_tensor,
)


class AtomExcludeMask(torch.nn.Module):
    """Computes the type exclusion mask for atoms."""

    def __init__(
        self,
        ntypes: int,
        exclude_types: List[int] = [],
    ):
        super().__init__()
        self.reinit(ntypes, exclude_types)

    def reinit(
        self,
        ntypes: int,
        exclude_types: List[int] = [],
    ):
        self.ntypes = ntypes
        self.exclude_types = exclude_types
        self.type_mask = np.array(
            [1 if tt_i not in self.exclude_types else 0 for tt_i in range(ntypes)],
            dtype=np.int32,
        )
        self.type_mask = to_torch_tensor(self.type_mask).view([-1])

    def forward(
        self,
        atype: torch.Tensor,
    ) -> torch.Tensor:
        """Compute type exclusion mask for atoms.

        Parameters
        ----------
        atype
            The extended aotm types. shape: nf x natom

        Returns
        -------
        mask
            The type exclusion mask for atoms. shape: nf x natom
            Element [ff,ii] being 0 if type(ii) is excluded,
            otherwise being 1.

        """
        nf, natom = atype.shape
        return self.type_mask[atype].view(nf, natom)


class PairExcludeMask(torch.nn.Module):
    """Computes the type exclusion mask for atom pairs."""

    def __init__(
        self,
        ntypes: int,
        exclude_types: List[Tuple[int, int]] = [],
    ):
        super().__init__()
        self.reinit(ntypes, exclude_types)

    def reinit(
        self,
        ntypes: int,
        exclude_types: List[Tuple[int, int]] = [],
    ):
        self.ntypes = ntypes
        self._exclude_types: Set[Tuple[int, int]] = set()
        for tt in exclude_types:
            assert len(tt) == 2
            self._exclude_types.add((tt[0], tt[1]))
            self._exclude_types.add((tt[1], tt[0]))
        # ntypes + 1 for nlist masks
        self.type_mask = np.array(
            [
                [
                    1 if (tt_i, tt_j) not in self._exclude_types else 0
                    for tt_i in range(ntypes + 1)
                ]
                for tt_j in range(ntypes + 1)
            ],
            dtype=np.int32,
        )
        # (ntypes+1 x ntypes+1)
        self.type_mask = to_torch_tensor(self.type_mask).view([-1])
        self.no_exclusion = len(self._exclude_types) == 0

    # may have a better place for this method...
    def forward(
        self,
        nlist: torch.Tensor,
        atype_ext: torch.Tensor,
    ) -> torch.Tensor:
        """Compute type exclusion mask.

        Parameters
        ----------
        nlist
            The neighbor list. shape: nf x nloc x nnei
        atype_ext
            The extended aotm types. shape: nf x nall

        Returns
        -------
        mask
            The type exclusion mask of shape: nf x nloc x nnei.
            Element [ff,ii,jj] being 0 if type(ii), type(nlist[ff,ii,jj]) is excluded,
            otherwise being 1.

        """
        if self.no_exclusion:
            # safely return 1 if nothing is excluded.
            return torch.ones_like(nlist, dtype=torch.int32, device=nlist.device)
        nf, nloc, nnei = nlist.shape
        nall = atype_ext.shape[1]
        # add virtual atom of type ntypes. nf x nall+1
        ae = torch.cat(
            [
                atype_ext,
                self.ntypes
                * torch.ones([nf, 1], dtype=atype_ext.dtype, device=atype_ext.device),
            ],
            dim=-1,
        )
        type_i = atype_ext[:, :nloc].view(nf, nloc) * (self.ntypes + 1)
        # nf x nloc x nnei
        index = torch.where(nlist == -1, nall, nlist).view(nf, nloc * nnei)
        type_j = torch.gather(ae, 1, index).view(nf, nloc, nnei)
        type_ij = type_i[:, :, None] + type_j
        # nf x (nloc x nnei)
        type_ij = type_ij.view(nf, nloc * nnei)
        mask = self.type_mask[type_ij].view(nf, nloc, nnei)
        return mask
