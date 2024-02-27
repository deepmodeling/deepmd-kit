# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    List,
    Tuple,
)

import numpy as np


class AtomExcludeMask:
    """Computes the type exclusion mask for atoms."""

    def __init__(
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
        # (ntypes)
        self.type_mask = self.type_mask.reshape([-1])

    def build_type_exclude_mask(
        self,
        atype: np.ndarray,
    ):
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
        return self.type_mask[atype].reshape(nf, natom)


class PairExcludeMask:
    """Computes the type exclusion mask for atom pairs."""

    def __init__(
        self,
        ntypes: int,
        exclude_types: List[Tuple[int, int]] = [],
    ):
        self.ntypes = ntypes
        self.exclude_types = set()
        for tt in exclude_types:
            assert len(tt) == 2
            self.exclude_types.add((tt[0], tt[1]))
            self.exclude_types.add((tt[1], tt[0]))
        # ntypes + 1 for nlist masks
        self.type_mask = np.array(
            [
                [
                    1 if (tt_i, tt_j) not in self.exclude_types else 0
                    for tt_i in range(ntypes + 1)
                ]
                for tt_j in range(ntypes + 1)
            ],
            dtype=np.int32,
        )
        # (ntypes+1 x ntypes+1)
        self.type_mask = self.type_mask.reshape([-1])

    def build_type_exclude_mask(
        self,
        nlist: np.ndarray,
        atype_ext: np.ndarray,
    ):
        """Compute type exclusion mask for atom pairs.

        Parameters
        ----------
        nlist
            The neighbor list. shape: nf x nloc x nnei
        atype_ext
            The extended aotm types. shape: nf x nall

        Returns
        -------
        mask
            The type exclusion mask for pair atoms of shape: nf x nloc x nnei.
            Element [ff,ii,jj] being 0 if type(ii), type(nlist[ff,ii,jj]) is excluded,
            otherwise being 1.

        """
        if len(self.exclude_types) == 0:
            # safely return 1 if nothing is excluded.
            return np.ones_like(nlist, dtype=np.int32)
        nf, nloc, nnei = nlist.shape
        nall = atype_ext.shape[1]
        # add virtual atom of type ntypes. nf x nall+1
        ae = np.concatenate(
            [atype_ext, self.ntypes * np.ones([nf, 1], dtype=atype_ext.dtype)], axis=-1
        )
        type_i = atype_ext[:, :nloc].reshape(nf, nloc) * (self.ntypes + 1)
        # nf x nloc x nnei
        index = np.where(nlist == -1, nall, nlist).reshape(nf, nloc * nnei)
        type_j = np.take_along_axis(ae, index, axis=1).reshape(nf, nloc, nnei)
        type_ij = type_i[:, :, None] + type_j
        # nf x (nloc x nnei)
        type_ij = type_ij.reshape(nf, nloc * nnei)
        mask = self.type_mask[type_ij].reshape(nf, nloc, nnei)
        return mask

    def __contains__(self, item):
        return item in self.exclude_types
