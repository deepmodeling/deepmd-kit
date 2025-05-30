# SPDX-License-Identifier: LGPL-3.0-or-later

import array_api_compat
import numpy as np

from deepmd.dpmodel.array_api import (
    xp_take_along_axis,
)


class AtomExcludeMask:
    """Computes the type exclusion mask for atoms."""

    def __init__(
        self,
        ntypes: int,
        exclude_types: list[int] = [],
    ) -> None:
        self.ntypes = ntypes
        self.exclude_types = exclude_types
        type_mask = np.array(
            [1 if tt_i not in self.exclude_types else 0 for tt_i in range(ntypes)],
            dtype=np.int32,
        )
        # (ntypes)
        self.type_mask = type_mask.reshape([-1])

    def get_exclude_types(self):
        return self.exclude_types

    def get_type_mask(self):
        return self.type_mask

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
        xp = array_api_compat.array_namespace(atype)
        nf, natom = atype.shape
        return xp.reshape(
            xp.take(self.type_mask[...], xp.reshape(atype, [-1]), axis=0),
            (nf, natom),
        )


class PairExcludeMask:
    """Computes the type exclusion mask for atom pairs."""

    def __init__(
        self,
        ntypes: int,
        exclude_types: list[tuple[int, int]] = [],
    ) -> None:
        self.ntypes = ntypes
        self.exclude_types = set()
        for tt in exclude_types:
            assert len(tt) == 2
            self.exclude_types.add((tt[0], tt[1]))
            self.exclude_types.add((tt[1], tt[0]))
        # ntypes + 1 for nlist masks
        type_mask = np.array(
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
        self.type_mask = type_mask.reshape([-1])

    def get_exclude_types(self):
        return self.exclude_types

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
        xp = array_api_compat.array_namespace(nlist, atype_ext)
        if len(self.exclude_types) == 0:
            # safely return 1 if nothing is excluded.
            return xp.ones_like(nlist, dtype=xp.int32)
        nf, nloc, nnei = nlist.shape
        nall = atype_ext.shape[1]
        # add virtual atom of type ntypes. nf x nall+1
        ae = xp.concat(
            [atype_ext, self.ntypes * xp.ones([nf, 1], dtype=atype_ext.dtype)], axis=-1
        )
        type_i = xp.reshape(atype_ext[:, :nloc], (nf, nloc)) * (self.ntypes + 1)
        # nf x nloc x nnei
        index = xp.reshape(
            xp.where(nlist == -1, xp.full_like(nlist, nall), nlist), (nf, nloc * nnei)
        )
        type_j = xp_take_along_axis(ae, index, axis=1)
        type_j = xp.reshape(type_j, (nf, nloc, nnei))
        type_ij = type_i[:, :, None] + type_j
        # nf x (nloc x nnei)
        type_ij = xp.reshape(type_ij, (nf, nloc * nnei))
        mask = xp.reshape(
            xp.take(self.type_mask[...], xp.reshape(type_ij, (-1,))),
            (nf, nloc, nnei),
        )
        return mask

    def __contains__(self, item) -> bool:
        return item in self.exclude_types
