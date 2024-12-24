# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import paddle

from deepmd.pd.utils.utils import (
    to_paddle_tensor,
)


class AtomExcludeMask(paddle.nn.Layer):
    """Computes the type exclusion mask for atoms."""

    def __init__(
        self,
        ntypes: int,
        exclude_types: list[int] = [],
    ) -> None:
        super().__init__()
        self.reinit(ntypes, exclude_types)

    def reinit(
        self,
        ntypes: int,
        exclude_types: list[int] = [],
    ) -> None:
        self.ntypes = ntypes
        self.exclude_types = exclude_types
        self.type_mask = np.array(
            [1 if tt_i not in self.exclude_types else 0 for tt_i in range(ntypes)],
            dtype=np.int32,
        )
        self.type_mask = to_paddle_tensor(self.type_mask).reshape([-1])

    def get_exclude_types(self):
        return self.exclude_types

    def get_type_mask(self):
        return self.type_mask

    def forward(
        self,
        atype: paddle.Tensor,
    ) -> paddle.Tensor:
        """Compute type exclusion mask for atoms.

        Parameters
        ----------
        atype
            The extended atom types. shape: nf x natom

        Returns
        -------
        mask
            The type exclusion mask for atoms. shape: nf x natom
            Element [ff,ii] being 0 if type(ii) is excluded,
            otherwise being 1.

        """
        nf, natom = atype.shape
        return self.type_mask[atype].reshape([nf, natom]).to(atype.place)


class PairExcludeMask(paddle.nn.Layer):
    """Computes the type exclusion mask for atom pairs."""

    def __init__(
        self,
        ntypes: int,
        exclude_types: list[tuple[int, int]] = [],
    ) -> None:
        super().__init__()
        self.reinit(ntypes, exclude_types)

    def reinit(
        self,
        ntypes: int,
        exclude_types: list[tuple[int, int]] = [],
    ) -> None:
        self.ntypes = ntypes
        self._exclude_types: set[tuple[int, int]] = set()
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
        self.type_mask = to_paddle_tensor(self.type_mask).reshape([-1])
        self.no_exclusion = len(self._exclude_types) == 0

    def get_exclude_types(self):
        return self._exclude_types

    # may have a better place for this method...
    def forward(
        self,
        nlist: paddle.Tensor,
        atype_ext: paddle.Tensor,
    ) -> paddle.Tensor:
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
            return paddle.ones_like(nlist, dtype=paddle.int32).to(device=nlist.place)
        nf, nloc, nnei = nlist.shape
        nall = atype_ext.shape[1]
        # add virtual atom of type ntypes. nf x nall+1
        ae = paddle.concat(
            [
                atype_ext,
                self.ntypes
                * paddle.ones([nf, 1], dtype=atype_ext.dtype).to(atype_ext.place),
            ],
            axis=-1,
        )
        type_i = atype_ext[:, :nloc].reshape([nf, nloc]) * (self.ntypes + 1)
        # nf x nloc x nnei
        index = paddle.where(nlist == -1, nall, nlist).reshape([nf, nloc * nnei])
        type_j = paddle.take_along_axis(ae, axis=1, indices=index).reshape(
            [nf, nloc, nnei]
        )
        type_ij = type_i[:, :, None] + type_j
        # nf x (nloc x nnei)
        type_ij = type_ij.reshape([nf, nloc * nnei])
        mask = (
            self.type_mask[type_ij]
            .reshape([nf, nloc, nnei])
            .to(atype_ext.place)
            .astype("bool")
        )
        return mask
