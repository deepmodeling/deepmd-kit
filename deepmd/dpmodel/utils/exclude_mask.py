# SPDX-License-Identifier: LGPL-3.0-or-later

import array_api_compat
import numpy as np

from deepmd.dpmodel.array_api import (
    Array,
    xp_take_along_axis,
    xp_take_first_n,
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

    def get_exclude_types(self) -> list[int]:
        return self.exclude_types

    def get_type_mask(self) -> Array:
        return self.type_mask

    def build_type_exclude_mask(
        self,
        atype: Array,
    ) -> Array:
        """Compute type exclusion mask for atoms.

        Parameters
        ----------
        atype
            The atom types. shape: nf x natom (dense) or N (graph / flat)

        Returns
        -------
        mask
            The type exclusion mask for atoms, same shape as ``atype``.
            Element being 0 if the type is excluded, otherwise being 1.

        """
        xp = array_api_compat.array_namespace(atype)
        lead = atype.shape  # (nf, natom) dense | (N,) graph
        return xp.reshape(
            xp.take(
                xp.asarray(self.type_mask[...], device=array_api_compat.device(atype)),
                xp.reshape(atype, (-1,)),
                axis=0,
            ),
            lead,
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

    def get_exclude_types(self) -> list[tuple[int, int]]:
        return self.exclude_types

    def build_type_exclude_mask(
        self,
        nlist: Array,
        atype_ext: Array,
    ) -> Array:
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
            [
                atype_ext,
                self.ntypes
                * xp.ones(
                    [nf, 1],
                    dtype=atype_ext.dtype,
                    device=array_api_compat.device(atype_ext),
                ),
            ],
            axis=-1,
        )
        type_i = xp.reshape(xp_take_first_n(atype_ext, 1, nloc), (nf, nloc)) * (
            self.ntypes + 1
        )
        # Map -1 entries to nall (the virtual atom index in ae)
        nlist_for_type = xp.where(nlist == -1, xp.full_like(nlist, nall), nlist)
        # Gather neighbor types using xp_take_along_axis along axis=1.
        # This avoids flat (nf*(nall+1),) indexing that creates Ne(nall, nloc)
        # constraints in torch.export, breaking NoPbc (nall == nloc).
        nlist_for_gather = xp.reshape(nlist_for_type, (nf, nloc * nnei))
        type_j = xp_take_along_axis(ae, nlist_for_gather, axis=1)
        type_j = xp.reshape(type_j, (nf, nloc, nnei))
        type_ij = type_i[:, :, None] + type_j
        # (nf * nloc * nnei,)
        type_ij_flat = xp.reshape(type_ij, (-1,))
        mask = xp.reshape(
            xp.take(
                xp.asarray(self.type_mask[...], device=array_api_compat.device(nlist)),
                type_ij_flat,
            ),
            (nf, nloc, nnei),
        )
        return mask

    def build_edge_exclude_mask(self, edge_index: Array, atype: Array) -> Array:
        """Graph-native pair exclusion: per-edge keep mask (1 keep, 0 exclude).

        Parameters
        ----------
        edge_index
            (2, E) [src, dst]; src = neighbor, dst = center; into [0, N).
        atype
            (N,) flat local node types (clamped >= 0).

        Returns
        -------
        mask
            (E,) int. ``type_mask[atype[dst]*(ntypes+1) + atype[src]]``.

        """
        xp = array_api_compat.array_namespace(atype)
        if len(self.exclude_types) == 0:
            return xp.ones(
                (edge_index.shape[1],),
                dtype=xp.int32,
                device=array_api_compat.device(atype),
            )
        src_t = xp.take(atype, edge_index[0, :], axis=0)
        dst_t = xp.take(atype, edge_index[1, :], axis=0)
        type_ij = dst_t * (self.ntypes + 1) + src_t
        return xp.take(
            xp.asarray(self.type_mask[...], device=array_api_compat.device(atype)),
            type_ij,
            axis=0,
        )

    def __contains__(self, item: tuple[int, int]) -> bool:
        return item in self.exclude_types
