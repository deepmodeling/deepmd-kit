# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)

import numpy as np

from deepmd.dpmodel.utils import (
    AtomExcludeMask,
    PairExcludeMask,
)

from .make_base_atomic_model import (
    make_base_atomic_model,
)

BaseAtomicModel_ = make_base_atomic_model(np.ndarray)


class BaseAtomicModel(BaseAtomicModel_):
    def __init__(
        self,
        atom_exclude_types: List[int] = [],
        pair_exclude_types: List[Tuple[int, int]] = [],
    ):
        super().__init__()
        self.reinit_atom_exclude(atom_exclude_types)
        self.reinit_pair_exclude(pair_exclude_types)

    def reinit_atom_exclude(
        self,
        exclude_types: List[int] = [],
    ):
        self.atom_exclude_types = exclude_types
        if exclude_types == []:
            self.atom_excl = None
        else:
            self.atom_excl = AtomExcludeMask(self.get_ntypes(), self.atom_exclude_types)

    def reinit_pair_exclude(
        self,
        exclude_types: List[Tuple[int, int]] = [],
    ):
        self.pair_exclude_types = exclude_types
        if exclude_types == []:
            self.pair_excl = None
        else:
            self.pair_excl = PairExcludeMask(self.get_ntypes(), self.pair_exclude_types)

    def forward_common_atomic(
        self,
        extended_coord: np.ndarray,
        extended_atype: np.ndarray,
        nlist: np.ndarray,
        mapping: Optional[np.ndarray] = None,
        fparam: Optional[np.ndarray] = None,
        aparam: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        nf, nloc, nnei = nlist.shape
        nf, nall = extended_atype.shape
        atype = extended_atype[:, :nloc]
        if self.pair_excl is not None:
            pair_mask = self.pair_excl.build_type_exclude_mask(nlist, extended_atype)
            # exclude neighbors in the nlist
            nlist = np.where(pair_mask == 1, nlist, -1)

        ret_dict = self.forward_atomic(
            extended_coord,
            extended_atype,
            nlist,
            mapping=mapping,
            fparam=fparam,
            aparam=aparam,
        )

        if self.atom_excl is not None:
            atom_mask = self.atom_excl.build_type_exclude_mask(atype)
            for kk in ret_dict.keys():
                ret_dict[kk] = ret_dict[kk] * atom_mask[:, :, None]

        return ret_dict

    def serialize(self) -> dict:
        return {
            "atom_exclude_types": self.atom_exclude_types,
            "pair_exclude_types": self.pair_exclude_types,
        }
