# SPDX-License-Identifier: LGPL-3.0-or-later


from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)

import torch

from deepmd.dpmodel.atomic_model import (
    make_base_atomic_model,
)
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    OutputVariableDef,
)
from deepmd.pt.utils import (
    AtomExcludeMask,
    PairExcludeMask,
)
from deepmd.utils.path import (
    DPPath,
)

BaseAtomicModel_ = make_base_atomic_model(torch.Tensor)


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

    # to make jit happy...
    def make_atom_mask(
        self,
        atype: torch.Tensor,
    ) -> torch.Tensor:
        """The atoms with type < 0 are treated as virutal atoms,
        which serves as place-holders for multi-frame calculations
        with different number of atoms in different frames.

        Parameters
        ----------
        atype
            Atom types. >= 0 for real atoms <0 for virtual atoms.

        Returns
        -------
        mask
            True for real atoms and False for virutal atoms.

        """
        # supposed to be supported by all backends
        return atype >= 0

    def atomic_output_def(self) -> FittingOutputDef:
        old_def = self.fitting_output_def()
        old_list = list(old_def.get_data().values())
        return FittingOutputDef(
            old_list  # noqa:RUF005
            + [
                OutputVariableDef(
                    name="mask",
                    shape=[1],
                    reduciable=False,
                    r_differentiable=False,
                    c_differentiable=False,
                )
            ]
        )

    def forward_common_atomic(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        comm_dict: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """Common interface for atomic inference.

        This method accept extended coordinates, extended atom typs, neighbor list,
        and predict the atomic contribution of the fit property.

        Parameters
        ----------
        extended_coord
            extended coodinates, shape: nf x (nall x 3)
        extended_atype
            extended atom typs, shape: nf x nall
            for a type < 0 indicating the atomic is virtual.
        nlist
            neighbor list, shape: nf x nloc x nsel
        mapping
            extended to local index mapping, shape: nf x nall
        fparam
            frame parameters, shape: nf x dim_fparam
        aparam
            atomic parameter, shape: nf x nloc x dim_aparam

        Returns
        -------
        ret_dict
            dict of output atomic properties.
            should implement the definition of `fitting_output_def`.
            ret_dict["mask"] of shape nf x nloc will be provided.
            ret_dict["mask"][ff,ii] == 1 indicating the ii-th atom of the ff-th frame is real.
            ret_dict["mask"][ff,ii] == 0 indicating the ii-th atom of the ff-th frame is virtual.

        """
        _, nloc, _ = nlist.shape
        atype = extended_atype[:, :nloc]

        if self.pair_excl is not None:
            pair_mask = self.pair_excl(nlist, extended_atype)
            # exclude neighbors in the nlist
            nlist = torch.where(pair_mask == 1, nlist, -1)

        ext_atom_mask = self.make_atom_mask(extended_atype)
        ret_dict = self.forward_atomic(
            extended_coord,
            torch.where(ext_atom_mask, extended_atype, 0),
            nlist,
            mapping=mapping,
            fparam=fparam,
            aparam=aparam,
            comm_dict=comm_dict
        )

        # nf x nloc
        atom_mask = ext_atom_mask[:, :nloc].to(torch.int32)
        if self.atom_excl is not None:
            atom_mask *= self.atom_excl(atype)

        for kk in ret_dict.keys():
            out_shape = ret_dict[kk].shape
            ret_dict[kk] = (
                ret_dict[kk].reshape([out_shape[0], out_shape[1], -1])
                * atom_mask[:, :, None]
            ).view(out_shape)
        ret_dict["mask"] = atom_mask

        return ret_dict

    def serialize(self) -> dict:
        return {
            "atom_exclude_types": self.atom_exclude_types,
            "pair_exclude_types": self.pair_exclude_types,
        }

    def compute_or_load_stat(
        self,
        sampled_func,
        stat_file_path: Optional[DPPath] = None,
    ):
        """
        Compute or load the statistics parameters of the model,
        such as mean and standard deviation of descriptors or the energy bias of the fitting net.
        When `sampled` is provided, all the statistics parameters will be calculated (or re-calculated for update),
        and saved in the `stat_file_path`(s).
        When `sampled` is not provided, it will check the existence of `stat_file_path`(s)
        and load the calculated statistics parameters.

        Parameters
        ----------
        sampled_func
            The sampled data frames from different data systems.
        stat_file_path
            The path to the statistics files.
        """
        raise NotImplementedError
