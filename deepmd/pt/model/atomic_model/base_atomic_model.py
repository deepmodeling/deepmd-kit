# SPDX-License-Identifier: LGPL-3.0-or-later


import logging
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

import numpy as np
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
from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
)
from deepmd.pt.utils.stat import (
    compute_output_stats,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
)
from deepmd.utils.path import (
    DPPath,
)

log = logging.getLogger(__name__)

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

    def get_forward_wrapper_func(self) -> Callable[..., torch.Tensor]:
        """Get a forward wrapper of the atomic model for output bias calculation."""

        def model_forward(coord, atype, box, fparam=None, aparam=None):
            with torch.no_grad():  # it's essential for pure torch forward function to use auto_batchsize
                (
                    extended_coord,
                    extended_atype,
                    mapping,
                    nlist,
                ) = extend_input_and_build_neighbor_list(
                    coord,
                    atype,
                    self.get_rcut(),
                    self.get_sel(),
                    mixed_types=self.mixed_types(),
                    box=box,
                )
                atomic_ret = self.forward_common_atomic(
                    extended_coord,
                    extended_atype,
                    nlist,
                    mapping=mapping,
                    fparam=fparam,
                    aparam=aparam,
                )
                return {kk: vv.detach() for kk, vv in atomic_ret.items()}

        return model_forward

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

    def change_out_bias(
        self,
        merged,
        origin_type_map,
        full_type_map,
        bias_adjust_mode="change-by-statistic",
    ) -> None:
        """Change the output bias according to the input data and the pretrained model.

        Parameters
        ----------
        merged : Union[Callable[[], List[dict]], List[dict]]
            - List[dict]: A list of data samples from various data systems.
                Each element, `merged[i]`, is a data dictionary containing `keys`: `torch.Tensor`
                originating from the `i`-th data system.
            - Callable[[], List[dict]]: A lazy function that returns data samples in the above format
                only when needed. Since the sampling process can be slow and memory-intensive,
                the lazy function helps by only sampling once.
        origin_type_map : List[str]
            The original type_map in dataset, they are targets to change the output bias.
        full_type_map : List[str]
            The full type_map in pre-trained model
        bias_adjust_mode : str
            The mode for changing output bias : ['change-by-statistic', 'set-by-statistic']
            'change-by-statistic' : perform predictions on labels of target dataset,
                    and do least square on the errors to obtain the target shift as bias.
            'set-by-statistic' : directly use the statistic output bias in the target dataset.
        """
        sorter = np.argsort(full_type_map)
        missing_types = [t for t in origin_type_map if t not in full_type_map]
        assert (
            not missing_types
        ), f"Some types are not in the pre-trained model: {list(missing_types)} !"
        idx_type_map = sorter[
            np.searchsorted(full_type_map, origin_type_map, sorter=sorter)
        ]
        original_bias = self.get_out_bias()
        if bias_adjust_mode == "change-by-statistic":
            delta_bias = compute_output_stats(
                merged,
                self.get_ntypes(),
                keys=list(self.fitting_output_def().keys()),
                model_forward=self.get_forward_wrapper_func(),
            )[list(self.fitting_output_def().keys())[0]]
            self.set_out_bias(delta_bias, add=True)
        elif bias_adjust_mode == "set-by-statistic":
            bias_atom = compute_output_stats(
                merged,
                self.get_ntypes(),
                keys=list(self.fitting_output_def().keys()),
            )[list(self.fitting_output_def().keys())[0]]
            self.set_out_bias(bias_atom)
        else:
            raise RuntimeError("Unknown bias_adjust_mode mode: " + bias_adjust_mode)
        bias_atom = self.get_out_bias()
        log.info(
            f"Change output bias of {origin_type_map!s} "
            f"from {to_numpy_array(original_bias[idx_type_map]).reshape(-1)!s} "
            f"to {to_numpy_array(bias_atom[idx_type_map]).reshape(-1)!s}."
        )
