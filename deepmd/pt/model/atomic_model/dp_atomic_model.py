# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import functools
import logging
from typing import (
    Dict,
    List,
    Optional,
)

import torch

from deepmd.dpmodel import (
    FittingOutputDef,
)
from deepmd.pt.model.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.pt.model.task.base_fitting import (
    BaseFitting,
)
from deepmd.utils.path import (
    DPPath,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .base_atomic_model import (
    BaseAtomicModel,
)

log = logging.getLogger(__name__)


@BaseAtomicModel.register("standard")
class DPAtomicModel(BaseAtomicModel):
    """Model give atomic prediction of some physical property.

    Parameters
    ----------
    descriptor
            Descriptor
    fitting_net
            Fitting net
    type_map
            Mapping atom type to the name (str) of the type.
            For example `type_map[1]` gives the name of the type 1.
    """

    def __init__(
        self,
        descriptor,
        fitting,
        type_map: List[str],
        **kwargs,
    ):
        super().__init__(type_map, **kwargs)
        ntypes = len(type_map)
        self.type_map = type_map
        self.ntypes = ntypes
        self.descriptor = descriptor
        self.rcut = self.descriptor.get_rcut()
        self.sel = self.descriptor.get_sel()
        self.fitting_net = fitting
        super().init_out_stat()

    @torch.jit.export
    def fitting_output_def(self) -> FittingOutputDef:
        """Get the output def of the fitting net."""
        return (
            self.fitting_net.output_def()
            if self.fitting_net is not None
            else self.coord_denoise_net.output_def()
        )

    @torch.jit.export
    def get_rcut(self) -> float:
        """Get the cut-off radius."""
        return self.rcut

    def get_sel(self) -> List[int]:
        """Get the neighbor selection."""
        return self.sel

    def mixed_types(self) -> bool:
        """If true, the model
        1. assumes total number of atoms aligned across frames;
        2. uses a neighbor list that does not distinguish different atomic types.

        If false, the model
        1. assumes total number of atoms of each atom type aligned across frames;
        2. uses a neighbor list that distinguishes different atomic types.

        """
        return self.descriptor.mixed_types()

    def change_type_map(
        self, type_map: List[str], model_with_new_type_stat=None
    ) -> None:
        """Change the type related params to new ones, according to `type_map` and the original one in the model.
        If there are new types in `type_map`, statistics will be updated accordingly to `model_with_new_type_stat` for these new types.
        """
        super().change_type_map(
            type_map=type_map, model_with_new_type_stat=model_with_new_type_stat
        )
        self.type_map = type_map
        self.ntypes = len(type_map)
        self.descriptor.change_type_map(
            type_map=type_map,
            model_with_new_type_stat=model_with_new_type_stat.descriptor
            if model_with_new_type_stat is not None
            else None,
        )
        self.fitting_net.change_type_map(type_map=type_map)

    def has_message_passing(self) -> bool:
        """Returns whether the atomic model has message passing."""
        return self.descriptor.has_message_passing()

    def serialize(self) -> dict:
        dd = BaseAtomicModel.serialize(self)
        dd.update(
            {
                "@class": "Model",
                "@version": 2,
                "type": "standard",
                "type_map": self.type_map,
                "descriptor": self.descriptor.serialize(),
                "fitting": self.fitting_net.serialize(),
            }
        )
        return dd

    @classmethod
    def deserialize(cls, data) -> "DPAtomicModel":
        data = copy.deepcopy(data)
        check_version_compatibility(data.pop("@version", 1), 2, 1)
        data.pop("@class", None)
        data.pop("type", None)
        descriptor_obj = BaseDescriptor.deserialize(data.pop("descriptor"))
        fitting_obj = BaseFitting.deserialize(data.pop("fitting"))
        data["descriptor"] = descriptor_obj
        data["fitting"] = fitting_obj
        obj = super().deserialize(data)
        return obj

    def forward_atomic(
        self,
        extended_coord,
        extended_atype,
        nlist,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        comm_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return atomic prediction.

        Parameters
        ----------
        extended_coord
            coodinates in extended region
        extended_atype
            atomic type in extended region
        nlist
            neighbor list. nf x nloc x nsel
        mapping
            mapps the extended indices to local indices
        fparam
            frame parameter. nf x ndf
        aparam
            atomic parameter. nf x nloc x nda

        Returns
        -------
        result_dict
            the result dict, defined by the `FittingOutputDef`.

        """
        nframes, nloc, nnei = nlist.shape
        atype = extended_atype[:, :nloc]
        if self.do_grad_r() or self.do_grad_c():
            extended_coord.requires_grad_(True)
        descriptor, rot_mat, g2, h2, sw = self.descriptor(
            extended_coord,
            extended_atype,
            nlist,
            mapping=mapping,
            comm_dict=comm_dict,
        )
        assert descriptor is not None
        # energy, force
        fit_ret = self.fitting_net(
            descriptor,
            atype,
            gr=rot_mat,
            g2=g2,
            h2=h2,
            fparam=fparam,
            aparam=aparam,
        )
        return fit_ret

    def get_out_bias(self) -> torch.Tensor:
        return self.out_bias

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
            The lazy sampled function to get data frames from different data systems.
        stat_file_path
            The dictionary of paths to the statistics files.
        """
        if stat_file_path is not None and self.type_map is not None:
            # descriptors and fitting net with different type_map
            # should not share the same parameters
            stat_file_path /= " ".join(self.type_map)

        @functools.lru_cache
        def wrapped_sampler():
            sampled = sampled_func()
            if self.pair_excl is not None:
                pair_exclude_types = self.pair_excl.get_exclude_types()
                for sample in sampled:
                    sample["pair_exclude_types"] = list(pair_exclude_types)
            if self.atom_excl is not None:
                atom_exclude_types = self.atom_excl.get_exclude_types()
                for sample in sampled:
                    sample["atom_exclude_types"] = list(atom_exclude_types)
            return sampled

        self.descriptor.compute_input_stats(wrapped_sampler, stat_file_path)
        self.compute_or_load_out_stat(wrapped_sampler, stat_file_path)

    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this atomic model."""
        return self.fitting_net.get_dim_fparam()

    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this atomic model."""
        return self.fitting_net.get_dim_aparam()

    def get_sel_type(self) -> List[int]:
        """Get the selected atom types of this model.

        Only atoms with selected atom types have atomic contribution
        to the result of the model.
        If returning an empty list, all atom types are selected.
        """
        return self.fitting_net.get_sel_type()

    def is_aparam_nall(self) -> bool:
        """Check whether the shape of atomic parameters is (nframes, nall, ndim).

        If False, the shape is (nframes, nloc, ndim).
        """
        return False
