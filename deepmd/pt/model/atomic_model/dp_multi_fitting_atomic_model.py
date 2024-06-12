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
    OutputVariableDef,
)
from deepmd.pt.model.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.pt.model.task.base_fitting import (
    BaseFitting,
)
from deepmd.pt.model.task.fitting import (
    Fitting,
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


@BaseAtomicModel.register("multi_fitting")
class DPMultiFittingAtomicModel(BaseAtomicModel):
    """Model give atomic prediction of some physical property.

    Parameters
    ----------
    descriptor
            Descriptor
    fitting_dict
            Dict of Fitting net
    type_map
            Mapping atom type to the name (str) of the type.
            For example `type_map[1]` gives the name of the type 1.
    """

    def __init__(
        self,
        descriptor,
        fitting_dict: Dict[str, Fitting],
        type_map: Optional[List[str]],
        **kwargs,
    ):
        super().__init__(type_map, **kwargs)
        ntypes = len(type_map)
        self.type_map = type_map
        self.ntypes = ntypes
        self.descriptor = descriptor
        self.rcut = self.descriptor.get_rcut()
        self.sel = self.descriptor.get_sel()
        fitting_dict = copy.deepcopy(fitting_dict)
        self.model_type = fitting_dict.pop("type")
        self.fitting_net_dict = fitting_dict
        self.fitting_net = fitting_dict
        self.var_defs: List[OutputVariableDef] = []
        for name, fitting_net in fitting_dict.items():
            for vdef in fitting_net.output_def().var_defs.values():
                vdef.name = name
                self.var_defs.append(vdef)
        self.test_fitting = fitting_net
        self.fittings = torch.nn.ModuleList(
            fitting for fitting in fitting_dict.values()
        )
        self.fitting_names = list(fitting_dict.keys())
        super().init_out_stat()

    @torch.jit.export
    def fitting_output_def(self) -> FittingOutputDef:
        """Get the output def of the fitting net."""
        return FittingOutputDef(self.var_defs)

    @torch.jit.export
    def get_rcut(self) -> float:
        """Get the cut-off radius."""
        return self.rcut

    @torch.jit.export
    def get_type_map(self) -> List[str]:
        """Get the type map."""
        return self.type_map

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

    def has_message_passing(self) -> bool:
        """Returns whether the atomic model has message passing."""
        return self.descriptor.has_message_passing()

    def serialize(self) -> dict:
        dd = BaseAtomicModel.serialize(self)
        fitting_dict = {}
        for name, fitting_net in self.fitting_net_dict.items():
            fitting_dict[name] = fitting_net.serialize()
        fitting_dict["type"] = self.model_type
        dd.update(
            {
                "@class": "Model",
                "@version": 2,
                "type": "multi_fitting",
                "type_map": self.type_map,
                "descriptor": self.descriptor.serialize(),
                "fitting_dict": fitting_dict,
            }
        )
        return dd

    @classmethod
    def deserialize(cls, data: dict) -> "DPMultiFittingAtomicModel":
        data = copy.deepcopy(data)
        check_version_compatibility(data.pop("@version", 1), 2, 1)
        data.pop("@class", None)
        data.pop("type", None)
        descriptor_obj = BaseDescriptor.deserialize(data.pop("descriptor"))

        fitting_dict = {}
        _fitting_dict = data["fitting_dict"]
        fitting_dict["type"] = _fitting_dict.pop("type")
        for name, fitting in _fitting_dict.items():
            fitting_obj = BaseFitting.deserialize(fitting)
            fitting_dict[name] = fitting_obj
        data["descriptor"] = descriptor_obj
        data["fitting_dict"] = fitting_dict
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
        fit_ret_dict = {}
        for ii, fitting_net in enumerate(self.fittings):
            fitting = fitting_net(
                descriptor,
                atype,
                gr=rot_mat,
                g2=g2,
                h2=h2,
                fparam=fparam,
                aparam=aparam,
            )
            for v in fitting.values():
                fit_ret_dict[self.fitting_names[ii]] = v
        return fit_ret_dict

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

    @torch.jit.export
    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this atomic model."""
        dim_fparam = None
        for fitting in self.fittings:
            if dim_fparam is not None:
                assert dim_fparam == fitting.get_dim_fparam()
            else:
                dim_fparam = fitting.get_dim_fparam()
        return dim_fparam

    @torch.jit.export
    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this atomic model."""
        dim_aparam = None
        for fitting in self.fittings:
            if dim_aparam is not None:
                assert dim_aparam == fitting.get_dim_aparam()
            else:
                dim_aparam = fitting.get_dim_aparam()
        return dim_aparam

    @torch.jit.export
    def get_sel_type(self) -> List[List[int]]:
        """Get the selected atom types of this model.

        Only atoms with selected atom types have atomic contribution
        to the result of the model.
        If returning an empty list, all atom types are selected.
        """
        sel_type: List[List[int]] = []
        for fitting_net in self.fittings:
            sel_type.append(fitting_net.get_sel_type())
        return sel_type

    @torch.jit.export
    def is_aparam_nall(self) -> bool:
        """Check whether the shape of atomic parameters is (nframes, nall, ndim).

        If False, the shape is (nframes, nloc, ndim).
        """
        return False
