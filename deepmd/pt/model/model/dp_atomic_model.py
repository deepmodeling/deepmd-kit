# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import logging
import os
import sys
from typing import (
    Dict,
    List,
    Optional,
    Union,
)

import torch

from deepmd.dpmodel import (
    FittingOutputDef,
)
from deepmd.pt.model.descriptor.se_a import (  # noqa # TODO: should import all descriptors!!!
    DescrptSeA,
)
from deepmd.pt.model.task.ener import (  # noqa # TODO: should import all fittings!
    InvarFitting,
)
from deepmd.pt.utils.utils import (
    dict_to_device,
)

from .base_atomic_model import (
    BaseAtomicModel,
)
from .model import (
    BaseModel,
)

log = logging.getLogger(__name__)


class DPAtomicModel(BaseModel, BaseAtomicModel):
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

    def __init__(self, descriptor, fitting, type_map: Optional[List[str]]):
        super().__init__()
        ntypes = len(type_map)
        self.type_map = type_map
        self.ntypes = ntypes
        self.descriptor = descriptor
        self.rcut = self.descriptor.get_rcut()
        self.sel = self.descriptor.get_sel()
        self.fitting_net = fitting

    def fitting_output_def(self) -> FittingOutputDef:
        """Get the output def of the fitting net."""
        return (
            self.fitting_net.output_def()
            if self.fitting_net is not None
            else self.coord_denoise_net.output_def()
        )

    def get_rcut(self) -> float:
        """Get the cut-off radius."""
        return self.rcut

    def get_sel(self) -> List[int]:
        """Get the neighbor selection."""
        return self.sel

    def distinguish_types(self) -> bool:
        """If distinguish different types by sorting."""
        return self.descriptor.distinguish_types()

    def serialize(self) -> dict:
        return {
            "type_map": self.type_map,
            "descriptor": self.descriptor.serialize(),
            "fitting": self.fitting_net.serialize(),
            "descriptor_name": self.descriptor.__class__.__name__,
            "fitting_name": self.fitting_net.__class__.__name__,
        }

    @classmethod
    def deserialize(cls, data) -> "DPAtomicModel":
        data = copy.deepcopy(data)
        descriptor_obj = getattr(
            sys.modules[__name__], data["descriptor_name"]
        ).deserialize(data["descriptor"])
        fitting_obj = getattr(sys.modules[__name__], data["fitting_name"]).deserialize(
            data["fitting"]
        )
        obj = cls(descriptor_obj, fitting_obj, type_map=data["type_map"])
        return obj

    def forward_atomic(
        self,
        extended_coord,
        extended_atype,
        nlist,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
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
        if self.do_grad():
            extended_coord.requires_grad_(True)
        descriptor, rot_mat, g2, h2, sw = self.descriptor(
            extended_coord,
            extended_atype,
            nlist,
            mapping=mapping,
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

    def compute_or_load_stat(
        self,
        type_map: Optional[List[str]] = None,
        sampled=None,
        stat_file_path_dict: Optional[Dict[str, Union[str, List[str]]]] = None,
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
        type_map
            Mapping atom type to the name (str) of the type.
            For example `type_map[1]` gives the name of the type 1.
        sampled
            The sampled data frames from different data systems.
        stat_file_path_dict
            The dictionary of paths to the statistics files.
        """
        if sampled is not None:  # move data to device
            for data_sys in sampled:
                dict_to_device(data_sys)
        if stat_file_path_dict is not None:
            if not isinstance(stat_file_path_dict["descriptor"], list):
                stat_file_dir = os.path.dirname(stat_file_path_dict["descriptor"])
            else:
                stat_file_dir = os.path.dirname(stat_file_path_dict["descriptor"][0])
            if not os.path.exists(stat_file_dir):
                os.mkdir(stat_file_dir)
        self.descriptor.set_stats(type_map, sampled, stat_file_path_dict["descriptor"])
        if self.fitting_net is not None:
            self.fitting_net.set_stats(
                type_map, sampled, stat_file_path_dict["fitting_net"]
            )
