# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import logging
import sys
from typing import (
    Dict,
    List,
    Optional,
)

import torch

from deepmd.dpmodel import (
    FittingOutputDef,
)
from deepmd.pt.model.descriptor.se_a import (  # noqa # TODO: should import all descriptors!!!
    DescrptSeA,
)
from deepmd.pt.model.task.ener import (  # noqa # TODO: should import all fittings!
    EnergyFittingNet,
    InvarFitting,
)
from deepmd.pt.utils.utils import (
    dict_to_device,
)
from deepmd.utils.path import (
    DPPath,
)

from .base_atomic_model import (
    BaseAtomicModel,
)

log = logging.getLogger(__name__)


class DPAtomicModel(torch.nn.Module, BaseAtomicModel):
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
        if self.do_grad_r() or self.do_grad_c():
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
        sampled,
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
        sampled
            The sampled data frames from different data systems.
        stat_file_path
            The dictionary of paths to the statistics files.
        """
        if stat_file_path is not None and self.type_map is not None:
            # descriptors and fitting net with different type_map
            # should not share the same parameters
            stat_file_path /= " ".join(self.type_map)
        for data_sys in sampled:
            dict_to_device(data_sys)
        if sampled is None:
            sampled = []
        self.descriptor.compute_input_stats(sampled, stat_file_path)
        if self.fitting_net is not None:
            self.fitting_net.compute_output_stats(sampled, stat_file_path)

    @torch.jit.export
    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this atomic model."""
        return self.fitting_net.get_dim_fparam()

    @torch.jit.export
    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this atomic model."""
        return self.fitting_net.get_dim_aparam()

    @torch.jit.export
    def get_sel_type(self) -> List[int]:
        """Get the selected atom types of this model.

        Only atoms with selected atom types have atomic contribution
        to the result of the model.
        If returning an empty list, all atom types are selected.
        """
        return self.fitting_net.get_sel_type()

    @torch.jit.export
    def is_aparam_nall(self) -> bool:
        """Check whether the shape of atomic parameters is (nframes, nall, ndim).

        If False, the shape is (nframes, nloc, ndim).
        """
        return False
