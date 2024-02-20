# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import sys
from typing import (
    Dict,
    List,
    Optional,
)

import numpy as np

from deepmd.dpmodel.descriptor import (  # noqa # TODO: should import all descriptors!
    DescrptSeA,
)
from deepmd.dpmodel.fitting import (  # noqa # TODO: should import all fittings!
    EnergyFittingNet,
    InvarFitting,
)
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
)

from .base_atomic_model import (
    BaseAtomicModel,
)


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
        type_map: Optional[List[str]] = None,
    ):
        super().__init__()
        self.type_map = type_map
        self.descriptor = descriptor
        self.fitting = fitting

    def fitting_output_def(self) -> FittingOutputDef:
        """Get the output def of the fitting net."""
        return self.fitting.output_def()

    def get_rcut(self) -> float:
        """Get the cut-off radius."""
        return self.descriptor.get_rcut()

    def get_sel(self) -> List[int]:
        """Get the neighbor selection."""
        return self.descriptor.get_sel()

    def get_type_map(self) -> Optional[List[str]]:
        """Get the type map."""
        return self.type_map

    def mixed_types(self) -> bool:
        """If true, the model
        1. assumes total number of atoms aligned across frames;
        2. uses a neighbor list that does not distinguish different atomic types.

        If false, the model
        1. assumes total number of atoms of each atom type aligned across frames;
        2. uses a neighbor list that distinguishes different atomic types.

        """
        return self.descriptor.mixed_types()

    def forward_atomic(
        self,
        extended_coord: np.ndarray,
        extended_atype: np.ndarray,
        nlist: np.ndarray,
        mapping: Optional[np.ndarray] = None,
        fparam: Optional[np.ndarray] = None,
        aparam: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Models' atomic predictions.

        Parameters
        ----------
        extended_coord
            coodinates in extended region
        extended_atype
            atomic type in extended region
        nlist
            neighbor list. nf x nloc x nsel
        mapping
            mapps the extended indices to local indices. nf x nall
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
        descriptor, rot_mat, g2, h2, sw = self.descriptor(
            extended_coord,
            extended_atype,
            nlist,
            mapping=mapping,
        )
        ret = self.fitting(
            descriptor,
            atype,
            gr=rot_mat,
            g2=g2,
            h2=h2,
            fparam=fparam,
            aparam=aparam,
        )
        return ret

    def serialize(self) -> dict:
        return {
            "type_map": self.type_map,
            "descriptor": self.descriptor.serialize(),
            "fitting": self.fitting.serialize(),
            "descriptor_name": self.descriptor.__class__.__name__,
            "fitting_name": self.fitting.__class__.__name__,
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

    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this atomic model."""
        return self.fitting.get_dim_fparam()

    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this atomic model."""
        return self.fitting.get_dim_aparam()

    def get_sel_type(self) -> List[int]:
        """Get the selected atom types of this model.

        Only atoms with selected atom types have atomic contribution
        to the result of the model.
        If returning an empty list, all atom types are selected.
        """
        return self.fitting.get_sel_type()

    def is_aparam_nall(self) -> bool:
        """Check whether the shape of atomic parameters is (nframes, nall, ndim).

        If False, the shape is (nframes, nloc, ndim).
        """
        return False
