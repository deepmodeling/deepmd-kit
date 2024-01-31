# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import sys
from typing import (
    Dict,
    List,
    Optional,
)

import numpy as np

from .base_atomic_model import (
    BaseAtomicModel,
)
from .fitting import InvarFitting  # noqa # TODO: should import all fittings!
from .output_def import (
    FittingOutputDef,
)
from .se_e2_a import DescrptSeA  # noqa # TODO: should import all descriptors!


class DPAtomicModel(BaseAtomicModel):
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

    def get_fitting_output_def(self) -> FittingOutputDef:
        """Get the output def of the fitting net."""
        return self.fitting.output_def()

    def get_rcut(self) -> float:
        """Get the cut-off radius."""
        return self.descriptor.get_rcut()

    def get_sel(self) -> List[int]:
        """Get the neighbor selection."""
        return self.descriptor.get_sel()

    def distinguish_types(self) -> bool:
        """If distinguish different types by sorting."""
        return self.descriptor.distinguish_types()

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
