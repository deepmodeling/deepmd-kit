# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
from typing import (
    Dict,
    List,
    Optional,
)

import numpy as np

from deepmd.dpmodel.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.dpmodel.fitting.base_fitting import (
    BaseFitting,
)
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .base_atomic_model import (
    BaseAtomicModel,
)


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
        self.type_map = type_map
        self.descriptor = descriptor
        self.fitting = fitting
        self.type_map = type_map
        super().__init__(**kwargs)

    def fitting_output_def(self) -> FittingOutputDef:
        """Get the output def of the fitting net."""
        return self.fitting.output_def()

    def get_rcut(self) -> float:
        """Get the cut-off radius."""
        return self.descriptor.get_rcut()

    def get_sel(self) -> List[int]:
        """Get the neighbor selection."""
        return self.descriptor.get_sel()

    def get_type_map(self) -> List[str]:
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

    def set_out_bias(self, out_bias: np.ndarray, add=False) -> None:
        """
        Modify the output bias for the atomic model.

        Parameters
        ----------
        out_bias : np.ndarray
            The new bias to be applied.
        add : bool, optional
            Whether to add the new bias to the existing one.
            If False, the output bias will be directly replaced by the new bias.
            If True, the new bias will be added to the existing one.
        """
        self.fitting["bias_atom_e"] = (
            out_bias + self.fitting["bias_atom_e"] if add else out_bias
        )

    def get_out_bias(self) -> np.ndarray:
        """Return the output bias of the atomic model."""
        return self.fitting["bias_atom_e"]

    def need_sorted_nlist_for_lower(self) -> bool:
        """Returns whether the atomic model needs sorted nlist when using `forward_lower`."""
        return self.descriptor.need_sorted_nlist_for_lower()

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
        dd = super().serialize()
        dd.update(
            {
                "@class": "Model",
                "type": "standard",
                "@version": 1,
                "type_map": self.type_map,
                "descriptor": self.descriptor.serialize(),
                "fitting": self.fitting.serialize(),
            }
        )
        return dd

    @classmethod
    def deserialize(cls, data) -> "DPAtomicModel":
        data = copy.deepcopy(data)
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        data.pop("@class")
        data.pop("type")
        descriptor_obj = BaseDescriptor.deserialize(data.pop("descriptor"))
        fitting_obj = BaseFitting.deserialize(data.pop("fitting"))
        type_map = data.pop("type_map")
        obj = cls(descriptor_obj, fitting_obj, type_map=type_map, **data)
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
