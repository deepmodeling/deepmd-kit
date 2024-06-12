# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
from typing import (
    Dict,
    List,
    Optional,
)

import numpy as np

from deepmd.dpmodel import (
    FittingOutputDef,
)
from deepmd.dpmodel.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.dpmodel.fitting.base_fitting import (
    BaseFitting,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .base_atomic_model import (
    BaseAtomicModel,
)


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
        fitting_dict,
        type_map: Optional[List[str]],
        **kwargs,
    ):
        super().__init__(type_map, **kwargs)
        self.type_map = type_map
        self.descriptor = descriptor
        fitting_dict = copy.deepcopy(fitting_dict)
        self.model_type = fitting_dict.pop("type")
        self.fitting_net_dict = fitting_dict
        self.fitting_net = fitting_dict
        super().init_out_stat()

    def fitting_output_def(self) -> FittingOutputDef:
        """Get the output def of the fitting net."""
        var_defs = []
        for name, fitting_net in self.fitting_net_dict.items():
            for vdef in fitting_net.output_def().var_defs.values():
                vdef.name = name
                var_defs.append(vdef)
        return FittingOutputDef(var_defs)

    def get_rcut(self) -> float:
        """Get the cut-off radius."""
        return self.descriptor.get_rcut()

    def get_sel(self) -> List[int]:
        """Get the neighbor selection."""
        return self.descriptor.get_sel()

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

    def forward_atomic(
        self,
        extended_coord: np.ndarray,
        extended_atype: np.ndarray,
        nlist: np.ndarray,
        mapping: Optional[np.ndarray] = None,
        fparam: Optional[np.ndarray] = None,
        aparam: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
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
        descriptor, rot_mat, g2, h2, sw = self.descriptor(
            extended_coord,
            extended_atype,
            nlist,
            mapping=mapping,
        )
        fit_ret_dict = {}
        for name, fitting_net in self.fitting_net_dict.items():
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
                fit_ret_dict[name] = v
        return fit_ret_dict

    def serialize(self) -> dict:
        dd = super().serialize(self)
        dd.update(
            {
                "@class": "Model",
                "@version": 2,
                "type": "multi_fitting",
                "type_map": self.type_map,
                "descriptor": self.descriptor.serialize(),
                "fitting": [
                    fitting_net.serialize()
                    for fitting_net in self.fitting_net_dict.values()
                ],
                "fitting_name": self.fitting_net_dict.keys(),
            }
        )
        return dd

    @classmethod
    def deserialize(cls, data) -> "DPMultiFittingAtomicModel":
        data = copy.deepcopy(data)
        check_version_compatibility(data.pop("@version", 1), 2, 1)
        data.pop("@class", None)
        data.pop("type", None)
        descriptor_obj = BaseDescriptor.deserialize(data.pop("descriptor"))

        fitting_dict = {}
        fitting_names = data["fitting_name"]
        for name, fitting in zip(fitting_names, data.pop("fitting")):
            fitting_obj = BaseFitting.deserialize(fitting)
            fitting_dict[name] = fitting_obj
        # type_map = data.pop("type_map", None)
        # obj = cls(descriptor_obj, fitting_dict, type_map=type_map, **data)
        data["descriptor"] = descriptor_obj
        data["fitting"] = list(fitting_dict.values())
        data["fitting_name"] = list(fitting_dict.keys())
        obj = super().deserialize(data)
        return obj

    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this atomic model."""
        dim_fparam = None
        for fitting in self.fitting_net_dict.values():
            if dim_fparam is not None:
                assert dim_fparam == fitting.get_dim_fparam()
            else:
                dim_fparam = fitting.get_dim_fparam()
        return dim_fparam

    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this atomic model."""
        dim_aparam = None
        for fitting in self.fitting_net_dict.values():
            if dim_aparam is not None:
                assert dim_aparam == fitting.get_dim_aparam()
            else:
                dim_aparam = fitting.get_dim_aparam()
        return dim_aparam

    def get_sel_type(self) -> List[List[int]]:
        """Get the selected atom types of this model.

        Only atoms with selected atom types have atomic contribution
        to the result of the model.
        If returning an empty list, all atom types are selected.
        """
        sel_type = []
        for fitting_net in self.fitting_net_dict.values():
            sel_type.append(fitting_net.get_sel_type())
        return sel_type

    def is_aparam_nall(self) -> bool:
        """Check whether the shape of atomic parameters is (nframes, nall, ndim).

        If False, the shape is (nframes, nloc, ndim).
        """
        return False
