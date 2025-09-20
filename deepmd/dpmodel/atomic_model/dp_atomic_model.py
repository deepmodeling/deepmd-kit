# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
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
        type_map: list[str],
        **kwargs,
    ) -> None:
        super().__init__(type_map, **kwargs)
        self.type_map = type_map
        self.descriptor = descriptor
        self.fitting = fitting
        self.type_map = type_map
        super().init_out_stat()

    def fitting_output_def(self) -> FittingOutputDef:
        """Get the output def of the fitting net."""
        return self.fitting.output_def()

    def get_rcut(self) -> float:
        """Get the cut-off radius."""
        return self.descriptor.get_rcut()

    def get_sel(self) -> list[int]:
        """Get the neighbor selection."""
        return self.descriptor.get_sel()

    def set_case_embd(self, case_idx: int):
        """
        Set the case embedding of this atomic model by the given case_idx,
        typically concatenated with the output of the descriptor and fed into the fitting net.
        """
        self.fitting.set_case_embd(case_idx)

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

    def need_sorted_nlist_for_lower(self) -> bool:
        """Returns whether the atomic model needs sorted nlist when using `forward_lower`."""
        return self.descriptor.need_sorted_nlist_for_lower()

    def enable_compression(
        self,
        min_nbor_dist: float,
        table_extrapolate: float = 5,
        table_stride_1: float = 0.01,
        table_stride_2: float = 0.1,
        check_frequency: int = -1,
    ) -> None:
        """Call descriptor enable_compression().

        Parameters
        ----------
        min_nbor_dist
            The nearest distance between atoms
        table_extrapolate
            The scale of model extrapolation
        table_stride_1
            The uniform stride of the first table
        table_stride_2
            The uniform stride of the second table
        check_frequency
            The overflow check frequency
        """
        self.descriptor.enable_compression(
            min_nbor_dist,
            table_extrapolate,
            table_stride_1,
            table_stride_2,
            check_frequency,
        )

    def forward_atomic(
        self,
        extended_coord: np.ndarray,
        extended_atype: np.ndarray,
        nlist: np.ndarray,
        mapping: Optional[np.ndarray] = None,
        fparam: Optional[np.ndarray] = None,
        aparam: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        """Models' atomic predictions.

        Parameters
        ----------
        extended_coord
            coordinates in extended region
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

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat=None
    ) -> None:
        """Change the type related params to new ones, according to `type_map` and the original one in the model.
        If there are new types in `type_map`, statistics will be updated accordingly to `model_with_new_type_stat` for these new types.
        """
        super().change_type_map(
            type_map=type_map, model_with_new_type_stat=model_with_new_type_stat
        )
        self.type_map = type_map
        self.descriptor.change_type_map(
            type_map=type_map,
            model_with_new_type_stat=model_with_new_type_stat.descriptor
            if model_with_new_type_stat is not None
            else None,
        )
        self.fitting_net.change_type_map(type_map=type_map)

    def serialize(self) -> dict:
        dd = super().serialize()
        dd.update(
            {
                "@class": "Model",
                "type": "standard",
                "@version": 2,
                "type_map": self.type_map,
                "descriptor": self.descriptor.serialize(),
                "fitting": self.fitting.serialize(),
            }
        )
        return dd

    # for subclass overridden
    base_descriptor_cls = BaseDescriptor
    """The base descriptor class."""
    base_fitting_cls = BaseFitting
    """The base fitting class."""

    @classmethod
    def deserialize(cls, data) -> "DPAtomicModel":
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 2, 2)
        data.pop("@class")
        data.pop("type")
        descriptor_obj = cls.base_descriptor_cls.deserialize(data.pop("descriptor"))
        fitting_obj = cls.base_fitting_cls.deserialize(data.pop("fitting"))
        data["descriptor"] = descriptor_obj
        data["fitting"] = fitting_obj
        obj = super().deserialize(data)
        return obj

    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this atomic model."""
        return self.fitting.get_dim_fparam()

    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this atomic model."""
        return self.fitting.get_dim_aparam()

    def get_sel_type(self) -> list[int]:
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
