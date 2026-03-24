# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.dpmodel.fitting.base_fitting import (
    BaseFitting,
)
from deepmd.dpmodel.fitting.xas_fitting import (
    XASFittingNet,
)

from .dp_atomic_model import (
    DPAtomicModel,
)


class DPXASAtomicModel(DPAtomicModel):
    """Atomic model for XAS spectrum fitting.

    Automatically sets ``atom_exclude_types`` to all non-absorbing atom types
    so that the intensive mean reduction in ``fit_output_to_model_output``
    computes the mean XAS over absorbing atoms only.

    Parameters
    ----------
    descriptor : BaseDescriptor
    fitting : BaseFitting
        Must be an instance of XASFittingNet.
    type_map : list[str]
        Mapping from type index to element symbol.
    absorbing_type : str
        Element symbol of the absorbing atom type (e.g. "Fe").
    **kwargs
        Passed to DPAtomicModel.
    """

    def __init__(
        self,
        descriptor: BaseDescriptor,
        fitting: BaseFitting,
        type_map: list[str],
        absorbing_type: str,
        **kwargs: Any,
    ) -> None:
        if not isinstance(fitting, XASFittingNet):
            raise TypeError(
                "fitting must be an instance of XASFittingNet for DPXASAtomicModel"
            )
        if absorbing_type not in type_map:
            raise ValueError(
                f"absorbing_type '{absorbing_type}' not found in type_map {type_map}"
            )
        self.absorbing_type = absorbing_type
        absorbing_idx = type_map.index(absorbing_type)
        # Exclude all types except the absorbing type so the intensive mean
        # reduction is computed only over absorbing atoms.
        atom_exclude_types = [i for i in range(len(type_map)) if i != absorbing_idx]
        kwargs["atom_exclude_types"] = atom_exclude_types
        super().__init__(descriptor, fitting, type_map, **kwargs)

    def get_intensive(self) -> bool:
        """XAS is an intensive property (mean over absorbing atoms)."""
        return True

    def serialize(self) -> dict:
        dd = super().serialize()
        dd["absorbing_type"] = self.absorbing_type
        return dd

    @classmethod
    def deserialize(cls, data: dict) -> "DPXASAtomicModel":
        data = data.copy()
        absorbing_type = data.pop("absorbing_type")
        # atom_exclude_types is already stored by base; rebuild absorbing_type param
        obj = super().deserialize(data)
        obj.absorbing_type = absorbing_type
        return obj
