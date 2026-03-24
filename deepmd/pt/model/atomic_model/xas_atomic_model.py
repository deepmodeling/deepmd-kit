# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.pt.model.task.xas import (
    XASFittingNet,
)

from .dp_atomic_model import (
    DPAtomicModel,
)


class DPXASAtomicModel(DPAtomicModel):
    """PyTorch atomic model for XAS spectrum fitting.

    Automatically excludes all non-absorbing atom types so that
    the intensive mean reduction computes the mean XAS over absorbing
    atoms only.

    Parameters
    ----------
    descriptor : Any
    fitting : Any
        Must be an instance of XASFittingNet.
    type_map : Any
        Mapping from type index to element symbol.
    absorbing_type : str
        Element symbol of the absorbing atom type (e.g. "Fe").
    **kwargs
        Passed to DPAtomicModel.
    """

    def __init__(
        self,
        descriptor: Any,
        fitting: Any,
        type_map: Any,
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
        atom_exclude_types = [i for i in range(len(type_map)) if i != absorbing_idx]
        kwargs["atom_exclude_types"] = atom_exclude_types
        super().__init__(descriptor, fitting, type_map, **kwargs)

    def get_intensive(self) -> bool:
        """XAS is an intensive property (mean over absorbing atoms)."""
        return True
