# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    abstractmethod,
)

import torch

from deepmd.dpmodel.modifier.base_modifier import (
    make_base_modifier,
)
from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.pt.utils.utils import to_torch_tensor, to_numpy_array
from deepmd.utils.data import (
    DeepmdData,
)


class BaseModifier(torch.nn.Module, make_base_modifier()):
    def __init__(self) -> None:
        """Construct a basic model for different tasks."""
        torch.nn.Module.__init__(self)
        self.modifier_type = "base"

    def serialize(self) -> dict:
        """Serialize the modifier.

        Returns
        -------
        dict
            The serialized data
        """
        data = {
            "@class": "Modifier",
            "type": self.modifier_type,
            "@version": 3,
        }
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "BaseModifier":
        """Deserialize the modifier.

        Parameters
        ----------
        data : dict
            The serialized data

        Returns
        -------
        BaseModifier
            The deserialized modifier
        """
        data = data.copy()
        modifier = cls(**data)
        return modifier

    @abstractmethod
    @torch.jit.export
    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute energy, force, and virial corrections."""

    @torch.jit.unused
    def modify_data(self, data: dict[str, Array | float], data_sys: DeepmdData) -> None:
        """Modify data.

        Parameters
        ----------
        data
            Internal data of DeepmdData.
            Be a dict, has the following keys
            - coord         coordinates
            - box           simulation box
            - atype         atom types
            - fparam        frame parameter
            - aparam        atom parameter
            - find_energy   tells if data has energy
            - find_force    tells if data has force
            - find_virial   tells if data has virial
            - energy        energy
            - force         force
            - virial        virial
        """
        if (
            "find_energy" not in data
            and "find_force" not in data
            and "find_virial" not in data
        ):
            return

        get_nframes = None
        t_coord = to_torch_tensor(data["coord"][:get_nframes, :])
        t_atype = to_torch_tensor(data["atype"][:get_nframes, :])
        if data["box"] is None:
            t_box = None
        else:
            t_box = to_torch_tensor(data["box"][:get_nframes, :])
        if data["fparam"] is None:
            t_fparam = None
        else:
            t_fparam = to_torch_tensor(data["fparam"][:get_nframes, :])
        if data["aparam"] is None:
            t_aparam = None
        else:
            t_aparam = to_torch_tensor(data["aparam"][:get_nframes, :])
        # 
        
        # implement data modification method in forward
        modifier_data = self.forward(t_coord, t_atype, t_box, t_fparam, t_aparam)

        if "find_energy" in data and data["find_energy"] == 1.0:
            data["energy"] -= to_numpy_array(modifier_data["energy"]).reshape(data["energy"].shape)
        if "find_force" in data and data["find_force"] == 1.0:
            data["force"] -= to_numpy_array(modifier_data["force"]).reshape(data["force"].shape)
        if "find_virial" in data and data["find_virial"] == 1.0:
            data["virial"] -= to_numpy_array(modifier_data["virial"]).reshape(data["virial"].shape)
