# SPDX-License-Identifier: LGPL-3.0-or-later
import torch

from deepmd.dpmodel.modifier.base_modifier import (
    make_base_modifier,
)
from deepmd.utils.data import (
    DeepmdData,
)


class BaseModifier(torch.nn.Module, make_base_modifier()):
    def __init__(self) -> None:
        """Construct a basic model for different tasks."""
        torch.nn.Module.__init__(self)

    def modify_data(self, data: dict, data_sys: DeepmdData) -> None:
        # TODO: data_sys parameter is currently unused but may be needed by subclasses in the future
        """Modify data.

        Parameters
        ----------
        data
            Internal data of DeepmdData.
            Be a dict, has the following keys
            - coord         coordinates
            - box           simulation box
            - atype          atom types
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
        coord = data["coord"][:get_nframes, :]
        if data["box"] is None:
            box = None
        else:
            box = data["box"][:get_nframes, :]
        atype = data["atype"][:get_nframes, :]

        # implement data modification method in forward
        tot_e, tot_f, tot_v = self.forward(coord, atype, box, False, None, None)

        if "find_energy" in data and data["find_energy"] == 1.0:
            data["energy"] -= tot_e.reshape(data["energy"].shape)
        if "find_force" in data and data["find_force"] == 1.0:
            data["force"] -= tot_f.reshape(data["force"].shape)
        if "find_virial" in data and data["find_virial"] == 1.0:
            data["virial"] -= tot_v.reshape(data["virial"].shape)
