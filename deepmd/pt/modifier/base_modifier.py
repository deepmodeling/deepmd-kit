# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    abstractmethod,
)

import numpy as np
import torch

from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.common import PRECISION_DICT as NP_PRECISION_DICT
from deepmd.dpmodel.modifier.base_modifier import (
    make_base_modifier,
)
from deepmd.pt.utils.env import (
    DEVICE,
    GLOBAL_PT_FLOAT_PRECISION,
    RESERVED_PRECISION_DICT,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
    to_torch_tensor,
)
from deepmd.utils.data import (
    DeepmdData,
)


class BaseModifier(torch.nn.Module, make_base_modifier()):
    def __init__(self, use_cache: bool = True) -> None:
        """Construct a base modifier for data modification tasks."""
        torch.nn.Module.__init__(self)
        self.modifier_type = "base"
        self.jitable = True

        self.use_cache = use_cache

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
        # Remove serialization metadata before passing to constructor
        data.pop("@class", None)
        data.pop("type", None)
        data.pop("@version", None)
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
        do_atomic_virial: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Compute energy, force, and virial corrections."""

    @torch.jit.unused
    def modify_data(self, data: dict[str, Array | float], data_sys: DeepmdData) -> None:
        """Modify data of single frame.

        Parameters
        ----------
        data
            Internal data of DeepmdData.
            Be a dict, has the following keys
            - coord         coordinates (nat, 3)
            - box           simulation box (9,)
            - atype         atom types (nat,)
            - fparam        frame parameter (nfp,)
            - aparam        atom parameter (nat, nap)
            - find_energy   tells if data has energy
            - find_force    tells if data has force
            - find_virial   tells if data has virial
            - energy        energy (1,)
            - force         force (nat, 3)
            - virial        virial (9,)
        """
        if (
            "find_energy" not in data
            and "find_force" not in data
            and "find_virial" not in data
        ):
            return

        prec = NP_PRECISION_DICT[RESERVED_PRECISION_DICT[GLOBAL_PT_FLOAT_PRECISION]]

        nframes = 1
        natoms = len(data["atype"])
        atom_types = np.tile(data["atype"], nframes).reshape(nframes, -1)

        coord_input = torch.tensor(
            data["coord"].reshape([nframes, natoms, 3]).astype(prec),
            dtype=GLOBAL_PT_FLOAT_PRECISION,
            device=DEVICE,
        )
        type_input = torch.tensor(
            atom_types.astype(NP_PRECISION_DICT[RESERVED_PRECISION_DICT[torch.long]]),
            dtype=torch.long,
            device=DEVICE,
        )
        if data["box"] is not None:
            box_input = torch.tensor(
                data["box"].reshape([nframes, 3, 3]).astype(prec),
                dtype=GLOBAL_PT_FLOAT_PRECISION,
                device=DEVICE,
            )
        else:
            box_input = None
        if "fparam" in data:
            fparam_input = to_torch_tensor(data["fparam"].reshape(nframes, -1))
        else:
            fparam_input = None
        if "aparam" in data:
            aparam_input = to_torch_tensor(data["aparam"].reshape(nframes, natoms, -1))
        else:
            aparam_input = None
        do_atomic_virial = False

        # implement data modification method in forward
        modifier_data = self.forward(
            coord_input,
            type_input,
            box_input,
            fparam_input,
            aparam_input,
            do_atomic_virial,
        )

        if data.get("find_energy") == 1.0:
            if "energy" not in modifier_data:
                raise KeyError(
                    f"Modifier {self.__class__.__name__} did not provide 'energy' "
                    "in its output while 'find_energy' is set."
                )
            data["energy"] -= to_numpy_array(modifier_data["energy"]).reshape(
                data["energy"].shape
            )
        if data.get("find_force") == 1.0:
            if "force" not in modifier_data:
                raise KeyError(
                    f"Modifier {self.__class__.__name__} did not provide 'force' "
                    "in its output while 'find_force' is set."
                )
            data["force"] -= to_numpy_array(modifier_data["force"]).reshape(
                data["force"].shape
            )
        if data.get("find_virial") == 1.0:
            if "virial" not in modifier_data:
                raise KeyError(
                    f"Modifier {self.__class__.__name__} did not provide 'virial' "
                    "in its output while 'find_virial' is set."
                )
            data["virial"] -= to_numpy_array(modifier_data["virial"]).reshape(
                data["virial"].shape
            )
