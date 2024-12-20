# SPDX-License-Identifier: LGPL-3.0-or-later


from typing import (
    Optional,
)

from torch.utils.data import (
    Dataset,
)

from deepmd.utils.data import (
    DataRequirementItem,
    DeepmdData,
)


class DeepmdDataSetForLoader(Dataset):
    def __init__(self, system: str, type_map: Optional[list[str]] = None) -> None:
        """Construct DeePMD-style dataset containing frames cross different systems.

        Args:
        - systems: Paths to systems.
        - type_map: Atom types.
        """
        self.system = system
        self._type_map = type_map
        self._data_system = DeepmdData(sys_path=system, type_map=self._type_map)
        self.mixed_type = self._data_system.mixed_type
        self._ntypes = self._data_system.get_ntypes()
        self._natoms = self._data_system.get_natoms()
        self._natoms_vec = self._data_system.get_natoms_vec(self._ntypes)

    def __len__(self) -> int:
        return self._data_system.nframes

    def __getitem__(self, index):
        """Get a frame from the selected system."""
        b_data = self._data_system.get_item_torch(index)
        b_data["natoms"] = self._natoms_vec
        return b_data

    def add_data_requirement(self, data_requirement: list[DataRequirementItem]) -> None:
        """Add data requirement for this data system."""
        for data_item in data_requirement:
            self._data_system.add(
                data_item["key"],
                data_item["ndof"],
                atomic=data_item["atomic"],
                must=data_item["must"],
                high_prec=data_item["high_prec"],
                type_sel=data_item["type_sel"],
                repeat=data_item["repeat"],
                default=data_item["default"],
                dtype=data_item["dtype"],
                output_natoms_for_type_sel=data_item["output_natoms_for_type_sel"],
            )
