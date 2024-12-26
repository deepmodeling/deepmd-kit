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
import numpy as np

class DeepmdDataSetForLoader(Dataset):
    def __init__(self, system: str, type_map: Optional[list[str]] = None) -> None:
        """Construct DeePMD-style dataset containing frames across different systems.

        Args:
        - system: Path to the system.
        - type_map: Atom types.
        """
        self.system = system
        self._type_map = type_map
        self._data_system = DeepmdData(sys_path=system, type_map=self._type_map)
        self.mixed_type = self._data_system.mixed_type
        self._ntypes = self._data_system.get_ntypes()
        self._natoms = self._data_system.get_natoms()
        self._natoms_vec = self._data_system.get_natoms_vec(self._ntypes)
        self.element_to_frames, self.get_all_atype = self._build_element_to_frames()

    def __len__(self) -> int:
        return self._data_system.nframes

    def __getitem__(self, index):
        """Get a frame from the selected system."""
        b_data = self._data_system.get_item_torch(index)
        b_data["natoms"] = self._natoms_vec
        return b_data
    
    def _build_element_to_frames(self):
        """Build mapping from element types to frame indexes and return all unique element types."""
        element_to_frames = {element: [] for element in range(self._ntypes)}  
        all_elements = set()  
        all_frame_data = self._data_system.get_batch(self._data_system.nframes)
        all_elements = np.unique(all_frame_data["type"])
        for i in range(len(self)):  
            for element in all_elements:
                element_to_frames[element].append(i)
        return element_to_frames, all_elements
    
    def get_frames_for_element(self, missing_element_name):
        """Get the frames that contain the specified element type."""
        element_index = self._type_map.index(missing_element_name) 
        return self.element_to_frames.get(element_index, [])

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
