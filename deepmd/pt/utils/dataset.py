# SPDX-License-Identifier: LGPL-3.0-or-later

from collections import (
    defaultdict,
)
from typing import (
    Optional,
)

import numpy as np
from torch.utils.data import (
    Dataset,
)

from deepmd.utils.data import (
    DataRequirementItem,
    DeepmdData,
)


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

    def __len__(self) -> int:
        return self._data_system.nframes

    def __getitem__(self, index):
        """Get a frame from the selected system."""
        b_data = self._data_system.get_item_torch(index)
        b_data["natoms"] = self._natoms_vec
        return b_data

    def get_frame_index_for_elements(self):
        """
        Get the frame index and the number of frames with all the elements in the system.
        Map the remapped atom_type_mix back to their element names in type_map,
        This function is only used in the mixed type.

        Returns
        -------
        element_counts : dict
            A dictionary where:
            - The key is the element type.
            - The value is another dictionary with the following keys:
                - "frames": int
                    The total number of frames in which the element appears.
                - "indices": list of int
                    A list of row indices where the element is found in the dataset.
        global_type_name : dict
            The key is the element index and the value is the element name.
        """
        element_counts = defaultdict(lambda: {"frames": 0, "indices": []})
        set_files = self._data_system.dirs
        base_offset = 0
        global_type_name = {}
        for set_file in set_files:
            element_data = self._data_system._load_type_mix(set_file)
            unique_elements = np.unique(element_data)
            type_name = self._data_system.build_reidx_to_name_map(
                element_data, set_file
            )
            for new_idx, elem_name in type_name.items():
                if new_idx not in global_type_name:
                    global_type_name[new_idx] = elem_name
            for elem in unique_elements:
                frames_with_elem = np.any(element_data == elem, axis=1)
                row_indices = np.where(frames_with_elem)[0]
                row_indices_global = np.where(frames_with_elem)[0] + base_offset
                element_counts[elem]["frames"] += len(row_indices)
                element_counts[elem]["indices"].extend(row_indices_global.tolist())
            base_offset += element_data.shape[0]
        element_counts = dict(element_counts)
        return element_counts, global_type_name

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
