# SPDX-License-Identifier: LGPL-3.0-or-later


import glob
import os
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
from deepmd.utils.path import (
    DPPath,
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

    def true_types(self):
        """Identify and count unique element types present in the dataset,
        and count the number of frames each element appears in.
        """
        element_counts = defaultdict(lambda: {"count": 0, "frames": 0})
        set_pattern = os.path.join(self.system, "set.*")
        set_files = sorted(glob.glob(set_pattern))
        for set_file in set_files:
            element_data = self._data_system._load_type_mix(DPPath(set_file))
            unique_elements, counts = np.unique(element_data, return_counts=True)
            for elem, cnt in zip(unique_elements, counts):
                element_counts[elem]["count"] += cnt
            for elem in unique_elements:
                frames_with_elem = np.any(element_data == elem, axis=1)
                row_count = np.sum(frames_with_elem)
                element_counts[elem]["frames"] += row_count
        element_counts = dict(element_counts)
        return element_counts

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
