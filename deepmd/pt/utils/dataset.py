# SPDX-License-Identifier: LGPL-3.0-or-later


from torch.utils.data import (
    Dataset,
)

from deepmd.utils.data import (
    DeepmdData,
)


class DeepmdDataSetForLoader(Dataset):
    def __init__(
        self,
        system: str,
        type_map: str,
        shuffle=True,
    ):
        """Construct DeePMD-style dataset containing frames cross different systems.

        Args:
        - systems: Paths to systems.
        - batch_size: Max frame count in a batch.
        - type_map: Atom types.
        """
        self._type_map = type_map
        self._data_system = DeepmdData(
            sys_path=system, shuffle_test=shuffle, type_map=self._type_map
        )
        self.mixed_type = self._data_system.mixed_type
        self._ntypes = self._data_system.get_ntypes()
        self._natoms = self._data_system.get_natoms()
        self._natoms_vec = self._data_system.get_natoms_vec(self._ntypes)

    def __len__(self):
        return self._data_system.nframes

    def __getitem__(self, index):
        """Get a frame from the selected system."""
        b_data = self._data_system.get_item_torch(index)
        b_data["natoms"] = self._natoms_vec
        return b_data

    def add_data_requirement(self, dict_of_keys):
        """Add data requirement for this data system."""
        for data_key in dict_of_keys:
            self._data_system.add(
                data_key,
                dict_of_keys[data_key]["ndof"],
                atomic=dict_of_keys[data_key].get("atomic", False),
                must=dict_of_keys[data_key].get("must", False),
                high_prec=dict_of_keys[data_key].get("high_prec", False),
                type_sel=dict_of_keys[data_key].get("type_sel", None),
                repeat=dict_of_keys[data_key].get("repeat", 1),
                default=dict_of_keys[data_key].get("default", 0.0),
                dtype=dict_of_keys[data_key].get("dtype", None),
            )
