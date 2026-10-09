# SPDX-License-Identifier: LGPL-3.0-or-later


from deepmd.dpmodel.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.dpmodel.fitting.base_fitting import (
    BaseFitting,
)
from deepmd.dpmodel.utils.neighbor_contract import (
    NeighborContract,
)
from deepmd.dpmodel.utils.update_sel import (
    UpdateSel,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)


# use "class" to resolve "Variable not allowed in type expression"
class DPModelCommon:
    r"""Common methods for DP models.

    This class provides common functionality for DeepPot models, including
    neighbor selection updates and fitting network access.
    """

    @classmethod
    def neighbor_contract_from_jdata(cls, local_jdata: dict) -> NeighborContract:
        """Resolve the model neighbor contract from config before construction."""
        return BaseDescriptor.neighbor_contract_from_jdata(local_jdata["descriptor"])

    @classmethod
    def prepare_neighbors(
        cls,
        train_data: DeepmdDataSystem,
        type_map: list[str] | None,
        local_jdata: dict,
    ) -> tuple[dict, float | None]:
        """Prepare neighbor config under the model neighbor contract.

        This is the single entrypoint preparation hook: it resolves the
        descriptor/model contract recursively, rewrites graph-native configs so
        they do not require capacity discovery, and only runs the legacy
        ``update_sel`` path when the contract still needs a dense capacity.
        """
        local_jdata_cpy = dict(local_jdata)
        descriptor_jdata = dict(local_jdata_cpy["descriptor"])
        contract = BaseDescriptor.neighbor_contract_from_jdata(descriptor_jdata)
        descriptor_jdata = BaseDescriptor.prepare_jdata_for_neighbor_contract(
            descriptor_jdata, contract
        )
        local_jdata_cpy["descriptor"] = descriptor_jdata
        if not contract.requires_capacity:
            # Capacity discovery is skipped. Still report min neighbor distance
            # when the descriptor exposes a cutoff (used by compression bounds).
            rcut = descriptor_jdata.get("rcut")
            if rcut is None:
                return local_jdata_cpy, None
            return local_jdata_cpy, float(UpdateSel().get_min_nbor_dist(train_data))
        local_jdata_cpy["descriptor"], min_nbor_dist = BaseDescriptor.update_sel(
            train_data, type_map, descriptor_jdata
        )
        return local_jdata_cpy, min_nbor_dist

    @classmethod
    def update_sel(
        cls,
        train_data: DeepmdDataSystem,
        type_map: list[str] | None,
        local_jdata: dict,
    ) -> tuple[dict, float | None]:
        """Update the selection and perform neighbor statistics.

        Parameters
        ----------
        train_data : DeepmdDataSystem
            data used to do neighbor statistics
        type_map : list[str], optional
            The name of each type of atoms
        local_jdata : dict
            The local data refer to the current class

        Returns
        -------
        dict
            The updated local data
        float
            The minimum distance between two atoms
        """
        # Prefer the contract-aware preparation hook so callers that still
        # invoke update_sel inherit graph-native capacity skipping.
        return cls.prepare_neighbors(train_data, type_map, local_jdata)

    def get_fitting_net(self) -> BaseFitting:
        """Get the fitting network."""
        return self.atomic_model.fitting_net

    def get_descriptor(self) -> BaseDescriptor:
        """Get the descriptor."""
        return self.atomic_model.descriptor
