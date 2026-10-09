# SPDX-License-Identifier: LGPL-3.0-or-later

from deepmd.dpmodel.utils.neighbor_contract import (
    NeighborContract,
)
from deepmd.pt.model.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.pt.utils.update_sel import (
    UpdateSel,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)


class DPModelCommon:
    """A base class to implement common methods for all the Models."""

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
        """Prepare neighbor config under the model neighbor contract."""
        local_jdata_cpy = dict(local_jdata)
        descriptor_jdata = dict(local_jdata_cpy["descriptor"])
        contract = BaseDescriptor.neighbor_contract_from_jdata(descriptor_jdata)
        descriptor_jdata = BaseDescriptor.prepare_jdata_for_neighbor_contract(
            descriptor_jdata, contract
        )
        local_jdata_cpy["descriptor"] = descriptor_jdata
        if not contract.requires_capacity:
            if descriptor_jdata.get("rcut") is None:
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
        return cls.prepare_neighbors(train_data, type_map, local_jdata)

    # sadly, use -> BaseFitting here will not make torchscript happy
    def get_fitting_net(self):  # noqa: ANN201
        """Get the fitting network."""
        return self.atomic_model.fitting_net

    def get_descriptor(self):  # noqa: ANN201
        """Get the descriptor."""
        return self.atomic_model.descriptor
