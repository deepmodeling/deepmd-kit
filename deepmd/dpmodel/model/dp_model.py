# SPDX-License-Identifier: LGPL-3.0-or-later


from deepmd.dpmodel.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.dpmodel.fitting.base_fitting import (
    BaseFitting,
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

    def adam_route_patterns(self) -> list[str]:
        """
        Name patterns of the parameters that take the AdamW path under HybridMuon.

        Returns
        -------
        list[str]
            Substrings of parameter names, composed from the tensors the
            descriptor declares through its own ``adam_route_patterns``: rows
            of these matrices that correspond to rarely visited inputs receive
            almost no gradient, and Adam moves each row with its own gradient
            history, whereas Muon's orthogonalized update moves every row of a
            matrix at the same rate.
        """
        descriptor = getattr(getattr(self, "atomic_model", None), "descriptor", None)
        declared = getattr(descriptor, "adam_route_patterns", None)
        if declared is None:
            return []
        return [f"descriptor.{p}" for p in declared()]

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
        local_jdata_cpy = local_jdata.copy()
        local_jdata_cpy["descriptor"], min_nbor_dist = BaseDescriptor.update_sel(
            train_data, type_map, local_jdata["descriptor"]
        )
        return local_jdata_cpy, min_nbor_dist

    def get_fitting_net(self) -> BaseFitting:
        """Get the fitting network."""
        return self.atomic_model.fitting_net

    def get_descriptor(self) -> BaseDescriptor:
        """Get the descriptor."""
        return self.atomic_model.descriptor
