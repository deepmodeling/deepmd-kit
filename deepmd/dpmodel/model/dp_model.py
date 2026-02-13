# SPDX-License-Identifier: LGPL-3.0-or-later


from deepmd.dpmodel.array_api import (
    Array,
)
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
        return self.atomic_model.fitting

    def get_descriptor(self) -> BaseDescriptor:
        """Get the descriptor."""
        return self.atomic_model.descriptor

    def set_eval_descriptor_hook(self, enable: bool) -> None:
        """Set the hook for evaluating descriptor."""
        self.atomic_model.set_eval_descriptor_hook(enable)

    def eval_descriptor(self) -> Array:
        """Evaluate the descriptor."""
        return self.atomic_model.eval_descriptor()

    def set_eval_fitting_last_layer_hook(self, enable: bool) -> None:
        """Set the hook for evaluating fitting last layer output."""
        self.atomic_model.set_eval_fitting_last_layer_hook(enable)

    def eval_fitting_last_layer(self) -> Array:
        """Evaluate the fitting last layer output."""
        return self.atomic_model.eval_fitting_last_layer()
