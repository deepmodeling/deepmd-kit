# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
)

import torch

from deepmd.pt.model.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)


class DPModelCommon:
    """A base class to implement common methods for all the Models."""

    @classmethod
    def update_sel(
        cls,
        train_data: DeepmdDataSystem,
        type_map: Optional[list[str]],
        local_jdata: dict,
    ) -> tuple[dict, Optional[float]]:
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

    # sadly, use -> BaseFitting here will not make torchscript happy
    def get_fitting_net(self):  # noqa: ANN201
        """Get the fitting network."""
        return self.atomic_model.fitting_net

    def get_descriptor(self):  # noqa: ANN201
        """Get the descriptor."""
        return self.atomic_model.descriptor

    @torch.jit.export
    def set_eval_descriptor_hook(self, enable: bool) -> None:
        """Set the hook for evaluating descriptor and clear the cache for descriptor list."""
        self.atomic_model.set_eval_descriptor_hook(enable)

    @torch.jit.export
    def eval_descriptor(self) -> torch.Tensor:
        """Evaluate the descriptor."""
        return self.atomic_model.eval_descriptor()

    @torch.jit.export
    def set_eval_fitting_last_layer_hook(self, enable: bool) -> None:
        """Set the hook for evaluating fitting_last_layer and clear the cache for fitting_last_layer list."""
        self.atomic_model.set_eval_fitting_last_layer_hook(enable)

    @torch.jit.export
    def eval_fitting_last_layer(self) -> torch.Tensor:
        """Evaluate the fitting_last_layer."""
        return self.atomic_model.eval_fitting_last_layer()
