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

    The model takes atomic predictions from the atomic model and computes
    global properties by reduction and differentiation:

    **Reduction** (for reducible quantities like energy):

    .. math::
        E = \sum_{i=1}^{N} E^i,

    where :math:`E^i` is the atomic energy from the atomic model.

    **Differentiation** (for forces and virials):

    .. math::
        \mathbf{F}_i = -\frac{\partial E}{\partial \mathbf{r}_i},

    .. math::
        \boldsymbol{\Xi} = -\sum_{i=1}^{N} \frac{\partial E^i}{\partial \mathbf{r}_i} \otimes \mathbf{r}_i,

    where :math:`\mathbf{F}_i` is the force on atom :math:`i` and
    :math:`\boldsymbol{\Xi}` is the virial tensor.
    """

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
