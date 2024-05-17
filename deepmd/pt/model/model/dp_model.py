# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import (
    annotations,
)

from deepmd.pt.model.descriptor.base_descriptor import (
    BaseDescriptor,
)


class DPModelCommon:
    """A base class to implement common methods for all the Models."""

    @classmethod
    def update_sel(cls, global_jdata: dict, local_jdata: dict):
        """Update the selection and perform neighbor statistics.

        Parameters
        ----------
        global_jdata : dict
            The global data, containing the training section
        local_jdata : dict
            The local data refer to the current class
        """
        local_jdata_cpy = local_jdata.copy()
        local_jdata_cpy["descriptor"] = BaseDescriptor.update_sel(
            global_jdata, local_jdata["descriptor"]
        )
        return local_jdata_cpy

    def get_fitting_net(self):
        """Get the fitting network."""
        return self.atomic_model.fitting_net

    def get_descriptor(self):
        """Get the descriptor."""
        return self.atomic_model.descriptor
