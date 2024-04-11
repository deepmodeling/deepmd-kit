# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.pt.model.descriptor.base_descriptor import (
    BaseDescriptor,
)


class DPModel:
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
