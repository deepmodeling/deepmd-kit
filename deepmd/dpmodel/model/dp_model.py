# SPDX-License-Identifier: LGPL-3.0-or-later


from deepmd.dpmodel.descriptor.base_descriptor import (
    BaseDescriptor,
)


# use "class" to resolve "Variable not allowed in type expression"
class DPModelCommon:
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
