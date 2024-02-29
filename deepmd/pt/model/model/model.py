# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
)

from deepmd.dpmodel.model.base_model import (
    make_base_model,
)
from deepmd.utils.path import (
    DPPath,
)


# trick: torch.nn.Module should not be inherbited here, otherwise,
# the abstract method will override the method from the atomic model
# as Python resolves method lookups using the C3 linearisation.
# See https://stackoverflow.com/a/47117600/9567349
# Take an example, this is the situation for only inheriting make_model():
#       torch.nn.Module        BaseAtomicModel        make_model()
#             |                       |                    |
#             -------------------------                    |
#                         |                                |
#                    DPAtomicModel                      BaseModel
#                         |                                |
#                make_model(DPAtomicModel)                 |
#                         |                                |
#                         ----------------------------------
#                                           |
#                                         DPModel
#
# The order is: DPModel -> make_model(DPAtomicModel) -> DPAtomicModel ->
# torch.nn.Module -> BaseAtomicModel -> BaseModel -> make_model()
#
# However, if BaseModel also inherbits from torch.nn.Module:
#         torch.nn.Module                      make_model()
#                |                                   |
#                |---------------------------        |
#                |                          |        |
#                |      BaseAtomicModel     |        |
#                |            |             |        |
#                |-------------             ----------
#                |                              |
#           DPAtomicModel                   BaseModel
#                |                              |
#                |                              |
#       make_model(DPAtomicModel)               |
#                |                              |
#                |                              |
#                --------------------------------
#                         |
#                         |
#                      DPModel
#
# The order is DPModel -> make_model(DPAtomicModel) -> DPAtomicModel ->
# BaseModel -> torch.nn.Module -> BaseAtomicModel -> make_model()
# BaseModel has higher proirity than BaseAtomicModel, which is not what
# we want.
# Alternatively, we can also make BaseAtomicModel in front of torch.nn.Module
# in DPAtomicModel (and other classes), but this requires the developer aware
# of it when developing it...
class BaseModel(make_base_model()):
    def __init__(self, *args, **kwargs):
        """Construct a basic model for different tasks."""
        super().__init__(*args, **kwargs)

    def compute_or_load_stat(
        self,
        sampled,
        stat_file_path: Optional[DPPath] = None,
    ):
        """
        Compute or load the statistics parameters of the model,
        such as mean and standard deviation of descriptors or the energy bias of the fitting net.
        When `sampled` is provided, all the statistics parameters will be calculated (or re-calculated for update),
        and saved in the `stat_file_path`(s).
        When `sampled` is not provided, it will check the existence of `stat_file_path`(s)
        and load the calculated statistics parameters.

        Parameters
        ----------
        sampled
            The sampled data frames from different data systems.
        stat_file_path
            The path to the statistics files.
        """
        raise NotImplementedError
