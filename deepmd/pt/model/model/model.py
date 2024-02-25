# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    abstractmethod,
)
from typing import (
    Callable,
    Optional,
    Type,
)

from deepmd.common import (
    j_get_type,
)
from deepmd.dpmodel.model.base_model import (
    BaseBaseModel,
)
from deepmd.utils.path import (
    DPPath,
)
from deepmd.utils.plugin import (
    Plugin,
)


# trick: torch.nn.Module should not be inherbited here, otherwise,
# the abstract method will override the method from the atomic model
# as Python resolves method lookups using the C3 linearisation.
# See https://stackoverflow.com/a/47117600/9567349
# Take an example, this is sitatuion for only inherbiting BaseBaseModel:
#       torch.nn.Module        BaseAtomicModel        BaseBaseModel
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
# torch.nn.Module -> BaseAtomicModel -> BaseBaseModel
#
# However, if BaseModel also inherbits from torch.nn.Module:
#         torch.nn.Module                      BaseBaseModel
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
# BaseModel -> torch.nn.Module -> BaseAtomicModel -> BaseBaseModel
# BaseModel has higher proirity than BaseAtomicModel, which is not what
# we want.
# Alternatively, we can also make BaseAtomicModel in front of torch.nn.Module
# in DPAtomicModel (and other classes), but this requires the developer aware
# of it when developing it...
class BaseModel(BaseBaseModel):
    __plugins = Plugin()

    @staticmethod
    def register(key: str) -> Callable[[object], object]:
        """Register a descriptor plugin.

        Parameters
        ----------
        key : str
            the key of a descriptor

        Returns
        -------
        callable[[object], object]
            the registered descriptor

        Examples
        --------
        >>> @Fitting.register("some_fitting")
            class SomeFitting(Fitting):
                pass
        """
        return BaseModel.__plugins.register(key)

    def __new__(cls, *args, **kwargs):
        if cls is BaseModel:
            cls = cls.get_class_by_type(j_get_type(kwargs, cls.__name__))
        return super().__new__(cls)

    @classmethod
    def get_class_by_type(cls, model_type: str) -> Type["BaseModel"]:
        if model_type in BaseModel.__plugins.plugins:
            return BaseModel.__plugins.plugins[model_type]
        else:
            raise RuntimeError("Unknown model type: " + model_type)

    @classmethod
    def deserialize(cls, data: dict) -> "BaseModel":
        """Deserialize the model.

        Parameters
        ----------
        data : dict
            The serialized data

        Returns
        -------
        BaseModel
            The deserialized model
        """
        if cls is BaseModel:
            return BaseModel.get_class_by_type(data["type"]).deserialize(data)
        raise NotImplementedError("Not implemented in class %s" % cls.__name__)

    def __init__(self):
        """Construct a basic model for different tasks."""
        super().__init__()

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

    model_def_script: str

    # currently, only pt needs the following methods
    @abstractmethod
    def get_model_def_script(self) -> str:
        """Get the model definition script."""
        raise NotImplementedError

    @abstractmethod
    def get_nnei(self) -> int:
        """Returns the total number of selected neighboring atoms in the cut-off radius."""
        # for C++ interface
        raise NotImplementedError

    @abstractmethod
    def get_nsel(self) -> int:
        """Returns the total number of selected neighboring atoms in the cut-off radius."""
        raise NotImplementedError
