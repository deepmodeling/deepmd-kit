# SPDX-License-Identifier: LGPL-3.0-or-later
import inspect
from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Any,
    List,
    Type,
)

from deepmd.utils.plugin import (
    PluginVariant,
    make_plugin_registry,
)


def make_base_model() -> Type[object]:
    class BaseBaseModel(ABC, PluginVariant, make_plugin_registry("model")):
        """Base class for final exported model that will be directly used for inference.

        The class defines some abstractmethods that will be directly called by the
        inference interface. If the final model class inherits some of those methods
        from other classes, `BaseModel` should be inherited as the last class to ensure
        the correct method resolution order.

        This class is backend-indepedent.

        See Also
        --------
        deepmd.dpmodel.model.base_model.BaseModel
            BaseModel class for DPModel backend.
        """

        def __new__(cls, *args, **kwargs):
            if inspect.isabstract(cls):
                cls = cls.get_class_by_type(kwargs.get("type", "standard"))
            return super().__new__(cls)

        @abstractmethod
        def __call__(self, *args: Any, **kwds: Any) -> Any:
            """Inference method.

            Parameters
            ----------
            *args : Any
                The input data for inference.
            **kwds : Any
                The input data for inference.

            Returns
            -------
            Any
                The output of the inference.
            """
            pass

        @abstractmethod
        def get_type_map(self) -> List[str]:
            """Get the type map."""

        @abstractmethod
        def get_rcut(self):
            """Get the cut-off radius."""

        @abstractmethod
        def get_dim_fparam(self):
            """Get the number (dimension) of frame parameters of this atomic model."""

        @abstractmethod
        def get_dim_aparam(self):
            """Get the number (dimension) of atomic parameters of this atomic model."""

        @abstractmethod
        def get_sel_type(self) -> List[int]:
            """Get the selected atom types of this model.

            Only atoms with selected atom types have atomic contribution
            to the result of the model.
            If returning an empty list, all atom types are selected.
            """

        @abstractmethod
        def is_aparam_nall(self) -> bool:
            """Check whether the shape of atomic parameters is (nframes, nall, ndim).

            If False, the shape is (nframes, nloc, ndim).
            """

        @abstractmethod
        def model_output_type(self) -> List[str]:
            """Get the output type for the model."""

        @abstractmethod
        def serialize(self) -> dict:
            """Serialize the model.

            Returns
            -------
            dict
                The serialized data
            """
            pass

        @classmethod
        def deserialize(cls, data: dict) -> "BaseBaseModel":
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
            if inspect.isabstract(cls):
                return cls.get_class_by_type(data["type"]).deserialize(data)
            raise NotImplementedError("Not implemented in class %s" % cls.__name__)

        model_def_script: str

        @abstractmethod
        def get_model_def_script(self) -> str:
            """Get the model definition script."""
            pass

        @abstractmethod
        def get_nnei(self) -> int:
            """Returns the total number of selected neighboring atoms in the cut-off radius."""
            # for C++ interface
            pass

        @abstractmethod
        def get_nsel(self) -> int:
            """Returns the total number of selected neighboring atoms in the cut-off radius."""
            pass

        @classmethod
        @abstractmethod
        def update_sel(cls, global_jdata: dict, local_jdata: dict):
            """Update the selection and perform neighbor statistics.

            Parameters
            ----------
            global_jdata : dict
                The global data, containing the training section
            local_jdata : dict
                The local data refer to the current class
            """
            cls = cls.get_class_by_type(local_jdata.get("type", "standard"))
            return cls.update_sel(global_jdata, local_jdata)

    return BaseBaseModel


class BaseModel(make_base_model()):
    """Base class for final exported model that will be directly used for inference.

    The class defines some abstractmethods that will be directly called by the
    inference interface. If the final model class inherbits some of those methods
    from other classes, `BaseModel` should be inherited as the last class to ensure
    the correct method resolution order.

    This class is for the DPModel backend.

    See Also
    --------
    deepmd.dpmodel.model.base_model.BaseBaseModel
        Backend-independent BaseModel class.
    """
