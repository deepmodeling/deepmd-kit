# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Any,
    Callable,
    List,
    Type,
)

from deepmd.common import (
    j_get_type,
)
from deepmd.utils.plugin import (
    Plugin,
)


class BaseBaseModel(ABC):
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
    def model_output_type(self) -> str:
        """Get the output type for the model."""


class BaseModel(BaseBaseModel):
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
