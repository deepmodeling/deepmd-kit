# SPDX-License-Identifier: LGPL-3.0-or-later
import inspect
import json
from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Any,
    Optional,
)

from deepmd.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.utils.plugin import (
    PluginVariant,
    make_plugin_registry,
)


def make_base_model() -> type[object]:
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
                # getting model type based on fitting type
                model_type = kwargs.get("type", "standard")
                if model_type == "standard":
                    model_type = kwargs.get("fitting", {}).get("type", "ener")
                cls = cls.get_class_by_type(model_type)
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
        def get_type_map(self) -> list[str]:
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
        def get_sel_type(self) -> list[int]:
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
        def model_output_type(self) -> list[str]:
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
                model_type = data.get("type", "standard")
                if model_type == "standard":
                    model_type = data.get("fitting", {}).get("type", "ener")
                return cls.get_class_by_type(model_type).deserialize(data)
            raise NotImplementedError(f"Not implemented in class {cls.__name__}")

        model_def_script: str
        """The model definition script."""
        min_nbor_dist: Optional[float]
        """The minimum distance between two atoms. Used for model compression.
        None when skipping neighbor statistics.
        """

        @abstractmethod
        def get_model_def_script(self) -> str:
            """Get the model definition script."""
            pass

        def get_min_nbor_dist(self) -> Optional[float]:
            """Get the minimum distance between two atoms."""
            return self.min_nbor_dist

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
            # getting model type based on fitting type
            model_type = local_jdata.get("type", "standard")
            if model_type == "standard":
                model_type = local_jdata.get("fitting", {}).get("type", "ener")
            cls = cls.get_class_by_type(model_type)
            return cls.update_sel(train_data, type_map, local_jdata)

        def enable_compression(
            self,
            table_extrapolate: float = 5,
            table_stride_1: float = 0.01,
            table_stride_2: float = 0.1,
            check_frequency: int = -1,
        ) -> None:
            """Enable model compression by tabulation.

            Parameters
            ----------
            table_extrapolate
                The scale of model extrapolation
            table_stride_1
                The uniform stride of the first table
            table_stride_2
                The uniform stride of the second table
            check_frequency
                The overflow check frequency
            """
            raise NotImplementedError("This atomic model doesn't support compression!")

        @classmethod
        def get_model(cls, model_params: dict) -> "BaseBaseModel":
            """Get the model by the parameters.

            By default, all the parameters are directly passed to the constructor.
            If not, override this method.

            Parameters
            ----------
            model_params : dict
                The model parameters

            Returns
            -------
            BaseBaseModel
                The model
            """
            model_params_old = model_params.copy()
            model_params = model_params.copy()
            model_params.pop("type", None)
            model = cls(**model_params)
            model.model_def_script = json.dumps(model_params_old)
            return model

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

    def __init__(self) -> None:
        self.model_def_script = ""
        self.min_nbor_dist = None

    def get_model_def_script(self) -> str:
        """Get the model definition script."""
        return self.model_def_script
