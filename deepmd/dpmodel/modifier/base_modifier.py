# SPDX-License-Identifier: LGPL-3.0-or-later
import inspect
from abc import (
    ABC,
    abstractmethod,
)

from deepmd.utils.plugin import (
    PluginVariant,
    make_plugin_registry,
)


def make_base_modifier() -> type[object]:
    class BaseModifier(ABC, PluginVariant, make_plugin_registry("modifier")):
        """Base class for data modifier."""

        def __new__(cls, *args, **kwargs):
            if cls is BaseModifier:
                cls = cls.get_class_by_type(kwargs["type"])
            return super().__new__(cls)

        @abstractmethod
        def serialize(self) -> dict:
            """Serialize the modifier.

            Returns
            -------
            dict
                The serialized data
            """
            pass

        @classmethod
        def deserialize(cls, data: dict) -> "BaseModifier":
            """Deserialize the modifier.

            Parameters
            ----------
            data : dict
                The serialized data

            Returns
            -------
            BaseModel
                The deserialized modifier
            """
            if inspect.isabstract(cls):
                return cls.get_class_by_type(data["type"]).deserialize(data)
            raise NotImplementedError(f"Not implemented in class {cls.__name__}")

        @classmethod
        def get_modifier(cls, modifier_params: dict) -> "BaseModifier":
            """Get the modifier by the parameters.

            By default, all the parameters are directly passed to the constructor.
            If not, override this method.

            Parameters
            ----------
            modifier_params : dict
                The modifier parameters

            Returns
            -------
            BaseModifier
                The modifier
            """
            modifier_params = modifier_params.copy()
            modifier_params.pop("type", None)
            modifier = cls(**modifier_params)
            return modifier

    return BaseModifier
