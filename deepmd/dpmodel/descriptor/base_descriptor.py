# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Callable,
    Type,
)

import numpy as np

from deepmd.utils.plugin import (
    Plugin,
)

from .make_base_descriptor import (
    make_base_descriptor,
)


class BaseDescriptor(make_base_descriptor(np.ndarray, "call")):
    __plugins = Plugin()

    @staticmethod
    def register(key: str) -> Callable:
        """Register a descriptor plugin.

        Parameters
        ----------
        key : str
            the key of a descriptor

        Returns
        -------
        Descriptor
            the registered descriptor

        Examples
        --------
        >>> @Descriptor.register("some_descrpt")
            class SomeDescript(Descriptor):
                pass
        """
        return BaseDescriptor.__plugins.register(key)

    def __new__(cls, *args, **kwargs):
        if cls is BaseDescriptor:
            cls = cls.get_class_by_input(kwargs)
        return super().__new__(cls)

    @classmethod
    def get_class_by_input(cls, input: dict) -> Type["BaseDescriptor"]:
        try:
            descrpt_type = input["type"]
        except KeyError:
            raise KeyError("the type of descriptor should be set by `type`")
        if descrpt_type in BaseDescriptor.__plugins.plugins:
            return BaseDescriptor.__plugins.plugins[descrpt_type]
        else:
            raise RuntimeError("Unknown descriptor type: " + descrpt_type)

    @classmethod
    def deserialize(cls, data: dict) -> "BaseDescriptor":
        """Deserialize the model.

        There is no suffix in a native DP model, but it is important
        for the TF backend.

        Parameters
        ----------
        data : dict
            The serialized data

        Returns
        -------
        Descriptor
            The deserialized descriptor
        """
        if cls is BaseDescriptor:
            return BaseDescriptor.get_class_by_input(data).deserialize(data)
        raise NotImplementedError("Not implemented in class %s" % cls.__name__)
