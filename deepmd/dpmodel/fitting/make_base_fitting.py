# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Dict,
    Optional,
)

from deepmd.common import (
    j_get_type,
)
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
)
from deepmd.utils.plugin import (
    PluginVariant,
    make_plugin_registry,
)


def make_base_fitting(
    t_tensor,
    fwd_method_name: str = "forward",
):
    """Make the base class for the fitting.

    Parameters
    ----------
    t_tensor
        The type of the tensor. used in the type hint.
    fwd_method_name
        Name of the forward method. For dpmodels, it should be "call".
        For torch models, it should be "forward".

    """

    class BF(ABC, PluginVariant, make_plugin_registry("fitting")):
        """Base fitting provides the interfaces of fitting net."""

        def __new__(cls, *args, **kwargs):
            if cls is BF:
                cls = cls.get_class_by_type(j_get_type(kwargs, cls.__name__))
            return super().__new__(cls)

        @abstractmethod
        def output_def(self) -> FittingOutputDef:
            """Returns the output def of the fitting net."""
            pass

        @abstractmethod
        def fwd(
            self,
            descriptor: t_tensor,
            atype: t_tensor,
            gr: Optional[t_tensor] = None,
            g2: Optional[t_tensor] = None,
            h2: Optional[t_tensor] = None,
            fparam: Optional[t_tensor] = None,
            aparam: Optional[t_tensor] = None,
        ) -> Dict[str, t_tensor]:
            """Calculate fitting."""
            pass

        def compute_output_stats(self, merged):
            """Update the output bias for fitting net."""
            raise NotImplementedError

        def init_fitting_stat(self, **kwargs):
            """Initialize the model bias by the statistics."""
            raise NotImplementedError

        @abstractmethod
        def serialize(self) -> dict:
            """Serialize the obj to dict."""
            pass

        @classmethod
        def deserialize(cls, data: dict) -> "BF":
            """Deserialize the fitting.

            Parameters
            ----------
            data : dict
                The serialized data

            Returns
            -------
            BF
                The deserialized fitting
            """
            if cls is BF:
                return BF.get_class_by_type(data["type"]).deserialize(data)
            raise NotImplementedError("Not implemented in class %s" % cls.__name__)

    setattr(BF, fwd_method_name, BF.fwd)
    delattr(BF, "fwd")

    return BF
