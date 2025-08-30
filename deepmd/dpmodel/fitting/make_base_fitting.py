# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Any,
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
            raise NotImplementedError

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
        ) -> dict[str, t_tensor]:
            """Calculate fitting."""
            raise NotImplementedError

        def compute_output_stats(self, merged) -> None:
            """Update the output bias for fitting net."""
            raise NotImplementedError

        @abstractmethod
        def get_type_map(self) -> list[str]:
            """Get the name to each type of atoms."""
            raise NotImplementedError

        @abstractmethod
        def change_type_map(
            self, type_map: list[str], model_with_new_type_stat=None
        ) -> None:
            """Change the type related params to new ones, according to `type_map` and the original one in the model.
            If there are new types in `type_map`, statistics will be updated accordingly to `model_with_new_type_stat` for these new types.
            """
            raise NotImplementedError

        @abstractmethod
        def serialize(self) -> dict[str, Any]:
            """Serialize the obj to dict."""
            raise NotImplementedError

        @classmethod
        def deserialize(cls, data: dict[str, Any]) -> "BF":
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
            # Note: This method should not be called during TorchScript compilation
            # It's only used for model serialization/deserialization
            raise NotImplementedError("deserialize not supported")

    setattr(BF, fwd_method_name, BF.fwd)
    delattr(BF, "fwd")

    return BF
