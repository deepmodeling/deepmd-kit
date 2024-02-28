# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Callable,
    List,
    Optional,
    Union,
)

from deepmd.common import (
    j_get_type,
)
from deepmd.utils.path import (
    DPPath,
)
from deepmd.utils.plugin import (
    PluginVariant,
    make_plugin_registry,
)


def make_base_descriptor(
    t_tensor,
    fwd_method_name: str = "forward",
):
    """Make the base class for the descriptor.

    Parameters
    ----------
    t_tensor
        The type of the tensor. used in the type hint.
    fwd_method_name
        Name of the forward method. For dpmodels, it should be "call".
        For torch models, it should be "forward".

    """

    class BD(ABC, PluginVariant, make_plugin_registry("descriptor")):
        """Base descriptor provides the interfaces of descriptor."""

        def __new__(cls, *args, **kwargs):
            if cls is BD:
                cls = cls.get_class_by_type(j_get_type(kwargs, cls.__name__))
            return super().__new__(cls)

        @abstractmethod
        def get_rcut(self) -> float:
            """Returns the cut-off radius."""
            pass

        @abstractmethod
        def get_sel(self) -> List[int]:
            """Returns the number of selected neighboring atoms for each type."""
            pass

        def get_nsel(self) -> int:
            """Returns the total number of selected neighboring atoms in the cut-off radius."""
            return sum(self.get_sel())

        def get_nnei(self) -> int:
            """Returns the total number of selected neighboring atoms in the cut-off radius."""
            return self.get_nsel()

        @abstractmethod
        def get_ntypes(self) -> int:
            """Returns the number of element types."""
            pass

        @abstractmethod
        def get_dim_out(self) -> int:
            """Returns the output descriptor dimension."""
            pass

        @abstractmethod
        def get_dim_emb(self) -> int:
            """Returns the embedding dimension of g2."""
            pass

        @abstractmethod
        def mixed_types(self) -> bool:
            """Returns if the descriptor requires a neighbor list that distinguish different
            atomic types or not.
            """
            pass

        @abstractmethod
        def share_params(self, base_class, shared_level, resume=False):
            """Share the parameters of self to the base_class with shared_level."""
            pass

        def compute_input_stats(
            self, merged: Union[Callable, List[dict]], path: Optional[DPPath] = None
        ):
            """Update mean and stddev for descriptor elements."""
            raise NotImplementedError

        @abstractmethod
        def fwd(
            self,
            extended_coord,
            extended_atype,
            nlist,
            mapping: Optional[t_tensor] = None,
        ):
            """Calculate descriptor."""
            pass

        @abstractmethod
        def serialize(self) -> dict:
            """Serialize the obj to dict."""
            pass

        @classmethod
        def deserialize(cls, data: dict) -> "BD":
            """Deserialize the model.

            Parameters
            ----------
            data : dict
                The serialized data

            Returns
            -------
            BD
                The deserialized descriptor
            """
            if cls is BD:
                return BD.get_class_by_type(data["type"]).deserialize(data)
            raise NotImplementedError("Not implemented in class %s" % cls.__name__)

    setattr(BD, fwd_method_name, BD.fwd)
    delattr(BD, "fwd")

    return BD
