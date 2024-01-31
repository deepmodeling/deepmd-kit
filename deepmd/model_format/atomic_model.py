# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    ABC,
    abstractclassmethod,
    abstractmethod,
)
from typing import (
    Dict,
    List,
    Optional,
)

from deepmd.model_format import (
    FittingOutputDef,
)


def make_base_atomic_model(T_Tensor):
    class BAM(ABC):
        """Base Atomic Model provides the interfaces of an atomic model."""

        @abstractmethod
        def get_fitting_output_def(self) -> FittingOutputDef:
            raise NotImplementedError

        @abstractmethod
        def get_rcut(self) -> float:
            raise NotImplementedError

        @abstractmethod
        def get_sel(self) -> List[int]:
            raise NotImplementedError

        @abstractmethod
        def distinguish_types(self) -> bool:
            raise NotImplementedError

        @abstractmethod
        def forward_atomic(
            self,
            extended_coord: T_Tensor,
            extended_atype: T_Tensor,
            nlist: T_Tensor,
            mapping: Optional[T_Tensor] = None,
            do_atomic_virial: bool = False,
        ) -> Dict[str, T_Tensor]:
            raise NotImplementedError

        @abstractmethod
        def serialize(self) -> dict:
            raise NotImplementedError

        @abstractclassmethod
        def deserialize(cls):
            raise NotImplementedError

    return BAM
