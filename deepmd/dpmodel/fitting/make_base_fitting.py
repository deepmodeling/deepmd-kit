# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    ABC,
    abstractclassmethod,
    abstractmethod,
)
from typing import (
    Dict,
    Optional,
)

from deepmd.dpmodel.output_def import (
    FittingOutputDef,
)


def make_base_fitting(
    T_Tensor,
    FWD_Method: str = "call",
):
    """Make the base class for the fitting."""

    class BF(ABC):
        """Base fitting provides the interfaces of fitting net."""

        @abstractmethod
        def output_def(self) -> FittingOutputDef:
            pass

        @abstractmethod
        def fwd(
            self,
            descriptor: T_Tensor,
            atype: T_Tensor,
            gr: Optional[T_Tensor] = None,
            g2: Optional[T_Tensor] = None,
            h2: Optional[T_Tensor] = None,
            fparam: Optional[T_Tensor] = None,
            aparam: Optional[T_Tensor] = None,
        ) -> Dict[str, T_Tensor]:
            pass

        @abstractmethod
        def serialize(self) -> dict:
            pass

        @abstractclassmethod
        def deserialize(cls):
            pass

    setattr(BF, FWD_Method, BF.fwd)
    delattr(BF, "fwd")

    return BF
