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

from deepmd.dpmodel.output_def import (
    FittingOutputDef,
)


def make_base_atomic_model(T_Tensor):
    class BAM(ABC):
        """Base Atomic Model provides the interfaces of an atomic model."""

        @abstractmethod
        def fitting_output_def(self) -> FittingOutputDef:
            pass

        @abstractmethod
        def get_rcut(self) -> float:
            pass

        @abstractmethod
        def get_sel(self) -> List[int]:
            pass

        @abstractmethod
        def distinguish_types(self) -> bool:
            pass

        @abstractmethod
        def forward_atomic(
            self,
            extended_coord: T_Tensor,
            extended_atype: T_Tensor,
            nlist: T_Tensor,
            mapping: Optional[T_Tensor] = None,
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

        def do_grad(
            self,
            var_name: Optional[str] = None,
        ) -> bool:
            """Tell if the output variable `var_name` is differentiable.
            if var_name is None, returns if any of the variable is differentiable.

            """
            odef = self.fitting_output_def()
            if var_name is None:
                require: List[bool] = []
                for vv in odef.keys():
                    require.append(self.do_grad_(vv))
                return any(require)
            else:
                return self.do_grad_(var_name)

        def do_grad_(
            self,
            var_name: str,
        ) -> bool:
            """Tell if the output variable `var_name` is differentiable."""
            assert var_name is not None
            return self.fitting_output_def()[var_name].differentiable

    return BAM
