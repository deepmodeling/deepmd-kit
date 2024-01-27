# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Dict,
    List,
    Optional,
)

import torch

from deepmd.model_format import (
    FittingOutputDef,
)


class AtomicModel(ABC):
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
        extended_coord,
        extended_atype,
        nlist,
        mapping: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def do_grad(
        self,
        var_name: Optional[str] = None,
    ) -> bool:
        """Tell if the output variable `var_name` is differentiable.
        if var_name is None, returns if any of the variable is differentiable.

        """
        odef = self.get_fitting_output_def()
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
        return self.get_fitting_output_def()[var_name].differentiable
