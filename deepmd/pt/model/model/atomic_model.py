# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    List,
    Optional,
)

import torch

from deepmd.model_format.atomic_model import (
    make_base_atomic_model,
)

BaseAtomicModel = make_base_atomic_model(torch.Tensor)


class AtomicModel(BaseAtomicModel):
    """Common base class for atomic model."""

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
