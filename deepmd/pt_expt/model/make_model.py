# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import torch

from deepmd.dpmodel.atomic_model.base_atomic_model import (
    BaseAtomicModel,
)
from deepmd.dpmodel.model.make_model import make_model as make_model_dp
from deepmd.pt_expt.common import (
    dpmodel_setattr,
)

from .transform_output import (
    fit_output_to_model_output,
)


def make_model(T_AtomicModel: type[BaseAtomicModel]) -> type:
    """Make a model as a derived class of an atomic model.

    Wraps dpmodel's make_model with torch.nn.Module and overrides
    forward_common_atomic to use autograd-based derivatives.

    Parameters
    ----------
    T_AtomicModel
        The atomic model.

    Returns
    -------
    CM
        The model.

    """
    DPModel = make_model_dp(T_AtomicModel)

    class CM(DPModel, torch.nn.Module):
        def __init__(
            self,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            torch.nn.Module.__init__(self)
            DPModel.__init__(self, *args, **kwargs)

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            # Ensure torch.nn.Module.__call__ drives forward() for export/tracing.
            return torch.nn.Module.__call__(self, *args, **kwargs)

        def __setattr__(self, name: str, value: Any) -> None:
            handled, value = dpmodel_setattr(self, name, value)
            if not handled:
                super().__setattr__(name, value)

        def forward(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
            """Default forward delegates to call().

            Subclasses (e.g. EnergyModel) override this with output translation.
            """
            return self.call(*args, **kwargs)

        def forward_common_atomic(
            self,
            extended_coord: torch.Tensor,
            extended_atype: torch.Tensor,
            nlist: torch.Tensor,
            mapping: torch.Tensor | None = None,
            fparam: torch.Tensor | None = None,
            aparam: torch.Tensor | None = None,
            do_atomic_virial: bool = False,
        ) -> dict[str, torch.Tensor]:
            atomic_ret = self.atomic_model.forward_common_atomic(
                extended_coord,
                extended_atype,
                nlist,
                mapping=mapping,
                fparam=fparam,
                aparam=aparam,
            )
            return fit_output_to_model_output(
                atomic_ret,
                self.atomic_output_def(),
                extended_coord,
                do_atomic_virial=do_atomic_virial,
                create_graph=self.training,
                mask=atomic_ret.get("mask"),
            )

    return CM
