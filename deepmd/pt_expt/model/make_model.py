# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import torch
from torch.fx.experimental.proxy_tensor import (
    make_fx,
)

from deepmd.dpmodel.atomic_model.base_atomic_model import (
    BaseAtomicModel,
)
from deepmd.dpmodel.model.make_model import make_model as make_model_dp
from deepmd.pt_expt.common import (
    torch_module,
)

from .transform_output import (
    fit_output_to_model_output,
)


def make_model(
    T_AtomicModel: type[BaseAtomicModel],
    T_Bases: tuple[type, ...] = (),
) -> type:
    """Make a model as a derived class of an atomic model.

    Wraps dpmodel's make_model with torch.nn.Module and overrides
    forward_common_atomic to use autograd-based derivatives.

    Parameters
    ----------
    T_AtomicModel
        The atomic model.
    T_Bases
        Additional base classes for the returned model class.
        For example, pass ``(BaseModel,)`` so that the concrete model
        inherits the pt_expt ``BaseModel`` plugin registry.

    Returns
    -------
    CM
        The model.

    """
    DPModel = make_model_dp(T_AtomicModel)

    @torch_module
    class CM(DPModel, *T_Bases):
        def forward(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
            """Default forward delegates to call().

            Subclasses (e.g. EnergyModel) override this with output translation.
            """
            return self.call(*args, **kwargs)

        def forward_common(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
            """Forward common delegates to call_common()."""
            return self.call_common(*args, **kwargs)

        def forward_common_lower(
            self, *args: Any, **kwargs: Any
        ) -> dict[str, torch.Tensor]:
            """Forward common lower delegates to call_common_lower()."""
            return self.call_common_lower(*args, **kwargs)

        def forward_common_atomic(
            self,
            extended_coord: torch.Tensor,
            extended_atype: torch.Tensor,
            nlist: torch.Tensor,
            mapping: torch.Tensor | None = None,
            fparam: torch.Tensor | None = None,
            aparam: torch.Tensor | None = None,
            do_atomic_virial: bool = False,
            extended_coord_corr: torch.Tensor | None = None,
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
                extended_coord_corr=extended_coord_corr,
            )

        def forward_common_lower_exportable(
            self,
            extended_coord: torch.Tensor,
            extended_atype: torch.Tensor,
            nlist: torch.Tensor,
            mapping: torch.Tensor | None = None,
            fparam: torch.Tensor | None = None,
            aparam: torch.Tensor | None = None,
            do_atomic_virial: bool = False,
            **make_fx_kwargs: Any,
        ) -> torch.nn.Module:
            """Trace ``forward_common_lower`` into an exportable module.

            Uses ``make_fx`` to trace through ``torch.autograd.grad``,
            decomposing the backward pass into primitive ops.  The returned
            module can be passed directly to ``torch.export.export``.

            The output uses internal key names (e.g. ``energy``,
            ``energy_redu``, ``energy_derv_r``) so that
            ``communicate_extended_output`` can be applied at inference
            time.

            Parameters
            ----------
            extended_coord, extended_atype, nlist, mapping, fparam, aparam, do_atomic_virial
                Sample inputs with representative shapes (used for tracing).
            **make_fx_kwargs
                Extra keyword arguments forwarded to ``make_fx``
                (e.g. ``tracing_mode="symbolic"``).

            Returns
            -------
            torch.nn.Module
                A traced module whose ``forward`` accepts
                ``(extended_coord, extended_atype, nlist, mapping,
                fparam, aparam)`` and returns a dict with the same keys
                as ``call_common_lower``.
            """
            model = self

            def fn(
                extended_coord: torch.Tensor,
                extended_atype: torch.Tensor,
                nlist: torch.Tensor,
                mapping: torch.Tensor | None,
                fparam: torch.Tensor | None,
                aparam: torch.Tensor | None,
            ) -> dict[str, torch.Tensor]:
                extended_coord = extended_coord.detach().requires_grad_(True)
                return model.forward_common_lower(
                    extended_coord,
                    extended_atype,
                    nlist,
                    mapping,
                    fparam=fparam,
                    aparam=aparam,
                    do_atomic_virial=do_atomic_virial,
                )

            return make_fx(fn, **make_fx_kwargs)(
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                fparam,
                aparam,
            )

    return CM
