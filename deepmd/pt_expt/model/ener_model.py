# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import torch
from torch.fx.experimental.proxy_tensor import (
    make_fx,
)

from deepmd.dpmodel.model.dp_model import (
    DPModelCommon,
)
from deepmd.pt_expt.atomic_model import (
    DPEnergyAtomicModel,
)

from .make_model import (
    make_model,
)

DPEnergyModel_ = make_model(DPEnergyAtomicModel)


class EnergyModel(DPModelCommon, DPEnergyModel_):
    model_type = "ener"

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        DPModelCommon.__init__(self)
        DPEnergyModel_.__init__(self, *args, **kwargs)

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, torch.Tensor]:
        model_ret = self.call(
            coord,
            atype,
            box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )
        model_predict = {}
        model_predict["atom_energy"] = model_ret["energy"]
        model_predict["energy"] = model_ret["energy_redu"]
        if self.do_grad_r("energy"):
            model_predict["force"] = model_ret["energy_derv_r"].squeeze(-2)
        if self.do_grad_c("energy"):
            model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
            if do_atomic_virial:
                model_predict["atom_virial"] = model_ret["energy_derv_c"].squeeze(-3)
        if "mask" in model_ret:
            model_predict["mask"] = model_ret["mask"]
        return model_predict

    def _forward_lower(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, torch.Tensor]:
        model_ret = self.call_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )
        model_predict = {}
        model_predict["atom_energy"] = model_ret["energy"]
        model_predict["energy"] = model_ret["energy_redu"]
        if self.do_grad_r("energy"):
            model_predict["extended_force"] = model_ret["energy_derv_r"].squeeze(-2)
        if self.do_grad_c("energy"):
            model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
            if do_atomic_virial:
                model_predict["extended_virial"] = model_ret["energy_derv_c"].squeeze(
                    -3
                )
        if "mask" in model_ret:
            model_predict["mask"] = model_ret["mask"]
        return model_predict

    def forward_lower(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
    ) -> torch.nn.Module:
        """Trace ``_forward_lower`` into an exportable module.

        Uses ``make_fx`` to trace through ``torch.autograd.grad``,
        decomposing the backward pass into primitive ops.  The returned
        module can be passed directly to ``torch.export.export``.

        Parameters
        ----------
        extended_coord, extended_atype, nlist, mapping, fparam, aparam, do_atomic_virial
            Sample inputs with representative shapes (used for tracing).

        Returns
        -------
        torch.nn.Module
            A traced module whose ``forward`` accepts
            ``(extended_coord, extended_atype, nlist, mapping, fparam, aparam)``
            and returns a dict with the same keys as ``_forward_lower``.
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
            return model._forward_lower(
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                fparam=fparam,
                aparam=aparam,
                do_atomic_virial=do_atomic_virial,
            )

        return make_fx(fn)(
            extended_coord, extended_atype, nlist, mapping, fparam, aparam
        )
