# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import torch
from torch.fx.experimental.proxy_tensor import (
    make_fx,
)

from deepmd.dpmodel.model.spin_model import SpinModel as SpinModelDP
from deepmd.pt_expt.common import (
    torch_module,
)
from deepmd.utils.spin import (
    Spin,
)

from .make_model import (
    make_model,
)
from .model import (
    BaseModel,
)


@torch_module
class SpinModel(SpinModelDP):
    def __getattr__(self, name: str) -> Any:
        """Get attribute from the wrapped model.

        In torch.nn.Module, submodules are stored in _modules, not __dict__.
        Override the dpmodel version to use torch.nn.Module's __getattr__
        first (which checks _parameters, _buffers, _modules), then fall
        back to backbone_model delegation for arbitrary attributes.
        """
        try:
            return torch.nn.Module.__getattr__(self, name)
        except AttributeError:
            pass
        # backbone_model is in _modules, access via _modules directly
        # to avoid re-entering __getattr__
        modules = self.__dict__.get("_modules", {})
        backbone = modules.get("backbone_model")
        if backbone is not None:
            return getattr(backbone, name)
        raise AttributeError(name)

    def forward_common_lower_exportable(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        extended_spin: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        **make_fx_kwargs: Any,
    ) -> torch.nn.Module:
        """Trace ``call_common_lower`` into an exportable module.

        Uses ``make_fx`` to trace through ``torch.autograd.grad``,
        decomposing the backward pass into primitive ops.  The returned
        module can be passed directly to ``torch.export.export``.

        The output uses internal key names (e.g. ``energy``,
        ``energy_redu``, ``energy_derv_r``) so that subclasses can
        apply their own key translation on top.

        Parameters
        ----------
        extended_coord, extended_atype, extended_spin, nlist, mapping, fparam, aparam, do_atomic_virial
            Sample inputs with representative shapes (used for tracing).
        **make_fx_kwargs
            Extra keyword arguments forwarded to ``make_fx``
            (e.g. ``tracing_mode="symbolic"``).

        Returns
        -------
        torch.nn.Module
            A traced module whose ``forward`` accepts
            ``(extended_coord, extended_atype, extended_spin, nlist,
            mapping, fparam, aparam)`` and returns a dict with the same
            keys as ``call_common_lower``.
        """
        model = self

        def fn(
            extended_coord: torch.Tensor,
            extended_atype: torch.Tensor,
            extended_spin: torch.Tensor,
            nlist: torch.Tensor,
            mapping: torch.Tensor | None,
            fparam: torch.Tensor | None,
            aparam: torch.Tensor | None,
        ) -> dict[str, torch.Tensor]:
            extended_coord = extended_coord.detach().requires_grad_(True)
            return model.forward_common_lower(
                extended_coord,
                extended_atype,
                extended_spin,
                nlist,
                mapping,
                fparam=fparam,
                aparam=aparam,
                do_atomic_virial=do_atomic_virial,
            )

        return make_fx(fn, **make_fx_kwargs)(
            extended_coord,
            extended_atype,
            extended_spin,
            nlist,
            mapping,
            fparam,
            aparam,
        )

    def forward_common_lower(
        self, *args: Any, **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        """Forward common lower delegates to call_common_lower()."""
        return self.call_common_lower(*args, **kwargs)

    @classmethod
    def deserialize(cls, data: dict) -> "SpinModel":
        from deepmd.dpmodel.atomic_model import (
            DPEnergyAtomicModel,
        )

        backbone_model_obj = make_model(
            DPEnergyAtomicModel, T_Bases=(BaseModel,)
        ).deserialize(data["backbone_model"])
        spin = Spin.deserialize(data["spin"])
        return cls(
            backbone_model=backbone_model_obj,
            spin=spin,
        )
