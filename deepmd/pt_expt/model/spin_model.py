# SPDX-License-Identifier: LGPL-3.0-or-later
import types
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
    _pad_nlist_for_export,
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
            nlist = _pad_nlist_for_export(nlist)
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

        # Force the sort branch of `_format_nlist` into the compiled graph by
        # overriding `need_sorted_nlist_for_lower` on the backbone (which is
        # where `call_common_lower` reads it).  Short-circuit `or` in
        # `_format_nlist` then skips the symbolic `n_nnei > nnei` comparison,
        # so no spurious shape guard is emitted.  See make_model.py for the
        # non-spin counterpart.
        backbone = model.backbone_model
        _orig_need_sort = backbone.need_sorted_nlist_for_lower
        backbone.need_sorted_nlist_for_lower = types.MethodType(
            lambda self: True, backbone
        )
        try:
            traced = make_fx(fn, **make_fx_kwargs)(
                extended_coord,
                extended_atype,
                extended_spin,
                nlist,
                mapping,
                fparam,
                aparam,
            )
        finally:
            backbone.need_sorted_nlist_for_lower = _orig_need_sort
        return traced

    def forward_common_lower_exportable_with_comm(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        extended_spin: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None,
        fparam: torch.Tensor | None,
        aparam: torch.Tensor | None,
        send_list: torch.Tensor,
        send_proc: torch.Tensor,
        recv_proc: torch.Tensor,
        send_num: torch.Tensor,
        recv_num: torch.Tensor,
        communicator: torch.Tensor,
        nlocal: torch.Tensor,
        nghost: torch.Tensor,
        do_atomic_virial: bool = False,
        **make_fx_kwargs: Any,
    ) -> torch.nn.Module:
        """Spin variant of ``forward_common_lower_exportable_with_comm``.

        Mirrors the non-spin version (see ``make_model.py``) but threads
        ``extended_spin`` through and injects ``has_spin`` into
        ``comm_dict`` so the pt_expt Repflow/Repformer override takes
        the spin branch (split real/virtual + concat_switch_virtual).
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
            send_list: torch.Tensor,
            send_proc: torch.Tensor,
            recv_proc: torch.Tensor,
            send_num: torch.Tensor,
            recv_num: torch.Tensor,
            communicator: torch.Tensor,
            nlocal: torch.Tensor,
            nghost: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            extended_coord = extended_coord.detach().requires_grad_(True)
            # Same nnei-dynamic-axis workaround as the regular variant.
            nlist = _pad_nlist_for_export(nlist)
            comm_dict = {
                "send_list": send_list,
                "send_proc": send_proc,
                "recv_proc": recv_proc,
                "send_num": send_num,
                "recv_num": recv_num,
                "communicator": communicator,
                "nlocal": nlocal,
                "nghost": nghost,
                # Trace-time marker so the override takes the spin path.
                # Value is irrelevant — only key presence matters.
                "has_spin": torch.tensor(
                    [1],
                    dtype=torch.int32,
                    device=extended_coord.device,
                ),
            }
            return model.forward_common_lower(
                extended_coord,
                extended_atype,
                extended_spin,
                nlist,
                mapping,
                fparam=fparam,
                aparam=aparam,
                do_atomic_virial=do_atomic_virial,
                comm_dict=comm_dict,
            )

        # Force the sort branch in ``_format_nlist`` so the compiled
        # graph's ``nnei`` axis stays dynamic (mirrors the regular
        # spin variant; backbone-level override is required).
        backbone = self.backbone_model
        _orig_need_sort = backbone.need_sorted_nlist_for_lower
        backbone.need_sorted_nlist_for_lower = types.MethodType(
            lambda self: True, backbone
        )
        try:
            traced = make_fx(fn, **make_fx_kwargs)(
                extended_coord,
                extended_atype,
                extended_spin,
                nlist,
                mapping,
                fparam,
                aparam,
                send_list,
                send_proc,
                recv_proc,
                send_num,
                recv_num,
                communicator,
                nlocal,
                nghost,
            )
        finally:
            backbone.need_sorted_nlist_for_lower = _orig_need_sort
        return traced

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

        data = data.copy()
        data.pop("type", None)
        backbone_model_obj = make_model(
            DPEnergyAtomicModel, T_Bases=(BaseModel,)
        ).deserialize(data["backbone_model"])
        spin = Spin.deserialize(data["spin"])
        return cls(
            backbone_model=backbone_model_obj,
            spin=spin,
        )
