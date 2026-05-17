# SPDX-License-Identifier: LGPL-3.0-or-later
import math
import types
from typing import (
    Any,
)

import torch
from torch.fx.experimental.proxy_tensor import (
    make_fx,
)

from deepmd.dpmodel import (
    get_hessian_name,
)
from deepmd.dpmodel.atomic_model.base_atomic_model import (
    BaseAtomicModel,
)
from deepmd.dpmodel.model.make_model import make_model as make_model_dp
from deepmd.dpmodel.output_def import (
    OutputVariableDef,
)
from deepmd.pt_expt.common import (
    torch_module,
)

from .transform_output import (
    fit_output_to_model_output,
)


def _pad_nlist_for_export(nlist: torch.Tensor) -> torch.Tensor:
    """Append a single ``-1`` column to ``nlist`` for export-time tracing.

    Used inside ``forward_common_lower_exportable`` (and its spin counterpart)
    so that ``_format_nlist``'s terminal slice ``ret[..., :nnei]`` truncates
    to a statically sized output.  Without the extra column, torch.export
    cannot prove the ``ret.shape[-1] == nnei`` assertion at trace time and
    would specialise the dynamic ``nnei`` dim to the sample value.

    Combined with the short-circuit order in ``_format_nlist``
    (``extra_nlist_sort`` on the left) and the ``need_sorted_nlist_for_lower``
    override during tracing, this keeps the compiled graph's ``nnei`` axis
    fully dynamic and free of symbolic shape guards.
    """
    pad = -torch.ones(
        (*nlist.shape[:2], 1),
        dtype=nlist.dtype,
        device=nlist.device,
    )
    return torch.cat([nlist, pad], dim=-1)


def _cal_hessian_ext(
    model: Any,
    kk: str,
    vdef: OutputVariableDef,
    extended_coord: torch.Tensor,
    extended_atype: torch.Tensor,
    nlist: torch.Tensor,
    mapping: torch.Tensor | None,
    fparam: torch.Tensor | None,
    aparam: torch.Tensor | None,
    create_graph: bool = False,
) -> torch.Tensor:
    """Compute hessian of reduced output w.r.t. extended coordinates.

    Mirrors the JAX approach: compute hessian on extended coordinates,
    then let communicate_extended_output map nall->nloc.

    Parameters
    ----------
    model
        The model (CM instance). Must have ``atomic_model.forward_common_atomic``.
    kk
        The output key (e.g. "energy").
    vdef
        The output variable definition.
    extended_coord
        Extended coordinates. Shape: [nf, nall, 3].
    extended_atype
        Extended atom types. Shape: [nf, nall].
    nlist
        Neighbor list. Shape: [nf, nloc, nsel].
    mapping
        Mapping from extended to local. Shape: [nf, nall] or None.
    fparam
        Frame parameters. Shape: [nf, nfp] or None.
    aparam
        Atomic parameters. Shape: [nf, nloc, nap] or None.
    create_graph
        Whether to create graph for higher-order derivatives.

    Returns
    -------
    torch.Tensor
        Hessian on extended coordinates. Shape: [nf, *def, nall, 3, nall, 3].
    """
    nf, nall, _ = extended_coord.shape
    vsize = math.prod(vdef.shape)
    coord_flat = extended_coord.reshape(nf, nall * 3)
    hessians = []
    for ii in range(nf):
        for ci in range(vsize):
            wrapper = _WrapperForwardEnergy(
                model,
                kk,
                ci,
                nall,
                extended_atype[ii],
                nlist[ii],
                mapping[ii] if mapping is not None else None,
                fparam[ii] if fparam is not None else None,
                aparam[ii] if aparam is not None else None,
            )
            hess = torch.autograd.functional.hessian(
                wrapper,
                coord_flat[ii],
                create_graph=create_graph,
            )
            hessians.append(hess)
    # [nf * vsize, nall*3, nall*3] -> [nf, *vshape, nall, 3, nall, 3]
    result = torch.stack(hessians).reshape(nf, *vdef.shape, nall, 3, nall, 3)
    return result


class _WrapperForwardEnergy:
    """Callable wrapper for torch.autograd.functional.hessian.

    Given flattened extended coordinates, recomputes the reduced energy
    for one frame and one output component.
    """

    def __init__(
        self,
        model: Any,
        kk: str,
        ci: int,
        nall: int,
        atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None,
        fparam: torch.Tensor | None,
        aparam: torch.Tensor | None,
    ) -> None:
        self.model = model
        self.kk = kk
        self.ci = ci
        self.nall = nall
        self.atype = atype
        self.nlist = nlist
        self.mapping = mapping
        self.fparam = fparam
        self.aparam = aparam

    def __call__(self, coord_flat: torch.Tensor) -> torch.Tensor:
        """Compute scalar reduced energy for one frame, one component.

        Parameters
        ----------
        coord_flat
            Flattened extended coordinates for one frame. Shape: [nall * 3].

        Returns
        -------
        torch.Tensor
            Scalar energy component.
        """
        cc_3d = coord_flat.reshape(1, self.nall, 3)
        atomic_ret = self.model.atomic_model.forward_common_atomic(
            cc_3d,
            self.atype.unsqueeze(0),
            self.nlist.unsqueeze(0),
            mapping=self.mapping.unsqueeze(0) if self.mapping is not None else None,
            fparam=self.fparam.unsqueeze(0) if self.fparam is not None else None,
            aparam=self.aparam.unsqueeze(0) if self.aparam is not None else None,
        )
        # atomic_ret[kk]: [1, nloc, *def]
        atom_energy = atomic_ret[self.kk][0]  # [nloc, *def]
        energy_redu = atom_energy.sum(dim=0).reshape(-1)[self.ci]
        return energy_redu


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
        @property  # type: ignore[override]
        def min_nbor_dist(self) -> float | None:
            """Minimum neighbor distance, stored as a buffer (survives serialization).

            Uses ``-1.0`` as sentinel for "not set", matching the pt backend.
            """
            buf = self.__dict__.get("_buffers", {}).get("_min_nbor_dist")
            if buf is None or buf.item() == -1.0:
                return None
            return buf.item()

        @min_nbor_dist.setter
        def min_nbor_dist(self, value: float | None) -> None:
            # Infer device from existing buffer or model parameters,
            # falling back to env.DEVICE only if nothing is available yet.
            buf = self.__dict__.get("_buffers", {}).get("_min_nbor_dist")
            if buf is not None:
                device = buf.device
            else:
                p = next(self.parameters(), None) or next(self.buffers(), None)
                if p is not None:
                    device = p.device
                else:
                    from deepmd.pt_expt.utils.env import (
                        DEVICE,
                    )

                    device = DEVICE

            t = torch.tensor(
                -1.0 if value is None else float(value),
                dtype=torch.float64,
                device=device,
            )
            if "_buffers" in self.__dict__ and "_min_nbor_dist" in self._buffers:
                self._buffers["_min_nbor_dist"] = t
            elif "_buffers" in self.__dict__:
                self.register_buffer("_min_nbor_dist", t)
            # else: too early (before Module.__init__), will be set again later

        def get_min_nbor_dist(self) -> float | None:
            """Get the minimum distance between two atoms."""
            return self.min_nbor_dist

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
            comm_dict: dict | None = None,
        ) -> dict[str, torch.Tensor]:
            atomic_ret = self.atomic_model.forward_common_atomic(
                extended_coord,
                extended_atype,
                nlist,
                mapping=mapping,
                fparam=fparam,
                aparam=aparam,
                comm_dict=comm_dict,
            )
            model_ret = fit_output_to_model_output(
                atomic_ret,
                self.atomic_output_def(),
                extended_coord,
                do_atomic_virial=do_atomic_virial,
                create_graph=self.training,
                mask=atomic_ret.get("mask"),
                extended_coord_corr=extended_coord_corr,
            )
            # Hessian computation (mirrors JAX's forward_common_atomic).
            # Produces hessian on extended coords [nf, *def, nall, 3, nall, 3],
            # then communicate_extended_output maps it to nloc x nloc.
            aod = self.atomic_output_def()
            for kk in aod.keys():
                vdef = aod[kk]
                if vdef.reducible and vdef.r_hessian:
                    kk_hess = get_hessian_name(kk)
                    model_ret[kk_hess] = _cal_hessian_ext(
                        self,
                        kk,
                        vdef,
                        extended_coord,
                        extended_atype,
                        nlist,
                        mapping,
                        fparam,
                        aparam,
                        create_graph=self.training,
                    )
            return model_ret

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
                nlist = _pad_nlist_for_export(nlist)
                return model.forward_common_lower(
                    extended_coord,
                    extended_atype,
                    nlist,
                    mapping,
                    fparam=fparam,
                    aparam=aparam,
                    do_atomic_virial=do_atomic_virial,
                )

            # Force `_format_nlist`'s sort branch into the compiled graph so the
            # exported model tolerates oversized nlists at runtime (LAMMPS builds
            # nlists with rcut+skin).  Combined with the short-circuit order in
            # `_format_nlist`, no symbolic guard on the dynamic `nnei` axis is
            # emitted.
            _orig_need_sort = model.need_sorted_nlist_for_lower
            model.need_sorted_nlist_for_lower = types.MethodType(
                lambda self: True, model
            )
            try:
                traced = make_fx(fn, **make_fx_kwargs)(
                    extended_coord,
                    extended_atype,
                    nlist,
                    mapping,
                    fparam,
                    aparam,
                )
            finally:
                model.need_sorted_nlist_for_lower = _orig_need_sort
            return traced

        def forward_common_lower_exportable_with_comm(
            self,
            extended_coord: torch.Tensor,
            extended_atype: torch.Tensor,
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
            """Trace forward_common_lower with comm_dict tensors as positional inputs.

            Used to compile a parallel-inference variant of the model
            (.pt2 with-comm artifact) that drives MPI ghost-atom exchange
            for GNN descriptors via the opaque
            ``deepmd_export::border_op`` wrapper. The comm tensors enter
            the exported program as 8 additional positional inputs after
            the usual (coord, atype, nlist, mapping, fparam, aparam) —
            this fixes the C++ ABI for ``DeepPotPTExpt`` (Phase 4).

            Tracing requires ``nswap >= 1`` (Phase 0 finding); with
            ``nswap == 0`` the dim specializes and the artifact would
            only run for that exact value. The C++ caller must always
            provide ``nswap >= 1``.
            """
            model = self

            def fn(
                extended_coord: torch.Tensor,
                extended_atype: torch.Tensor,
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
                # Same nnei-dynamic-axis workaround as the regular variant
                # (see ``_pad_nlist_for_export``).  Without it the with-comm
                # trace specialises ``nnei`` to the sample width.
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
                }
                return model.forward_common_lower(
                    extended_coord,
                    extended_atype,
                    nlist,
                    mapping,
                    fparam=fparam,
                    aparam=aparam,
                    do_atomic_virial=do_atomic_virial,
                    comm_dict=comm_dict,
                )

            # Force the sort branch in ``_format_nlist`` (mirrors the regular
            # variant) so the compiled graph's ``nnei`` axis stays dynamic.
            _orig_need_sort = model.need_sorted_nlist_for_lower
            model.need_sorted_nlist_for_lower = types.MethodType(
                lambda self: True, model
            )
            try:
                traced = make_fx(fn, **make_fx_kwargs)(
                    extended_coord,
                    extended_atype,
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
                model.need_sorted_nlist_for_lower = _orig_need_sort
            return traced

    return CM
