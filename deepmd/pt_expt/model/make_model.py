# SPDX-License-Identifier: LGPL-3.0-or-later
import math
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
        ) -> dict[str, torch.Tensor]:
            atomic_ret = self.atomic_model.forward_common_atomic(
                extended_coord,
                extended_atype,
                nlist,
                mapping=mapping,
                fparam=fparam,
                aparam=aparam,
            )
            model_ret = fit_output_to_model_output(
                atomic_ret,
                self.atomic_output_def(),
                extended_coord,
                do_atomic_virial=do_atomic_virial,
                create_graph=self.training,
                mask=atomic_ret.get("mask"),
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
