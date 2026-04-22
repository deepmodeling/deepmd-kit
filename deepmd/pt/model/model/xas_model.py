# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import torch

from deepmd.pt.model.atomic_model import (
    DPXASAtomicModel,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)

from .property_model import (
    PropertyModel,
)


@BaseModel.register("xas")
class XASModel(PropertyModel):
    """Model for XAS spectrum fitting.

    Identical to :class:`PropertyModel` but uses :class:`DPXASAtomicModel`
    as the underlying atomic model, which carries the per-(absorbing_type,
    edge) energy reference buffer ``xas_e_ref`` in the checkpoint.  This
    buffer is populated by :meth:`deepmd.pt.loss.xas.XASLoss.compute_output_stats`
    before training starts and restored at inference time so that absolute
    edge energies are available without any external reference files.

    Two corrections are applied in ``forward`` that are absent in the generic
    :class:`PropertyModel`:

    1. **sel_type reduction** — only atoms of the absorbing type (looked up
       from ``fparam`` via ``xas_edge_to_seltype``) contribute to the
       reduced spectrum.  The parent reduction sums *all* atoms, which is
       incorrect for XAS where only one atom type absorbs.

    2. **e_ref restoration** — during training the energy dimensions (E_min,
       E_max at indices 0–1) are trained against chemical shifts
       ``label − e_ref``.  At inference we add ``e_ref`` back so the output
       is in absolute edge-energy units (eV).
    """

    model_type = "xas"

    def __init__(
        self,
        descriptor: Any,
        fitting: Any,
        type_map: Any,
        **kwargs: Any,
    ) -> None:
        xas_atomic = DPXASAtomicModel(descriptor, fitting, type_map, **kwargs)
        super().__init__(
            descriptor, fitting, type_map, atomic_model_=xas_atomic, **kwargs
        )

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, torch.Tensor]:
        model_predict = super().forward(
            coord, atype, box, fparam, aparam, do_atomic_virial
        )
        if fparam is None or fparam.numel() == 0:
            return model_predict

        am = self.atomic_model
        edge_to_sel = getattr(am, "xas_edge_to_seltype", None)
        if edge_to_sel is None:
            return model_predict

        var_name = self.get_var_name()
        atom_xas = model_predict[f"atom_{var_name}"]  # [nf, nloc, task_dim]
        nf = atype.shape[0]

        # Derive absorbing atom type from one-hot fparam
        edge_idx = fparam.reshape(nf, -1).argmax(dim=-1).clamp(
            0, edge_to_sel.shape[0] - 1
        )  # [nf]
        sel_type_per_frame = edge_to_sel[edge_idx.to(edge_to_sel.device)].to(
            atype.device
        )  # [nf]

        # Sum only sel_type atoms per frame
        mask_3d = atype.unsqueeze(-1) == sel_type_per_frame.view(nf, 1, 1)  # [nf, nloc, 1]
        xas_redu = (atom_xas * mask_3d.to(atom_xas.dtype)).sum(dim=1)  # [nf, task_dim]

        xas_redu = xas_redu.clone()

        # Restore energy dims to absolute eV: pred_abs = pred + e_ref
        xas_e_ref = getattr(am, "xas_e_ref", None)
        if xas_e_ref is not None:
            e_ref_frame = xas_e_ref[
                sel_type_per_frame.to(xas_e_ref.device),
                edge_idx.to(xas_e_ref.device),
            ].to(dtype=xas_redu.dtype, device=xas_redu.device)  # [nf, 2]
            xas_redu[:, :2] = xas_redu[:, :2] + e_ref_frame

        # Restore intensity dims to absolute scale:
        #   pred_abs = pred_standardised * intensity_std + intensity_ref
        xas_intensity_ref = getattr(am, "xas_intensity_ref", None)
        if xas_intensity_ref is not None:
            xas_intensity_std = am.xas_intensity_std
            _st = sel_type_per_frame.to(xas_intensity_ref.device)
            _ei = edge_idx.to(xas_intensity_ref.device)
            int_ref = xas_intensity_ref[_st, _ei].to(
                dtype=xas_redu.dtype, device=xas_redu.device
            )  # [nf, n_pts]
            int_std = xas_intensity_std[_st, _ei].to(
                dtype=xas_redu.dtype, device=xas_redu.device
            )  # [nf, n_pts]
            xas_redu[:, 2:] = xas_redu[:, 2:] * int_std + int_ref

        model_predict[var_name] = xas_redu
        return model_predict
