# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    Any,
)

import torch
import torch.nn.functional as F

from deepmd.pt.loss.loss import (
    TaskLoss,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.utils.data import (
    DataRequirementItem,
)

log = logging.getLogger(__name__)


class XASLoss(TaskLoss):
    """Loss for XAS spectrum fitting via property fitting + sel_type reduction.

    The model outputs per-atom property vectors (atom_xas).  For each frame
    this loss sums the contributions of atoms matching ``sel_type`` (read from
    ``sel_type.npy`` per system) and computes a loss against the per-frame XAS
    label.

    Normalisation statistics (``xas_e_ref``, ``xas_intensity_ref/std``,
    ``out_bias``, ``out_std``) are computed once before training by
    :meth:`DPXASAtomicModel.compute_or_load_out_stat` via the standard
    :meth:`compute_or_load_stat` pipeline and stored as model buffers.

    Parameters
    ----------
    task_dim : int
        Output dimension of the fitting net (e.g. 102 = E_min + E_max + 100 pts).
    nfparam : int
        Length of the fparam one-hot vector (= number of edge types).
    var_name : str
        Property name, must match ``property_name`` in the fitting config.
    loss_func : str
        One of ``smooth_mae``, ``mae``, ``mse``, ``rmse``.
    metric : list[str]
        Metrics to display during training (absolute scale).
    beta : float
        Beta parameter for smooth_l1 loss.
    pref_energy : float
        Weight multiplier for the two energy dimensions (E_min, E_max).
    pref_spectrum : float
        Weight multiplier for the intensity dimensions (index 2 onward).
    smooth_reg : float
        Coefficient of the second-order smoothness regulariser applied to the
        predicted intensity dimensions in standardised space.  0.0 disables (default).
    """

    def __init__(
        self,
        task_dim: int,
        nfparam: int,
        var_name: str = "xas",
        loss_func: str = "smooth_mae",
        metric: list[str] = ["mae"],
        beta: float = 1.0,
        pref_energy: float = 1.0,
        pref_spectrum: float = 1.0,
        smooth_reg: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.task_dim = task_dim
        self.nfparam = nfparam
        self.var_name = var_name
        self.loss_func = loss_func
        self.metric = metric
        self.beta = beta
        self.pref_energy = pref_energy
        self.pref_spectrum = pref_spectrum
        self.smooth_reg = smooth_reg

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        model: torch.nn.Module,
        label: dict[str, torch.Tensor],
        natoms: int,
        learning_rate: float = 0.0,
        mae: bool = False,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        model_pred = model(**input_dict)

        # per-atom outputs: [nf, nloc, task_dim]
        atom_prop = model_pred[f"atom_{self.var_name}"]
        atype = input_dict["atype"]  # [nf, nloc]

        sel_type = label["sel_type"][:, 0].long()  # [nf]

        nf, nloc, td = atom_prop.shape
        mask_3d = atype.unsqueeze(-1) == sel_type.view(nf, 1, 1)  # [nf, nloc, 1]
        pred = (atom_prop * mask_3d).sum(dim=1)  # [nf, task_dim]

        label_xas = label[self.var_name]  # [nf, task_dim]

        # --- per-(type, edge) stat lookup from model buffers ---
        fparam = input_dict.get("fparam")
        if fparam is not None and fparam.numel() > 0:
            edge_idx = fparam.reshape(nf, -1).argmax(dim=-1).clamp(0, self.nfparam - 1)
        else:
            edge_idx = torch.zeros(nf, dtype=torch.long, device=pred.device)

        am = model.atomic_model
        e_ref = am.xas_e_ref  # [ntypes, nfparam, 2]
        intensity_ref = am.xas_intensity_ref  # [ntypes, nfparam, n_pts]
        intensity_std = am.xas_intensity_std  # [ntypes, nfparam, n_pts]

        _dev = e_ref.device
        _sel = sel_type.to(_dev)
        _eidx = edge_idx.to(_dev)

        e_ref_frame = e_ref[_sel, _eidx].to(pred.device)  # [nf, 2]
        intensity_ref_frame = intensity_ref[_sel, _eidx].to(pred.device)  # [nf, n_pts]
        intensity_std_frame = intensity_std[_sel, _eidx].to(pred.device)  # [nf, n_pts]

        # Normalised targets:
        #   energy dims   → chemical shift:  label - e_ref
        #   intensity dims → standardised:   (label - ref) / std
        label_energy_norm = label_xas[:, :2] - e_ref_frame
        label_intens_norm = (label_xas[:, 2:] - intensity_ref_frame) / intensity_std_frame

        def _elem_loss(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            if self.loss_func == "smooth_mae":
                return F.smooth_l1_loss(p, t, reduction="sum", beta=self.beta)
            elif self.loss_func == "mae":
                return F.l1_loss(p, t, reduction="sum")
            elif self.loss_func == "mse":
                return F.mse_loss(p, t, reduction="sum")
            elif self.loss_func == "rmse":
                return torch.sqrt(F.mse_loss(p, t, reduction="mean"))
            else:
                raise RuntimeError(f"Unknown loss function: {self.loss_func}")

        loss = torch.zeros(1, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)[0]
        loss += self.pref_energy * _elem_loss(pred[:, :2], label_energy_norm)
        loss += self.pref_spectrum * _elem_loss(pred[:, 2:], label_intens_norm)

        # Smoothness regulariser on standardised intensity dims (scale-invariant).
        n_pts = self.task_dim - 2
        if self.smooth_reg > 0.0 and n_pts >= 3:
            pi = pred[:, 2:]  # [nf, n_pts] in standardised space
            curv = pi[:, 2:] - 2.0 * pi[:, 1:-1] + pi[:, :-2]
            loss += self.smooth_reg * (curv**2).mean()

        # --- metrics (reported on absolute scale) ---
        pred_abs = pred.clone()
        pred_abs[:, :2] = pred[:, :2] + e_ref_frame
        pred_abs[:, 2:] = pred[:, 2:] * intensity_std_frame + intensity_ref_frame

        more_loss: dict[str, torch.Tensor] = {}
        if "mae" in self.metric:
            more_loss["mae"] = F.l1_loss(
                pred_abs, label_xas, reduction="mean"
            ).detach()
        if "rmse" in self.metric:
            more_loss["rmse"] = torch.sqrt(
                F.mse_loss(pred_abs, label_xas, reduction="mean")
            ).detach()

        model_pred[self.var_name] = pred_abs
        return model_pred, loss, more_loss

    @property
    def label_requirement(self) -> list[DataRequirementItem]:
        """Declare required data files: xas label + sel_type."""
        return [
            DataRequirementItem(
                self.var_name,
                ndof=self.task_dim,
                atomic=False,
                must=True,
                high_prec=True,
            ),
            DataRequirementItem(
                "sel_type",
                ndof=1,
                atomic=False,
                must=True,
                high_prec=False,
            ),
        ]
