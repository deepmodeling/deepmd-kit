# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from collections import (
    defaultdict,
)
from typing import (
    Any,
)

import numpy as np
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

    Normalisation strategy
    ----------------------
    XAS labels have two fundamentally different components that require separate
    treatment:

    **Energy dimensions (indices 0–1: E_min, E_max)**
        Absolute edge energies vary enormously across element-edge pairs
        (H K-edge ~14 eV, Th K-edge ~110 000 eV).  ``compute_output_stats``
        computes a per-(absorbing_type, edge) reference energy ``e_ref[t, e]``
        (the mean E_min/E_max across training frames for that group).  During
        training the label is shifted to a chemical-shift representation
        ``label[:, :2] - e_ref``, which is on a ±few-eV scale.  At inference
        ``e_ref`` is added back so the model output is in absolute eV.

    **Intensity dimensions (indices 2+)**
        Cross-section intensities for different (absorbing_type, edge)
        combinations can differ by orders of magnitude.  Normalising by a
        *per-frame* quantity (e.g. peak intensity) equalises the loss value
        across frames but leaves a ``1/norm_factor`` factor in the gradient
        chain, causing high-intensity frames to receive proportionally weaker
        training signal.

        Instead, ``compute_output_stats`` computes per-point statistics
        ``intensity_ref[t, e, :]`` (mean) and ``intensity_std[t, e, :]``
        (std) for every (absorbing_type, edge) group.  The loss is computed on
        the standardised residual::

            (pred[:, 2:] - intensity_ref[t, e]) / intensity_std[t, e]

        Since ``intensity_std`` is a fixed constant (not per-frame), all frames
        within the same (t, e) group receive identical gradient magnitudes —
        exactly the same mechanism used by ``PropertyLoss`` with its global
        ``out_std``.  The model's ``out_bias[:, 2:]`` and ``out_std[:, 2:]``
        are reset to 0/1 so the NN directly predicts the standardised
        spectrum.  At inference ``XASModel.forward`` restores absolute scale
        via ``pred * intensity_std + intensity_ref``.

    Parameters
    ----------
    task_dim : int
        Output dimension of the fitting net (e.g. 102 = E_min + E_max + 100 pts).
    ntypes : int
        Number of atom types in the model.
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
        predicted intensity dimensions in standardised space.  Penalises
        ``(pred[i+1] - 2*pred[i] + pred[i-1])^2``.  0.0 disables (default).
    """

    def __init__(
        self,
        task_dim: int,
        ntypes: int,
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
        self.ntypes = ntypes
        self.nfparam = nfparam
        self.var_name = var_name
        self.loss_func = loss_func
        self.metric = metric
        self.beta = beta
        self.pref_energy = pref_energy
        self.pref_spectrum = pref_spectrum
        self.smooth_reg = smooth_reg

        n_pts = max(task_dim - 2, 0)

        # e_ref[t, e, 0/1] = mean E_min / E_max for (absorbing_type t, edge e).
        # Shape [ntypes, nfparam, 2]. Filled by compute_output_stats; zero until then.
        self.register_buffer(
            "e_ref",
            torch.zeros(ntypes, nfparam, 2, dtype=env.GLOBAL_PT_FLOAT_PRECISION),
        )
        # e_std[t, e, 0/1] = std of chemical shifts; used to scale out_std for energy dims.
        self.register_buffer(
            "e_std",
            torch.ones(ntypes, nfparam, 2, dtype=env.GLOBAL_PT_FLOAT_PRECISION),
        )
        # intensity_ref[t, e, i] = per-point mean intensity for (t, e) group.
        # Shape [ntypes, nfparam, n_pts]. Filled by compute_output_stats; zero until then.
        self.register_buffer(
            "intensity_ref",
            torch.zeros(ntypes, nfparam, n_pts, dtype=env.GLOBAL_PT_FLOAT_PRECISION),
        )
        # intensity_std[t, e, i] = per-point std of intensities; floored at 1e-6.
        # All frames in a (t, e) group are divided by the same constant → equal
        # gradient magnitudes across frames (cf. PropertyLoss / out_std approach).
        self.register_buffer(
            "intensity_std",
            torch.ones(ntypes, nfparam, n_pts, dtype=env.GLOBAL_PT_FLOAT_PRECISION),
        )

    # ------------------------------------------------------------------
    # Stat phase
    # ------------------------------------------------------------------
    def compute_output_stats(
        self,
        sampled: list[dict],
        model: "torch.nn.Module | None" = None,
    ) -> None:
        """Compute per-(absorbing_type, edge) statistics and update model buffers.

        Called once before training starts.  Requires ``xas``, ``sel_type``,
        and ``fparam`` in at least some samples.

        Energy dims
            ``e_ref`` = mean absolute edge energy per group → label shifted to
            chemical shifts.  ``out_bias[:, :2] = 0``,
            ``out_std[:, :2] = e_std_global`` so the NN works in ±O(1) space.

        Intensity dims
            ``intensity_ref`` = per-point mean intensity per group.
            ``intensity_std`` = per-point std (floored at 1e-6).
            ``out_bias[:, 2:] = 0``, ``out_std[:, 2:] = 1`` so the NN
            directly outputs the standardised residual
            ``(label - intensity_ref) / intensity_std``.
        """
        accum: dict[tuple[int, int], list] = defaultdict(list)

        for frame in sampled:
            if (
                self.var_name not in frame
                or "sel_type" not in frame
                or "fparam" not in frame
            ):
                continue
            xas = frame[self.var_name]
            sel_type = frame["sel_type"]
            fparam = frame["fparam"]

            xas = xas.reshape(-1, self.task_dim)
            sel_type = sel_type.reshape(-1).long()
            fparam = fparam.reshape(-1, self.nfparam)
            edge_idx = fparam.argmax(dim=-1)

            nf = xas.shape[0]
            for i in range(nf):
                t = int(sel_type[i].item())
                e = int(edge_idx[i].item())
                if 0 <= t < self.ntypes and 0 <= e < self.nfparam:
                    accum[(t, e)].append(xas[i].detach().cpu().numpy())

        if not accum:
            log.warning(
                "XASLoss.compute_output_stats: no frames with xas+sel_type+fparam found; "
                "stats remain at defaults. Training may be unstable."
            )
            return

        e_ref = torch.zeros(
            self.ntypes, self.nfparam, 2, dtype=env.GLOBAL_PT_FLOAT_PRECISION
        )
        e_std = torch.ones(
            self.ntypes, self.nfparam, 2, dtype=env.GLOBAL_PT_FLOAT_PRECISION
        )
        n_pts = max(self.task_dim - 2, 0)
        intensity_ref = torch.zeros(
            self.ntypes, self.nfparam, n_pts, dtype=env.GLOBAL_PT_FLOAT_PRECISION
        )
        intensity_std = torch.ones(
            self.ntypes, self.nfparam, n_pts, dtype=env.GLOBAL_PT_FLOAT_PRECISION
        )

        for (t, e), vals in accum.items():
            arr = np.array(vals)  # [n, task_dim]

            # Energy dims
            energy_mean = np.mean(arr[:, :2], axis=0)
            energy_std = np.std(arr[:, :2], axis=0).clip(min=1.0)
            e_ref[t, e] = torch.tensor(energy_mean, dtype=env.GLOBAL_PT_FLOAT_PRECISION)
            e_std[t, e] = torch.tensor(energy_std, dtype=env.GLOBAL_PT_FLOAT_PRECISION)

            # Intensity dims
            if n_pts > 0:
                intens_mean = np.mean(arr[:, 2:], axis=0)  # [n_pts]
                intens_std = np.std(arr[:, 2:], axis=0).clip(min=1e-6)
                intensity_ref[t, e] = torch.tensor(
                    intens_mean, dtype=env.GLOBAL_PT_FLOAT_PRECISION
                )
                intensity_std[t, e] = torch.tensor(
                    intens_std, dtype=env.GLOBAL_PT_FLOAT_PRECISION
                )

            log.info(
                f"XASLoss stats: type={t}, edge={e} | "
                f"E_min_ref={float(e_ref[t,e,0]):.2f} eV (std={float(e_std[t,e,0]):.2f}), "
                f"E_max_ref={float(e_ref[t,e,1]):.2f} eV (std={float(e_std[t,e,1]):.2f}) | "
                f"intensity_ref mean={float(intensity_ref[t,e].mean()):.4g}, "
                f"intensity_std mean={float(intensity_std[t,e].mean()):.4g}  "
                f"(n={len(vals)})"
            )

        self.e_ref.copy_(e_ref)
        self.e_std.copy_(e_std)
        self.intensity_ref.copy_(intensity_ref)
        self.intensity_std.copy_(intensity_std)
        log.info(
            f"XASLoss: stats computed for {len(accum)} (sel_type, edge) combinations."
        )

        if model is not None:
            try:
                am = model.atomic_model

                # Copy e_ref / intensity stats into model buffers so they are
                # saved in the checkpoint and available at inference time.
                if getattr(am, "xas_e_ref", None) is not None:
                    am.xas_e_ref.copy_(e_ref.to(am.xas_e_ref.dtype))
                    log.info("XASLoss: copied e_ref → model.atomic_model.xas_e_ref.")

                if getattr(am, "xas_intensity_ref", None) is not None:
                    am.xas_intensity_ref.copy_(
                        intensity_ref.to(am.xas_intensity_ref.dtype)
                    )
                    am.xas_intensity_std.copy_(
                        intensity_std.to(am.xas_intensity_std.dtype)
                    )
                    log.info(
                        "XASLoss: copied intensity_ref/std → "
                        "model.atomic_model.xas_intensity_ref/std."
                    )

                # edge_idx → absorbing atom type mapping (needed by inference forward).
                if getattr(am, "xas_edge_to_seltype", None) is not None:
                    mapping = torch.zeros(
                        self.nfparam,
                        dtype=torch.long,
                        device=am.xas_edge_to_seltype.device,
                    )
                    for (t, e) in accum.keys():
                        mapping[e] = t
                    am.xas_edge_to_seltype.copy_(mapping)
                    log.info("XASLoss: populated xas_edge_to_seltype mapping.")

                key_idx = am.bias_keys.index(self.var_name)

                # Energy dims: out_bias = 0, out_std = e_std_global
                # → NN predicts chemical shifts in ±O(1) normalised units.
                populated = e_std.abs().gt(1.0)
                if populated.any():
                    e_std_global = e_std[populated].mean(dim=0)  # [2]
                else:
                    e_std_global = torch.ones(2, dtype=e_std.dtype, device=e_std.device)
                with torch.no_grad():
                    am.out_bias[key_idx, :, :2] = 0.0
                    am.out_std[key_idx, :, :2] = e_std_global.to(am.out_std.dtype)
                log.info(
                    f"XASLoss: out_bias[:,:2]=0, out_std[:,:2]={e_std_global.tolist()} eV."
                )

                # Intensity dims: out_bias = 0, out_std = 1
                # → NN directly outputs the standardised residual
                #   (label - intensity_ref) / intensity_std.
                # Since intensity_std is a fixed constant per (t,e) group, the
                # gradient magnitude is identical for every frame in the group —
                # the same mechanism PropertyLoss uses with its global out_std.
                if n_pts > 0:
                    with torch.no_grad():
                        am.out_bias[key_idx, :, 2:] = 0.0
                        am.out_std[key_idx, :, 2:] = 1.0
                    log.info(
                        "XASLoss: out_bias[:,2:]=0, out_std[:,2:]=1 "
                        "(NN predicts standardised intensity)."
                    )

            except Exception as exc:
                log.warning(f"XASLoss: could not update model stats: {exc}")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
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

        # --- per-(type, edge) stat lookup ---
        fparam = input_dict.get("fparam")
        if fparam is not None and fparam.numel() > 0:
            edge_idx = fparam.reshape(nf, -1).argmax(dim=-1).clamp(0, self.nfparam - 1)
        else:
            edge_idx = torch.zeros(nf, dtype=torch.long, device=pred.device)

        _dev = self.e_ref.device
        _sel = sel_type.to(_dev)
        _eidx = edge_idx.to(_dev)

        e_ref_frame = self.e_ref[_sel, _eidx].to(pred.device)          # [nf, 2]
        intensity_ref_frame = self.intensity_ref[_sel, _eidx].to(pred.device)  # [nf, n_pts]
        intensity_std_frame = self.intensity_std[_sel, _eidx].to(pred.device)  # [nf, n_pts]

        # Normalised targets:
        #   energy dims   → chemical shift:  label - e_ref          (eV scale, ≈ ±few eV)
        #   intensity dims → standardised:   (label - ref) / std    (unit-variance)
        label_energy_norm = label_xas[:, :2] - e_ref_frame
        label_intens_norm = (label_xas[:, 2:] - intensity_ref_frame) / intensity_std_frame

        # pred[:, :2]  = NN_raw * e_std_global  ≈ chemical shift  (from apply_out_stat)
        # pred[:, 2:]  = NN_raw * 1 + 0         ≈ standardised intensity
        # Both are already in the correct normalised space after compute_output_stats.

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
