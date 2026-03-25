# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from deepmd.pt.loss.loss import TaskLoss
from deepmd.pt.utils import env
from deepmd.utils.data import DataRequirementItem

log = logging.getLogger(__name__)


class XASLoss(TaskLoss):
    """Loss for XAS spectrum fitting via property fitting + sel_type reduction.

    The model outputs per-atom property vectors (atom_xas).  For each frame
    this loss selects the atoms of type ``sel_type`` (read from ``sel_type.npy``
    in each training system) and takes their mean, then computes a loss against
    the per-frame XAS label.

    Energy normalization
    --------------------
    XAS labels contain absolute edge energies (E_min, E_max in eV) that vary
    enormously across element-edge pairs (H_K ~14 eV, Th_K ~110000 eV).
    Training directly on absolute values causes gradient instability because
    the energy dimensions dwarf the intensity dimensions.

    ``compute_output_stats`` computes a reference energy ``e_ref[t, e]`` for
    every ``(absorbing_type t, edge_index e)`` combination from the training
    data and stores it as a registered buffer.  During training, ``forward``
    normalises labels and predictions by subtracting the per-frame reference
    so that the loss is computed on chemical shifts (±few eV) and normalised
    intensities—quantities of comparable magnitude.

    The buffer is saved in the model checkpoint, eliminating any need for
    external normalisation files.

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
        Metrics to display during training.
    beta : float
        Beta parameter for smooth_l1 loss.
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

        # e_ref[sel_type_idx, edge_idx, 0] = mean E_min  (eV)
        # e_ref[sel_type_idx, edge_idx, 1] = mean E_max  (eV)
        # Shape: [ntypes, nfparam, 2]. Filled by compute_output_stats; zero until then.
        self.register_buffer(
            "e_ref",
            torch.zeros(ntypes, nfparam, 2, dtype=env.GLOBAL_PT_FLOAT_PRECISION),
        )

    # ------------------------------------------------------------------
    # Stat phase: compute per-(absorbing_type, edge) reference energies
    # ------------------------------------------------------------------
    def compute_output_stats(
        self,
        sampled: list[dict],
        model: "torch.nn.Module | None" = None,
    ) -> None:
        """Compute ``e_ref`` and fix model energy-dim bias/std.

        Called once before training starts.  Requires ``xas``, ``sel_type``,
        and ``fparam`` in at least some samples.

        Parameters
        ----------
        sampled : list[dict]
            List of data batches from ``make_stat_input``.
        model : nn.Module, optional
            The full DeePMD model.  When given, the per-atom property model's
            ``out_bias`` and ``out_std`` for the two energy dimensions (E_min,
            E_max) are reset to 0 / 1 so the NN predicts *chemical shifts*
            (±few eV) instead of absolute energies (~thousands of eV).
            Without this reset the stat-initialised ``out_std ≈ 26 000 eV``
            amplifies weight-update steps by 26 000×, causing immediate
            gradient explosion.
        """
        accum: dict[tuple[int, int], list] = defaultdict(list)

        for frame in sampled:
            if (
                self.var_name not in frame
                or "sel_type" not in frame
                or "fparam" not in frame
            ):
                continue
            xas = frame[self.var_name]  # tensor, various shapes
            sel_type = frame["sel_type"]
            fparam = frame["fparam"]

            # flatten to [nf, task_dim], [nf], [nf, nfparam]
            xas = xas.reshape(-1, self.task_dim)
            sel_type = sel_type.reshape(-1).long()
            fparam = fparam.reshape(-1, self.nfparam)
            edge_idx = fparam.argmax(dim=-1)

            nf = xas.shape[0]
            for i in range(nf):
                t = int(sel_type[i].item())
                e = int(edge_idx[i].item())
                if 0 <= t < self.ntypes and 0 <= e < self.nfparam:
                    accum[(t, e)].append(xas[i, :2].detach().cpu().numpy())

        if not accum:
            log.warning(
                "XASLoss.compute_output_stats: no frames with xas+sel_type+fparam found; "
                "e_ref remains zero. Training may be unstable."
            )
            return

        e_ref = torch.zeros(
            self.ntypes, self.nfparam, 2, dtype=env.GLOBAL_PT_FLOAT_PRECISION
        )
        for (t, e), vals in accum.items():
            e_ref[t, e] = torch.tensor(
                np.mean(vals, axis=0), dtype=env.GLOBAL_PT_FLOAT_PRECISION
            )
            log.info(
                f"XASLoss e_ref: type={t}, edge={e} -> "
                f"E_min_ref={float(e_ref[t,e,0]):.2f} eV, "
                f"E_max_ref={float(e_ref[t,e,1]):.2f} eV  "
                f"(n={len(vals)})"
            )

        self.e_ref.copy_(e_ref)
        log.info(
            f"XASLoss: e_ref computed for {len(accum)} (sel_type, edge) combinations."
        )

        if model is not None:
            try:
                am = model.atomic_model

                # 1. Copy e_ref into the model's own buffer so it is saved
                #    in the checkpoint and available at inference time without
                #    any external reference file (analogous to out_bias).
                if getattr(am, "xas_e_ref", None) is not None:
                    am.xas_e_ref.copy_(e_ref.to(am.xas_e_ref.dtype))
                    log.info("XASLoss: copied e_ref → model.atomic_model.xas_e_ref.")

                # 2. Reset energy-dim out_bias/out_std so the NN predicts
                #    chemical shifts instead of absolute energies.
                #
                #    Why this is necessary
                #    ----------------------
                #    The model stat phase initialises
                #      out_bias[:, :2] ≈ global_mean(E_min, E_max) ≈ 19 000 eV
                #      out_std[:, :2]  ≈ global_std(E_min, E_max)  ≈ 26 000 eV
                #    so  atom_xas[:, 0] = NN_raw[:, 0] * 26 000 + 19 000.
                #    A single Adam step changes NN_raw by ~lr, which changes
                #    the physical output by lr × 26 000 = 2.7 eV — the same
                #    instability as out_bias for energy fitting if the reference
                #    is wrong.  With out_std=1 / out_bias=0, the NN output for
                #    energy dims is interpreted directly as a chemical shift
                #    (target ≈ label − e_ref ≈ ±few eV), keeping gradient
                #    magnitudes O(1) and training stable.
                key_idx = am.bias_keys.index(self.var_name)
                with torch.no_grad():
                    am.out_bias[key_idx, :, :2] = 0.0
                    am.out_std[key_idx, :, :2] = 1.0
                log.info(
                    "XASLoss: reset out_bias[:,:2]=0 and out_std[:,:2]=1 "
                    "for energy dims (model predicts chemical shifts; "
                    "xas_e_ref restores absolute energies at inference)."
                )
            except Exception as exc:
                log.warning(f"XASLoss: could not update model energy-dim stats: {exc}")

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

        # sel_type from label: [nf, 1] float → [nf] int
        sel_type = label["sel_type"][:, 0].long()

        # element-wise mean: average atom_prop over atoms of sel_type per frame
        nf, nloc, td = atom_prop.shape
        pred = torch.zeros(nf, td, dtype=atom_prop.dtype, device=atom_prop.device)
        for i in range(nf):
            t = int(sel_type[i].item())
            mask = (atype[i] == t).unsqueeze(-1)  # [nloc, 1]
            count = mask.sum().clamp(min=1)
            pred[i] = (atom_prop[i] * mask).sum(dim=0) / count

        label_xas = label[self.var_name]  # [nf, task_dim]

        # --- per-frame reference energy lookup ---
        # edge_idx = argmax of one-hot fparam
        fparam = input_dict.get("fparam")
        if fparam is not None and fparam.numel() > 0:
            edge_idx = fparam.reshape(nf, -1).argmax(dim=-1).clamp(0, self.nfparam - 1)
        else:
            edge_idx = torch.zeros(nf, dtype=torch.long, device=pred.device)

        # e_ref_frame: [nf, 2]  (E_min_ref, E_max_ref for each frame)
        e_ref_frame = self.e_ref[sel_type, edge_idx]  # [nf, 2]

        # Shift the energy-dim TARGETS only.
        #
        # After compute_output_stats has reset out_bias[:,:2]=0 / out_std[:,:2]=1,
        # the model outputs raw NN values ≈ 0 for dims 0,1.  We train those
        # dims against (label − e_ref), i.e. the chemical shift (±few eV),
        # keeping gradient magnitudes O(1).  Intensity dims (2:) are trained
        # against the original label values unchanged.
        #
        # At inference, we add e_ref back to get the absolute edge energy.
        label_shifted = label_xas.clone()
        label_shifted[:, :2] = label_xas[:, :2] - e_ref_frame

        # --- loss ---
        loss = torch.zeros(1, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)[0]
        if self.loss_func == "smooth_mae":
            loss += F.smooth_l1_loss(
                pred, label_shifted, reduction="sum", beta=self.beta
            )
        elif self.loss_func == "mae":
            loss += F.l1_loss(pred, label_shifted, reduction="sum")
        elif self.loss_func == "mse":
            loss += F.mse_loss(pred, label_shifted, reduction="sum")
        elif self.loss_func == "rmse":
            loss += torch.sqrt(F.mse_loss(pred, label_shifted, reduction="mean"))
        else:
            raise RuntimeError(f"Unknown loss function: {self.loss_func}")

        # --- metrics ---
        more_loss: dict[str, torch.Tensor] = {}
        if "mae" in self.metric:
            more_loss["mae"] = F.l1_loss(
                pred, label_shifted, reduction="mean"
            ).detach()
        if "rmse" in self.metric:
            more_loss["rmse"] = torch.sqrt(
                F.mse_loss(pred, label_shifted, reduction="mean")
            ).detach()

        # Absolute prediction: add e_ref back to energy dims for eval / output
        pred_abs = pred.clone()
        pred_abs[:, :2] = pred[:, :2] + e_ref_frame
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
