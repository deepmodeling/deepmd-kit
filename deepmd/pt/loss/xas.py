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

    Weighted loss
    -------------
    ``pref_energy`` and ``pref_spectrum`` allow independent weighting of the
    two energy dimensions (E_min, E_max at indices 0–1) and the intensity
    dimensions (indices 2:).  If the loss is dominated by the energy shift
    terms, reduce ``pref_energy`` or increase ``pref_spectrum``.

    Smoothness regularisation
    -------------------------
    XAS spectra should be smooth.  A second-order finite-difference penalty
    on the predicted intensity dimensions discourages high-frequency wiggles::

        L_smooth = smooth_reg * mean(
            (pred[:, i + 1] - 2 * pred[:, i] + pred[:, i - 1]) ^ 2
        )

    The regulariser is applied to the raw NN output (before adding ``e_ref``),
    so it acts on chemical-shift–normalised intensities.

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
    pref_energy : float
        Weight multiplier for the two energy dimensions (E_min, E_max).
        Default 1.0.  Reduce this if energy terms dominate training.
    pref_spectrum : float
        Weight multiplier for the intensity dimensions (index 2 onward).
        Default 1.0.  Increase this to focus training on spectral shape.
    smooth_reg : float
        Coefficient of the second-order smoothness regulariser applied to
        the predicted intensity dimensions.  0.0 disables it (default).
    intensity_norm : str
        Normalisation applied to intensity dimensions **inside the loss only**
        (model outputs remain in absolute units; freeze/test are unaffected).

        ``"none"`` (default)
            No normalisation.  Loss is dominated by high-intensity regions,
            which can cause small features to be ignored and large peaks to be
            over-smoothed by smooth_reg.

        ``"log1p"``
            Apply ``log(1 + x)`` to both predicted and label intensities before
            computing the loss and smooth_reg.  Compresses the dynamic range so
            that features at 1e-2 and features at 10 receive comparable gradient
            signal.  Requires non-negative intensities (physically guaranteed for
            XAS cross-sections; negative transients during training are clamped
            to 0 before the transform).

        ``"max_frame"``
            Divide each frame's intensities by ``max(|label|) + eps`` so that
            every spectrum has a peak of ~1 in loss space.  Equalises the
            per-frame contribution regardless of absolute cross-section scale.
            Use this when absolute intensity differences between frames are not
            physically meaningful.
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
        intensity_norm: str = "none",
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
        if intensity_norm not in ("none", "log1p", "max_frame"):
            raise ValueError(
                f"intensity_norm must be 'none', 'log1p', or 'max_frame', got {intensity_norm!r}"
            )
        self.intensity_norm = intensity_norm

        # e_ref[sel_type_idx, edge_idx, 0] = mean E_min  (eV)
        # e_ref[sel_type_idx, edge_idx, 1] = mean E_max  (eV)
        # Shape: [ntypes, nfparam, 2]. Filled by compute_output_stats; zero until then.
        self.register_buffer(
            "e_ref",
            torch.zeros(ntypes, nfparam, 2, dtype=env.GLOBAL_PT_FLOAT_PRECISION),
        )
        # e_std[sel_type_idx, edge_idx, 0/1] = std of chemical shifts for E_min/E_max.
        # Used to normalise the energy-dim loss so that chemical shifts (eV) are
        # on the same scale as the intensity dims after intensity_norm.
        # Initialised to 1.0 (= no normalisation) until compute_output_stats runs.
        self.register_buffer(
            "e_std",
            torch.ones(ntypes, nfparam, 2, dtype=env.GLOBAL_PT_FLOAT_PRECISION),
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
        e_std = torch.ones(
            self.ntypes, self.nfparam, 2, dtype=env.GLOBAL_PT_FLOAT_PRECISION
        )
        for (t, e), vals in accum.items():
            arr = np.array(vals)  # [n, 2]
            mean_val = np.mean(arr, axis=0)
            std_val = np.std(arr, axis=0).clip(min=1.0)  # floor at 1 eV
            e_ref[t, e] = torch.tensor(mean_val, dtype=env.GLOBAL_PT_FLOAT_PRECISION)
            e_std[t, e] = torch.tensor(std_val, dtype=env.GLOBAL_PT_FLOAT_PRECISION)
            log.info(
                f"XASLoss e_ref: type={t}, edge={e} -> "
                f"E_min_ref={float(e_ref[t, e, 0]):.2f} eV (std={float(e_std[t, e, 0]):.2f}), "
                f"E_max_ref={float(e_ref[t, e, 1]):.2f} eV (std={float(e_std[t, e, 1]):.2f})  "
                f"(n={len(vals)})"
            )

        self.e_ref.copy_(e_ref)
        self.e_std.copy_(e_std)
        log.info(
            f"XASLoss: e_ref/e_std computed for {len(accum)} (sel_type, edge) combinations."
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

                # 2. Set energy-dim out_bias/out_std so the NN predicts
                #    *normalised* chemical shifts instead of absolute energies.
                #
                #    out_bias[:, :2] = 0       → NN output = 0 means ΔE = 0 (= e_ref)
                #    out_std[:, :2]  = e_std   → NN output ±1 maps to ±e_std eV
                #
                #    This keeps the NN's raw output in ±O(1) range (similar to the
                #    intensity dims after max_frame normalisation), so pref_energy=1
                #    gives a balanced gradient signal without requiring manual tuning.
                #    The loss is computed directly on the physical chemical shift
                #    (pred = NN_raw × e_std ≈ ΔE), so no explicit e_std division is
                #    needed in the loss — avoiding the e_std² gradient suppression that
                #    would otherwise cause systematic energy prediction errors.
                #
                #    e_std is the global mean across all (type,edge) combinations,
                #    since out_std is shared across types in the current implementation.
                key_idx = am.bias_keys.index(self.var_name)
                # Compute a single representative e_std for the two energy dims
                # as the mean over all populated (type, edge) entries.
                populated = e_std.abs().gt(1.0)  # mask: where e_std > floor
                if populated.any():
                    e_std_global = e_std[populated].mean(dim=0)  # [2]
                else:
                    e_std_global = torch.ones(2, dtype=e_std.dtype, device=e_std.device)
                with torch.no_grad():
                    am.out_bias[key_idx, :, :2] = 0.0
                    am.out_std[key_idx, :, :2] = e_std_global.to(am.out_std.dtype)
                log.info(
                    f"XASLoss: set out_bias[:,:2]=0, out_std[:,:2]={e_std_global.tolist()} eV "
                    "(NN output ±1 ≈ ±e_std eV chemical shift)."
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

        # Sum atomic contributions over atoms of sel_type per frame.
        # The label represents the total XAS spectrum from all sel_type atoms
        # in the supercell, so the correct reduction is sum (not mean).
        nf, nloc, td = atom_prop.shape
        mask_3d = (atype.unsqueeze(-1) == sel_type.view(nf, 1, 1))  # [nf, nloc, 1]
        pred = (atom_prop * mask_3d).sum(dim=1)  # [nf, td]

        label_xas = label[self.var_name]  # [nf, task_dim]

        # --- per-frame reference energy lookup ---
        # edge_idx = argmax of one-hot fparam
        fparam = input_dict.get("fparam")
        if fparam is not None and fparam.numel() > 0:
            edge_idx = fparam.reshape(nf, -1).argmax(dim=-1).clamp(0, self.nfparam - 1)
        else:
            edge_idx = torch.zeros(nf, dtype=torch.long, device=pred.device)

        # e_ref_frame / e_std_frame: [nf, 2]
        _dev = self.e_ref.device
        _sel = sel_type.to(_dev)
        _eidx = edge_idx.to(_dev)
        e_ref_frame = self.e_ref[_sel, _eidx].to(pred.device)
        e_std_frame = self.e_std[_sel, _eidx].to(pred.device)  # [nf, 2]

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

        # --- intensity normalisation (loss space only; model output unchanged) ---
        # Separate intensity dims for possible normalisation before loss/smooth_reg.
        pred_intens = pred[:, 2:]  # [nf, n_pts]
        label_intens = label_shifted[:, 2:]
        if self.intensity_norm == "log1p":
            # log(1+x): compresses dynamic range so 1e-2 and 10 get comparable
            # gradient signal.  Clamp predictions to >=0 (physically required for
            # cross-sections; may transiently go negative early in training).
            pred_intens = torch.log1p(pred_intens.clamp(min=0.0))
            label_intens = torch.log1p(label_intens.clamp(min=0.0))
        elif self.intensity_norm == "max_frame":
            # Per-frame normalisation: divide by frame's peak label intensity.
            # Every spectrum contributes equally regardless of absolute scale.
            norm_factor = (
                label_intens.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-6)
            )
            pred_intens = pred_intens / norm_factor
            label_intens = label_intens / norm_factor

        # --- weighted loss ---
        # Build per-dimension weight vector: pref_energy for dims 0-1,
        # pref_spectrum for dims 2+.
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
        loss += self.pref_energy * _elem_loss(pred[:, :2], label_shifted[:, :2])
        loss += self.pref_spectrum * _elem_loss(pred_intens, label_intens)

        # --- smoothness regulariser on intensity dims ---
        # Computed on the (possibly normalised) intensity tensor so that the
        # curvature penalty is scale-invariant when intensity_norm is active:
        #   - "none"      : penalises absolute curvature (dominated by large peaks)
        #   - "log1p"     : penalises relative curvature uniformly across scales
        #   - "max_frame" : penalises curvature relative to each frame's peak
        if self.smooth_reg > 0.0 and self.task_dim > 4:
            curv = pred_intens[:, 2:] - 2.0 * pred_intens[:, 1:-1] + pred_intens[:, :-2]
            loss += self.smooth_reg * (curv**2).mean()

        # --- metrics ---
        more_loss: dict[str, torch.Tensor] = {}
        if "mae" in self.metric:
            more_loss["mae"] = F.l1_loss(pred, label_shifted, reduction="mean").detach()
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
