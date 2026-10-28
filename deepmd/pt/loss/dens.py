# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import torch

from deepmd.pt.loss.ener import (
    EnergyStdLoss,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    GLOBAL_PT_FLOAT_PRECISION,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


class DeNSLoss(EnergyStdLoss):
    """
    Joint energy and direct-force/denoising loss for SeZM `dens` mode.

    This loss follows the EquiformerV3 DeNS training semantics:

    - energy is supervised in one global normalized space
    - clean atoms predict globally normalized direct forces
    - corrupted atoms predict normalized Gaussian noise `epsilon / sigma`

    A batch enters the denoising path with probability `dens_prob`. Otherwise the
    batch falls back to clean direct-force supervision while still using the `dens`
    head. When only part of the batch is corrupted, each subset loss is weighted by
    its atom fraction so the mixed objective matches one full-batch per-atom average.
    """

    def __init__(
        self,
        starter_learning_rate: float = 1.0,
        start_pref_e: float = 20,
        limit_pref_e: float = 20,
        start_pref_f: float = 20,
        limit_pref_f: float = 20,
        loss_func: str = "mae",
        inference: bool = False,
        dens_prob: float = 0.5,
        dens_fixed_noise_std: bool = True,
        dens_std: float = 0.025,
        dens_corrupt_ratio: float | None = 0.5,
        dens_denoising_pos_coefficient: float = 10.0,
        start_pref_v: float = 0.0,
        limit_pref_v: float = 0.0,
        start_pref_ae: float = 0.0,
        limit_pref_ae: float = 0.0,
        start_pref_pf: float = 0.0,
        limit_pref_pf: float = 0.0,
        start_pref_gf: float = 0.0,
        limit_pref_gf: float = 0.0,
        numb_generalized_coord: int = 0,
        **kwargs: Any,
    ) -> None:
        unsupported = sorted(key for key in kwargs if key not in {"type"})
        if unsupported:
            unsupported_str = ", ".join(unsupported)
            raise ValueError(f"Unsupported `dens` loss options: {unsupported_str}.")
        if not dens_fixed_noise_std:
            raise NotImplementedError(
                "`dens_fixed_noise_std=false` is not supported. "
                "This matches the current EquiformerV3 DeNS trainer path, "
                "which only uses the fixed-noise-std setting."
            )
        if not 0.0 <= float(dens_prob) <= 1.0:
            raise ValueError("`dens_prob` must be within [0, 1].")
        if (
            dens_corrupt_ratio is not None
            and not 0.0 <= float(dens_corrupt_ratio) <= 1.0
        ):
            raise ValueError("`dens_corrupt_ratio` must be within [0, 1] or None.")
        if float(dens_std) <= 0.0:
            raise ValueError("`dens_std` must be > 0.")
        if float(dens_denoising_pos_coefficient) < 0.0:
            raise ValueError("`dens_denoising_pos_coefficient` must be >= 0.")
        unsupported_prefactors = (
            float(start_pref_v),
            float(limit_pref_v),
            float(start_pref_ae),
            float(limit_pref_ae),
            float(start_pref_pf),
            float(limit_pref_pf),
            float(start_pref_gf),
            float(limit_pref_gf),
            float(numb_generalized_coord),
        )
        if any(value != 0.0 for value in unsupported_prefactors):
            raise ValueError(
                "`dens` loss currently supports only energy and force/noise supervision."
            )
        super().__init__(
            starter_learning_rate=starter_learning_rate,
            start_pref_e=start_pref_e,
            limit_pref_e=limit_pref_e,
            start_pref_f=start_pref_f,
            limit_pref_f=limit_pref_f,
            start_pref_v=0.0,
            limit_pref_v=0.0,
            start_pref_ae=0.0,
            limit_pref_ae=0.0,
            start_pref_pf=0.0,
            limit_pref_pf=0.0,
            relative_f=None,
            enable_atom_ener_coeff=False,
            start_pref_gf=0.0,
            limit_pref_gf=0.0,
            numb_generalized_coord=0,
            loss_func=loss_func,
            inference=inference,
            use_huber=False,
            f_use_norm=(loss_func == "mae"),
            huber_delta=0.01,
        )
        self.dens_prob = float(dens_prob)
        self.dens_fixed_noise_std = bool(dens_fixed_noise_std)
        self.dens_std = float(dens_std)
        self.dens_corrupt_ratio = (
            None if dens_corrupt_ratio is None else float(dens_corrupt_ratio)
        )
        self.dens_denoising_pos_coefficient = float(dens_denoising_pos_coefficient)

    @staticmethod
    def _canonicalize_vec3_tensor(
        tensor: torch.Tensor,
        *,
        nf: int,
        nloc: int,
        name: str,
    ) -> torch.Tensor:
        """Convert `(nf, nloc*3)` or `(nf, nloc, 3)` to `(nf, nloc, 3)`."""
        if tensor.ndim == 3:
            if tensor.shape != (nf, nloc, 3):
                raise ValueError(
                    f"`{name}` must have shape ({nf}, {nloc}, 3), got {tuple(tensor.shape)}."
                )
            return tensor
        if tensor.ndim == 2:
            if tensor.shape != (nf, nloc * 3):
                raise ValueError(
                    f"`{name}` must have shape ({nf}, {nloc * 3}) when flattened, got {tuple(tensor.shape)}."
                )
            return tensor.view(nf, nloc, 3)
        raise ValueError(
            f"`{name}` must have shape ({nf}, {nloc}, 3) or ({nf}, {nloc * 3})."
        )

    def _prepare_dens_inputs(
        self,
        input_dict: dict[str, torch.Tensor],
        label: dict[str, torch.Tensor],
        *,
        enable_dens: bool,
    ) -> tuple[
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        bool,
    ]:
        """Build noisy coordinates and mixed targets for one forward pass.

        Returns the corrupted and clean atom subsets as two masks. Both are
        restricted to real atoms: a mixed-nloc batch is padded to a rectangular
        shape with phantom slots that carry no physical site, and a phantom is
        neither corrupted nor clean. The two masks are built independently
        rather than as complements of one another, since complements over the
        padded width would hand every phantom to the clean subset.
        """
        atype = input_dict["atype"]
        nf, nloc = atype.shape[:2]
        coord_raw = input_dict["coord"]
        coord = self._canonicalize_vec3_tensor(
            coord_raw, nf=nf, nloc=nloc, name="coord"
        )
        force_label = self._canonicalize_vec3_tensor(
            label["force"], nf=nf, nloc=nloc, name="force"
        ).to(device=coord.device, dtype=coord.dtype)

        use_dens = bool(
            enable_dens
            and self.dens_prob > 0.0
            and torch.rand(
                (), dtype=GLOBAL_PT_FLOAT_PRECISION, device=coord.device
            ).item()
            < self.dens_prob
        )
        real_mask = atype >= 0
        noise_mask = torch.zeros((nf, nloc), dtype=torch.bool, device=coord.device)
        noise_vec = torch.zeros_like(coord)
        if use_dens:
            if self.dens_corrupt_ratio is None:
                noise_mask = real_mask
            else:
                noise_mask = (
                    torch.rand(
                        (nf, nloc), dtype=GLOBAL_PT_FLOAT_PRECISION, device=coord.device
                    )
                    < self.dens_corrupt_ratio
                ) & real_mask
            noise_vec = torch.randn_like(coord) * self.dens_std
            noise_vec = noise_vec * noise_mask.unsqueeze(-1)
        clean_mask = real_mask & ~noise_mask
        coord_model = coord + noise_vec

        # DeNS predicts normalized noise epsilon / sigma for corrupted atoms.
        noise_target = noise_vec / self.dens_std

        model_input = dict(input_dict)
        if coord_raw.ndim == 2:
            model_input["coord"] = coord_model.view(nf, nloc * 3)
        else:
            model_input["coord"] = coord_model
        model_input["noise_mask"] = noise_mask
        if use_dens:
            model_input["force_input"] = force_label
        return (
            model_input,
            force_label,
            noise_target,
            noise_mask,
            clean_mask,
            use_dens,
        )

    @staticmethod
    def _get_sezm_atomic_model(model: torch.nn.Module) -> Any:
        """Return the SeZM atomic model used by `dens` training."""
        atomic_model = getattr(model, "atomic_model", None)
        if atomic_model is None:
            raise TypeError("SeZM `dens` loss expects `model.atomic_model` to exist.")
        required = (
            "norm_dens_energy",
            "denorm_dens_energy",
            "norm_dens_force",
            "denorm_dens_force",
        )
        missing = [name for name in required if not hasattr(atomic_model, name)]
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise TypeError(
                f"SeZM `dens` loss requires atomic_model methods: {missing_str}."
            )
        return atomic_model

    def _compute_force_subset_loss(
        self,
        force_pred: torch.Tensor,
        force_target: torch.Tensor,
        coefficient: float | torch.Tensor,
    ) -> torch.Tensor:
        """Compute one clean-force or denoising-force subset loss."""
        if force_pred.numel() == 0:
            return force_pred.new_zeros((), dtype=GLOBAL_PT_FLOAT_PRECISION)
        diff_f = (force_target - force_pred).reshape(-1)
        if self.loss_func == "mse":
            subset_loss = torch.mean(torch.square(diff_f))
        elif self.loss_func == "mae":
            subset_loss = torch.linalg.vector_norm(
                (force_target - force_pred).reshape(-1, 3),
                ord=2,
                dim=1,
                keepdim=True,
            ).mean()
        else:
            raise NotImplementedError(
                f"Loss type {self.loss_func} is not implemented for `dens` force loss."
            )
        return (coefficient * subset_loss).to(GLOBAL_PT_FLOAT_PRECISION)

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        model: torch.nn.Module,
        label: dict[str, torch.Tensor],
        natoms: int,
        learning_rate: float,
        mae: bool = False,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        """Return loss on SeZM `dens` energy and direct-force/noise outputs.

        ``natoms`` is the padded width of the batch and is superseded here by
        the per-frame real atom count read off ``atype``; it remains in the
        signature to satisfy the shared loss interface.
        """
        del natoms
        model_input, force_label, noise_target, noise_mask, clean_mask, use_dens = (
            self._prepare_dens_inputs(
                input_dict,
                label,
                enable_dens=model.training,
            )
        )
        model_pred = model(**model_input)
        atomic_model = self._get_sezm_atomic_model(model)

        coef = learning_rate / self.starter_learning_rate
        pref_e = self.limit_pref_e + (self.start_pref_e - self.limit_pref_e) * coef
        pref_f = self.limit_pref_f + (self.start_pref_f - self.limit_pref_f) * coef
        denoise_pref = self.dens_denoising_pos_coefficient

        loss = force_label.new_zeros((), dtype=env.GLOBAL_PT_FLOAT_PRECISION)
        more_loss: dict[str, torch.Tensor] = {}
        # The energy terms are extensive, so a frame's residual is divided by
        # that frame's own real atom count; a mixed-nloc batch pads to a common
        # width with phantom atoms that belong to no system. The factor is
        # therefore applied inside the mean over frames, where on a batch of
        # uniform atom count it reduces to the scalar ``1 / natoms``.
        inv_natoms = 1.0 / (input_dict["atype"] >= 0).sum(dim=-1, keepdim=True).to(
            dtype=force_label.dtype
        )  # [nf, 1]

        if self.has_e and "energy" in model_pred and "energy" in label:
            energy_pred = model_pred.get("energy_norm", model_pred["energy"])
            energy_label = label["energy"].to(
                device=energy_pred.device, dtype=energy_pred.dtype
            )
            energy_label_norm = atomic_model.norm_dens_energy(
                energy_label,
                input_dict["atype"],
            )
            if "energy_norm" in model_pred:
                energy_pred_phys = model_pred["energy"].to(
                    device=energy_pred.device,
                    dtype=energy_pred.dtype,
                )
            else:
                energy_pred_phys = atomic_model.denorm_dens_energy(
                    energy_pred,
                    input_dict["atype"],
                )
            find_energy = label.get("find_energy", 0.0)
            pref_e = pref_e * find_energy
            if self.loss_func == "mse":
                l2_ener_loss = torch.mean(torch.square(energy_pred - energy_label_norm))
                if not self.inference:
                    more_loss["l2_ener_loss"] = self.display_if_exist(
                        l2_ener_loss.detach(),
                        find_energy,
                    )
                loss += pref_e * torch.mean(
                    torch.square(energy_pred - energy_label_norm) * inv_natoms
                )
                rmse_e = torch.mean(
                    torch.square(energy_pred_phys - energy_label) * inv_natoms**2
                ).sqrt()
                more_loss["rmse_e"] = self.display_if_exist(
                    rmse_e.detach(),
                    find_energy,
                )
            elif self.loss_func == "mae":
                l1_ener_loss = torch.mean(
                    torch.abs(energy_pred - energy_label_norm) * inv_natoms
                )
                loss += pref_e * l1_ener_loss
                mae_e = torch.mean(
                    torch.abs(energy_pred_phys - energy_label) * inv_natoms
                )
                more_loss["mae_e"] = self.display_if_exist(
                    mae_e.detach(),
                    find_energy,
                )
            else:
                raise NotImplementedError(
                    f"Loss type {self.loss_func} is not implemented for `dens` energy loss."
                )
            if mae:
                mae_e = torch.mean(
                    torch.abs(energy_pred_phys - energy_label) * inv_natoms
                )
                more_loss["mae_e"] = self.display_if_exist(mae_e.detach(), find_energy)
                mae_e_all = torch.mean(torch.abs(energy_pred_phys - energy_label))
                more_loss["mae_e_all"] = self.display_if_exist(
                    mae_e_all.detach(),
                    find_energy,
                )

        if self.has_f and "force" in model_pred and "force" in label:
            find_force = label.get("find_force", 0.0)
            clean_force_pred_norm = self._canonicalize_vec3_tensor(
                model_pred.get(
                    "clean_force_norm",
                    model_pred.get("force_norm", model_pred["force"]),
                ),
                nf=force_label.shape[0],
                nloc=force_label.shape[1],
                name="predicted normalized clean force",
            )
            denoising_force_pred_norm = self._canonicalize_vec3_tensor(
                model_pred.get(
                    "denoising_force_norm",
                    model_pred.get("force_norm", model_pred["force"]),
                ),
                nf=force_label.shape[0],
                nloc=force_label.shape[1],
                name="predicted normalized denoising force",
            )
            if "force_norm" in model_pred:
                force_pred_phys = self._canonicalize_vec3_tensor(
                    model_pred["force"],
                    nf=force_label.shape[0],
                    nloc=force_label.shape[1],
                    name="predicted physical force",
                )
            else:
                force_pred_phys = atomic_model.denorm_dens_force(clean_force_pred_norm)
            force_target_norm = atomic_model.norm_dens_force(force_label)
            noise_only_mask = noise_mask if use_dens else torch.zeros_like(noise_mask)
            # Each subset loss averages over its own atoms and is then scaled
            # by that subset's share, so the two recombine into one average
            # over the real atoms. The share is taken against the real-atom
            # count rather than the padded width, or phantom slots would
            # dilute both terms by the padding fraction.
            real_count = (
                (input_dict["atype"] >= 0).sum().to(dtype=GLOBAL_PT_FLOAT_PRECISION)
            )
            clean_fraction = clean_mask.sum() / real_count
            noise_fraction = noise_only_mask.sum() / real_count
            clean_force_loss = self._compute_force_subset_loss(
                clean_force_pred_norm[clean_mask].reshape(-1, 3),
                force_target_norm[clean_mask].reshape(-1, 3),
                coefficient=(pref_f * find_force) * clean_fraction,
            )
            loss += clean_force_loss
            if use_dens:
                noise_force_loss = self._compute_force_subset_loss(
                    denoising_force_pred_norm[noise_only_mask].reshape(-1, 3),
                    noise_target[noise_only_mask].reshape(-1, 3),
                    coefficient=(denoise_pref * find_force) * noise_fraction,
                )
                loss += noise_force_loss
            if self.loss_func == "mse":
                diff_clean = clean_force_pred_norm[clean_mask].reshape(
                    -1, 3
                ) - force_target_norm[clean_mask].reshape(-1, 3)
                diff_noise = denoising_force_pred_norm[noise_only_mask].reshape(
                    -1, 3
                ) - noise_target[noise_only_mask].reshape(-1, 3)
                l2_num = torch.sum(torch.square(diff_clean))
                l2_den = max(diff_clean.numel(), 1)
                if noise_count := int(noise_only_mask.sum().item()):
                    l2_num = l2_num + torch.sum(torch.square(diff_noise))
                    l2_den += diff_noise.numel()
                l2_force_loss = l2_num / l2_den
                if not self.inference:
                    more_loss["l2_force_loss"] = self.display_if_exist(
                        l2_force_loss.detach(),
                        find_force,
                    )
            elif self.loss_func == "mae":
                pass
            clean_count = int(clean_mask.sum().item())
            if clean_count > 0:
                clean_force_pred_phys = force_pred_phys[clean_mask].reshape(-1, 3)
                clean_force_label_phys = force_label[clean_mask].reshape(-1, 3)
                if self.loss_func == "mse":
                    clean_rmse_f = torch.mean(
                        torch.square(clean_force_pred_phys - clean_force_label_phys)
                    ).sqrt()
                    more_loss["rmse_f"] = self.display_if_exist(
                        clean_rmse_f.detach(),
                        find_force,
                    )
                elif self.loss_func == "mae":
                    clean_mae_f = torch.linalg.vector_norm(
                        clean_force_pred_phys - clean_force_label_phys,
                        ord=2,
                        dim=1,
                        keepdim=True,
                    ).mean()
                    more_loss["mae_f"] = self.display_if_exist(
                        clean_mae_f.detach(),
                        find_force,
                    )
        if not self.inference:
            more_loss["rmse"] = torch.sqrt(loss.detach())
        return model_pred, loss, more_loss

    def serialize(self) -> dict:
        """Serialize the `dens` loss."""
        return {
            "@class": "DeNSLoss",
            "@version": 1,
            "starter_learning_rate": self.starter_learning_rate,
            "start_pref_e": self.start_pref_e,
            "limit_pref_e": self.limit_pref_e,
            "start_pref_f": self.start_pref_f,
            "limit_pref_f": self.limit_pref_f,
            "loss_func": self.loss_func,
            "dens_prob": self.dens_prob,
            "dens_fixed_noise_std": self.dens_fixed_noise_std,
            "dens_std": self.dens_std,
            "dens_corrupt_ratio": self.dens_corrupt_ratio,
            "dens_denoising_pos_coefficient": self.dens_denoising_pos_coefficient,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "DeNSLoss":
        """Deserialize the `dens` loss."""
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        data.pop("@class", None)
        return cls(**data)
