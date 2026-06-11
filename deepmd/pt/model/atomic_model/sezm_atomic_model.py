# SPDX-License-Identifier: LGPL-3.0-or-later
"""SeZM atomic model definitions."""

from __future__ import (
    annotations,
)

import copy
import math
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np
import torch

from deepmd.pt.model.atomic_model.dp_atomic_model import (
    DPAtomicModel,
)
from deepmd.pt.model.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.pt.model.descriptor.sezm_nn import (
    SeZMDeNSFittingNet,
)
from deepmd.pt.model.task.base_fitting import (
    BaseFitting,
)
from deepmd.pt.model.task.ener import (
    EnergyFittingNet,
    EnergyFittingNetDirect,
    InvarFitting,
)
from deepmd.pt.model.task.sezm_ener import (
    SeZMEnergyFittingNet,
)
from deepmd.pt.utils.utils import (
    to_torch_tensor,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

if TYPE_CHECKING:
    from deepmd.dpmodel import (
        FittingOutputDef,
    )
    from deepmd.utils.path import (
        DPPath,
    )


class SeZMAtomicModel(DPAtomicModel):
    """Atomic model scaffold for SeZM parallel `ener` / `dens` fitting.

    Parameters
    ----------
    descriptor
        Descriptor instance.
    fitting
        Standard `ener` fitting network instance.
    dens_fitting
        Optional parallel `dens` fitting network instance.
    type_map
        Atom type map.
    active_mode
        Default active execution mode.
    **kwargs
        Additional keyword arguments forwarded to DPAtomicModel.

    Raises
    ------
    TypeError
        If fitting is not an energy fitting network.
    """

    def __init__(
        self,
        descriptor: Any,
        fitting: Any,
        type_map: Any,
        dens_fitting: Any | None = None,
        active_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        if not (
            isinstance(fitting, EnergyFittingNet)
            or isinstance(fitting, EnergyFittingNetDirect)
            or isinstance(fitting, InvarFitting)
        ):
            raise TypeError(
                "fitting must be an instance of EnergyFittingNet, EnergyFittingNetDirect or InvarFitting for SeZMAtomicModel"
            )
        if dens_fitting is not None and not isinstance(
            dens_fitting, SeZMDeNSFittingNet
        ):
            raise TypeError(
                "dens_fitting must be an instance of SeZMDeNSFittingNet for SeZMAtomicModel"
            )
        super().__init__(descriptor, fitting, type_map, **kwargs)
        self.register_buffer(
            "dens_force_rmsd",
            self.out_std.new_tensor(1.0),
        )
        self.dens_fitting_net = dens_fitting
        # Start unlocked when `active_mode` is not provided.
        # The mode will be decided later by training setup (`loss.type`)
        # or inferred from checkpoint contents during state_dict loading.
        self._mode_locked = active_mode is not None
        self._active_mode = "ener"
        if active_mode is not None:
            self.set_active_mode(active_mode)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Materialize the optional `dens` head before recursive loading."""
        dens_rmsd_key = prefix + "dens_force_rmsd"
        if dens_rmsd_key not in state_dict:
            state_dict[dens_rmsd_key] = self.dens_force_rmsd.data.clone()
        has_dens_state = any(
            key.startswith(prefix + "dens_fitting_net.") for key in state_dict
        )
        if self.dens_fitting_net is None and has_dens_state:
            self._ensure_dens_fitting_net()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        # Training mode should normally come from `loss.type`.
        # This is only a fallback for bare state_dict loads when mode was not restored.
        if has_dens_state and not self._mode_locked:
            self._active_mode = "dens"

    def get_active_mode(self) -> str:
        """Return the current SeZM execution mode."""
        return str(getattr(self, "_active_mode", "ener"))

    def _compute_or_load_dens_force_stat(
        self,
        sampled_func: Any,
        stat_file_path: DPPath | None = None,
    ) -> None:
        """
        Compute or load the SeZM `dens` direct-force RMSD scale.

        Parameters
        ----------
        sampled_func
            Packed statistics samples or a lazy callable that returns them.
        stat_file_path
            Statistics file path.

        Raises
        ------
        ValueError
            If force labels are unavailable for SeZM `dens` statistics.
        """
        force_stat_path = (
            None if stat_file_path is None else stat_file_path / "rmsd_dforce"
        )
        if force_stat_path is not None and force_stat_path.is_file():
            force_rmsd = float(np.asarray(force_stat_path.load_numpy()).reshape(-1)[0])
        else:
            sampled = sampled_func() if callable(sampled_func) else sampled_func
            force_square_sum = 0.0
            force_atom_count = 0
            for sample in sampled:
                find_force = sample.get("find_force", 0.0)
                if isinstance(find_force, torch.Tensor):
                    find_force = float(find_force.detach().cpu().item())
                if not bool(find_force):
                    continue

                force = sample.get("force")
                atype = sample.get("atype")
                if force is None or atype is None:
                    continue

                force_np = (
                    force.detach().cpu().numpy()
                    if isinstance(force, torch.Tensor)
                    else np.asarray(force)
                )
                atype_np = (
                    atype.detach().cpu().numpy()
                    if isinstance(atype, torch.Tensor)
                    else np.asarray(atype)
                )
                if force_np.ndim == 2 and atype_np.ndim == 2:
                    force_np = force_np.reshape(*atype_np.shape, 3)
                if force_np.ndim != 3 or atype_np.ndim != 2:
                    raise ValueError(
                        "SeZM `dens` force statistics expect `force` with shape "
                        "(nf, nloc, 3) or (nf, nloc*3)."
                    )

                atom_mask = atype_np >= 0
                exclude_types = sample.get("atom_exclude_types", [])
                for type_idx in exclude_types:
                    atom_mask &= atype_np != type_idx
                valid_force = force_np[atom_mask]
                if valid_force.size == 0:
                    continue
                force_square_sum += float(np.square(valid_force).sum())
                force_atom_count += int(valid_force.shape[0])

            if force_atom_count == 0:
                raise ValueError(
                    "SeZM `dens` statistics require atomic `force` labels so that "
                    "the global direct-force RMSD can be computed."
                )
            force_rmsd = math.sqrt(force_square_sum / force_atom_count)
            if force_stat_path is not None:
                force_stat_path.save_numpy(np.asarray([force_rmsd], dtype=np.float64))

        if force_rmsd <= 0.0:
            raise ValueError("SeZM `dens` direct-force RMSD must be positive.")
        self.dens_force_rmsd.copy_(self.dens_force_rmsd.new_tensor(force_rmsd))

    def _get_dens_energy_stat_tensors(
        self,
        atype: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return the SeZM `dens` energy bias/std tensors derived from `out_stat`.

        Parameters
        ----------
        atype
            Local atom types with shape `(nf, nloc)`.
        dtype
            Target floating-point dtype.
        device
            Target device.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Per-atom energy bias, per-atom broadcast energy std, and system-level
            global energy std.
        """
        out_bias, out_std = self._fetch_out_stat(["energy"])
        atom_mask = self.make_atom_mask(atype)
        if self.atom_excl is not None:
            atom_mask *= self.atom_excl(atype)
        safe_atype = atype.clamp_min(0)
        energy_bias_atom = out_bias["energy"][safe_atype].to(device=device, dtype=dtype)
        energy_std_atom = out_std["energy"][safe_atype].to(device=device, dtype=dtype)
        atom_mask_float = atom_mask.to(device=device, dtype=dtype).unsqueeze(-1)
        energy_bias_atom = energy_bias_atom * atom_mask_float
        energy_std_atom = energy_std_atom * atom_mask_float
        energy_std = out_std["energy"][0].to(device=device, dtype=dtype).view(1, -1)
        return energy_bias_atom, energy_std_atom, energy_std

    def norm_dens_energy(
        self,
        energy: torch.Tensor,
        atype: torch.Tensor,
    ) -> torch.Tensor:
        """
        Normalize `dens` system energies using the standard energy bias and
        the global residual std.

        Parameters
        ----------
        energy
            System energy tensor.
        atype
            Local atom types with shape `(nf, nloc)`.

        Returns
        -------
        torch.Tensor
            Normalized energy tensor.
        """
        energy_bias_atom, _, energy_std = self._get_dens_energy_stat_tensors(
            atype,
            dtype=energy.dtype,
            device=energy.device,
        )
        energy_bias = energy_bias_atom.sum(dim=1)
        return (energy - energy_bias) / energy_std

    def denorm_dens_energy(
        self,
        energy: torch.Tensor,
        atype: torch.Tensor,
    ) -> torch.Tensor:
        """
        Denormalize `dens` system energies using the standard energy bias
        and the global residual std.

        Parameters
        ----------
        energy
            Normalized system energy tensor.
        atype
            Local atom types with shape `(nf, nloc)`.

        Returns
        -------
        torch.Tensor
            Physical energy tensor.
        """
        energy_bias_atom, _, energy_std = self._get_dens_energy_stat_tensors(
            atype,
            dtype=energy.dtype,
            device=energy.device,
        )
        energy_bias = energy_bias_atom.sum(dim=1)
        return energy * energy_std + energy_bias

    def norm_dens_force(self, force: torch.Tensor) -> torch.Tensor:
        """
        Normalize `dens` direct-force targets with the global RMSD.

        Parameters
        ----------
        force
            Physical direct-force tensor.

        Returns
        -------
        torch.Tensor
            Normalized force tensor.
        """
        force_rmsd = self.dens_force_rmsd.to(device=force.device, dtype=force.dtype)
        return force / force_rmsd

    def denorm_dens_force(self, force: torch.Tensor) -> torch.Tensor:
        """
        Denormalize `dens` direct-force predictions with the global RMSD.

        Parameters
        ----------
        force
            Normalized direct-force tensor.

        Returns
        -------
        torch.Tensor
            Physical direct-force tensor.
        """
        force_rmsd = self.dens_force_rmsd.to(device=force.device, dtype=force.dtype)
        return force * force_rmsd

    def apply_out_stat_dens(
        self,
        ret: dict[str, torch.Tensor],
        atype: torch.Tensor,
        *,
        noise_mask: torch.Tensor,
        energy_redu_dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        """
        Apply SeZM `dens` output-stat semantics for both normalized training
        outputs and public physical predictions.

        Parameters
        ----------
        ret
            Raw normalized `dens` outputs with keys `energy`, `clean_dforce`, and
            `denoising_dforce`.
        atype
            Local atom types with shape `(nf, nloc)`.
        noise_mask
            Corruption mask with shape `(nf, nloc)`.
        energy_redu_dtype
            Reduction dtype used for summed system energies.

        Returns
        -------
        dict[str, torch.Tensor]
            Outputs carrying normalized tensors for loss calculation together
            with public DeePMD-style physical predictions.
        """
        atom_mask = self.make_atom_mask(atype).to(torch.int32)
        if self.atom_excl is not None:
            atom_mask *= self.atom_excl(atype)

        atom_mask_float = atom_mask.to(dtype=ret["energy"].dtype)
        energy_bias_atom, energy_std_atom, _ = self._get_dens_energy_stat_tensors(
            atype,
            dtype=ret["energy"].dtype,
            device=ret["energy"].device,
        )
        energy_norm = ret["energy"] * atom_mask_float.unsqueeze(-1)
        energy = energy_norm * energy_std_atom + energy_bias_atom
        energy_redu_norm = torch.sum(energy_norm.to(energy_redu_dtype), dim=1)
        energy_redu = torch.sum(energy.to(energy_redu_dtype), dim=1)

        clean_dforce_norm = ret["clean_dforce"] * atom_mask.to(
            dtype=ret["clean_dforce"].dtype
        ).unsqueeze(-1)
        denoising_dforce_norm = ret["denoising_dforce"] * atom_mask.to(
            dtype=ret["denoising_dforce"].dtype
        ).unsqueeze(-1)
        dforce_norm = torch.where(
            noise_mask.unsqueeze(-1),
            denoising_dforce_norm,
            clean_dforce_norm,
        )
        clean_dforce = self.denorm_dens_force(clean_dforce_norm)
        return {
            "energy": energy,
            "energy_redu": energy_redu,
            "dforce": clean_dforce,
            "energy_norm": energy_redu_norm,
            "atom_energy_norm": energy_norm,
            "dforce_norm": dforce_norm,
            "clean_dforce_norm": clean_dforce_norm,
            "denoising_dforce_norm": denoising_dforce_norm,
            "mask": atom_mask,
        }

    def _ensure_dens_fitting_net(self) -> SeZMDeNSFittingNet:
        """
        Materialize the optional `dens` fitting head from the current energy head.

        Returns
        -------
        SeZMDeNSFittingNet
            The existing or newly created `dens` fitting head.
        """
        dens_fitting = getattr(self, "dens_fitting_net", None)
        if dens_fitting is not None:
            return dens_fitting
        self.dens_fitting_net = SeZMDeNSFittingNet(**self._build_dens_fitting_kwargs())
        return self.dens_fitting_net

    def get_dens_fitting_net(self) -> SeZMDeNSFittingNet:
        """Return the `dens` fitting head, materializing it on demand."""
        return self._ensure_dens_fitting_net()

    def set_active_mode(self, mode: str) -> None:
        """
        Switch the active SeZM execution mode.

        Parameters
        ----------
        mode
            Target mode. Must be `ener` or `dens`.
        """
        normalized = str(mode).lower()
        if normalized not in {"ener", "dens"}:
            raise ValueError(f"Unsupported SeZM mode: {mode!r}")
        if normalized == "dens":
            self._ensure_dens_fitting_net()
        self._mode_locked = True
        self._active_mode = normalized

    def get_active_fitting_net(self) -> Any:
        """Return the fitting network selected by the current active mode."""
        if self.get_active_mode() == "dens":
            return self._ensure_dens_fitting_net()
        return self.fitting_net

    def reset_head_for_mode(self, mode: str) -> None:
        """
        Reinitialize the fitting head of certain mode from stored kwargs.

        Parameters
        ----------
        mode
            Target mode to reset.
        """
        normalized = str(mode).lower()
        if normalized == "ener":
            self.fitting_net = SeZMEnergyFittingNet(**self._build_ener_fitting_kwargs())
        elif normalized == "dens":
            self.dens_fitting_net = None
            self._ensure_dens_fitting_net()
        else:
            raise ValueError(f"Unsupported SeZM mode: {mode!r}")

    @torch.jit.unused
    def fitting_output_def(self) -> FittingOutputDef:
        """Return the fitting output definition of the active SeZM mode."""
        active_fitting = self.get_active_fitting_net()
        if active_fitting is None:
            return super().fitting_output_def()
        return active_fitting.output_def()

    def set_eval_fitting_last_layer_hook(self, enable: bool) -> None:
        """
        Set the fitting-last-layer evaluation hook for the active fitting path.

        Parameters
        ----------
        enable
            Whether to enable the hook.
        """
        self.enable_eval_fitting_last_layer_hook = enable
        active_fitting = self.get_active_fitting_net()
        if active_fitting is not None and hasattr(
            active_fitting, "set_return_middle_output"
        ):
            active_fitting.set_return_middle_output(enable)
        self.eval_fitting_last_layer_list.clear()

    def change_type_map(
        self,
        type_map: list[str],
        model_with_new_type_stat: SeZMAtomicModel | None = None,
    ) -> None:
        """
        Change the type map for the descriptor and both SeZM fitting heads.

        Parameters
        ----------
        type_map
            New atom type map.
        model_with_new_type_stat
            Optional reference model that carries new-type statistics.
        """
        super().change_type_map(
            type_map=type_map,
            model_with_new_type_stat=model_with_new_type_stat,
        )
        if self.dens_fitting_net is not None:
            ref_dens = (
                None
                if model_with_new_type_stat is None
                else model_with_new_type_stat.dens_fitting_net
            )
            self.dens_fitting_net.change_type_map(
                type_map=type_map,
                model_with_new_type_stat=ref_dens,
            )

    def compute_or_load_stat(
        self,
        sampled_func: Any,
        stat_file_path: Any = None,
        compute_or_load_out_stat: bool = True,
        preset_observed_type: list[str] | None = None,
    ) -> None:
        """
        Compute/load SeZM statistics for the active execution mode.

        Parameters
        ----------
        sampled_func
            Lazy sampler providing training frames.
        stat_file_path
            Statistics file path.
        compute_or_load_out_stat
            Whether to compute or load output statistics. `dens` mode keeps the
            standard `ener`-branch statistics intact and additionally fits one
            global direct-force RMSD scale for the normalized DeNS training
            path. The `dens` energy path reuses the standard per-type energy bias
            and the broadcast global residual std already stored in `out_stat`.
        preset_observed_type
            Optional observed-type override.
        """
        original_mode = self.get_active_mode()
        if stat_file_path is not None and self.type_map is not None:
            stat_file_path /= " ".join(self.type_map)

        wrapped_sampler = self._make_wrapped_sampler(sampled_func)
        self.descriptor.compute_input_stats(wrapped_sampler, stat_file_path)
        self.compute_fitting_input_stat(wrapped_sampler, stat_file_path)
        if compute_or_load_out_stat:
            self.set_active_mode("ener")
            try:
                self.compute_or_load_out_stat(wrapped_sampler, stat_file_path)
            finally:
                self.set_active_mode(original_mode)
            if original_mode == "dens":
                self._compute_or_load_dens_force_stat(wrapped_sampler, stat_file_path)

        self._collect_and_set_observed_type(
            wrapped_sampler,
            stat_file_path,
            preset_observed_type,
        )

    def apply_out_stat(
        self,
        ret: dict[str, torch.Tensor],
        atype: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Apply SeZM-specific output statistics.

        Parameters
        ----------
        ret
            Atomic fitting outputs.
        atype
            Local atom types with shape `(nf, nloc)`.

        Returns
        -------
        dict[str, torch.Tensor]
            Outputs after SeZM output-stat post-processing.
        """
        if "energy" in ret:
            out_bias, _ = self._fetch_out_stat(["energy"])
            ret["energy"] = ret["energy"] + out_bias["energy"][atype]
        return ret

    def get_dim_fparam(self) -> int:
        """Return frame-parameter width of the active SeZM branch."""
        active_fitting = self.get_active_fitting_net()
        if active_fitting is not None and hasattr(active_fitting, "get_dim_fparam"):
            return active_fitting.get_dim_fparam()
        return super().get_dim_fparam()

    def has_default_fparam(self) -> bool:
        """Return whether the active SeZM branch has default frame parameters."""
        active_fitting = self.get_active_fitting_net()
        if active_fitting is not None and hasattr(active_fitting, "has_default_fparam"):
            return active_fitting.has_default_fparam()
        return super().has_default_fparam()

    def get_default_fparam(self) -> torch.Tensor | None:
        """Return default frame parameters of the active SeZM branch."""
        active_fitting = self.get_active_fitting_net()
        if active_fitting is not None and hasattr(active_fitting, "get_default_fparam"):
            return active_fitting.get_default_fparam()
        return super().get_default_fparam()

    def has_chg_spin_ebd(self) -> bool:
        """Return whether charge/spin condition embedding is enabled."""
        return bool(getattr(self.descriptor, "add_chg_spin_ebd", False))

    def get_dim_chg_spin(self) -> int:
        """Return charge/spin condition width."""
        if self.has_chg_spin_ebd() and hasattr(self.descriptor, "get_dim_chg_spin"):
            return self.descriptor.get_dim_chg_spin()
        return 0

    def has_default_chg_spin(self) -> bool:
        """Return whether default charge/spin conditions are configured."""
        if self.has_chg_spin_ebd() and hasattr(self.descriptor, "has_default_chg_spin"):
            return self.descriptor.has_default_chg_spin()
        return False

    def get_default_chg_spin(self) -> torch.Tensor | None:
        """Return default charge/spin conditions as a tensor."""
        if self.has_chg_spin_ebd() and hasattr(self.descriptor, "get_default_chg_spin"):
            default_chg_spin = self.descriptor.get_default_chg_spin()
            if default_chg_spin is not None:
                return self.out_std.new_tensor(default_chg_spin)
        return None

    def get_dim_aparam(self) -> int:
        """Return atomic-parameter width of the active SeZM branch."""
        active_fitting = self.get_active_fitting_net()
        if active_fitting is not None and hasattr(active_fitting, "get_dim_aparam"):
            return active_fitting.get_dim_aparam()
        return super().get_dim_aparam()

    def get_sel_type(self) -> list[int]:
        """Return selected atom types of the active SeZM branch."""
        active_fitting = self.get_active_fitting_net()
        if active_fitting is not None and hasattr(active_fitting, "get_sel_type"):
            return active_fitting.get_sel_type()
        return super().get_sel_type()

    def serialize(self) -> dict:
        """Serialize the SeZM atomic model including the optional `dens` head."""
        data = DPAtomicModel.serialize(self)
        data["@variables"]["dens_force_rmsd"] = (
            self.dens_force_rmsd.detach().cpu().numpy()
        )
        data.update(
            {
                "@version": 3,
                "type": "sezm_atomic",
                "dens_fitting": None
                if self.dens_fitting_net is None
                else self.dens_fitting_net.serialize(),
                "active_mode": self.get_active_mode(),
            }
        )
        return data

    def _build_ener_fitting_kwargs(self) -> dict[str, Any]:
        """Reconstruct SeZM energy-head kwargs from the current fitting head."""
        fitting = self.fitting_net
        return {
            "ntypes": int(fitting.ntypes),
            "dim_descrpt": int(fitting.dim_descrpt),
            "neuron": copy.deepcopy(list(fitting.neuron)),
            "bias_atom_e": None
            if fitting.bias_atom_e is None
            else fitting.bias_atom_e.detach().cpu().numpy().copy(),
            "resnet_dt": bool(fitting.resnet_dt),
            "numb_fparam": int(fitting.numb_fparam),
            "numb_aparam": int(fitting.numb_aparam),
            "dim_case_embd": int(fitting.dim_case_embd),
            "case_film_embd": bool(getattr(fitting, "case_film_embd", False)),
            "activation_function": str(fitting.activation_function),
            "bias_out": bool(getattr(fitting, "bias_out", False)),
            "precision": str(fitting.precision),
            "mixed_types": bool(fitting.mixed_types),
            "seed": copy.deepcopy(fitting.seed),
            "type_map": None if fitting.type_map is None else list(fitting.type_map),
            "default_fparam": copy.deepcopy(fitting.default_fparam),
            "rcond": fitting.rcond,
            "exclude_types": copy.deepcopy(fitting.exclude_types),
            "trainable": copy.deepcopy(fitting.trainable),
            "atom_ener": copy.deepcopy(fitting.atom_ener),
            "use_aparam_as_mask": bool(fitting.use_aparam_as_mask),
        }

    def _build_dens_fitting_kwargs(self) -> dict[str, Any]:
        """Reconstruct SeZM `dens`-head kwargs from energy head and descriptor."""
        descriptor = self.descriptor
        kwargs = self._build_ener_fitting_kwargs()
        node_l_schedule = getattr(descriptor, "node_l_schedule", descriptor.l_schedule)
        kwargs["condition_lmax"] = int(node_l_schedule[0])
        kwargs["latent_lmax"] = int(node_l_schedule[-1])
        kwargs["channels"] = int(descriptor.channels)
        return kwargs

    @classmethod
    def deserialize(cls, data: dict) -> SeZMAtomicModel:
        """
        Deserialize the SeZM atomic model.

        Parameters
        ----------
        data
            Serialized atomic-model data.

        Returns
        -------
        SeZMAtomicModel
            Deserialized SeZM atomic model.
        """
        payload = data.copy()
        version = int(payload.pop("@version", 2))
        check_version_compatibility(version, 3, 2)
        payload.pop("@class", None)
        payload.pop("type", None)

        descriptor_obj = BaseDescriptor.deserialize(payload.pop("descriptor"))
        fitting_payload = payload.pop("fitting")
        fitting_obj = BaseFitting.deserialize(fitting_payload)
        dens_payload = payload.pop("dens_fitting", None)
        dens_obj = (
            None
            if dens_payload is None
            else SeZMDeNSFittingNet.deserialize(dens_payload)
        )
        active_mode = payload.pop("active_mode", None)
        payload["descriptor"] = descriptor_obj
        payload["fitting"] = fitting_obj
        payload["dens_fitting"] = dens_obj
        payload["active_mode"] = active_mode
        variables = payload.pop("@variables", None)
        obj = cls(**payload)
        variables = (
            {"out_bias": None, "out_std": None} if variables is None else variables
        )
        obj["out_bias"] = (
            to_torch_tensor(variables["out_bias"])
            if variables["out_bias"] is not None
            else obj._default_bias()
        )
        obj["out_std"] = (
            to_torch_tensor(variables["out_std"])
            if variables["out_std"] is not None
            else obj._default_std()
        )
        dens_force_rmsd = variables.get("dens_force_rmsd")
        if dens_force_rmsd is not None:
            obj.dens_force_rmsd.copy_(to_torch_tensor(dens_force_rmsd))
        return obj
