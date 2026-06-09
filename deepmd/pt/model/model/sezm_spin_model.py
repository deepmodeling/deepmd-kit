# SPDX-License-Identifier: LGPL-3.0-or-later
"""Spin-enabled SeZM energy model."""

import functools
from collections.abc import (
    Callable,
)
from copy import (
    deepcopy,
)
from typing import (
    Any,
)

import torch

from deepmd.dpmodel import (
    ModelOutputDef,
)
from deepmd.pt.model.atomic_model.sezm_atomic_model import (
    SeZMAtomicModel,
)
from deepmd.pt.model.descriptor.sezm_nn import (
    nvtx_range,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)
from deepmd.pt.model.model.sezm_model import (
    InterPotential,
    SeZMModel,
)
from deepmd.pt.model.model.spin_model import (
    SpinModel,
    _lookup_type_values,
)
from deepmd.pt.utils.utils import (
    to_torch_tensor,
)
from deepmd.utils.path import (
    DPPath,
)
from deepmd.utils.spin import (
    Spin,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


@BaseModel.register("sezm_spin")
class SeZMSpinModel(SeZMModel):
    """SeZM energy model with virtual spin atoms.

    Parameters
    ----------
    spin
        Spin metadata describing magnetic real types and virtual displacement
        scales.
    *args
        Positional arguments forwarded to :class:`SeZMModel`.
    **kwargs
        Keyword arguments forwarded to :class:`SeZMModel`.
    """

    model_type = "sezm_spin"

    def __init__(
        self,
        *args: Any,
        spin: Spin,
        real_sel: list[int],
        **kwargs: Any,
    ) -> None:
        # Delay InterPotential construction until ntypes_real is available.
        bridging_method = str(kwargs.pop("bridging_method", "none")).upper()
        kwargs["bridging_method"] = "none"

        super().__init__(*args, **kwargs)
        self.spin = spin
        self.ntypes_real = self.spin.ntypes_real
        self.real_sel = [int(sel) for sel in real_sel]
        self.register_buffer(
            "virtual_scale_mask",
            to_torch_tensor(self.spin.get_virtual_scale_mask()),
            persistent=False,
        )
        self.register_buffer(
            "spin_mask",
            to_torch_tensor(self.spin.get_spin_mask()),
            persistent=False,
        )

        self.bridging_method = bridging_method
        self.inter_potential = (
            InterPotential(type_map=self.get_type_map(), mode=self.bridging_method)
            if self.bridging_method != "NONE"
            else None
        )

    # =========================================================================
    # Forward Methods
    # =========================================================================

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        spin: torch.Tensor,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return spin-aware SeZM predictions with public output keys."""
        model_ret = self.forward_common(
            coord,
            atype,
            spin,
            box=box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            charge_spin=charge_spin,
        )
        model_predict: dict[str, torch.Tensor] = {
            "atom_energy": model_ret["energy"],
            "energy": model_ret["energy_redu"],
            "mask_mag": model_ret["mask_mag"],
        }
        if self.do_grad_r("energy"):
            model_predict["force"] = model_ret["energy_derv_r"].squeeze(-2)
            model_predict["force_mag"] = model_ret["energy_derv_r_mag"].squeeze(-2)
        if self.do_grad_c("energy"):
            model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
            if do_atomic_virial:
                model_predict["atom_virial"] = model_ret["energy_derv_c"].squeeze(-2)
        return model_predict

    def forward_common(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        spin: torch.Tensor,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return spin-aware SeZM predictions with internal output keys."""
        with nvtx_range("SeZMSpin/forward_common"):
            cc, bb, fp, ap, input_prec = self._input_type_cast(
                coord, box=box, fparam=fparam, aparam=aparam
            )
            del coord, box, fparam, aparam
            atype = atype.to(device=cc.device, dtype=torch.long)
            nf, nloc = atype.shape[:2]
            if cc.ndim == 2:
                cc = cc.view(nf, nloc, 3)
            spin = spin.to(dtype=cc.dtype, device=cc.device).reshape(nf, nloc, 3)

            extended_coord, extended_atype, mapping, nlist = self.build_neighbor_list(
                cc, atype, bb
            )
            extended_spin = torch.gather(
                spin,
                1,
                mapping.unsqueeze(-1).expand(-1, -1, 3),
            )
            (
                extended_coord_updated,
                extended_atype_updated,
                nlist_updated,
                mapping_updated,
                extended_coord_corr,
            ) = self.process_spin_input_lower(
                extended_coord,
                extended_atype,
                extended_spin,
                nlist,
                mapping=mapping,
            )
            if ap is not None:
                ap = self.expand_aparam(ap, nloc * 2)
            model_ret = self.forward_common_after_nlist(
                extended_coord_updated,
                extended_atype_updated,
                mapping_updated,
                nlist_updated,
                extended_atype_updated[:, : nloc * 2],
                fp,
                ap,
                input_prec,
                do_atomic_virial=do_atomic_virial,
                extended_coord_corr=extended_coord_corr,
                charge_spin=charge_spin,
            )
            return self._split_spin_common_output(model_ret, atype, nloc)

    def forward_lower(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        extended_spin: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        comm_dict: dict[str, torch.Tensor] | None = None,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return spin-aware SeZM lower-interface predictions."""
        model_ret = self.forward_common_lower(
            extended_coord,
            extended_atype,
            extended_spin,
            nlist,
            mapping=mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            comm_dict=comm_dict,
            charge_spin=charge_spin,
            extra_nlist_sort=self.need_sorted_nlist_for_lower(),
        )
        model_predict: dict[str, torch.Tensor] = {
            "atom_energy": model_ret["energy"],
            "energy": model_ret["energy_redu"],
            "extended_mask_mag": model_ret["mask_mag"],
        }
        if self.do_grad_r("energy"):
            model_predict["extended_force"] = model_ret["energy_derv_r"].squeeze(-2)
            model_predict["extended_force_mag"] = model_ret[
                "energy_derv_r_mag"
            ].squeeze(-2)
        if self.do_grad_c("energy"):
            model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
            if do_atomic_virial:
                model_predict["extended_virial"] = model_ret["energy_derv_c"].squeeze(
                    -2
                )
        return model_predict

    def forward_common_lower(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        extended_spin: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        comm_dict: dict[str, torch.Tensor] | None = None,
        extra_nlist_sort: bool = False,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return spin-aware lower-interface predictions with internal keys."""
        extended_atype = extended_atype.to(
            device=extended_coord.device, dtype=torch.long
        )
        _, nloc = nlist.shape[:2]
        (
            extended_coord_updated,
            extended_atype_updated,
            nlist_updated,
            mapping_updated,
            extended_coord_corr,
        ) = self.process_spin_input_lower(
            extended_coord,
            extended_atype,
            extended_spin,
            nlist,
            mapping=mapping,
        )
        if aparam is not None:
            aparam = self.expand_aparam(aparam, nloc * 2)
        model_ret = super().forward_common_lower(
            extended_coord_updated,
            extended_atype_updated,
            nlist_updated,
            mapping=mapping_updated,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            comm_dict=comm_dict,
            extra_nlist_sort=extra_nlist_sort,
            extended_coord_corr=extended_coord_corr,
            charge_spin=charge_spin,
        )
        return self._split_spin_lower_output(model_ret, extended_atype, nloc)

    def forward_common_lower_exportable(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        extended_spin: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        charge_spin: torch.Tensor | None = None,
        *,
        do_atomic_virial: bool = False,
    ) -> torch.nn.Module:
        """Trace the spin lower interface into an exportable FX graph."""
        extra_sort = self.need_sorted_nlist_for_lower()

        def lower_fn(
            ext_coord: torch.Tensor,
            ext_atype: torch.Tensor,
            ext_spin: torch.Tensor,
            nlist_: torch.Tensor,
            mapping_: torch.Tensor | None,
            fparam_: torch.Tensor | None,
            aparam_: torch.Tensor | None,
            charge_spin_: torch.Tensor | None,
        ) -> dict[str, torch.Tensor]:
            ext_coord = ext_coord.detach().requires_grad_(True)
            return self.forward_common_lower(
                ext_coord,
                ext_atype,
                ext_spin,
                nlist_,
                mapping_,
                fparam=fparam_,
                aparam=aparam_,
                do_atomic_virial=do_atomic_virial,
                extra_nlist_sort=extra_sort,
                charge_spin=charge_spin_,
            )

        def fn(
            ext_coord: torch.Tensor,
            ext_atype: torch.Tensor,
            ext_spin: torch.Tensor,
            nlist_: torch.Tensor,
            mapping_: torch.Tensor | None,
            fparam_: torch.Tensor | None,
            aparam_: torch.Tensor | None,
            charge_spin_: torch.Tensor | None,
        ) -> dict[str, torch.Tensor]:
            return lower_fn(
                ext_coord,
                ext_atype,
                ext_spin,
                nlist_,
                mapping_,
                fparam_,
                aparam_,
                charge_spin_,
            )

        trace_inputs = (
            extended_coord,
            extended_atype,
            extended_spin,
            nlist,
            mapping,
            fparam,
            aparam,
        )
        if self.get_dim_chg_spin() > 0:
            charge_spin = self.convert_charge_spin(
                charge_spin,
                nf=extended_atype.shape[0],
                dtype=extended_coord.dtype,
                device=extended_coord.device,
            )
        trace_inputs = (*trace_inputs, charge_spin)

        return self._trace_lower_exportable(
            fn,
            *trace_inputs,
        )

    # =========================================================================
    # Statistics and Mode Methods
    # =========================================================================

    def compute_or_load_stat(
        self,
        sampled_func: Callable[[], list[dict[str, Any]]],
        stat_file_path: DPPath | None = None,
        preset_observed_type: list[str] | None = None,
    ) -> None:
        """Compute or load statistics with virtual spin atoms included."""
        super().compute_or_load_stat(
            self._get_spin_sampled_func(sampled_func),
            stat_file_path,
            preset_observed_type=preset_observed_type,
        )

    def change_out_bias(
        self,
        merged: Callable[[], list[dict[str, Any]]] | list[dict[str, Any]],
        bias_adjust_mode: str = "change-by-statistic",
    ) -> None:
        """Change output bias using spin-expanded sampled data."""
        spin_sampled_func = self._get_spin_sampled_func(
            merged if callable(merged) else lambda: merged
        )
        super().change_out_bias(
            spin_sampled_func,
            bias_adjust_mode=bias_adjust_mode,
        )

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat: Any = None
    ) -> None:
        """Change real type map and rebuild corresponding virtual spin types."""
        type_map_with_spin = type_map + [item + "_spin" for item in type_map]
        super().change_type_map(type_map_with_spin, model_with_new_type_stat)
        self.ntypes_real = len(type_map)

    def set_active_mode(self, mode: str) -> None:
        """Switch mode, allowing only the conservative energy path."""
        normalized = str(mode).lower()
        if normalized != "ener":
            raise NotImplementedError("SeZM spin supports only the `ener` path.")
        super().set_active_mode(normalized)

    def set_active_mode_from_loss(self, loss_type: str) -> None:
        """Select execution mode from loss type."""
        normalized = str(loss_type).lower()
        if normalized == "dens":
            raise NotImplementedError("SeZM spin supports only the `ener` path.")
        if normalized in {"ener", "ener_spin"}:
            self.set_active_mode("ener")

    # =========================================================================
    # Output Definitions and Metadata
    # =========================================================================

    def has_spin(self) -> bool:
        """Return whether this model consumes spin input."""
        return True

    def get_type_map(self) -> list[str]:
        """Return the real atom type map."""
        return super().get_type_map()[: self.ntypes_real]

    def get_ntypes(self) -> int:
        """Return the number of real atom types."""
        return len(self.get_type_map())

    def get_sel(self) -> list[int]:
        """Return the public real-atom neighbor selection."""
        return self.real_sel

    def get_nsel(self) -> int:
        """Return the public real-atom total neighbor count."""
        return int(sum(self.real_sel))

    def get_nnei(self) -> int:
        """Return the public real-atom total neighbor count."""
        return int(sum(self.real_sel))

    def get_observed_type_list(self) -> list[str]:
        """Return observed real types according to the output bias."""
        type_map = self.get_type_map()
        out_bias = self.atomic_model.get_out_bias()[0]
        assert out_bias is not None, "No out_bias found in the model."
        assert out_bias.dim() == 2, "The supported out_bias should be a 2D tensor."
        assert out_bias.size(0) >= self.ntypes_real, (
            "The out_bias shape is smaller than the number of real types."
        )
        bias_mask = (
            torch.gt(torch.abs(out_bias[: self.ntypes_real]), 1e-6).any(dim=-1).cpu()
        )
        result: list[str] = []
        for t, m in zip(type_map, bias_mask.tolist()):
            if m:
                result.append(t)
        return result

    def model_output_def(self) -> ModelOutputDef:
        """Return the spin-aware model output definition."""
        var_name = self._get_output_var_name()
        atomic_output_def = self.atomic_output_def()
        atomic_output_def[var_name].magnetic = True
        return ModelOutputDef(atomic_output_def)

    def translated_output_def(self) -> dict[str, Any]:
        """Translate internal output definitions to public spin keys."""
        out_def_data = self.model_output_def().get_data()
        output_def = {
            "atom_energy": out_def_data["energy"],
            "energy": out_def_data["energy_redu"],
            "mask_mag": out_def_data["mask_mag"],
        }
        if self.do_grad_r("energy"):
            output_def["force"] = deepcopy(out_def_data["energy_derv_r"])
            output_def["force"].squeeze(-2)
            output_def["force_mag"] = deepcopy(out_def_data["energy_derv_r_mag"])
            output_def["force_mag"].squeeze(-2)
        if self.do_grad_c("energy"):
            output_def["virial"] = deepcopy(out_def_data["energy_derv_c_redu"])
            output_def["virial"].squeeze(-2)
            output_def["atom_virial"] = deepcopy(out_def_data["energy_derv_c"])
            output_def["atom_virial"].squeeze(-2)
        return output_def

    # =========================================================================
    # Serialization
    # =========================================================================

    def serialize(self) -> dict[str, Any]:
        """Serialize the SeZM spin model."""
        data = super().serialize()
        data["type"] = self.model_type
        data["spin"] = self.spin.serialize()
        data["real_sel"] = self.real_sel
        return data

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "SeZMSpinModel":
        """Deserialize a SeZM spin model."""
        data = data.copy()
        version = int(data.pop("@version", 1))
        check_version_compatibility(version, 1, 1)
        data.pop("@class", None)
        data.pop("type", None)
        spin = Spin.deserialize(data.pop("spin"))
        real_sel = data.pop("real_sel")
        atomic_model = SeZMAtomicModel.deserialize(data.pop("atomic_model"))
        return cls(atomic_model_=atomic_model, spin=spin, real_sel=real_sel, **data)

    # =========================================================================
    # Small Utilities
    # =========================================================================

    def build_neighbor_list(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the real-atom neighbor list before spin expansion."""
        return super().build_neighbor_list(coord, atype, box)

    def format_nlist(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        extra_nlist_sort: bool = False,
    ) -> torch.Tensor:
        """Format spin-expanded nlist to the internal descriptor capacity."""
        del extended_atype
        return self._format_nlist(
            extended_coord,
            nlist,
            sum(self.atomic_model.get_sel()),
            extra_nlist_sort=extra_nlist_sort,
        )

    def _get_inter_potential_real_type_count(self) -> int:
        """Return the number of real types for real-only ZBL masking."""
        return self.ntypes_real

    def _get_output_var_name(self) -> str:
        """Return the primary atomic output variable name."""
        return "energy"

    def _get_spin_sampled_func(
        self, sampled_func: Callable[[], list[dict[str, Any]]]
    ) -> Callable[[], list[dict[str, Any]]]:
        """Wrap a data sampler so statistics see real and virtual atoms."""

        @functools.lru_cache
        def spin_sampled_func() -> list[dict[str, Any]]:
            sampled = sampled_func()
            spin_sampled = []
            for sys in sampled:
                coord_updated, atype_updated, _ = self.process_spin_input(
                    sys["coord"], sys["atype"], sys["spin"]
                )
                tmp_dict = {
                    "coord": coord_updated,
                    "atype": atype_updated,
                }
                if "aparam" in sys:
                    tmp_dict["aparam"] = self.expand_aparam(
                        sys["aparam"], atype_updated.shape[1]
                    )
                if "natoms" in sys:
                    natoms = sys["natoms"]
                    tmp_dict["natoms"] = torch.cat(
                        [2 * natoms[:, :2], natoms[:, 2:], natoms[:, 2:]], dim=-1
                    )
                for item_key in sys.keys():
                    if item_key not in [
                        "coord",
                        "atype",
                        "spin",
                        "natoms",
                        "aparam",
                    ]:
                        tmp_dict[item_key] = sys[item_key]
                spin_sampled.append(tmp_dict)
            return spin_sampled

        return self.atomic_model._make_wrapped_sampler(spin_sampled_func)

    def _ensure_mask_mag(
        self,
        model_ret: dict[str, torch.Tensor],
        atype: torch.Tensor,
    ) -> None:
        """Ensure the magnetic atom mask exists in ``model_ret``."""
        if "mask_mag" in model_ret:
            return
        nframes, nloc = atype.shape[:2]
        atomic_mask = _lookup_type_values(self.virtual_scale_mask, atype).reshape(
            [nframes, nloc, 1]
        )
        model_ret["mask_mag"] = atomic_mask > 0.0

    def _split_spin_common_output(
        self,
        model_ret: dict[str, torch.Tensor],
        atype: torch.Tensor,
        nloc: int,
    ) -> dict[str, torch.Tensor]:
        """Split full-interface SeZM outputs into real and magnetic parts."""
        var_name = self._get_output_var_name()
        model_ret[var_name] = torch.split(model_ret[var_name], [nloc, nloc], dim=1)[0]
        if self.do_grad_r(var_name) and model_ret.get(f"{var_name}_derv_r") is not None:
            (
                model_ret[f"{var_name}_derv_r"],
                model_ret[f"{var_name}_derv_r_mag"],
                model_ret["mask_mag"],
            ) = self.process_spin_output(atype, model_ret[f"{var_name}_derv_r"])
        if self.do_grad_c(var_name) and model_ret.get(f"{var_name}_derv_c") is not None:
            (
                model_ret[f"{var_name}_derv_c"],
                model_ret[f"{var_name}_derv_c_mag"],
                model_ret["mask_mag"],
            ) = self.process_spin_output(
                atype,
                model_ret[f"{var_name}_derv_c"],
                add_mag=True,
                virtual_scale=False,
            )
        self._ensure_mask_mag(model_ret, atype)
        return model_ret

    def _split_spin_lower_output(
        self,
        model_ret: dict[str, torch.Tensor],
        extended_atype: torch.Tensor,
        nloc: int,
    ) -> dict[str, torch.Tensor]:
        """Split lower-interface SeZM outputs into real and magnetic parts."""
        var_name = self._get_output_var_name()
        model_ret[var_name] = torch.split(model_ret[var_name], [nloc, nloc], dim=1)[0]
        if self.do_grad_r(var_name) and model_ret.get(f"{var_name}_derv_r") is not None:
            (
                model_ret[f"{var_name}_derv_r"],
                model_ret[f"{var_name}_derv_r_mag"],
                model_ret["mask_mag"],
            ) = self.process_spin_output_lower(
                extended_atype, model_ret[f"{var_name}_derv_r"], nloc
            )
        if self.do_grad_c(var_name) and model_ret.get(f"{var_name}_derv_c") is not None:
            (
                model_ret[f"{var_name}_derv_c"],
                model_ret[f"{var_name}_derv_c_mag"],
                model_ret["mask_mag"],
            ) = self.process_spin_output_lower(
                extended_atype,
                model_ret[f"{var_name}_derv_c"],
                nloc,
                add_mag=True,
                virtual_scale=False,
            )
        self._ensure_mask_mag(model_ret, extended_atype)
        return model_ret

    process_spin_input = SpinModel.process_spin_input
    process_spin_input_lower = SpinModel.process_spin_input_lower
    process_spin_output = SpinModel.process_spin_output
    process_spin_output_lower = SpinModel.process_spin_output_lower
    extend_nlist = staticmethod(SpinModel.extend_nlist)
    expand_aparam = staticmethod(SpinModel.expand_aparam)
