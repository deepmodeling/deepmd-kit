# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
    Optional,
)

import torch
import pytorch_finufft

from deepmd.pt.model.model.transform_output import (
    communicate_extended_output,
)
from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
)

from deepmd.pt.model.atomic_model import (
    LESEnergyAtomicModel,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)

from .dp_model import (
    DPModelCommon,
)
from .make_hessian_model import (
    make_hessian_model,
)
from .make_model import (
    make_model,
)

LESEnergyModel_ = make_model(LESEnergyAtomicModel)


@BaseModel.register("les_ener")
class LESEnergyModel(DPModelCommon, LESEnergyModel_):
    model_type = "les_ener"

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        DPModelCommon.__init__(self)
        LESEnergyModel_.__init__(self, *args, **kwargs)
        self._hessian_enabled = False

    def enable_hessian(self) -> None:
        self.__class__ = make_hessian_model(type(self))
        self.hess_fitting_def = super(type(self), self).atomic_output_def()
        self.requires_hessian("energy")
        self._hessian_enabled = True

    @torch.jit.export
    def get_observed_type_list(self) -> list[str]:
        """Get observed types (elements) of the model during data statistics.

        Returns
        -------
        observed_type_list: a list of the observed types in this model.
        """
        type_map = self.get_type_map()
        out_bias = self.atomic_model.get_out_bias()[0]

        assert out_bias is not None, "No out_bias found in the model."
        assert out_bias.dim() == 2, "The supported out_bias should be a 2D tensor."
        assert out_bias.size(0) == len(type_map), (
            "The out_bias shape does not match the type_map length."
        )
        bias_mask = (
            torch.gt(torch.abs(out_bias), 1e-6).any(dim=-1).detach().cpu()
        )  # 1e-6 for stability

        observed_type_list: list[str] = []
        for i in range(len(type_map)):
            if bias_mask[i]:
                observed_type_list.append(type_map[i])
        return observed_type_list

    def translated_output_def(self) -> dict[str, Any]:
        out_def_data = self.model_output_def().get_data()
        output_def = {
            "atom_energy": out_def_data["energy"],
            "energy": out_def_data["energy_redu"],
        }
        if self.do_grad_r("energy"):
            output_def["force"] = out_def_data["energy_derv_r"]
            output_def["force"].squeeze(-2)
        if self.do_grad_c("energy"):
            output_def["virial"] = out_def_data["energy_derv_c_redu"]
            output_def["virial"].squeeze(-2)
            output_def["atom_virial"] = out_def_data["energy_derv_c"]
            output_def["atom_virial"].squeeze(-3)
        if "mask" in out_def_data:
            output_def["mask"] = out_def_data["mask"]
        if self._hessian_enabled:
            output_def["hessian"] = out_def_data["energy_derv_r_derv_r"]
        return output_def

    def _compute_les_frame_correction(
        self,
        coord: torch.Tensor,
        latent_charge: torch.Tensor,
        box: torch.Tensor,
    ) -> torch.Tensor:
        fitting = self.get_fitting_net()
        runtime_device = coord.device
        real_dtype = coord.dtype
        complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64
        latent_charge = latent_charge.to(device=runtime_device, dtype=real_dtype)
        box = box.to(device=runtime_device, dtype=real_dtype)

        sigma_raw = getattr(fitting, "sigma", None)
        if sigma_raw is None:
            raise ValueError("LES fitting net should provide `sigma` for frame correction.")
        sigma = torch.as_tensor(
            sigma_raw,
            dtype=real_dtype,
            device=runtime_device,
        ).reshape(-1)[0]
        sigma = torch.clamp(sigma, min=torch.finfo(real_dtype).eps)
        remove_self_interaction = bool(fitting.remove_self_interaction)
        n_dl = int(fitting.n_dl)

        nf, _, _ = coord.shape
        corr = torch.zeros((nf, 1), dtype=real_dtype, device=runtime_device)

        for ff in range(nf):
            r_raw = coord[ff]
            q = latent_charge[ff]
            box_frame = box[ff]

            volume = torch.det(box_frame)
            if torch.abs(volume) <= torch.finfo(real_dtype).eps:
                raise ValueError("`box` is singular (near-zero volume), cannot run NUFFT.")

            cell_inv = torch.linalg.inv(box_frame)
            r_frac = torch.matmul(r_raw, cell_inv)
            r_frac = torch.remainder(r_frac + 0.5, 1.0) - 0.5
            pi_tensor = torch.tensor(torch.pi, dtype=real_dtype, device=runtime_device)
            point_limit = pi_tensor - 32.0 * torch.finfo(real_dtype).eps
            r_in = torch.clamp(
                2.0 * pi_tensor * r_frac,
                min=-point_limit,
                max=point_limit,
            ).contiguous()
            nufft_points = r_in.transpose(0, 1).contiguous()

            norms = torch.norm(box_frame, dim=1)
            nk = [max(1, int(n.item() / n_dl)) for n in norms]
            n1 = torch.arange(-nk[0], nk[0] + 1, device=runtime_device, dtype=real_dtype)
            n2 = torch.arange(-nk[1], nk[1] + 1, device=runtime_device, dtype=real_dtype)
            n3 = torch.arange(-nk[2], nk[2] + 1, device=runtime_device, dtype=real_dtype)

            kx_grid, ky_grid, kz_grid = torch.meshgrid(n1, n2, n3, indexing="ij")
            k_sq = kx_grid**2 + ky_grid**2 + kz_grid**2
            zero_mask = k_sq == 0

            k_sq_safe = torch.where(zero_mask, torch.ones_like(k_sq), k_sq)
            kfac = torch.exp(-0.5 * (sigma**2) * k_sq_safe) / k_sq_safe
            kfac = kfac.to(dtype=real_dtype)
            kfac[zero_mask] = 0.0

            q_t = q.transpose(0, 1).contiguous()
            charge = torch.complex(q_t, torch.zeros_like(q_t)).to(dtype=complex_dtype)
            recon = pytorch_finufft.functional.finufft_type1(
                nufft_points,
                charge,
                output_shape=tuple(int(x) for x in kx_grid.shape),
                eps=1e-4,
                isign=-1,
            )
            conv = kfac.unsqueeze(0) * recon
            ifft_conv = pytorch_finufft.functional.finufft_type2(
                nufft_points,
                conv,
                eps=1e-4,
                isign=1,
            ) / (2.0 * volume)
            corr[ff, 0] = (charge * ifft_conv).real.sum()

            if remove_self_interaction:
                diag_sum = kfac.sum(dim=-1).sum(dim=-1).sum(dim=-1) / (2.0 * volume)
                corr[ff, 0] -= torch.sum(q**2) * diag_sum

        return corr

    def _apply_frame_correction_lower(
        self,
        model_ret: dict[str, torch.Tensor],
        extended_coord: torch.Tensor,
        nlist: torch.Tensor,
        box: Optional[torch.Tensor],
        do_atomic_virial: bool,
    ) -> dict[str, torch.Tensor]:
        if box is None or "latent_charge" not in model_ret:
            return model_ret

        nf, nloc, _ = nlist.shape
        nall = extended_coord.shape[1]
        coord_local = extended_coord[:, :nloc, :]
        box_local = box.view(nf, 3, 3)
        latent_charge = model_ret["latent_charge"]
        corr_redu = self._compute_les_frame_correction(coord_local, latent_charge, box_local)

        model_ret["energy_redu"] = model_ret["energy_redu"] + corr_redu.to(model_ret["energy_redu"].dtype)

        if self.do_grad_r("energy") or self.do_grad_c("energy"):
            corr_force_ext = -torch.autograd.grad(
                corr_redu.sum(),
                extended_coord,
                create_graph=self.training,
                retain_graph=True,
            )[0].view(nf, nall, 3)
            if "energy_derv_r" in model_ret:
                model_ret["energy_derv_r"] = model_ret["energy_derv_r"] + corr_force_ext.unsqueeze(-2).to(
                    model_ret["energy_derv_r"].dtype
                )

            if self.do_grad_c("energy"):
                corr_virial_ext = torch.einsum(
                    "fai,faj->faij",
                    corr_force_ext,
                    extended_coord,
                ).reshape(nf, nall, 1, 9)
                corr_virial_redu = corr_virial_ext.sum(dim=1)
                if "energy_derv_c_redu" in model_ret:
                    model_ret["energy_derv_c_redu"] = model_ret["energy_derv_c_redu"] + corr_virial_redu.to(
                        model_ret["energy_derv_c_redu"].dtype
                    )
                if do_atomic_virial and "energy_derv_c" in model_ret:
                    corr_atom_virial = torch.zeros(
                        (nf, nall, 1, 9),
                        dtype=corr_virial_ext.dtype,
                        device=corr_virial_ext.device,
                    )
                    corr_atom_virial[:, :nloc, :, :] = corr_virial_ext[:, :nloc, :, :]
                    model_ret["energy_derv_c"] = model_ret["energy_derv_c"] + corr_atom_virial.to(
                        model_ret["energy_derv_c"].dtype
                    )

        return model_ret

    @torch.jit.export
    def forward_common_lower(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
        comm_dict: Optional[dict[str, torch.Tensor]] = None,
        extra_nlist_sort: bool = False,
        extended_coord_corr: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if self.do_grad_r("energy") or self.do_grad_c("energy"):
            extended_coord = extended_coord.requires_grad_(True)
        model_ret = super().forward_common_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            comm_dict=comm_dict,
            extra_nlist_sort=extra_nlist_sort,
            extended_coord_corr=extended_coord_corr,
        )
        box = None
        if comm_dict is not None and "box" in comm_dict:
            box = comm_dict["box"]
        return self._apply_frame_correction_lower(
            model_ret,
            extended_coord,
            nlist,
            box,
            do_atomic_virial,
        )

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, torch.Tensor]:
        cc, bb, fp, ap, input_prec = self._input_type_cast(
            coord, box=box, fparam=fparam, aparam=aparam
        )
        (
            extended_coord,
            extended_atype,
            mapping,
            nlist,
        ) = extend_input_and_build_neighbor_list(
            cc,
            atype,
            self.get_rcut(),
            self.get_sel(),
            mixed_types=True,
            box=bb,
        )
        comm_dict: dict[str, torch.Tensor] | None = None
        if bb is not None:
            comm_dict = {"box": bb}
        model_predict_lower = self.forward_common_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam=fp,
            aparam=ap,
            do_atomic_virial=do_atomic_virial,
            comm_dict=comm_dict,
        )
        model_ret = communicate_extended_output(
            model_predict_lower,
            self.model_output_def(),
            mapping,
            do_atomic_virial=do_atomic_virial,
        )
        model_ret = self._output_type_cast(model_ret, input_prec)
        if self.get_fitting_net() is not None:
            model_predict = {}
            model_predict["atom_energy"] = model_ret["energy"]
            model_predict["energy"] = model_ret["energy_redu"]
            if self.do_grad_r("energy"):
                model_predict["force"] = model_ret["energy_derv_r"].squeeze(-2)
            if self.do_grad_c("energy"):
                model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
                if do_atomic_virial:
                    model_predict["atom_virial"] = model_ret["energy_derv_c"].squeeze(
                        -3
                    )
            else:
                model_predict["force"] = model_ret["dforce"]
            if "mask" in model_ret:
                model_predict["mask"] = model_ret["mask"]
            if self._hessian_enabled:
                model_predict["hessian"] = model_ret["energy_derv_r_derv_r"].squeeze(-2)
        else:
            model_predict = model_ret
            model_predict["updated_coord"] += coord
        return model_predict

    @torch.jit.export
    def forward_lower(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
        comm_dict: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        model_ret = self.forward_common_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            comm_dict=comm_dict,
            extra_nlist_sort=self.need_sorted_nlist_for_lower(),
        )
        if self.get_fitting_net() is not None:
            model_predict = {}
            model_predict["atom_energy"] = model_ret["energy"]
            model_predict["energy"] = model_ret["energy_redu"]
            if self.do_grad_r("energy"):
                model_predict["extended_force"] = model_ret["energy_derv_r"].squeeze(-2)
            if self.do_grad_c("energy"):
                model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
                if do_atomic_virial:
                    model_predict["extended_virial"] = model_ret[
                        "energy_derv_c"
                    ].squeeze(-3)
            else:
                assert model_ret["dforce"] is not None
                model_predict["dforce"] = model_ret["dforce"]
        else:
            model_predict = model_ret
        return model_predict
