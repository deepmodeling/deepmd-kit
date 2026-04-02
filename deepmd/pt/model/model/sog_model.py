# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import pytorch_finufft
import torch

from deepmd.pt.model.atomic_model import (
    SOGEnergyAtomicModel,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)
from deepmd.pt.model.model.transform_output import (
    communicate_extended_output,
)
from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
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

SOGEnergyModel_ = make_model(SOGEnergyAtomicModel)


@BaseModel.register("sog_ener")
class SOGEnergyModel(DPModelCommon, SOGEnergyModel_):
    model_type = "sog_ener"

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        DPModelCommon.__init__(self)
        SOGEnergyModel_.__init__(self, *args, **kwargs)
        self._hessian_enabled = False
        # Runtime-only caches for NUFFT correction path.
        self._sog_param_cache: dict[
            tuple[Any, ...], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}

    @staticmethod
    def _device_key(device: torch.device) -> str:
        if device.index is None:
            return device.type
        return f"{device.type}:{device.index}"

    @staticmethod
    def _trim_cache(cache: dict[Any, Any], max_size: int = 8) -> None:
        if len(cache) > max_size:
            oldest_key = next(iter(cache.keys()))
            cache.pop(oldest_key, None)

    def _get_cached_sog_params(
        self,
        fitting: Any,
        runtime_device: torch.device,
        real_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wl_raw = fitting.wl
        sl_raw = fitting.sl
        grad_mode = torch.is_grad_enabled() and (
            wl_raw.requires_grad or sl_raw.requires_grad
        )

        wl = (
            wl_raw
            if (wl_raw.device == runtime_device and wl_raw.dtype == real_dtype)
            else wl_raw.to(dtype=real_dtype, device=runtime_device)
        )
        sl = (
            sl_raw
            if (sl_raw.device == runtime_device and sl_raw.dtype == real_dtype)
            else sl_raw.to(dtype=real_dtype, device=runtime_device)
        )
        min_term = -1.0 / torch.exp(-2.0 * sl)

        # Do not cache differentiable tensors across iterations.
        if grad_mode:
            return wl, sl, min_term

        wl_version = int(getattr(fitting.wl, "_version", 0))
        sl_version = int(getattr(fitting.sl, "_version", 0))
        cache_key = (
            self._device_key(runtime_device),
            str(real_dtype),
            wl_version,
            sl_version,
        )
        cached = self._sog_param_cache.get(cache_key)
        if cached is not None:
            return cached

        self._sog_param_cache[cache_key] = (wl, sl, min_term)
        self._trim_cache(self._sog_param_cache)
        return wl, sl, min_term

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

    def _compute_sog_frame_correction_bundle(
        self,
        coord: torch.Tensor,
        latent_charge: torch.Tensor,
        box: torch.Tensor,
        *,
        need_force: bool,
        need_virial: bool,
    ) -> dict[str, torch.Tensor]:
        if coord.dim() != 3:
            raise ValueError(
                f"`coord` should be [nf, nloc, 3], got shape {tuple(coord.shape)}"
            )
        if latent_charge.dim() != 3:
            raise ValueError(
                f"`latent_charge` should be [nf, nloc, nq], got shape {tuple(latent_charge.shape)}"
            )
        if coord.shape[:2] != latent_charge.shape[:2]:
            raise ValueError(
                "`coord` and `latent_charge` local dimensions mismatch: "
                f"{tuple(coord.shape[:2])} vs {tuple(latent_charge.shape[:2])}"
            )

        fitting = self.get_fitting_net()
        runtime_device = coord.device
        real_dtype = coord.dtype
        complex_dtype = (
            torch.complex128 if real_dtype == torch.float64 else torch.complex64
        )
        latent_charge = latent_charge.to(device=runtime_device, dtype=real_dtype)
        box = box.to(device=runtime_device, dtype=real_dtype)
        if box.dim() != 3 or box.shape[-2:] != (3, 3):
            raise ValueError(
                f"`box` should be [nf, 3, 3], got shape {tuple(box.shape)}"
            )

        wl, _sl, min_term = self._get_cached_sog_params(
            fitting,
            runtime_device,
            real_dtype,
        )
        remove_self_interaction = bool(fitting.remove_self_interaction)
        n_dl = int(fitting.n_dl)
        pi_tensor = torch.tensor(torch.pi, dtype=real_dtype, device=runtime_device)
        two_pi = torch.tensor(2.0 * torch.pi, dtype=real_dtype, device=runtime_device)

        nf, nloc, _ = coord.shape
        corr = torch.zeros((nf, 1), dtype=real_dtype, device=runtime_device)
        force_local = (
            torch.zeros((nf, nloc, 3), dtype=real_dtype, device=runtime_device)
            if need_force
            else None
        )
        virial_local = (
            torch.zeros((nf, nloc, 1, 9), dtype=real_dtype, device=runtime_device)
            if need_virial
            else None
        )

        for ff in range(nf):
            r_raw = coord[ff]
            q = latent_charge[ff]
            box_frame = box[ff]

            volume = torch.det(box_frame)
            if torch.abs(volume) <= torch.finfo(real_dtype).eps:
                raise ValueError(
                    "`box` is singular (near-zero volume), cannot run NUFFT."
                )

            cell_inv = torch.linalg.inv(box_frame)
            r_frac = torch.matmul(r_raw, cell_inv)
            r_frac = torch.remainder(r_frac + 0.5, 1.0) - 0.5
            point_limit = pi_tensor - 32.0 * torch.finfo(real_dtype).eps
            r_in = torch.clamp(
                2.0 * pi_tensor * r_frac,
                min=-point_limit,
                max=point_limit,
            ).contiguous()
            nufft_points = r_in.transpose(0, 1).contiguous()

            norms = torch.norm(box_frame, dim=1)
            nk = tuple(max(1, int(n.item() / n_dl)) for n in norms)
            n1 = torch.arange(
                -nk[0], nk[0] + 1, device=runtime_device, dtype=real_dtype
            )
            n2 = torch.arange(
                -nk[1], nk[1] + 1, device=runtime_device, dtype=real_dtype
            )
            n3 = torch.arange(
                -nk[2], nk[2] + 1, device=runtime_device, dtype=real_dtype
            )
            kx_grid, ky_grid, kz_grid = torch.meshgrid(n1, n2, n3, indexing="ij")
            k_sq = kx_grid**2 + ky_grid**2 + kz_grid**2
            zero_mask = k_sq == 0

            kfac = wl.view(1, 1, 1, -1) * torch.exp(k_sq.unsqueeze(-1) * min_term)
            kfac = kfac.sum(dim=-1)
            kfac = kfac.to(dtype=real_dtype)
            kfac[zero_mask] = 0.0
            output_shape = tuple(int(x) for x in kx_grid.shape)

            q_t = q.transpose(0, 1).contiguous()
            charge = (
                torch.complex(q_t, torch.zeros_like(q_t))
                .to(dtype=complex_dtype)
                .contiguous()
            )
            recon = pytorch_finufft.functional.finufft_type1(
                nufft_points,
                charge,
                output_shape=output_shape,
                eps=1e-4,
                isign=-1,
            )

            rho_sq = recon.real.square() + recon.imag.square()
            corr[ff, 0] = (kfac.unsqueeze(0) * rho_sq).sum() / (2.0 * volume)

            conv = None
            if need_force:
                conv = kfac.unsqueeze(0).to(dtype=complex_dtype) * recon

            if need_force:
                assert conv is not None
                # Reuse the already built k-grid and only reorder it to the FFT
                # storage order required by type-2 inputs.
                kk1 = torch.fft.ifftshift(kx_grid, dim=0)
                kk2 = torch.fft.ifftshift(ky_grid, dim=1)
                kk3 = torch.fft.ifftshift(kz_grid, dim=2)
                k_grid = torch.stack((kk1, kk2, kk3), dim=0)
                g_cart = two_pi * torch.einsum("ik,k...->i...", cell_inv, k_grid)
                grad_conv = (
                    1j * g_cart.unsqueeze(1).to(dtype=complex_dtype)
                ) * conv.unsqueeze(0)
                grad_field = pytorch_finufft.functional.finufft_type2(
                    nufft_points,
                    grad_conv,
                    eps=1e-4,
                    isign=1,
                )
                force_frame = (
                    -(q_t.unsqueeze(0) * grad_field.real.to(dtype=real_dtype))
                    .sum(dim=1)
                    .transpose(0, 1)
                )
                force_frame = force_frame / volume
                force_local[ff] = force_frame

                if need_virial:
                    virial_local[ff] = torch.einsum(
                        "ai,aj->aij",
                        force_frame,
                        r_raw,
                    ).reshape(nloc, 1, 9)

            if remove_self_interaction:
                diag_sum = kfac.sum(dim=-1).sum(dim=-1).sum(dim=-1) / (2.0 * volume)
                corr[ff, 0] -= torch.sum(q**2) * diag_sum

        out: dict[str, torch.Tensor] = {"corr_redu": corr}
        if force_local is not None:
            out["force_local"] = force_local
        if virial_local is not None:
            out["virial_local"] = virial_local
        return out

    def _compute_sog_frame_correction(
        self,
        coord: torch.Tensor,
        latent_charge: torch.Tensor,
        box: torch.Tensor,
    ) -> torch.Tensor:
        out = self._compute_sog_frame_correction_bundle(
            coord,
            latent_charge,
            box,
            need_force=False,
            need_virial=False,
        )
        return out["corr_redu"]

    def _apply_frame_correction_lower(
        self,
        model_ret: dict[str, torch.Tensor],
        extended_coord: torch.Tensor,
        nlist: torch.Tensor,
        box: torch.Tensor | None,
        do_atomic_virial: bool,
    ) -> dict[str, torch.Tensor]:
        if box is None or "latent_charge" not in model_ret:
            return model_ret

        nf, nloc, _ = nlist.shape
        nall = extended_coord.shape[1]
        coord_local = extended_coord[:, :nloc, :]
        box_local = box.view(nf, 3, 3)
        latent_charge = model_ret["latent_charge"]
        need_force = self.do_grad_r("energy") or self.do_grad_c("energy")
        need_virial = self.do_grad_c("energy")
        latent_charge_runtime = (
            latent_charge if self.training else latent_charge.detach()
        )
        corr_bundle = self._compute_sog_frame_correction_bundle(
            coord_local,
            latent_charge_runtime,
            box_local,
            need_force=need_force,
            need_virial=need_virial,
        )
        corr_redu = corr_bundle["corr_redu"]

        model_ret["energy_redu"] = model_ret["energy_redu"] + corr_redu.to(
            model_ret["energy_redu"].dtype
        )

        if need_force:
            corr_force_local = corr_bundle["force_local"].to(coord_local.dtype)

            corr_force_ext = torch.zeros(
                (nf, nall, 3),
                dtype=corr_force_local.dtype,
                device=corr_force_local.device,
            )
            corr_force_ext[:, :nloc, :] = corr_force_local
            if "energy_derv_r" in model_ret:
                model_ret["energy_derv_r"] = model_ret[
                    "energy_derv_r"
                ] + corr_force_ext.unsqueeze(-2).to(model_ret["energy_derv_r"].dtype)

            if need_virial:
                corr_virial_local = corr_bundle["virial_local"].to(
                    corr_force_local.dtype
                )
                corr_virial_redu = corr_virial_local.sum(dim=1)
                if "energy_derv_c_redu" in model_ret:
                    model_ret["energy_derv_c_redu"] = model_ret[
                        "energy_derv_c_redu"
                    ] + corr_virial_redu.to(model_ret["energy_derv_c_redu"].dtype)
                if do_atomic_virial and "energy_derv_c" in model_ret:
                    corr_atom_virial = torch.zeros(
                        (nf, nall, 1, 9),
                        dtype=corr_virial_local.dtype,
                        device=corr_virial_local.device,
                    )
                    corr_atom_virial[:, :nloc, :, :] = corr_virial_local
                    model_ret["energy_derv_c"] = model_ret[
                        "energy_derv_c"
                    ] + corr_atom_virial.to(model_ret["energy_derv_c"].dtype)

        return model_ret

    @torch.jit.export
    def forward_common_lower(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        comm_dict: dict[str, torch.Tensor] | None = None,
        extra_nlist_sort: bool = False,
        extended_coord_corr: torch.Tensor | None = None,
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
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
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
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        comm_dict: dict[str, torch.Tensor] | None = None,
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
