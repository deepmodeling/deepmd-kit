#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Profile SOG model runtime breakdown using input_torch.json as model config."""

from __future__ import (
    annotations,
)

import argparse
import json
import time
import types
from collections import (
    defaultdict,
)
from contextlib import (
    nullcontext,
)
from pathlib import (
    Path,
)
from typing import (
    Any,
)

import pytorch_finufft
import torch

from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.model.model.sog_model import (
    SOGEnergyModel_,
)
from deepmd.pt.model.model.transform_output import (
    communicate_extended_output,
)
from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
)


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_block(name: str, timings: dict[str, float], device: torch.device):
    class _Timer:
        def __enter__(self_inner):
            _sync_if_cuda(device)
            self_inner.t0 = time.perf_counter()
            return self_inner

        def __exit__(self_inner, exc_type, exc, tb):
            _sync_if_cuda(device)
            timings[name] += time.perf_counter() - self_inner.t0

    return _Timer()


def _build_synthetic_input(
    nframes: int,
    nloc: int,
    box_len: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coord = torch.rand((nframes, nloc, 3), device=device, dtype=dtype) * box_len
    atype = torch.zeros((nframes, nloc), device=device, dtype=torch.long)

    # Water-like type ratio: O:H = 1:2
    atype[:, 1::3] = 1
    atype[:, 2::3] = 1

    box = torch.zeros((nframes, 3, 3), device=device, dtype=dtype)
    box[:, 0, 0] = box_len
    box[:, 1, 1] = box_len
    box[:, 2, 2] = box_len
    return coord, atype, box


def _load_model(config_path: Path, device: torch.device) -> Any:
    cfg = json.loads(config_path.read_text())
    model = get_model(cfg["model"])
    model = model.to(device)
    model.eval()
    return model


def _install_fine_frame_corr_profiler(
    model: Any,
    detail_times: dict[str, float],
    device: torch.device,
    collect_flag: dict[str, bool],
) -> tuple[Any, Any]:
    orig_bundle = model._compute_sog_frame_correction_bundle
    orig_apply = model._apply_frame_correction_lower

    def _timed_bundle(
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
        with (
            _time_block("fc_cast_inputs", detail_times, device)
            if collect_flag["on"]
            else nullcontext()
        ):
            latent_charge = latent_charge.to(device=runtime_device, dtype=real_dtype)
            box = box.to(device=runtime_device, dtype=real_dtype)
        if box.dim() != 3 or box.shape[-2:] != (3, 3):
            raise ValueError(
                f"`box` should be [nf, 3, 3], got shape {tuple(box.shape)}"
            )

        with (
            _time_block("fc_param_prepare", detail_times, device)
            if collect_flag["on"]
            else nullcontext()
        ):
            wl, _sl, min_term = self._get_cached_sog_params(
                fitting,
                runtime_device,
                real_dtype,
            )
            remove_self_interaction = bool(fitting.remove_self_interaction)
            n_dl = int(fitting.n_dl)
            pi_tensor = torch.tensor(torch.pi, dtype=real_dtype, device=runtime_device)
            two_pi = torch.tensor(
                2.0 * torch.pi, dtype=real_dtype, device=runtime_device
            )

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

            with (
                _time_block("fc_geom_and_points", detail_times, device)
                if collect_flag["on"]
                else nullcontext()
            ):
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

            with (
                _time_block("fc_build_k_grid", detail_times, device)
                if collect_flag["on"]
                else nullcontext()
            ):
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

            with (
                _time_block("fc_build_kfac", detail_times, device)
                if collect_flag["on"]
                else nullcontext()
            ):
                kfac = wl.view(1, 1, 1, -1) * torch.exp(k_sq.unsqueeze(-1) * min_term)
                kfac = kfac.sum(dim=-1)
                kfac = kfac.to(dtype=real_dtype)
                kfac[zero_mask] = 0.0

            with (
                _time_block("fc_prepare_charge", detail_times, device)
                if collect_flag["on"]
                else nullcontext()
            ):
                q_t = q.transpose(0, 1).contiguous()
                charge = (
                    torch.complex(q_t, torch.zeros_like(q_t))
                    .to(dtype=complex_dtype)
                    .contiguous()
                )

            with (
                _time_block("fc_nufft_type1", detail_times, device)
                if collect_flag["on"]
                else nullcontext()
            ):
                recon = pytorch_finufft.functional.finufft_type1(
                    nufft_points,
                    charge,
                    output_shape=tuple(int(x) for x in kx_grid.shape),
                    eps=1e-4,
                    isign=-1,
                )

            with (
                _time_block("fc_energy_reduce", detail_times, device)
                if collect_flag["on"]
                else nullcontext()
            ):
                rho_sq = recon.real.square() + recon.imag.square()
                corr[ff, 0] = (kfac.unsqueeze(0) * rho_sq).sum() / (2.0 * volume)

            if need_force:
                with (
                    _time_block("fc_prepare_force_conv", detail_times, device)
                    if collect_flag["on"]
                    else nullcontext()
                ):
                    conv = kfac.unsqueeze(0).to(dtype=complex_dtype) * recon

                with (
                    _time_block("fc_prepare_force_kgrid", detail_times, device)
                    if collect_flag["on"]
                    else nullcontext()
                ):
                    kk1 = torch.fft.ifftshift(kx_grid, dim=0)
                    kk2 = torch.fft.ifftshift(ky_grid, dim=1)
                    kk3 = torch.fft.ifftshift(kz_grid, dim=2)
                    k_grid = torch.stack((kk1, kk2, kk3), dim=0)
                    g_cart = two_pi * torch.einsum("ik,k...->i...", cell_inv, k_grid)
                    grad_conv = (
                        1j * g_cart.unsqueeze(1).to(dtype=complex_dtype)
                    ) * conv.unsqueeze(0)

                with (
                    _time_block("fc_nufft_type2_force", detail_times, device)
                    if collect_flag["on"]
                    else nullcontext()
                ):
                    grad_field = pytorch_finufft.functional.finufft_type2(
                        nufft_points,
                        grad_conv,
                        eps=1e-4,
                        isign=1,
                    )

                with (
                    _time_block("fc_force_reduce", detail_times, device)
                    if collect_flag["on"]
                    else nullcontext()
                ):
                    force_frame = (
                        -(q_t.unsqueeze(0) * grad_field.real.to(dtype=real_dtype))
                        .sum(dim=1)
                        .transpose(0, 1)
                    )
                    force_frame = force_frame / volume
                    force_local[ff] = force_frame

                if need_virial:
                    with (
                        _time_block("fc_virial_local", detail_times, device)
                        if collect_flag["on"]
                        else nullcontext()
                    ):
                        virial_local[ff] = torch.einsum(
                            "ai,aj->aij",
                            force_frame,
                            r_raw,
                        ).reshape(nloc, 1, 9)

            if remove_self_interaction:
                with (
                    _time_block("fc_self_interaction", detail_times, device)
                    if collect_flag["on"]
                    else nullcontext()
                ):
                    diag_sum = kfac.sum(dim=-1).sum(dim=-1).sum(dim=-1) / (2.0 * volume)
                    corr[ff, 0] -= torch.sum(q**2) * diag_sum

        out: dict[str, torch.Tensor] = {"corr_redu": corr}
        if force_local is not None:
            out["force_local"] = force_local
        if virial_local is not None:
            out["virial_local"] = virial_local
        return out

    def _timed_apply(
        self,
        model_ret: dict[str, torch.Tensor],
        extended_coord: torch.Tensor,
        nlist: torch.Tensor,
        box: torch.Tensor | None,
        do_atomic_virial: bool,
    ) -> dict[str, torch.Tensor]:
        with (
            _time_block("fc_guard_and_slice", detail_times, device)
            if collect_flag["on"]
            else nullcontext()
        ):
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

        with (
            _time_block("fc_compute_corr_bundle", detail_times, device)
            if collect_flag["on"]
            else nullcontext()
        ):
            corr_bundle = self._compute_sog_frame_correction_bundle(
                coord_local,
                latent_charge_runtime,
                box_local,
                need_force=need_force,
                need_virial=need_virial,
            )
            corr_redu = corr_bundle["corr_redu"]

        with (
            _time_block("fc_add_energy", detail_times, device)
            if collect_flag["on"]
            else nullcontext()
        ):
            model_ret["energy_redu"] = model_ret["energy_redu"] + corr_redu.to(
                model_ret["energy_redu"].dtype
            )

        if need_force:
            corr_force_local = corr_bundle["force_local"].to(coord_local.dtype)

            with (
                _time_block("fc_scatter_force", detail_times, device)
                if collect_flag["on"]
                else nullcontext()
            ):
                corr_force_ext = torch.zeros(
                    (nf, nall, 3),
                    dtype=corr_force_local.dtype,
                    device=corr_force_local.device,
                )
                corr_force_ext[:, :nloc, :] = corr_force_local
                if "energy_derv_r" in model_ret:
                    model_ret["energy_derv_r"] = model_ret[
                        "energy_derv_r"
                    ] + corr_force_ext.unsqueeze(-2).to(
                        model_ret["energy_derv_r"].dtype
                    )

            if need_virial:
                corr_virial_local = corr_bundle["virial_local"].to(
                    corr_force_local.dtype
                )
                with (
                    _time_block("fc_virial_update", detail_times, device)
                    if collect_flag["on"]
                    else nullcontext()
                ):
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

    model._compute_sog_frame_correction_bundle = types.MethodType(_timed_bundle, model)
    model._apply_frame_correction_lower = types.MethodType(_timed_apply, model)
    return orig_bundle, orig_apply


def profile(
    model: Any,
    nframes: int,
    nloc: int,
    box_len: float,
    repeats: int,
    warmup: int,
    do_atomic_virial: bool,
    dtype: torch.dtype,
    fine_frame_profile: bool = False,
) -> dict[str, float]:
    device = next(model.parameters()).device

    # NUFFT fine-grained timers by monkeypatching function calls.
    nufft_times: dict[str, float] = defaultdict(float)
    collect_nufft = False
    orig_type1 = pytorch_finufft.functional.finufft_type1
    orig_type2 = pytorch_finufft.functional.finufft_type2

    def timed_type1(*args, **kwargs):
        if collect_nufft:
            with _time_block("nufft_type1", nufft_times, device):
                return orig_type1(*args, **kwargs)
        return orig_type1(*args, **kwargs)

    def timed_type2(*args, **kwargs):
        if collect_nufft:
            with _time_block("nufft_type2", nufft_times, device):
                return orig_type2(*args, **kwargs)
        return orig_type2(*args, **kwargs)

    pytorch_finufft.functional.finufft_type1 = timed_type1
    pytorch_finufft.functional.finufft_type2 = timed_type2

    timings: dict[str, float] = defaultdict(float)
    detail_times: dict[str, float] = defaultdict(float)
    collect_detail = {"on": False}
    orig_bundle = None
    orig_apply = None

    if fine_frame_profile:
        orig_bundle, orig_apply = _install_fine_frame_corr_profiler(
            model,
            detail_times,
            device,
            collect_detail,
        )

    try:
        for _ in range(warmup + repeats):
            coord, atype, box = _build_synthetic_input(
                nframes=nframes,
                nloc=nloc,
                box_len=box_len,
                device=device,
                dtype=dtype,
            )

            is_warmup = _ < warmup
            collect_nufft = not is_warmup
            collect_detail["on"] = not is_warmup
            iter_times: dict[str, float] = defaultdict(float)

            with _time_block("input_cast", iter_times, device):
                cc, bb, fp, ap, input_prec = model._input_type_cast(
                    coord, box=box, fparam=None, aparam=None
                )

            with _time_block("build_nlist", iter_times, device):
                (
                    extended_coord,
                    extended_atype,
                    mapping,
                    nlist,
                ) = extend_input_and_build_neighbor_list(
                    cc,
                    atype,
                    model.get_rcut(),
                    model.get_sel(),
                    mixed_types=True,
                    box=bb,
                )

            comm_dict: dict[str, torch.Tensor] | None = {"box": bb}

            if model.do_grad_r("energy") or model.do_grad_c("energy"):
                extended_coord = extended_coord.requires_grad_(True)

            with _time_block("lower_super", iter_times, device):
                model_ret = SOGEnergyModel_.forward_common_lower(
                    model,
                    extended_coord,
                    extended_atype,
                    nlist,
                    mapping,
                    fparam=fp,
                    aparam=ap,
                    do_atomic_virial=do_atomic_virial,
                    comm_dict=comm_dict,
                    extra_nlist_sort=False,
                    extended_coord_corr=None,
                )

            with _time_block("lower_frame_corr", iter_times, device):
                model_ret = model._apply_frame_correction_lower(
                    model_ret,
                    extended_coord,
                    nlist,
                    bb,
                    do_atomic_virial,
                )

            with _time_block("communicate_output", iter_times, device):
                model_ret = communicate_extended_output(
                    model_ret,
                    model.model_output_def(),
                    mapping,
                    do_atomic_virial=do_atomic_virial,
                )

            with _time_block("output_cast", iter_times, device):
                model_ret = model._output_type_cast(model_ret, input_prec)
                _ = model_ret["energy_redu"]

            if not is_warmup:
                iter_times["total"] = sum(
                    iter_times[k]
                    for k in [
                        "input_cast",
                        "build_nlist",
                        "lower_super",
                        "lower_frame_corr",
                        "communicate_output",
                        "output_cast",
                    ]
                )
                for k, v in iter_times.items():
                    timings[k] += v

        # Only keep averaged timings for measured iterations.
        for k in list(timings.keys()):
            timings[k] /= repeats
        for k, v in nufft_times.items():
            timings[k] = v / repeats
        for k, v in detail_times.items():
            timings[k] = v / repeats

        return dict(timings)
    finally:
        pytorch_finufft.functional.finufft_type1 = orig_type1
        pytorch_finufft.functional.finufft_type2 = orig_type2
        if fine_frame_profile and orig_bundle is not None and orig_apply is not None:
            model._compute_sog_frame_correction_bundle = orig_bundle
            model._apply_frame_correction_lower = orig_apply


def _format_report(timings: dict[str, float]) -> str:
    total = sum(
        timings.get(k, 0.0)
        for k in [
            "input_cast",
            "build_nlist",
            "lower_super",
            "lower_frame_corr",
            "communicate_output",
            "output_cast",
        ]
    )

    keys = [
        "input_cast",
        "build_nlist",
        "lower_super",
        "lower_frame_corr",
        "nufft_type1",
        "nufft_type2",
        "communicate_output",
        "output_cast",
        "fc_guard_and_slice",
        "fc_compute_corr_bundle",
        "fc_cast_inputs",
        "fc_param_prepare",
        "fc_geom_and_points",
        "fc_build_k_grid",
        "fc_build_kfac",
        "fc_prepare_charge",
        "fc_nufft_type1",
        "fc_energy_reduce",
        "fc_prepare_force_conv",
        "fc_prepare_force_kgrid",
        "fc_nufft_type2_force",
        "fc_force_reduce",
        "fc_virial_local",
        "fc_self_interaction",
        "fc_add_energy",
        "fc_scatter_force",
        "fc_virial_update",
    ]

    lines = []
    lines.append("Timing breakdown (avg per iteration):")
    for k in keys:
        if k in timings:
            ms = timings[k] * 1000.0
            ratio = (timings[k] / total * 100.0) if total > 0 else 0.0
            lines.append(f"  - {k:20s}: {ms:10.3f} ms  ({ratio:6.2f}%)")

    lines.append(f"  - {'total(sum)':20s}: {total * 1000.0:10.3f} ms  (100.00%)")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/water/sog/input_torch.json"),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dtype", type=str, default="float32", choices=["float32", "float64"]
    )
    parser.add_argument("--nframes", type=int, default=1)
    parser.add_argument("--nloc", type=int, default=192)
    parser.add_argument("--box-len", type=float, default=20.0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--atomic-virial", action="store_true")
    parser.add_argument("--n-dl-override", type=int, default=2)
    parser.add_argument("--disable-energy-grad", action="store_true")
    parser.add_argument("--fine-frame-profile", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    model = _load_model(args.config, device)
    if args.n_dl_override > 0 and hasattr(model.get_fitting_net(), "n_dl"):
        model.get_fitting_net().n_dl = int(args.n_dl_override)
    if args.disable_energy_grad:
        model.do_grad_r = types.MethodType(lambda self, _name: False, model)
        model.do_grad_c = types.MethodType(lambda self, _name: False, model)
    timings = profile(
        model=model,
        nframes=args.nframes,
        nloc=args.nloc,
        box_len=args.box_len,
        repeats=args.repeats,
        warmup=args.warmup,
        do_atomic_virial=args.atomic_virial,
        dtype=dtype,
        fine_frame_profile=args.fine_frame_profile,
    )
    print(_format_report(timings))


if __name__ == "__main__":
    main()
