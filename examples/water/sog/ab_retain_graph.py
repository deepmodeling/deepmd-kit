#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import (
    annotations,
)

import json
import time
from pathlib import (
    Path,
)
from types import (
    MethodType,
)

import torch

from deepmd.pt.model.model import (
    get_model,
)


def sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def build_input(
    nf: int = 1,
    nloc: int = 192,
    box_len: float = 20.0,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.device]:
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    g = torch.Generator(device=dev)
    g.manual_seed(1234)
    coord = torch.rand((nf, nloc, 3), device=dev, dtype=dtype, generator=g) * box_len
    atype = torch.zeros((nf, nloc), device=dev, dtype=torch.long)
    atype[:, 1::3] = 1
    atype[:, 2::3] = 1
    box = torch.zeros((nf, 3, 3), device=dev, dtype=dtype)
    box[:, 0, 0] = box_len
    box[:, 1, 1] = box_len
    box[:, 2, 2] = box_len
    return coord, atype, box, dev


def bench(
    model,
    coord: torch.Tensor,
    atype: torch.Tensor,
    box: torch.Tensor,
    reps: int = 20,
    warmup: int = 5,
) -> float:
    model.eval()
    for _ in range(warmup):
        _ = model(coord, atype, box=box)
    sync(coord.device)
    t0 = time.perf_counter()
    for _ in range(reps):
        _ = model(coord, atype, box=box)
    sync(coord.device)
    return (time.perf_counter() - t0) / reps * 1000.0


def make_patched_apply():
    def patched_apply(self, model_ret, extended_coord, nlist, box, do_atomic_virial):
        if box is None or "latent_charge" not in model_ret:
            return model_ret
        nf, nloc, _ = nlist.shape
        nall = extended_coord.shape[1]
        coord_local = extended_coord[:, :nloc, :]
        box_local = box.view(nf, 3, 3)
        latent_charge = model_ret["latent_charge"]
        corr_redu = self._compute_sog_frame_correction(
            coord_local, latent_charge, box_local
        )
        model_ret["energy_redu"] = model_ret["energy_redu"] + corr_redu.to(
            model_ret["energy_redu"].dtype
        )

        if self.do_grad_r("energy") or self.do_grad_c("energy"):
            corr_force_local = -torch.autograd.grad(
                corr_redu.sum(),
                coord_local,
                create_graph=self.training,
                retain_graph=False,
            )[0].view(nf, nloc, 3)

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

            if self.do_grad_c("energy"):
                corr_virial_local = torch.einsum(
                    "fai,faj->faij",
                    corr_force_local,
                    coord_local,
                ).reshape(nf, nloc, 1, 9)
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

    return patched_apply


def main() -> None:
    cfg = json.loads(Path("examples/water/sog/input_torch.json").read_text())["model"]
    coord, atype, box, dev = build_input()
    model = get_model(cfg).to(dev)

    base = bench(model, coord, atype, box)

    orig_apply = model._apply_frame_correction_lower
    model._apply_frame_correction_lower = MethodType(make_patched_apply(), model)
    patched = bench(model, coord, atype, box)

    model._apply_frame_correction_lower = orig_apply
    out0 = model(coord, atype, box=box)
    model._apply_frame_correction_lower = MethodType(make_patched_apply(), model)
    out1 = model(coord, atype, box=box)

    max_de = (out0["energy"] - out1["energy"]).abs().max().item()
    max_df = (out0["force"] - out1["force"]).abs().max().item()

    print(f"baseline_ms={base:.3f}")
    print(f"retain_graph_false_ms={patched:.3f}")
    print(f"speedup={(base / patched):.3f}x")
    print(f"max|dE|={max_de:.3e}")
    print(f"max|dF|={max_df:.3e}")


if __name__ == "__main__":
    main()
