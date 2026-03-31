#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import pathlib

import torch

from deepmd.pt.model.model import get_model


def main() -> None:
    cfg = json.loads(pathlib.Path("examples/water/sog/input_torch.json").read_text())["model"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg).to(dev).eval()
    fit = model.get_fitting_net()

    sog_path = pathlib.Path("/data/zyjin/cace/SOG-Net/CACE-SOG/cace/modules/sog.py")
    spec = importlib.util.spec_from_file_location("cace_sog_module", sog_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load cace sog.py module")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    SOGPotential = mod.SOGPotential

    cace_sog = SOGPotential(N_dl=int(fit.n_dl), Periodic=True).to(dev).eval()
    with torch.no_grad():
        cace_sog.wl.copy_(fit.wl.detach().to(cace_sog.wl.dtype).to(dev))
        cace_sog.sl.copy_(fit.sl.detach().to(cace_sog.sl.dtype).to(dev))

    nf, nloc, nq = 3, 32, 1
    coord = torch.rand(nf, nloc, 3, device=dev, dtype=torch.float32) * 10.0
    box = torch.zeros(nf, 3, 3, device=dev, dtype=torch.float32)
    box[:, 0, 0] = 10.0
    box[:, 1, 1] = 11.0
    box[:, 2, 2] = 12.0
    latent = torch.randn(nf, nloc, nq, device=dev, dtype=torch.float32)

    with torch.no_grad():
        corr_deepmd = model._compute_sog_frame_correction(coord, latent, box).squeeze(-1)
        corr_cace = []
        for i in range(nf):
            v = cace_sog.compute_potential_SOG_triclinic_NUFFT(coord[i], latent[i], box[i])
            corr_cace.append(v.squeeze())
        corr_cace = torch.stack(corr_cace)

    abs_diff = (corr_deepmd - corr_cace).abs()
    rel_diff = abs_diff / torch.clamp(corr_cace.abs(), min=1e-8)

    print("corr_deepmd", corr_deepmd.detach().cpu().numpy())
    print("corr_cace  ", corr_cace.detach().cpu().numpy())
    print("max_abs_diff", abs_diff.max().item())
    print("mean_abs_diff", abs_diff.mean().item())
    print("max_rel_diff", rel_diff.max().item())


if __name__ == "__main__":
    main()
