#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from deepmd.pt.model.model import get_model
from profile_sog_timing import profile


CFG_PATH = Path("examples/water/sog/input_torch.json")


def run(tag: str, model_cfg: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_cfg).to(device)
    model.eval()
    t = profile(
        model,
        nframes=1,
        nloc=192,
        box_len=20.0,
        repeats=6,
        warmup=2,
        do_atomic_virial=False,
        dtype=torch.float32,
    )
    total = (
        t["input_cast"]
        + t["build_nlist"]
        + t["lower_super"]
        + t["lower_frame_corr"]
        + t["communicate_output"]
        + t["output_cast"]
    ) * 1000.0
    print(
        f"{tag}: total={total:.3f}ms, "
        f"lower_super={t['lower_super']*1000.0:.3f}ms, "
        f"lower_frame_corr={t['lower_frame_corr']*1000.0:.3f}ms, "
        f"nufft1={t.get('nufft_type1', 0.0)*1000.0:.3f}ms, "
        f"nufft2={t.get('nufft_type2', 0.0)*1000.0:.3f}ms"
    )


def main() -> None:
    cfg = json.loads(CFG_PATH.read_text())
    base_model_cfg = cfg["model"]

    run("baseline", copy.deepcopy(base_model_cfg))

    cfg_lr1 = copy.deepcopy(base_model_cfg)
    cfg_lr1["fitting_net"]["dim_out_lr"] = 1
    run("dim_out_lr=1", cfg_lr1)

    cfg_small = copy.deepcopy(base_model_cfg)
    cfg_small["fitting_net"]["neuron_sr"] = [128, 128, 128]
    cfg_small["fitting_net"]["neuron_lr"] = [128, 128, 128]
    run("neurons=128", cfg_small)

    cfg_both = copy.deepcopy(cfg_lr1)
    cfg_both["fitting_net"]["neuron_sr"] = [128, 128, 128]
    cfg_both["fitting_net"]["neuron_lr"] = [128, 128, 128]
    run("dim_out_lr=1 + neurons=128", cfg_both)


if __name__ == "__main__":
    main()
