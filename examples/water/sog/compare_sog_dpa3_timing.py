#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from deepmd.pt.model.model import get_model


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
    coord = torch.rand((nf, nloc, 3), device=dev, dtype=dtype) * box_len
    atype = torch.zeros((nf, nloc), device=dev, dtype=torch.long)
    atype[:, 1::3] = 1
    atype[:, 2::3] = 1
    box = torch.zeros((nf, 3, 3), device=dev, dtype=dtype)
    box[:, 0, 0] = box_len
    box[:, 1, 1] = box_len
    box[:, 2, 2] = box_len
    return coord, atype, box, dev


def bench_model(
    model,
    coord: torch.Tensor,
    atype: torch.Tensor,
    box: torch.Tensor,
    warmup: int = 5,
    reps: int = 30,
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


def main() -> None:
    sog_cfg = json.loads(Path("examples/water/sog/input_torch.json").read_text())["model"]
    dpa_cfg = json.loads(Path("examples/water/dpa3/input_torch_copy.json").read_text())["model"]

    coord, atype, box, dev = build_input()

    sog = get_model(sog_cfg).to(dev)
    dpa = get_model(dpa_cfg).to(dev)
    if hasattr(sog.get_fitting_net(), "n_dl"):
        sog.get_fitting_net().n_dl = 2

    sog_ms = bench_model(sog, coord, atype, box)

    orig = sog._apply_frame_correction_lower
    sog._apply_frame_correction_lower = lambda model_ret, *args, **kwargs: model_ret
    sog_nocorr_ms = bench_model(sog, coord, atype, box)
    sog._apply_frame_correction_lower = orig

    dpa_ms = bench_model(dpa, coord, atype, box)

    print(f"dpa3_total_ms={dpa_ms:.3f}")
    print(f"sog_total_ms={sog_ms:.3f}")
    print(f"sog_without_frame_corr_ms={sog_nocorr_ms:.3f}")
    print(f"sog_extra_frame_corr_ms={sog_ms - sog_nocorr_ms:.3f}")
    print(f"sog_vs_dpa3_delta_ms={sog_ms - dpa_ms:.3f}")


if __name__ == "__main__":
    main()
