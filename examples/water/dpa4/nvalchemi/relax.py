# SPDX-License-Identifier: LGPL-3.0-or-later
"""Geometry optimization (FIRE) with a DPA-4 / SeZM model.

This example relaxes atomic positions toward a local energy minimum using the
FIRE2 optimizer in ``nvalchemi``, driven by a trained DPA-4 / SeZM potential.
The cell is held fixed; only atomic coordinates move.

Convergence is controlled by a force criterion: the optimizer stops once the
maximum per-atom force norm (``fmax``) drops below the threshold. This is the
same criterion ASE's optimizers use, so the threshold is directly comparable.

Usage
-----
::

    python relax.py \
        --model ../lmp/pretrained.pt \
        --data ../../data/data_0 \
        --fmax 0.05 --max-steps 200 --dt 1.0
"""

from __future__ import (
    annotations,
)

import argparse
from pathlib import (
    Path,
)

import numpy as np
import torch
from ase.data import atomic_numbers as ASE_Z
from nvalchemi.data import (
    AtomicData,
    Batch,
)
from nvalchemi.dynamics.base import (
    ConvergenceHook,
    DynamicsStage,
)
from nvalchemi.dynamics.optimizers import (
    FIRE2,
)
from nvalchemi.hooks import (
    NeighborListHook,
)
from nvalchemi.neighbors import (
    compute_neighbors,
)

from deepmd.pt.nvalchemi import (
    DPA4Wrapper,
)


def load_frame(
    data_dir: str | Path,
    frame: int = 0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load one periodic frame from a DeePMD-kit ``npy`` data system."""
    data_dir = Path(data_dir)
    set_dir = sorted(data_dir.glob("set.*"))[0]
    coord = np.load(set_dir / "coord.npy")[frame].reshape(-1, 3)
    box = np.load(set_dir / "box.npy")[frame].reshape(3, 3)
    type_index = np.loadtxt(data_dir / "type.raw", dtype=int).reshape(-1)
    type_map = (data_dir / "type_map.raw").read_text().split()
    z = np.array([ASE_Z[type_map[t]] for t in type_index], dtype=np.int64)
    return (
        torch.tensor(z, dtype=torch.long, device=device),
        torch.tensor(coord, dtype=dtype, device=device),
        torch.tensor(box, dtype=dtype, device=device).reshape(1, 3, 3),
    )


def fmax(batch: Batch) -> float:
    """Maximum per-atom force norm in eV/A."""
    return batch.forces.norm(dim=-1).max().item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="../lmp/pretrained.pt")
    parser.add_argument("--data", default="../../data/data_0")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument(
        "--fmax", type=float, default=0.05, help="force convergence threshold (eV/A)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=200, help="maximum optimizer steps"
    )
    parser.add_argument("--dt", type=float, default=1.0, help="initial FIRE step (fs)")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.log_every <= 0:
        parser.error("--log-every must be a positive integer")
    device = torch.device(args.device)

    model = DPA4Wrapper.from_checkpoint(args.model, device=device)
    model.eval()

    atomic_numbers, positions, cell = load_frame(
        args.data, frame=args.frame, device=device
    )
    n_atoms = atomic_numbers.shape[0]
    # FIRE reuses the ``velocities`` field as its internal velocity (starting
    # from rest); ``forces`` and ``energy`` are written back by ``compute()``.
    data = AtomicData(
        atomic_numbers=atomic_numbers,
        positions=positions,
        cell=cell,
        pbc=torch.ones(1, 3, dtype=torch.bool, device=device),
        forces=torch.zeros_like(positions),
        energy=torch.zeros((1, 1), dtype=positions.dtype, device=device),
    )
    batch = Batch.from_data_list([data], device=device)

    nl_hook = NeighborListHook(
        model.model_config.neighbor_config, stage=DynamicsStage.BEFORE_COMPUTE
    )
    opt = FIRE2(
        model,
        dt=args.dt,
        hooks=[nl_hook],
        convergence_hook=ConvergenceHook.from_fmax(threshold=args.fmax),
    )

    compute_neighbors(batch, config=model.model_config.neighbor_config)
    opt.compute(batch)

    e0 = batch.energy.item()
    print(f"model     : {args.model}  (rcut={model.rcut} A)")
    print(f"system    : {n_atoms} atoms, fmax target={args.fmax} eV/A")
    print(f"{'step':>8} {'E_pot[eV]':>14} {'fmax[eV/A]':>12}")
    print(f"{0:>8} {e0:>14.4f} {fmax(batch):>12.5f}")

    converged = False
    step = 0
    while step < args.max_steps:
        chunk = min(args.log_every, args.max_steps - step)
        batch = opt.run(batch, n_steps=chunk)
        step += chunk
        print(f"{step:>8} {batch.energy.item():>14.4f} {fmax(batch):>12.5f}")
        if fmax(batch) <= args.fmax:
            converged = True
            break

    e1 = batch.energy.item()
    status = "converged" if converged else f"not converged in {args.max_steps} steps"
    print(f"\n{status}: fmax={fmax(batch):.5f} eV/A")
    print(f"energy change: {(e1 - e0) / n_atoms * 1e3:.4f} meV/atom")


if __name__ == "__main__":
    main()
