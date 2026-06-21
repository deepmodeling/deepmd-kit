# SPDX-License-Identifier: LGPL-3.0-or-later
"""Canonical (NVT) molecular dynamics with a DPA-4 / SeZM model.

This example runs Langevin NVT dynamics through ``nvalchemi`` driven by a trained
DPA-4 / SeZM potential. A Langevin thermostat couples the system to a heat bath,
so the instantaneous temperature fluctuates around the target value rather than
being conserved (as in NVE).

The wiring mirrors :mod:`run_nve` -- wrap the model, build a periodic batch,
seed Maxwell-Boltzmann velocities, register a neighbour-list hook -- but uses
:class:`~nvalchemi.dynamics.integrators.NVTLangevin`, which additionally takes a
target ``temperature`` and a ``friction`` coefficient.

Usage
-----
::

    python run_nvt.py \
        --model ../lmp/pretrained.pt \
        --data ../../data/data_0 \
        --steps 300 --dt 0.5 --temperature 330 --friction 0.01
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
from nvalchemi.dynamics import (
    initialize_velocities,
)
from nvalchemi.dynamics.base import (
    DynamicsStage,
)
from nvalchemi.dynamics.integrators import (
    NVTLangevin,
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

# Boltzmann constant in eV/K (positions in A, masses in amu, energy in eV).
_KB_EV = 8.617333262e-5


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


def temperature_kelvin(batch: Batch, n_atoms: int) -> float:
    """Instantaneous kinetic temperature from ``T = 2 KE / (3 N k_B)``."""
    ke = (0.5 * batch.atomic_masses * (batch.velocities**2).sum(-1)).sum().item()
    return 2.0 * ke / (3.0 * n_atoms * _KB_EV)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="../lmp/pretrained.pt")
    parser.add_argument("--data", default="../../data/data_0")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--steps", type=int, default=300, help="number of MD steps")
    parser.add_argument("--dt", type=float, default=0.5, help="timestep in fs")
    parser.add_argument(
        "--temperature", type=float, default=330.0, help="target temperature in K"
    )
    parser.add_argument(
        "--friction", type=float, default=0.01, help="Langevin friction in 1/fs"
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
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
    data = AtomicData(
        atomic_numbers=atomic_numbers,
        positions=positions,
        cell=cell,
        pbc=torch.ones(1, 3, dtype=torch.bool, device=device),
        forces=torch.zeros_like(positions),
        energy=torch.zeros((1, 1), dtype=positions.dtype, device=device),
    )
    batch = Batch.from_data_list([data], device=device)

    temperature = torch.full(
        (batch.num_graphs,), args.temperature, dtype=positions.dtype, device=device
    )
    initialize_velocities(
        batch.velocities,
        batch.atomic_masses,
        temperature,
        batch.batch_idx.int(),
        random_seed=args.seed,
    )

    nl_hook = NeighborListHook(
        model.model_config.neighbor_config, stage=DynamicsStage.BEFORE_COMPUTE
    )
    nvt = NVTLangevin(
        model,
        dt=args.dt,
        temperature=args.temperature,
        friction=args.friction,
        random_seed=args.seed,
        hooks=[nl_hook],
    )

    # Prime the neighbour list and forces before the first half-kick.
    compute_neighbors(batch, config=model.model_config.neighbor_config)
    nvt.compute(batch)

    print(f"model     : {args.model}  (rcut={model.rcut} A)")
    print(
        f"system    : {n_atoms} atoms, dt={args.dt} fs, "
        f"T_target={args.temperature} K, friction={args.friction}/fs"
    )
    print(f"{'step':>8} {'E_pot[eV]':>14} {'T[K]':>9} {'fmax':>9}")
    print(
        f"{0:>8} {batch.energy.item():>14.4f} "
        f"{temperature_kelvin(batch, n_atoms):>9.2f} "
        f"{batch.forces.norm(dim=-1).max().item():>9.4f}"
    )

    step = 0
    while step < args.steps:
        chunk = min(args.log_every, args.steps - step)
        batch = nvt.run(batch, n_steps=chunk)
        step += chunk
        print(
            f"{step:>8} {batch.energy.item():>14.4f} "
            f"{temperature_kelvin(batch, n_atoms):>9.2f} "
            f"{batch.forces.norm(dim=-1).max().item():>9.4f}"
        )


if __name__ == "__main__":
    main()
