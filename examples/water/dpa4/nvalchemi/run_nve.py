# SPDX-License-Identifier: LGPL-3.0-or-later
"""Microcanonical (NVE) molecular dynamics with a DPA-4 / SeZM model.

This example runs velocity-Verlet NVE dynamics through ``nvalchemi`` driven by a
trained DPA-4 / SeZM potential. It demonstrates the full molecular-dynamics
wiring:

* wrap the model with :class:`deepmd.pt.nvalchemi.DPA4Wrapper`;
* build a periodic ``nvalchemi`` batch with allocated ``forces`` / ``velocities``;
* draw initial velocities from the Maxwell-Boltzmann distribution;
* register a COO :class:`~nvalchemi.hooks.NeighborListHook` so the neighbour
  list is rebuilt before each force evaluation;
* integrate with :class:`~nvalchemi.dynamics.integrators.NVE` and monitor the
  conserved total energy ``E_pot + E_kin``.

NVE conserves the total energy, so the drift over the run is a direct measure of
the integration quality for the given timestep.

Usage
-----
::

    python run_nve.py \
        --model ../lmp/pretrained.pt \
        --data ../../data/data_0 \
        --steps 200 --dt 0.5 --temperature 300
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
    NVE,
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


def thermo(batch: Batch, n_atoms: int) -> tuple[float, float, float, float]:
    """Return (potential energy, kinetic energy, temperature, max force).

    Kinetic energy is ``0.5 * sum(m v^2)`` in eV (the integrator's internal
    velocity unit makes this expression directly an energy), and the
    temperature follows from equipartition with ``3N`` degrees of freedom.
    """
    ke = (0.5 * batch.atomic_masses * (batch.velocities**2).sum(-1)).sum().item()
    pe = batch.energy.item()
    temperature = 2.0 * ke / (3.0 * n_atoms * _KB_EV)
    fmax = batch.forces.norm(dim=-1).max().item()
    return pe, ke, temperature, fmax


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="../lmp/pretrained.pt")
    parser.add_argument("--data", default="../../data/data_0")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200, help="number of MD steps")
    parser.add_argument("--dt", type=float, default=0.5, help="timestep in fs")
    parser.add_argument(
        "--temperature", type=float, default=300.0, help="initial temperature in K"
    )
    parser.add_argument("--log-every", type=int, default=20)
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
    # ``forces`` and ``energy`` are pre-allocated so the integrator can read
    # forces and ``compute()`` can write energy / forces back in place.
    data = AtomicData(
        atomic_numbers=atomic_numbers,
        positions=positions,
        cell=cell,
        pbc=torch.ones(1, 3, dtype=torch.bool, device=device),
        forces=torch.zeros_like(positions),
        energy=torch.zeros((1, 1), dtype=positions.dtype, device=device),
    )
    batch = Batch.from_data_list([data], device=device)

    # Draw Maxwell-Boltzmann velocities at the target temperature (in-place).
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
    nve = NVE(model, dt=args.dt, hooks=[nl_hook])

    # Prime the neighbour list and forces so the first half-kick is exact.
    compute_neighbors(batch, config=model.model_config.neighbor_config)
    nve.compute(batch)

    pe0, ke0, t0, fmax0 = thermo(batch, n_atoms)
    e_tot0 = pe0 + ke0
    print(f"model     : {args.model}  (rcut={model.rcut} A)")
    print(f"system    : {n_atoms} atoms, dt={args.dt} fs, T0={args.temperature} K")
    print(f"{'step':>8} {'E_pot[eV]':>14} {'E_tot[eV]':>14} {'T[K]':>9} {'fmax':>9}")
    print(f"{0:>8} {pe0:>14.4f} {e_tot0:>14.4f} {t0:>9.2f} {fmax0:>9.4f}")

    step = 0
    while step < args.steps:
        chunk = min(args.log_every, args.steps - step)
        batch = nve.run(batch, n_steps=chunk)
        step += chunk
        pe, ke, temperature_now, fmax = thermo(batch, n_atoms)
        print(
            f"{step:>8} {pe:>14.4f} {pe + ke:>14.4f} {temperature_now:>9.2f} "
            f"{fmax:>9.4f}"
        )

    pe, ke, _, _ = thermo(batch, n_atoms)
    drift = (pe + ke - e_tot0) / n_atoms
    print(f"\ntotal-energy drift: {drift * 1e3:.4f} meV/atom over {args.steps} steps")


if __name__ == "__main__":
    main()
