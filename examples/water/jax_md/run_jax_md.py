# SPDX-License-Identifier: LGPL-3.0-or-later
"""Run a short JAX-MD trajectory with a DeePMD JAX checkpoint."""

from __future__ import (
    annotations,
)

import argparse
import sys
from pathlib import (
    Path,
)
from typing import (
    Any,
)

import numpy as np
from jax_md import (
    quantity,
    simulate,
    space,
)

K_B_EV_PER_K = 8.617333262145e-5
AMU_TO_EV_PS2_PER_A2 = 1.0364269656262175e-4
WATER_TYPE_MAP = ("O", "H")
WATER_MASS_AMU = {
    "O": 16.0,
    "H": 2.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DeePMD JAX checkpoint through JAX-MD on water.lmp."
    )
    parser.add_argument(
        "--model",
        default="../se_e2_a/model.ckpt.jax",
        help="Path to a DeePMD JAX checkpoint directory or stable checkpoint pointer.",
    )
    parser.add_argument(
        "--data",
        default="../lmp/water.lmp",
        help="LAMMPS data file used as initial coordinates.",
    )
    parser.add_argument("--steps", type=int, default=10, help="NVE integration steps.")
    parser.add_argument(
        "--dt",
        type=float,
        default=0.0005,
        help="Timestep in ps, matching the LAMMPS metal-unit example.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=330.0,
        help="Initial temperature in K.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23456789,
        help="Random seed matching the LAMMPS velocity command.",
    )
    parser.add_argument(
        "--dr-threshold",
        type=float,
        default=0.2,
        help="JAX-MD neighbor-list update threshold in Angstrom.",
    )
    parser.add_argument(
        "--capacity-multiplier",
        type=float,
        default=1.5,
        help="JAX-MD neighbor-list capacity multiplier.",
    )
    return parser.parse_args()


def read_lammps_water(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read the water LAMMPS data file with dpdata."""
    import dpdata

    system = dpdata.System(path, fmt="lammps/lmp", type_map=list(WATER_TYPE_MAP))
    coord = np.asarray(system.data["coords"][0], dtype=np.float64)
    atom_types = np.asarray(system.data["atom_types"], dtype=np.int32)
    cell = np.asarray(system.data["cells"][0], dtype=np.float64)
    off_diagonal = cell - np.diag(np.diag(cell))
    if np.any(np.abs(off_diagonal) > 1e-12):
        raise ValueError("This JAX-MD example only supports orthogonal boxes.")
    type_names = np.asarray(WATER_TYPE_MAP, dtype=object)[atom_types]
    masses = np.asarray([WATER_MASS_AMU[name] for name in type_names], dtype=np.float64)
    masses = masses[:, None] * AMU_TO_EV_PS2_PER_A2
    box = np.diag(cell)
    return coord, atom_types, masses, box


def emit(line: str) -> None:
    sys.stdout.write(line + "\n")


def main() -> None:
    from deepmd.jax.env import (
        jax,
        jnp,
    )
    from deepmd.jax.jax_md import (
        as_jax_md,
    )

    args = parse_args()
    backend = jax.default_backend()
    devices = jax.devices()

    coord_np, atom_types_np, masses_np, box_np = read_lammps_water(Path(args.data))
    coord = jnp.asarray(coord_np)
    atom_types = jnp.asarray(atom_types_np)
    masses = jnp.asarray(masses_np)
    box = jnp.asarray(box_np)
    kT = K_B_EV_PER_K * args.temperature

    displacement_fn, shift_fn = space.periodic(box)
    neighbor_fn, potential_fn = as_jax_md(
        args.model,
        displacement_fn,
        box,
        atom_types,
        dr_threshold=args.dr_threshold,
        capacity_multiplier=args.capacity_multiplier,
    )
    neighbor = neighbor_fn.allocate(coord)
    init_fn, step_fn = simulate.nve(potential_fn, shift_fn, dt=args.dt)
    key = jax.random.key(args.seed)
    state = init_fn(key, coord, kT=kT, mass=masses, neighbor=neighbor)

    emit(f"jax_backend {backend}")
    emit("jax_devices " + ", ".join(str(device) for device in devices))
    emit(f"neighbor_idx_shape {tuple(neighbor.idx.shape)}")
    emit("# step potential_eV kinetic_eV temperature_K neighbor_overflow")

    def thermo(current_state: Any, current_neighbor: Any) -> tuple[Any, Any, Any]:
        energy = potential_fn(current_state.position, neighbor=current_neighbor)
        kinetic = quantity.kinetic_energy(
            momentum=current_state.momentum,
            mass=current_state.mass,
        )
        temperature = quantity.temperature(
            momentum=current_state.momentum,
            mass=current_state.mass,
        )
        return energy, kinetic, temperature

    @jax.jit
    def md_step(current_state: Any, current_neighbor: Any) -> tuple[Any, ...]:
        current_neighbor = neighbor_fn.update(current_state.position, current_neighbor)
        current_state = step_fn(current_state, neighbor=current_neighbor)
        return current_state, current_neighbor, *thermo(current_state, current_neighbor)

    energy, kinetic, temperature = thermo(state, neighbor)
    emit(
        f"0 {float(energy):.12e} {float(kinetic):.12e} "
        f"{float(temperature / K_B_EV_PER_K):.6f} {bool(neighbor.did_buffer_overflow)}"
    )
    for step in range(1, args.steps + 1):
        state, neighbor, energy, kinetic, temperature = md_step(state, neighbor)
        emit(
            f"{step} {float(energy):.12e} {float(kinetic):.12e} "
            f"{float(temperature / K_B_EV_PER_K):.6f} "
            f"{bool(neighbor.did_buffer_overflow)}"
        )


if __name__ == "__main__":
    main()
