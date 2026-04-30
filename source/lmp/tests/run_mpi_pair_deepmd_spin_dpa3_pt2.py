# SPDX-License-Identifier: LGPL-3.0-or-later
"""Multi-rank LAMMPS driver for the DPA3 spin GNN .pt2 fixture.

Mirrors ``run_mpi_pair_deepmd_dpa3_pt2.py`` but for spin models:
``atom_style spin`` / ``pair_style deepspin`` and gathers the
per-atom magnetic force ``fm`` in addition to the normal force and
per-atom virial. The DPA3 spin .pt2 with ``use_loc_mapping=False``
carries a with-comm AOTI artifact (Phase 3 dual-artifact layout); the
C++ ``DeepSpinPTExpt`` (Phase 4c) routes to it when LAMMPS reports
nswap > 0 (multi-rank), driving MPI ghost-atom exchange via
``deepmd_export::border_op``.

Rank 0 writes potential energy + per-atom forces (3 cols) +
per-atom force_mag (3 cols) + per-atom virial (9 cols, from
``compute centroid/stress/atom NULL pair`` in LAMMPS internal units)
to ``OUTPUT`` so the parent pytest process can compare against the
single-rank reference.
"""

from __future__ import (
    annotations,
)

import argparse

import numpy as np
from lammps import (
    PyLammps,
)
from mpi4py import (
    MPI,
)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

parser = argparse.ArgumentParser()
parser.add_argument(
    "DATAFILE", type=str, help="LAMMPS data file (atom positions + spin)"
)
parser.add_argument("PB_FILE", type=str, help=".pt2 model file (spin GNN)")
parser.add_argument(
    "OUTPUT", type=str, help="Output file for energies + forces + force_mag + virial"
)
parser.add_argument(
    "--nsteps",
    type=int,
    default=0,
    help="Number of MD steps to run after the initial force evaluation. "
    "Note: integrating spin requires fix nve/spin which is outside the "
    "scope of this multi-rank correctness test; we only run static "
    "force/energy evaluations and an optional run > 0 to exercise the "
    "with-comm dispatch across neighbour-list rebuilds.",
)
parser.add_argument(
    "--processors",
    type=str,
    default="2 1 1",
    help="LAMMPS processors grid. Default '2 1 1' forces multi-rank "
    "domain decomposition (nswap>0). Pass '1 1 1' for a single-rank "
    "reference run on the same archive.",
)
args = parser.parse_args()

lammps = PyLammps()
lammps.processors(args.processors)
lammps.units("metal")
lammps.boundary("p p p")
lammps.atom_style("spin")
lammps.atom_modify("map yes")
lammps.neighbor("2.0 bin")
lammps.neigh_modify("every 10 delay 0 check no")
lammps.read_data(args.DATAFILE)
lammps.mass("1 58")
lammps.mass("2 16")
lammps.timestep(0.0005)
lammps.fix("1 all nve")

lammps.pair_style(f"deepspin {args.PB_FILE}")
lammps.pair_coeff("* *")
lammps.compute("virial all centroid/stress/atom NULL pair")
# Per-atom magnetic force components. LAMMPS does not expose ``fm``
# through the legacy ``extract``/``gather_atoms`` registry, so we go
# via ``compute property/atom fmx fmy fmz`` + ``gather`` to obtain a
# global, id-ordered (nlocal+nghost reduced) array on every rank.
lammps.compute("fmprop all property/atom fmx fmy fmz")
lammps.run(0)

if args.nsteps > 0:
    lammps.run(args.nsteps)

# All per-atom data goes through the LAMMPS global gather API.
# ``c_fmprop`` is the compute defined above (fmx/fmy/fmz columns).
forces_global = lammps.lmp.gather_atoms("f", 1, 3)
ids_global = lammps.lmp.gather_atoms("id", 0, 1)
virial_global = lammps.lmp.gather("c_virial", 1, 9)
fm_global = lammps.lmp.gather("c_fmprop", 1, 3)

if rank == 0:
    pe_global = lammps.eval("pe")
    natoms = lammps.atoms.natoms
    forces = np.array(forces_global, dtype=np.float64).reshape(natoms, 3)
    fm = np.array(fm_global, dtype=np.float64).reshape(natoms, 3)
    virials = np.array(virial_global, dtype=np.float64).reshape(natoms, 9)
    ids = np.array(ids_global, dtype=np.int64).reshape(natoms)
    order = np.argsort(ids)
    forces = forces[order]
    fm = fm[order]
    virials = virials[order]
    with open(args.OUTPUT, "w") as f:
        f.write(f"{pe_global:.16e}\n")
        # Each row: 3 force + 3 force_mag + 9 virial = 15 columns.
        for fi, fmi, vi in zip(forces, fm, virials, strict=True):
            row = np.concatenate([fi, fmi, vi])
            f.write(" ".join(f"{v:.16e}" for v in row) + "\n")

MPI.Finalize()
