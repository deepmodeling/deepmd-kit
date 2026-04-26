# SPDX-License-Identifier: LGPL-3.0-or-later
"""Multi-rank LAMMPS driver for DPA3 .pt2 (Phase 5 of GNN MPI).

Run via ``mpirun -n N python run_mpi_pair_deepmd_dpa3_pt2.py DATAFILE PB_FILE OUTPUT``.
Mirrors ``run_mpi_pair_deepmd.py`` but targets a GNN model whose .pt2 archive
carries the with-comm artifact (Phase 3 dual-artifact layout). The C++
``DeepPotPTExpt`` (Phase 4) routes to the with-comm artifact when LAMMPS
reports nswap > 0 (multi-rank), driving MPI ghost-atom exchange via
``deepmd_export::border_op`` per layer.

Rank 0 writes potential energy + per-atom forces to ``OUTPUT`` so the parent
pytest process can compare against the single-rank reference.
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
parser.add_argument("DATAFILE", type=str, help="LAMMPS data file (atom positions)")
parser.add_argument("PB_FILE", type=str, help=".pt2 model file")
parser.add_argument("OUTPUT", type=str, help="Output file for energies + forces")
args = parser.parse_args()

lammps = PyLammps()
# Force a non-trivial domain decomposition: 2 x 1 x 1 across ranks.
# Combined with the simulation box this guarantees nswap > 0 on the C++
# side, so DeepPotPTExpt routes to the with-comm AOTI artifact.
lammps.processors("2 1 1")
lammps.units("metal")
lammps.boundary("p p p")
lammps.atom_style("atomic")
lammps.neighbor("2.0 bin")
lammps.neigh_modify("every 10 delay 0 check no")
lammps.read_data(args.DATAFILE)
lammps.mass("1 16")
lammps.mass("2 2")
lammps.timestep(0.0005)
lammps.fix("1 all nve")

lammps.pair_style(f"deepmd {args.PB_FILE}")
lammps.pair_coeff("* *")
lammps.run(0)

# Forces need to be gathered across ranks. PyLammps's ``atoms[i]``
# only exposes rank-local atoms; ``gather_atoms`` returns the global,
# id-ordered array on every rank.
forces_global = lammps.lmp.gather_atoms("f", 1, 3)
# ``PyLammps.eval`` is rank-0-only.
if rank == 0:
    pe_global = lammps.eval("pe")
    natoms = lammps.atoms.natoms
    forces = np.array(forces_global, dtype=np.float64).reshape(natoms, 3)
    with open(args.OUTPUT, "w") as f:
        f.write(f"{pe_global:.16e}\n")
        for row in forces:
            f.write(" ".join(f"{v:.16e}" for v in row) + "\n")

MPI.Finalize()
