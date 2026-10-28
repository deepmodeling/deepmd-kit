# SPDX-License-Identifier: LGPL-3.0-or-later
"""Multi-rank LAMMPS driver for the native-spin DPA4 graph ``.pt2`` fixture.

``atom_style spin`` / ``pair_style deepspin`` runner for
``test_lammps_dpa4_spin_graph_pt2.py``'s multi-rank comparison.  The
native-spin DPA4 graph archive now carries the nested with-comm AOTI
artifact, so ``DeepSpinPTExpt::compute_inner`` drives a real domain-
decomposed run through ``run_model_graph_with_comm`` (per-block ghost
FEATURE refresh via ``border_op``; ghost SPINS arrive through the LAMMPS
``sp`` forward-comm).

Rank 0 writes potential energy + per-atom force (3 cols) + per-atom
force_mag (3 cols) to ``OUTPUT``, id-ordered, so the parent pytest process
can compare a 2-rank run against a 1-rank run on the SAME archive.  Same
output convention as ``run_mpi_pair_deepmd_spin_dpa3_pt2.py`` minus the
virial columns (the native-spin fixture takes no fparam/aparam).
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

rank = MPI.COMM_WORLD.Get_rank()

parser = argparse.ArgumentParser()
parser.add_argument(
    "DATAFILE", type=str, help="LAMMPS data file (atom positions + spin)"
)
parser.add_argument("PB_FILE", type=str, help=".pt2 model file (native-spin graph)")
parser.add_argument("OUTPUT", type=str, help="Unused; kept for CLI-shape parity")
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
# Per-atom magnetic force components: LAMMPS does not expose ``fm`` through
# the legacy extract/gather_atoms registry, so go via
# ``compute property/atom fmx fmy fmz`` + ``gather``.
lammps.compute("fmprop all property/atom fmx fmy fmz")
lammps.run(0)

forces_global = lammps.lmp.gather_atoms("f", 1, 3)
ids_global = lammps.lmp.gather_atoms("id", 0, 1)
fm_global = lammps.lmp.gather("c_fmprop", 1, 3)

if rank == 0:
    pe_global = lammps.eval("pe")
    natoms = lammps.atoms.natoms
    forces = np.array(forces_global, dtype=np.float64).reshape(natoms, 3)
    fm = np.array(fm_global, dtype=np.float64).reshape(natoms, 3)
    ids = np.array(ids_global, dtype=np.int64).reshape(natoms)
    order = np.argsort(ids)
    forces = forces[order]
    fm = fm[order]
    with open(args.OUTPUT, "w") as f:
        f.write(f"{pe_global:.16e}\n")
        # Each row: 3 force + 3 force_mag = 6 columns.
        for fi, fmi in zip(forces, fm, strict=True):
            row = np.concatenate([fi, fmi])
            f.write(" ".join(f"{v:.16e}" for v in row) + "\n")

# Tear down LAMMPS before MPI.Finalize() so its destructor's MPI calls run
# while the communicator is still valid (see the dpa3 spin runner).
del lammps
MPI.Finalize()
