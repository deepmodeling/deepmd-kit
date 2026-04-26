# SPDX-License-Identifier: LGPL-3.0-or-later
"""Multi-rank LAMMPS driver for DPA3 .pt2 (Phase 5 of GNN MPI).

Run via ``mpirun -n N python run_mpi_pair_deepmd_dpa3_pt2.py DATAFILE PB_FILE OUTPUT``.
Mirrors ``run_mpi_pair_deepmd.py`` but targets a GNN model whose .pt2 archive
carries the with-comm artifact (Phase 3 dual-artifact layout). The C++
``DeepPotPTExpt`` (Phase 4) routes to the with-comm artifact when LAMMPS
reports nswap > 0 (multi-rank), driving MPI ghost-atom exchange via
``deepmd_export::border_op`` per layer.

Rank 0 writes potential energy + per-atom forces (3 cols) + per-atom
virial (9 cols, from ``compute centroid/stress/atom NULL pair`` in
LAMMPS internal units) to ``OUTPUT`` so the parent pytest process can
compare against the single-rank reference.
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
parser.add_argument(
    "--nsteps",
    type=int,
    default=0,
    help="Number of MD steps to run after the initial force evaluation; "
    "with --nsteps > 10 (LAMMPS neigh_modify every=10) the dispatch path "
    "is exercised across at least one neighbor-list rebuild.",
)
parser.add_argument(
    "--processors",
    type=str,
    default="2 1 1",
    help="LAMMPS processors grid. Default '2 1 1' forces multi-rank "
    "domain decomposition (nswap>0). Pass '1 1 1' for a single-rank "
    "reference run on the same archive (single-artifact dispatch).",
)
args = parser.parse_args()

lammps = PyLammps()
# Force the requested domain decomposition. The default "2 1 1"
# combined with the simulation box guarantees nswap > 0 on the C++
# side, so DeepPotPTExpt routes to the with-comm AOTI artifact. Pass
# "1 1 1" to obtain a single-rank reference using the same archive
# (the regular artifact handles nswap==0).
lammps.processors(args.processors)
lammps.units("metal")
lammps.boundary("p p p")
lammps.atom_style("atomic")
# ``atom_modify map yes`` is required when single-rank dispatch goes
# through the regular artifact of a use_loc_mapping=False .pt2: the
# C++ side needs the LAMMPS global-id->local-index map to build the
# ``mapping`` tensor. It is harmless under multi-rank.
lammps.atom_modify("map yes")
lammps.neighbor("2.0 bin")
lammps.neigh_modify("every 10 delay 0 check no")
lammps.read_data(args.DATAFILE)
lammps.mass("1 16")
lammps.mass("2 2")
lammps.timestep(0.0005)
lammps.fix("1 all nve")

lammps.pair_style(f"deepmd {args.PB_FILE}")
lammps.pair_coeff("* *")
# Per-atom virial from the pair contribution. ``centroid/stress/atom``
# is parallel-safe (rank-local data, gathered below). LAMMPS computes
# stress*volume per atom in internal units; the parent test reverses
# the unit conversion (divide by ``constants.nktv2p``) before comparing
# against the reference virial.
lammps.compute("virial all centroid/stress/atom NULL pair")
lammps.run(0)

# Optional: run additional MD steps to exercise the with-comm
# dispatch across neighbor-list rebuilds (LAMMPS rebuilds every
# 10 steps with our neigh_modify config, so any nsteps >= 10
# triggers at least one rebuild).
if args.nsteps > 0:
    lammps.run(args.nsteps)

# Forces need to be gathered across ranks. PyLammps's ``atoms[i]``
# only exposes rank-local atoms; ``gather_atoms`` returns the global,
# id-ordered array on every rank. We also gather ``id`` and reorder
# explicitly by id rather than trusting an implicit ordering — this
# is robust against subdomain layout, empty subdomains, and any
# future LAMMPS change in gather ordering.
forces_global = lammps.lmp.gather_atoms("f", 1, 3)
ids_global = lammps.lmp.gather_atoms("id", 0, 1)
# Gather the per-atom virial across ranks. ``lmp.gather`` accepts
# named per-atom computes (``c_<id>``) and returns the global,
# id-ordered array on every rank.
virial_global = lammps.lmp.gather("c_virial", 1, 9)
# ``PyLammps.eval`` is rank-0-only.
if rank == 0:
    pe_global = lammps.eval("pe")
    natoms = lammps.atoms.natoms
    forces = np.array(forces_global, dtype=np.float64).reshape(natoms, 3)
    virials = np.array(virial_global, dtype=np.float64).reshape(natoms, 9)
    ids = np.array(ids_global, dtype=np.int64).reshape(natoms)
    # Sort by atom id so output is unambiguously id-ordered (id 1 first).
    order = np.argsort(ids)
    forces = forces[order]
    virials = virials[order]
    with open(args.OUTPUT, "w") as f:
        f.write(f"{pe_global:.16e}\n")
        # Each row: 3 force components followed by 9 virial components.
        for fi, vi in zip(forces, virials, strict=True):
            row = np.concatenate([fi, vi])
            f.write(" ".join(f"{v:.16e}" for v in row) + "\n")

MPI.Finalize()
