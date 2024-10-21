# SPDX-License-Identifier: LGPL-3.0-or-later
"""Use mpi4py to run a LAMMPS pair_deepmd + model deviation (atomic, relative) task."""

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
parser.add_argument("DATAFILE", type=str)
parser.add_argument("PBFILE", type=str)
parser.add_argument("PBFILE2", type=str)
parser.add_argument("MD_FILE", type=str)
parser.add_argument("OUTPUT", type=str)
parser.add_argument("--balance", action="store_true")
parser.add_argument("--nopbc", action="store_true")

args = parser.parse_args()
data_file = args.DATAFILE
pb_file = args.PBFILE
pb_file2 = args.PBFILE2
md_file = args.MD_FILE
output = args.OUTPUT
balance = args.balance

lammps = PyLammps()
if balance:
    # 4 and 2 atoms
    lammps.processors("2 1 1")
else:
    # 6 and 0 atoms
    lammps.processors("1 2 1")
lammps.units("metal")
if args.nopbc:
    lammps.boundary("f f f")
else:
    lammps.boundary("p p p")
lammps.atom_style("atomic")
lammps.neighbor("2.0 bin")
lammps.neigh_modify("every 10 delay 0 check no")
lammps.read_data(data_file)
lammps.mass("1 16")
lammps.mass("2 2")
lammps.timestep(0.0005)
lammps.fix("1 all nve")

relative = 1.0
lammps.pair_style(
    f"deepmd {pb_file} {pb_file2} out_file {md_file} out_freq 1 atomic relative {relative}"
)
lammps.pair_coeff("* *")
lammps.run(0)
if rank == 0:
    pe = lammps.eval("pe")
    arr = [pe]
    np.savetxt(output, np.array(arr))
MPI.Finalize()
