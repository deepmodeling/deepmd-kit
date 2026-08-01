# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared MPI driver for DeepMD and DeepSpin LAMMPS pair-style tests."""

import argparse
from typing import (
    NamedTuple,
)

import numpy as np
from lammps import (
    PyLammps,
)
from mpi4py import (
    MPI,
)


class PairStyleConfig(NamedTuple):
    """LAMMPS commands that differ between the DeepMD and DeepSpin runners."""

    atom_style: str
    masses: tuple[str, str]
    pair_style: str


def run_mpi_pair_deepmd(config: PairStyleConfig) -> None:
    """Run the common two-rank model-deviation scenario.

    The public runner scripts remain separate because their model and data-file
    contracts differ.  Keeping those scripts as wrappers also preserves their
    command-line entry points for pytest and external build tooling.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("DATAFILE", type=str)
    parser.add_argument("PBFILE", type=str)
    parser.add_argument("PBFILE2", type=str)
    parser.add_argument("MD_FILE", type=str)
    parser.add_argument("OUTPUT", type=str)
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--nopbc", action="store_true")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    lammps = PyLammps()
    if args.balance:
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
    lammps.atom_style(config.atom_style)
    lammps.neighbor("2.0 bin")
    lammps.neigh_modify("every 10 delay 0 check no")
    lammps.read_data(args.DATAFILE)
    lammps.mass(f"1 {config.masses[0]}")
    lammps.mass(f"2 {config.masses[1]}")
    lammps.timestep(0.0005)
    lammps.fix("1 all nve")

    relative = 1.0
    lammps.pair_style(
        f"{config.pair_style} {args.PBFILE} {args.PBFILE2} "
        f"out_file {args.MD_FILE} out_freq 1 atomic relative {relative}"
    )
    lammps.pair_coeff("* *")
    lammps.run(0)
    if rank == 0:
        pe = lammps.eval("pe")
        np.savetxt(args.OUTPUT, np.array([pe]))

    # LAMMPS owns MPI resources, so its destructor must run before finalization.
    # Changing this order can make the destructor call MPI after MPI_Finalize.
    del lammps
    MPI.Finalize()
