# SPDX-License-Identifier: LGPL-3.0-or-later
"""Run two independent DeepMD LAMMPS instances on MPI subcommunicators."""

import argparse

import numpy as np
from lammps import (
    lammps,
)
from mpi4py import (
    MPI,
)


def main() -> None:
    """Load a different model in each two-rank LAMMPS communicator."""
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file")
    parser.add_argument("model_a")
    parser.add_argument("model_b")
    parser.add_argument("output_a")
    parser.add_argument("output_b")
    args = parser.parse_args()

    parent = MPI.COMM_WORLD
    if parent.Get_size() != 4:
        raise RuntimeError("This regression runner requires exactly four ranks")

    color = parent.Get_rank() // 2
    comm = parent.Split(color, parent.Get_rank())
    model = args.model_a if color == 0 else args.model_b
    output = args.output_a if color == 0 else args.output_b

    # Use the low-level wrapper because PyLammps is deprecated and its output
    # capture is explicitly experimental with more than one MPI rank.  Raw
    # commands keep this regression focused on communicator ownership.
    lmp = lammps(comm=comm)
    lmp.commands_list(
        [
            "processors 1 2 1",
            "units metal",
            "atom_style atomic",
            "neighbor 2.0 bin",
            "neigh_modify every 10 delay 0 check no",
            f"read_data {args.data_file}",
            "mass 1 16",
            "mass 2 2",
            "timestep 0.0005",
            "fix 1 all nve",
            f"pair_style deepmd {model}",
            "pair_coeff * *",
            "run 0",
        ]
    )

    # LAMMPS gather is collective on the instance communicator.  Sorting by
    # atom id makes each subcommunicator's output directly comparable with the
    # single-instance reference arrays in test_lammps.py.
    forces_global = lmp.gather_atoms("f", 1, 3)
    ids_global = lmp.gather_atoms("id", 0, 1)
    if comm.Get_rank() == 0:
        natoms = lmp.get_natoms()
        forces = np.array(forces_global, dtype=np.float64).reshape(natoms, 3)
        ids = np.array(ids_global, dtype=np.int64).reshape(natoms)
        np.savetxt(output, forces[np.argsort(ids)])

    # Destroy LAMMPS before freeing the communicator it was constructed with.
    lmp.close()
    comm.Free()
    parent.Barrier()
    MPI.Finalize()


if __name__ == "__main__":
    main()
