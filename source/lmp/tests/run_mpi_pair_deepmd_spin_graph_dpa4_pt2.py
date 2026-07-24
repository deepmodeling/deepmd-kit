# SPDX-License-Identifier: LGPL-3.0-or-later
"""Multi-rank LAMMPS driver for the native-spin DPA4 graph ``.pt2`` fixture.

Minimal ``atom_style spin`` / ``pair_style deepspin`` runner used ONLY by
``test_lammps_dpa4_spin_graph_pt2.py``'s multi-rank fail-fast test: the
native-spin DPA4 graph archive (``deeppot_dpa4_spin_graph.pt2``) is exported
with ``has_comm_artifact=false`` unconditionally (no nested with-comm AOTI
artifact -- see ``source/tests/infer/gen_dpa4_spin.py``), so
``DeepSpinPTExpt::compute_inner`` throws on ANY ``nprocs > 1`` run before
doing meaningful work. Unlike
``run_mpi_pair_deepmd_spin_dpa3_pt2.py`` (the virtual-atom-scheme spin GNN
twin, which DOES have a with-comm artifact and drives a real multi-rank
comparison), this runner is not expected to produce usable output -- it
exists to reach the C++ fail-fast throw, not to complete a run. No
fparam/aparam handling: the native-spin DPA4 fixture takes neither.
"""

from __future__ import (
    annotations,
)

import argparse

from lammps import (
    PyLammps,
)
from mpi4py import (
    MPI,
)

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
    "domain decomposition, which alone is sufficient to trigger the "
    "has_comm_artifact=false fail-fast guard.",
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
# Expected to throw (deepmd::deepmd_exception -> LAMMPS error->all() ->
# MPI_Abort) before returning -- the test only checks the process exits
# nonzero with the documented message; no output file is written.
lammps.run(0)

del lammps
MPI.Finalize()
