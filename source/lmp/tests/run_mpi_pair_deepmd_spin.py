# SPDX-License-Identifier: LGPL-3.0-or-later
"""Run the pair_deepspin MPI model-deviation test scenario."""

from mpi_pair_deepmd import (
    PairStyleConfig,
    run_mpi_pair_deepmd,
)

if __name__ == "__main__":
    run_mpi_pair_deepmd(
        PairStyleConfig(atom_style="spin", masses=("58", "16"), pair_style="deepspin")
    )
