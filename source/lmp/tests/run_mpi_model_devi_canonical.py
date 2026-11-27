# SPDX-License-Identifier: LGPL-3.0-or-later
"""Run the model-deviation scenarios of one pair style on one or more ranks.

``mpirun -n N python run_mpi_model_devi_canonical.py DATA OUTPUT --style STYLE
--models M0 M1`` evaluates, on the configuration of ``DATA``:

- each model alone at step 0: its force, for a spin style the negative
  energy gradient with respect to the spin (the output of the model in metal
  units, recovered from the stored precession magnetic force), and the sum of
  its per-atom centroid virial;
- the ensemble of both models at step 0, with the force that drives the
  dynamics, the deviation of every atom and the number of local and ghost
  atoms of every rank;
- unless ``--point`` is given, the ensemble over a run of six steps of
  dynamics with a neighbor rebuild every third step, a run of one step, and,
  after a reset of the step to one and a displacement that changes the
  neighbors of an atom, a run of one step more, writing the deviation every
  step and every second step, with the number of local atoms of every rank
  before and after the six steps.

Rank 0 writes the outputs and the rows of the deviation files as JSON. Every
rank runs on the one visible GPU.
"""

from __future__ import (
    annotations,
)

import argparse
import json
from contextlib import (
    contextmanager,
)
from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
)

import numpy as np
from lammps import (
    lammps,
)
from mpi4py import (
    MPI,
)

if TYPE_CHECKING:
    from collections.abc import (
        Iterator,
    )

# Reduced Planck constant in eV.ps, which scales the stored magnetic force.
HBAR_METAL = 6.5821191e-04


class _Session:
    """LAMMPS instances of one pair style, which share one Kokkos runtime."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.kokkos = args.style.endswith("/kk")
        self.magnetic = args.style in ("dpa4spin", "dpa4spin/kk")
        self.library = None

    @contextmanager
    def open(self, pair_style: str) -> Iterator[lammps]:
        """Open an instance whose pair style evaluates ``pair_style``."""
        cmdargs = ["-log", "none", "-screen", "none"]
        if self.kokkos:
            cmdargs += ["-k", "on", "g", "1", "-pk", "kokkos", "neigh", "full"]
            cmdargs += ["newton", "off", "comm", "host", "gpu/aware", "off"]
        with lammps(cmdargs=cmdargs, comm=MPI.COMM_WORLD) as lmp:
            self.library = lmp.lib
            self._setup(lmp, pair_style)
            yield lmp

    def _setup(self, lmp: lammps, pair_style: str) -> None:
        """Read the system and define the pair style of an instance."""
        suffix = "/kk" if self.kokkos else ""
        atom_style = "spin" if self.magnetic else "atomic"
        masses = ("58.6934", "15.9994") if self.magnetic else ("16", "2")
        for command in (
            "units metal",
            f"processors {MPI.COMM_WORLD.size} 1 1",
            f"atom_style {atom_style}{suffix}",
            "atom_modify map yes",
            f"read_data {self.args.data.resolve()}",
            f"run_style verlet{suffix}",
            f"mass 1 {masses[0]}",
            f"mass 2 {masses[1]}",
            f"pair_style {self.args.style} {pair_style}",
            "pair_coeff * *",
            "neighbor 1.0 bin",
            "neigh_modify every 3 delay 0 check no",
            "timestep 0.0005",
        ):
            lmp.command(command)
        assert lmp.extract_global("pair_style") == self.args.style

    def single(self, model: Path) -> dict:
        """Outputs of one model alone at step 0."""
        with self.open(str(model.resolve())) as lmp:
            lmp.command("compute vatom all centroid/stress/atom NULL pair")
            if self.magnetic:
                lmp.command("compute fmprop all property/atom fmx fmy fmz")
                lmp.command("compute moment all property/atom sp")
            lmp.command("run 0")
            # centroid/stress/atom reports the negative per-atom virial in
            # pressure*volume units; dividing by -nktv2p gives eV.
            scale = -1.0 / lmp.extract_global("nktv2p")
            virial = scale * np.sum(_gather(lmp, "c_vatom", 9), axis=0)
            result = {"force": _gather(lmp, "f", 3), "virial": virial.tolist()}
            if self.magnetic:
                # The pair style stores |m|/hbar times -dE/dS, whereas the
                # deviation is that of -dE/dS, the output of the models.
                moment = np.array(_gather(lmp, "c_moment", 1))[:, 0]
                assert np.all(moment > 0), "every atom of the fixture is magnetic"
                magnetic = np.array(_gather(lmp, "c_fmprop", 3))
                gradient = magnetic * (HBAR_METAL / moment)[:, None]
                result["magnetic_force"] = gradient.tolist()
            return result

    def ensemble(self, out_file: Path, out_freq: int, nsteps: int) -> dict:
        """Run the ensemble of both models and read its deviation file."""
        models = " ".join(str(model.resolve()) for model in self.args.models)
        comm = MPI.COMM_WORLD
        pair_style = f"{models} out_file {out_file} out_freq {out_freq} atomic"
        with self.open(pair_style) as lmp:
            if nsteps > 0:
                # The generic host fix marks every atom array as modified on
                # the host, the magnetic-force DualViews included; Kokkos NVE
                # modifies only the positions and the velocities.
                suffix = "/kk" if self.kokkos else ""
                lmp.command(f"fix integrate all nve{suffix}")
            # read_data has assigned every atom to the rank that owns it.
            counts = [comm.allgather(lmp.extract_setting("nlocal"))]
            lmp.command(f"run {nsteps}")
            atoms = lmp.extract_setting("nlocal") + lmp.extract_setting("nghost")
            counts.append(comm.allgather(lmp.extract_setting("nlocal")))
            result = {
                "force": _gather(lmp, "f", 3),
                "local_counts": counts,
                "atom_counts": comm.allgather(atoms),
            }
            if nsteps > 0:
                # The further models hold the list of their last evaluation.
                # A new run counts its neighbor builds afresh, and a reset of
                # the step repeats steps, so the last run, which is due only
                # at its second step, finds the count and the step of a list
                # they have seen, after the neighbors have changed.
                lmp.command("run 1")
                lmp.command("reset_timestep 1")
                # The move brings an atom to 2.4 Angstrom of atom 1 that lay
                # beyond the cutoff and the skin of its earlier list.
                lmp.command("group moved id 1")
                lmp.command("displace_atoms moved move -1.0 2.0 -2.0 units box")
                lmp.command("run 1")
        # The pair style closes the deviation file with its instance.
        if comm.rank == 0:
            result["rows"] = np.atleast_2d(np.loadtxt(out_file)).tolist()
        return result


def _gather(lmp: lammps, name: str, count: int) -> list[list[float]]:
    # gather() returns atom-ID order, and every rank takes part in it.
    return np.array(lmp.gather(name, 1, count)).reshape(-1, count).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--style", required=True)
    parser.add_argument("--models", type=Path, nargs=2, required=True)
    parser.add_argument(
        "--point", action="store_true", help="Run the single point only."
    )
    args = parser.parse_args()
    session = _Session(args)
    work = args.output.parent
    try:
        # Every run writes a deviation row at step 0, so each writes its own
        # file.
        results = {
            "single": [session.single(model) for model in args.models],
            "ensemble": session.ensemble(work / "md_single_point.out", 1, 0),
        }
        if not args.point:
            results["every_step"] = session.ensemble(work / "md_every_step.out", 1, 6)
            results["every_second_step"] = session.ensemble(
                work / "md_every_second.out", 2, 6
            )
        if MPI.COMM_WORLD.rank == 0:
            args.output.write_text(json.dumps(results))
    finally:
        # close() destroys each instance but keeps Kokkos initialized for
        # later instances. Finalize once, even on error, before mpi4py
        # finalizes MPI.
        if session.kokkos and session.library is not None:
            session.library.lammps_kokkos_finalize()


if __name__ == "__main__":
    main()
