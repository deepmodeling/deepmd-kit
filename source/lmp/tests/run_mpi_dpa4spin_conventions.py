# SPDX-License-Identifier: LGPL-3.0-or-later
"""Run native-spin conventions through a real host or CUDA Kokkos plugin.

Each MPI process uses the same GPU, so two-rank coverage needs only one device.
Host-staged communication deliberately does not require CUDA-aware MPI.
"""

from __future__ import (
    annotations,
)

import argparse
import ctypes
import json
from pathlib import (
    Path,
)

import numpy as np
from lammps import (
    lammps,
)
from mpi4py import (
    MPI,
)


def _pair_pointer(lmp, name, pointer_type):
    extract = lmp.lib.lammps_extract_pair
    previous = extract.restype
    extract.restype = pointer_type
    try:
        pointer = extract(lmp.lmp, name.encode())
    finally:
        extract.restype = previous
    assert pointer, f"Pair::extract({name!r}) returned null"
    return pointer


def _snapshot(lmp, virial=False):
    # gather() returns atom-ID order, including contributions reverse-communicated
    # from ghost nodes. All ranks must participate in these collective calls.
    def gather(name, count):
        return np.array(lmp.gather(name, 1, count)).reshape(-1, count).tolist()

    snapshot = {
        "ids": list(lmp.gather_atoms("id", 0, 1)),
        "force": gather("f", 3),
        "magnetic_force": gather("c_fmprop", 3),
        "spin": gather("c_spprop", 4),
        "energy": lmp.get_thermo("pe"),
    }
    if virial:
        # centroid/stress/atom reports the negative per-atom virial in
        # pressure*volume units; dividing by -nktv2p gives eV (units metal).
        scale = -1.0 / lmp.extract_global("nktv2p")
        snapshot["virial"] = (scale * np.array(gather("c_vatom", 9))).tolist()
    return snapshot


def _cmdargs(kokkos):
    cmdargs = ["-log", "none", "-screen", "none"]
    if kokkos:
        cmdargs += [
            "-k",
            "on",
            "g",
            "1",
            "-pk",
            "kokkos",
            "neigh",
            "full",
            "newton",
            "off",
            "comm",
            "host",
            "gpu/aware",
            "off",
        ]
    return cmdargs


def _run(lmp, args, selection, overlay=False):
    comm = MPI.COMM_WORLD
    assert lmp.get_mpi_comm().size == args.ranks
    pair = "dpa4spin/kk" if args.kokkos else "dpa4spin"
    atom = "spin/kk" if args.kokkos else "spin"
    integrator = "verlet/kk" if args.kokkos else "verlet"
    assert lmp.has_style("pair", pair), f"Missing plugin pair style {pair}"
    pair_args = "" if selection == "default" else f" magnetic_force {selection}"
    substyle = f"{pair} {args.model.resolve()}{pair_args}"
    if overlay:
        style = "hybrid/overlay/kk" if args.kokkos else "hybrid/overlay"
        pair_command = f"pair_style {style} {substyle} {substyle}"
        coefficients = [f"pair_coeff * * {pair} {i} Ni O" for i in (1, 2)]
    else:
        style = pair
        pair_command = f"pair_style {substyle}"
        coefficients = ["pair_coeff * * Ni O"]
    for command in (
        "units metal",
        f"processors {comm.size} 1 1",
        f"atom_style {atom}",
        "atom_modify map yes",
        f"read_data {args.data.resolve()}",
        # No global -sf kk: select the matching integrator explicitly too.
        f"run_style {integrator}",
        "mass 1 58.6934",
        "mass 2 15.9994",
        pair_command,
        *coefficients,
        "neighbor 1.0 bin",
        "compute fmprop all property/atom fmx fmy fmz",
        "compute spprop all property/atom spx spy spz sp",
        "run 0",
    ):
        lmp.command(command)
    assert lmp.extract_pair_dimension("fm_cartesian_force") == 0
    flag = _pair_pointer(lmp, "fm_cartesian_force", ctypes.POINTER(ctypes.c_int))[0]
    result = {
        "convention": flag,
        "pair_style": lmp.extract_global("pair_style"),
        "run_style_active": lmp.eval(f"is_active(run_style,{integrator})"),
        "atom_style": lmp.extract_global("atom_style"),
        "kokkos_active": lmp.extract_setting("kokkos_active"),
        "kokkos_ngpus": lmp.extract_setting("kokkos_ngpus"),
        "newton_pair": lmp.extract_setting("newton_pair"),
        "local_counts": comm.allgather(lmp.extract_setting("nlocal")),
        "ghost_counts": comm.allgather(lmp.extract_setting("nghost")),
        "snapshots": [_snapshot(lmp)],
    }
    assert result["pair_style"] == style
    assert result["atom_style"] == atom
    # Repeated computes expose stale ghost magnetic-force accumulation.
    for _ in range(2):
        lmp.command("run 0")
        result["snapshots"].append(_snapshot(lmp))
    # A per-atom centroid virial switches the graph build and the operator to
    # their source-attributed path; the snapshots above cover the path without
    # it. A compute defined between runs is scheduled by every later run.
    lmp.command("compute vatom all centroid/stress/atom NULL pair")
    result["virial_snapshots"] = []
    for _ in range(2):
        lmp.command("run 0")
        result["virial_snapshots"].append(_snapshot(lmp, virial=True))
    if not overlay:
        scale = _pair_pointer(
            lmp, "scale", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))
        )
        scale[1][1] = 2.75
        lmp.command("run 0")
        result["scaled"] = _snapshot(lmp, virial=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--kokkos", action="store_true")
    parser.add_argument("--ranks", type=int, default=1)
    args = parser.parse_args()
    assert MPI.COMM_WORLD.size == args.ranks, (
        f"Expected {args.ranks} MPI ranks, got {MPI.COMM_WORLD.size}: "
        f"{MPI.Get_library_version()}"
    )
    results = {}
    library = None
    try:
        for selection, overlay in (
            ("default", False),
            ("precession", False),
            ("cartesian", False),
            ("precession", True),
            ("cartesian", True),
        ):
            with lammps(cmdargs=_cmdargs(args.kokkos), comm=MPI.COMM_WORLD) as lmp:
                library = lmp.lib
                key = f"overlay_{selection}" if overlay else selection
                results[key] = _run(lmp, args, selection, overlay)
        if MPI.COMM_WORLD.rank == 0:
            args.output.write_text(json.dumps(results))
    finally:
        # close() destroys each instance but keeps Kokkos initialized for later
        # instances. Finalize once, even on error, before mpi4py finalizes MPI.
        if args.kokkos and library is not None:
            library.lammps_kokkos_finalize()


if __name__ == "__main__":
    main()
