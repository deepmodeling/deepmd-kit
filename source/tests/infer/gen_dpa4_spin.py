#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate deeppot_dpa4_spin_graph.pt2 test model (native-spin DPA4, graph route).

Mirrors ``gen_dpa4.py``'s Section B pattern: the dpmodel is built in-process
from the inline ``NATIVE_SPIN_CONFIG`` below with a fixed weight-init seed,
its zero-initialized residual projections are jittered away from exact zero
with a fixed RNG seed (``jitter_zero_arrays``, imported from
``source/tests/dpa4_fixtures.py``), and the result is frozen directly to the
graph-kind ``.pt2`` -- no intermediate ``.yaml`` is read or written. Both
``get_model``/weight-init and ``np.random.default_rng(seed)`` are
deterministic, so this reproduces byte-identical weights on every machine/CI
run without committing a serialized-weights file to git.

The native spin scheme has NO dense/nlist lower at all, spin rides the
NeighborGraph lower exclusively (see
``deepmd/pt_expt/model/dpa4_native_spin_model.py``'s module docstring and
``source/tests/pt_expt/model/test_dpa4_export.py`` Task 6). Without the
jitter, a freshly built DPA4 collapses to a type-embedding-only descriptor
(see ``jitter_zero_arrays``'s docstring): every force AND force_mag would be
identically zero regardless of geometry/spin, making this fixture vacuous
for the C++/LAMMPS consumers (Tasks 9/10).

Also writes a sidecar ``.expected`` reference file (PBC and NoPbc per-atom
energy/force/force_mag/virial) consumed by the C++ tests, mirroring
``gen_spin.py``'s field convention.
"""

import copy
import json
import os
import sys
import zipfile

import numpy as np

# Ensure the source tree is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# Ensure source/tests is on the path for dpa4_fixtures (this script runs
# standalone, outside pytest's package machinery, so the usual `from
# ...dpa4_fixtures import ...` relative import used by the test suite does
# not apply here).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dpa4_fixtures import (
    jitter_zero_arrays,
)
from gen_common import (
    ensure_inductor_compiler,
    load_custom_ops,
    write_expected_ref,
)

# Small fp64 DPA4/SeZM native-spin config. Mirrors ``NATIVE_SPIN_CONFIG`` in
# source/tests/common/dpmodel/test_dpa4_native_spin_model.py and
# source/tests/pt_expt/model/test_dpa4_native_spin.py (the config exercised
# by Tasks 1-7's dpmodel/pt_expt/export tests): channels 16, n_radial 8,
# lmax 2, mmax 1, n_blocks 2 -- large enough to exercise the SO(2)/SO(3) +
# attention + spin-embedding paths, small enough to keep the AOTInductor
# compile bounded. ``use_spin=[True, False]``: type 0 ("Ni") carries a
# magnetic moment, type 1 ("O") does not. ``scheme="native"`` selects the
# NeighborGraph-only spin route (no virtual atoms, unlike the deepspin
# ``spin_ener`` scheme used by ``gen_spin.py``).
NATIVE_SPIN_CONFIG = {
    "type_map": ["Ni", "O"],
    "descriptor": {
        "type": "dpa4",
        "rcut": 4.0,
        "sel": 8,
        "channels": 16,
        "n_radial": 8,
        "lmax": 2,
        "mmax": 1,
        "n_blocks": 2,
        "precision": "float64",
        "seed": 7,
        "random_gamma": False,
    },
    "fitting_net": {
        "type": "dpa4_ener",
        "neuron": [8, 8],
        "precision": "float64",
        "seed": 7,
    },
    "spin": {"use_spin": [True, False], "scheme": "native"},
}

# Fixed seed for jittering the zero-initialized residual projections away
# from exact zero (see ``jitter_zero_arrays``'s docstring and module
# docstring above). Kept as the value the fixture was originally generated
# with, for continuity of the fixed 6-atom reference numbers below.
_JITTER_SEED = 20260720


def _build_model_dict() -> dict:
    """Build the native-spin dpmodel from config+seed and jitter in place."""
    from deepmd.dpmodel.model.model import (
        get_model,
    )

    model = get_model(copy.deepcopy(NATIVE_SPIN_CONFIG))
    model_dict = model.serialize()
    jitter_zero_arrays(model_dict, np.random.default_rng(_JITTER_SEED))
    return model_dict


# Fixed 6-atom system (3 Ni, spin-active; 3 O, non-magnetic). Coordinates
# and spins reused verbatim from
# source/tests/pt_expt/model/test_dpa4_export.py's
# ``_SPIN_EVAL_{COORDS,CELL,SPINS}`` / ``_SPIN_EVAL_ATYPES`` -- a system
# already validated (Task 7) to yield a non-degenerate ``force_mag`` with
# this same architecture (rcut=4.0, sel=8). Spin is deliberately NOT
# pre-masked by type: the model's own descriptor gating must zero the
# non-spin (type 1 / O) rows internally.
_NATOMS = 6
_ATYPES = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)  # Ni, Ni, Ni, O, O, O
_COORDS = np.array(
    [
        [1.0, 1.0, 1.0],
        [3.2, 1.4, 1.1],
        [1.3, 1.8, 1.0],
        [0.4, 1.2, 1.6],
        [3.6, 2.0, 1.3],
        [3.4, 0.7, 1.7],
    ],
    dtype=np.float64,
).reshape(1, _NATOMS, 3)
_CELL = (np.eye(3, dtype=np.float64) * 6.0).reshape(1, 9)
_SPINS = np.array(
    [
        [0.11, 0.05, -0.02],
        [-0.07, 0.09, 0.03],
        [0.02, -0.06, 0.08],
        [0.01, -0.01, 0.02],
        [-0.02, 0.03, -0.01],
        [0.015, 0.02, -0.03],
    ],
    dtype=np.float64,
).reshape(1, _NATOMS, 3)


def main():
    from deepmd.infer import (
        DeepPot,
    )
    from deepmd.pt_expt.utils.serialization import (
        deserialize_to_file as pt_expt_deserialize_to_file,
    )

    ensure_inductor_compiler()
    load_custom_ops()

    base_dir = os.path.dirname(__file__)
    pt2_path = os.path.join(base_dir, "deeppot_dpa4_spin_graph.pt2")

    # ---- 1. Build the jittered dpmodel dict from config+seed ----
    model_dict = _build_model_dict()
    data = {
        "model": model_dict,
        "model_def_script": NATIVE_SPIN_CONFIG,
        "backend": "dpmodel",
        "software": "deepmd-kit",
        "version": "3.0.0",
    }

    # ---- 2. Freeze directly to graph-kind .pt2 ----
    # Native-spin DPA4 has NO dense/nlist lower at all (spin rides the
    # NeighborGraph lower exclusively -- see the module docstring above), so
    # ``lower_kind="auto"`` resolves to "graph" for this model
    # (``_resolve_lower_kind``); the virtual-atom ``spin_ener`` scheme would
    # instead hard-stop at "nlist". Pinned explicitly here for clarity.
    print(f"Exporting to {pt2_path} (lower_kind='graph') ...")  # noqa: T201
    pt_expt_deserialize_to_file(
        pt2_path, data, do_atomic_virial=True, lower_kind="graph"
    )
    print("Export done.")  # noqa: T201

    # ---- 3. Sanity-check the frozen archive's metadata ----
    with zipfile.ZipFile(pt2_path) as zf:
        md = json.loads(zf.read("model/extra/metadata.json").decode("utf-8"))
    print("\n// metadata:")  # noqa: T201
    print(  # noqa: T201
        json.dumps(
            {
                k: md[k]
                for k in (
                    "type_map",
                    "lower_input_kind",
                    "is_spin",
                    "has_comm_artifact",
                    "has_message_passing",
                    "ntypes_spin",
                    "use_spin",
                    "output_keys",
                )
                if k in md
            },
            indent=2,
        )
    )
    assert md["type_map"] == NATIVE_SPIN_CONFIG["type_map"]
    assert md["lower_input_kind"] == "graph", (
        f"expected native-spin DPA4 to freeze to the graph lower, got "
        f"{md.get('lower_input_kind')!r}"
    )
    assert md["is_spin"] is True
    assert md["has_comm_artifact"] is False
    assert md["has_message_passing"] is True
    assert md["use_spin"] == [True, False]
    for key in ("atom_energy", "energy", "force", "force_mag", "virial"):
        assert key in md["output_keys"]

    # ---- 4. Run inference (PBC) ----
    dp = DeepPot(pt2_path)
    assert dp.has_spin

    e1, f1, v1, ae1, av1, fm1, _mm1 = dp.eval(
        _COORDS, _CELL, _ATYPES, atomic=True, spin=_SPINS
    )
    print(f"\n// PBC total energy: {e1[0, 0]:.18e}")  # noqa: T201

    # ---- 5. Run inference (NoPbc) ----
    e_np, f_np, v_np, ae_np, av_np, fm_np, _mm_np = dp.eval(
        _COORDS, None, _ATYPES, atomic=True, spin=_SPINS
    )
    print(f"\n// NoPbc total energy: {e_np[0, 0]:.18e}")  # noqa: T201

    # ---- 6. Sanity checks ----
    spin_mask = _ATYPES == 0  # Ni carries spin; O does not
    for label, e, f, fm in (
        ("PBC", e1, f1, fm1),
        ("NoPbc", e_np, f_np, fm_np),
    ):
        assert np.all(np.isfinite(e)), f"{label}: non-finite energy"
        assert np.all(np.isfinite(f)), f"{label}: non-finite force"
        assert np.all(np.isfinite(fm)), f"{label}: non-finite force_mag"

        fm_flat = fm.reshape(_NATOMS, 3)
        fm_spin_max = float(np.max(np.abs(fm_flat[spin_mask])))
        fm_nospin_max = float(np.max(np.abs(fm_flat[~spin_mask])))
        print(  # noqa: T201
            f"// {label} max |force_mag| on spin atoms:    {fm_spin_max:.6e}"
        )
        print(  # noqa: T201
            f"// {label} max |force_mag| on non-spin atoms: {fm_nospin_max:.6e}"
        )
        # Anti-vacuity: a fresh (non-jittered) DPA4 zero-initializes its
        # residual projections, making force_mag identically zero on the
        # spin-carrying atoms too -- would silently produce a degenerate
        # fixture (see the jitter docstring above).
        assert fm_spin_max > 1e-6, (
            f"{label}: expected non-trivial force_mag on spin-active (Ni) "
            f"atoms; got max |force_mag| = {fm_spin_max:.3e} (jitter not "
            f"effective -- this fixture would be vacuous)."
        )
        # The non-spin (O) rows must be exactly gated to zero by the
        # model's own type mask -- not merely small.
        assert fm_nospin_max == 0.0, (
            f"{label}: expected force_mag to be EXACTLY zero on non-spin "
            f"(O) atoms; got max |force_mag| = {fm_nospin_max:.3e}."
        )

    # ---- 7. Write sidecar reference file consumed by C++ tests ----
    ref_path = os.path.join(base_dir, "deeppot_dpa4_spin_graph.expected")
    write_expected_ref(
        ref_path,
        sections={
            "pbc": {
                "expected_e": ae1[0, :, 0],
                "expected_f": f1[0],
                "expected_fm": fm1[0],
                "expected_tot_v": v1[0],
                "expected_atom_v": av1[0],
            },
            "nopbc": {
                "expected_e": ae_np[0, :, 0],
                "expected_f": f_np[0],
                "expected_fm": fm_np[0],
                "expected_tot_v": v_np[0],
                "expected_atom_v": av_np[0],
            },
        },
        source_script="source/tests/infer/gen_dpa4_spin.py",
    )
    print(f"Wrote {ref_path}")  # noqa: T201

    print("\nDone!")  # noqa: T201


if __name__ == "__main__":
    main()
