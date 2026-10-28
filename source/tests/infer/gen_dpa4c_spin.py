#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate the native-spin DPA4C graph-route .pt2 test model.

Two classes serve the native spin scheme, and the archive decides which:
``NativeSpinPTExpt`` takes those that carry no nested with-comm artifact,
everything else stays with ``DeepSpinPTExpt``. DPA4's graph lower exchanges
per-layer ghost features across ranks and so always ships that artifact,
which leaves DPA4C -- whose compact descriptor keeps its messaging local --
as the family that reaches ``NativeSpinPTExpt``. Without a fixture from this
family the class has no end-to-end C++ coverage at all.

The model carries frame and atomic parameters and a charge state on top of
native spin, so that one archive reaches every input the standalone entry
point has to divide among frames: coordinates and spins, which the caller
supplies per frame; a cell per frame; parameters, which the caller may
either supply per frame or supply once for every frame to reuse; and a
charge state, which the caller may supply per frame or once.

Everything is fp64 and small enough to compile on CPU, as the C++ suite has
no GPU.
"""

import copy
import json
import os
import sys
import zipfile

import numpy as np

# The gen scripts run standalone, outside pytest's package machinery, so the
# relative imports the test suite uses do not apply here.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dpa4_fixtures import (
    jitter_zero_arrays,
)
from gen_common import (
    ensure_inductor_compiler,
    load_custom_ops,
    write_expected_ref,
)

# Small fp64 DPA4C native-spin config. ``scheme="native"`` selects the
# native spin model, whose spin rides the NeighborGraph lower rather than
# the virtual atoms of the ``spin_ener`` scheme.
NATIVE_SPIN_CONFIG = {
    "type_map": ["Ni", "O"],
    "descriptor": {
        "type": "dpa4c",
        "rcut": 4.0,
        "channels": 8,
        "lmax": 2,
        "n_radial": 4,
        "precision": "float64",
        "seed": 7,
        "add_chg_spin_ebd": True,
        "default_chg_spin": [0.0, 1.0],
    },
    "fitting_net": {
        "neuron": [8, 8],
        "precision": "float64",
        "seed": 7,
        "numb_fparam": 2,
        "numb_aparam": 1,
    },
    "spin": {"use_spin": [True, False], "scheme": "native"},
}

# Jitters the zero-initialized residual projections away from exact zero;
# without it the descriptor is edge independent and ``force_mag`` vanishes
# identically, which would make every reference below vacuous.
_JITTER_SEED = 20260817

# Fixed 6-atom system: 3 Ni carry a magnetic moment, 3 O do not. Spin is
# deliberately not pre-masked by type, so that the model's own gating is
# what zeroes the non-magnetic rows.
_NATOMS = 6
_ATYPES = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
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
_FPARAM = np.array([[0.3, -0.2]], dtype=np.float64)
_APARAM = np.array([[0.1, -0.4, 0.7, -0.3, 0.5, 0.2]], dtype=np.float64).reshape(
    1, _NATOMS, 1
)

# A second frame the C++ suite evaluates alongside the first, to pin that
# the frames are answered independently rather than from the first alone.
_COORDS_ALT = np.array(
    [
        [1.2, 0.9, 1.3],
        [3.0, 1.6, 0.9],
        [1.5, 2.0, 1.2],
        [0.6, 1.0, 1.4],
        [3.4, 2.2, 1.5],
        [3.6, 0.5, 1.9],
    ],
    dtype=np.float64,
).reshape(1, _NATOMS, 3)

# The state the archive is frozen with, and one the caller asks for instead.
_DEFAULT_CHG_SPIN = np.array([[0.0, 1.0]], dtype=np.float64)
_OTHER_CHG_SPIN = np.array([[1.0, 2.0]], dtype=np.float64)


def _build_model_dict() -> dict:
    """Build the native-spin dpmodel from config and seed, jittered in place."""
    from deepmd.dpmodel.model.model import (
        get_model,
    )

    model = get_model(copy.deepcopy(NATIVE_SPIN_CONFIG))
    return jitter_zero_arrays(model.serialize(), np.random.default_rng(_JITTER_SEED))


def _check_metadata(pt2_path: str) -> None:
    """Assert the archive routes to ``NativeSpinPTExpt`` and carries its inputs."""
    with zipfile.ZipFile(pt2_path) as zf:
        md = json.loads(zf.read("model/extra/metadata.json").decode("utf-8"))
    print("\n// metadata:")  # noqa: T201
    print(  # noqa: T201
        json.dumps(
            {
                k: md.get(k)
                for k in (
                    "type_map",
                    "lower_input_kind",
                    "spin_scheme",
                    "has_comm_artifact",
                    "dim_fparam",
                    "dim_aparam",
                    "dim_chg_spin",
                    "use_spin",
                )
            },
            indent=2,
        )
    )
    assert md["spin_scheme"] == "native"
    # The dispatch in DeepPotPTExptPlugin.cc reads exactly these two fields,
    # and a with-comm artifact would divert the archive to DeepSpinPTExpt,
    # leaving NativeSpinPTExpt untested by everything built on this fixture.
    assert md.get("has_comm_artifact") is False, (
        "this archive must carry no with-comm artifact, or it is served by "
        "DeepSpinPTExpt and the fixture covers the wrong class"
    )
    assert md["lower_input_kind"] == "graph"
    assert md["dim_fparam"] == 2
    assert md["dim_aparam"] == 1
    assert md["dim_chg_spin"] == 2
    assert md["use_spin"] == [True, False]
    for key in ("atom_energy", "energy", "force", "force_mag", "virial"):
        assert key in md["output_keys"]


def _evaluate(dp, coords, charge_spin):
    """Evaluate one frame and return its energy, force, magnetic force, virial."""
    e, f, v, ae, av, fm, _mm = dp.eval(
        coords,
        _CELL,
        _ATYPES,
        atomic=True,
        spin=_SPINS,
        fparam=_FPARAM,
        aparam=_APARAM,
        charge_spin=charge_spin,
    )
    return e, f, v, ae, av, fm


def _write_ref(pt2_path: str, ref_path: str) -> None:
    """Evaluate the reference cases and write the C++ sidecar."""
    from deepmd.infer import (
        DeepPot,
    )

    dp = DeepPot(pt2_path)
    assert dp.has_spin

    cases = {
        "frame0": (_COORDS, _DEFAULT_CHG_SPIN),
        "frame1": (_COORDS_ALT, _DEFAULT_CHG_SPIN),
        "frame0_other_state": (_COORDS, _OTHER_CHG_SPIN),
    }
    sections = {}
    energies = {}
    spin_mask = _ATYPES == 0
    for name, (coords, charge_spin) in cases.items():
        e, f, v, ae, av, fm = _evaluate(dp, coords, charge_spin)
        energies[name] = float(e[0, 0])
        print(f"// {name} total energy: {e[0, 0]:.18e}")  # noqa: T201

        fm_flat = fm.reshape(_NATOMS, 3)
        fm_spin_max = float(np.max(np.abs(fm_flat[spin_mask])))
        fm_nospin_max = float(np.max(np.abs(fm_flat[~spin_mask])))
        assert np.all(np.isfinite(e)) and np.all(np.isfinite(f))
        # A zero-initialized descriptor yields a vanishing force_mag even on
        # the magnetic atoms, which would make the whole fixture vacuous.
        assert fm_spin_max > 1e-6, (
            f"{name}: force_mag vanishes on the magnetic (Ni) atoms "
            f"({fm_spin_max:.3e}); the jitter is not effective"
        )
        # The model's own gating must zero the non-magnetic rows exactly.
        assert fm_nospin_max == 0.0, (
            f"{name}: force_mag is {fm_nospin_max:.3e} on the non-magnetic "
            f"(O) atoms, where it must be exactly zero"
        )
        sections[name] = {
            "expected_e": ae[0, :, 0],
            "expected_f": f[0],
            "expected_fm": fm[0],
            "expected_tot_v": v[0],
            "expected_atom_v": av[0],
        }

    # Anti-vacuity: the C++ regressions compare frames against each other and
    # states against each other, and both comparisons need a real difference.
    assert abs(energies["frame1"] - energies["frame0"]) > 1e-6, (
        "the two frames must differ, or a multi-frame call answered from the "
        "first frame alone would still match"
    )
    assert abs(energies["frame0_other_state"] - energies["frame0"]) > 1e-6, (
        "the two charge states must differ, or a call that ignored the "
        "requested state would still match"
    )

    write_expected_ref(
        ref_path,
        sections=sections,
        source_script="source/tests/infer/gen_dpa4c_spin.py",
    )
    print(f"Wrote {ref_path}")  # noqa: T201


def main():
    from deepmd.pt_expt.utils.serialization import (
        deserialize_to_file as pt_expt_deserialize_to_file,
    )

    ensure_inductor_compiler()
    load_custom_ops()

    base_dir = os.path.dirname(__file__)
    pt2_path = os.path.join(base_dir, "deeppot_dpa4c_spin_graph.pt2")

    data = {
        "model": _build_model_dict(),
        "model_def_script": NATIVE_SPIN_CONFIG,
        "backend": "dpmodel",
        "software": "deepmd-kit",
        "version": "3.0.0",
    }

    # Native spin has no dense lower; spin rides the NeighborGraph lower
    # exclusively, which is what "auto" resolves to. Pinned for clarity.
    print(f"Exporting to {pt2_path} (lower_kind='graph') ...")  # noqa: T201
    pt_expt_deserialize_to_file(
        pt2_path, data, do_atomic_virial=True, lower_kind="graph"
    )
    print("Export done.")  # noqa: T201

    _check_metadata(pt2_path)
    _write_ref(pt2_path, os.path.join(base_dir, "deeppot_dpa4c_spin_graph.expected"))

    print("\nDone!")  # noqa: T201


if __name__ == "__main__":
    main()
