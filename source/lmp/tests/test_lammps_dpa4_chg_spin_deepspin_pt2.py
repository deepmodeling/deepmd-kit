# SPDX-License-Identifier: LGPL-3.0-or-later
"""Test the LAMMPS ``charge_spin`` keyword through ``pair_style deepspin``.

The sibling ``test_lammps_chg_spin_pt2.py`` exercises the keyword through
``pair_style deepmd`` (DPA3, non-spin); the DeepSpin ingestion seam is a
SEPARATE code path (``pair_deepspin.cpp``'s own ``charge_spin`` keyword
parse feeding ``DeepSpin::compute(..., charge_spin)``, see
``source/lmp/pair_deepspin.cpp``), so a regression confined to it was
invisible to that file -- every pre-existing spin fixture has
``dim_chg_spin == 0``, making the whole argument inert.  This file mirrors
the sibling's three-test structure (default run, explicit run, sensitivity)
against the COMBINED native-spin + charge-spin FiLM DPA4 archive
``deeppot_dpa4_spin_chgspin.pt2`` (issue #5906 Task 12b: DPA4-variant
behaviour must align with what the C++ gtest
``source/api_cc/tests/test_deepspin_dpa4_chgspin_ptexpt.cc`` already proves
for the C API -- the LAMMPS pair style is a distinct consumer).

The archive and its stored ``default_chg_spin = [0.0, 1.0]`` come from
``source/tests/infer/gen_dpa4_spin_chgspin.py``; the explicit probe
``charge_spin 1.0 2.0`` matches the generator's ``_EXPLICIT_CHG_SPIN``
(the embedding is CATEGORICAL, so the two probes land on distinct rows in
BOTH components -- see the generator's module docstring).

Reference values are computed LIVE at test-setup time via
``deepmd.infer.DeepPot.eval`` on the archive itself (mirroring
``test_lammps_dpa4_spin_graph_pt2.py``'s ``_compute_expected``, which
explains the reasoning in full) rather than read from the generator's
``.expected`` sidecar: the sidecar's evaluation uses a 6x6x6 A cell whose
edge length equals DPA4's LAMMPS ghost cutoff exactly
(rcut(4.0)+skin(2.0)=6.0), which is not a safe geometry for a periodic
LAMMPS run.  This module keeps the generator's 6-atom NiO geometry and
spins verbatim, in a 13x13x13 A box instead (the same box-swap the ZBL twin
``test_lammps_dpa4_zbl_pt2.py`` applies to its generator geometry).
"""

import json
import os
import subprocess as sp
import sys
import textwrap
from pathlib import (
    Path,
)

import numpy as np
import pytest
from lammps import (
    PyLammps,
)
from write_lmp_data import (
    write_lmp_data_spin,
)

pb_file = (
    Path(__file__).parent.parent.parent
    / "tests"
    / "infer"
    / "deeppot_dpa4_spin_chgspin.pt2"
)
data_file = Path(__file__).parent / "data_dpa4_chg_spin_deepspin_pt2.lmp"

# 6-atom NiO system: coordinates, types and spins verbatim from
# ``gen_dpa4_spin_chgspin.py`` (3 spin-active Ni + 3 O; the O spins are
# deliberately nonzero there -- the model's own descriptor gating must zero
# the non-spin rows internally), in a 13x13x13 A box instead of the
# generator's 6x6x6 (see the module docstring).
box = np.array([0, 13, 0, 13, 0, 13, 0, 0, 0])
coord = np.array(
    [
        [1.0, 1.0, 1.0],
        [3.2, 1.4, 1.1],
        [1.3, 1.8, 1.0],
        [0.4, 1.2, 1.6],
        [3.6, 2.0, 1.3],
        [3.4, 0.7, 1.7],
    ]
)
spin = np.array(
    [
        [0.11, 0.05, -0.02],
        [-0.07, 0.09, 0.03],
        [0.02, -0.06, 0.08],
        [0.01, -0.01, 0.02],
        [-0.02, 0.03, -0.01],
        [0.015, 0.02, -0.03],
    ]
)
# Model ``type_map`` is ["Ni", "O"]; the generator's atype [0,0,0,1,1,1]
# -> LAMMPS types [1,1,1,2,2,2] under identity ``pair_coeff * *``.
type_NiO = np.array([1, 1, 1, 2, 2, 2])

# Explicit runtime probe, matching the generator's ``_EXPLICIT_CHG_SPIN``
# (distinct from the stored default [0.0, 1.0] in BOTH categorical
# components, so neither component alone can explain a response).
_EXPLICIT_CHG_SPIN = [1.0, 2.0]

# LAMMPS's ``fm`` (what ``compute property/atom fmx fmy fmz`` reports) is
# NOT the raw DeepEval force_mag: pair_deepspin.cpp scales it by
# ``spin_norm / hbar`` per atom (metal-units ``hbar = 6.5821191e-04``; same
# convention as test_lammps_dpa4_spin_graph_pt2.py, which documents it).
_HBAR_METAL = 6.5821191e-04

# Reference values (energy / force / force_mag, default and explicit
# charge_spin), populated by ``_compute_expected`` in ``setup_module``.
expected_e_default = None
expected_f_default = None
expected_fm_default = None
expected_e_explicit = None
expected_f_explicit = None
expected_fm_explicit = None


def _cell_from_lammps_box(lmp_box: np.ndarray) -> np.ndarray:
    """Convert a LAMMPS ``xlo xhi ylo yhi zlo zhi xy xz yz`` box spec to a
    flat, row-major 3x3 cell matrix (deepmd's ``box`` convention).
    """
    xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz = lmp_box
    return np.array([xhi - xlo, 0.0, 0.0, xy, yhi - ylo, 0.0, xz, yz, zhi - zlo])


def _compute_expected() -> None:
    """Load ``deeppot_dpa4_spin_chgspin.pt2`` via ``DeepPot`` and evaluate
    the module's fixed 6-atom NiO system, once with NO ``charge_spin``
    (stored default) and once with the explicit probe.

    Runs in a subprocess to avoid importing ``deepmd`` in the LAMMPS test
    process (the LAMMPS plugin already loads ``libdeepmd_op_pt.so`` at the
    C++ level, and importing the Python package on top of that can
    segfault) -- the same precaution as ``test_lammps_dpa4_spin_graph_pt2.py``.
    """
    global expected_e_default, expected_f_default, expected_fm_default
    global expected_e_explicit, expected_f_explicit, expected_fm_explicit

    cell = _cell_from_lammps_box(box)
    atype = (type_NiO - 1).tolist()  # LAMMPS 1-based -> deepmd 0-based (Ni=0, O=1)

    # The archive lives in ``source/tests/infer`` next to ``gen_common.py``,
    # whose ``load_custom_ops()`` loads the build-tree ``libdeepmd_op_pt.so``
    # (registering ``deepmd::edge_force_virial``, which graph ``.pt2``
    # inference needs); ``import deepmd.pt`` alone only loads the op library
    # from SHARED_LIB_DIR, which the build-test env does not populate.
    infer_dir = str(pb_file.resolve().parent)
    script = textwrap.dedent(f"""\
        import json
        import sys
        import numpy as np

        sys.path.insert(0, {infer_dir!r})
        import deepmd.pt  # noqa: F401  (triggers the base op-library load)
        from gen_common import load_custom_ops

        load_custom_ops()
        from deepmd.infer import DeepPot

        dp = DeepPot({str(pb_file.resolve())!r})
        assert dp.deep_eval.get_dim_chg_spin() == 2
        out = {{}}
        for label, chg_spin in (
            ("default", None),
            ("explicit", {_EXPLICIT_CHG_SPIN!r}),
        ):
            kwargs = {{}}
            if chg_spin is not None:
                kwargs["charge_spin"] = np.array([chg_spin], dtype=np.float64)
            e, f, v, ae, av, fm, mm = dp.eval(
                np.array({coord.tolist()!r}).reshape(1, -1, 3),
                np.array({cell.tolist()!r}).reshape(1, 9),
                {atype!r},
                atomic=True,
                spin=np.array({spin.tolist()!r}).reshape(1, -1, 3),
                **kwargs,
            )
            out[label] = {{
                "e": float(e[0, 0]),
                "f": np.asarray(f[0]).tolist(),
                "fm": np.asarray(fm[0]).tolist(),
            }}
        print(json.dumps(out))
    """)
    proc = sp.run([sys.executable, "-c", script], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to compute expected values:\n{proc.stderr}")
    result = json.loads(proc.stdout.strip())

    # Raw DeepEval force_mag (dE/dspin), scaled by LAMMPS's own
    # spin_norm / hbar unit convention (see ``_HBAR_METAL`` above) before
    # comparison.
    spin_norm_scale = (np.linalg.norm(spin, axis=1) / _HBAR_METAL)[:, None]

    expected_e_default = result["default"]["e"]
    expected_f_default = np.array(result["default"]["f"])
    expected_fm_default = np.array(result["default"]["fm"]) * spin_norm_scale
    expected_e_explicit = result["explicit"]["e"]
    expected_f_explicit = np.array(result["explicit"]["f"])
    expected_fm_explicit = np.array(result["explicit"]["fm"]) * spin_norm_scale

    # Anti-vacuity, checked once here so every test below is known to compare
    # against a non-degenerate reference: the explicit probe must MOVE the
    # energy (the generator asserts the same at generation time; re-asserting
    # on THIS geometry keeps the sensitivity test below meaningful), and the
    # charge-spin FiLM must not have killed the spin response.
    assert abs(expected_e_explicit - expected_e_default) > 1e-6, (
        f"charge_spin={_EXPLICIT_CHG_SPIN} left the energy unchanged vs the "
        f"stored default ({expected_e_default:.18e} vs "
        f"{expected_e_explicit:.18e}); the FiLM conditioning is not reaching "
        f"the forward on this geometry, so the sensitivity test is vacuous."
    )
    assert np.max(np.abs(expected_fm_default[:3])) > 1e-6, (
        "expected non-trivial force_mag on the spin-active (Ni) atoms; the "
        "fixture would be vacuous for the spin leaf."
    )


def setup_module() -> None:
    if os.environ.get("ENABLE_PYTORCH", "1") != "1":
        pytest.skip("Skip test because PyTorch support is not enabled.")
    if not pb_file.exists():
        pytest.skip(
            "deeppot_dpa4_spin_chgspin.pt2 not found (run "
            "source/tests/infer/gen_dpa4_spin_chgspin.py)."
        )
    _compute_expected()
    write_lmp_data_spin(box, coord, spin, type_NiO, data_file)


def teardown_module() -> None:
    if data_file.exists():
        os.remove(data_file)


def _lammps(data_file, units="metal") -> PyLammps:
    """Standard DeepSpin LAMMPS system, plus ``atom_modify map yes``.

    Same setup as ``test_lammps_dpa4_spin_graph_pt2.py``: the native-spin
    DPA4 GRAPH ``.pt2`` needs the LAMMPS atom-map to resolve ghost-atom
    indices to local owners for single-rank inference.
    """
    if units != "metal":
        raise ValueError("units for spin should be metal")

    lammps = PyLammps()
    lammps.units(units)
    lammps.boundary("p p p")
    lammps.atom_style("spin")
    lammps.atom_modify("map yes")
    lammps.neighbor("2.0 bin")
    lammps.neigh_modify("every 10 delay 0 check no")
    lammps.read_data(data_file.resolve())
    lammps.mass("1 58")  # Ni
    lammps.mass("2 16")  # O
    lammps.timestep(0.0005)
    lammps.fix("1 all nve")
    return lammps


@pytest.fixture
def lammps():
    lmp = _lammps(data_file=data_file)
    yield lmp
    lmp.close()


def _gather_force_mag(lammps: PyLammps, natoms: int) -> np.ndarray:
    """Extract per-atom force_mag in atom-id order via
    ``compute property/atom fmx fmy fmz`` + ``gather`` (LAMMPS does not
    expose ``fm`` through the legacy ``extract``/``gather_atoms`` registry;
    see ``test_lammps_dpa4_spin_graph_pt2.py``).
    """
    fm_global = lammps.lmp.gather("c_fmprop", 1, 3)
    return np.array(fm_global, dtype=np.float64).reshape(natoms, 3)


def _assert_run0_matches(
    lammps: PyLammps,
    e_ref: float,
    f_ref: np.ndarray,
    fm_ref: np.ndarray,
) -> None:
    """Run 0 steps and compare pe / force / force_mag to the given reference
    (both sides run the SAME compiled artifact, so ``atol=1e-8`` is a
    cross-consumer bound, not a cross-backend one -- same rationale as the
    sibling DPA4 LAMMPS tests).
    """
    natoms = coord.shape[0]
    lammps.compute("fmprop all property/atom fmx fmy fmz")
    lammps.run(0)

    assert lammps.eval("pe") == pytest.approx(e_ref, rel=1e-10)

    forces = np.array(
        [lammps.atoms[ii].force for ii in range(natoms)], dtype=np.float64
    )
    ids = np.array([lammps.atoms[ii].id for ii in range(natoms)])
    np.testing.assert_allclose(forces, f_ref[ids - 1], atol=1e-8, rtol=0)

    force_mag = _gather_force_mag(lammps, natoms)
    np.testing.assert_allclose(force_mag, fm_ref, atol=1e-8, rtol=0)
    # Native-spin design invariant: force_mag on the non-spin (O) atoms must
    # be exactly zero -- the model's own type gating, not the spin values,
    # decides (the O spins in this fixture are deliberately nonzero).
    np.testing.assert_array_equal(force_mag[3:], np.zeros((3, 3)))


def test_pair_deepspin_charge_spin_default(lammps) -> None:
    """No charge_spin keyword -> the model's stored default_chg_spin is used
    (the DeepSpin twin of the backward-compatibility contract the C++ gtest
    pins for an EMPTY runtime charge_spin).
    """
    lammps.pair_style(f"deepspin {pb_file.resolve()}")
    lammps.pair_coeff("* *")
    _assert_run0_matches(
        lammps, expected_e_default, expected_f_default, expected_fm_default
    )
    lammps.run(1)


def test_pair_deepspin_charge_spin_explicit(lammps) -> None:
    """Explicit ``charge_spin`` keyword is parsed by pair_deepspin and
    threaded through DeepSpin to the model (energy, force AND force_mag --
    the spin-only output -- must all follow the explicit conditioning).
    """
    cs = " ".join(str(v) for v in _EXPLICIT_CHG_SPIN)
    lammps.pair_style(f"deepspin {pb_file.resolve()} charge_spin {cs}")
    lammps.pair_coeff("* *")
    _assert_run0_matches(
        lammps, expected_e_explicit, expected_f_explicit, expected_fm_explicit
    )
    lammps.run(1)


def test_charge_spin_changes_result(lammps) -> None:
    """Different charge_spin must give a different energy (keyword takes
    effect through pair_deepspin; ``_compute_expected`` already pinned that
    the two references differ, so this catches the keyword being silently
    dropped on the LAMMPS side).
    """
    cs = " ".join(str(v) for v in _EXPLICIT_CHG_SPIN)
    lammps.pair_style(f"deepspin {pb_file.resolve()} charge_spin {cs}")
    lammps.pair_coeff("* *")
    lammps.run(0)
    assert lammps.eval("pe") != pytest.approx(expected_e_default)
