#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate the COMBINED native-spin + ZBL-bridging DPA4 .pt2 fixture.

One archive, ``deeppot_dpa4_spin_zbl_graph.pt2``: a native-spin
(``scheme="native"``) DPA4 that is ALSO bridged (``bridging_method: "ZBL"``),
i.e. a ``LinearEnergyModel`` over ``[learned DPA4, InterPotentialAtomicModel]``
with ``weights="sum"`` wrapped by the native-spin model class.

Why this fixture exists
-----------------------
Native spin and analytical bridging each had coverage down to C++/LAMMPS
(``gen_dpa4_spin.py`` / ``gen_dpa4_zbl.py``), but their COMBINATION had none
below the pt_expt Python layer: its only export-seam test
(``source/tests/pt_expt/model/test_zbl_bridging.py::
test_native_spin_with_bridging_graph_freeze_and_deep_eval``) is
``pytest.skip``ped when ``CI=true``, so CI validated the combination through
eager Python alone.  The combination is not the conjunction of the two
covered paths: the spin route feeds ``spin`` into a COMPOSITION (the learned
child consumes it, the analytical child accepts and ignores it), and the
freeze must carry ``is_spin`` metadata for a model whose top-level class is
the linear composition -- neither single-feature fixture exercises that.

Single-rank only, and this fixture PINS that limitation
-------------------------------------------------------
Bridging enables the descriptor's Source Freeze Propagation Gate, whose
per-node ``eta_j = prod_{e: src_e = j} w_e`` folds a node's FULL outgoing-edge
set.  Edges exist only for owned centres, so eta is incomplete on every rank
and no with-comm artifact may be exported.  ``_check_metadata`` asserts BOTH
``has_comm_artifact is False`` AND that the nested
``model/extra/forward_lower_with_comm.pt2`` entry is absent (mirroring
``gen_dpa4_zbl.py``) -- an artifact appearing there would silently promise a
multi-rank capability the model cannot honour.  Note this is the opposite of
the UNbridged native-spin fixture (``gen_dpa4_spin.py``), which does carry
the with-comm twin: bridging is what removes it.

Generation mirrors ``gen_dpa4_spin_chgspin.py`` (the closest precedent): the
dpmodel is built in-process from ``NATIVE_SPIN_CONFIG`` imported from
``gen_dpa4_spin.py`` -- this fixture is THAT model plus the three bridging
keys, and nothing else -- with a fixed weight-init seed, its zero-initialized
residual projections are jittered away from exact zero with a fixed RNG seed
(``jitter_zero_arrays``), and the result is frozen directly to the graph-kind
``.pt2``.  Both ``get_model``/weight-init and ``np.random.default_rng(seed)``
are deterministic, so this reproduces byte-identical weights on every
machine/CI run without committing a serialized-weights file to git.

Without the jitter a freshly built DPA4 collapses to a type-embedding-only
descriptor (see ``jitter_zero_arrays``'s docstring): force and force_mag
would be identically zero regardless of geometry/spin, so the LEARNED half of
the composition would contribute nothing and the fixture would only ever test
the analytical term.

The sidecar ``deeppot_dpa4_spin_zbl_graph.expected`` carries the usual ``pbc``
/ ``nopbc`` sections (per-atom energy, force, force_mag, total and atomic
virial).  Consumed by
``source/api_cc/tests/test_deepspin_dpa4_zbl_ptexpt.cc``.
"""

import copy
import json
import math
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

# The base native-spin DPA4 config, shared verbatim with the sibling
# native-spin fixture: this archive is THAT model plus the bridging keys, so
# importing (rather than re-typing) the config keeps the single difference
# between the two fixtures visible in the three lines below.
from gen_dpa4_spin import (
    NATIVE_SPIN_CONFIG,
)

# Bridging radii, identical to gen_dpa4_zbl.py's: they feed the descriptor's
# InnerClamp AND BridgingSwitch (built together from the same radii) on the
# LEARNED child, and are what makes the composition single-rank only.
_BRIDGING_R_INNER = 0.8
_BRIDGING_R_OUTER = 1.2

SPIN_ZBL_CONFIG = copy.deepcopy(NATIVE_SPIN_CONFIG)
SPIN_ZBL_CONFIG["bridging_method"] = "ZBL"
SPIN_ZBL_CONFIG["bridging_r_inner"] = _BRIDGING_R_INNER
SPIN_ZBL_CONFIG["bridging_r_outer"] = _BRIDGING_R_OUTER

# Fixed seed for jittering the zero-initialized residual projections away from
# exact zero (see ``jitter_zero_arrays``'s docstring and the module docstring
# above). Deliberately NOT gen_dpa4_spin.py's nor gen_dpa4_spin_chgspin.py's
# seed: this is a different model (the composition adds a second child, so the
# traversal differs), and sharing a seed would only suggest a weight
# relationship that does not exist.
_JITTER_SEED = 20260727

# Fixed 6-atom system (3 Ni, spin-active; 3 O, non-magnetic).  Coordinates and
# cell verbatim from gen_dpa4_zbl.py -- atoms 0 and 1 sit 0.9 A apart, inside
# ``bridging_r_outer``, so the analytical ZBL term contributes a large,
# unmistakable repulsion instead of a numerical afterthought (and BOTH members
# of that close pair are spin-carrying Ni, so the ZBL and spin channels act on
# the same atoms).  Spins verbatim from gen_dpa4_spin.py.  Spin is deliberately
# NOT pre-masked by type: the model's own descriptor gating must zero the
# non-spin (type 1 / O) rows internally.
_NATOMS = 6
_ATYPES = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)  # Ni, Ni, Ni, O, O, O
_COORDS = np.array(
    [
        [1.0, 1.0, 1.0],
        [1.9, 1.0, 1.0],
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

# ZBL screening parameters, repeated here as an INDEPENDENT reference (they
# mirror deepmd/dpmodel/atomic_model/inter_potential.py, deliberately not
# imported from it: a reference that shares its constants with the code under
# test cannot catch a wrong constant).
_ZBL_A_COEFF = (0.18175, 0.50986, 0.28022, 0.028171)
_ZBL_B_COEFF = (3.1998, 0.94229, 0.4029, 0.20162)
_KE_EV_A = 14.3996
_A_BOHR = 0.5291772109
_Z_OF_TYPE = {0: 28.0, 1: 8.0}  # type_map ["Ni", "O"]


def _analytic_zbl_total(coord: np.ndarray, atype: np.ndarray, rcut: float) -> float:
    """Gas-phase ZBL total energy: a direct double loop over pairs < rcut.

    Parameters
    ----------
    coord : np.ndarray
        (natoms, 3) coordinates in Angstrom.  No periodic images are
        considered, so this is only a complete reference for a gas-phase
        (no-box) evaluation.
    atype : np.ndarray
        (natoms,) atom types, indexing ``_Z_OF_TYPE``.
    rcut : float
        Cutoff radius; pairs at or beyond it contribute nothing.

    Returns
    -------
    float
        The summed ZBL pair energy in eV.
    """
    zs = [_Z_OF_TYPE[int(t)] for t in atype]
    total = 0.0
    natoms = len(zs)
    for ii in range(natoms):
        for jj in range(ii + 1, natoms):
            r = float(np.linalg.norm(coord[ii] - coord[jj]))
            if r >= rcut:
                continue
            a_screen = 0.88534 * _A_BOHR / (zs[ii] ** 0.23 + zs[jj] ** 0.23)
            phi = sum(
                a_k * math.exp(-b_k * (r / a_screen))
                for a_k, b_k in zip(_ZBL_A_COEFF, _ZBL_B_COEFF, strict=True)
            )
            total += _KE_EV_A * zs[ii] * zs[jj] / r * phi
    return total


def _build_model_dict() -> dict:
    """Build the combined native-spin + bridged dpmodel and jitter it.

    Returns
    -------
    dict
        The jittered serialized model tree, ready to be frozen.
    """
    from deepmd.dpmodel.model.model import (
        get_model,
    )

    model = get_model(copy.deepcopy(SPIN_ZBL_CONFIG))
    assert model.has_spin() is True
    kinds = [type(child).__name__ for child in model.atomic_model.models]
    assert kinds[1] == "InterPotentialAtomicModel", (
        f"expected the bridged composition's second child to be the "
        f"analytical InterPotentialAtomicModel, got {kinds!r}; without it "
        f"this fixture is just the plain native-spin model again."
    )
    model_dict = model.serialize()
    model_dict = jitter_zero_arrays(model_dict, np.random.default_rng(_JITTER_SEED))
    return model_dict


def _assert_zbl_term_is_active(model_dict: dict) -> float:
    """Pin that the analytical term genuinely contributes, and by how much.

    Re-builds the jittered model and its LEARNED child alone (the same model
    minus the analytical energy), and checks their gas-phase energy difference
    against the independent double-loop ZBL reference above.

    The identity is exact rather than approximate, but it is not simply
    ``delta == ZBL``: ``jitter_zero_arrays`` also perturbs the all-zero
    ``out_bias`` of the analytical child and of the linear composition itself,
    and neither of those two per-type constants exists in the learned-child-
    only model.  Both are read straight out of the serialized tree and added
    to the reference, so the check stays a bit-level identity (1e-10) instead
    of a hand-tuned tolerance.

    Gas phase (no box) is used because the double-loop reference has no
    periodic images; with this fixture's 6 A cell and 4 A cutoff the PBC case
    would additionally involve images and a binding ``sel``.

    Parameters
    ----------
    model_dict : dict
        The jittered serialized model tree.

    Returns
    -------
    float
        The EAGER gas-phase total energy of the bridged model, so the caller
        can hold the frozen archive to it (see ``main``).
    """
    from deepmd.dpmodel.model.base_model import (
        BaseModel,
    )
    from deepmd.dpmodel.model.native_spin_model import (
        NativeSpinEnergyModel,
    )

    model = BaseModel.deserialize(copy.deepcopy(model_dict))
    # Same wrapper class, same learned child, no analytical term: the ONLY
    # difference from ``model`` is the composition's second child.
    learned_only = NativeSpinEnergyModel(
        atomic_model_=model.atomic_model.models[0], spin=model.spin
    )
    # The dpmodel ``call`` API is framed (nf x nloc x ...); DeepPot.eval below
    # takes the flat atype instead, hence the reshape here only.
    atype_framed = _ATYPES.reshape(1, _NATOMS)
    e_bridged = float(
        np.reshape(model.call(_COORDS, atype_framed, _SPINS)["energy"], (-1,))[0]
    )
    e_learned = float(
        np.reshape(learned_only.call(_COORDS, atype_framed, _SPINS)["energy"], (-1,))[0]
    )
    delta = e_bridged - e_learned

    zbl_ref = _analytic_zbl_total(
        _COORDS[0], _ATYPES, float(SPIN_ZBL_CONFIG["descriptor"]["rcut"])
    )
    # The two per-type constant offsets that live only in the composition.
    bias_inter = np.reshape(np.asarray(model.atomic_model.models[1].out_bias), (-1,))
    bias_linear = np.reshape(np.asarray(model.atomic_model.out_bias), (-1,))
    offset = float(np.sum(bias_inter[_ATYPES]) + np.sum(bias_linear[_ATYPES]))

    print(  # noqa: T201
        f"\n// gas-phase bridged energy: {e_bridged:.18e}\n"
        f"// gas-phase learned energy: {e_learned:.18e}\n"
        f"// delta:                    {delta:.18e}\n"
        f"// analytic ZBL + jittered out_bias offset: "
        f"{zbl_ref + offset:.18e}"
    )
    # (a) the analytical term must be LARGE -- it is the whole point of the
    #     fixture that a bridged energy is nowhere near the unbridged one.
    assert abs(delta) > 1e-3, (
        f"bridging left the energy essentially unchanged (delta = {delta:.3e}); "
        f"the analytical ZBL term is not reaching the composition, so this "
        f"fixture would be vacuous."
    )
    # (b) and it must be exactly the ZBL energy (plus the two jitter-induced
    #     constants), not merely "something large".
    np.testing.assert_allclose(
        delta,
        zbl_ref + offset,
        rtol=1e-10,
        atol=1e-10,
        err_msg=(
            "the bridged-minus-learned energy is not the analytical ZBL sum; "
            "the composition's analytical child is computing something else."
        ),
    )
    # (c) the learned half must not be swamped into irrelevance either: a
    #     fixture whose learned energy were zero would not test the DPA4 half.
    assert abs(e_learned) > 1e-6, (
        f"learned-child energy is {e_learned:.3e}; the jitter did not take "
        f"effect and the fixture would only exercise the analytical term."
    )
    return e_bridged


def _check_metadata(pt2_path: str) -> None:
    """Assert the frozen archive's metadata and the with-comm ABSENCE.

    Parameters
    ----------
    pt2_path : str
        Path to the frozen ``.pt2`` archive.
    """
    with zipfile.ZipFile(pt2_path) as zf:
        md = json.loads(zf.read("model/extra/metadata.json").decode("utf-8"))
        names = zf.namelist()
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
                    "use_spin",
                    "output_keys",
                )
                if k in md
            },
            indent=2,
        )
    )
    assert md["type_map"] == SPIN_ZBL_CONFIG["type_map"]
    assert md["lower_input_kind"] == "graph", (
        f"expected native-spin DPA4 to freeze to the graph lower, got "
        f"{md.get('lower_input_kind')!r}"
    )
    # Without is_spin the C++ side would route through DeepPot and the whole
    # DeepSpin regression that consumes this archive would test nothing.
    assert md["is_spin"] is True, (
        f"{pt2_path}: metadata is_spin = {md.get('is_spin')!r}, expected True; "
        f"the composition dropped the spin flag on the way to the freeze."
    )
    assert md["use_spin"] == [True, False]
    # Single-rank only -- see the module docstring (the bridging gate's eta is
    # incomplete per rank).  BOTH halves matter: the flag is what the C++
    # dispatch reads, the archive entry is what it would load.
    assert md["has_comm_artifact"] is False, (
        f"{pt2_path}: metadata has_comm_artifact = "
        f"{md.get('has_comm_artifact')!r}, expected False; a bridged model "
        f"cannot support multi-rank message passing (its Source Freeze "
        f"Propagation Gate folds each node's full outgoing-edge set, which no "
        f"rank owns), so advertising one would promise a capability the model "
        f"cannot honour."
    )
    assert "model/extra/forward_lower_with_comm.pt2" not in names, (
        f"{pt2_path}: a nested forward_lower_with_comm.pt2 was exported for a "
        f"bridged model; see above -- the archive must not carry one."
    )
    # The descriptor still message-passes WITHIN a rank; it is only the
    # cross-rank exchange that bridging forbids.  This is the flag that makes
    # the C++ side fail fast under mpirun instead of answering wrongly.
    assert md["has_message_passing"] is True
    for key in ("atom_energy", "energy", "force", "force_mag", "virial"):
        assert key in md["output_keys"]


def _eval_and_check(dp, cell, label: str) -> dict:
    """Evaluate one (PBC or NoPbc) case and run its anti-vacuity checks.

    Parameters
    ----------
    dp : deepmd.infer.DeepPot
        The loaded archive.
    cell : np.ndarray or None
        The simulation cell, or ``None`` for a gas-phase evaluation.
    label : str
        Human-readable case name, used in messages.

    Returns
    -------
    dict
        The reference arrays for this case, in ``write_expected_ref``'s
        field convention.
    """
    e, f, v, ae, av, fm, _mm = dp.eval(_COORDS, cell, _ATYPES, atomic=True, spin=_SPINS)
    print(f"\n// {label} total energy: {e[0, 0]:.18e}")  # noqa: T201

    assert np.all(np.isfinite(e)), f"{label}: non-finite energy"
    assert np.all(np.isfinite(f)), f"{label}: non-finite force"
    assert np.all(np.isfinite(fm)), f"{label}: non-finite force_mag"

    fmax = float(np.max(np.abs(f)))
    print(f"//   max |force|: {fmax:.6e}")  # noqa: T201
    # Anti-vacuity: the 0.9 A Ni-Ni pair drives a large analytical ZBL
    # repulsion, so a small max-force means the term (or the jitter) is not
    # reaching the forward.
    assert fmax > 1e-3, (
        f"{label}: expected a non-trivial force (the 0.9 A Ni-Ni pair drives "
        f"the analytical ZBL term); got {fmax:.3e}."
    )

    spin_mask = _ATYPES == 0  # Ni carries spin; O does not
    fm_flat = fm.reshape(_NATOMS, 3)
    fm_spin_max = float(np.max(np.abs(fm_flat[spin_mask])))
    fm_nospin_max = float(np.max(np.abs(fm_flat[~spin_mask])))
    print(  # noqa: T201
        f"//   max |force_mag| spin / non-spin atoms: "
        f"{fm_spin_max:.6e} / {fm_nospin_max:.6e}"
    )
    # Anti-vacuity: a fresh (non-jittered) DPA4 zero-initializes its residual
    # projections, making force_mag identically zero on the spin-carrying
    # atoms too (see the jitter docstring in the module header).
    assert fm_spin_max > 1e-6, (
        f"{label}: expected non-trivial force_mag on spin-active (Ni) atoms; "
        f"got max |force_mag| = {fm_spin_max:.3e} (jitter not effective -- "
        f"this fixture would be vacuous)."
    )
    # The non-spin (O) rows must be EXACTLY gated to zero by the model's own
    # type mask -- not merely small.  The analytical child, which knows
    # nothing about spin, must not leak into this channel either.
    assert fm_nospin_max == 0.0, (
        f"{label}: expected force_mag to be EXACTLY zero on non-spin (O) "
        f"atoms; got max |force_mag| = {fm_nospin_max:.3e}."
    )
    return {
        "expected_e": ae[0, :, 0],
        "expected_f": f[0],
        "expected_fm": fm[0],
        "expected_tot_v": v[0],
        "expected_atom_v": av[0],
    }


def main():
    from deepmd.pt_expt.utils.serialization import (
        deserialize_to_file as pt_expt_deserialize_to_file,
    )

    ensure_inductor_compiler()
    load_custom_ops()

    base_dir = os.path.dirname(__file__)
    pt2_path = os.path.join(base_dir, "deeppot_dpa4_spin_zbl_graph.pt2")
    ref_path = os.path.join(base_dir, "deeppot_dpa4_spin_zbl_graph.expected")

    # ---- 1. Build the jittered dpmodel dict from config+seed ----
    model_dict = _build_model_dict()

    # ---- 2. Pin that the analytical term is genuinely active (eager) ----
    # Done BEFORE the (slow) inductor compile so a degenerate fixture fails in
    # seconds rather than minutes.
    e_gas_eager = _assert_zbl_term_is_active(model_dict)

    data = {
        "model": model_dict,
        "model_def_script": SPIN_ZBL_CONFIG,
        "backend": "dpmodel",
        "software": "deepmd-kit",
        "version": "3.0.0",
    }

    # ---- 3. Freeze directly to graph-kind .pt2 ----
    # Native-spin DPA4 has NO dense/nlist lower at all (spin rides the
    # NeighborGraph lower exclusively), and the analytical child is likewise
    # graph-route only, so ``lower_kind="auto"`` would resolve to "graph"
    # anyway; pinned explicitly here for clarity.
    print(f"\nExporting to {pt2_path} (lower_kind='graph') ...")  # noqa: T201
    pt_expt_deserialize_to_file(
        pt2_path, data, do_atomic_virial=True, lower_kind="graph"
    )
    print("Export done.")  # noqa: T201
    _check_metadata(pt2_path)

    # ---- 4. Evaluate the two reference cases ----
    from deepmd.infer import (
        DeepPot,
    )

    dp = DeepPot(pt2_path)
    assert dp.has_spin
    pbc = _eval_and_check(dp, _CELL, "PBC")
    nopbc = _eval_and_check(dp, None, "NoPbc")

    # ---- 5. The freeze must preserve the composition ----
    # The checks above only prove the FROZEN archive is self-consistent; this
    # holds it to the eager dpmodel it was built from, so a freeze that
    # silently dropped the analytical child (or the spin injection) is caught
    # here rather than becoming the "reference" every downstream C++/LAMMPS
    # test then agrees with.  Gas phase, to compare against the same eager
    # evaluation the ZBL identity above used.  1e-10 is the project's
    # cross-backend fp64 bound (dpmodel/NumPy vs the compiled torch artifact);
    # the observed gap is ~1e-13 relative.
    e_gas_frozen = float(np.sum(nopbc["expected_e"]))
    np.testing.assert_allclose(
        e_gas_frozen,
        e_gas_eager,
        rtol=1e-10,
        atol=0.0,
        err_msg=(
            "the frozen archive's gas-phase energy disagrees with the eager "
            "dpmodel it was built from; the export dropped part of the "
            "native-spin + bridging composition."
        ),
    )

    # ---- 6. Sidecar reference consumed by the C++ test ----
    write_expected_ref(
        ref_path,
        sections={"pbc": pbc, "nopbc": nopbc},
        source_script="source/tests/infer/gen_dpa4_spin_zbl.py",
    )
    print(f"\nWrote {ref_path}")  # noqa: T201

    print("\nDone!")  # noqa: T201


if __name__ == "__main__":
    main()
