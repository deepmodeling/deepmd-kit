#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate the COMBINED native-spin + charge-spin FiLM DPA4 .pt2 fixture.

One archive, ``deeppot_dpa4_spin_chgspin.pt2``: a native-spin (``scheme=
"native"``) DPA4 whose descriptor ALSO carries ``add_chg_spin_ebd=True``
(``get_dim_chg_spin() == 2``) and a stored ``default_chg_spin``. That
combination is a supported public configuration (see
``source/tests/pt_expt/model/test_dpa4_native_spin.py``'s
``TestCombinedChargeSpin*``); ``charge_spin`` rides the CONDITIONAL slot-13
tail of the graph-spin ABI, after the native ``spin`` input at slot 10.

Why this fixture exists
-----------------------
The C++/LAMMPS spin inference path grew a runtime ``charge_spin`` argument
(``DeepSpin::compute(..., charge_spin)``, ``DP_DeepSpinCompute3``,
``pair_deepspin``'s keyword). Before this fixture nothing demonstrated that a
runtime ``charge_spin`` actually reaches the model: every existing spin
fixture has ``dim_chg_spin == 0``, so the whole argument was inert and a dead
seam would have been invisible. The sibling non-spin fixture
(``gen_chg_spin.py`` -> ``chg_spin.pt2``, DPA3) covers ``DeepPot`` only; the
``DeepSpin`` ingestion seam is a SEPARATE code path
(``DeepSpinPTExpt::compute``, two overloads) and needs its own model.

Generation mirrors ``gen_dpa4_spin.py`` exactly: the dpmodel is built
in-process from ``NATIVE_SPIN_CONFIG`` (imported from that script -- this
fixture is that model plus the charge-spin FiLM, and nothing else) with a
fixed weight-init seed, its zero-initialized residual projections are
jittered away from exact zero with a fixed RNG seed
(``jitter_zero_arrays``), and the result is frozen directly to the
graph-kind ``.pt2``. Both ``get_model``/weight-init and
``np.random.default_rng(seed)`` are deterministic, so this reproduces
byte-identical weights on every machine/CI run without committing a
serialized-weights file to git.

Without the jitter a freshly built DPA4 collapses to a type-embedding-only
descriptor (see ``jitter_zero_arrays``'s docstring): force and force_mag
would be identically zero regardless of geometry/spin, and -- since the
charge-spin FiLM feeds those same zero-initialized residual projections --
the ``charge_spin`` response would vanish too, making the fixture vacuous
for exactly the property it exists to pin.

The sidecar ``deeppot_dpa4_spin_chgspin.expected`` carries FOUR sections,
PBC and NoPbc x default and explicit ``charge_spin``:

- ``pbc_default`` / ``nopbc_default``   -- eval with NO charge_spin, i.e. the
  model's stored ``default_chg_spin = [0.0, 1.0]``. This is what an EMPTY
  runtime ``charge_spin`` must reproduce in C++ (backward compatibility).
- ``pbc_explicit`` / ``nopbc_explicit`` -- eval with ``charge_spin =
  [1.0, 2.0]``.

``ChargeSpinEmbedding`` is CATEGORICAL (it casts the frame ``(charge, spin)``
pair to int64 lookup indices: ``charge + 100`` and ``spin``), so the two
probes are integer-valued and land on distinct rows: ``[0.0, 1.0]`` ->
(100, 1), ``[1.0, 2.0]`` -> (101, 2).

Consumed by ``source/api_cc/tests/test_deepspin_dpa4_chgspin_ptexpt.cc``.
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

# The base native-spin DPA4 config, shared verbatim with the sibling
# native-spin fixture: this archive is THAT model plus the charge-spin FiLM,
# so importing (rather than re-typing) the config keeps the single difference
# between the two fixtures visible in one place below.
from gen_dpa4_spin import (
    NATIVE_SPIN_CONFIG,
)

# Stored fallback used whenever the caller passes no charge_spin. Written to
# the .pt2 metadata as ``default_chg_spin`` and read back by
# ``DeepSpinPTExpt::init`` into ``default_chg_spin_``; an EMPTY runtime
# charge_spin must reproduce exactly this. Same value as gen_chg_spin.py's
# non-spin DPA3 fixture, for cross-fixture consistency.
_DEFAULT_CHG_SPIN = [0.0, 1.0]

# Explicit runtime probe. Distinct in BOTH components from the default:
# charge idx 101 vs 100, spin idx 2 vs 1 (the embedding is categorical, see
# the module docstring), so neither component alone can explain a response.
_EXPLICIT_CHG_SPIN = [1.0, 2.0]

CHG_SPIN_CONFIG = copy.deepcopy(NATIVE_SPIN_CONFIG)
CHG_SPIN_CONFIG["descriptor"]["add_chg_spin_ebd"] = True
CHG_SPIN_CONFIG["descriptor"]["default_chg_spin"] = _DEFAULT_CHG_SPIN

# Fixed seed for jittering the zero-initialized residual projections away
# from exact zero (see ``jitter_zero_arrays``'s docstring and the module
# docstring above). Deliberately NOT gen_dpa4_spin.py's seed: this is a
# different model (extra FiLM weights change the traversal), so sharing a
# seed would only suggest a weight relationship that does not exist.
_JITTER_SEED = 20260726


def _build_model_dict() -> dict:
    """Build the combined dpmodel from config+seed and jitter in place."""
    from deepmd.dpmodel.model.model import (
        get_model,
    )

    model = get_model(copy.deepcopy(CHG_SPIN_CONFIG))
    assert model.get_dim_chg_spin() == 2, (
        f"expected the combined native-spin DPA4 to expose dim_chg_spin == 2, "
        f"got {model.get_dim_chg_spin()}"
    )
    assert model.get_default_chg_spin() is not None
    model_dict = model.serialize()
    model_dict = jitter_zero_arrays(model_dict, np.random.default_rng(_JITTER_SEED))
    return model_dict


# Fixed 6-atom system (3 Ni, spin-active; 3 O, non-magnetic) -- coordinates,
# cell and spins verbatim from gen_dpa4_spin.py, a system already validated
# to yield a non-degenerate ``force_mag`` with this architecture (rcut=4.0,
# sel=8). Spin is deliberately NOT pre-masked by type: the model's own
# descriptor gating must zero the non-spin (type 1 / O) rows internally.
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


def _check_metadata(pt2_path: str) -> None:
    """Assert the frozen archive's metadata, including the charge-spin slot."""
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
                    "use_spin",
                    "dim_chg_spin",
                    "has_default_chg_spin",
                    "default_chg_spin",
                    "output_keys",
                )
                if k in md
            },
            indent=2,
        )
    )
    assert md["type_map"] == CHG_SPIN_CONFIG["type_map"]
    assert md["lower_input_kind"] == "graph", (
        f"expected native-spin DPA4 to freeze to the graph lower, got "
        f"{md.get('lower_input_kind')!r}"
    )
    # Without BOTH of these the artifact does not exercise the feature at all:
    # is_spin=False would route the C++ side through DeepPot, and
    # dim_chg_spin=0 would make every runtime charge_spin argument inert (which
    # is precisely the state every pre-existing spin fixture is in).
    assert md["is_spin"] is True, (
        f"{pt2_path}: metadata is_spin = {md.get('is_spin')!r}, expected True; "
        f"the DeepSpin charge_spin seam would never be reached."
    )
    assert md.get("dim_chg_spin") == 2, (
        f"{pt2_path}: metadata dim_chg_spin = {md.get('dim_chg_spin')!r}, "
        f"expected 2; a runtime charge_spin would be silently ignored, making "
        f"the C++ regression that consumes this archive vacuous."
    )
    assert md.get("has_default_chg_spin") is True, (
        f"{pt2_path}: metadata has_default_chg_spin = "
        f"{md.get('has_default_chg_spin')!r}, expected True; the empty-"
        f"charge_spin (backward-compatibility) C++ case would throw instead."
    )
    stored_default = [float(x) for x in md.get("default_chg_spin", [])]
    assert stored_default == _DEFAULT_CHG_SPIN, (
        f"{pt2_path}: metadata default_chg_spin = {stored_default!r}, expected "
        f"{_DEFAULT_CHG_SPIN!r}"
    )
    # Native spin rides the with-comm artifact on the graph lower, so the
    # archive carries the nested forward_lower_with_comm.pt2 for the C++
    # multi-rank path (a SECOND inductor compile).
    assert md["has_comm_artifact"] is True
    assert md["has_message_passing"] is True
    assert md["use_spin"] == [True, False]
    for key in ("atom_energy", "energy", "force", "force_mag", "virial"):
        assert key in md["output_keys"]


def _eval_one(dp, cell, charge_spin, label: str) -> tuple[dict, float]:
    """Evaluate one (cell, charge_spin) case; return its ref arrays + energy.

    ``charge_spin=None`` means "pass no charge_spin at all", i.e. exercise the
    model's stored ``default_chg_spin`` -- the Python twin of an EMPTY
    ``std::vector<double>`` on the C++ side.
    """
    kwargs = {}
    if charge_spin is not None:
        kwargs["charge_spin"] = np.array([charge_spin], dtype=np.float64)
    e, f, v, ae, av, fm, _mm = dp.eval(
        _COORDS, cell, _ATYPES, atomic=True, spin=_SPINS, **kwargs
    )
    print(f"// {label} total energy: {e[0, 0]:.18e}")  # noqa: T201

    assert np.all(np.isfinite(e)), f"{label}: non-finite energy"
    assert np.all(np.isfinite(f)), f"{label}: non-finite force"
    assert np.all(np.isfinite(fm)), f"{label}: non-finite force_mag"

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
    # atoms too (see the jitter docstring above).
    assert fm_spin_max > 1e-6, (
        f"{label}: expected non-trivial force_mag on spin-active (Ni) atoms; "
        f"got max |force_mag| = {fm_spin_max:.3e} (jitter not effective -- "
        f"this fixture would be vacuous)."
    )
    # The non-spin (O) rows must be exactly gated to zero by the model's own
    # type mask -- not merely small.
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
    }, float(e[0, 0])


def main():
    from deepmd.pt_expt.utils.serialization import (
        deserialize_to_file as pt_expt_deserialize_to_file,
    )

    ensure_inductor_compiler()
    load_custom_ops()

    base_dir = os.path.dirname(__file__)
    pt2_path = os.path.join(base_dir, "deeppot_dpa4_spin_chgspin.pt2")
    ref_path = os.path.join(base_dir, "deeppot_dpa4_spin_chgspin.expected")

    # ---- 1. Build the jittered dpmodel dict from config+seed ----
    model_dict = _build_model_dict()
    data = {
        "model": model_dict,
        "model_def_script": CHG_SPIN_CONFIG,
        "backend": "dpmodel",
        "software": "deepmd-kit",
        "version": "3.0.0",
    }

    # ---- 2. Freeze directly to graph-kind .pt2 ----
    # Native-spin DPA4 has NO dense/nlist lower at all (spin rides the
    # NeighborGraph lower exclusively), so ``lower_kind="auto"`` would resolve
    # to "graph" anyway; pinned explicitly here for clarity.
    print(f"Exporting to {pt2_path} (lower_kind='graph') ...")  # noqa: T201
    pt_expt_deserialize_to_file(
        pt2_path, data, do_atomic_virial=True, lower_kind="graph"
    )
    print("Export done.")  # noqa: T201
    _check_metadata(pt2_path)

    # ---- 3. Evaluate the four reference cases ----
    from deepmd.infer import (
        DeepPot,
    )

    dp = DeepPot(pt2_path)
    assert dp.has_spin
    dim = dp.deep_eval.get_dim_chg_spin()
    assert dim == 2, f"expected dim_chg_spin == 2 from DeepEval, got {dim}"

    print("")  # noqa: T201
    pbc_default, e_pbc_default = _eval_one(dp, _CELL, None, "PBC default")
    pbc_explicit, e_pbc_explicit = _eval_one(
        dp, _CELL, _EXPLICIT_CHG_SPIN, f"PBC explicit {_EXPLICIT_CHG_SPIN}"
    )
    nopbc_default, e_nopbc_default = _eval_one(dp, None, None, "NoPbc default")
    nopbc_explicit, e_nopbc_explicit = _eval_one(
        dp, None, _EXPLICIT_CHG_SPIN, f"NoPbc explicit {_EXPLICIT_CHG_SPIN}"
    )

    # ---- 4. Anti-vacuity: charge_spin must MOVE the output ----
    # Equal energies would mean the charge-spin FiLM never reached the graph
    # forward, so the C++ regression consuming these references would pass for
    # the wrong reason (it asserts two charge_spin values give two energies).
    for label, e_def, e_exp in (
        ("PBC", e_pbc_default, e_pbc_explicit),
        ("NoPbc", e_nopbc_default, e_nopbc_explicit),
    ):
        print(  # noqa: T201
            f"\n// {label} default  energy: {e_def:.18e}\n"
            f"// {label} explicit energy: {e_exp:.18e}\n"
            f"// {label} delta:           {abs(e_exp - e_def):.6e}"
        )
        assert abs(e_exp - e_def) > 1e-6, (
            f"{label}: charge_spin={_EXPLICIT_CHG_SPIN} left the energy "
            f"unchanged vs the stored default {_DEFAULT_CHG_SPIN} "
            f"({e_def:.18e} vs {e_exp:.18e}); the FiLM conditioning is not "
            f"reaching the forward, so this fixture would be vacuous."
        )

    # ---- 5. Pin the "no charge_spin == stored default" contract ----
    # The C++ empty-charge_spin case is checked against the ``*_default``
    # sections, so those sections must really BE the stored default and not
    # some third behaviour. Passing the default value explicitly has to
    # reproduce them bit-for-bit (same tensor, same code path).
    explicit_default, e_explicit_default = _eval_one(
        dp, _CELL, _DEFAULT_CHG_SPIN, f"PBC explicit {_DEFAULT_CHG_SPIN} (== default)"
    )
    for key, arr in explicit_default.items():
        np.testing.assert_allclose(
            arr,
            pbc_default[key],
            rtol=1e-14,
            atol=1e-14,
            err_msg=(
                f"passing charge_spin={_DEFAULT_CHG_SPIN} explicitly disagrees "
                f"with omitting it ({key}); the stored default_chg_spin is not "
                f"what the omitted case uses."
            ),
        )
    assert abs(e_explicit_default - e_pbc_default) < 1e-12

    # ---- 6. Write the sidecar reference ----
    write_expected_ref(
        ref_path,
        sections={
            "pbc_default": pbc_default,
            "pbc_explicit": pbc_explicit,
            "nopbc_default": nopbc_default,
            "nopbc_explicit": nopbc_explicit,
        },
        source_script="source/tests/infer/gen_dpa4_spin_chgspin.py",
    )
    print(f"\nWrote {ref_path}")  # noqa: T201

    print("\nDone!")  # noqa: T201


if __name__ == "__main__":
    main()
