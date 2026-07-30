#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate deeppot_dpa4_zbl_graph.pt2: DPA4 with analytical ZBL bridging.

``bridging_method: ZBL`` builds a COMPOSITION -- ``LinearEnergyModel`` over
``[learned DPA4, InterPotentialAtomicModel]`` with ``weights="sum"`` -- so
the frozen archive exercises a code path no other C++ fixture covers: the
graph lower of a linear composition rather than a single learned model.
Before this fixture, ZBL bridging had NO C++ or LAMMPS coverage at all; its
only end-to-end test drove the ``.pt2`` through the PYTHON ``DeepPot``, which
never touches ``DeepPotPTExpt``.

Multi-rank capable (issue #5906): bridging enables the descriptor's Source
Freeze Propagation Gate, whose per-node ``eta_j = prod_{e: src_e = j} w_e``
folds a node's FULL outgoing-edge set.  Edges exist only for owned centres,
so the per-node ``[log_eta, zero_count]`` partials are rank-incomplete; the
with-comm artifact completes them via one reverse-accumulate
(``border_op_backward``) + forward-broadcast (``border_op``) exchange before
the gate is applied, so the nested ``forward_lower_with_comm.pt2`` is
embedded (``has_comm_artifact=true``, asserted below).

Generation mirrors ``gen_dpa4_spin.py``: the dpmodel is built in-process from
the inline config with a fixed weight-init seed, its zero-initialised
residual projections are jittered away from exact zero with a fixed RNG seed,
and the result is frozen straight to the graph-kind ``.pt2``.  Without the
jitter a fresh DPA4 collapses to a type-embedding-only descriptor and every
force would be identically zero, making the fixture vacuous.
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
# standalone, outside pytest's package machinery).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dpa4_fixtures import (
    jitter_zero_arrays,
)
from gen_common import (
    ensure_inductor_compiler,
    load_custom_ops,
    write_expected_ref,
)

# Small fp64 DPA4 + ZBL config.  ``bridging_r_inner``/``r_outer`` feed the
# descriptor's InnerClamp AND BridgingSwitch (they are built together from
# the same radii), and the model-level InterPotential term.
ZBL_CONFIG = {
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
    "bridging_method": "ZBL",
    "bridging_r_inner": 0.8,
    "bridging_r_outer": 1.2,
}

_JITTER_SEED = 20260725

# Fixed 6-atom system.  Atoms 0 and 1 sit 0.9 A apart -- inside
# ``bridging_r_outer`` -- so the analytical ZBL term contributes a large,
# unmistakable repulsion instead of a numerical afterthought.
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


def main():
    from deepmd.dpmodel.model.model import (
        get_model,
    )
    from deepmd.infer import (
        DeepPot,
    )
    from deepmd.pt_expt.utils.serialization import (
        deserialize_to_file as pt_expt_deserialize_to_file,
    )

    ensure_inductor_compiler()
    load_custom_ops()

    base_dir = os.path.dirname(__file__)
    pt2_path = os.path.join(base_dir, "deeppot_dpa4_zbl_graph.pt2")

    # ---- 1. Build the jittered composition ----
    model = get_model(copy.deepcopy(ZBL_CONFIG))
    model_dict = jitter_zero_arrays(
        model.serialize(), np.random.default_rng(_JITTER_SEED)
    )
    data = {
        "model": model_dict,
        "model_def_script": ZBL_CONFIG,
        "backend": "dpmodel",
        "software": "deepmd-kit",
        "version": "3.0.0",
    }

    # ---- 2. Freeze to graph-kind .pt2 ----
    print(f"Exporting to {pt2_path} (lower_kind='graph') ...")  # noqa: T201
    pt_expt_deserialize_to_file(
        pt2_path, data, do_atomic_virial=True, lower_kind="graph"
    )
    print("Export done.")  # noqa: T201

    # ---- 3. Check metadata ----
    with zipfile.ZipFile(pt2_path) as zf:
        md = json.loads(zf.read("model/extra/metadata.json").decode("utf-8"))
        names = zf.namelist()
    print(  # noqa: T201
        json.dumps(
            {
                k: md[k]
                for k in ("type_map", "lower_input_kind", "has_comm_artifact")
                if k in md
            },
            indent=2,
        )
    )
    assert md["type_map"] == ZBL_CONFIG["type_map"]
    assert md["lower_input_kind"] == "graph"
    # Multi-rank capable (issue #5906): the SFPG per-node partials are
    # completed across ranks, so the nested with-comm artifact is embedded.
    assert md["has_comm_artifact"] is True
    assert "model/extra/forward_lower_with_comm.pt2" in names

    # ---- 4. Evaluate (PBC + NoPbc) ----
    dp = DeepPot(pt2_path)
    e1, f1, v1, ae1, av1 = dp.eval(_COORDS, _CELL, _ATYPES, atomic=True)
    e_np, f_np, v_np, ae_np, av_np = dp.eval(_COORDS, None, _ATYPES, atomic=True)
    print(f"\n// PBC   total energy: {e1[0, 0]:.18e}")  # noqa: T201
    print(f"// NoPbc total energy: {e_np[0, 0]:.18e}")  # noqa: T201

    for label, e, f in (("PBC", e1, f1), ("NoPbc", e_np, f_np)):
        assert np.all(np.isfinite(e)), f"{label}: non-finite energy"
        assert np.all(np.isfinite(f)), f"{label}: non-finite force"
        fmax = float(np.max(np.abs(f)))
        print(f"// {label} max |force|: {fmax:.6e}")  # noqa: T201
        # Anti-vacuity: an unjittered DPA4 gives identically zero forces.
        # The close Ni-Ni pair alone drives a large ZBL repulsion, so a
        # small max-force means the fixture is degenerate.
        assert fmax > 1e-3, (
            f"{label}: expected a non-trivial force (the 0.9 A Ni-Ni pair "
            f"drives the analytical ZBL term); got {fmax:.3e} -- jitter or "
            f"bridging is not effective and this fixture would be vacuous."
        )

    # ---- 5. Sidecar reference consumed by the C++ test ----
    ref_path = os.path.join(base_dir, "deeppot_dpa4_zbl_graph.expected")
    write_expected_ref(
        ref_path,
        sections={
            "pbc": {
                "expected_e": ae1[0, :, 0],
                "expected_f": f1[0],
                "expected_tot_v": v1[0],
                "expected_atom_v": av1[0],
            },
            "nopbc": {
                "expected_e": ae_np[0, :, 0],
                "expected_f": f_np[0],
                "expected_tot_v": v_np[0],
                "expected_atom_v": av_np[0],
            },
        },
        source_script="source/tests/infer/gen_dpa4_zbl.py",
    )
    print(f"Wrote {ref_path}")  # noqa: T201
    print("\nDone!")  # noqa: T201


if __name__ == "__main__":
    main()
