#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate DPA1 test models carrying model-level ``pair_exclude_types``.

Produces two graph-eligible DPA1(attn_layer=0) models with identical weights,
one exported through each C++ ingestion route, plus a no-exclusion baseline:

  - deeppot_dpa1_pairexcl_graph.pt2  (lower_kind="graph", pair_exclude=[[0,1]])
  - deeppot_dpa1_pairexcl_nlist.pt2  (lower_kind="nlist", pair_exclude=[[0,1]])
  - deeppot_dpa1_pairexcl_none.pt2   (lower_kind="graph", NO exclusion)

The pair models exercise the C++ pair-exclusion ingestion seam:
``applyPairExclusion`` (graph route) / ``applyPairExclusionNlist`` (dense route)
plus the ``pair_exclude_types`` metadata round-trip in ``DeepPotPTExpt::init``.
Model-level pair exclusion is a graph-BUILD transform (decision #18): it is
folded into ``edge_mask`` at build time (``applyPairExclusion`` in C++, the
NeighborGraph builder in Python ``DeepEval``), NOT inside the exported ``.pt2``
lower. The gtest validates C++ energy/force vs the Python ``DeepEval`` reference
at 1e-10 and, by comparing against the ``_none`` baseline, confirms the exclusion
is actually active.

Reference sidecar files (.expected) consumed by the C++ gtest are written from
the Python ``DeepEval`` evaluation of each pair model (PBC + NoPBC sections).

exclude_types = [[0, 1]] drops every O-H (cross-type) interaction while keeping
O-O and H-H, so both energy and forces change measurably but stay non-degenerate
(H-H pairs survive).

This is a SEPARATE script from ``gen_dpa1.py`` so it can be run independently on
boxes where the unrelated attn_layer=2 model in ``gen_dpa1.py`` trips a torch
inductor codegen bug.
"""

import copy
import os
import sys

import numpy as np

# Ensure the source tree is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from gen_common import (
    ensure_inductor_compiler,
    load_custom_ops,
    write_expected_ref,
)


def main():
    from deepmd.dpmodel.model.model import (
        get_model,
    )

    ensure_inductor_compiler()

    base_dir = os.path.dirname(__file__)

    # Graph-eligible DPA1 (attn_layer=0); same descriptor as gen_dpa1.py
    # Section B so the two fixtures stay comparable.
    base_config = {
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "se_atten",
            "sel": 30,
            "rcut_smth": 2.0,
            "rcut": 6.0,
            "neuron": [2, 4, 8],
            "axis_neuron": 4,
            "attn": 5,
            "attn_layer": 0,
            "attn_dotr": True,
            "attn_mask": False,
            "activation_function": "tanh",
            "scaling_factor": 1.0,
            "normalize": True,
            "temperature": 1.0,
            "type_one_side": True,
            "seed": 1,
        },
        "fitting_net": {
            "neuron": [5, 5, 5],
            "resnet_dt": True,
            "seed": 1,
        },
    }

    from deepmd.pt_expt.utils.serialization import (
        deserialize_to_file as pt_expt_deserialize_to_file,
    )

    load_custom_ops()

    from deepmd.infer import (
        DeepPot,
    )

    coord = np.array(
        [
            12.83, 2.56, 2.18, 12.09, 2.87, 2.74, 0.25, 3.32, 1.68,
            3.36, 3.00, 1.81, 3.51, 2.51, 2.60, 4.27, 3.22, 1.56,
        ],
        dtype=np.float64,
    )  # fmt: skip
    atype = [0, 1, 1, 0, 1, 1]
    box = np.array([13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0], dtype=np.float64)

    def build_data(exclude_types):
        cfg = copy.deepcopy(base_config)
        if exclude_types:
            cfg["pair_exclude_types"] = exclude_types
        model = get_model(copy.deepcopy(cfg))
        return {
            "model": copy.deepcopy(model.serialize()),
            "model_def_script": cfg,
            "backend": "dpmodel",
            "software": "deepmd-kit",
            "version": "3.0.0",
        }

    # ---- No-exclusion baseline (graph route) ----
    none_pt2 = os.path.join(base_dir, "deeppot_dpa1_pairexcl_none.pt2")
    print(f"Exporting no-exclusion baseline to {none_pt2} ...")  # noqa: T201
    pt_expt_deserialize_to_file(
        none_pt2, build_data(None), do_atomic_virial=True, lower_kind="graph"
    )
    dp_none = DeepPot(none_pt2)
    e_none = dp_none.eval(coord, box, atype, atomic=False)[0][0, 0]
    print(f"  baseline PBC energy: {e_none:.18e}")  # noqa: T201

    # ---- Pair-exclusion models (graph + dense routes) ----
    exclude_types = [[0, 1]]
    data_e = build_data(exclude_types)
    for lower_kind, tag in (("graph", "graph"), ("nlist", "nlist")):
        pt2_path = os.path.join(base_dir, f"deeppot_dpa1_pairexcl_{tag}.pt2")
        print(  # noqa: T201
            f"Exporting to {pt2_path} (lower_kind='{lower_kind}', "
            f"pair_exclude_types={exclude_types}) ..."
        )
        pt_expt_deserialize_to_file(
            pt2_path,
            copy.deepcopy(data_e),
            do_atomic_virial=True,
            lower_kind=lower_kind,
        )
        dp_e = DeepPot(pt2_path)
        e_e1, f_e1, v_e1, ae_e1, av_e1 = dp_e.eval(coord, box, atype, atomic=True)
        e_enp, f_enp, v_enp, ae_enp, av_enp = dp_e.eval(coord, None, atype, atomic=True)

        # Confirm the exclusion is ACTIVE: energy must differ from the
        # no-exclusion baseline (identical weights minus pair_exclude_types).
        e_diff = float(abs(e_e1[0, 0] - e_none))
        print(f"  {tag}: |E(excl) - E(no-excl)| = {e_diff:.3e}")  # noqa: T201
        if e_diff < 1e-6:
            raise RuntimeError(
                f"BLOCKED: pair_exclude_types had no effect on the {tag} model "
                f"(energy delta {e_diff:.2e} < 1e-6); exclusion may be dropped."
            )
        f_max = float(np.max(np.abs(f_e1)))
        if f_max < 1e-10:
            raise RuntimeError(
                f"Pair-exclude {tag} forces are degenerate (max={f_max:.2e})."
            )

        ref_path = os.path.join(base_dir, f"deeppot_dpa1_pairexcl_{tag}.expected")
        write_expected_ref(
            ref_path,
            sections={
                "pbc": {
                    "expected_e": ae_e1[0, :, 0],
                    "expected_f": f_e1[0],
                    "expected_v": av_e1[0],
                },
                "nopbc": {
                    "expected_e": ae_enp[0, :, 0],
                    "expected_f": f_enp[0],
                    "expected_v": av_enp[0],
                },
            },
            source_script="source/tests/infer/gen_dpa1_pairexcl.py",
        )
        print(f"  Wrote {ref_path}")  # noqa: T201

    print("\nAll pair-exclude models generated.")  # noqa: T201
    print("Done!")  # noqa: T201


if __name__ == "__main__":
    main()
