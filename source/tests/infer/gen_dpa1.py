#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate deeppot_dpa1.pth, deeppot_dpa1.pt2, and deeppot_dpa1_graph.pt2 test models.

Creates two DPA1 models from dpmodel configs:
  - deeppot_dpa1.pt2 / deeppot_dpa1.pth  (attn_layer=2, dense nlist-form export)
  - deeppot_dpa1_graph.pt2                (attn_layer=0, graph-form export via
                                            lower_kind="graph"; the graph forward
                                            is eligible only when attn_layer==0)

Both are serialized and exported to their respective formats from the same weights.
Reference sidecar files (.expected) consumed by C++ gtests are also written:
  - deeppot_dpa1.expected   — from the nlist .pt2 eval (existing)
  - deeppot_dpa1_graph.expected — from an independent NLIST .pt2 eval (NOT the
      graph .pt2; dpmodel se_atten has no analytical force, so the dense nlist
      path is the independent ground truth). At non-binding sel the graph and
      nlist paths see the same neighbor set, so the graph .pt2 is sanity-checked
      against this reference at ≤1e-5.
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

    # ---- 1. DPA1 model config with type_one_side=True ----
    config = {
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "se_atten",
            "sel": 30,
            "rcut_smth": 2.0,
            "rcut": 6.0,
            "neuron": [2, 4, 8],
            "axis_neuron": 4,
            "attn": 5,
            "attn_layer": 2,
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

    # ---- 2. Build dpmodel and serialize ----
    model = get_model(copy.deepcopy(config))
    model_dict = model.serialize()

    data = {
        "model": model_dict,
        "model_def_script": config,
        "backend": "dpmodel",
        "software": "deepmd-kit",
        "version": "3.0.0",
    }

    # ---- 3. Export to .pt2 and .pth ----
    from deepmd.pt.utils.serialization import (
        deserialize_to_file as pt_deserialize_to_file,
    )
    from deepmd.pt_expt.utils.serialization import (
        deserialize_to_file as pt_expt_deserialize_to_file,
    )

    # Load custom ops after deepmd.pt import to avoid double registration
    load_custom_ops()

    base_dir = os.path.dirname(__file__)

    pt2_path = os.path.join(base_dir, "deeppot_dpa1.pt2")
    print(f"Exporting to {pt2_path} ...")  # noqa: T201
    pt_expt_deserialize_to_file(pt2_path, copy.deepcopy(data), do_atomic_virial=True)

    pth_path = os.path.join(base_dir, "deeppot_dpa1.pth")
    print(f"Exporting to {pth_path} ...")  # noqa: T201
    try:
        pt_deserialize_to_file(pth_path, copy.deepcopy(data))
    except RuntimeError as e:
        # Custom ops (e.g. tabulate_fusion_se_t_tebd) may not be available
        # in all build environments; .pth generation is not critical.
        print(f"WARNING: .pth export failed ({e}), skipping.")  # noqa: T201

    print("Export done.")  # noqa: T201

    # ---- 4. Run inference for PBC test ----
    from deepmd.infer import (
        DeepPot,
    )

    dp = DeepPot(pt2_path)

    coord = np.array(
        [
            12.83,
            2.56,
            2.18,
            12.09,
            2.87,
            2.74,
            0.25,
            3.32,
            1.68,
            3.36,
            3.00,
            1.81,
            3.51,
            2.51,
            2.60,
            4.27,
            3.22,
            1.56,
        ],
        dtype=np.float64,
    )
    atype = [0, 1, 1, 0, 1, 1]
    box = np.array([13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0], dtype=np.float64)

    e1, f1, v1, ae1, av1 = dp.eval(coord, box, atype, atomic=True)
    print(f"\n// PBC total energy: {e1[0, 0]:.18e}")  # noqa: T201

    # ---- 5. Run inference for NoPbc test ----
    e_np, f_np, v_np, ae_np, av_np = dp.eval(coord, None, atype, atomic=True)
    print(f"\n// NoPbc total energy: {e_np[0, 0]:.18e}")  # noqa: T201

    # ---- 5b. Write sidecar reference file consumed by C++ tests ----
    ref_path = os.path.join(base_dir, "deeppot_dpa1.expected")
    write_expected_ref(
        ref_path,
        sections={
            "pbc": {
                "expected_e": ae1[0, :, 0],
                "expected_f": f1[0],
                "expected_v": av1[0],
            },
            "nopbc": {
                "expected_e": ae_np[0, :, 0],
                "expected_f": f_np[0],
                "expected_v": av_np[0],
            },
        },
        source_script="source/tests/infer/gen_dpa1.py",
    )
    print(f"Wrote {ref_path}")  # noqa: T201

    # ---- 6. Verify .pth gives same results ----
    dp_pth = DeepPot(pth_path)
    e_pth, f_pth, v_pth, ae_pth, av_pth = dp_pth.eval(coord, box, atype, atomic=True)
    print(f"\n// .pth PBC total energy: {e_pth[0, 0]:.18e}")  # noqa: T201
    print(f"// .pth vs .pt2 energy diff: {abs(e1[0, 0] - e_pth[0, 0]):.2e}")  # noqa: T201
    print(f"// .pth vs .pt2 force max diff: {np.max(np.abs(f1 - f_pth)):.2e}")  # noqa: T201

    e_pth_np, f_pth_np, _, ae_pth_np, av_pth_np = dp_pth.eval(
        coord, None, atype, atomic=True
    )
    print(f"// .pth NoPbc total energy: {e_pth_np[0, 0]:.18e}")  # noqa: T201
    print(f"// .pth vs .pt2 NoPbc energy diff: {abs(e_np[0, 0] - e_pth_np[0, 0]):.2e}")  # noqa: T201

    # ============================================================
    # Section B: graph-eligible DPA1 (attn_layer=0) model
    # ============================================================
    # attn_layer=0 disables the attention layers, making the descriptor
    # a plain two-body embedding (se_e2_a-like) that is eligible for the
    # NeighborGraph forward path (forward_lower_graph_exportable).
    # Config mirrors DPA1_CONFIG in
    # source/tests/pt_expt/utils/test_graph_pt2_metadata.py
    graph_config = {
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

    print("\n---- Building graph-eligible DPA1 (attn_layer=0) ----")  # noqa: T201

    # ---- B.1  Build dpmodel, serialize ----
    model_g = get_model(copy.deepcopy(graph_config))
    model_dict_g = model_g.serialize()

    data_g = {
        "model": copy.deepcopy(model_dict_g),
        "model_def_script": graph_config,
        "backend": "dpmodel",
        "software": "deepmd-kit",
        "version": "3.0.0",
    }

    # ---- B.2  Compute reference via nlist .pt2 (independent of graph path) ----
    # The reference for deeppot_dpa1_graph.expected comes from the NLIST .pt2
    # (dense-quartet forward), NOT the graph .pt2.  This ensures the C++ gtest
    # (B2.5) independently validates the graph AOTI path against a known-good
    # nlist evaluation.
    #
    # The nlist .pt2 is also PERSISTED (deeppot_dpa1_graph_nlist_ref.pt2): the
    # C++ gtest loads it alongside the graph .pt2 to cross-check graph≈dense at
    # 1e-9 on arbitrary system sizes (dynamic-edge-axis cases) without baking a
    # second reference block into the .expected sidecar.  Same weights as the
    # graph model, so at non-binding sel the two paths must agree.
    nlist_ref_pt2 = os.path.join(base_dir, "deeppot_dpa1_graph_nlist_ref.pt2")
    print(f"Exporting reference nlist .pt2 to {nlist_ref_pt2} ...")  # noqa: T201
    pt_expt_deserialize_to_file(
        nlist_ref_pt2,
        copy.deepcopy(data_g),
        do_atomic_virial=True,
        lower_kind="nlist",  # independent: dense nlist, NOT graph
    )
    dp_nlist_ref = DeepPot(nlist_ref_pt2)

    # PBC reference from nlist path
    e_r1, f_r1, v_r1, ae_r1, av_r1 = dp_nlist_ref.eval(coord, box, atype, atomic=True)
    # NoPBC reference from nlist path
    e_rnp, f_rnp, v_rnp, ae_rnp, av_rnp = dp_nlist_ref.eval(
        coord, None, atype, atomic=True
    )

    print(f"Nlist ref PBC energy: {e_r1[0, 0]:.18e}")  # noqa: T201
    print(f"Nlist ref NoPBC energy: {e_rnp[0, 0]:.18e}")  # noqa: T201
    max_ref_force_pbc = float(np.max(np.abs(f_r1)))
    max_ref_force_nopbc = float(np.max(np.abs(f_rnp)))
    print(f"Nlist ref PBC max |force|: {max_ref_force_pbc:.6e}")  # noqa: T201
    print(f"Nlist ref NoPBC max |force|: {max_ref_force_nopbc:.6e}")  # noqa: T201
    if max_ref_force_pbc < 1e-10:
        raise RuntimeError(
            f"Graph model nlist-ref forces are degenerate "
            f"(max={max_ref_force_pbc:.2e}); weights may need perturbation."
        )

    # ---- B.3  Write sidecar reference file ----
    graph_ref_path = os.path.join(base_dir, "deeppot_dpa1_graph.expected")
    write_expected_ref(
        graph_ref_path,
        sections={
            "pbc": {
                "expected_e": ae_r1[0, :, 0],
                "expected_f": f_r1[0],
                "expected_v": av_r1[0],
            },
            "nopbc": {
                "expected_e": ae_rnp[0, :, 0],
                "expected_f": f_rnp[0],
                "expected_v": av_rnp[0],
            },
        },
        source_script="source/tests/infer/gen_dpa1.py",
    )
    print(f"Wrote {graph_ref_path}")  # noqa: T201

    # ---- B.4  Export graph-form .pt2 ----
    graph_pt2_path = os.path.join(base_dir, "deeppot_dpa1_graph.pt2")
    print(f"Exporting to {graph_pt2_path} (lower_kind='graph') ...")  # noqa: T201
    pt_expt_deserialize_to_file(
        graph_pt2_path,
        copy.deepcopy(data_g),
        do_atomic_virial=True,
        lower_kind="graph",
    )
    print("Graph .pt2 export done.")  # noqa: T201

    # ---- B.5  Sanity-check: graph .pt2 vs nlist reference ----
    # Both use the SAME weights; at non-binding sel the math is equivalent.
    # Verifies that forward_lower_graph_exportable + edge_energy_deriv match
    # the nlist forward for this concrete system.
    dp_graph = DeepPot(graph_pt2_path)

    # PBC sanity check
    e_g1, f_g1, v_g1, ae_g1, av_g1 = dp_graph.eval(coord, box, atype, atomic=True)
    force_diff_pbc = float(np.max(np.abs(f_g1[0] - f_r1[0])))
    print(  # noqa: T201
        f"Graph .pt2 vs nlist ref PBC force max diff: {force_diff_pbc:.2e}"
    )
    if force_diff_pbc > 1e-5:
        raise RuntimeError(
            f"BLOCKED: graph .pt2 PBC force differs from nlist reference by "
            f"{force_diff_pbc:.2e} (threshold 1e-5)."
        )

    # NoPBC sanity check
    e_gnp, f_gnp, v_gnp, ae_gnp, av_gnp = dp_graph.eval(coord, None, atype, atomic=True)
    force_diff_nopbc = float(np.max(np.abs(f_gnp[0] - f_rnp[0])))
    print(  # noqa: T201
        f"Graph .pt2 vs nlist ref NoPBC force max diff: {force_diff_nopbc:.2e}"
    )
    if force_diff_nopbc > 1e-5:
        raise RuntimeError(
            f"BLOCKED: graph .pt2 NoPBC force differs from nlist reference by "
            f"{force_diff_nopbc:.2e} (threshold 1e-5)."
        )

    print("\nAll graph sanity checks passed.")  # noqa: T201
    print("\nDone!")  # noqa: T201


if __name__ == "__main__":
    main()
