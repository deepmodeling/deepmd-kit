#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate deeppot_dpa2.pth and deeppot_dpa2.pt2 test models.

Creates a DPA2 model from dpmodel config (with three-body, type_one_side=True),
serializes, and exports to both .pth and .pt2 from the same weights.
Also prints reference values for C++ tests (PBC and NoPbc).
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

    # ---- 1. DPA2 model config with type_one_side=True, use_three_body=True ----
    config = {
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "dpa2",
            "repinit": {
                "rcut": 6.0,
                "rcut_smth": 2.0,
                "nsel": 30,
                "neuron": [2, 4, 8],
                "axis_neuron": 4,
                "tebd_dim": 8,
                "tebd_input_mode": "concat",
                "set_davg_zero": True,
                "type_one_side": True,
                "use_three_body": True,
                "three_body_neuron": [2, 4],
                "three_body_sel": 20,
                "three_body_rcut": 4.0,
                "three_body_rcut_smth": 0.5,
            },
            "repformer": {
                "rcut": 3.0,
                "rcut_smth": 1.5,
                "nsel": 15,
                "nlayers": 2,
                "g1_dim": 8,
                "g2_dim": 5,
                "axis_neuron": 4,
                "update_g1_has_conv": True,
                "update_g1_has_drrd": True,
                "update_g1_has_grrg": True,
                "update_g2_has_attn": True,
                "attn1_hidden": 8,
                "attn1_nhead": 2,
                "attn2_hidden": 5,
                "attn2_nhead": 1,
                "update_style": "res_avg",
                "set_davg_zero": True,
            },
            "concat_output_tebd": True,
            "precision": "float64",
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

    pt2_path = os.path.join(base_dir, "deeppot_dpa2.pt2")
    print(f"Exporting to {pt2_path} ...")  # noqa: T201
    # DPA2's repformer block has no ``use_loc_mapping`` knob (unlike
    # DPA3), so a single .pt2 already carries the dual-artifact layout
    # (regular + with-comm) — ``has_message_passing_across_ranks``
    # returns True and the serializer produces both. No separate _mpi.pt2
    # needed.
    pt_expt_deserialize_to_file(pt2_path, copy.deepcopy(data), do_atomic_virial=True)

    pth_path = os.path.join(base_dir, "deeppot_dpa2.pth")
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
    ref_path = os.path.join(base_dir, "deeppot_dpa2.expected")
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
        source_script="source/tests/infer/gen_dpa2.py",
    )
    print(f"Wrote {ref_path}")  # noqa: T201

    # ---- 6. Verify .pth gives same results ----
    if os.path.exists(pth_path):
        dp_pth = DeepPot(pth_path)
        e_pth, f_pth, v_pth, ae_pth, av_pth = dp_pth.eval(
            coord, box, atype, atomic=True
        )
        print(f"\n// .pth PBC total energy: {e_pth[0, 0]:.18e}")  # noqa: T201
        print(f"// .pth vs .pt2 energy diff: {abs(e1[0, 0] - e_pth[0, 0]):.2e}")  # noqa: T201
        print(f"// .pth vs .pt2 force max diff: {np.max(np.abs(f1 - f_pth)):.2e}")  # noqa: T201

        e_pth_np, f_pth_np, _, ae_pth_np, av_pth_np = dp_pth.eval(
            coord, None, atype, atomic=True
        )
        print(f"// .pth NoPbc total energy: {e_pth_np[0, 0]:.18e}")  # noqa: T201
        print(  # noqa: T201
            f"// .pth vs .pt2 NoPbc energy diff: {abs(e_np[0, 0] - e_pth_np[0, 0]):.2e}"
        )
    else:
        print("\n// Skipping .pth verification (file not generated).")  # noqa: T201

    # ============================================================
    # Section B: graph-eligible DPA2 (use_three_body=False) model
    # ============================================================
    # Three-body (repinit's optional se_t_tebd sub-block) is graph-ineligible,
    # so the graph-form export needs a config with use_three_body=False.
    # Everything else stays at the section-A defaults (attn toggles ON in
    # repformer -- that's the point: repformer's own attention is graph-
    # native, unlike DPA1's se_atten which requires attn_layer=0).
    #
    # Skip the whole graph section under LeakSanitizer. The C++ memleak matrix
    # runs these gen scripts with ``LD_PRELOAD=liblsan`` (the sanitizer-
    # instrumented deepmd op .so requires the LSAN runtime; see
    # source/install/test_cc_local.sh). Evaluating the AOTInductor-compiled
    # graph .pt2's BACKWARD (forces) under that runtime segfaults inside the
    # repformer's fused backward kernel -- an AOTI-compiled-code vs
    # LeakSanitizer allocator incompatibility, NOT a graph-code bug: the same
    # backward is bit-identical and finite in eager, in the non-memleak C++
    # ctest, and in LAMMPS. dpa1's simpler graph .pt2 does not trip it. Leak-
    # checking torch's own compiled kernels is meaningless anyway (the memleak
    # build exists to leak-check deepmd's C++ ops). The dense DPA2 .pt2
    # (section A) is unaffected and still generated. The C++ dpa2_graph_ptexpt
    # row GTEST_SKIPs when this artifact is absent (skip_if_artifact_missing).
    if "lsan" in os.environ.get("LD_PRELOAD", "").lower():
        print(  # noqa: T201
            "\n// Skipping DPA2 graph section under LeakSanitizer "
            "(AOTInductor .pt2 backward is incompatible with the LSAN runtime; "
            "covered by the non-memleak C++/LAMMPS matrix)."
        )
        return

    graph_config = copy.deepcopy(config)
    graph_config["descriptor"]["repinit"]["use_three_body"] = False

    print("\n---- Building graph-eligible DPA2 (use_three_body=False) ----")  # noqa: T201

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

    # ---- B.2  Independent cross-check via nlist .pt2 (dense-quartet) ----
    # Unlike the DPA1 graph fixture, deeppot_dpa2_graph.expected is NOT
    # copied from the nlist artifact (see B.5): DPA2 is a message-passing
    # model, so the graph path's per-edge full-to-source atomic-virial
    # decomposition genuinely differs from the dense per-atom decomposition
    # (only the SUM agrees), and cross-path energy/force agreement (~1e-10)
    # sits at the universal gtest's double tolerance (1e-10).  The nlist
    # artifact is instead used here as the independent gen-time oracle:
    # atomic energies, forces and the TOTAL virial of the graph .pt2 must
    # match it (checked in B.4) or generation aborts.
    #
    # The nlist .pt2 is PERSISTED (deeppot_dpa2_graph_nlist_ref.pt2): a
    # C++ gtest could load it alongside the graph .pt2 to cross-check
    # graph≈dense on arbitrary system sizes without baking a second reference
    # block into the .expected sidecar.  Same weights as the graph model, so
    # at non-binding sel the two paths must agree.
    nlist_ref_pt2 = os.path.join(base_dir, "deeppot_dpa2_graph_nlist_ref.pt2")
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
    # ``not (x >= th)`` (rather than ``x < th``) so NaN forces -- e.g. from
    # an inductor SIMD miscompile of the AOTI artifact -- fail the check
    # instead of slipping through.
    if not (max_ref_force_pbc >= 1e-10) or not (
        np.all(np.isfinite(f_r1)) and np.all(np.isfinite(f_rnp))
    ):
        raise RuntimeError(
            f"Graph model nlist-ref forces are degenerate or non-finite "
            f"(max={max_ref_force_pbc:.2e}); weights may need perturbation, "
            f"or the AOTI compile is broken (known inductor CPU-SIMD bug; "
            f"workaround: torch._inductor.config.cpp.simdlen = 1)."
        )

    # ---- B.3  Export graph-form .pt2 ----
    # For DPA2, has_message_passing_across_ranks is True, so the
    # lower_kind="graph" export automatically embeds the nested with-comm
    # artifact (forward_lower_with_comm.pt2) alongside the graph forward --
    # no separate export call is needed.
    graph_pt2_path = os.path.join(base_dir, "deeppot_dpa2_graph.pt2")
    print(f"Exporting to {graph_pt2_path} (lower_kind='graph') ...")  # noqa: T201
    pt_expt_deserialize_to_file(
        graph_pt2_path,
        copy.deepcopy(data_g),
        do_atomic_virial=True,
        lower_kind="graph",
    )
    print("Graph .pt2 export done.")  # noqa: T201

    # ---- B.4  Cross-check: graph .pt2 vs independent nlist reference ----
    # Both use the SAME weights; at non-binding sel the math is equivalent.
    # Atomic energies, forces and the TOTAL virial must agree.  The per-atom
    # virial is deliberately NOT compared: the graph path assigns each edge's
    # virial contribution fully to the source atom, which for a
    # message-passing model is a different (equally valid) decomposition
    # than the dense path's -- only the sum is convention-independent.
    dp_graph = DeepPot(graph_pt2_path)

    e_g1, f_g1, v_g1, ae_g1, av_g1 = dp_graph.eval(coord, box, atype, atomic=True)
    e_gnp, f_gnp, v_gnp, ae_gnp, av_gnp = dp_graph.eval(coord, None, atype, atomic=True)

    cross_tol = 1e-8
    for label, (f_g, ae_g, v_g), (f_r, ae_r, v_r) in (
        ("PBC", (f_g1, ae_g1, v_g1), (f_r1, ae_r1, v_r1)),
        ("NoPBC", (f_gnp, ae_gnp, v_gnp), (f_rnp, ae_rnp, v_rnp)),
    ):
        f_diff = float(np.max(np.abs(f_g[0] - f_r[0])))
        ae_diff = float(np.max(np.abs(ae_g[0] - ae_r[0])))
        v_diff = float(np.max(np.abs(v_g[0] - v_r[0])))
        print(  # noqa: T201
            f"Graph .pt2 vs nlist ref {label}: ae {ae_diff:.2e}, "
            f"f {f_diff:.2e}, total-virial {v_diff:.2e}"
        )
        # NaN-safe: NaN fails ``<=``
        if not (f_diff <= cross_tol and ae_diff <= cross_tol and v_diff <= cross_tol):
            raise RuntimeError(
                f"BLOCKED: graph .pt2 {label} differs from nlist reference "
                f"(ae {ae_diff:.2e}, f {f_diff:.2e}, v {v_diff:.2e}; "
                f"threshold {cross_tol:.0e})."
            )

    # ---- B.5  Write sidecar reference file from the graph .pt2 eval ----
    # Self-referential like the dense fixtures' .expected (the C++ gtest is a
    # regression test of the C++ inference path against the Python eval of
    # the same artifact); independence from the graph path is enforced above
    # in B.4.  Sourcing e/f/v from the nlist artifact instead would break the
    # per-atom virial comparison (convention, see B.4) and sit at the gtest's
    # 1e-10 double tolerance for energies/forces (cross-path noise ~1e-10).
    graph_ref_path = os.path.join(base_dir, "deeppot_dpa2_graph.expected")
    write_expected_ref(
        graph_ref_path,
        sections={
            "pbc": {
                "expected_e": ae_g1[0, :, 0],
                "expected_f": f_g1[0],
                "expected_v": av_g1[0],
            },
            "nopbc": {
                "expected_e": ae_gnp[0, :, 0],
                "expected_f": f_gnp[0],
                "expected_v": av_gnp[0],
            },
        },
        source_script="source/tests/infer/gen_dpa2.py",
    )
    print(f"Wrote {graph_ref_path}")  # noqa: T201

    print("\nAll graph sanity checks passed.")  # noqa: T201
    print("\nDone!")  # noqa: T201


if __name__ == "__main__":
    main()
