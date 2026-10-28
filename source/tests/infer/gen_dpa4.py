#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate deeppot_dpa4.pth and deeppot_dpa4.pt2 test models.

Creates a DPA4/SeZM model from a pt_expt config, serializes, and exports
to both .pt2 (pt_expt / AOTInductor) and .pth (pt) from the same weights.
Also writes a sidecar reference file (PBC and NoPbc per-atom energy/force/
virial) consumed by the C++ tests.
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
    import torch

    from deepmd.pt_expt.model.get_model import (
        get_model,
    )

    ensure_inductor_compiler()

    # ---- 1. DPA4/SeZM model config (small, fast to compile) ----
    # Mirrors test_dpa4_export.py: channels 16, n_radial 8, lmax 2, mmax 1,
    # n_blocks 2 — large enough to exercise the SO(2)/SO(3) + attention +
    # embedding paths, small enough to keep the AOTInductor compile bounded.
    config = {
        "type": "dpa4",
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "dpa4",
            "sel": 20,
            "rcut": 4.0,
            "channels": 16,
            "n_radial": 8,
            "lmax": 2,
            "mmax": 1,
            "n_blocks": 2,
            "precision": "float64",
            "seed": 1,
        },
        "fitting_net": {
            "type": "dpa4_ener",
            "neuron": [16],
            "precision": "float64",
            "seed": 1,
        },
    }

    # ---- 2. Build the pt_expt model and serialize ----
    # dpmodel ``get_model`` has no DPA4 dispatch; the model-type alias lives
    # in pt_expt ``get_model``.  Build there, then serialize to a backend-
    # neutral dict that both pt_expt and pt can deserialize.
    model = get_model(copy.deepcopy(config))
    model.to("cpu")
    model.eval()

    # ---- 2b. Activate the zero-initialised residual branches ----
    # DPA4/SeZM follows the standard residual-network convention of
    # ZERO-initialising the output projection of every residual branch
    # (``*.so3_linear_2.weight``, ``post_focus_mix.weight``,
    # ``env_seed_embedding.output_proj.w``) and the final descriptor output
    # projection (``output_ffn.so3_linear_2.weight``).  At random init these
    # branches therefore contribute EXACTLY zero, so a freshly built DPA4
    # collapses to a type-embedding-only descriptor: the per-atom energy is a
    # pure per-type constant and every force/virial is identically zero,
    # regardless of geometry.  Such a fixture exercises none of the
    # force/virial code paths and would make the C++ inference test vacuous.
    #
    # A trained model has non-zero weights in these branches, so to obtain a
    # representative (geometry-dependent, non-zero-force) reference we fill the
    # all-zero parameters with small deterministic pseudo-random values.  This
    # is the minimal change that makes the descriptor coordinate-dependent
    # while leaving the rest of the random init untouched.  (Unlike DPA3,
    # whose random init already yields non-zero forces, DPA4 needs this step.)
    generator = torch.Generator().manual_seed(20240614)
    with torch.no_grad():
        for _name, param in model.named_parameters():
            if float(param.detach().abs().max()) == 0.0:
                param.copy_(
                    0.1
                    * torch.randn(param.shape, dtype=param.dtype, generator=generator)
                )

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

    pt2_path = os.path.join(base_dir, "deeppot_dpa4.pt2")
    print(f"Exporting to {pt2_path} ...")  # noqa: T201
    # Pinned explicitly: DPA4 is graph-native end-to-end, so
    # ``lower_kind="auto"`` now resolves to "graph" (_resolve_lower_kind).
    # This dense fixture must keep testing the dense-nlist ABI regardless of
    # that default; the graph ABI is exercised separately by Section B below.
    pt_expt_deserialize_to_file(
        pt2_path, copy.deepcopy(data), do_atomic_virial=True, lower_kind="nlist"
    )

    pth_path = os.path.join(base_dir, "deeppot_dpa4.pth")
    print(f"Exporting to {pth_path} ...")  # noqa: T201
    try:
        pt_deserialize_to_file(pth_path, copy.deepcopy(data))
    except RuntimeError as e:
        # Custom ops may not be available in all build environments;
        # .pth generation is not critical.
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
    ref_path = os.path.join(base_dir, "deeppot_dpa4.expected")
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
        source_script="source/tests/infer/gen_dpa4.py",
    )
    print(f"Wrote {ref_path}")  # noqa: T201

    # ---- 6. Verify .pth gives same results ----
    if os.path.exists(pth_path):
        dp_pth = DeepPot(pth_path)
        e_pth, f_pth, v_pth, ae_pth, av_pth = dp_pth.eval(
            coord, box, atype, atomic=True
        )
        # PBC parity assertions
        pbc_e_diff = abs(e1[0, 0] - e_pth[0, 0])
        pbc_f_diff = np.max(np.abs(f1 - f_pth))
        pbc_v_diff = np.max(np.abs(v1 - v_pth))
        print(f"\n// .pth PBC total energy: {e_pth[0, 0]:.18e}")  # noqa: T201
        print(f"// .pth vs .pt2 energy diff: {pbc_e_diff:.2e}")  # noqa: T201
        print(f"// .pth vs .pt2 force max diff: {pbc_f_diff:.2e}")  # noqa: T201
        print(f"// .pth vs .pt2 virial max diff: {pbc_v_diff:.2e}")  # noqa: T201
        tol = 1e-10
        assert pbc_e_diff < tol, f"PBC energy parity failed: diff={pbc_e_diff:.2e}"
        assert pbc_f_diff < tol, f"PBC force parity failed: diff={pbc_f_diff:.2e}"
        # NOTE: ``v1``/``v_pth`` are the *global* virials (3rd return value).
        # The per-atom virial distribution legitimately differs between pt's
        # edge-force scatter and pt_expt's generic assembly (#5518); only the
        # global virial (their sum) is a physical observable, so we assert on
        # the global virial here.
        assert pbc_v_diff < tol, f"PBC virial parity failed: diff={pbc_v_diff:.2e}"

        e_pth_np, f_pth_np, v_pth_np, ae_pth_np, av_pth_np = dp_pth.eval(
            coord, None, atype, atomic=True
        )
        # NoPbc parity assertions
        np_e_diff = abs(e_np[0, 0] - e_pth_np[0, 0])
        np_f_diff = np.max(np.abs(f_np - f_pth_np))
        np_v_diff = np.max(np.abs(v_np - v_pth_np))
        print(f"// .pth NoPbc total energy: {e_pth_np[0, 0]:.18e}")  # noqa: T201
        print(f"// .pth vs .pt2 NoPbc energy diff: {np_e_diff:.2e}")  # noqa: T201
        print(f"// .pth vs .pt2 NoPbc force diff: {np_f_diff:.2e}")  # noqa: T201
        print(f"// .pth vs .pt2 NoPbc virial diff: {np_v_diff:.2e}")  # noqa: T201
        assert np_e_diff < tol, f"NoPbc energy parity failed: diff={np_e_diff:.2e}"
        assert np_f_diff < tol, f"NoPbc force parity failed: diff={np_f_diff:.2e}"
        assert np_v_diff < tol, f"NoPbc virial parity failed: diff={np_v_diff:.2e}"
    else:
        print("\n// Skipping .pth verification (file not generated).")  # noqa: T201

    # ============================================================
    # Section B: graph .pt2 export (jittered weights, non-vacuous)
    # ============================================================
    # DPA4 is graph-native end-to-end: there is no dense-only sub-block to
    # toggle off for graph-eligibility (unlike DPA2's use_three_body), so
    # the SAME config as Section A is graph-eligible and, before the pin
    # above, ``lower_kind="auto"`` already resolved to "graph" for it (see
    # deepmd/pt_expt/utils/serialization.py:_resolve_lower_kind).
    #
    # Skip the whole graph section under LeakSanitizer. The C++ memleak
    # matrix runs these gen scripts with ``LD_PRELOAD=liblsan`` (the
    # sanitizer-instrumented deepmd op .so requires the LSAN runtime; see
    # source/install/test_cc_local.sh). Evaluating the AOTInductor-compiled
    # graph .pt2's BACKWARD (forces) under that runtime INTERMITTENTLY
    # segfaults -- an AOTI-compiled-code vs LeakSanitizer allocator
    # incompatibility, NOT a graph-code bug (see gen_dpa2.py's identical
    # section for the full rationale -- dpa2's repformer trips the same
    # issue). The dense DPA4 .pt2 (Section A) is unaffected and still
    # generated. The C++ dpa4_graph_pytorch_pt2 row GTEST_SKIPs when this
    # artifact is absent (skip_if_artifact_missing).
    #
    # Detection is via the explicit DP_GEN_UNDER_SANITIZER flag set by
    # test_cc_local.sh next to the preload: sniffing LD_PRELOAD here does
    # NOT reliably work -- the LSAN runtime removes its own entry from the
    # process environment during startup on some platforms. The LD_PRELOAD
    # check is kept only as a belt-and-braces fallback for manual
    # invocations where the runtime leaves the variable intact.
    if (
        os.environ.get("DP_GEN_UNDER_SANITIZER", "") == "lsan"
        or "lsan" in os.environ.get("LD_PRELOAD", "").lower()
    ):
        # remove any graph artifacts left by a previous non-LSAN run of a
        # REUSED workspace: skipping regeneration alone leaves them present,
        # and the C++ tests' skip_if_artifact_missing would then execute
        # them under LSAN and hit the very crash this branch avoids
        for name in (
            "deeppot_dpa4_graph_nlist_ref.pt2",
            "deeppot_dpa4_graph.pt2",
            "deeppot_dpa4_graph.expected",
        ):
            stale = os.path.join(base_dir, name)
            if os.path.exists(stale):
                os.remove(stale)
        print(  # noqa: T201
            "\n// Skipping DPA4 graph section under LeakSanitizer "
            "(AOTInductor .pt2 backward is incompatible with the LSAN runtime; "
            "covered by the non-memleak C++/LAMMPS matrix)."
        )
        print("\nDone!")  # noqa: T201
        return

    print("\n---- Building graph DPA4 (jittered weights) ----")  # noqa: T201

    # ---- B.1  Build a fresh DPA4 model; jitter zero-init residuals ----
    # A freshly built DPA4 zero-initializes several residual output
    # projections (see step 2b above and the ``jitter_zero_arrays``
    # docstring in source/tests/dpa4_fixtures.py), so its output is
    # architecturally edge-independent until those branches are perturbed
    # away from exactly zero. Section A already does this via in-place
    # torch-parameter replacement (step 2b); this section instead follows
    # the dict-level ``jitter_zero_arrays`` pattern used by
    # test_dpa4_call_graph.py / test_dpa4_graph_lower.py, for consistency
    # with the rest of the DPA4 graph test suite. Inlined here (rather than
    # imported from source/tests/dpa4_fixtures.py) because gen_dpa4.py is a
    # standalone script run outside pytest's package machinery -- importing
    # a source/tests/... module from it would need ad hoc sys.path /
    # package surgery for no real benefit.
    # Mirror of source/tests/dpa4_fixtures.py:jitter_zero_arrays -- keep in sync.
    def _jitter_zero_arrays(node, rng: np.random.Generator):
        # Mirror of source/tests/dpa4_fixtures.py:jitter_zero_arrays -- keep in
        # sync. PURE rebuild (returns a new tree, does not mutate ``node``) to
        # avoid CodeQL's py/modification-of-default-value dataflow; behavior
        # (RNG draws, shapes, dtype) is bit-identical.
        if isinstance(node, dict):
            return {k: _jitter_zero_arrays(v, rng) for k, v in node.items()}
        if isinstance(node, list):
            return [_jitter_zero_arrays(v, rng) for v in node]
        if (
            isinstance(node, np.ndarray)
            and node.dtype.kind == "f"
            and node.size > 0
            and np.all(node == 0.0)
        ):
            return rng.normal(0.0, 0.05, size=node.shape).astype(node.dtype)
        return node

    model_g = get_model(copy.deepcopy(config))
    model_g.to("cpu")
    model_g.eval()
    model_dict_g = model_g.serialize()
    model_dict_g = _jitter_zero_arrays(model_dict_g, np.random.default_rng(20240615))

    data_g = {
        "model": copy.deepcopy(model_dict_g),
        "model_def_script": config,
        "backend": "dpmodel",
        "software": "deepmd-kit",
        "version": "3.0.0",
    }

    # ---- B.2  Independent cross-check via nlist .pt2 (dense-quartet) ----
    # Like gen_dpa2.py's Section B.2: deeppot_dpa4_graph.expected is NOT
    # copied from the nlist artifact (see B.5), since DPA4's graph path
    # assigns each edge's force/virial contribution fully to the source atom
    # (edge_force_virial full-to-src), a different (equally valid)
    # decomposition than the dense per-atom one -- only the SUM agrees. The
    # nlist .pt2 is instead used here as the independent gen-time oracle:
    # atomic energies, forces and the TOTAL virial of the graph .pt2 must
    # match it (checked in B.4) or generation aborts.
    #
    # The nlist .pt2 is PERSISTED (deeppot_dpa4_graph_nlist_ref.pt2), reused
    # directly by the LAMMPS graph-vs-nlist-ref test and by the DeepEval
    # graph parity test, both exercised on the SAME jittered weights as the
    # graph model, so at non-binding sel the two paths must agree.
    nlist_ref_pt2 = os.path.join(base_dir, "deeppot_dpa4_graph_nlist_ref.pt2")
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
    # Anti-vacuity guard: a fresh DPA4 is edge-independent (zero-init
    # residual projections), so a broken (or accidentally skipped) jitter
    # above would silently produce a degenerate, geometry-insensitive
    # fixture. ``not (x >= th)`` (rather than ``x < th``) so NaN forces --
    # e.g. from an inductor SIMD miscompile of the AOTI artifact -- fail the
    # check instead of slipping through.
    if (
        not (max_ref_force_pbc > 1e-6)
        or not (max_ref_force_nopbc > 1e-6)
        or not (np.all(np.isfinite(f_r1)) and np.all(np.isfinite(f_rnp)))
    ):
        raise RuntimeError(
            f"BLOCKED: graph DPA4 nlist-ref forces are degenerate or "
            f"non-finite (PBC max={max_ref_force_pbc:.2e}, "
            f"NoPBC max={max_ref_force_nopbc:.2e}); the zero-init-residual "
            f"jitter may have failed to perturb the descriptor, or the AOTI "
            f"compile is broken (known inductor CPU-SIMD bug; workaround: "
            f"torch._inductor.config.cpp.simdlen = 1)."
        )

    # ---- B.3  Export graph-form .pt2 (SAME jittered weights) ----
    graph_pt2_path = os.path.join(base_dir, "deeppot_dpa4_graph.pt2")
    print(f"Exporting to {graph_pt2_path} (lower_kind='graph') ...")  # noqa: T201
    # has_message_passing_across_ranks() is True -> the graph export
    # auto-embeds model/extra/forward_lower_with_comm.pt2 (multi-rank
    # LAMMPS).
    pt_expt_deserialize_to_file(
        graph_pt2_path,
        copy.deepcopy(data_g),
        do_atomic_virial=True,
        lower_kind="graph",
    )
    print("Graph .pt2 export done.")  # noqa: T201

    # ---- B.4  Cross-check: graph .pt2 vs independent nlist reference ----
    # Both use the SAME weights; at non-binding sel the math is equivalent.
    # Atomic energies, forces and the TOTAL virial must agree. The per-atom
    # virial is deliberately NOT compared: see B.2.
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
    # in B.4. Sourcing e/f/v from the nlist artifact instead would break the
    # per-atom virial comparison (convention, see B.2) and sit at the
    # gtest's 1e-10 double tolerance for energies/forces (cross-path noise
    # is only checked to 1e-8 here).
    graph_ref_path = os.path.join(base_dir, "deeppot_dpa4_graph.expected")
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
        source_script="source/tests/infer/gen_dpa4.py",
    )
    print(f"Wrote {graph_ref_path}")  # noqa: T201

    print("\nAll graph sanity checks passed.")  # noqa: T201

    print("\nDone!")  # noqa: T201


if __name__ == "__main__":
    main()
