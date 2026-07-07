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
    pt_expt_deserialize_to_file(pt2_path, copy.deepcopy(data), do_atomic_virial=True)

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

    print("\nDone!")  # noqa: T201


if __name__ == "__main__":
    main()
