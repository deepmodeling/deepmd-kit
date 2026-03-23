#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate deeppot_dpa3.pth and deeppot_dpa3.pt2 test models.

Creates a DPA3 model from dpmodel config,
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
    print_cpp_values,
)


def main():
    from deepmd.dpmodel.model.model import (
        get_model,
    )

    ensure_inductor_compiler()

    # ---- 1. DPA3 model config (small, fast to compile) ----
    config = {
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "dpa3",
            "repflow": {
                "n_dim": 8,
                "e_dim": 5,
                "a_dim": 4,
                "nlayers": 2,
                "e_rcut": 6.0,
                "e_rcut_smth": 2.0,
                "e_sel": 30,
                "a_rcut": 4.0,
                "a_rcut_smth": 2.0,
                "a_sel": 20,
                "axis_neuron": 4,
                "update_angle": True,
                "update_style": "res_residual",
                "update_residual_init": "const",
                "smooth_edge_update": True,
            },
            "concat_output_tebd": True,
            "precision": "float64",
            "seed": 1,
        },
        "fitting_net": {"neuron": [5, 5, 5], "resnet_dt": True, "seed": 1},
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

    pt2_path = os.path.join(base_dir, "deeppot_dpa3.pt2")
    print(f"Exporting to {pt2_path} ...")  # noqa: T201
    pt_expt_deserialize_to_file(pt2_path, copy.deepcopy(data))

    pth_path = os.path.join(base_dir, "deeppot_dpa3.pth")
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
    print_cpp_values("PBC reference values", ae1, f1, av1)

    # ---- 5. Run inference for NoPbc test ----
    e_np, f_np, v_np, ae_np, av_np = dp.eval(coord, None, atype, atomic=True)
    print(f"\n// NoPbc total energy: {e_np[0, 0]:.18e}")  # noqa: T201
    print_cpp_values("NoPbc reference values", ae_np, f_np, av_np)

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
