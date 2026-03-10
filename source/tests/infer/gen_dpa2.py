#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate deeppot_dpa2.pth and deeppot_dpa2.pt2 test models.

Creates a DPA2 model from dpmodel config (with three-body, type_one_side=True),
serializes, and exports to both .pth and .pt2 from the same weights.
Also prints reference values for C++ tests (PBC and NoPbc).
"""

import copy
import glob
import os
import shutil
import sys

import numpy as np

# Ensure the source tree is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _ensure_inductor_compiler():
    """Ensure torch._inductor can find a C++ compiler.

    torch._inductor searches for 'g++' by default.  On some CI images only
    versioned binaries (e.g. g++-11) or 'c++' exist.  Fall back to those.
    """
    import torch._inductor.config as inductor_config

    search = inductor_config.cpp.cxx
    if isinstance(search, (list, tuple)):
        search = list(search)
    else:
        search = [search]
    # Append common fallbacks that are not in the default search list
    for fallback in ["c++", "g++-14", "g++-13", "g++-12", "g++-11"]:
        if fallback not in search and shutil.which(fallback):
            search.append(fallback)
    inductor_config.cpp.cxx = tuple(search)


def _load_custom_ops():
    """Load custom op library if not already registered.

    Must be called AFTER importing deepmd (which may register ops from the
    pip-installed library) to avoid double-registration crashes.
    """
    import torch

    if hasattr(torch.ops, "deepmd") and hasattr(torch.ops.deepmd, "border_op"):
        return
    _search_base = os.path.realpath(os.path.dirname(__file__))
    for pattern in [
        os.path.join(
            _search_base, "..", "..", "..", "build*", "op", "pt", "libdeepmd_op_pt.so"
        ),
        os.path.join(
            _search_base, "..", "..", "build*", "op", "pt", "libdeepmd_op_pt.so"
        ),
    ]:
        libs = glob.glob(pattern)
        if libs:
            try:
                torch.ops.load_library(libs[0])
            except Exception as e:
                print(f"NOTE: custom op library not loaded ({e})", file=sys.stderr)  # noqa: T201
            break


def print_cpp_values(label, ae, f, av):
    """Print C++ reference arrays."""
    print(f"\n// ---- {label} ----")  # noqa: T201
    atom_energy = ae[0, :, 0]
    print("  std::vector<VALUETYPE> expected_e = {")  # noqa: T201
    for ii, e in enumerate(atom_energy):
        comma = "," if ii < len(atom_energy) - 1 else ""
        print(f"      {e:.18e}{comma}")  # noqa: T201
    print("  };")  # noqa: T201

    print("  std::vector<VALUETYPE> expected_f = {")  # noqa: T201
    force_flat = f[0].flatten()
    for ii, fv in enumerate(force_flat):
        comma = "," if ii < len(force_flat) - 1 else ""
        print(f"      {fv:.18e}{comma}")  # noqa: T201
    print("  };")  # noqa: T201

    print("  std::vector<VALUETYPE> expected_v = {")  # noqa: T201
    virial_flat = av[0].flatten()
    for ii, v in enumerate(virial_flat):
        comma = "," if ii < len(virial_flat) - 1 else ""
        print(f"      {v:.18e}{comma}")  # noqa: T201
    print("  };")  # noqa: T201


def main():
    from deepmd.dpmodel.model.model import (
        get_model,
    )

    # Load custom ops after deepmd import to avoid double registration
    _load_custom_ops()
    _ensure_inductor_compiler()

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

    base_dir = os.path.dirname(__file__)

    pt2_path = os.path.join(base_dir, "deeppot_dpa2.pt2")
    print(f"Exporting to {pt2_path} ...")  # noqa: T201
    pt_expt_deserialize_to_file(pt2_path, copy.deepcopy(data))

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

    print("\nDone!")  # noqa: T201


if __name__ == "__main__":
    main()
