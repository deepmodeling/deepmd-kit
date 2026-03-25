#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate fparam_aparam.pth and fparam_aparam.pt2 test models.

Creates a dpmodel model from config with type_one_side=True, serializes,
and exports to both .pth and .pt2 from the same weights.
Also prints reference values for C++ tests.
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
)


def main():
    from deepmd.dpmodel.model.model import (
        get_model,
    )

    ensure_inductor_compiler()

    # ---- 1. Model config (type_one_side=True) ----
    config = {
        "type_map": ["O"],
        "descriptor": {
            "type": "se_e2_a",
            "sel": [60],
            "rcut": 6.0,
            "rcut_smth": 1.8,
            "neuron": [5, 10, 20],
            "axis_neuron": 8,
            "activation_function": "tanh",
            "resnet_dt": False,
            "type_one_side": True,
            "exclude_types": [],
            "set_davg_zero": False,
            "precision": "default",
            "trainable": True,
            "seed": 1,
        },
        "fitting_net": {
            "type": "ener",
            "neuron": [5, 5, 5],
            "activation_function": "tanh",
            "resnet_dt": True,
            "numb_fparam": 1,
            "numb_aparam": 1,
            "precision": "default",
            "seed": 1,
            "atom_ener": [],
            "rcond": 0.001,
            "trainable": True,
            "use_aparam_as_mask": False,
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

    pt2_path = os.path.join(base_dir, "fparam_aparam.pt2")
    print(f"Exporting to {pt2_path} ...")  # noqa: T201
    pt_expt_deserialize_to_file(pt2_path, copy.deepcopy(data))

    pth_path = os.path.join(base_dir, "fparam_aparam.pth")
    pth_exported = False
    print(f"Exporting to {pth_path} ...")  # noqa: T201
    try:
        pt_deserialize_to_file(pth_path, copy.deepcopy(data))
        pth_exported = True
    except RuntimeError as e:
        # Custom ops (e.g. tabulate_fusion_se_t_tebd) may not be available
        # in all build environments; .pth generation is not critical.
        print(f"WARNING: .pth export failed ({e}), skipping.")  # noqa: T201

    print("Export done.")  # noqa: T201

    # ---- 3b. Export a model with default_fparam to .pt2 ----
    config_default = copy.deepcopy(config)
    config_default["fitting_net"]["default_fparam"] = [0.25852028]
    model_default = get_model(config_default)
    data_default = {
        "model": model_default.serialize(),
        "model_def_script": config_default,
        "backend": "dpmodel",
        "software": "deepmd-kit",
        "version": "3.0.0",
    }
    pt2_default_path = os.path.join(base_dir, "fparam_aparam_default.pt2")
    print(f"Exporting to {pt2_default_path} ...")  # noqa: T201
    pt_expt_deserialize_to_file(pt2_default_path, copy.deepcopy(data_default))

    # ---- 4. Run inference via DeepPot to get reference values ----
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
    atype = [0, 0, 0, 0, 0, 0]
    box = np.array([13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0], dtype=np.float64)
    fparam_val = np.array([0.25852028], dtype=np.float64)
    aparam_val = np.array([0.25852028] * 6, dtype=np.float64)

    e, f, v, ae, av = dp.eval(
        coord,
        box,
        atype,
        fparam=fparam_val,
        aparam=aparam_val,
        atomic=True,
    )

    atom_energy = ae[0, :, 0]
    force = f[0]
    atom_virial = av[0]

    # Print C++ format
    print("\n// ---- Reference values for C++ test (shared by .pth and .pt2) ----")  # noqa: T201
    print(f"// Total energy: {e[0, 0]:.18e}")  # noqa: T201
    print()  # noqa: T201
    print("  std::vector<VALUETYPE> expected_e = {")  # noqa: T201
    for ii, ev in enumerate(atom_energy):
        comma = "," if ii < len(atom_energy) - 1 else ""
        print(f"      {ev:.18e}{comma}")  # noqa: T201
    print("  };")  # noqa: T201

    print("  std::vector<VALUETYPE> expected_f = {")  # noqa: T201
    force_flat = force.flatten()
    for ii, fv in enumerate(force_flat):
        comma = "," if ii < len(force_flat) - 1 else ""
        print(f"      {fv:.18e}{comma}")  # noqa: T201
    print("  };")  # noqa: T201

    print("  std::vector<VALUETYPE> expected_v = {")  # noqa: T201
    virial_flat = atom_virial.flatten()
    for ii, vv in enumerate(virial_flat):
        comma = "," if ii < len(virial_flat) - 1 else ""
        print(f"      {vv:.18e}{comma}")  # noqa: T201
    print("  };")  # noqa: T201

    # ---- 5. Verify .pth gives same results ----
    if pth_exported:
        dp_pth = DeepPot(pth_path)
        e_pth, f_pth, v_pth, ae_pth, av_pth = dp_pth.eval(
            coord,
            box,
            atype,
            fparam=fparam_val,
            aparam=aparam_val,
            atomic=True,
        )
        tol = 1e-10
        e_diff = abs(e[0, 0] - e_pth[0, 0])
        f_diff = np.max(np.abs(f - f_pth))
        v_diff = np.max(np.abs(v - v_pth))
        ae_diff = np.max(np.abs(ae - ae_pth))
        av_diff = np.max(np.abs(av - av_pth))
        print(f"\n// .pth vs .pt2 energy diff: {e_diff:.2e}")  # noqa: T201
        print(f"// .pth vs .pt2 force max diff: {f_diff:.2e}")  # noqa: T201
        print(f"// .pth vs .pt2 virial max diff: {v_diff:.2e}")  # noqa: T201
        print(f"// .pth vs .pt2 atom_energy max diff: {ae_diff:.2e}")  # noqa: T201
        print(f"// .pth vs .pt2 atom_virial max diff: {av_diff:.2e}")  # noqa: T201
        assert e_diff < tol, f"Energy parity failed: diff={e_diff:.2e}"
        assert f_diff < tol, f"Force parity failed: diff={f_diff:.2e}"
        assert v_diff < tol, f"Virial parity failed: diff={v_diff:.2e}"
        assert ae_diff < tol, f"Atom energy parity failed: diff={ae_diff:.2e}"
        assert av_diff < tol, f"Atom virial parity failed: diff={av_diff:.2e}"
    else:
        print("\n// Skipping .pth verification (export failed).")  # noqa: T201

    print("\nDone!")  # noqa: T201


if __name__ == "__main__":
    main()
