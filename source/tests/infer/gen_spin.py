#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate deeppot_dpa_spin.pth and deeppot_dpa_spin.pt2 test models.

The canonical model weights are stored in ``deeppot_dpa_spin.yaml`` (dpmodel
serialization, committed to git).  This script converts the .yaml to both
.pth (torch.jit) and .pt2 (torch.export) formats.

If the .yaml does not yet exist, it is created from a dpmodel built with
a deterministic config+seed — but this should only be done once (the .yaml
is then committed).

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
    print_cpp_spin_values,
)


def _build_yaml(yaml_path: str) -> None:
    """Build the dpmodel from config+seed and save as .yaml."""
    from deepmd.dpmodel.model.model import (
        get_model,
    )
    from deepmd.dpmodel.utils.serialization import (
        save_dp_model,
    )

    config = {
        "type_map": ["Ni", "O"],
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
        "spin": {
            "use_spin": [True, False],
            "virtual_scale": [0.3140, 0.0],
        },
    }

    model = get_model(copy.deepcopy(config))
    model_dict = model.serialize()

    data = {
        "model": model_dict,
        "model_def_script": config,
        "backend": "dpmodel",
        "software": "deepmd-kit",
        "version": "3.0.0",
    }

    print(f"Building dpmodel and saving to {yaml_path} ...")  # noqa: T201
    save_dp_model(yaml_path, data)


def main():
    from deepmd.entrypoints.convert_backend import (
        convert_backend,
    )

    ensure_inductor_compiler()

    base_dir = os.path.dirname(__file__)
    yaml_path = os.path.join(base_dir, "deeppot_dpa_spin.yaml")
    pth_path = os.path.join(base_dir, "deeppot_dpa_spin.pth")
    pt2_path = os.path.join(base_dir, "deeppot_dpa_spin.pt2")

    # ---- 1. Build .yaml if it doesn't exist ----
    if not os.path.exists(yaml_path):
        _build_yaml(yaml_path)
    else:
        print(f"Using existing {yaml_path}")  # noqa: T201

    # ---- 2. Convert .yaml -> .pth and .yaml -> .pt2 ----
    # Import deepmd.pt to register the backend (needed for convert_backend)
    import deepmd.pt  # noqa: F401

    load_custom_ops()

    print(f"Converting to {pth_path} ...")  # noqa: T201
    convert_backend(INPUT=yaml_path, OUTPUT=pth_path)

    print(f"Converting to {pt2_path} ...")  # noqa: T201
    convert_backend(INPUT=yaml_path, OUTPUT=pt2_path)

    print("Export done.")  # noqa: T201

    # ---- 3. Run inference for PBC test ----
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
    spin = np.array(
        [
            0.13,
            0.02,
            0.03,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.14,
            0.10,
            0.12,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )
    atype = [0, 1, 1, 0, 1, 1]
    box = np.array([13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0], dtype=np.float64)

    e1, f1, v1, ae1, av1, fm1, _ = dp.eval(coord, box, atype, atomic=True, spin=spin)
    print(f"\n// PBC total energy: {e1[0, 0]:.18e}")  # noqa: T201
    print_cpp_spin_values("PBC reference values", ae1, f1, fm1, v1, av1)

    # ---- 4. Run inference for NoPbc test ----
    e_np, f_np, v_np, ae_np, av_np, fm_np, _ = dp.eval(
        coord, None, atype, atomic=True, spin=spin
    )
    print(f"\n// NoPbc total energy: {e_np[0, 0]:.18e}")  # noqa: T201
    print_cpp_spin_values("NoPbc reference values", ae_np, f_np, fm_np, v_np, av_np)

    # ---- 5. Verify .pth gives same results ----
    if os.path.exists(pth_path):
        dp_pth = DeepPot(pth_path)
        e_pth, f_pth, v_pth, ae_pth, av_pth, fm_pth, _ = dp_pth.eval(
            coord, box, atype, atomic=True, spin=spin
        )
        print(f"\n// .pth PBC total energy: {e_pth[0, 0]:.18e}")  # noqa: T201
        print(f"// .pth vs .pt2 energy diff: {abs(e1[0, 0] - e_pth[0, 0]):.2e}")  # noqa: T201
        print(f"// .pth vs .pt2 force max diff: {np.max(np.abs(f1 - f_pth)):.2e}")  # noqa: T201
        print(f"// .pth vs .pt2 force_mag max diff: {np.max(np.abs(fm1 - fm_pth)):.2e}")  # noqa: T201

    print("\nDone!")  # noqa: T201


if __name__ == "__main__":
    main()
