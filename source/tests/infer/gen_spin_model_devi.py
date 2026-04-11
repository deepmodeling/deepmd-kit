#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate deeppot_dpa_spin_md0.pt2 and deeppot_dpa_spin_md1.pt2 test models.

The canonical model weights are stored in .yaml files (dpmodel serialization,
committed to git).  This script converts them to .pt2 format.

If the .yaml files do not yet exist, they are created from dpmodel with
different seeds — but this should only be done once (the .yaml files are
then committed).

Prints reference values for C++ tests.
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

# Model config (same architecture as gen_spin.py, different seeds)
_BASE_CONFIG = {
    "type_map": ["Ni", "O", "H"],
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
    },
    "fitting_net": {
        "neuron": [5, 5, 5],
        "resnet_dt": True,
    },
    "spin": {
        "use_spin": [True, False, False],
        "virtual_scale": [0.3140, 0.0, 0.0],
    },
}


def _build_yaml(yaml_path: str, seed: int) -> None:
    """Build a dpmodel with given seed and save as .yaml."""
    from deepmd.dpmodel.model.model import (
        get_model,
    )
    from deepmd.dpmodel.utils.serialization import (
        save_dp_model,
    )

    cfg = copy.deepcopy(_BASE_CONFIG)
    cfg["descriptor"]["seed"] = seed
    cfg["fitting_net"]["seed"] = seed
    model = get_model(cfg)
    model_dict = model.serialize()

    data = {
        "model": model_dict,
        "model_def_script": cfg,
        "backend": "dpmodel",
        "software": "deepmd-kit",
        "version": "3.0.0",
    }

    print(f"Building dpmodel (seed={seed}) and saving to {yaml_path} ...")  # noqa: T201
    save_dp_model(yaml_path, data)


def main():
    from deepmd.entrypoints.convert_backend import (
        convert_backend,
    )

    ensure_inductor_compiler()

    base_dir = os.path.dirname(__file__)

    # ---- 1. Ensure .yaml files exist ----
    seeds = [1, 2]
    yaml_paths = []
    pt2_paths = []
    for idx, seed in enumerate(seeds):
        yaml_path = os.path.join(base_dir, f"deeppot_dpa_spin_md{idx}.yaml")
        yaml_paths.append(yaml_path)
        if not os.path.exists(yaml_path):
            _build_yaml(yaml_path, seed)
        else:
            print(f"Using existing {yaml_path}")  # noqa: T201

    # ---- 2. Convert .yaml -> .pt2 ----
    # Import deepmd.pt to register the backend
    import deepmd.pt  # noqa: F401

    load_custom_ops()

    for idx, yaml_path in enumerate(yaml_paths):
        pt2_path = os.path.join(base_dir, f"deeppot_dpa_spin_md{idx}.pt2")
        pt2_paths.append(pt2_path)
        print(f"Converting to {pt2_path} ...")  # noqa: T201
        convert_backend(INPUT=yaml_path, OUTPUT=pt2_path)

    print("Export done.")  # noqa: T201

    # ---- 3. Run inference for both models ----
    from deepmd.infer import (
        DeepPot,
    )

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

    for idx, pt2_path in enumerate(pt2_paths):
        dp = DeepPot(pt2_path)
        e, f, v, ae, av, fm, _ = dp.eval(coord, box, atype, atomic=True, spin=spin)
        print(f"\n// Model {idx} total energy: {e[0, 0]:.18e}")  # noqa: T201
        print_cpp_spin_values(f"Model {idx} reference values", ae, f, fm, v, av)

    # ---- 4. Also print LAMMPS 4-atom system reference values ----
    lmp_coord = np.array(
        [
            12.83,
            2.56,
            2.18,
            12.09,
            2.87,
            2.74,
            3.51,
            2.51,
            2.60,
            4.27,
            3.22,
            1.56,
        ],
        dtype=np.float64,
    )
    lmp_spin = np.array(
        [
            0,
            0,
            1.2737,
            0,
            0,
            1.2737,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        dtype=np.float64,
    )
    lmp_atype = [0, 0, 1, 1]
    lmp_box = np.array(
        [13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0], dtype=np.float64
    )

    print("\n// ---- LAMMPS 4-atom system (PBC) ----")  # noqa: T201
    for idx, pt2_path in enumerate(pt2_paths):
        dp = DeepPot(pt2_path)
        e, f, v, ae, av, fm, _ = dp.eval(
            lmp_coord, lmp_box, lmp_atype, atomic=True, spin=lmp_spin
        )
        print(f"\n// LAMMPS Model {idx} total energy: {e[0, 0]:.18e}")  # noqa: T201
        print(f"// LAMMPS Model {idx} force:")  # noqa: T201
        for ii in range(4):
            print(f"//   [{f[0, ii, 0]:.16e}, {f[0, ii, 1]:.16e}, {f[0, ii, 2]:.16e}]")  # noqa: T201
        print(f"// LAMMPS Model {idx} force_mag:")  # noqa: T201
        for ii in range(4):
            msg = (
                f"//   [{fm[0, ii, 0]:.16e}, {fm[0, ii, 1]:.16e}, {fm[0, ii, 2]:.16e}]"
            )
            print(msg)  # noqa: T201

    # NoPBC for LAMMPS
    print("\n// ---- LAMMPS 4-atom system (NoPBC) ----")  # noqa: T201
    for idx, pt2_path in enumerate(pt2_paths):
        dp = DeepPot(pt2_path)
        e, f, v, ae, av, fm, _ = dp.eval(
            lmp_coord, None, lmp_atype, atomic=True, spin=lmp_spin
        )
        print(f"\n// LAMMPS NoPBC Model {idx} total energy: {e[0, 0]:.18e}")  # noqa: T201
        print(f"// LAMMPS NoPBC Model {idx} force:")  # noqa: T201
        for ii in range(4):
            print(f"//   [{f[0, ii, 0]:.16e}, {f[0, ii, 1]:.16e}, {f[0, ii, 2]:.16e}]")  # noqa: T201
        print(f"// LAMMPS NoPBC Model {idx} force_mag:")  # noqa: T201
        for ii in range(4):
            msg = (
                f"//   [{fm[0, ii, 0]:.16e}, {fm[0, ii, 1]:.16e}, {fm[0, ii, 2]:.16e}]"
            )
            print(msg)  # noqa: T201

    print("\nDone!")  # noqa: T201


if __name__ == "__main__":
    main()
