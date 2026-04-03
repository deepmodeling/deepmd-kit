#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate model_devi_md0.pt2 and model_devi_md1.pt2 test models.

Creates two SE(A) models with fparam/aparam + default_fparam, using different
seeds so they produce different weights. This gives meaningful deviations for
DeepPotModelDevi tests. Prints precomputed reference values for C++ tests.
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

    # ---- 1. Model config (SE(A) + fparam + aparam + default_fparam) ----
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
            "default_fparam": [0.25852028],
            "precision": "default",
            "seed": 1,
            "atom_ener": [],
            "rcond": 0.001,
            "trainable": True,
            "use_aparam_as_mask": False,
        },
    }

    # ---- 2. Build two models with different seeds AND fitting sizes ----
    # Using different fitting neuron sizes ensures meaningfully different outputs
    # (same arch + different seeds produces near-identical small random forces).
    fitting_neurons = [[5, 5, 5], [10, 10, 10]]

    from deepmd.pt.utils.serialization import (  # noqa: F401
        deserialize_to_file,
    )
    from deepmd.pt_expt.utils.serialization import (
        deserialize_to_file as pt_expt_deserialize_to_file,
    )

    # Load custom ops after deepmd.pt import to avoid double registration
    load_custom_ops()

    base_dir = os.path.dirname(__file__)

    models = []
    for idx, seed in enumerate([1, 2]):
        cfg = copy.deepcopy(config)
        cfg["descriptor"]["seed"] = seed
        cfg["fitting_net"]["seed"] = seed
        cfg["fitting_net"]["neuron"] = fitting_neurons[idx]
        model = get_model(cfg)
        model_dict = model.serialize()
        data = {
            "model": model_dict,
            "model_def_script": cfg,
            "backend": "dpmodel",
            "software": "deepmd-kit",
            "version": "3.0.0",
        }
        pt2_path = os.path.join(base_dir, f"model_devi_md{idx}.pt2")
        print(f"Exporting to {pt2_path} ...")  # noqa: T201
        pt_expt_deserialize_to_file(pt2_path, copy.deepcopy(data))
        models.append(pt2_path)

    print("Export done.")  # noqa: T201

    # ---- 3. Run inference via DeepPot ----
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
    atype = [0, 0, 0, 0, 0, 0]
    box = np.array([13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0], dtype=np.float64)
    fparam_val = np.array([0.25852028], dtype=np.float64)
    aparam_val = np.array([0.25852028] * 6, dtype=np.float64)

    # Inference with explicit fparam for both models
    for idx, pt2_path in enumerate(models):
        dp = DeepPot(pt2_path)

        # With explicit fparam + aparam
        e, f, v, ae, av = dp.eval(
            coord,
            box,
            atype,
            fparam=fparam_val,
            aparam=aparam_val,
            atomic=True,
        )

        # Note: default_fparam path is tested at the C++ level;
        # the Python pt2 runner filters None args so it can't be tested here.

        atom_energy = ae[0, :, 0]
        force = f[0]
        atom_virial = av[0]

        print(f"\n// ---- Model {idx} reference values (LAMMPS nlist path) ----")  # noqa: T201
        print(f"// Total energy: {e[0, 0]:.18e}")  # noqa: T201
        print()  # noqa: T201
        print(f"  // model {idx}")  # noqa: T201
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

    # ---- 4. Compute deviation stats (LAMMPS nlist) ----
    dp0 = DeepPot(models[0])
    dp1 = DeepPot(models[1])

    e0, f0, v0, ae0, av0 = dp0.eval(
        coord, box, atype, fparam=fparam_val, aparam=aparam_val, atomic=True
    )
    e1, f1, v1, ae1, av1 = dp1.eval(
        coord, box, atype, fparam=fparam_val, aparam=aparam_val, atomic=True
    )

    nloc = len(atype)
    nmodel = 2

    # std_f: per-atom force std
    f0_flat = f0[0].flatten()  # (nloc*3,)
    f1_flat = f1[0].flatten()
    std_f = np.zeros(nloc)
    for ii in range(nloc):
        avg_f = np.zeros(3)
        for dd in range(3):
            avg_f[dd] = (f0_flat[ii * 3 + dd] + f1_flat[ii * 3 + dd]) / nmodel
        std_val = 0.0
        for kk, fk in enumerate([f0_flat, f1_flat]):
            for dd in range(3):
                tmp = fk[ii * 3 + dd] - avg_f[dd]
                std_val += tmp * tmp
        std_val /= nmodel
        std_f[ii] = np.sqrt(std_val)

    max_std_f = np.max(std_f)
    min_std_f = np.min(std_f)
    avg_std_f = np.mean(std_f)

    print("\n// ---- Deviation stats ----")  # noqa: T201
    print(f"// std_f: max={max_std_f:.18e}  min={min_std_f:.18e}  avg={avg_std_f:.18e}")  # noqa: T201

    # std_v: per-component virial std (virial normalized by natoms)
    v0_flat = v0[0].flatten() / nloc  # (9,)
    v1_flat = v1[0].flatten() / nloc
    std_v = np.zeros(9)
    for ii in range(9):
        avg_v = (v0_flat[ii] + v1_flat[ii]) / nmodel
        std_val = 0.0
        for vk in [v0_flat, v1_flat]:
            tmp = vk[ii] - avg_v
            std_val += tmp * tmp
        std_val /= nmodel
        std_v[ii] = np.sqrt(std_val)

    max_std_v = np.max(std_v)
    min_std_v = np.min(std_v)
    # mystd: sqrt(mean(x^2))
    mystd_v = np.sqrt(np.mean(std_v**2))

    print(f"// std_v: max={max_std_v:.18e}  min={min_std_v:.18e}  mystd={mystd_v:.18e}")  # noqa: T201

    print("\n  std::vector<VALUETYPE> expected_md_f = {")  # noqa: T201
    print(f"      {max_std_f:.18e},")  # noqa: T201
    print(f"      {min_std_f:.18e},")  # noqa: T201
    print(f"      {avg_std_f:.18e}}}; // max min avg")  # noqa: T201
    print("  std::vector<VALUETYPE> expected_md_v = {")  # noqa: T201
    print(f"      {max_std_v:.18e},")  # noqa: T201
    print(f"      {min_std_v:.18e},")  # noqa: T201
    print(f"      {mystd_v:.18e}}}; // max min mystd")  # noqa: T201

    print("\nDone!")  # noqa: T201


if __name__ == "__main__":
    main()
