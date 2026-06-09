#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate chg_spin.pt2 test model and reference values.

Creates a DPA3 model with add_chg_spin_ebd=True and default_chg_spin=[0.0, 1.0],
exports to .pt2, and writes chg_spin.expected with two sections:
  [default]  -- eval with no charge_spin (uses stored default [0.0, 1.0])
  [explicit] -- eval with charge_spin=[0.5, 0.8]
"""

import copy
import os
import sys

import numpy as np

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
            "add_chg_spin_ebd": True,
            "default_chg_spin": [0.0, 1.0],
            "seed": 1,
        },
        "fitting_net": {"neuron": [5, 5, 5], "resnet_dt": True, "seed": 1},
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

    from deepmd.pt_expt.utils.serialization import (
        deserialize_to_file as pt_expt_deserialize_to_file,
    )

    load_custom_ops()

    base_dir = os.path.dirname(__file__)
    pt2_path = os.path.join(base_dir, "chg_spin.pt2")
    print(f"Exporting to {pt2_path} ...")  # noqa: T201
    pt_expt_deserialize_to_file(pt2_path, copy.deepcopy(data), do_atomic_virial=True)
    print("Export done.")  # noqa: T201

    from deepmd.infer import (
        DeepPot,
    )

    dp = DeepPot(pt2_path)

    dim = dp.deep_eval.get_dim_chg_spin()
    assert dim == 2, f"Expected dim_chg_spin == 2, got {dim}"
    print(f"dim_chg_spin = {dim}")  # noqa: T201

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

    # Default charge_spin: no argument → uses stored [0.0, 1.0]
    e_def, f_def, v_def, ae_def, av_def = dp.eval(coord, box, atype, atomic=True)
    print(f"\n// Default charge_spin total energy: {e_def[0, 0]:.18e}")  # noqa: T201

    # Explicit charge_spin = [0.5, 0.8]
    charge_spin_exp = np.array([[0.5, 0.8]], dtype=np.float64)
    e_exp, f_exp, v_exp, ae_exp, av_exp = dp.eval(
        coord, box, atype, atomic=True, charge_spin=charge_spin_exp
    )
    print(f"\n// Explicit charge_spin [0.5, 0.8] total energy: {e_exp[0, 0]:.18e}")  # noqa: T201

    # Sanity: different charge_spin should yield different outputs
    assert not np.allclose(e_def, e_exp), (
        "Default and explicit charge_spin gave identical energy — charge_spin may be ignored"
    )

    ref_path = os.path.join(base_dir, "chg_spin.expected")
    write_expected_ref(
        ref_path,
        sections={
            "default": {
                "expected_e": ae_def[0, :, 0],
                "expected_f": f_def[0],
                "expected_v": av_def[0],
            },
            "explicit": {
                "expected_e": ae_exp[0, :, 0],
                "expected_f": f_exp[0],
                "expected_v": av_exp[0],
            },
        },
        source_script="source/tests/infer/gen_chg_spin.py",
    )
    print(f"Wrote {ref_path}")  # noqa: T201
    print("\nDone!")  # noqa: T201


if __name__ == "__main__":
    main()
