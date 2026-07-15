#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate charge-spin test models and reference values.

Creates a DPA3 model with add_chg_spin_ebd=True and default_chg_spin=[0.0, 1.0],
exports it to .pt2 (AOTInductor), .pth (TorchScript), and .savedmodel (JAX)
from the same weights, and writes chg_spin.expected with two sections:
  [default]  -- eval with no charge_spin (uses stored default [0.0, 1.0])
  [explicit] -- eval with charge_spin=[1.0, 2.0]
The .pth is verified to match the .pt2 reference for both sections.
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

    from deepmd.jax.utils.serialization import (
        deserialize_to_file as jax_deserialize_to_file,
    )
    from deepmd.pt.utils.serialization import (
        deserialize_to_file as pt_deserialize_to_file,
    )
    from deepmd.pt_expt.utils.serialization import (
        deserialize_to_file as pt_expt_deserialize_to_file,
    )

    load_custom_ops()

    base_dir = os.path.dirname(__file__)
    pt2_path = os.path.join(base_dir, "chg_spin.pt2")
    print(f"Exporting to {pt2_path} ...")  # noqa: T201
    pt_expt_deserialize_to_file(pt2_path, copy.deepcopy(data), do_atomic_virial=True)

    savedmodel_path = os.path.join(base_dir, "chg_spin.savedmodel")
    print(f"Exporting to {savedmodel_path} ...")  # noqa: T201
    jax_deserialize_to_file(savedmodel_path, copy.deepcopy(data))

    pth_path = os.path.join(base_dir, "chg_spin.pth")
    # Remove any stale .pth first so a failed export below cannot leave an old
    # artifact that the parity check would then validate against.
    if os.path.exists(pth_path):
        os.remove(pth_path)
    pth_generated = False
    print(f"Exporting to {pth_path} ...")  # noqa: T201
    try:
        pt_deserialize_to_file(pth_path, copy.deepcopy(data))
        pth_generated = True
    except RuntimeError as e:
        # Custom ops may not be available in all build environments; .pth
        # generation is not critical (the .pth test skips if the file is
        # missing / PyTorch support is off).
        print(f"WARNING: .pth export failed ({e}), skipping.")  # noqa: T201
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

    # Explicit charge_spin = [1.0, 2.0] (charge idx 101, spin idx 2 — both
    # differ from the default [0.0, 1.0] which maps to charge idx 100, spin 1)
    charge_spin_exp = np.array([[1.0, 2.0]], dtype=np.float64)
    e_exp, f_exp, v_exp, ae_exp, av_exp = dp.eval(
        coord, box, atype, atomic=True, charge_spin=charge_spin_exp
    )
    print(f"\n// Explicit charge_spin [1.0, 2.0] total energy: {e_exp[0, 0]:.18e}")  # noqa: T201

    # Sanity: different charge_spin should yield different outputs
    assert not np.allclose(e_def, e_exp), (
        "Default and explicit charge_spin gave identical energy — charge_spin may be ignored"
    )

    # NoPbc variants (box=None) — used by the .pth nlist test, which mirrors the
    # established DPA3 .pth pattern (NoPbc lmp_nlist, nghost=0) and so needs its
    # own reference values.
    e_np_def, f_np_def, v_np_def, ae_np_def, av_np_def = dp.eval(
        coord, None, atype, atomic=True
    )
    e_np_exp, f_np_exp, v_np_exp, ae_np_exp, av_np_exp = dp.eval(
        coord, None, atype, atomic=True, charge_spin=charge_spin_exp
    )
    assert not np.allclose(e_np_def, e_np_exp), (
        "NoPbc: default and explicit charge_spin gave identical energy"
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
            "nopbc_default": {
                "expected_e": ae_np_def[0, :, 0],
                "expected_f": f_np_def[0],
                "expected_v": av_np_def[0],
            },
            "nopbc_explicit": {
                "expected_e": ae_np_exp[0, :, 0],
                "expected_f": f_np_exp[0],
                "expected_v": av_np_exp[0],
            },
        },
        source_script="source/tests/infer/gen_chg_spin.py",
    )
    print(f"Wrote {ref_path}")  # noqa: T201

    # ---- Verify .pth reproduces the .pt2 reference (PBC + NoPbc) ----
    if pth_generated:
        dp_pth = DeepPot(pth_path)
        # Note: the .pth (pt) DeepEval does not expose get_dim_chg_spin (only
        # the .pt2 / pt_expt one does); the energy/force parity below is the
        # real check that charge_spin is threaded for the .pth backend.
        tol = 1e-10
        e_def_p, f_def_p, v_def_p = dp_pth.eval(coord, box, atype)
        e_exp_p, f_exp_p, v_exp_p = dp_pth.eval(
            coord, box, atype, charge_spin=charge_spin_exp
        )
        np.testing.assert_allclose(e_def_p, e_def, atol=tol, err_msg="pth default e")
        np.testing.assert_allclose(f_def_p, f_def, atol=tol, err_msg="pth default f")
        np.testing.assert_allclose(e_exp_p, e_exp, atol=tol, err_msg="pth explicit e")
        np.testing.assert_allclose(f_exp_p, f_exp, atol=tol, err_msg="pth explicit f")
        # NoPbc parity (the .pth nlist test uses NoPbc)
        e_np_def_p, f_np_def_p, _ = dp_pth.eval(coord, None, atype)
        e_np_exp_p, f_np_exp_p, _ = dp_pth.eval(
            coord, None, atype, charge_spin=charge_spin_exp
        )
        np.testing.assert_allclose(
            e_np_def_p, e_np_def, atol=tol, err_msg="pth nopbc default e"
        )
        np.testing.assert_allclose(
            e_np_exp_p, e_np_exp, atol=tol, err_msg="pth nopbc explicit e"
        )
        # different charge_spin must change the .pth output too
        assert not np.allclose(e_def_p, e_exp_p), (
            ".pth: default and explicit charge_spin gave identical energy"
        )
        print("// .pth matches .pt2 reference (PBC + NoPbc).")  # noqa: T201
    else:
        print("\n// Skipping .pth verification (file not generated).")  # noqa: T201

    print("\nDone!")  # noqa: T201


if __name__ == "__main__":
    main()
