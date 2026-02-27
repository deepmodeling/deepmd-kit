# SPDX-License-Identifier: LGPL-3.0-or-later
"""Consistency test between universal make_stat_input and pt make_stat_input.

The universal make_stat_input (deepmd.utils.model_stat) uses DeepmdDataSystem
(numpy-based). The pt make_stat_input (deepmd.pt.utils.stat) uses DpLoaderSet +
DataLoader (torch-based). This test verifies that both produce equivalent
per-system stat dicts for the keys consumed by compute_or_load_stat.
"""

import os
import unittest

import numpy as np

from deepmd.utils.argcheck import (
    normalize,
)
from deepmd.utils.compat import (
    update_deepmd_input,
)
from deepmd.utils.data import (
    DataRequirementItem,
)

from .common import (
    INSTALLED_PT,
)

TESTS_DIR = os.path.dirname(os.path.dirname(__file__))
EXAMPLE_DIR = os.path.join(TESTS_DIR, "..", "..", "examples", "water")


def _build_config(
    systems: list[str],
    type_map: list[str],
    *,
    data_stat_nbatch: int = 2,
    numb_fparam: int = 0,
    numb_aparam: int = 0,
) -> dict:
    config = {
        "model": {
            "type_map": type_map,
            "descriptor": {
                "type": "se_e2_a",
                "sel": [6, 12],
                "rcut_smth": 0.50,
                "rcut": 3.00,
                "neuron": [8, 16],
                "resnet_dt": False,
                "axis_neuron": 4,
                "type_one_side": True,
                "seed": 1,
            },
            "fitting_net": {
                "neuron": [16, 16],
                "resnet_dt": True,
                "seed": 1,
            },
            "data_stat_nbatch": data_stat_nbatch,
        },
        "learning_rate": {
            "type": "exp",
            "decay_steps": 500,
            "start_lr": 0.001,
            "stop_lr": 3.51e-8,
        },
        "loss": {
            "type": "ener",
            "start_pref_e": 0.02,
            "limit_pref_e": 1,
            "start_pref_f": 1000,
            "limit_pref_f": 1,
            "start_pref_v": 0,
            "limit_pref_v": 0,
        },
        "training": {
            "training_data": {"systems": systems, "batch_size": 1},
            "validation_data": {
                "systems": systems[:1],
                "batch_size": 1,
                "numb_btch": 1,
            },
            "numb_steps": 1,
            "seed": 10,
            "disp_file": "lcurve.out",
            "disp_freq": 99999,
            "save_freq": 99999,
        },
    }
    if numb_fparam > 0:
        config["model"]["fitting_net"]["numb_fparam"] = numb_fparam
    if numb_aparam > 0:
        config["model"]["fitting_net"]["numb_aparam"] = numb_aparam
    config = update_deepmd_input(config, warning=False)
    config = normalize(config)
    return config


def _get_universal_stat(config: dict, data_requirement: list[DataRequirementItem]):
    """Get stat using the universal make_stat_input (DeepmdDataSystem)."""
    from deepmd.utils.data_system import (
        DeepmdDataSystem,
    )
    from deepmd.utils.model_stat import (
        make_stat_input,
    )

    model_params = config["model"]
    training_params = config["training"]
    systems = training_params["training_data"]["systems"]
    nbatch = model_params.get("data_stat_nbatch", 10)

    data = DeepmdDataSystem(
        systems=systems,
        batch_size=training_params["training_data"]["batch_size"],
        test_size=1,
        rcut=model_params["descriptor"]["rcut"],
        type_map=model_params["type_map"],
    )
    for item in data_requirement:
        data.add(
            item.key,
            item.ndof,
            atomic=item.atomic,
            must=item.must,
            high_prec=item.high_prec,
            type_sel=item.type_sel,
            repeat=item.repeat,
            default=item.default,
            dtype=item.dtype,
            output_natoms_for_type_sel=item.output_natoms_for_type_sel,
        )

    return make_stat_input(data, nbatch)


def _get_pt_stat(config: dict, data_requirement: list[DataRequirementItem]):
    """Get stat using the pt make_stat_input (DpLoaderSet + DataLoader)."""
    from deepmd.pt.utils.dataloader import (
        DpLoaderSet,
    )
    from deepmd.pt.utils.stat import (
        make_stat_input,
    )

    model_params = config["model"]
    training_params = config["training"]
    systems = training_params["training_data"]["systems"]
    nbatch = model_params.get("data_stat_nbatch", 10)

    loader = DpLoaderSet(
        systems,
        training_params["training_data"]["batch_size"],
        model_params["type_map"],
        seed=10,
    )
    for item in data_requirement:
        loader.add_data_requirement([item])

    return make_stat_input(loader.systems, loader.dataloaders, nbatch)


def _to_numpy(val):
    """Convert torch.Tensor or np.ndarray to numpy."""
    import torch

    if isinstance(val, torch.Tensor):
        return val.detach().cpu().numpy()
    return val


def _compare_stat(
    test_case: unittest.TestCase,
    universal_stat: list[dict],
    pt_stat: list[dict],
    check_keys: list[str],
) -> None:
    """Compare universal and pt stat outputs for the given keys.

    Verifies structural equivalence: same number of systems, same key
    presence, matching find_* flags, consistent nframes, and consistent
    per-frame sizes.
    """
    test_case.assertEqual(len(universal_stat), len(pt_stat))
    for sys_idx in range(len(universal_stat)):
        for key in check_keys:
            in_uni = key in universal_stat[sys_idx]
            in_pt = key in pt_stat[sys_idx]
            # pt pops fparam/find_fparam when find_fparam==0 but
            # universal keeps them. Skip when find_* is 0.
            if in_uni and not in_pt:
                find_key = f"find_{key}" if not key.startswith("find_") else key
                find_val = universal_stat[sys_idx].get(find_key, None)
                if find_val is not None and not find_val:
                    continue
            test_case.assertEqual(
                in_uni, in_pt, f"system {sys_idx}: key '{key}' presence mismatch"
            )
            if not in_uni:
                continue

            v_uni = _to_numpy(universal_stat[sys_idx][key])
            v_pt = _to_numpy(pt_stat[sys_idx][key])

            if key.startswith("find_"):
                # universal returns bool, pt returns float32
                test_case.assertEqual(
                    bool(v_uni),
                    bool(float(np.ravel(v_pt)[0]) > 0.5),
                    f"system {sys_idx}, key '{key}': find flag mismatch",
                )
                continue

            v_uni = np.asarray(v_uni, dtype=np.float64)
            v_pt = np.asarray(v_pt, dtype=np.float64)

            nf_uni = v_uni.shape[0] if v_uni.ndim >= 2 else 1
            nf_pt = v_pt.shape[0] if v_pt.ndim >= 2 else 1
            test_case.assertEqual(
                nf_uni,
                nf_pt,
                f"system {sys_idx}, key '{key}': nframes mismatch",
            )
            # coord shape differs: universal [nf, natoms*3], pt [nf, natoms, 3].
            # Compare per-frame size.
            test_case.assertEqual(
                v_uni.size // max(nf_uni, 1),
                v_pt.size // max(nf_pt, 1),
                f"system {sys_idx}, key '{key}': per-frame size mismatch "
                f"(uni shape {v_uni.shape}, pt shape {v_pt.shape})",
            )


# --- Standard data requirements for energy model ---
_ENER_DATA_REQ = [
    DataRequirementItem("energy", 1, atomic=False, must=False, high_prec=True),
    DataRequirementItem("force", 3, atomic=True, must=False, high_prec=False),
]

_COMMON_CHECK_KEYS = [
    "atype",
    "box",
    "coord",
    "energy",
    "natoms",
    "find_energy",
    "find_force",
]


@unittest.skipUnless(INSTALLED_PT, "PyTorch backend not installed")
class TestMakeStatInputNormal(unittest.TestCase):
    """Test with normal (non-mixed-type) water data, multiple systems."""

    def test_consistency(self) -> None:
        data_dir = os.path.join(EXAMPLE_DIR, "data")
        if not os.path.isdir(data_dir):
            self.skipTest(f"Example data not found: {data_dir}")

        systems = [
            os.path.join(data_dir, "data_0"),
            os.path.join(data_dir, "data_1"),
        ]
        config = _build_config(systems, ["O", "H"])

        universal_stat = _get_universal_stat(config, _ENER_DATA_REQ)
        pt_stat = _get_pt_stat(config, _ENER_DATA_REQ)

        self.assertEqual(len(universal_stat), 2)
        _compare_stat(self, universal_stat, pt_stat, _COMMON_CHECK_KEYS)


@unittest.skipUnless(INSTALLED_PT, "PyTorch backend not installed")
class TestMakeStatInputMixedType(unittest.TestCase):
    """Test with mixed-type data."""

    def test_consistency(self) -> None:
        data_dir = os.path.join(TESTS_DIR, "tf", "finetune", "data_mixed_type")
        if not os.path.isdir(data_dir):
            self.skipTest(f"Mixed-type data not found: {data_dir}")

        config = _build_config([data_dir], ["O", "H"])

        universal_stat = _get_universal_stat(config, _ENER_DATA_REQ)
        pt_stat = _get_pt_stat(config, _ENER_DATA_REQ)

        _compare_stat(
            self,
            universal_stat,
            pt_stat,
            [*_COMMON_CHECK_KEYS, "real_natoms_vec"],
        )

        # For mixed-type data, real_natoms_vec is the per-frame version.
        # Verify it is present in the universal output.
        for sys_idx in range(len(universal_stat)):
            self.assertIn(
                "real_natoms_vec",
                universal_stat[sys_idx],
                f"system {sys_idx}: real_natoms_vec should be present "
                f"for mixed-type data",
            )


@unittest.skipUnless(INSTALLED_PT, "PyTorch backend not installed")
class TestMakeStatInputFparamAparam(unittest.TestCase):
    """Test with data containing fparam and aparam, multiple systems."""

    def test_consistency(self) -> None:
        data_dir = os.path.join(TESTS_DIR, "pt", "model", "water", "data")
        if not os.path.isdir(data_dir):
            self.skipTest(f"Water fparam data not found: {data_dir}")

        # data_0 has fparam/aparam, data_1 does not — tests find_fparam=0 case
        systems = [
            os.path.join(data_dir, "data_0"),
            os.path.join(data_dir, "data_1"),
        ]
        config = _build_config(systems, ["O", "H"], numb_fparam=2, numb_aparam=1)

        data_requirement = [
            *_ENER_DATA_REQ,
            DataRequirementItem("fparam", 2, atomic=False, must=False, high_prec=False),
            DataRequirementItem("aparam", 1, atomic=True, must=False, high_prec=False),
        ]
        universal_stat = _get_universal_stat(config, data_requirement)
        pt_stat = _get_pt_stat(config, data_requirement)

        self.assertEqual(len(universal_stat), 2)
        _compare_stat(
            self,
            universal_stat,
            pt_stat,
            [*_COMMON_CHECK_KEYS, "fparam", "aparam", "find_fparam", "find_aparam"],
        )


@unittest.skipUnless(INSTALLED_PT, "PyTorch backend not installed")
class TestMakeStatInputSpin(unittest.TestCase):
    """Test with data containing spin, multiple systems."""

    def test_consistency(self) -> None:
        data_dir = os.path.join(TESTS_DIR, "pt", "NiO", "data")
        if not os.path.isdir(data_dir):
            self.skipTest(f"NiO spin data not found: {data_dir}")

        systems = [
            os.path.join(data_dir, "data_0"),
            os.path.join(data_dir, "data_0"),
        ]
        config = _build_config(systems, ["Ni", "O"])

        data_requirement = [
            *_ENER_DATA_REQ,
            DataRequirementItem("spin", 3, atomic=True, must=True, high_prec=False),
        ]
        universal_stat = _get_universal_stat(config, data_requirement)
        pt_stat = _get_pt_stat(config, data_requirement)

        self.assertEqual(len(universal_stat), 2)
        _compare_stat(
            self,
            universal_stat,
            pt_stat,
            [*_COMMON_CHECK_KEYS, "spin", "find_spin"],
        )


if __name__ == "__main__":
    unittest.main()
