# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for spin model inference via the DeepPot high-level API.

Verifies that .pt2 and .pte spin models produce correct results when loaded
through the pt_expt inference backend (DeepEval → DeepPot).
"""

import copy
import os
import tempfile

import numpy as np
import pytest
import torch

from deepmd.dpmodel.model.model import get_model as get_model_dp
from deepmd.pt_expt.model.spin_ener_model import (
    SpinEnergyModel,
)
from deepmd.pt_expt.utils import (
    env,
)
from deepmd.pt_expt.utils.serialization import (
    deserialize_to_file,
)

SPIN_CONFIG = {
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

COORD = np.array(
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
SPIN = np.array(
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
ATYPE = [0, 1, 1, 0, 1, 1]
BOX = np.array([13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0], dtype=np.float64)


def _build_reference():
    """Build pt_expt model and run eager reference inference.

    Returns data dict, and reference dicts for PBC and NoPBC.
    """
    dp_model = get_model_dp(copy.deepcopy(SPIN_CONFIG))
    model_dict = dp_model.serialize()
    data = {
        "model": model_dict,
        "model_def_script": SPIN_CONFIG,
        "backend": "dpmodel",
        "software": "deepmd-kit",
        "version": "3.0.0",
    }

    # Build pt_expt model for eager reference
    pt_model = SpinEnergyModel.deserialize(dp_model.serialize()).to(env.DEVICE)
    pt_model.eval()

    natoms = len(ATYPE)
    coord_t = torch.tensor(
        COORD.reshape(1, natoms, 3), dtype=torch.float64, device=env.DEVICE
    )
    coord_t.requires_grad_(True)
    atype_t = torch.tensor([ATYPE], dtype=torch.int64, device=env.DEVICE)
    spin_t = torch.tensor(
        SPIN.reshape(1, natoms, 3), dtype=torch.float64, device=env.DEVICE
    )
    box_t = torch.tensor(BOX.reshape(1, 9), dtype=torch.float64, device=env.DEVICE)

    # PBC reference
    ref_pbc = pt_model(coord_t, atype_t, spin_t, box_t)
    ref_pbc = {k: v.detach().cpu().numpy() for k, v in ref_pbc.items()}

    # NoPBC reference
    ref_nopbc = pt_model(coord_t, atype_t, spin_t, None)
    ref_nopbc = {k: v.detach().cpu().numpy() for k, v in ref_nopbc.items()}

    return data, ref_pbc, ref_nopbc


@pytest.fixture(scope="module")
def spin_model_files():
    """Create .pt2 and .pte spin model files and compute reference values."""
    data, ref_pbc, ref_nopbc = _build_reference()
    files = {}
    tmpdir = tempfile.mkdtemp()
    for ext in (".pt2", ".pte"):
        path = os.path.join(tmpdir, f"spin_test{ext}")
        # AOTInductor (.pt2) internally creates tensors using the PyTorch
        # default device.  Clear it so compilation stays on CPU.
        prev = torch.get_default_device()
        torch.set_default_device(None)
        try:
            deserialize_to_file(path, copy.deepcopy(data))
        finally:
            torch.set_default_device(prev)
        files[ext] = path
    yield files, ref_pbc, ref_nopbc
    for path in files.values():
        if os.path.exists(path):
            os.unlink(path)
    os.rmdir(tmpdir)


@pytest.mark.parametrize("ext", [".pt2", ".pte"])  # model format
class TestSpinInference:
    """Test spin model inference through DeepPot high-level API."""

    def test_get_has_spin(self, spin_model_files, ext) -> None:
        """Test that get_has_spin returns True for spin models."""
        from deepmd.infer import (
            DeepPot,
        )

        files, _, _ = spin_model_files
        dp = DeepPot(files[ext])
        assert dp.has_spin

    def test_get_use_spin(self, spin_model_files, ext) -> None:
        """Test that use_spin returns per-type spin usage."""
        from deepmd.infer import (
            DeepPot,
        )

        files, _, _ = spin_model_files
        dp = DeepPot(files[ext])
        use_spin = dp.use_spin
        assert use_spin == [True, False]

    def test_get_ntypes_spin(self, spin_model_files, ext) -> None:
        """Test that get_ntypes_spin returns 0 (new spin implementation)."""
        from deepmd.infer import (
            DeepPot,
        )

        files, _, _ = spin_model_files
        dp = DeepPot(files[ext])
        assert dp.get_ntypes_spin() == 0

    def test_eval_spin_model_requires_spin(self, spin_model_files, ext) -> None:
        """Spin model must raise ValueError when spin is not provided."""
        from deepmd.infer import (
            DeepPot,
        )

        files, _, _ = spin_model_files
        dp = DeepPot(files[ext])
        with pytest.raises(ValueError, match="no `spin` argument was provided"):
            dp.eval(COORD, BOX, ATYPE)

    def test_eval_pbc_atomic(self, spin_model_files, ext) -> None:
        """Test PBC evaluation with atomic=True."""
        from deepmd.infer import (
            DeepPot,
        )

        files, ref, _ = spin_model_files
        dp = DeepPot(files[ext])
        natoms = len(ATYPE)

        e, f, v, ae, av, fm, mm = dp.eval(COORD, BOX, ATYPE, atomic=True, spin=SPIN)

        np.testing.assert_allclose(
            e.reshape(-1), ref["energy"].reshape(-1), rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            ae.reshape(-1),
            ref["atom_energy"].reshape(-1),
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            f.reshape(-1), ref["force"].reshape(-1), rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            fm.reshape(-1),
            ref["force_mag"].reshape(-1),
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            v.reshape(-1), ref["virial"].reshape(-1), rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(mm.reshape(-1), ref["mask_mag"].reshape(-1))
        # Shape checks
        assert e.shape == (1, 1)
        assert f.shape == (1, natoms, 3)
        assert v.shape == (1, 9)
        assert ae.shape == (1, natoms, 1)
        assert av.shape == (1, natoms, 9)
        assert fm.shape == (1, natoms, 3)
        assert mm.shape == (1, natoms, 1)

    def test_eval_pbc_nonatomic(self, spin_model_files, ext) -> None:
        """Test PBC evaluation with atomic=False."""
        from deepmd.infer import (
            DeepPot,
        )

        files, ref, _ = spin_model_files
        dp = DeepPot(files[ext])

        e, f, v, fm, mm = dp.eval(COORD, BOX, ATYPE, atomic=False, spin=SPIN)

        np.testing.assert_allclose(
            e.reshape(-1), ref["energy"].reshape(-1), rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            f.reshape(-1), ref["force"].reshape(-1), rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            fm.reshape(-1),
            ref["force_mag"].reshape(-1),
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            v.reshape(-1), ref["virial"].reshape(-1), rtol=1e-10, atol=1e-10
        )

    def test_eval_nopbc_atomic(self, spin_model_files, ext) -> None:
        """Test NoPBC evaluation with atomic=True."""
        from deepmd.infer import (
            DeepPot,
        )

        files, _, ref = spin_model_files
        dp = DeepPot(files[ext])

        e, f, v, ae, av, fm, mm = dp.eval(COORD, None, ATYPE, atomic=True, spin=SPIN)

        np.testing.assert_allclose(
            e.reshape(-1), ref["energy"].reshape(-1), rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            ae.reshape(-1),
            ref["atom_energy"].reshape(-1),
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            f.reshape(-1), ref["force"].reshape(-1), rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            fm.reshape(-1),
            ref["force_mag"].reshape(-1),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_eval_nopbc_nonatomic(self, spin_model_files, ext) -> None:
        """Test NoPBC evaluation with atomic=False."""
        from deepmd.infer import (
            DeepPot,
        )

        files, _, ref = spin_model_files
        dp = DeepPot(files[ext])

        e, f, v, fm, mm = dp.eval(COORD, None, ATYPE, atomic=False, spin=SPIN)

        np.testing.assert_allclose(
            e.reshape(-1), ref["energy"].reshape(-1), rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            f.reshape(-1), ref["force"].reshape(-1), rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            fm.reshape(-1),
            ref["force_mag"].reshape(-1),
            rtol=1e-10,
            atol=1e-10,
        )


SPIN_FPARAM_CONFIG = copy.deepcopy(SPIN_CONFIG)
SPIN_FPARAM_CONFIG["fitting_net"]["numb_fparam"] = 1
SPIN_FPARAM_CONFIG["fitting_net"]["default_fparam"] = [0.5]


@pytest.fixture(scope="module")
def spin_fparam_model_files():
    """Create .pt2 and .pte spin model files with default fparam."""
    dp_model = get_model_dp(copy.deepcopy(SPIN_FPARAM_CONFIG))
    model_dict = dp_model.serialize()
    data = {
        "model": model_dict,
        "model_def_script": SPIN_FPARAM_CONFIG,
        "backend": "dpmodel",
        "software": "deepmd-kit",
        "version": "3.0.0",
    }
    files = {}
    tmpdir = tempfile.mkdtemp()
    for ext in (".pt2", ".pte"):
        path = os.path.join(tmpdir, f"spin_fparam_test{ext}")
        prev = torch.get_default_device()
        torch.set_default_device(None)
        try:
            deserialize_to_file(path, copy.deepcopy(data))
        finally:
            torch.set_default_device(prev)
        files[ext] = path
    yield files
    for path in files.values():
        if os.path.exists(path):
            os.unlink(path)
    os.rmdir(tmpdir)


@pytest.mark.parametrize("ext", [".pt2", ".pte"])  # model format
class TestSpinDefaultFparam:
    """Test spin model with default_fparam via DeepPot API."""

    def test_eval_without_fparam_matches_explicit(
        self, spin_fparam_model_files, ext
    ) -> None:
        """Eval without fparam should use default and match explicit fparam."""
        from deepmd.infer import (
            DeepPot,
        )

        files = spin_fparam_model_files
        dp = DeepPot(files[ext])

        # Eval WITHOUT fparam — should use default_fparam=[0.5]
        e_no, f_no, v_no, fm_no, mm_no = dp.eval(
            COORD, BOX, ATYPE, atomic=False, spin=SPIN
        )
        # Eval WITH explicit fparam=[0.5]
        e_ex, f_ex, v_ex, fm_ex, mm_ex = dp.eval(
            COORD, BOX, ATYPE, atomic=False, spin=SPIN, fparam=[0.5]
        )

        np.testing.assert_allclose(e_no, e_ex, atol=1e-10)
        np.testing.assert_allclose(f_no, f_ex, atol=1e-10)
        np.testing.assert_allclose(v_no, v_ex, atol=1e-10)
        np.testing.assert_allclose(fm_no, fm_ex, atol=1e-10)


SPIN_APARAM_CONFIG = copy.deepcopy(SPIN_CONFIG)
SPIN_APARAM_CONFIG["fitting_net"]["numb_aparam"] = 2


@pytest.fixture(scope="module")
def spin_aparam_model_files():
    """Create .pt2 and .pte spin model files with aparam."""
    dp_model = get_model_dp(copy.deepcopy(SPIN_APARAM_CONFIG))
    model_dict = dp_model.serialize()
    data = {
        "model": model_dict,
        "model_def_script": SPIN_APARAM_CONFIG,
        "backend": "dpmodel",
        "software": "deepmd-kit",
        "version": "3.0.0",
    }
    files = {}
    tmpdir = tempfile.mkdtemp()
    for ext in (".pt2", ".pte"):
        path = os.path.join(tmpdir, f"spin_aparam_test{ext}")
        prev = torch.get_default_device()
        torch.set_default_device(None)
        try:
            deserialize_to_file(path, copy.deepcopy(data))
        finally:
            torch.set_default_device(prev)
        files[ext] = path
    yield files
    for path in files.values():
        if os.path.exists(path):
            os.unlink(path)
    os.rmdir(tmpdir)


@pytest.mark.parametrize("ext", [".pt2", ".pte"])  # model format
class TestSpinAparam:
    """Test spin model with aparam via DeepPot API (.pt2/.pte)."""

    def test_aparam_takes_effect(self, spin_aparam_model_files, ext) -> None:
        """Verify that different aparam values produce different outputs."""
        from deepmd.infer import (
            DeepPot,
        )

        files = spin_aparam_model_files
        dp = DeepPot(files[ext])
        natoms = len(ATYPE)

        aparam_zero = np.zeros(natoms * 2, dtype=np.float64)
        aparam_nonzero = np.full(natoms * 2, 0.5, dtype=np.float64)

        e0, f0, v0, fm0, mm0 = dp.eval(
            COORD, BOX, ATYPE, atomic=False, spin=SPIN, aparam=aparam_zero
        )
        e1, f1, v1, fm1, mm1 = dp.eval(
            COORD, BOX, ATYPE, atomic=False, spin=SPIN, aparam=aparam_nonzero
        )

        # Different aparam must produce different energy
        assert not np.allclose(e0, e1), (
            "Changing aparam did not change output — aparam may be ignored"
        )

    def test_eval_without_aparam_raises(self, spin_aparam_model_files, ext) -> None:
        """Model with dim_aparam > 0 must raise when aparam not provided."""
        from deepmd.infer import (
            DeepPot,
        )

        files = spin_aparam_model_files
        dp = DeepPot(files[ext])

        with pytest.raises(ValueError, match="aparam is required"):
            dp.eval(COORD, BOX, ATYPE, atomic=False, spin=SPIN)
