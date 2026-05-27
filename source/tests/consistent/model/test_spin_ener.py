# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import unittest
from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.model.model import get_model as get_model_dp
from deepmd.dpmodel.model.spin_model import SpinModel as SpinModelDP
from deepmd.dpmodel.utils.nlist import (
    build_neighbor_list,
    extend_coord_with_ghosts,
)
from deepmd.dpmodel.utils.region import (
    normalize_coord,
)
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)

from ..common import (
    INSTALLED_ARRAY_API_STRICT,
    INSTALLED_PT,
    INSTALLED_PT_EXPT,
    CommonTest,
)
from .common import (
    ModelTest,
    compare_variables_recursive,
)

if INSTALLED_PT:
    from deepmd.pt.model.model import get_model as get_model_pt
    from deepmd.pt.model.model.spin_model import SpinEnergyModel as SpinEnergyModelPT
    from deepmd.pt.utils.utils import to_numpy_array as torch_to_numpy
    from deepmd.pt.utils.utils import to_torch_tensor as numpy_to_torch
else:
    SpinEnergyModelPT = None
if INSTALLED_PT_EXPT:
    from deepmd.pt_expt.common import to_torch_array as pt_expt_numpy_to_torch
    from deepmd.pt_expt.model.spin_ener_model import (
        SpinEnergyModel as SpinEnergyModelPTExpt,
    )
else:
    SpinEnergyModelPTExpt = None
if INSTALLED_ARRAY_API_STRICT:
    import array_api_strict

from deepmd.utils.argcheck import (
    model_args,
)

SPIN_DATA = {
    "type_map": ["O", "H", "B"],
    "descriptor": {
        "type": "se_e2_a",
        "sel": [20, 20, 20],
        "rcut_smth": 0.50,
        "rcut": 4.00,
        "neuron": [
            3,
            6,
        ],
        "resnet_dt": False,
        "axis_neuron": 2,
        "precision": "float64",
        "type_one_side": True,
        "seed": 1,
    },
    "fitting_net": {
        "neuron": [
            5,
            5,
        ],
        "resnet_dt": True,
        "precision": "float64",
        "seed": 1,
        "numb_fparam": 2,
        "numb_aparam": 3,
    },
    "spin": {
        "use_spin": [True, False, False],
        "virtual_scale": [0.3140],
    },
}


class TestSpinEner(CommonTest, ModelTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        return SPIN_DATA

    tf_class = None
    dp_class = SpinModelDP
    pt_class = SpinEnergyModelPT
    pd_class = None
    pt_expt_class = SpinEnergyModelPTExpt
    jax_class = None
    args = model_args()

    skip_tf = True
    skip_jax = True
    skip_pd = True

    @property
    def skip_pt_expt(self):
        return not INSTALLED_PT_EXPT

    def get_reference_backend(self):
        """Get the reference backend.

        We need a reference backend that can reproduce forces.
        """
        if not self.skip_pt:
            return self.RefBackend.PT
        if not self.skip_pt_expt:
            return self.RefBackend.PT_EXPT
        if not self.skip_dp:
            return self.RefBackend.DP
        raise ValueError("No available reference")

    def pass_data_to_cls(self, cls, data) -> Any:
        """Pass data to the class."""
        data = copy.deepcopy(data)
        if cls is SpinModelDP:
            return get_model_dp(data)
        elif cls is SpinEnergyModelPT:
            return get_model_pt(data)
        elif cls is SpinEnergyModelPTExpt:
            dp_model = get_model_dp(data)
            return SpinEnergyModelPTExpt.deserialize(dp_model.serialize())
        return cls(**data, **self.additional_data)

    def setUp(self) -> None:
        CommonTest.setUp(self)

        self.ntypes = 3
        self.coords = np.array(
            [
                12.83,
                2.56,
                2.18,
                12.09,
                2.87,
                2.74,
                00.25,
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
            dtype=GLOBAL_NP_FLOAT_PRECISION,
        ).reshape(1, -1, 3)
        self.atype = np.array([0, 0, 1, 0, 1, 1], dtype=np.int32).reshape(1, -1)
        self.box = np.array(
            [13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0],
            dtype=GLOBAL_NP_FLOAT_PRECISION,
        ).reshape(1, 9)
        self.natoms = np.array([6, 6, 3, 3, 0], dtype=np.int32)
        self.spin = np.array(
            [
                0.50,
                0.30,
                0.20,
                0.40,
                0.25,
                0.15,
                0.10,
                0.05,
                0.08,
                0.12,
                0.07,
                0.09,
                0.45,
                0.35,
                0.28,
                0.11,
                0.06,
                0.03,
            ],
            dtype=GLOBAL_NP_FLOAT_PRECISION,
        ).reshape(1, -1, 3)
        nframes = 1
        nloc = 6
        numb_fparam = 2
        numb_aparam = 3
        rng = np.random.default_rng(42)
        self.fparam = rng.normal(size=(nframes, numb_fparam)).astype(
            GLOBAL_NP_FLOAT_PRECISION
        )
        self.aparam = rng.normal(size=(nframes, nloc, numb_aparam)).astype(
            GLOBAL_NP_FLOAT_PRECISION
        )

    def build_tf(self, obj: Any, suffix: str) -> tuple[list, dict]:
        raise NotImplementedError("no TF in this test")

    def eval_dp(self, dp_obj: Any) -> Any:
        return dp_obj(
            self.coords,
            self.atype,
            self.spin,
            box=self.box,
            fparam=self.fparam,
            aparam=self.aparam,
        )

    def eval_pt(self, pt_obj: Any) -> Any:
        return {
            kk: torch_to_numpy(vv)
            for kk, vv in pt_obj(
                numpy_to_torch(self.coords),
                numpy_to_torch(self.atype),
                numpy_to_torch(self.spin),
                box=numpy_to_torch(self.box),
                fparam=numpy_to_torch(self.fparam),
                aparam=numpy_to_torch(self.aparam),
            ).items()
        }

    def eval_pt_expt(self, pt_expt_obj: Any) -> Any:
        coord_tensor = pt_expt_numpy_to_torch(self.coords)
        coord_tensor.requires_grad_(True)
        return {
            kk: vv.detach().cpu().numpy()
            for kk, vv in pt_expt_obj(
                coord_tensor,
                pt_expt_numpy_to_torch(self.atype),
                pt_expt_numpy_to_torch(self.spin),
                box=pt_expt_numpy_to_torch(self.box),
                fparam=pt_expt_numpy_to_torch(self.fparam),
                aparam=pt_expt_numpy_to_torch(self.aparam),
            ).items()
        }

    def extract_ret(self, ret: Any, backend) -> tuple[np.ndarray, ...]:
        # shape not matched. ravel...
        from ..common import (
            SKIP_FLAG,
        )

        if backend is self.RefBackend.DP:
            return (
                ret["energy"].ravel(),
                ret["atom_energy"].ravel(),
                ret["mask_mag"].ravel(),
                SKIP_FLAG,
                SKIP_FLAG,
                SKIP_FLAG,
            )
        elif backend is self.RefBackend.PT:
            return (
                ret["energy"].ravel(),
                ret["atom_energy"].ravel(),
                ret["mask_mag"].ravel(),
                ret["force"].ravel(),
                ret["force_mag"].ravel(),
                ret["virial"].ravel(),
            )
        elif backend is self.RefBackend.PT_EXPT:
            return (
                ret["energy"].ravel(),
                ret["atom_energy"].ravel(),
                ret["mask_mag"].ravel(),
                ret["force"].ravel(),
                ret["force_mag"].ravel(),
                ret["virial"].ravel(),
            )
        raise ValueError(f"Unknown backend: {backend}")


class TestSpinEnerLower(CommonTest, ModelTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        return SPIN_DATA

    tf_class = None
    dp_class = SpinModelDP
    pt_class = SpinEnergyModelPT
    pd_class = None
    pt_expt_class = SpinEnergyModelPTExpt
    jax_class = None
    array_api_strict_class = SpinModelDP
    args = model_args()

    skip_tf = True
    skip_jax = True
    skip_pd = True
    # The backbone model (make_model) is not yet array_api_strict compatible
    # at the full model call_lower level (indexing issues in descriptor/fitting).
    skip_array_api_strict = True

    @property
    def skip_pt_expt(self):
        return not INSTALLED_PT_EXPT

    def get_reference_backend(self):
        """Get the reference backend.

        We need a reference backend that can reproduce forces.
        """
        if not self.skip_pt:
            return self.RefBackend.PT
        if not self.skip_pt_expt:
            return self.RefBackend.PT_EXPT
        if not self.skip_dp:
            return self.RefBackend.DP
        raise ValueError("No available reference")

    def pass_data_to_cls(self, cls, data) -> Any:
        """Pass data to the class."""
        data = copy.deepcopy(data)
        if cls is SpinModelDP:
            return get_model_dp(data)
        elif cls is SpinEnergyModelPT:
            return get_model_pt(data)
        elif cls is SpinEnergyModelPTExpt:
            dp_model = get_model_dp(data)
            return SpinEnergyModelPTExpt.deserialize(dp_model.serialize())
        return cls(**data, **self.additional_data)

    def setUp(self) -> None:
        CommonTest.setUp(self)

        self.ntypes = 3
        coords = np.array(
            [
                12.83,
                2.56,
                2.18,
                12.09,
                2.87,
                2.74,
                00.25,
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
            dtype=GLOBAL_NP_FLOAT_PRECISION,
        ).reshape(1, -1, 3)
        atype = np.array([0, 0, 1, 0, 1, 1], dtype=np.int32).reshape(1, -1)
        box = np.array(
            [13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0],
            dtype=GLOBAL_NP_FLOAT_PRECISION,
        ).reshape(1, 9)
        self.spin = np.array(
            [
                0.50,
                0.30,
                0.20,
                0.40,
                0.25,
                0.15,
                0.10,
                0.05,
                0.08,
                0.12,
                0.07,
                0.09,
                0.45,
                0.35,
                0.28,
                0.11,
                0.06,
                0.03,
            ],
            dtype=GLOBAL_NP_FLOAT_PRECISION,
        ).reshape(1, -1, 3)

        rcut = 4.0
        nframes, nloc = atype.shape[:2]
        coord_normalized = normalize_coord(
            coords.reshape(nframes, nloc, 3),
            box.reshape(nframes, 3, 3),
        )
        extended_coord, extended_atype, mapping = extend_coord_with_ghosts(
            coord_normalized, atype, box, rcut
        )
        nlist = build_neighbor_list(
            extended_coord,
            extended_atype,
            nloc,
            rcut,
            [20, 20, 20],
            distinguish_types=True,
        )
        extended_coord = extended_coord.reshape(nframes, -1, 3)
        self.nlist = nlist
        self.extended_coord = extended_coord
        self.extended_atype = extended_atype
        self.mapping = mapping

        # Build extended spin from mapping
        self.extended_spin = np.take_along_axis(
            self.spin,
            np.repeat(mapping[:, :, np.newaxis], 3, axis=2),
            axis=1,
        )

        nframes = 1
        nloc = 6
        numb_fparam = 2
        numb_aparam = 3
        rng = np.random.default_rng(42)
        self.fparam = rng.normal(size=(nframes, numb_fparam)).astype(
            GLOBAL_NP_FLOAT_PRECISION
        )
        self.aparam = rng.normal(size=(nframes, nloc, numb_aparam)).astype(
            GLOBAL_NP_FLOAT_PRECISION
        )

    def build_tf(self, obj: Any, suffix: str) -> tuple[list, dict]:
        raise NotImplementedError("no TF in this test")

    def eval_dp(self, dp_obj: Any) -> Any:
        return dp_obj.call_lower(
            self.extended_coord,
            self.extended_atype,
            self.extended_spin,
            self.nlist,
            self.mapping,
            fparam=self.fparam,
            aparam=self.aparam,
        )

    def eval_pt(self, pt_obj: Any) -> Any:
        return {
            kk: torch_to_numpy(vv)
            for kk, vv in pt_obj.forward_lower(
                numpy_to_torch(self.extended_coord),
                numpy_to_torch(self.extended_atype),
                numpy_to_torch(self.extended_spin),
                numpy_to_torch(self.nlist),
                numpy_to_torch(self.mapping),
                fparam=numpy_to_torch(self.fparam),
                aparam=numpy_to_torch(self.aparam),
            ).items()
        }

    def eval_pt_expt(self, pt_expt_obj: Any) -> Any:
        extended_coord_tensor = pt_expt_numpy_to_torch(self.extended_coord)
        extended_coord_tensor.requires_grad_(True)
        return {
            kk: vv.detach().cpu().numpy()
            for kk, vv in pt_expt_obj.forward_lower(
                extended_coord_tensor,
                pt_expt_numpy_to_torch(self.extended_atype),
                pt_expt_numpy_to_torch(self.extended_spin),
                pt_expt_numpy_to_torch(self.nlist),
                pt_expt_numpy_to_torch(self.mapping),
                fparam=pt_expt_numpy_to_torch(self.fparam),
                aparam=pt_expt_numpy_to_torch(self.aparam),
            ).items()
        }

    def eval_array_api_strict(self, array_api_strict_obj: Any) -> Any:
        return {
            kk: to_numpy_array(vv) if hasattr(vv, "__array_namespace__") else vv
            for kk, vv in array_api_strict_obj.call_lower(
                array_api_strict.asarray(self.extended_coord),
                array_api_strict.asarray(self.extended_atype),
                array_api_strict.asarray(self.extended_spin),
                array_api_strict.asarray(self.nlist),
                array_api_strict.asarray(self.mapping),
                fparam=array_api_strict.asarray(self.fparam),
                aparam=array_api_strict.asarray(self.aparam),
            ).items()
        }

    def extract_ret(self, ret: Any, backend) -> tuple[np.ndarray, ...]:
        # shape not matched. ravel...
        from ..common import (
            SKIP_FLAG,
        )

        if backend is self.RefBackend.DP:
            return (
                ret["energy"].ravel(),
                ret["atom_energy"].ravel(),
                ret["extended_mask_mag"].ravel(),
                SKIP_FLAG,
                SKIP_FLAG,
                SKIP_FLAG,
            )
        elif backend is self.RefBackend.PT:
            return (
                ret["energy"].ravel(),
                ret["atom_energy"].ravel(),
                ret["extended_mask_mag"].ravel(),
                ret["extended_force"].ravel(),
                ret["extended_force_mag"].ravel(),
                ret["virial"].ravel(),
            )
        elif backend is self.RefBackend.PT_EXPT:
            return (
                ret["energy"].ravel(),
                ret["atom_energy"].ravel(),
                ret["extended_mask_mag"].ravel(),
                ret["extended_force"].ravel(),
                ret["extended_force_mag"].ravel(),
                ret["virial"].ravel(),
            )
        elif backend is self.RefBackend.ARRAY_API_STRICT:
            return (
                ret["energy"].ravel(),
                ret["atom_energy"].ravel(),
                ret["extended_mask_mag"].ravel(),
                SKIP_FLAG,
                SKIP_FLAG,
                SKIP_FLAG,
            )
        raise ValueError(f"Unknown backend: {backend}")


@unittest.skipUnless(INSTALLED_PT and INSTALLED_PT_EXPT, "PT and PT_EXPT are required")
class TestSpinEnerComputeOrLoadStat(unittest.TestCase):
    """Test that compute_or_load_stat produces identical statistics on dp, pt, and pt_expt
    for spin models.
    """

    def setUp(self) -> None:
        data = model_args().normalize_value(
            copy.deepcopy(SPIN_DATA),
            trim_pattern="_*",
        )

        self._model_data = data

        # Build dp model, then deserialize into pt and pt_expt to share weights
        self.dp_model = get_model_dp(data)
        serialized = self.dp_model.serialize()
        self.pt_model = SpinEnergyModelPT.deserialize(serialized)
        self.pt_expt_model = SpinEnergyModelPTExpt.deserialize(serialized)

        # Test coords / atype / box for forward evaluation
        self.coords = np.array(
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
            dtype=GLOBAL_NP_FLOAT_PRECISION,
        ).reshape(1, -1, 3)
        self.atype = np.array([0, 0, 1, 0, 1, 1], dtype=np.int32).reshape(1, -1)
        self.box = np.array(
            [13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0],
            dtype=GLOBAL_NP_FLOAT_PRECISION,
        ).reshape(1, 9)
        self.spin = np.array(
            [
                0.50,
                0.30,
                0.20,
                0.40,
                0.25,
                0.15,
                0.10,
                0.05,
                0.08,
                0.12,
                0.07,
                0.09,
                0.45,
                0.35,
                0.28,
                0.11,
                0.06,
                0.03,
            ],
            dtype=GLOBAL_NP_FLOAT_PRECISION,
        ).reshape(1, -1, 3)

        # Mock training data for compute_or_load_stat
        natoms = 6
        nframes = 3
        numb_fparam = 2
        numb_aparam = 3
        rng = np.random.default_rng(42)
        coords_stat = rng.normal(size=(nframes, natoms, 3)).astype(
            GLOBAL_NP_FLOAT_PRECISION
        )
        atype_stat = np.array([[0, 0, 1, 0, 1, 1]] * nframes, dtype=np.int32)
        box_stat = np.tile(
            np.eye(3, dtype=GLOBAL_NP_FLOAT_PRECISION).reshape(1, 3, 3) * 13.0,
            (nframes, 1, 1),
        )
        # natoms: [total, real, type0_count, type1_count, type2_count]
        natoms_stat = np.array([[natoms, natoms, 3, 3, 0]] * nframes, dtype=np.int32)
        energy_stat = rng.normal(size=(nframes, 1)).astype(GLOBAL_NP_FLOAT_PRECISION)
        spin_stat = rng.normal(size=(nframes, natoms, 3)).astype(
            GLOBAL_NP_FLOAT_PRECISION
        )
        fparam_stat = rng.normal(size=(nframes, numb_fparam)).astype(
            GLOBAL_NP_FLOAT_PRECISION
        )
        aparam_stat = rng.normal(size=(nframes, natoms, numb_aparam)).astype(
            GLOBAL_NP_FLOAT_PRECISION
        )

        # dp sample (numpy)
        np_sample = {
            "coord": coords_stat,
            "atype": atype_stat,
            "atype_ext": atype_stat,
            "box": box_stat,
            "natoms": natoms_stat,
            "energy": energy_stat,
            "find_energy": np.float32(1.0),
            "spin": spin_stat,
            "fparam": fparam_stat,
            "find_fparam": np.float32(1.0),
            "aparam": aparam_stat,
            "find_aparam": np.float32(1.0),
        }
        # pt / pt_expt sample (torch tensors)
        pt_sample = {
            "coord": numpy_to_torch(coords_stat),
            "atype": numpy_to_torch(atype_stat),
            "atype_ext": numpy_to_torch(atype_stat),
            "box": numpy_to_torch(box_stat),
            "natoms": numpy_to_torch(natoms_stat),
            "energy": numpy_to_torch(energy_stat),
            "find_energy": np.float32(1.0),
            "spin": numpy_to_torch(spin_stat),
            "fparam": numpy_to_torch(fparam_stat),
            "find_fparam": np.float32(1.0),
            "aparam": numpy_to_torch(aparam_stat),
            "find_aparam": np.float32(1.0),
        }

        self.np_sampled = [np_sample]
        self.pt_sampled = [pt_sample]

        # fparam/aparam for eval
        nframes_eval = 1
        nloc_eval = 6
        self.fparam = rng.normal(size=(nframes_eval, numb_fparam)).astype(
            GLOBAL_NP_FLOAT_PRECISION
        )
        self.aparam = rng.normal(size=(nframes_eval, nloc_eval, numb_aparam)).astype(
            GLOBAL_NP_FLOAT_PRECISION
        )

    def _eval_dp(self) -> dict:
        return self.dp_model(
            self.coords,
            self.atype,
            self.spin,
            box=self.box,
            fparam=self.fparam,
            aparam=self.aparam,
        )

    def _eval_pt(self) -> dict:
        return {
            kk: torch_to_numpy(vv)
            for kk, vv in self.pt_model(
                numpy_to_torch(self.coords),
                numpy_to_torch(self.atype),
                numpy_to_torch(self.spin),
                box=numpy_to_torch(self.box),
                fparam=numpy_to_torch(self.fparam),
                aparam=numpy_to_torch(self.aparam),
            ).items()
        }

    def _eval_pt_expt(self) -> dict:
        coord_t = pt_expt_numpy_to_torch(self.coords)
        coord_t.requires_grad_(True)
        return {
            k: v.detach().cpu().numpy()
            for k, v in self.pt_expt_model(
                coord_t,
                pt_expt_numpy_to_torch(self.atype),
                pt_expt_numpy_to_torch(self.spin),
                box=pt_expt_numpy_to_torch(self.box),
                fparam=pt_expt_numpy_to_torch(self.fparam),
                aparam=pt_expt_numpy_to_torch(self.aparam),
            ).items()
        }

    def test_compute_stat(self) -> None:
        # 1. Pre-stat forward consistency
        dp_ret0 = self._eval_dp()
        pt_ret0 = self._eval_pt()
        pe_ret0 = self._eval_pt_expt()
        for key in ("energy", "atom_energy"):
            np.testing.assert_allclose(
                dp_ret0[key],
                pt_ret0[key],
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"Pre-stat dp vs pt mismatch in {key}",
            )
            np.testing.assert_allclose(
                dp_ret0[key],
                pe_ret0[key],
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"Pre-stat dp vs pt_expt mismatch in {key}",
            )

        # 2. Run compute_or_load_stat on all three backends
        from copy import (
            deepcopy,
        )

        self.dp_model.compute_or_load_stat(lambda: deepcopy(self.np_sampled))
        self.pt_model.compute_or_load_stat(lambda: deepcopy(self.pt_sampled))
        self.pt_expt_model.compute_or_load_stat(lambda: deepcopy(self.pt_sampled))

        # 3. Serialize all three and compare @variables
        dp_ser = self.dp_model.serialize()
        pt_ser = self.pt_model.serialize()
        pe_ser = self.pt_expt_model.serialize()
        compare_variables_recursive(dp_ser, pt_ser)
        compare_variables_recursive(dp_ser, pe_ser)

        # 4. Post-stat forward consistency
        dp_ret1 = self._eval_dp()
        pt_ret1 = self._eval_pt()
        pe_ret1 = self._eval_pt_expt()
        for key in ("energy", "atom_energy"):
            np.testing.assert_allclose(
                dp_ret1[key],
                pt_ret1[key],
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"Post-stat dp vs pt mismatch in {key}",
            )
            np.testing.assert_allclose(
                dp_ret1[key],
                pe_ret1[key],
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"Post-stat dp vs pt_expt mismatch in {key}",
            )

    def test_load_stat_from_file(self) -> None:
        import tempfile
        from pathlib import (
            Path,
        )

        import h5py

        from deepmd.utils.path import (
            DPPath,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create separate stat files for each backend
            dp_h5 = str((Path(tmpdir) / "dp_stat.h5").resolve())
            pt_h5 = str((Path(tmpdir) / "pt_stat.h5").resolve())
            pe_h5 = str((Path(tmpdir) / "pe_stat.h5").resolve())
            for p in (dp_h5, pt_h5, pe_h5):
                with h5py.File(p, "w"):
                    pass

            # 1. Compute stats and save to file
            self.dp_model.compute_or_load_stat(
                lambda: self.np_sampled, stat_file_path=DPPath(dp_h5, "a")
            )
            self.pt_model.compute_or_load_stat(
                lambda: self.pt_sampled, stat_file_path=DPPath(pt_h5, "a")
            )
            self.pt_expt_model.compute_or_load_stat(
                lambda: self.pt_sampled, stat_file_path=DPPath(pe_h5, "a")
            )

            # Save the computed serializations as reference
            dp_ser_computed = self.dp_model.serialize()
            pt_ser_computed = self.pt_model.serialize()
            pe_ser_computed = self.pt_expt_model.serialize()

            # 2. Build fresh models from the same initial weights
            dp_model2 = get_model_dp(self._model_data)
            pt_model2 = SpinEnergyModelPT.deserialize(dp_model2.serialize())
            pe_model2 = SpinEnergyModelPTExpt.deserialize(dp_model2.serialize())

            # 3. Load stats from file (should NOT call the sampled func)
            def raise_error():
                raise RuntimeError("Should load from file, not recompute")

            dp_model2.compute_or_load_stat(
                raise_error, stat_file_path=DPPath(dp_h5, "a")
            )
            pt_model2.compute_or_load_stat(
                raise_error, stat_file_path=DPPath(pt_h5, "a")
            )
            pe_model2.compute_or_load_stat(
                raise_error, stat_file_path=DPPath(pe_h5, "a")
            )

            # 4. Loaded models should match the computed ones
            dp_ser_loaded = dp_model2.serialize()
            pt_ser_loaded = pt_model2.serialize()
            pe_ser_loaded = pe_model2.serialize()
            compare_variables_recursive(dp_ser_computed, dp_ser_loaded)
            compare_variables_recursive(pt_ser_computed, pt_ser_loaded)
            compare_variables_recursive(pe_ser_computed, pe_ser_loaded)

            # 5. Cross-backend consistency after loading
            compare_variables_recursive(dp_ser_loaded, pt_ser_loaded)
            compare_variables_recursive(dp_ser_loaded, pe_ser_loaded)


@unittest.skipUnless(INSTALLED_PT and INSTALLED_PT_EXPT, "PT and PT_EXPT are required")
class TestSpinEnerModelAPIs(unittest.TestCase):
    """Test consistency of model-level APIs between dp, pt, and pt_expt for spin models.

    Both models are constructed from the same serialized weights
    (dpmodel -> serialize -> pt/pt_expt deserialize) so that numerical outputs
    can be compared directly.
    """

    def setUp(self) -> None:
        data = model_args().normalize_value(
            copy.deepcopy(SPIN_DATA),
            trim_pattern="_*",
        )
        self.dp_model = get_model_dp(data)
        serialized = self.dp_model.serialize()
        self.pt_model = SpinEnergyModelPT.deserialize(serialized)
        self.pt_expt_model = SpinEnergyModelPTExpt.deserialize(serialized)

    def test_translated_output_def(self) -> None:
        """translated_output_def should return the same keys on dp, pt, and pt_expt."""
        dp_def = self.dp_model.translated_output_def()
        pt_def = self.pt_model.translated_output_def()
        pt_expt_def = self.pt_expt_model.translated_output_def()
        self.assertEqual(set(dp_def.keys()), set(pt_def.keys()))
        self.assertEqual(set(dp_def.keys()), set(pt_expt_def.keys()))
        for key in dp_def:
            self.assertEqual(dp_def[key].shape, pt_def[key].shape)
            self.assertEqual(dp_def[key].shape, pt_expt_def[key].shape)

    def test_model_output_def(self) -> None:
        """model_output_def should return the same keys and shapes on dp and pt."""
        dp_def = self.dp_model.model_output_def().get_data()
        pt_def = self.pt_model.model_output_def().get_data()
        self.assertEqual(set(dp_def.keys()), set(pt_def.keys()))
        for key in dp_def:
            self.assertEqual(dp_def[key].shape, pt_def[key].shape)

    def test_model_output_type(self) -> None:
        """model_output_type should return the same list on dp and pt."""
        self.assertEqual(
            self.dp_model.model_output_type(),
            self.pt_model.model_output_type(),
        )

    def test_do_grad_r(self) -> None:
        """do_grad_r should return the same value on dp and pt."""
        self.assertEqual(
            self.dp_model.do_grad_r("energy"),
            self.pt_model.do_grad_r("energy"),
        )
        self.assertTrue(self.dp_model.do_grad_r("energy"))

    def test_do_grad_c(self) -> None:
        """do_grad_c should return the same value on dp and pt."""
        self.assertEqual(
            self.dp_model.do_grad_c("energy"),
            self.pt_model.do_grad_c("energy"),
        )
        self.assertTrue(self.dp_model.do_grad_c("energy"))

    def test_get_rcut(self) -> None:
        """get_rcut should return the same value on dp, pt, and pt_expt."""
        self.assertEqual(self.dp_model.get_rcut(), self.pt_model.get_rcut())
        self.assertEqual(self.dp_model.get_rcut(), self.pt_expt_model.get_rcut())
        self.assertAlmostEqual(self.dp_model.get_rcut(), 4.0)

    def test_get_type_map(self) -> None:
        """get_type_map should return the same list on dp, pt, and pt_expt."""
        self.assertEqual(self.dp_model.get_type_map(), self.pt_model.get_type_map())
        self.assertEqual(
            self.dp_model.get_type_map(), self.pt_expt_model.get_type_map()
        )
        self.assertEqual(self.dp_model.get_type_map(), ["O", "H", "B"])

    def test_get_ntypes(self) -> None:
        """get_ntypes should return the same value on dp, pt, and pt_expt."""
        self.assertEqual(self.dp_model.get_ntypes(), self.pt_model.get_ntypes())
        self.assertEqual(self.dp_model.get_ntypes(), self.pt_expt_model.get_ntypes())
        self.assertEqual(self.dp_model.get_ntypes(), 3)

    def test_get_sel_type(self) -> None:
        """get_sel_type should return the same list on dp and pt."""
        self.assertEqual(self.dp_model.get_sel_type(), self.pt_model.get_sel_type())

    def test_get_dim_fparam(self) -> None:
        """get_dim_fparam should return the same value on dp and pt."""
        self.assertEqual(self.dp_model.get_dim_fparam(), self.pt_model.get_dim_fparam())
        self.assertEqual(self.dp_model.get_dim_fparam(), 2)

    def test_get_dim_aparam(self) -> None:
        """get_dim_aparam should return the same value on dp and pt."""
        self.assertEqual(self.dp_model.get_dim_aparam(), self.pt_model.get_dim_aparam())
        self.assertEqual(self.dp_model.get_dim_aparam(), 3)

    def test_get_nnei(self) -> None:
        """get_nnei should return the same value on dp and pt."""
        self.assertEqual(self.dp_model.get_nnei(), self.pt_model.get_nnei())

    def test_get_nsel(self) -> None:
        """get_nsel should return the same value on dp and pt."""
        self.assertEqual(self.dp_model.get_nsel(), self.pt_model.get_nsel())

    def test_is_aparam_nall(self) -> None:
        """is_aparam_nall should return the same value on dp and pt."""
        self.assertEqual(self.dp_model.is_aparam_nall(), self.pt_model.is_aparam_nall())

    def test_has_spin(self) -> None:
        """has_spin should return True on all backends."""
        self.assertTrue(self.dp_model.has_spin())
        self.assertTrue(self.pt_model.has_spin())
        self.assertTrue(self.pt_expt_model.has_spin())

    def test_get_model_def_script(self) -> None:
        """get_model_def_script should return the same value on dp, pt, and pt_expt."""
        dp_val = self.dp_model.get_model_def_script()
        pt_val = self.pt_model.get_model_def_script()
        pe_val = self.pt_expt_model.get_model_def_script()
        self.assertEqual(dp_val, pt_val)
        self.assertEqual(dp_val, pe_val)

    def test_get_min_nbor_dist(self) -> None:
        """get_min_nbor_dist should return the same value on dp, pt, and pt_expt."""
        dp_val = self.dp_model.get_min_nbor_dist()
        pt_val = self.pt_model.get_min_nbor_dist()
        pe_val = self.pt_expt_model.get_min_nbor_dist()
        self.assertEqual(dp_val, pt_val)
        self.assertEqual(dp_val, pe_val)
