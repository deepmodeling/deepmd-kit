# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import os
import unittest
from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.model.dp_zbl_model import DPZBLModel as DPZBLModelDP
from deepmd.dpmodel.model.model import get_model as get_model_dp
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
    INSTALLED_JAX,
    INSTALLED_PT,
    INSTALLED_PT_EXPT,
    SKIP_FLAG,
    CommonTest,
    parameterized,
)
from .common import (
    ModelTest,
    compare_variables_recursive,
)

if INSTALLED_PT:
    from deepmd.pt.model.model import get_model as get_model_pt
    from deepmd.pt.model.model.dp_zbl_model import DPZBLModel as DPZBLModelPT
    from deepmd.pt.utils.utils import to_numpy_array as torch_to_numpy
    from deepmd.pt.utils.utils import to_torch_tensor as numpy_to_torch
else:
    DPZBLModelPT = None
if INSTALLED_JAX:
    from deepmd.jax.model.dp_zbl_model import DPZBLModel as DPZBLModelJAX
    from deepmd.jax.model.model import get_model as get_model_jax
else:
    DPZBLModelJAX = None
if INSTALLED_PT_EXPT:
    from deepmd.pt_expt.common import to_torch_array as pt_expt_numpy_to_torch
    from deepmd.pt_expt.model import DPZBLModel as DPZBLModelPTExpt
else:
    DPZBLModelPTExpt = None
from deepmd.utils.argcheck import (
    model_args,
)

TESTS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


@parameterized(
    (
        [],
        [[0, 1]],
    ),
    (
        [],
        [1],
    ),
)
class TestEner(CommonTest, ModelTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        pair_exclude_types, atom_exclude_types = self.param
        return {
            "type_map": ["O", "H", "B"],
            "use_srtab": f"{TESTS_DIR}/pt/water/data/zbl_tab_potential/H2O_tab_potential.txt",
            "smin_alpha": 0.1,
            "sw_rmin": 0.2,
            "sw_rmax": 4.0,
            "pair_exclude_types": pair_exclude_types,
            "atom_exclude_types": atom_exclude_types,
            "descriptor": {
                "type": "se_atten",
                "sel": 40,
                "rcut_smth": 0.5,
                "rcut": 4.0,
                "neuron": [3, 6],
                "axis_neuron": 2,
                "attn": 8,
                "attn_layer": 2,
                "attn_dotr": True,
                "attn_mask": False,
                "activation_function": "tanh",
                "scaling_factor": 1.0,
                "normalize": False,
                "temperature": 1.0,
                "set_davg_zero": True,
                "type_one_side": True,
                "seed": 1,
            },
            "fitting_net": {
                "neuron": [5, 5],
                "resnet_dt": True,
                "seed": 1,
            },
        }

    dp_class = DPZBLModelDP
    pt_class = DPZBLModelPT
    pt_expt_class = DPZBLModelPTExpt
    jax_class = DPZBLModelJAX
    args = model_args()

    def get_reference_backend(self):
        """Get the reference backend.

        We need a reference backend that can reproduce forces.
        """
        if not self.skip_pt:
            return self.RefBackend.PT
        if not self.skip_tf:
            return self.RefBackend.TF
        if not self.skip_pt_expt and self.pt_expt_class is not None:
            return self.RefBackend.PT_EXPT
        if not self.skip_jax:
            return self.RefBackend.JAX
        if not self.skip_dp:
            return self.RefBackend.DP
        raise ValueError("No available reference")

    @property
    def skip_tf(self) -> bool:
        return True

    @property
    def skip_jax(self) -> bool:
        return not INSTALLED_JAX

    def pass_data_to_cls(self, cls, data) -> Any:
        """Pass data to the class."""
        data = data.copy()
        if cls is DPZBLModelDP:
            return get_model_dp(data)
        elif cls is DPZBLModelPT:
            return get_model_pt(data)
        elif cls is DPZBLModelPTExpt:
            dp_model = get_model_dp(data)
            return DPZBLModelPTExpt.deserialize(dp_model.serialize())
        elif cls is DPZBLModelJAX:
            return get_model_jax(data)
        return cls(**data, **self.additional_data)

    def setUp(self) -> None:
        CommonTest.setUp(self)

        self.ntypes = 2
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
        self.atype = np.array([0, 1, 1, 0, 1, 1], dtype=np.int32).reshape(1, -1)
        self.box = np.array(
            [13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0],
            dtype=GLOBAL_NP_FLOAT_PRECISION,
        ).reshape(1, 9)
        self.natoms = np.array([6, 6, 2, 4], dtype=np.int32)

        # TF requires the atype to be sort
        idx_map = np.argsort(self.atype.ravel())
        self.atype = self.atype[:, idx_map]
        self.coords = self.coords[:, idx_map]

    def build_tf(self, obj: Any, suffix: str) -> tuple[list, dict]:
        return self.build_tf_model(
            obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
            suffix,
        )

    def eval_dp(self, dp_obj: Any) -> Any:
        return self.eval_dp_model(
            dp_obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
        )

    def eval_pt(self, pt_obj: Any) -> Any:
        return self.eval_pt_model(
            pt_obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
        )

    def eval_pt_expt(self, pt_expt_obj: Any) -> Any:
        return self.eval_pt_expt_model(
            pt_expt_obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
        )

    def eval_jax(self, jax_obj: Any) -> Any:
        return self.eval_jax_model(
            jax_obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
        )

    def extract_ret(self, ret: Any, backend) -> tuple[np.ndarray, ...]:
        # shape not matched. ravel...
        if backend is self.RefBackend.TF:
            return (ret[0].ravel(), ret[1].ravel(), ret[2].ravel(), ret[3].ravel())
        elif backend is self.RefBackend.DP:
            return (
                ret["energy"].ravel(),
                ret["atom_energy"].ravel(),
                SKIP_FLAG,
                SKIP_FLAG,
            )
        elif backend in {
            self.RefBackend.PT,
            self.RefBackend.PT_EXPT,
            self.RefBackend.JAX,
        }:
            return (
                ret["energy"].ravel(),
                ret["atom_energy"].ravel(),
                ret["force"].ravel(),
                ret["virial"].ravel(),
            )
        raise ValueError(f"Unknown backend: {backend}")


@unittest.skipUnless(INSTALLED_PT and INSTALLED_PT_EXPT, "PyTorch is not installed")
class TestZBLEnerModelAPIs(unittest.TestCase):
    """Test consistency of model-level APIs between pt and dpmodel backends.

    Both models are constructed from the same serialized weights
    (dpmodel -> serialize -> pt deserialize) so that numerical outputs
    can be compared directly.

    DPZBLModel is a linear combination model (DP + ZBL) and does NOT
    support get_descriptor() or get_fitting_net() at the top level.
    """

    def setUp(self) -> None:
        data = model_args().normalize_value(
            {
                "type_map": ["O", "H", "B"],
                "use_srtab": f"{TESTS_DIR}/pt/water/data/zbl_tab_potential/H2O_tab_potential.txt",
                "smin_alpha": 0.1,
                "sw_rmin": 0.2,
                "sw_rmax": 4.0,
                "descriptor": {
                    "type": "se_atten",
                    "sel": 40,
                    "rcut_smth": 0.5,
                    "rcut": 4.0,
                    "neuron": [3, 6],
                    "axis_neuron": 2,
                    "attn": 8,
                    "attn_layer": 2,
                    "attn_dotr": True,
                    "attn_mask": False,
                    "activation_function": "tanh",
                    "scaling_factor": 1.0,
                    "normalize": False,
                    "temperature": 1.0,
                    "set_davg_zero": True,
                    "type_one_side": True,
                    "seed": 1,
                },
                "fitting_net": {
                    "neuron": [5, 5],
                    "resnet_dt": True,
                    "precision": "float64",
                    "seed": 1,
                },
            },
            trim_pattern="_*",
        )
        # Build dpmodel first, then deserialize into pt/pt_expt to share weights
        self.dp_model = get_model_dp(data)
        serialized = self.dp_model.serialize()
        self.pt_model = DPZBLModelPT.deserialize(serialized)
        self.pt_expt_model = DPZBLModelPTExpt.deserialize(serialized)

        # Coords / atype / box
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
        self.atype = np.array([0, 1, 1, 0, 1, 1], dtype=np.int32).reshape(1, -1)
        self.box = np.array(
            [13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0],
            dtype=GLOBAL_NP_FLOAT_PRECISION,
        ).reshape(1, 9)

        # Build extended coords + nlist for lower-level calls
        rcut = self.dp_model.get_rcut()
        sel = self.dp_model.get_sel()
        nframes, nloc = self.atype.shape[:2]
        coord_normalized = normalize_coord(
            self.coords.reshape(nframes, nloc, 3),
            self.box.reshape(nframes, 3, 3),
        )
        extended_coord, extended_atype, mapping = extend_coord_with_ghosts(
            coord_normalized, self.atype, self.box, rcut
        )
        nlist = build_neighbor_list(
            extended_coord,
            extended_atype,
            nloc,
            rcut,
            sel,
            distinguish_types=False,
        )
        self.extended_coord = extended_coord.reshape(nframes, -1, 3)
        self.extended_atype = extended_atype
        self.mapping = mapping
        self.nlist = nlist

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

    def test_get_out_bias(self) -> None:
        """get_out_bias should return numerically equal values on dp and pt."""
        dp_bias = to_numpy_array(self.dp_model.get_out_bias())
        pt_bias = torch_to_numpy(self.pt_model.get_out_bias())
        np.testing.assert_allclose(dp_bias, pt_bias, rtol=1e-10, atol=1e-10)
        # Verify shape: ntypes=3 for ZBL model
        self.assertEqual(dp_bias.shape[1], 3)
        self.assertGreater(dp_bias.shape[0], 0)

    def test_set_out_bias(self) -> None:
        """set_out_bias should update the bias on both backends."""
        dp_bias = to_numpy_array(self.dp_model.get_out_bias())
        new_bias = dp_bias + 1.0
        # dp
        self.dp_model.set_out_bias(new_bias)
        np.testing.assert_allclose(
            to_numpy_array(self.dp_model.get_out_bias()),
            new_bias,
            rtol=1e-10,
            atol=1e-10,
        )
        # pt
        self.pt_model.set_out_bias(numpy_to_torch(new_bias))
        np.testing.assert_allclose(
            torch_to_numpy(self.pt_model.get_out_bias()),
            new_bias,
            rtol=1e-10,
            atol=1e-10,
        )

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
        """get_rcut should return the same value on dp and pt."""
        self.assertEqual(self.dp_model.get_rcut(), self.pt_model.get_rcut())

    def test_get_type_map(self) -> None:
        """get_type_map should return the same list on dp and pt."""
        self.assertEqual(self.dp_model.get_type_map(), self.pt_model.get_type_map())
        self.assertEqual(self.dp_model.get_type_map(), ["O", "H", "B"])

    def test_get_sel(self) -> None:
        """get_sel should return the same list on dp and pt."""
        self.assertEqual(self.dp_model.get_sel(), self.pt_model.get_sel())

    def test_get_nsel(self) -> None:
        """get_nsel should return the same value on dp and pt."""
        self.assertEqual(self.dp_model.get_nsel(), self.pt_model.get_nsel())

    def test_get_nnei(self) -> None:
        """get_nnei should return the same value on dp and pt."""
        self.assertEqual(self.dp_model.get_nnei(), self.pt_model.get_nnei())

    def test_mixed_types(self) -> None:
        """mixed_types should return the same value on dp and pt."""
        self.assertEqual(self.dp_model.mixed_types(), self.pt_model.mixed_types())
        # DPZBLModel (LinearEnergyAtomicModel) always uses mixed types
        self.assertTrue(self.dp_model.mixed_types())

    def test_has_message_passing(self) -> None:
        """has_message_passing should return the same value on dp and pt."""
        self.assertEqual(
            self.dp_model.has_message_passing(),
            self.pt_model.has_message_passing(),
        )
        self.assertFalse(self.dp_model.has_message_passing())

    def test_need_sorted_nlist_for_lower(self) -> None:
        """need_sorted_nlist_for_lower should return the same value on dp and pt."""
        self.assertEqual(
            self.dp_model.need_sorted_nlist_for_lower(),
            self.pt_model.need_sorted_nlist_for_lower(),
        )

    def test_get_dim_fparam(self) -> None:
        """get_dim_fparam should return the same value on dp and pt."""
        self.assertEqual(self.dp_model.get_dim_fparam(), self.pt_model.get_dim_fparam())
        self.assertEqual(self.dp_model.get_dim_fparam(), 0)

    def test_get_dim_aparam(self) -> None:
        """get_dim_aparam should return the same value on dp and pt."""
        self.assertEqual(self.dp_model.get_dim_aparam(), self.pt_model.get_dim_aparam())
        self.assertEqual(self.dp_model.get_dim_aparam(), 0)

    def test_get_sel_type(self) -> None:
        """get_sel_type should return the same list on dp and pt."""
        self.assertEqual(self.dp_model.get_sel_type(), self.pt_model.get_sel_type())

    def test_is_aparam_nall(self) -> None:
        """is_aparam_nall should return the same value on dp and pt."""
        self.assertEqual(self.dp_model.is_aparam_nall(), self.pt_model.is_aparam_nall())
        self.assertFalse(self.dp_model.is_aparam_nall())

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

    def test_set_case_embd(self) -> None:
        """set_case_embd should produce consistent results across backends.

        Also verifies that different case indices produce different outputs,
        confirming the embedding is actually used.
        """
        from deepmd.utils.argcheck import (
            model_args,
        )

        # Build a model with dim_case_embd > 0
        data = model_args().normalize_value(
            {
                "type_map": ["O", "H", "B"],
                "use_srtab": f"{TESTS_DIR}/pt/water/data/zbl_tab_potential/H2O_tab_potential.txt",
                "smin_alpha": 0.1,
                "sw_rmin": 0.2,
                "sw_rmax": 4.0,
                "descriptor": {
                    "type": "se_atten",
                    "sel": 40,
                    "rcut_smth": 0.5,
                    "rcut": 4.0,
                    "neuron": [3, 6],
                    "axis_neuron": 2,
                    "attn": 8,
                    "attn_layer": 2,
                    "attn_dotr": True,
                    "attn_mask": False,
                    "activation_function": "tanh",
                    "scaling_factor": 1.0,
                    "normalize": False,
                    "temperature": 1.0,
                    "set_davg_zero": True,
                    "type_one_side": True,
                    "seed": 1,
                },
                "fitting_net": {
                    "neuron": [5, 5],
                    "resnet_dt": True,
                    "precision": "float64",
                    "seed": 1,
                    "dim_case_embd": 3,
                },
            },
            trim_pattern="_*",
        )
        dp_model = get_model_dp(data)
        serialized = dp_model.serialize()
        pt_model = DPZBLModelPT.deserialize(serialized)
        pe_model = DPZBLModelPTExpt.deserialize(serialized)

        def _eval(case_idx):
            dp_model.set_case_embd(case_idx)
            pt_model.set_case_embd(case_idx)
            pe_model.set_case_embd(case_idx)
            dp_ret = dp_model(self.coords, self.atype, box=self.box)
            pt_ret = {
                k: torch_to_numpy(v)
                for k, v in pt_model(
                    numpy_to_torch(self.coords),
                    numpy_to_torch(self.atype),
                    box=numpy_to_torch(self.box),
                ).items()
            }
            coord_t = pt_expt_numpy_to_torch(self.coords)
            coord_t.requires_grad_(True)
            pe_ret = {
                k: v.detach().cpu().numpy()
                for k, v in pe_model(
                    coord_t,
                    pt_expt_numpy_to_torch(self.atype),
                    box=pt_expt_numpy_to_torch(self.box),
                ).items()
            }
            return dp_ret, pt_ret, pe_ret

        dp0, pt0, pe0 = _eval(0)
        dp1, pt1, pe1 = _eval(1)

        # Cross-backend consistency for each case index
        for key in ("energy", "atom_energy"):
            np.testing.assert_allclose(
                dp0[key],
                pt0[key],
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"case 0: dp vs pt mismatch in {key}",
            )
            np.testing.assert_allclose(
                dp0[key],
                pe0[key],
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"case 0: dp vs pt_expt mismatch in {key}",
            )
            np.testing.assert_allclose(
                dp1[key],
                pt1[key],
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"case 1: dp vs pt mismatch in {key}",
            )
            np.testing.assert_allclose(
                dp1[key],
                pe1[key],
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"case 1: dp vs pt_expt mismatch in {key}",
            )
        # Different case indices should produce different outputs
        self.assertFalse(
            np.allclose(dp0["energy"], dp1["energy"]),
            "set_case_embd(0) and set_case_embd(1) produced the same energy",
        )

    def test_atomic_output_def(self) -> None:
        """atomic_output_def should return the same keys and shapes on dp and pt."""
        dp_def = self.dp_model.atomic_output_def()
        pt_def = self.pt_model.atomic_output_def()
        self.assertEqual(set(dp_def.keys()), set(pt_def.keys()))
        for key in dp_def.keys():
            self.assertEqual(dp_def[key].shape, pt_def[key].shape)

    def test_get_ntypes(self) -> None:
        """get_ntypes should return the same value on dp and pt."""
        self.assertEqual(self.dp_model.get_ntypes(), self.pt_model.get_ntypes())
        self.assertEqual(self.dp_model.get_ntypes(), 3)

    def test_format_nlist(self) -> None:
        """format_nlist should produce the same result on dp and pt."""
        dp_nlist = self.dp_model.format_nlist(
            self.extended_coord,
            self.extended_atype,
            self.nlist,
        )
        pt_nlist = torch_to_numpy(
            self.pt_model.format_nlist(
                numpy_to_torch(self.extended_coord),
                numpy_to_torch(self.extended_atype),
                numpy_to_torch(self.nlist),
            )
        )
        np.testing.assert_equal(dp_nlist, pt_nlist)

    def test_forward_common_atomic(self) -> None:
        """forward_common_atomic should produce consistent results on dp and pt.

        Compares at the atomic_model level, where both backends define this method.
        DPZBLModel has no aparam, so we don't pass aparam here.
        """
        dp_ret = self.dp_model.atomic_model.forward_common_atomic(
            self.extended_coord,
            self.extended_atype,
            self.nlist,
            mapping=self.mapping,
        )
        pt_ret = self.pt_model.atomic_model.forward_common_atomic(
            numpy_to_torch(self.extended_coord),
            numpy_to_torch(self.extended_atype),
            numpy_to_torch(self.nlist),
            mapping=numpy_to_torch(self.mapping),
        )
        # Compare the common keys
        common_keys = set(dp_ret.keys()) & set(pt_ret.keys())
        self.assertTrue(len(common_keys) > 0)
        for key in common_keys:
            if dp_ret[key] is not None and pt_ret[key] is not None:
                np.testing.assert_allclose(
                    dp_ret[key],
                    torch_to_numpy(pt_ret[key]),
                    rtol=1e-10,
                    atol=1e-10,
                    err_msg=f"Mismatch in forward_common_atomic key '{key}'",
                )

    def test_change_out_bias(self) -> None:
        """change_out_bias should produce consistent bias on dp, pt, and pt_expt.

        Tests both set-by-statistic and change-by-statistic modes.
        DPZBLModel has no fparam/aparam, so fitting stats checks are skipped.
        """
        nframes = 2

        # Use realistic coords (from setUp, tiled for 2 frames)
        coords_2f = np.tile(self.coords, (nframes, 1, 1))  # (2, 6, 3)
        atype_2f = np.array([[0, 0, 1, 1, 1, 1], [0, 1, 1, 0, 1, 1]], dtype=np.int32)
        box_2f = np.tile(self.box.reshape(1, 3, 3), (nframes, 1, 1))
        # natoms: [nloc, nloc, n_type0, n_type1, n_type2] — 3 types
        natoms_data = np.array([[6, 6, 2, 4, 0], [6, 6, 2, 4, 0]], dtype=np.int32)
        energy_data = np.array([10.0, 20.0]).reshape(nframes, 1)

        # dpmodel stat data (numpy)
        dp_merged = [
            {
                "coord": coords_2f,
                "atype": atype_2f,
                "atype_ext": atype_2f,
                "box": box_2f,
                "natoms": natoms_data,
                "energy": energy_data,
                "find_energy": np.float32(1.0),
            }
        ]
        # pt stat data (torch tensors)
        pt_merged = [
            {
                "coord": numpy_to_torch(coords_2f),
                "atype": numpy_to_torch(atype_2f),
                "atype_ext": numpy_to_torch(atype_2f),
                "box": numpy_to_torch(box_2f),
                "natoms": numpy_to_torch(natoms_data),
                "energy": numpy_to_torch(energy_data),
                "find_energy": np.float32(1.0),
            }
        ]
        # pt_expt stat data (numpy, same as dp)
        pe_merged = dp_merged

        # Save initial (zero) bias
        dp_bias_init = to_numpy_array(self.dp_model.get_out_bias()).copy()

        # --- Test "set-by-statistic" mode ---
        self.dp_model.change_out_bias(dp_merged, bias_adjust_mode="set-by-statistic")
        self.pt_model.change_out_bias(pt_merged, bias_adjust_mode="set-by-statistic")
        self.pt_expt_model.change_out_bias(
            pe_merged, bias_adjust_mode="set-by-statistic"
        )

        # Verify out bias consistency
        dp_bias = to_numpy_array(self.dp_model.get_out_bias())
        pt_bias = torch_to_numpy(self.pt_model.get_out_bias())
        pe_bias = to_numpy_array(self.pt_expt_model.get_out_bias())
        np.testing.assert_allclose(dp_bias, pt_bias, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(dp_bias, pe_bias, rtol=1e-10, atol=1e-10)
        self.assertFalse(
            np.allclose(dp_bias, dp_bias_init),
            "set-by-statistic did not change the bias from initial values",
        )

        # --- Test "change-by-statistic" mode ---
        dp_bias_before = dp_bias.copy()
        self.dp_model.change_out_bias(dp_merged, bias_adjust_mode="change-by-statistic")
        self.pt_model.change_out_bias(pt_merged, bias_adjust_mode="change-by-statistic")
        self.pt_expt_model.change_out_bias(
            pe_merged, bias_adjust_mode="change-by-statistic"
        )

        # Verify out bias consistency
        dp_bias2 = to_numpy_array(self.dp_model.get_out_bias())
        pt_bias2 = torch_to_numpy(self.pt_model.get_out_bias())
        pe_bias2 = to_numpy_array(self.pt_expt_model.get_out_bias())
        np.testing.assert_allclose(dp_bias2, pt_bias2, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(dp_bias2, pe_bias2, rtol=1e-10, atol=1e-10)
        self.assertFalse(
            np.allclose(dp_bias2, dp_bias_before),
            "change-by-statistic did not further change the bias",
        )

    # test_change_type_map: NOT applicable — PairTabAtomicModel does not
    # support changing type map (would require rebuilding the tab file),
    # so LinearEnergyAtomicModel.change_type_map always fails for DPZBLModel
    # when the new type_map differs from the original.

    # test_change_type_map_extend_stat: NOT applicable — same reason.

    def test_update_sel(self) -> None:
        """update_sel should return the same result on dp and pt."""
        from unittest.mock import (
            patch,
        )

        from deepmd.dpmodel.model.dp_model import DPModelCommon as DPModelCommonDP
        from deepmd.pt.model.model.dp_model import DPModelCommon as DPModelCommonPT

        mock_min_nbor_dist = 0.5
        mock_sel = [30]
        local_jdata = {
            "type_map": ["O", "H", "B"],
            "use_srtab": f"{TESTS_DIR}/pt/water/data/zbl_tab_potential/H2O_tab_potential.txt",
            "smin_alpha": 0.1,
            "sw_rmin": 0.2,
            "sw_rmax": 4.0,
            "descriptor": {
                "type": "se_atten",
                "sel": "auto",
                "rcut_smth": 0.5,
                "rcut": 4.0,
            },
            "fitting_net": {
                "neuron": [5, 5],
            },
        }
        type_map = ["O", "H", "B"]

        with patch(
            "deepmd.dpmodel.utils.update_sel.UpdateSel.get_nbor_stat",
            return_value=(mock_min_nbor_dist, mock_sel),
        ):
            dp_result, dp_min_dist = DPModelCommonDP.update_sel(
                None, type_map, local_jdata
            )

        with patch(
            "deepmd.pt.utils.update_sel.UpdateSel.get_nbor_stat",
            return_value=(mock_min_nbor_dist, mock_sel),
        ):
            pt_result, pt_min_dist = DPModelCommonPT.update_sel(
                None, type_map, local_jdata
            )

        self.assertEqual(dp_result, pt_result)
        self.assertEqual(dp_min_dist, pt_min_dist)
        # Verify sel was actually updated (not still "auto")
        self.assertNotEqual(dp_result["descriptor"]["sel"], "auto")

    def test_compute_or_load_out_stat(self) -> None:
        """compute_or_load_out_stat should produce consistent bias on dp and pt.

        Tests both the compute path (from data) and the load path (from file).
        """
        import tempfile
        from pathlib import (
            Path,
        )

        import h5py

        from deepmd.utils.path import (
            DPPath,
        )

        nframes = 2
        coords_2f = np.tile(self.coords, (nframes, 1, 1))
        atype_2f = np.array([[0, 0, 1, 1, 1, 1], [0, 1, 1, 0, 1, 1]], dtype=np.int32)
        box_2f = np.tile(self.box.reshape(1, 3, 3), (nframes, 1, 1))
        # natoms: [nloc, nloc, n_type0, n_type1, n_type2] — 3 types
        natoms_data = np.array([[6, 6, 2, 4, 0], [6, 6, 2, 4, 0]], dtype=np.int32)
        energy_data = np.array([10.0, 20.0]).reshape(nframes, 1)

        dp_merged = [
            {
                "coord": coords_2f,
                "atype": atype_2f,
                "atype_ext": atype_2f,
                "box": box_2f,
                "natoms": natoms_data,
                "energy": energy_data,
                "find_energy": np.float32(1.0),
            }
        ]
        pt_merged = [
            {
                "coord": numpy_to_torch(coords_2f),
                "atype": numpy_to_torch(atype_2f),
                "atype_ext": numpy_to_torch(atype_2f),
                "box": numpy_to_torch(box_2f),
                "natoms": numpy_to_torch(natoms_data),
                "energy": numpy_to_torch(energy_data),
                "find_energy": np.float32(1.0),
            }
        ]

        # Verify bias is initially identical
        dp_bias_before = to_numpy_array(self.dp_model.get_out_bias()).copy()
        pt_bias_before = torch_to_numpy(self.pt_model.get_out_bias()).copy()
        np.testing.assert_allclose(
            dp_bias_before, pt_bias_before, rtol=1e-10, atol=1e-10
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create separate h5 files for dp and pt
            dp_h5 = str((Path(tmpdir) / "dp_stat.h5").resolve())
            pt_h5 = str((Path(tmpdir) / "pt_stat.h5").resolve())
            with h5py.File(dp_h5, "w"):
                pass
            with h5py.File(pt_h5, "w"):
                pass
            dp_stat_path = DPPath(dp_h5, "a")
            pt_stat_path = DPPath(pt_h5, "a")

            # 1. Compute stats and save to file
            self.dp_model.atomic_model.compute_or_load_out_stat(
                dp_merged, stat_file_path=dp_stat_path
            )
            self.pt_model.atomic_model.compute_or_load_out_stat(
                pt_merged, stat_file_path=pt_stat_path
            )

            dp_bias_after = to_numpy_array(self.dp_model.get_out_bias())
            pt_bias_after = torch_to_numpy(self.pt_model.get_out_bias())
            np.testing.assert_allclose(
                dp_bias_after, pt_bias_after, rtol=1e-10, atol=1e-10
            )

            # Verify bias actually changed (not still all zeros)
            self.assertFalse(
                np.allclose(dp_bias_after, dp_bias_before),
                "compute_or_load_out_stat did not change the bias",
            )

            # 2. Verify both backends saved the same file content
            with h5py.File(dp_h5, "r") as dp_f, h5py.File(pt_h5, "r") as pt_f:
                dp_keys = sorted(dp_f.keys())
                pt_keys = sorted(pt_f.keys())
                self.assertEqual(dp_keys, pt_keys)
                for key in dp_keys:
                    np.testing.assert_allclose(
                        np.array(dp_f[key]),
                        np.array(pt_f[key]),
                        rtol=1e-10,
                        atol=1e-10,
                        err_msg=f"Stat file content mismatch for key {key}",
                    )

            # 3. Reset biases to zero, then load from file
            zero_bias = np.zeros_like(dp_bias_after)
            self.dp_model.set_out_bias(zero_bias)
            self.pt_model.set_out_bias(numpy_to_torch(zero_bias))

            # Use a callable that raises to ensure it loads from file, not recomputes
            def raise_error():
                raise RuntimeError("Should not recompute — should load from file")

            self.dp_model.atomic_model.compute_or_load_out_stat(
                raise_error, stat_file_path=dp_stat_path
            )
            self.pt_model.atomic_model.compute_or_load_out_stat(
                raise_error, stat_file_path=pt_stat_path
            )

            dp_bias_loaded = to_numpy_array(self.dp_model.get_out_bias())
            pt_bias_loaded = torch_to_numpy(self.pt_model.get_out_bias())

            # Loaded biases should match between backends
            np.testing.assert_allclose(
                dp_bias_loaded, pt_bias_loaded, rtol=1e-10, atol=1e-10
            )
            # Loaded biases should match the originally computed biases
            np.testing.assert_allclose(
                dp_bias_loaded, dp_bias_after, rtol=1e-10, atol=1e-10
            )

    def test_get_observed_type_list(self) -> None:
        """get_observed_type_list should be consistent across dp, pt, pt_expt.

        Uses mock data containing only type 0 ("O") so that types 1 ("H")
        and 2 ("B") are unobserved and should be absent from the returned list.
        """
        nframes = 2
        natoms = 6
        # All atoms are type 0 — types 1, 2 are unobserved
        atype_2f = np.zeros((nframes, natoms), dtype=np.int32)
        coords_2f = np.tile(self.coords, (nframes, 1, 1))
        box_2f = np.tile(self.box.reshape(1, 3, 3), (nframes, 1, 1))
        natoms_data = np.array(
            [[natoms, natoms, natoms, 0, 0]] * nframes, dtype=np.int32
        )
        energy_data = np.array([10.0, 20.0]).reshape(nframes, 1)

        dp_merged = [
            {
                "coord": coords_2f,
                "atype": atype_2f,
                "atype_ext": atype_2f,
                "box": box_2f,
                "natoms": natoms_data,
                "energy": energy_data,
                "find_energy": np.float32(1.0),
            }
        ]
        pt_merged = [
            {
                "coord": numpy_to_torch(coords_2f),
                "atype": numpy_to_torch(atype_2f),
                "atype_ext": numpy_to_torch(atype_2f),
                "box": numpy_to_torch(box_2f),
                "natoms": numpy_to_torch(natoms_data),
                "energy": numpy_to_torch(energy_data),
                "find_energy": np.float32(1.0),
            }
        ]

        self.dp_model.atomic_model.compute_or_load_out_stat(dp_merged)
        self.pt_model.atomic_model.compute_or_load_out_stat(pt_merged)
        self.pt_expt_model.atomic_model.compute_or_load_out_stat(dp_merged)

        dp_observed = self.dp_model.get_observed_type_list()
        pt_observed = self.pt_model.get_observed_type_list()
        pe_observed = self.pt_expt_model.get_observed_type_list()

        self.assertEqual(dp_observed, pt_observed)
        self.assertEqual(dp_observed, pe_observed)
        # Only type 0 ("O") should be observed
        self.assertEqual(dp_observed, ["O"])


@parameterized(
    (([], []), ([[0, 1]], [1])),  # (pair_exclude_types, atom_exclude_types)
)
@unittest.skipUnless(INSTALLED_PT and INSTALLED_PT_EXPT, "PT and PT_EXPT are required")
class TestZBLComputeOrLoadStat(unittest.TestCase):
    """Test that compute_or_load_stat produces identical statistics on dp, pt, and pt_expt.

    Covers descriptor stats (dstd) and output bias for DPZBLModel.
    Parameterized over exclusion types only (no fparam — LinearEnergyAtomicModel
    does not expose fitting_net for param stats).
    """

    def setUp(self) -> None:
        ((pair_exclude_types, atom_exclude_types),) = self.param
        data = model_args().normalize_value(
            {
                "type_map": ["O", "H", "B"],
                "use_srtab": f"{TESTS_DIR}/pt/water/data/zbl_tab_potential/H2O_tab_potential.txt",
                "smin_alpha": 0.1,
                "sw_rmin": 0.2,
                "sw_rmax": 4.0,
                "pair_exclude_types": pair_exclude_types,
                "atom_exclude_types": atom_exclude_types,
                "descriptor": {
                    "type": "se_atten",
                    "sel": 40,
                    "rcut_smth": 0.5,
                    "rcut": 4.0,
                    "neuron": [3, 6],
                    "axis_neuron": 2,
                    "attn": 8,
                    "attn_layer": 2,
                    "attn_dotr": True,
                    "attn_mask": False,
                    "activation_function": "tanh",
                    "scaling_factor": 1.0,
                    "normalize": False,
                    "temperature": 1.0,
                    "set_davg_zero": True,
                    "type_one_side": True,
                    "seed": 1,
                },
                "fitting_net": {
                    "neuron": [5, 5],
                    "resnet_dt": True,
                    "precision": "float64",
                    "seed": 1,
                },
            },
            trim_pattern="_*",
        )

        # Save data for reuse in load-from-file test
        self._model_data = data

        # Build dp model, then deserialize into pt and pt_expt to share weights
        self.dp_model = get_model_dp(data)
        serialized = self.dp_model.serialize()
        self.pt_model = DPZBLModelPT.deserialize(serialized)
        self.pt_expt_model = DPZBLModelPTExpt.deserialize(serialized)

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
        self.atype = np.array([0, 1, 1, 0, 1, 1], dtype=np.int32).reshape(1, -1)
        self.box = np.array(
            [13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0],
            dtype=GLOBAL_NP_FLOAT_PRECISION,
        ).reshape(1, 9)

        # Mock training data for compute_or_load_stat
        natoms = 6
        nframes = 3
        rng = np.random.default_rng(42)
        coords_stat = rng.normal(size=(nframes, natoms, 3)).astype(
            GLOBAL_NP_FLOAT_PRECISION
        )
        atype_stat = np.array([[0, 0, 1, 1, 1, 1]] * nframes, dtype=np.int32)
        box_stat = np.tile(
            np.eye(3, dtype=GLOBAL_NP_FLOAT_PRECISION).reshape(1, 3, 3) * 13.0,
            (nframes, 1, 1),
        )
        natoms_stat = np.array(
            [[natoms, natoms, 2, 4, 0]] * nframes, dtype=np.int32
        )  # 3 types: O=2, H=4, B=0
        energy_stat = rng.normal(size=(nframes, 1)).astype(GLOBAL_NP_FLOAT_PRECISION)

        # dp / pt_expt sample (numpy)
        np_sample = {
            "coord": coords_stat,
            "atype": atype_stat,
            "atype_ext": atype_stat,
            "box": box_stat,
            "natoms": natoms_stat,
            "energy": energy_stat,
            "find_energy": np.float32(1.0),
        }
        # pt sample (torch tensors)
        pt_sample = {
            "coord": numpy_to_torch(coords_stat),
            "atype": numpy_to_torch(atype_stat),
            "atype_ext": numpy_to_torch(atype_stat),
            "box": numpy_to_torch(box_stat),
            "natoms": numpy_to_torch(natoms_stat),
            "energy": numpy_to_torch(energy_stat),
            "find_energy": np.float32(1.0),
        }

        self.np_sampled = [np_sample]
        self.pt_sampled = [pt_sample]

    def _eval_dp(self) -> dict:
        return self.dp_model(self.coords, self.atype, box=self.box)

    def _eval_pt(self) -> dict:
        return {
            kk: torch_to_numpy(vv)
            for kk, vv in self.pt_model(
                numpy_to_torch(self.coords),
                numpy_to_torch(self.atype),
                box=numpy_to_torch(self.box),
                do_atomic_virial=True,
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
                box=pt_expt_numpy_to_torch(self.box),
                do_atomic_virial=True,
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
        # deepcopy samples: the ZBL model's stat path mutates natoms in-place
        # (stat.py applies atom_exclude_types mask via natoms[:, 2:] *= type_mask),
        # so each backend must receive its own copy.
        self.dp_model.compute_or_load_stat(lambda: copy.deepcopy(self.np_sampled))
        self.pt_model.compute_or_load_stat(lambda: copy.deepcopy(self.pt_sampled))
        self.pt_expt_model.compute_or_load_stat(lambda: copy.deepcopy(self.np_sampled))

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

            # 1. Compute stats and save to file (deepcopy: stat path mutates natoms)
            self.dp_model.compute_or_load_stat(
                lambda: copy.deepcopy(self.np_sampled),
                stat_file_path=DPPath(dp_h5, "a"),
            )
            self.pt_model.compute_or_load_stat(
                lambda: copy.deepcopy(self.pt_sampled),
                stat_file_path=DPPath(pt_h5, "a"),
            )
            self.pt_expt_model.compute_or_load_stat(
                lambda: copy.deepcopy(self.np_sampled),
                stat_file_path=DPPath(pe_h5, "a"),
            )

            # Save the computed serializations as reference
            dp_ser_computed = self.dp_model.serialize()
            pt_ser_computed = self.pt_model.serialize()
            pe_ser_computed = self.pt_expt_model.serialize()

            # 2. Build fresh models from the same initial weights
            dp_model2 = get_model_dp(self._model_data)
            pt_model2 = DPZBLModelPT.deserialize(dp_model2.serialize())
            pe_model2 = DPZBLModelPTExpt.deserialize(dp_model2.serialize())

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
