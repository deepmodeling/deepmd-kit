# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.model.ener_model import EnergyModel as EnergyModelDP
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
    INSTALLED_PD,
    INSTALLED_PT,
    INSTALLED_PT_EXPT,
    INSTALLED_TF,
    SKIP_FLAG,
    CommonTest,
    parameterized,
)
from .common import (
    ModelTest,
)

if INSTALLED_PT:
    from deepmd.pt.model.model import get_model as get_model_pt
    from deepmd.pt.model.model.ener_model import EnergyModel as EnergyModelPT
    from deepmd.pt.utils.utils import to_numpy_array as torch_to_numpy
    from deepmd.pt.utils.utils import to_torch_tensor as numpy_to_torch
else:
    EnergyModelPT = None
if INSTALLED_TF:
    from deepmd.tf.model.ener import EnerModel as EnergyModelTF
else:
    EnergyModelTF = None
if INSTALLED_PD:
    from deepmd.pd.model.model import get_model as get_model_pd
    from deepmd.pd.model.model.ener_model import EnergyModel as EnergyModelPD
    from deepmd.pd.utils.utils import to_numpy_array as paddle_to_numpy
    from deepmd.pd.utils.utils import to_paddle_tensor as numpy_to_paddle
else:
    EnergyModelPD = None
if INSTALLED_PT_EXPT:
    from deepmd.pt_expt.common import to_torch_array as pt_expt_numpy_to_torch
    from deepmd.pt_expt.model import EnergyModel as EnergyModelPTExpt
else:
    EnergyModelPTExpt = None
from deepmd.utils.argcheck import (
    model_args,
)

if INSTALLED_JAX:
    from deepmd.jax.common import (
        to_jax_array,
    )
    from deepmd.jax.model.ener_model import EnergyModel as EnergyModelJAX
    from deepmd.jax.model.model import get_model as get_model_jax
else:
    EnergyModelJAX = None


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
            "type_map": ["O", "H"],
            "pair_exclude_types": pair_exclude_types,
            "atom_exclude_types": atom_exclude_types,
            "descriptor": {
                "type": "se_e2_a",
                "sel": [20, 20],
                "rcut_smth": 0.50,
                "rcut": 6.00,
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
            },
        }

    tf_class = EnergyModelTF
    dp_class = EnergyModelDP
    pt_class = EnergyModelPT
    pd_class = EnergyModelPD
    pt_expt_class = EnergyModelPTExpt
    jax_class = EnergyModelJAX
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
        if not self.skip_pd:
            return self.RefBackend.PD
        if not self.skip_dp:
            return self.RefBackend.DP
        raise ValueError("No available reference")

    @property
    def skip_tf(self):
        return (
            self.data["pair_exclude_types"] != []
            or self.data["atom_exclude_types"] != []
        )

    @property
    def skip_jax(self) -> bool:
        return not INSTALLED_JAX

    def pass_data_to_cls(self, cls, data) -> Any:
        """Pass data to the class."""
        data = data.copy()
        if cls is EnergyModelDP:
            return get_model_dp(data)
        elif cls is EnergyModelPT:
            model = get_model_pt(data)
            model.atomic_model.out_bias.uniform_()
            return model
        elif cls is EnergyModelPTExpt:
            dp_model = get_model_dp(data)
            return EnergyModelPTExpt.deserialize(dp_model.serialize())
        elif cls is EnergyModelJAX:
            return get_model_jax(data)
        elif cls is EnergyModelPD:
            return get_model_pd(data)
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

    def eval_pd(self, pd_obj: Any) -> Any:
        return self.eval_pd_model(
            pd_obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
        )

    def extract_ret(self, ret: Any, backend) -> tuple[np.ndarray, ...]:
        # shape not matched. ravel...
        if backend is self.RefBackend.DP:
            return (
                ret["energy_redu"].ravel(),
                ret["energy"].ravel(),
                SKIP_FLAG,
                SKIP_FLAG,
                SKIP_FLAG,
            )
        elif backend is self.RefBackend.PT:
            return (
                ret["energy"].ravel(),
                ret["atom_energy"].ravel(),
                ret["force"].ravel(),
                ret["virial"].ravel(),
                ret["atom_virial"].ravel(),
            )
        elif backend is self.RefBackend.PT_EXPT:
            return (
                ret["energy"].ravel(),
                ret["atom_energy"].ravel(),
                ret["force"].ravel(),
                ret["virial"].ravel(),
                ret["atom_virial"].ravel(),
            )
        elif backend is self.RefBackend.TF:
            return (
                ret[0].ravel(),
                ret[1].ravel(),
                ret[2].ravel(),
                ret[3].ravel(),
                ret[4].ravel(),
            )
        elif backend is self.RefBackend.JAX:
            return (
                ret["energy_redu"].ravel(),
                ret["energy"].ravel(),
                ret["energy_derv_r"].ravel(),
                ret["energy_derv_c_redu"].ravel(),
                ret["energy_derv_c"].ravel(),
            )
        elif backend is self.RefBackend.PD:
            return (
                ret["energy"].flatten(),
                ret["atom_energy"].flatten(),
                ret["force"].flatten(),
                ret["virial"].flatten(),
                ret["atom_virial"].flatten(),
            )
        raise ValueError(f"Unknown backend: {backend}")


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
class TestEnerLower(CommonTest, ModelTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        pair_exclude_types, atom_exclude_types = self.param
        return {
            "type_map": ["O", "H"],
            "pair_exclude_types": pair_exclude_types,
            "atom_exclude_types": atom_exclude_types,
            "descriptor": {
                "type": "se_e2_a",
                "sel": [20, 20],
                "rcut_smth": 0.50,
                "rcut": 6.00,
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
            },
        }

    tf_class = EnergyModelTF
    dp_class = EnergyModelDP
    pt_class = EnergyModelPT
    pt_expt_class = EnergyModelPTExpt
    jax_class = EnergyModelJAX
    pd_class = EnergyModelPD
    args = model_args()

    def get_reference_backend(self):
        """Get the reference backend.

        We need a reference backend that can reproduce forces.
        """
        if not self.skip_pt:
            return self.RefBackend.PT
        if not self.skip_pt_expt and self.pt_expt_class is not None:
            return self.RefBackend.PT_EXPT
        if not self.skip_jax:
            return self.RefBackend.JAX
        if not self.skip_dp:
            return self.RefBackend.DP
        if not self.skip_pd:
            return self.RefBackend.PD
        raise ValueError("No available reference")

    @property
    def skip_tf(self) -> bool:
        # TF does not have lower interface
        return True

    @property
    def skip_jax(self) -> bool:
        return not INSTALLED_JAX

    def pass_data_to_cls(self, cls, data) -> Any:
        """Pass data to the class."""
        data = data.copy()
        if cls is EnergyModelDP:
            return get_model_dp(data)
        elif cls is EnergyModelPT:
            return get_model_pt(data)
        elif cls is EnergyModelPTExpt:
            dp_model = get_model_dp(data)
            return EnergyModelPTExpt.deserialize(dp_model.serialize())
        elif cls is EnergyModelJAX:
            return get_model_jax(data)
        elif cls is EnergyModelPD:
            return get_model_pd(data)
        return cls(**data, **self.additional_data)

    def setUp(self) -> None:
        CommonTest.setUp(self)

        self.ntypes = 2
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
        atype = np.array([0, 1, 1, 0, 1, 1], dtype=np.int32).reshape(1, -1)
        box = np.array(
            [13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0],
            dtype=GLOBAL_NP_FLOAT_PRECISION,
        ).reshape(1, 9)

        rcut = 6.0
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
            6.0,
            [20, 20],
            distinguish_types=True,
        )
        extended_coord = extended_coord.reshape(nframes, -1, 3)
        self.nlist = nlist
        self.extended_coord = extended_coord
        self.extended_atype = extended_atype
        self.mapping = mapping

    def build_tf(self, obj: Any, suffix: str) -> tuple[list, dict]:
        raise NotImplementedError("no TF in this test")

    def eval_dp(self, dp_obj: Any) -> Any:
        return dp_obj.call_lower(
            self.extended_coord,
            self.extended_atype,
            self.nlist,
            self.mapping,
            do_atomic_virial=True,
        )

    def eval_pt(self, pt_obj: Any) -> Any:
        return {
            kk: torch_to_numpy(vv)
            for kk, vv in pt_obj.forward_lower(
                numpy_to_torch(self.extended_coord),
                numpy_to_torch(self.extended_atype),
                numpy_to_torch(self.nlist),
                numpy_to_torch(self.mapping),
                do_atomic_virial=True,
            ).items()
        }

    def eval_pt_expt(self, pt_expt_obj: Any) -> Any:
        coord_tensor = pt_expt_numpy_to_torch(self.extended_coord)
        coord_tensor.requires_grad_(True)
        return {
            kk: vv.detach().cpu().numpy() if vv is not None else None
            for kk, vv in pt_expt_obj.call_lower(
                coord_tensor,
                pt_expt_numpy_to_torch(self.extended_atype),
                pt_expt_numpy_to_torch(self.nlist),
                pt_expt_numpy_to_torch(self.mapping),
                do_atomic_virial=True,
            ).items()
        }

    def eval_jax(self, jax_obj: Any) -> Any:
        return {
            kk: to_numpy_array(vv)
            for kk, vv in jax_obj.call_lower(
                to_jax_array(self.extended_coord),
                to_jax_array(self.extended_atype),
                to_jax_array(self.nlist),
                to_jax_array(self.mapping),
                do_atomic_virial=True,
            ).items()
        }

    def eval_pd(self, pd_obj: Any) -> Any:
        return {
            kk: paddle_to_numpy(vv)
            for kk, vv in pd_obj.forward_lower(
                numpy_to_paddle(self.extended_coord),
                numpy_to_paddle(self.extended_atype),
                numpy_to_paddle(self.nlist),
                numpy_to_paddle(self.mapping),
                do_atomic_virial=True,
            ).items()
        }

    def extract_ret(self, ret: Any, backend) -> tuple[np.ndarray, ...]:
        # shape not matched. ravel...
        if backend is self.RefBackend.DP:
            return (
                ret["energy_redu"].ravel(),
                ret["energy"].ravel(),
                SKIP_FLAG,
                SKIP_FLAG,
                SKIP_FLAG,
            )
        elif backend is self.RefBackend.PT:
            return (
                ret["energy"].ravel(),
                ret["atom_energy"].ravel(),
                ret["extended_force"].ravel(),
                ret["virial"].ravel(),
                ret["extended_virial"].ravel(),
            )
        elif backend is self.RefBackend.PT_EXPT:
            return (
                ret["energy_redu"].ravel(),
                ret["energy"].ravel(),
                ret["energy_derv_r"].ravel(),
                ret["energy_derv_c_redu"].ravel(),
                ret["energy_derv_c"].ravel(),
            )
        elif backend is self.RefBackend.JAX:
            return (
                ret["energy_redu"].ravel(),
                ret["energy"].ravel(),
                ret["energy_derv_r"].ravel(),
                ret["energy_derv_c_redu"].ravel(),
                ret["energy_derv_c"].ravel(),
            )
        elif backend is self.RefBackend.PD:
            return (
                ret["energy"].flatten(),
                ret["atom_energy"].flatten(),
                ret["extended_force"].flatten(),
                ret["virial"].flatten(),
                ret["extended_virial"].flatten(),
            )
        raise ValueError(f"Unknown backend: {backend}")


@unittest.skipUnless(INSTALLED_PT, "PyTorch is not installed")
class TestEnerModelAPIs(unittest.TestCase):
    """Test consistency of model-level APIs between pt and dpmodel backends.

    Both models are constructed from the same serialized weights
    (dpmodel -> serialize -> pt deserialize) so that numerical outputs
    can be compared directly.
    """

    def setUp(self) -> None:
        from deepmd.utils.argcheck import (
            model_args,
        )

        data = model_args().normalize_value(
            {
                "type_map": ["O", "H"],
                "descriptor": {
                    "type": "se_e2_a",
                    "sel": [20, 20],
                    "rcut_smth": 0.50,
                    "rcut": 6.00,
                    "neuron": [3, 6],
                    "resnet_dt": False,
                    "axis_neuron": 2,
                    "precision": "float64",
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
        # Build dpmodel first, then deserialize into pt to share weights
        self.dp_model = get_model_dp(data)
        serialized = self.dp_model.serialize()
        self.pt_model = EnergyModelPT.deserialize(serialized)

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
        rcut = 6.0
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
            [20, 20],
            distinguish_types=True,
        )
        self.extended_coord = extended_coord.reshape(nframes, -1, 3)
        self.extended_atype = extended_atype
        self.mapping = mapping
        self.nlist = nlist

    def test_translated_output_def(self) -> None:
        """translated_output_def should return the same keys on dp and pt."""
        dp_def = self.dp_model.translated_output_def()
        pt_def = self.pt_model.translated_output_def()
        self.assertEqual(set(dp_def.keys()), set(pt_def.keys()))
        for key in dp_def:
            self.assertEqual(dp_def[key].shape, pt_def[key].shape)

    def test_get_descriptor(self) -> None:
        """get_descriptor should return a non-None object on both backends."""
        self.assertIsNotNone(self.dp_model.get_descriptor())
        self.assertIsNotNone(self.pt_model.get_descriptor())

    def test_get_fitting_net(self) -> None:
        """get_fitting_net should return a non-None object on both backends."""
        self.assertIsNotNone(self.dp_model.get_fitting_net())
        self.assertIsNotNone(self.pt_model.get_fitting_net())

    def test_get_out_bias(self) -> None:
        """get_out_bias should return numerically equal values on dp and pt.

        Freshly constructed models have zero bias; the shape (n_output x ntypes x odim)
        is verified. Non-zero bias round-trip is covered by test_set_out_bias.
        """
        dp_bias = to_numpy_array(self.dp_model.get_out_bias())
        pt_bias = torch_to_numpy(self.pt_model.get_out_bias())
        np.testing.assert_allclose(dp_bias, pt_bias, rtol=1e-10, atol=1e-10)
        # Verify shape is sensible (n_output_keys x ntypes x odim)
        self.assertEqual(dp_bias.shape[1], 2)  # ntypes
        self.assertGreater(dp_bias.shape[0], 0)  # at least one output key

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

    def test_forward_common_alias(self) -> None:
        """forward_common should be the same as call on dpmodel."""
        ret_call = self.dp_model.call(
            self.coords,
            self.atype,
            box=self.box,
        )
        ret_fc = self.dp_model.forward_common(
            self.coords,
            self.atype,
            box=self.box,
        )
        for key in ret_call:
            np.testing.assert_equal(ret_call[key], ret_fc[key])

    def test_forward_common_lower_alias(self) -> None:
        """forward_common_lower should be the same as call_lower on dpmodel."""
        ret_call = self.dp_model.call_lower(
            self.extended_coord,
            self.extended_atype,
            self.nlist,
            self.mapping,
        )
        ret_fc = self.dp_model.forward_common_lower(
            self.extended_coord,
            self.extended_atype,
            self.nlist,
            self.mapping,
        )
        for key in ret_call:
            np.testing.assert_equal(ret_call[key], ret_fc[key])

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
        self.assertAlmostEqual(self.dp_model.get_rcut(), 6.0)

    def test_get_type_map(self) -> None:
        """get_type_map should return the same list on dp and pt."""
        self.assertEqual(self.dp_model.get_type_map(), self.pt_model.get_type_map())
        self.assertEqual(self.dp_model.get_type_map(), ["O", "H"])

    def test_get_sel(self) -> None:
        """get_sel should return the same list on dp and pt."""
        self.assertEqual(self.dp_model.get_sel(), self.pt_model.get_sel())
        self.assertEqual(self.dp_model.get_sel(), [20, 20])

    def test_get_nsel(self) -> None:
        """get_nsel should return the same value on dp and pt."""
        self.assertEqual(self.dp_model.get_nsel(), self.pt_model.get_nsel())
        self.assertEqual(self.dp_model.get_nsel(), 40)

    def test_get_nnei(self) -> None:
        """get_nnei should return the same value on dp and pt."""
        self.assertEqual(self.dp_model.get_nnei(), self.pt_model.get_nnei())
        self.assertEqual(self.dp_model.get_nnei(), 40)

    def test_mixed_types(self) -> None:
        """mixed_types should return the same value on dp and pt."""
        self.assertEqual(self.dp_model.mixed_types(), self.pt_model.mixed_types())
        # se_e2_a is not mixed-types
        self.assertFalse(self.dp_model.mixed_types())

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
        self.assertFalse(self.dp_model.need_sorted_nlist_for_lower())

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
        # For this model config, all types are selected (empty list)
        self.assertEqual(self.dp_model.get_sel_type(), [0, 1])

    def test_is_aparam_nall(self) -> None:
        """is_aparam_nall should return the same value on dp and pt."""
        self.assertEqual(self.dp_model.is_aparam_nall(), self.pt_model.is_aparam_nall())
        self.assertFalse(self.dp_model.is_aparam_nall())

    def test_atomic_output_def(self) -> None:
        """atomic_output_def should return the same keys and shapes on dp and pt."""
        dp_def = self.dp_model.atomic_output_def()
        pt_def = self.pt_model.atomic_output_def()
        self.assertEqual(set(dp_def.keys()), set(pt_def.keys()))
        for key in dp_def.keys():
            self.assertEqual(dp_def[key].shape, pt_def[key].shape)

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

    def test_has_default_fparam(self) -> None:
        """has_default_fparam should return the same value on dp and pt."""
        self.assertEqual(
            self.dp_model.has_default_fparam(),
            self.pt_model.has_default_fparam(),
        )
        self.assertFalse(self.dp_model.has_default_fparam())

    def test_get_default_fparam(self) -> None:
        """get_default_fparam should return None on both dp and pt (no fparam configured)."""
        dp_val = self.dp_model.get_default_fparam()
        pt_val = self.pt_model.get_default_fparam()
        self.assertIsNone(dp_val)
        self.assertIsNone(pt_val)
        # Note: both return None because no default_fparam is configured.
        # A non-trivial return requires configuring default_fparam in the fitting net.

    def test_change_out_bias(self) -> None:
        """change_out_bias should produce consistent bias on dp and pt."""
        nframes = 2
        # Use realistic coords (from setUp, tiled for 2 frames)
        coords_2f = np.tile(self.coords, (nframes, 1, 1))  # (2, 6, 3)
        atype_2f = np.array([[0, 0, 1, 1, 1, 1], [0, 1, 1, 0, 1, 1]], dtype=np.int32)
        box_2f = np.tile(self.box.reshape(1, 3, 3), (nframes, 1, 1))
        natoms_data = np.array([[6, 6, 2, 4], [6, 6, 2, 4]], dtype=np.int32)
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

        # Save initial (zero) bias
        dp_bias_init = to_numpy_array(self.dp_model.get_out_bias()).copy()

        # Test "set-by-statistic" mode
        self.dp_model.change_out_bias(dp_merged, bias_adjust_mode="set-by-statistic")
        self.pt_model.change_out_bias(pt_merged, bias_adjust_mode="set-by-statistic")
        dp_bias = to_numpy_array(self.dp_model.get_out_bias())
        pt_bias = torch_to_numpy(self.pt_model.get_out_bias())
        np.testing.assert_allclose(dp_bias, pt_bias, rtol=1e-10, atol=1e-10)
        # Verify bias actually changed from initial zeros
        self.assertFalse(
            np.allclose(dp_bias, dp_bias_init),
            "set-by-statistic did not change the bias from initial values",
        )

        # Test "change-by-statistic" mode (adjusts bias based on model predictions)
        dp_bias_before = dp_bias.copy()
        self.dp_model.change_out_bias(dp_merged, bias_adjust_mode="change-by-statistic")
        self.pt_model.change_out_bias(pt_merged, bias_adjust_mode="change-by-statistic")
        dp_bias2 = to_numpy_array(self.dp_model.get_out_bias())
        pt_bias2 = torch_to_numpy(self.pt_model.get_out_bias())
        np.testing.assert_allclose(dp_bias2, pt_bias2, rtol=1e-10, atol=1e-10)
        # Verify change-by-statistic further modified the bias
        self.assertFalse(
            np.allclose(dp_bias2, dp_bias_before),
            "change-by-statistic did not further change the bias",
        )

    def test_change_type_map(self) -> None:
        """change_type_map should produce consistent results on dp and pt.

        Uses a DPA1 (se_atten) descriptor since se_e2_a does not support
        change_type_map (non-mixed-types descriptors raise NotImplementedError).
        """
        from deepmd.utils.argcheck import model_args as model_args_fn

        data = model_args_fn().normalize_value(
            {
                "type_map": ["O", "H"],
                "descriptor": {
                    "type": "se_atten",
                    "sel": 20,
                    "rcut_smth": 0.50,
                    "rcut": 6.00,
                    "neuron": [3, 6],
                    "resnet_dt": False,
                    "axis_neuron": 2,
                    "precision": "float64",
                    "seed": 1,
                    "attn": 6,
                    "attn_layer": 0,
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
        dp_model = get_model_dp(data)
        pt_model = EnergyModelPT.deserialize(dp_model.serialize())

        # Set non-zero out_bias so the swap is non-trivial
        dp_bias_orig = to_numpy_array(dp_model.get_out_bias()).copy()
        new_bias = dp_bias_orig.copy()
        new_bias[:, 0, :] = 1.5  # type 0 ("O")
        new_bias[:, 1, :] = -3.7  # type 1 ("H")
        dp_model.set_out_bias(new_bias)
        pt_model.set_out_bias(numpy_to_torch(new_bias))

        new_type_map = ["H", "O"]
        dp_model.change_type_map(new_type_map)
        pt_model.change_type_map(new_type_map)

        # Both should have the new type_map
        self.assertEqual(dp_model.get_type_map(), new_type_map)
        self.assertEqual(pt_model.get_type_map(), new_type_map)

        # Out_bias should be reordered consistently between backends
        dp_bias_new = to_numpy_array(dp_model.get_out_bias())
        pt_bias_new = torch_to_numpy(pt_model.get_out_bias())
        np.testing.assert_allclose(dp_bias_new, pt_bias_new, rtol=1e-10, atol=1e-10)

        # Verify the reorder is correct: old type 0 -> new type 1, old type 1 -> new type 0
        np.testing.assert_allclose(
            dp_bias_new[:, 0, :],
            new_bias[:, 1, :],
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            dp_bias_new[:, 1, :],
            new_bias[:, 0, :],
            rtol=1e-10,
            atol=1e-10,
        )

    def test_update_sel(self) -> None:
        """update_sel should return the same result on dp and pt."""
        from unittest.mock import (
            patch,
        )

        from deepmd.dpmodel.model.dp_model import DPModelCommon as DPModelCommonDP
        from deepmd.pt.model.model.dp_model import DPModelCommon as DPModelCommonPT

        mock_min_nbor_dist = 0.5
        mock_sel = [10, 20]
        local_jdata = {
            "type_map": ["O", "H"],
            "descriptor": {
                "type": "se_e2_a",
                "sel": "auto",
                "rcut_smth": 0.50,
                "rcut": 6.00,
            },
            "fitting_net": {
                "neuron": [5, 5],
            },
        }
        type_map = ["O", "H"]

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
        self.assertIsInstance(dp_result["descriptor"]["sel"], list)
        self.assertNotEqual(dp_result["descriptor"]["sel"], "auto")

    def test_get_ntypes(self) -> None:
        """get_ntypes should return the same value on dp and pt."""
        self.assertEqual(self.dp_model.get_ntypes(), self.pt_model.get_ntypes())
        self.assertEqual(self.dp_model.get_ntypes(), 2)

    def test_compute_or_load_out_stat(self) -> None:
        """compute_or_load_out_stat should produce consistent bias on dp and pt.

        Tests both the compute path (from data) and the load path (from file).
        Both backends should save the same stat file content and load identical
        biases from file.
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
        natoms_data = np.array([[6, 6, 2, 4], [6, 6, 2, 4]], dtype=np.int32)
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

        # Verify bias is initially zero (or at least identical)
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
