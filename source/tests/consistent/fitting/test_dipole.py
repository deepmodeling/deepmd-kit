# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.fitting.dipole_fitting import DipoleFitting as DipoleFittingDP
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)

from ..common import (
    INSTALLED_ARRAY_API_STRICT,
    INSTALLED_JAX,
    INSTALLED_PT,
    INSTALLED_TF,
    CommonTest,
    parameterized,
)
from .common import (
    DipoleFittingTest,
)

if INSTALLED_PT:
    import torch

    from deepmd.pt.model.task.dipole import DipoleFittingNet as DipoleFittingPT
    from deepmd.pt.utils.env import DEVICE as PT_DEVICE
else:
    DipoleFittingPT = object
if INSTALLED_TF:
    from deepmd.tf.fit.dipole import DipoleFittingSeA as DipoleFittingTF
else:
    DipoleFittingTF = object
if INSTALLED_JAX:
    from deepmd.jax.env import (
        jnp,
    )
    from deepmd.jax.fitting.fitting import DipoleFittingNet as DipoleFittingJAX
else:
    DipoleFittingJAX = object
if INSTALLED_ARRAY_API_STRICT:
    import array_api_strict

    from ...array_api_strict.fitting.fitting import (
        DipoleFittingNet as DipoleFittingArrayAPIStrict,
    )
else:
    DipoleFittingArrayAPIStrict = object
from deepmd.utils.argcheck import (
    fitting_dipole,
)


@parameterized(
    (True, False),  # resnet_dt
    ("float64", "float32"),  # precision
    (True, False),  # mixed_types
    ([], [0, 1]),  # sel_type
)
class TestDipole(CommonTest, DipoleFittingTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        (
            resnet_dt,
            precision,
            mixed_types,
            sel_type,
        ) = self.param
        data = {
            "neuron": [5, 5, 5],
            "resnet_dt": resnet_dt,
            "precision": precision,
            "seed": 20240217,
        }
        # Only add sel_type if it's not empty (for TF backend compatibility)
        if sel_type:
            data["sel_type"] = sel_type
        return data

    @property
    def skip_pt(self) -> bool:
        (
            resnet_dt,
            precision,
            mixed_types,
            sel_type,
        ) = self.param
        return CommonTest.skip_pt

    tf_class = DipoleFittingTF
    dp_class = DipoleFittingDP
    pt_class = DipoleFittingPT
    jax_class = DipoleFittingJAX
    array_api_strict_class = DipoleFittingArrayAPIStrict
    args = fitting_dipole()
    skip_jax = not INSTALLED_JAX
    skip_array_api_strict = not INSTALLED_ARRAY_API_STRICT

    def setUp(self) -> None:
        CommonTest.setUp(self)

        self.ntypes = 2
        self.natoms = np.array([6, 6, 2, 4], dtype=np.int32)
        self.inputs = np.ones((1, 6, 20), dtype=GLOBAL_NP_FLOAT_PRECISION)
        self.gr = np.ones((1, 6, 30, 3), dtype=GLOBAL_NP_FLOAT_PRECISION)
        self.atype = np.array([0, 1, 1, 0, 1, 1], dtype=np.int32)
        # inconsistent if not sorted
        self.atype.sort()

    @property
    def additional_data(self) -> dict:
        (
            resnet_dt,
            precision,
            mixed_types,
            sel_type,
        ) = self.param
        additional = {
            "ntypes": self.ntypes,
            "dim_descrpt": self.inputs.shape[-1],
            "mixed_types": mixed_types,
            "embedding_width": 30,
        }
        # For DP/PT backends, use exclude_types instead of sel_type
        if sel_type:
            all_types = list(range(self.ntypes))
            exclude_types = [t for t in all_types if t not in sel_type]
            additional["exclude_types"] = exclude_types
        return additional

    def build_tf(self, obj: Any, suffix: str) -> tuple[list, dict]:
        (
            resnet_dt,
            precision,
            mixed_types,
            sel_type,
        ) = self.param
        return self.build_tf_fitting(
            obj,
            self.inputs.ravel(),
            self.gr,
            self.natoms,
            self.atype,
            None,
            suffix,
        )

    def eval_pt(self, pt_obj: Any) -> Any:
        (
            resnet_dt,
            precision,
            mixed_types,
            sel_type,
        ) = self.param
        return (
            pt_obj(
                torch.from_numpy(self.inputs).to(device=PT_DEVICE),
                torch.from_numpy(self.atype.reshape(1, -1)).to(device=PT_DEVICE),
                torch.from_numpy(self.gr).to(device=PT_DEVICE),
                None,
            )["dipole"]
            .detach()
            .cpu()
            .numpy()
        )

    def eval_dp(self, dp_obj: Any) -> Any:
        (
            resnet_dt,
            precision,
            mixed_types,
            sel_type,
        ) = self.param
        return dp_obj(
            self.inputs,
            self.atype.reshape(1, -1),
            self.gr,
            None,
        )["dipole"]

    def eval_jax(self, jax_obj: Any) -> Any:
        return np.asarray(
            jax_obj(
                jnp.asarray(self.inputs),
                jnp.asarray(self.atype.reshape(1, -1)),
                jnp.asarray(self.gr),
                None,
            )["dipole"]
        )

    def eval_array_api_strict(self, array_api_strict_obj: Any) -> Any:
        return to_numpy_array(
            array_api_strict_obj(
                array_api_strict.asarray(self.inputs),
                array_api_strict.asarray(self.atype.reshape(1, -1)),
                array_api_strict.asarray(self.gr),
                None,
            )["dipole"]
        )

    def extract_ret(self, ret: Any, backend) -> tuple[np.ndarray, ...]:
        if backend == self.RefBackend.TF:
            # shape is not same
            ret = ret[0].reshape(-1, self.natoms[0], 1)
        return (ret,)

    @property
    def rtol(self) -> float:
        """Relative tolerance for comparing the return value."""
        (
            resnet_dt,
            precision,
            mixed_types,
            sel_type,
        ) = self.param
        if precision == "float64":
            return 1e-10
        elif precision == "float32":
            return 1e-4
        else:
            raise ValueError(f"Unknown precision: {precision}")

    @property
    def atol(self) -> float:
        """Absolute tolerance for comparing the return value."""
        (
            resnet_dt,
            precision,
            mixed_types,
            sel_type,
        ) = self.param
        if precision == "float64":
            return 1e-10
        elif precision == "float32":
            return 1e-4
        else:
            raise ValueError(f"Unknown precision: {precision}")


class TestDipoleSelTypeBehavior(unittest.TestCase):
    """Test sel_type behavior specifically, without cross-backend consistency."""

    def setUp(self) -> None:
        self.ntypes = 2
        self.natoms = np.array([6, 6, 2, 4], dtype=np.int32)

    def test_tf_sel_type_all_types(self):
        """Test that TF dipole fitting creates networks for all selected types."""
        if not INSTALLED_TF:
            self.skipTest("TensorFlow not available")

        sel_type = [0, 1]  # Select all types

        tf_obj = DipoleFittingTF(
            ntypes=self.ntypes,
            dim_descrpt=20,
            embedding_width=30,
            neuron=[5, 5, 5],
            sel_type=sel_type,
        )

        # Verify sel_type is set correctly
        self.assertEqual(set(tf_obj.sel_type), set(sel_type))

        # Verify sel_mask is correct
        expected_mask = np.array([i in sel_type for i in range(self.ntypes)])
        np.testing.assert_array_equal(tf_obj.sel_mask, expected_mask)

    def test_tf_sel_type_partial(self):
        """Test that TF dipole fitting works with partial type selection."""
        if not INSTALLED_TF:
            self.skipTest("TensorFlow not available")

        sel_type = [0]  # Select only type 0

        tf_obj = DipoleFittingTF(
            ntypes=self.ntypes,
            dim_descrpt=20,
            embedding_width=30,
            neuron=[5, 5, 5],
            sel_type=sel_type,
        )

        # Verify sel_type is set correctly
        self.assertEqual(set(tf_obj.sel_type), set(sel_type))

        # Verify sel_mask is correct
        expected_mask = np.array([i in sel_type for i in range(self.ntypes)])
        np.testing.assert_array_equal(tf_obj.sel_mask, expected_mask)

    def test_dp_exclude_types_behavior(self):
        """Test that DP dipole fitting excludes the correct types."""
        sel_type = [0]  # Select only type 0
        all_types = list(range(self.ntypes))
        exclude_types = [t for t in all_types if t not in sel_type]

        dp_obj = DipoleFittingDP(
            ntypes=self.ntypes,
            dim_descrpt=20,
            embedding_width=30,
            neuron=[5, 5, 5],
            exclude_types=exclude_types,
        )

        # Verify exclude_types is set correctly
        self.assertEqual(set(dp_obj.exclude_types), set(exclude_types))

        # Verify get_sel_type returns the correct types
        selected_types = dp_obj.get_sel_type()
        self.assertEqual(set(selected_types), set(sel_type))

    def test_serialization_with_excluded_types(self):
        """Test that sel_type is correctly stored in DipoleFittingSeA."""
        if not INSTALLED_TF:
            self.skipTest("TensorFlow not available")

        # Test with excluding one type
        sel_type = [0]  # Only select type 0, exclude type 1

        tf_obj = DipoleFittingTF(
            ntypes=self.ntypes,
            dim_descrpt=20,
            embedding_width=30,
            neuron=[5, 5, 5],
            sel_type=sel_type,
        )

        # Verify that sel_type is correctly stored
        self.assertEqual(tf_obj.sel_type, sel_type)

        # Verify that sel_mask reflects the excluded types
        expected_mask = np.array([True, False])  # Only type 0 is selected
        np.testing.assert_array_equal(tf_obj.sel_mask, expected_mask)

    def test_network_collection_none_handling(self):
        """Test that NetworkCollection properly handles None networks."""
        from deepmd.dpmodel.utils.network import (
            NetworkCollection,
        )

        # Create a NetworkCollection with some None entries
        collection = NetworkCollection(ndim=1, ntypes=2)

        # Test that None values can be set
        collection[0] = None
        collection[1] = None

        # Test serialization with None values
        serialized = collection.serialize()
        self.assertIn("networks", serialized)
        networks = serialized["networks"]
        self.assertEqual(len(networks), 2)
        self.assertTrue(all(net is None for net in networks))
