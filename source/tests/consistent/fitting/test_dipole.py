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
    (None, [0]),  # sel_type
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
            "sel_type": sel_type,
            "seed": 20240217,
        }
        return data

    def pass_data_to_cls(self, cls, data) -> Any:
        """Pass data to the class."""
        if cls not in (self.tf_class,):
            sel_type = data.pop("sel_type", None)
            if sel_type is not None:
                all_types = list(range(self.ntypes))
                exclude_types = [t for t in all_types if t not in sel_type]
                data["exclude_types"] = exclude_types
        return cls(**data, **self.additional_data)

    @property
    def skip_tf(self) -> bool:
        (
            resnet_dt,
            precision,
            mixed_types,
            sel_type,
        ) = self.param
        # mixed_types + sel_type is not supported
        return CommonTest.skip_tf or (mixed_types and sel_type is not None)

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
        return {
            "ntypes": self.ntypes,
            "dim_descrpt": self.inputs.shape[-1],
            "mixed_types": mixed_types,
            "embedding_width": 30,
        }

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

    def test_tf_consistent_with_ref(self) -> None:
        """Test whether TF and reference are consistent."""
        # Special handle for sel_types
        if self.skip_tf:
            self.skipTest("Unsupported backend")
        ref_backend = self.get_reference_backend()
        if ref_backend == self.RefBackend.TF:
            self.skipTest("Reference is self")
        ret1, data1 = self.get_reference_ret_serialization(ref_backend)
        ret1 = self.extract_ret(ret1, ref_backend)
        self.reset_unique_id()
        tf_obj = self.tf_class.deserialize(data1, suffix=self.unique_id)
        ret2, data2 = self.get_tf_ret_serialization_from_cls(tf_obj)
        ret2 = self.extract_ret(ret2, self.RefBackend.TF)
        if tf_obj.__class__.__name__.startswith(("Polar", "Dipole", "DOS")):
            # tf, pt serialization mismatch
            common_keys = set(data1.keys()) & set(data2.keys())
            data1 = {k: data1[k] for k in common_keys}
            data2 = {k: data2[k] for k in common_keys}

        # not comparing version
        data1.pop("@version")
        data2.pop("@version")

        if tf_obj.__class__.__name__.startswith("Polar"):
            data1["@variables"].pop("bias_atom_e")
        for ii, networks in enumerate(data2["nets"]["networks"]):
            if networks is None:
                data1["nets"]["networks"][ii] = None
        np.testing.assert_equal(data1, data2)
        for rr1, rr2 in zip(ret1, ret2):
            np.testing.assert_allclose(
                rr1.ravel()[: rr2.size], rr2.ravel(), rtol=self.rtol, atol=self.atol
            )
            assert rr1.dtype == rr2.dtype, f"{rr1.dtype} != {rr2.dtype}"
