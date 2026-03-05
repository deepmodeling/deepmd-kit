# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel.model.ener_model import EnergyModel as EnergyModelDP
from deepmd.dpmodel.model.model import get_model as get_model_dp
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)

from ..common import (
    INSTALLED_JAX,
    INSTALLED_PT,
    INSTALLED_PT_EXPT,
    CommonTest,
)
from .common import (
    ModelTest,
)

if INSTALLED_PT:
    from deepmd.pt.model.model import get_model as get_model_pt
    from deepmd.pt.model.model.ener_model import EnergyModel as EnergyModelPT
else:
    EnergyModelPT = None
if INSTALLED_PT_EXPT:
    from deepmd.pt_expt.model import EnergyModel as EnergyModelPTExpt
else:
    EnergyModelPTExpt = None
if INSTALLED_JAX:
    from deepmd.jax.model.ener_model import EnergyModel as EnergyModelJAX
    from deepmd.jax.model.model import get_model as get_model_jax
else:
    EnergyModelJAX = None
from deepmd.utils.argcheck import (
    model_args,
)


class TestEnerHessian(CommonTest, ModelTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        return {
            "type_map": ["O", "H"],
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

    tf_class = None
    dp_class = EnergyModelDP
    pt_class = EnergyModelPT
    pt_expt_class = EnergyModelPTExpt
    jax_class = EnergyModelJAX
    pd_class = None
    args = model_args()

    @property
    def skip_tf(self) -> bool:
        return True

    @property
    def skip_dp(self) -> bool:
        return True

    @property
    def skip_pd(self) -> bool:
        return True

    @property
    def skip_jax(self) -> bool:
        return not INSTALLED_JAX

    @property
    def skip_array_api_strict(self) -> bool:
        return True

    def get_reference_backend(self):
        """Get the reference backend.

        We need a reference backend that can compute hessian.
        """
        if not self.skip_pt:
            return self.RefBackend.PT
        if not self.skip_pt_expt and self.pt_expt_class is not None:
            return self.RefBackend.PT_EXPT
        if not self.skip_jax:
            return self.RefBackend.JAX
        raise ValueError("No available reference")

    def pass_data_to_cls(self, cls, data) -> Any:
        """Pass data to the class and enable hessian."""
        data = data.copy()
        if cls is EnergyModelDP:
            model = get_model_dp(data)
        elif cls is EnergyModelPT:
            model = get_model_pt(data)
            model.atomic_model.out_bias.uniform_()
        elif cls is EnergyModelPTExpt:
            dp_model = get_model_dp(data)
            model = EnergyModelPTExpt.deserialize(dp_model.serialize())
        elif cls is EnergyModelJAX:
            model = get_model_jax(data)
        else:
            model = cls(**data)
        model.enable_hessian()
        return model

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
        raise NotImplementedError("no TF in this test")

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
        if backend in {
            self.RefBackend.PT,
            self.RefBackend.PT_EXPT,
            self.RefBackend.JAX,
        }:
            return (
                ret["energy"].ravel(),
                ret["atom_energy"].ravel(),
                ret["force"].ravel(),
                ret["virial"].ravel(),
                ret["hessian"].ravel(),
            )
        raise ValueError(f"Unknown backend: {backend}")

    def _enable_hessian_on(self, obj):
        """Enable hessian on a model object."""
        obj.enable_hessian()
        return obj

    def test_pt_consistent_with_ref(self) -> None:
        """Test PT consistent with reference, re-enabling hessian after deserialize."""
        if self.skip_pt:
            self.skipTest("Unsupported backend")
        ref_backend = self.get_reference_backend()
        if ref_backend == self.RefBackend.PT:
            self.skipTest("Reference is self")
        ret1, data1 = self.get_reference_ret_serialization(ref_backend)
        ret1 = self.extract_ret(ret1, ref_backend)
        obj = self._enable_hessian_on(self.pt_class.deserialize(data1))
        ret2 = self.eval_pt(obj)
        ret2 = self.extract_ret(ret2, self.RefBackend.PT)
        for rr1, rr2 in zip(ret1, ret2, strict=True):
            np.testing.assert_allclose(rr1, rr2, rtol=self.rtol, atol=self.atol)

    def test_pt_expt_consistent_with_ref(self) -> None:
        """Test pt_expt consistent with reference, re-enabling hessian after deserialize."""
        if self.skip_pt_expt or self.pt_expt_class is None:
            self.skipTest("Unsupported backend")
        ref_backend = self.get_reference_backend()
        if ref_backend == self.RefBackend.PT_EXPT:
            self.skipTest("Reference is self")
        ret1, data1 = self.get_reference_ret_serialization(ref_backend)
        ret1 = self.extract_ret(ret1, ref_backend)
        obj = self._enable_hessian_on(self.pt_expt_class.deserialize(data1))
        ret2 = self.eval_pt_expt(obj)
        ret2 = self.extract_ret(ret2, self.RefBackend.PT_EXPT)
        for rr1, rr2 in zip(ret1, ret2, strict=True):
            np.testing.assert_allclose(rr1, rr2, rtol=self.rtol, atol=self.atol)

    def test_jax_consistent_with_ref(self) -> None:
        """Test JAX consistent with reference, re-enabling hessian after deserialize."""
        if self.skip_jax or self.jax_class is None:
            self.skipTest("Unsupported backend")
        ref_backend = self.get_reference_backend()
        if ref_backend == self.RefBackend.JAX:
            self.skipTest("Reference is self")
        ret1, data1 = self.get_reference_ret_serialization(ref_backend)
        ret1 = self.extract_ret(ret1, ref_backend)
        obj = self._enable_hessian_on(self.jax_class.deserialize(data1))
        ret2 = self.eval_jax(obj)
        ret2 = self.extract_ret(ret2, self.RefBackend.JAX)
        for rr1, rr2 in zip(ret1, ret2, strict=True):
            np.testing.assert_allclose(rr1, rr2, rtol=self.rtol, atol=self.atol)

    def test_pt_self_consistent(self) -> None:
        """Skip: hessian is a runtime flag, not preserved by serialize/deserialize."""
        self.skipTest("Hessian state is not serialized")

    def test_pt_expt_self_consistent(self) -> None:
        """Skip: hessian is a runtime flag, not preserved by serialize/deserialize."""
        self.skipTest("Hessian state is not serialized")

    def test_jax_self_consistent(self) -> None:
        """Skip: hessian is a runtime flag, not preserved by serialize/deserialize."""
        self.skipTest("Hessian state is not serialized")

    def test_dp_self_consistent(self) -> None:
        """Skip: hessian is a runtime flag, not preserved by serialize/deserialize."""
        self.skipTest("Hessian state is not serialized")


if __name__ == "__main__":
    unittest.main()
